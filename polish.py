"""Local-LLM polish pass for Roman Urdu transcripts (runs on CPU through Ollama; free, offline).

Design (based on research on LLM ASR-error-correction):
  * Small, self-contained work units. Whisper segments are first merged into COMPLETE SENTENCES (a segment that
    stops mid-sentence is joined to the next one), and the LLM only ever sees a chunk of 3-4 such sentences
    (about 70 words). No neighbouring lines, no whole-meeting context.
  * Context is cheap and local: the user's context file plus a short topic summary made from the FIRST FEW MINUTES
    of the recording only (default 5 min), never from the whole transcript.
  * Optional public word list (lexicon.py): unknown words are shown to the LLM with their closest known spelling.
  * The LLM gives a verdict for EVERY sentence it is shown: OK|id, FIX|id|text or FLAG|id|meaning|best guess.
    It never rewrites the whole transcript, so it cannot silently drop or invent lines.
  * Coverage is enforced: a sentence with no verdict is re-asked in smaller batches; if it still gets none it is
    marked "not reviewed" (yellow) instead of being silently skipped.
  * Every FIX passes deterministic guardrails (similarity, length, numbers, script). A fix that fails is NOT applied;
    it becomes a yellow flag with the suggestion attached.
  * Yellow flags keep the original text. The review page shows what the model thinks was meant, the changed words
    highlighted, and Accept / Keep-original buttons; the final text follows your choices.
  * Levels: conservative (spelling/grammar only) < standard (also repairs mis-heard words) <
    maximum (every detail, two review passes, smaller chunks).
"""
import difflib
import hashlib
import html as _html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434"
LEVELS = ("conservative", "standard", "maximum")

CONTEXT_MINUTES = 5          # the topic summary is built from the first N minutes of the recording only
CONTEXT_MAX_SENTENCES = 25   # ... and from at most this many sentences
MAX_SENTENCE_WORDS = 28      # one review unit is at most this long: short segments are merged, long ones split
MERGE_GAP_S = 0.7            # a pause at least this long ends a sentence even without punctuation
CHUNK_SENTENCES = 12         # sentences per LLM request (6 for maximum)
CHUNK_WORDS = 220            # ... and about this many words per request (130 for maximum): whole paragraphs, never cut

SYSTEM_TMPL = """You are a careful copy-editor for meeting transcripts.

SETTING
- These are voice recordings of meetings at a company called TMC, transcribed automatically for meeting records.
- The text is Roman Urdu / Hinglish: a mix of English, Urdu and possibly Hindi, written in Latin letters. Stray Perso-Arabic or Devanagari characters may appear.
- The speech-to-text model was weak and the microphone poor, so the text has misheard words, English words spelled by sound (for example "dedlain" for "deadline"), broken grammar and wrong word forms.
- You are shown one or two complete paragraphs at a time, split into numbered sentences. Use the whole paragraph to understand each sentence, but give a verdict on EVERY numbered sentence. Never guess what was said elsewhere in the meeting.

{rules}

OUTPUT FORMAT (strict)
Output exactly one line for EVERY numbered sentence you are given, in order, and nothing else:
OK|<id>
FIX|<id>|<the full corrected sentence>
FLAG|<id>|<short English note: what the speaker probably means>|<your best-guess corrected sentence in Roman Urdu>
<id> is the number in square brackets in front of the sentence you are talking about, and FIX must contain the corrected version of THAT SAME sentence (never of a neighbouring one).
OK means you checked the sentence and nothing needs to change. In a FLAG, leave out the last part (and its "|") if you cannot make any guess. Every id must appear exactly once; never skip a sentence.
No explanations, no markdown, no extra text.

EXAMPLES
Input:
[1] mein kal office nahi aa rha hon
[2] yeh project ka dedlain agle haftay hai
[3] aap ne email bhej di thi na
[4] wo clayant ko prozel bhejna tha kal
[5] aaz ka sthaanpaar zindagi mein jaan hai kab
Output:
FIX|1|main kal office nahi aa raha hoon
FIX|2|yeh project ki deadline agle hafte hai
OK|3
FLAG|4|Probably: the client needed the proposal sent yesterday; the wording is uncertain.|woh client ko proposal bhejna tha kal
FLAG|5|Possibly asking when something will happen today; too garbled to repair.

{context_block}"""

RULES_STANDARD = """RULES
1. Fix in place: spelling, grammar, verb forms and gender agreement, spacing, punctuation, and words that are clearly mis-transcribed when the context makes the intended word obvious.
   Replace a word by a different word ONLY if you are highly confident AND the new word sounds like the misheard one. If you are not sure, leave the sentence unchanged or FLAG it.
2. Keep the speaker's words, meaning, order and tone. Do not rephrase, summarize, translate, shorten, or add information. Never invent content just to make a sentence make sense.
3. Latin letters only. English words stay English words. Keep the existing Roman Urdu spelling style.
4. Keep numbers, names and dates unchanged unless they are clearly a transcription error of a known word.
5. If a sentence is too garbled to repair with confidence, do NOT change it. Output a FLAG with your best guess of what it probably means.
6. A sentence that is already fine gets OK|<id>."""

RULES_CONSERVATIVE = """RULES
1. Only fix spelling, grammar, verb forms, gender agreement, spacing and punctuation of the words that are already there.
2. Do not replace one word by a different word, do not rephrase, summarize, translate, shorten or add anything.
3. Latin letters only. English words stay English words. Keep the existing Roman Urdu spelling style.
4. Keep numbers, names and dates unchanged.
5. If a sentence does not make sense, do NOT change it. Output a FLAG with your best guess of what it probably means.
6. A sentence that is already fine gets OK|<id>."""

RULES_MAXIMUM = """RULES (MAXIMUM CARE: examine every sentence word by word and correct everything that is wrong, down to the smallest detail)
1. Check every sentence and every word in it. Look at: spelling, grammar, verb forms and tense, gender and number agreement, postpositions (ka/ki/ke, ko, se, mein, par), words that are clearly missing or are clearly transcription noise, spacing, capitalization (names, places, acronyms, start of a sentence) and punctuation (full stops, commas, question marks).
2. Repair mis-transcribed words, including ones that are only slightly wrong. Use the company, the context notes and the other words of the sentence to work out the intended word. Replace a word when the replacement sounds like the misheard one and fits the sentence. English words and technical terms must be spelled correctly.
3. Spell names and recurring terms exactly as they are written in the context notes.
4. Remove clear stutters, accidentally repeated words and filler fragments caused by the speech model. Keep real emphasis and everything the speaker actually meant.
5. Keep the speaker's meaning, order and tone. Do not summarize, translate, restyle, or add information that was not said. Never invent content.
6. Latin letters only. Keep the existing Roman Urdu spelling style (hai, nahi, mein, ...) unless a spelling is clearly wrong.
7. Keep numbers and dates unchanged unless they are clearly a transcription error of a known word.
8. If a sentence is too garbled to repair with confidence, do NOT change it. Output a FLAG with your best guess of what it probably means.
9. Output OK only after you have checked the sentence and found nothing at all to improve. When in doubt, look again before choosing OK."""

RULES = {"standard": RULES_STANDARD, "conservative": RULES_CONSERVATIVE, "maximum": RULES_MAXIMUM}

SECOND_PASS_NOTE = (
    "SECOND AND FINAL REVIEW. The sentences below have already been through one correction pass. Read each one again "
    "slowly and fix anything that is still wrong: spelling, grammar, punctuation, capitalization, mis-heard words. "
    "Reply OK only if you are sure nothing is left to improve.\n\n"
)

HINT_NOTE = (
    "SPELLING HINTS from a public Roman Hindi/Urdu word list (word in the sentences -> closest known spelling). "
    "They are only hints: ignore any that do not fit; names and English words are often missing from the list.\n"
)

SUMMARY_PROMPT = (
    "Below are the opening sentences (the first few minutes) of a noisy Roman Urdu / English meeting transcript at a "
    "company called TMC. In English, write at most 3 short sentences saying what the meeting seems to be about, then "
    "a line 'Likely terms:' listing up to 15 names or technical/English terms that probably recur, spelled correctly. "
    "Only use what is supported by the sentences. Do not invent.\n\n"
)

_FIX_RE = re.compile(r"^\W*(FIX|FLAG)\s*[|:]\s*\[?(\d+)\]?\s*[|:]\s*(.+?)\s*$", re.IGNORECASE)
_OK_RE = re.compile(r"^\W*OK\s*[|:]\s*([\[\]\d\s,]+)(?:[|:].*)?$", re.IGNORECASE)
_TERMINAL = re.compile(r"[.?!\u0964\u061f\u06d4][\"')\]]*\s*$")

# how far a fix may drift from the original sentence, per level
_MIN_RATIO = {"maximum": 0.45, "standard": 0.55, "conservative": 0.75}
_LEN_RANGE = {"maximum": (0.5, 1.7)}


# ---------------------------------------------------------------- Ollama client (stdlib only)
def _post(host, path, body, timeout):
    req = urllib.request.Request(host + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_chat(model, system, user, host=OLLAMA_URL, num_predict=600, num_ctx=4096, timeout=1800):
    body = {
        "model": model, "stream": False, "keep_alive": "30m", "think": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    try:
        data = _post(host, "/api/chat", body, timeout)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        if "think" in msg.lower():           # model/server without a thinking switch
            body.pop("think")
            data = _post(host, "/api/chat", body, timeout)
        else:
            raise RuntimeError(f"Ollama error {e.code}: {msg[:300]}")
    text = data.get("message", {}).get("content", "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ---------------------------------------------------------------- parsing + guardrails
def parse_actions(text: str, valid_ids):
    """-> {id: {'fix', 'flag', 'flag_fix', 'ok'}} for every id the model gave a verdict on.
    FLAG|id|note|best guess  ->  flag = note, flag_fix = best guess (optional)."""
    out = {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        m = _FIX_RE.match(line)
        if m:
            kind, i, payload = m.group(1).upper(), int(m.group(2)), m.group(3).strip()
            if i not in valid_ids:
                continue
            d = out.setdefault(i, _blank())
            if kind == "FIX" and d["fix"] is None:
                d["fix"] = payload
            elif kind == "FLAG" and d["flag"] is None:
                parts = [p.strip() for p in payload.split("|")]
                d["flag"] = parts[0] or "unclear"
                if len(parts) > 1 and parts[1]:
                    d["flag_fix"] = " ".join(parts[1:]).strip()
            continue
        m = _OK_RE.match(line)
        if m:
            for tok in re.findall(r"\d+", m.group(1)):
                i = int(tok)
                if i in valid_ids:
                    out.setdefault(i, _blank())["ok"] = True
    return out


def _blank():
    return {"fix": None, "flag": None, "flag_fix": None, "ok": False}


def _latin_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return 1.0 if not letters else sum(c.isascii() for c in letters) / len(letters)


def validate_fix(orig: str, new: str, level: str = "standard"):
    """Deterministic checks that keep the LLM from rewriting. -> (ok, reason)."""
    orig, new = " ".join(orig.split()), " ".join(new.split())
    if not new or new == orig:
        return False, "unchanged"
    if re.findall(r"\d+", orig) != re.findall(r"\d+", new):
        return False, "numbers changed"
    if _latin_ratio(new) < 0.95:
        return False, "non-Latin output"
    lo, hi = _LEN_RANGE.get(level, (0.6, 1.5))
    if not lo <= len(new) / max(1, len(orig)) <= hi:
        return False, "length changed too much"
    ratio = difflib.SequenceMatcher(None, orig.lower(), new.lower()).ratio()
    if ratio < _MIN_RATIO.get(level, 0.55):
        return False, "too different from original"
    return True, ""


def usable_guess(orig: str, guess: str) -> bool:
    """A best-guess sentence attached to a FLAG is only a suggestion the user can accept, so the checks are lighter
    than for an automatic FIX: it must be Latin text, differ from the original and be of a similar size."""
    orig, guess = " ".join(orig.split()), " ".join((guess or "").split())
    if not guess or guess == orig or _latin_ratio(guess) < 0.9:
        return False
    return 0.4 <= len(guess) / max(1, len(orig)) <= 2.0


_SENT_SPLIT = re.compile(r"(?:(?<=[.?!\u0964\u061f\u06d4])|(?<=[.?!\u0964\u061f\u06d4][\"')\]]))\s+")


def _split_long(sentence, max_words):
    """A sentence longer than max_words is cut into near-equal parts, preferably right after a comma."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    n = -(-len(words) // max_words)
    size = len(words) / n
    parts, start = [], 0
    for k in range(1, n):
        target = round(k * size)
        window = range(max(start + 3, target - 5), min(len(words) - 3, target + 5) + 1)
        cut = min((j for j in window if words[j - 1].endswith((",", ";", ":"))), key=lambda j: abs(j - target), default=target)
        parts.append(" ".join(words[start:cut]))
        start = cut
    parts.append(" ".join(words[start:]))
    return [p for p in parts if p]


def split_pieces(text, max_words=MAX_SENTENCE_WORDS):
    """Text -> pieces of at most max_words that break at sentence ends (. ? ! danda) wherever possible. Short
    consecutive sentences are packed together so a piece is not needlessly tiny."""
    out, cur = [], ""
    for sent in (s for s in _SENT_SPLIT.split(text) if s.strip()):
        for part in _split_long(sent.strip(), max_words):
            if cur and len(cur.split()) + len(part.split()) <= max_words:
                cur += " " + part
            else:
                if cur:
                    out.append(cur)
                cur = part
    if cur:
        out.append(cur)
    return out


def build_units(segments, merge=True, max_words=MAX_SENTENCE_WORDS, max_gap=MERGE_GAP_S):
    """Whisper segments -> review units of complete sentences, never longer than max_words.
      * A segment that is longer than that (Whisper often returns 60-100 words at once) is SPLIT at sentence ends
        (or at a comma / evenly when there is no punctuation); the split pieces get interpolated timestamps.
      * A segment that does not end in . ? ! (or the Hindi danda) is JOINED to the next one if the same person is
        talking, the pause between them is short, and the result is not too long.
    With merge=False every segment stays exactly as it is. Without timestamps nothing is joined.
    -> [{'roman', 'start', 'end', 'conf', 'speaker', 'new_seg'}]  (new_seg: first piece of a transcript segment)"""
    units, cur = [], None
    for s in segments:
        text = " ".join(str(s.get("roman", "")).split())
        if not text:
            continue
        st, en, conf, spk = s.get("start"), s.get("end"), s.get("conf"), s.get("speaker")
        pieces = split_pieces(text, max_words) if merge else [text]
        total = sum(len(p) for p in pieces) or 1
        done = 0
        for k, piece in enumerate(pieces):
            if st is not None and en is not None and len(pieces) > 1:
                p_st = st + (en - st) * done / total
                p_en = st + (en - st) * (done + len(piece)) / total
            else:
                p_st, p_en = st, en
            done += len(piece)
            if (cur is not None and merge and p_st is not None and cur["end"] is not None and k == 0
                    and cur["speaker"] == spk and not _TERMINAL.search(cur["roman"])
                    and p_st - cur["end"] < max_gap
                    and len(cur["roman"].split()) + len(piece.split()) <= max_words):
                cur["roman"] += " " + piece
                cur["end"] = p_en if p_en is not None else cur["end"]
                cur["conf"] = None if (cur["conf"] is None or conf is None) else min(cur["conf"], conf)
                continue
            if cur is not None:
                units.append(cur)
            cur = dict(roman=piece, start=p_st, end=p_en, conf=conf, speaker=spk, new_seg=(k == 0))
    if cur is not None:
        units.append(cur)
    return units


def load_context(path):
    if not path or not Path(path).exists():
        return ""
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
    return "\n".join(l for l in lines if l and not l.startswith("#"))


# ---------------------------------------------------------------- the polisher
class Polisher:
    def __init__(self, model=DEFAULT_MODEL, level="standard", window=None, context_file=None,
                 summary=True, mode="all", low_conf=-0.5, host=OLLAMA_URL, llm=None, verbose=True,
                 cache_path="auto", context_minutes=CONTEXT_MINUTES, lexicon=None, merge=True):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        self.model, self.level = model, level
        # window = sentences per LLM request (whole paragraphs). maximum = somewhat smaller chunks, every sentence
        self.window = max(1, window or (6 if level == "maximum" else CHUNK_SENTENCES))
        self.chunk_words = 130 if level == "maximum" else CHUNK_WORDS
        self.context_minutes = context_minutes
        self.merge = merge
        if lexicon is not None and not hasattr(lexicon, "hints"):      # a path / comma list of paths
            from lexicon import Lexicon
            lexicon = Lexicon(lexicon)
        self.lexicon = lexicon
        self.context_text = load_context(context_file)
        self.use_summary, self.low_conf = summary, low_conf
        self.mode = "all" if level == "maximum" else mode
        self.host, self.verbose = host, verbose
        self._raw_llm = llm or (lambda system, user, **kw: ollama_chat(model, system, user, host=host, **kw))
        # replies are cached on disk, so re-polishing the same transcript is instant
        if cache_path == "auto":
            cache_path = None if llm else Path(__file__).with_name(".polish_cache.json")
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        self.stats = {}
        self.summary_text = ""
        self._done = self._total = 0
        self._t0 = time.time()
        self._cb = None

    # ---- caching ----
    def _key(self, system, user):
        return hashlib.sha1(f"{self.model}|{self.level}|{system}|{user}".encode("utf-8")).hexdigest()

    def _save_cache(self):
        if self.cache_path is not None:
            try:
                self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    def _forget(self, system, user):
        """Drop a cached reply that turned out incomplete, so a re-run asks the model again."""
        if self._cache.pop(self._key(system, user), None) is not None:
            self._save_cache()

    def _llm(self, system, user, **kw):
        key = self._key(system, user)
        if key in self._cache:
            return self._cache[key]
        reply = self._raw_llm(system, user, **kw)
        if self.cache_path is not None:
            self._cache[key] = reply
            self._save_cache()
        return reply

    def _log(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def check(self):
        """Fail early, with instructions, if Ollama or the model is missing."""
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r:
                names = [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
        except Exception:
            raise RuntimeError(
                "Cannot reach Ollama at " + self.host + ".\n"
                "  1) Install it (free) from https://ollama.com  2) make sure it is running\n"
                f"  3) run:  ollama pull {self.model}")
        ok = any(n == self.model or (":" not in self.model and n.split(":")[0] == self.model) for n in names)
        if not ok:
            raise RuntimeError(f"Model '{self.model}' is not installed. Run:  ollama pull {self.model}\n"
                               f"Installed models: {', '.join(names) or '(none)'}")

    def _system(self, summary_text):
        parts = []
        if self.context_text:
            parts.append("CONTEXT PROVIDED BY THE USER (trust this):\n" + self.context_text)
        if summary_text:
            parts.append("TOPIC NOTES (auto-written from the first minutes of the meeting; may be imperfect):\n"
                         + summary_text)
        return SYSTEM_TMPL.format(rules=RULES[self.level], context_block="\n\n".join(parts)).strip()

    def _opening(self, units):
        """Sentences from the first few minutes only (or the first sentences when there are no timestamps)."""
        limit = self.context_minutes * 60
        head = []
        for u in units:
            if u["start"] is not None and u["start"] > limit:
                break
            head.append(u["roman"])
            if len(head) >= CONTEXT_MAX_SENTENCES:
                break
        return head if len(head) >= 3 else [u["roman"] for u in units[:8]]

    def _summarize(self, units):
        if not self.use_summary:
            return ""
        head = self._opening(units)
        if len(head) < 3:
            return ""
        sample = "\n".join(head)[:1800]
        try:
            return self._llm("You summarize meeting transcripts briefly and faithfully.",
                             SUMMARY_PROMPT + sample, num_predict=160).strip()
        except Exception as e:
            self._log(f"[polish] summary skipped ({e})")
            return ""

    def _select(self, segments):
        """Indices to send to the LLM. mode 'all' (default, and always for level 'maximum') = every line that
        contains letters. 'uncertain' = only lines Whisper was unsure about (faster, but skips the rest)."""
        idx = []
        for i, s in enumerate(segments):
            text = s["roman"]
            if not any(c.isalpha() for c in text):
                continue
            if self.mode == "all":
                idx.append(i)
                continue
            if len(text.split()) < 2:
                continue
            c = s.get("conf")
            if c is None or c < self.low_conf or _latin_ratio(text) < 0.95:
                idx.append(i)
        return idx

    # ---- request sizing: never let Ollama truncate the prompt or the answer ----
    @staticmethod
    def _budget(system, user, ids, res):
        body = sum(len(res[i]["text"]) for i in ids)
        est_out = 40 * len(ids) + body / 1.2                 # worst case: every sentence is a FLAG with a guess
        num_predict = max(600, int(est_out * 1.4) + 100)
        need = (len(system) + len(user)) / 2.5 + num_predict + 300   # Roman Urdu is ~2.5 chars per token
        num_ctx = next((c for c in (4096, 8192, 16384, 32768) if need <= c), 32768)
        return dict(num_predict=num_predict, num_ctx=num_ctx)

    def _chunks(self, ids, res):
        """Group sentence indices into requests made of WHOLE paragraphs (all sentences that came from one transcript
        segment stay together, so the model keeps the paragraph's context). A request holds about `chunk_words`
        words / `window` sentences; one paragraph that is bigger than that still goes out whole."""
        groups = []
        for i in ids:
            if groups and not res[i].get("new_seg", True) and groups[-1][-1] == i - 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        chunks, cur, words = [], [], 0
        for g in groups:
            w = sum(len(res[i]["text"].split()) for i in g)
            if cur and (len(cur) + len(g) > self.window or words + w > self.chunk_words):
                chunks.append(cur)
                cur, words = [], 0
            cur += g
            words += w
        if cur:
            chunks.append(cur)
        return chunks

    def _build_user(self, ids, res, rev):
        """Only the chunk itself: no earlier or later lines, no whole-transcript context."""
        user = SECOND_PASS_NOTE if rev else ""
        if self.lexicon is not None:
            hints = self.lexicon.hints(" ".join(res[i]["text"] for i in ids))
            if hints:
                user += HINT_NOTE + hints + "\n\n"
        # small per-request numbers (1..N) are far less error-prone for a small model than transcript-wide ones
        lines = []
        for n, i in enumerate(ids, 1):
            if n > 1 and res[i].get("new_seg", True):
                lines.append("")                      # blank line = next paragraph
            lines.append(f"[{n}] {res[i]['text']}")
        user += (f"SENTENCES TO REVIEW: {len(ids)} numbered sentences, give a verdict for EVERY one "
                 "(sentences of one paragraph are listed together; a blank line starts the next paragraph):\n"
                 + "\n".join(lines))
        return user

    @staticmethod
    def _sim(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower(), autojunk=False).ratio()

    def _align(self, ids, acts, res):
        """Models sometimes attach a fix to the wrong line number (usually off by one). A fix must resemble the
        line it is attached to; if it resembles another line of this request far better, it is moved there;
        otherwise it is dropped (the line then has no verdict and is simply asked again)."""
        moves = {}
        for i, act in acts.items():
            if act.get("flag_fix") and self._sim(res[i]["text"], act["flag_fix"]) < 0.25:
                act["flag_fix"] = None                   # a guess that looks nothing like its line: drop the guess
            fix = act["fix"]
            if not fix:
                continue
            own = self._sim(res[i]["text"], fix)
            if own >= 0.4:
                continue
            act["fix"] = None
            best, score = None, 0.0
            for j in ids:
                if j != i:
                    s = self._sim(res[j]["text"], fix)
                    if s > score:
                        best, score = j, s
            self._log(f"[polish] fix for line {i} does not match it (similarity {own:.2f})"
                      + (f"; belongs to line {best} ({score:.2f}), moved" if best is not None and score >= 0.6
                         and score > own + 0.2 else "; dropped"))
            if best is not None and score >= 0.6 and score > own + 0.2:
                moves.setdefault(best, fix)
        for j, fix in moves.items():
            d = acts.setdefault(j, _blank())
            if d["fix"] is None:
                d["fix"] = fix
        return {i: a for i, a in acts.items() if a["fix"] or a["flag"] or a["ok"]}

    def _tick(self):
        self._done += 1
        el = time.time() - self._t0
        eta = el / self._done * max(0, self._total - self._done)
        self._log(f"[polish] request {self._done}/{self._total}  elapsed {el:.0f}s  eta {eta:.0f}s")
        if self._cb:
            self._cb(min(self._done, self._total), self._total, eta, "polish")

    def _review(self, ids, res, system, rev, attempt=0):
        """Ask for a verdict on every id. Any id without a verdict is re-asked in smaller batches (up to 2 retries)."""
        if attempt:
            self._total += 1
        user = self._build_user(ids, res, rev)
        try:
            reply = self._llm(system, user, **self._budget(system, user, ids, res))
        except Exception as e:
            self._log(f"[polish] lines {ids[0]}-{ids[-1]} failed: {e}")
            reply = ""
        local = {n: i for n, i in enumerate(ids, 1)}          # number shown to the model -> transcript line
        acts = {local[n]: a for n, a in parse_actions(reply, set(local)).items()}
        acts = self._align(ids, acts, res)
        for i, act in acts.items():
            self._apply(res[i], act)
        self._tick()
        missing = [i for i in ids if i not in acts]
        if not missing:
            return
        self._forget(system, user)
        self._log(f"[polish] {len(missing)} of {len(ids)} lines got no verdict"
                  + (", retrying in smaller batches" if attempt < 2 else ""))
        if attempt >= 2:
            if rev == 0:                  # never leave a line silently unchecked
                for i in missing:
                    res[i].update(status="flagged", unchecked=True, text=res[i]["original"],
                                  note="The LLM never reviewed this line. Please check it by hand.")
            return
        step = max(1, (len(missing) + 1) // 2)       # always smaller batches than the one that failed
        for k in range(0, len(missing), step):
            self._review(missing[k:k + step], res, system, rev, attempt + 1)

    def polish(self, segments, on_progress=None):
        """segments: dicts with 'roman' (+ optional 'start','end','conf','speaker'). Segments are merged into complete
        sentences first, so the returned list holds one result dict per SENTENCE (with the start of its first segment).
        on_progress(done, total, eta_seconds, phase) is called after each LLM request."""
        units = build_units(segments, merge=self.merge)
        n = len(units)
        res = [dict(start=u["start"], end=u["end"], original=u["roman"], text=u["roman"],
                    status="ok", note="", suggestion="", unchecked=False, speaker=u["speaker"],
                    new_seg=u.get("new_seg", True), sent=False) for u in units]
        select = self._select(units)
        for i in select:
            res[i]["sent"] = True
        passes = 2 if self.level == "maximum" else 1
        self._log(f"[polish] model {self.model}, level {self.level}, mode {self.mode}, {passes} pass(es): "
                  f"{len(segments)} segments -> {n} sentences, {len(select)} go to the LLM")
        if on_progress:
            on_progress(0, 1, None, "summary")
        summary = self._summarize(units) if select else ""
        system = self._system(summary)
        self.summary_text = summary

        chunks = self._chunks(select, res)
        self._total, self._done, self._t0, self._cb = len(chunks) * passes, 0, time.time(), on_progress
        for rev in range(passes):
            # a flagged sentence is already judged unrepairable, so the second pass only re-reads the others
            todo = select if rev == 0 else [i for i in select if res[i]["status"] != "flagged"]
            for ids in (chunks if rev == 0 else self._chunks(todo, res)):
                self._review(ids, res, system, rev)
        rechecked = 0
        if passes == 1:
            # Anti-laziness: a chunk of real speech from a weak microphone where EVERY sentence got OK is not credible,
            # and a sentence the word list still doubts should not stay OK either. Ask those once more.
            again = set()
            for ids in chunks:
                if all(res[i]["status"] == "ok" for i in ids) and sum(len(res[i]["text"].split()) for i in ids) >= 30:
                    again.update(ids)
            if self.lexicon is not None:
                again.update(i for i in select if res[i]["status"] == "ok" and self.lexicon.hints(res[i]["text"]))
            again = sorted(i for i in again if res[i]["status"] == "ok")
            if again:
                rechecked = len(again)
                more = self._chunks(again, res)
                self._total += len(more)
                self._log(f"[polish] re-checking {rechecked} sentences that were all called OK")
                for ids in more:
                    self._review(ids, res, system, 1)
        if on_progress and self._total:
            on_progress(self._total, self._total, 0.0, "polish")
        self.stats = dict(lines=n, segments=len(segments), sent=len(select),
                          fixed=sum(r["status"] == "fixed" for r in res),
                          flagged=sum(r["status"] == "flagged" for r in res),
                          unchecked=sum(bool(r["unchecked"]) for r in res), rechecked=rechecked,
                          skipped=sum(1 for r in res if not r["sent"] and any(c.isalpha() for c in r["original"])))
        return res

    def _apply(self, r, act):
        fix, flag, guess = act["fix"], act["flag"], act.get("flag_fix")
        if flag:                          # unclear sentence: never edit it, explain it and offer a suggestion
            was = r["text"] if r["text"] != r["original"] else ""   # a pass-1 fix becomes a suggestion
            r.update(status="flagged", note=flag, text=r["original"])
            if fix and " ".join(fix.split()) != r["original"]:
                r["suggestion"] = " ".join(fix.split())
            elif was:
                r["suggestion"] = was
            elif guess and usable_guess(r["original"], guess):
                r["suggestion"] = " ".join(guess.split())
            return
        if fix:
            ok, why = validate_fix(r["original"], fix, self.level)
            if ok:
                r.update(status="fixed", text=" ".join(fix.split()))
            elif why != "unchanged" and r["status"] != "fixed":
                r.update(status="flagged", suggestion=fix,
                         note=f"The model wanted a larger change ({why}), so the original was kept.")


# ---------------------------------------------------------------- rendering
def _esc(s):
    return _html.escape(str(s), quote=True)


def _diff_html(old, new):
    """The NEW text with every changed word highlighted green (hover: the word it replaced)."""
    a, b = old.split(), new.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(_esc(" ".join(b[j1:j2])))
        elif tag in ("replace", "insert"):
            was = " ".join(a[i1:i2]) or "(nothing)"
            out.append(f'<mark class="chg" title="was: {_esc(was)}">{_esc(" ".join(b[j1:j2]))}</mark>')
        else:
            out.append(f'<mark class="del" title="removed: {_esc(" ".join(a[i1:i2]))}">(removed)</mark>')
    return " ".join(out)


def _diff_pair(old, new):
    """-> (old_html, new_html): words that differ are red in the original and green in the suggestion."""
    a, b = old.split(), new.split()
    o, n = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            o.append(_esc(" ".join(a[i1:i2])))
            n.append(_esc(" ".join(b[j1:j2])))
            continue
        if i2 > i1:
            o.append(f'<mark class="rm">{_esc(" ".join(a[i1:i2]))}</mark>')
        if j2 > j1:
            was = " ".join(a[i1:i2]) or "(nothing)"
            n.append(f'<mark class="chg" title="was: {_esc(was)}">{_esc(" ".join(b[j1:j2]))}</mark>')
    return " ".join(o), " ".join(n)


def group_paragraphs(results):
    """Sentence results -> one entry per original transcript paragraph: [{'speaker', 'text', 'start'}]."""
    out = []
    for r in results:
        if out and not r.get("new_seg", True) and out[-1]["speaker"] == r.get("speaker"):
            out[-1]["text"] += " " + r["text"]
        else:
            out.append(dict(speaker=r.get("speaker"), text=r["text"], start=r["start"]))
    return out


def _ts(t):
    if t is None:
        return ""
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


CSS = """<style>
.tr{font-family:system-ui,Segoe UI,Arial,sans-serif;line-height:1.6;max-width:900px}
.tr .para{margin:6px 0}
.tr div.seg{margin:3px 0}
.tr .ts{color:#8a8a8a;font-size:.8em;margin-right:8px}
.tr mark.chg{background:#c9f0c0!important;color:#10300b!important;border-radius:3px;padding:0 3px}
.tr mark.rm{background:#f8c9c9!important;color:#5a0d0d!important;border-radius:3px;padding:0 3px;text-decoration:line-through}
.tr mark.del{background:#f8c9c9!important;color:#5a0d0d!important;border-radius:3px;padding:0 3px;font-size:.8em}
.tr details.flag{margin:4px 0;background:#fff1b8!important;border-left:5px solid #d19a00;border-radius:5px;padding:4px 10px}
.tr details.flag,.tr details.flag *{color:#1b1b1b!important}
.tr details.flag .ts{color:#6b5a00!important}
.tr details.flag summary{cursor:pointer;font-weight:500}
.tr .pop{margin:8px 0 4px;padding:9px 12px;background:#ffffff!important;border:1px solid #d19a00;border-radius:6px;font-size:.93em}
.tr .cmp{margin:8px 0}
.tr .cmp>div{margin:4px 0}
.tr .cmp .lbl{display:inline-block;min-width:104px;font-size:.78em;font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:#6b5a00!important}
.tr .acts{margin-top:8px}
.tr .rvb{font:inherit;font-size:.88em;cursor:pointer;border-radius:6px;padding:3px 12px;margin:0 6px 4px 0;border:1px solid #888;background:#f2f2f2!important;color:#1b1b1b!important}
.tr .rvb:hover{filter:brightness(.94)}
.tr .rvb.ok{background:#c9f0c0!important;border-color:#3d8a31;color:#10300b!important}
.tr .rvb.no{background:#f4f4f4!important;border-color:#8a8a8a}
.tr .rvb.lnk{border:none;background:none!important;text-decoration:underline;padding:0 4px;color:#4a6d9c!important}
.tr .tag{font-size:.78em;border-radius:10px;padding:0 8px;margin:0 4px;color:#1b1b1b!important}
.tr .tag.okt{background:#c9f0c0!important}
.tr .tag.rej{background:#e0e0e0!important}
.tr .rv[data-state="pending"] .v-accepted,.tr .rv[data-state="pending"] .v-rejected{display:none}
.tr .rv[data-state="accepted"] .v-pending,.tr .rv[data-state="accepted"] .v-rejected{display:none}
.tr .rv[data-state="rejected"] .v-pending,.tr .rv[data-state="rejected"] .v-accepted{display:none}
.tr .bar{margin:0 0 12px;padding:6px 0}
.tr .bar .cnt{font-size:.88em;margin-right:10px}
.tr .newspk{margin-top:12px}
.tr .spk{display:inline-block;font-size:.78em;font-weight:600;border-radius:10px;padding:0 8px;margin-right:6px;color:#1b1b1b!important;background:#e0e0e0!important}
.tr .spk.s0{background:#cfe3ff!important}.tr .spk.s1{background:#ffd6d6!important}.tr .spk.s2{background:#d4f0d0!important}
.tr .spk.s3{background:#e6d6ff!important}.tr .spk.s4{background:#ffe0b8!important}.tr .spk.s5{background:#c9f0ee!important}
.tr .spk.s6{background:#f5d0e6!important}
.tr .legend{font-size:.85em;margin-bottom:10px}
.tr .legend .y{background:#fff1b8;color:#1b1b1b;padding:0 5px;border-radius:3px}
</style>"""

# Event handling for the Accept / Keep original buttons. Plain JS, no libraries. Each suggestion row keeps three
# hidden views (pending / accepted / rejected) and the script only switches data-state; CSS shows the matching view.
# Gradio does not run <script> tags inside gr.HTML, so app.py puts this in the page <head> instead.
REVIEW_JS = r"""
(function () {
  if (window.__rvReady) return;
  window.__rvReady = true;
  var STATE = {accept: 'accepted', reject: 'rejected', undo: 'pending'};
  function finalText(tr) {
    var out = [];
    tr.querySelectorAll('.para').forEach(function (p) {
      var parts = [];
      p.querySelectorAll('.row').forEach(function (r) {
        parts.push(r.classList.contains('rv') ? (r.dataset.state === 'accepted' ? r.dataset.sugg : r.dataset.orig) : r.dataset.text);
      });
      out.push((p.dataset.spk ? p.dataset.spk + ': ' : '') + parts.join(' '));
    });
    return out.join('\n');
  }
  function refresh(tr) {
    var el = tr.querySelector('[data-cnt]');
    if (!el) return;
    var c = {pending: 0, accepted: 0, rejected: 0};
    tr.querySelectorAll('.rv').forEach(function (r) { c[r.dataset.state] += 1; });
    el.textContent = c.pending + ' suggestion' + (c.pending === 1 ? '' : 's') + ' waiting \u00b7 ' +
                     c.accepted + ' accepted \u00b7 ' + c.rejected + ' kept as original';
  }
  function flash(b, msg) {
    var old = b.textContent;
    b.textContent = msg;
    setTimeout(function () { b.textContent = old; }, 1400);
  }
  function copy(txt, b) {
    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = txt; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta); flash(b, 'Copied');
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(txt).then(function () { flash(b, 'Copied'); }, fallback);
    } else { fallback(); }
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('.rvb');
    if (!b) return;
    var tr = b.closest('.tr');
    if (!tr) return;
    var a = b.dataset.act;
    if (STATE[a]) {
      var row = b.closest('.rv');
      if (row) row.dataset.state = STATE[a];
    } else if (a === 'accept-all' || a === 'reject-all') {
      tr.querySelectorAll('.rv').forEach(function (r) {
        if (r.dataset.state === 'pending') r.dataset.state = (a === 'accept-all') ? 'accepted' : 'rejected';
      });
    } else if (a === 'copy') {
      copy(finalText(tr), b);
    } else if (a === 'download') {
      var url = URL.createObjectURL(new Blob([finalText(tr) + '\n'], {type: 'text/plain;charset=utf-8'}));
      var l = document.createElement('a');
      l.href = url; l.download = 'final_transcript.txt';
      document.body.appendChild(l); l.click(); document.body.removeChild(l);
      setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    }
    refresh(tr);
    e.preventDefault();
  });
})();
"""


def _spk_badge(name):
    if not name:
        return ""
    m = re.search(r"(\d+)", name)
    k = (int(m.group(1)) - 1) % 7 if m else 7
    return f'<span class="spk s{k}">{_esc(name)}</span>'


def _row_html(r, ts, new):
    """One sentence -> one HTML block. Rows carry data-* so the final text can be rebuilt from the page."""
    spk = r.get("speaker") or ""
    base = f'data-spk="{_esc(spk)}"'
    if r["status"] == "flagged":
        if r.get("unchecked"):
            pop = f'<b>Not reviewed:</b> {_esc(r["note"])}'
        else:
            pop = f'<b>Probably means:</b> {_esc(r["note"])}'
        sugg = r.get("suggestion") or ""
        if not sugg or sugg == r["original"]:
            if not r.get("unchecked"):
                pop += '<br><i>The original text was left unchanged (no rewrite was suggested).</i>'
            return (f'<details class="flag row" data-text="{_esc(r["original"])}" {base}>'
                    f'<summary>{ts}{_esc(r["original"])} &#9888;</summary><div class="pop">{pop}</div></details>')
        old_h, new_h = _diff_pair(r["original"], sugg)
        pop += (f'<div class="cmp"><div><span class="lbl">Original</span>{old_h}</div>'
                f'<div><span class="lbl">AI suggestion</span>{new_h}</div></div>'
                '<div class="acts"><button type="button" class="rvb ok" data-act="accept">Accept AI suggestion</button>'
                '<button type="button" class="rvb no" data-act="reject">Keep original</button></div>')
        return (f'<div class="rv row" data-state="pending" data-orig="{_esc(r["original"])}" '
                f'data-sugg="{_esc(sugg)}" {base}>'
                f'<details class="flag v-pending"><summary>{ts}{_esc(r["original"])} &#9888;</summary>'
                f'<div class="pop">{pop}</div></details>'
                f'<div class="seg v-accepted">{ts}{_diff_html(r["original"], sugg)}'
                '<span class="tag okt">&#10003; AI suggestion accepted</span>'
                '<button type="button" class="rvb lnk" data-act="undo">undo</button></div>'
                f'<div class="seg v-rejected">{ts}{_esc(r["original"])}'
                '<span class="tag rej">original kept</span>'
                '<button type="button" class="rvb lnk" data-act="undo">undo</button></div></div>')
    if r["status"] == "fixed":
        return (f'<span class="seg row" data-text="{_esc(r["text"])}" {base}>'
                f'{ts}{_diff_html(r["original"], r["text"])}</span> ')
    skipped = ('<span class="tag rej" title="Polish mode \'uncertain\' skipped this sentence">not sent to the LLM</span>'
               if not r.get("sent", True) and any(ch.isalpha() for ch in r["original"]) else "")
    return f'<span class="seg row" data-text="{_esc(r["text"])}" {base}>{ts}{_esc(r["text"])}{skipped}</span> '


def render_html(results, standalone=False, title="Transcript review"):
    paras, prev = [], None            # one block per original paragraph; its sentences flow together
    for r in results:
        spk = r.get("speaker")
        if not paras or r.get("new_seg", True) or spk != prev:
            ts = f'<span class="ts">{_ts(r["start"])}</span>' if r["start"] is not None else ""
            new = " newspk" if paras and spk != prev else ""
            paras.append([f'<div class="para{new}" data-spk="{_esc(spk or "")}">{ts}{_spk_badge(spk)}'])
        paras[-1].append(_row_html(r, "", ""))
        prev = spk
    rows = ["".join(p) + "</div>" for p in paras]
    waiting = sum(1 for r in results if r["status"] == "flagged" and r.get("suggestion")
                  and r["suggestion"] != r["original"])
    bar = (f'<div class="bar"><span class="cnt" data-cnt>{waiting} suggestion{"" if waiting == 1 else "s"} waiting'
           ' &middot; 0 accepted &middot; 0 kept as original</span>')
    if waiting:
        bar += ('<button type="button" class="rvb ok" data-act="accept-all">Accept all suggestions</button>'
                '<button type="button" class="rvb no" data-act="reject-all">Keep all originals</button>')
    bar += ('<button type="button" class="rvb" data-act="copy">Copy final text</button>'
            '<button type="button" class="rvb" data-act="download">Download .txt</button></div>')
    legend = ('<div class="legend"><mark class="chg">green</mark> = changed (hover to see the original word) &nbsp; '
              '<span class="y">yellow &#9888;</span> = unclear: click it, compare the original with the AI suggestion '
              '(changed words highlighted) and choose Accept or Keep original</div>')
    body = f'{CSS}<div class="tr">{legend}{bar}{"".join(rows)}</div>'
    if standalone:
        return (f'<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)}</title></head>'
                f'<body>{body}<script>{REVIEW_JS}</script></body></html>')
    return body


def write_polish_outputs(results, base: Path):
    """base: path without extension. Writes .polished.txt, .review.html, .polished.json
    (.polished.txt holds the automatic fixes only; accept/reject choices live in the review page, whose
    'Copy final text' / 'Download .txt' buttons give the text that follows them.)"""
    lines = [f'{r["speaker"]}: {r["text"]}' if r.get("speaker") else r["text"] for r in group_paragraphs(results)]
    (base.parent / (base.name + ".polished.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (base.parent / (base.name + ".review.html")).write_text(render_html(results, standalone=True, title=base.name), encoding="utf-8")
    (base.parent / (base.name + ".polished.json")).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- polish an existing transcript file
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Polish an existing Roman Urdu transcript (one segment per line) with a local LLM")
    ap.add_argument("transcript", help="transcript .json from transcribe.py (best: keeps confidences) or .txt, one line per segment")
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--polish-level", choices=list(LEVELS), default="standard")
    ap.add_argument("--context-file", default="context_tmc.txt")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--polish-mode", choices=["uncertain", "all"], default="all")
    ap.add_argument("--low-conf", type=float, default=-0.5)
    ap.add_argument("--window", type=int, default=None, help="sentences per LLM request (default 4; 3 for maximum)")
    ap.add_argument("--context-minutes", type=float, default=CONTEXT_MINUTES,
                    help="the topic summary is built from the first N minutes only (default 5)")
    ap.add_argument("--lexicon", default=None,
                    help="public word list(s), comma separated, used as spelling hints (see lexicon.py)")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    raw = Path(a.transcript).read_text(encoding="utf-8")
    if a.transcript.lower().endswith(".json"):
        segs = [s for s in json.loads(raw) if s.get("roman", "").strip()]
    else:   # plain text has no confidences, so every line is sent
        segs = [dict(start=None, end=None, roman=l.strip()) for l in raw.splitlines() if l.strip()]
    p = Polisher(model=a.llm_model, level=a.polish_level, window=a.window, context_file=a.context_file,
                 summary=not a.no_summary, mode=a.polish_mode, low_conf=a.low_conf,
                 context_minutes=a.context_minutes, lexicon=a.lexicon)
    p.check()
    res = p.polish(segs)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    write_polish_outputs(res, Path(a.out_dir) / Path(a.transcript).stem)
    print(f"[polish: {p.stats['fixed']} fixed, {p.stats['flagged']} flagged "
          f"({p.stats['unchecked']} not reviewed) of {p.stats['lines']} sentences "
          f"({p.stats['sent']} sent to the LLM)]", file=sys.stderr)


if __name__ == "__main__":
    main()
