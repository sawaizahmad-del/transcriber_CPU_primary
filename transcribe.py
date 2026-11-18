#!/usr/bin/env python3
"""Fast, CPU-only, free Roman Hindi/Urdu speech transcriber.

Pipeline: audio -> faster-whisper (CTranslate2 int8, CPU, VAD, batched, anti-loop decoding)
          -> guards: loops/hallucinations and cut-off ("?" / U+FFFD) segments are re-decoded or dropped
          -> Roman text (directly from a Hinglish model, or via translit.py for Devanagari models)
          -> optional local-LLM polish (polish.py)

Usage:
    python transcribe.py talk.mp3
    python transcribe.py talk.mp3 --autotune          # one-time: find the fastest threads/batch for THIS CPU
    python transcribe.py talk.mp3 --polish            # + local LLM polish (needs Ollama)
    python transcribe.py talk.mp3 --polish --polish-level maximum   # fix every last detail (slowest)
    python transcribe.py talk.mp3 --speakers                        # label who talks: Person 1, Person 2, ...
    python transcribe.py talk.mp3 --speakers --num-speakers 4       # most accurate if you know how many people
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from audio_utils import SR, clean_audio, collapse_repeats, has_truncation, is_degenerate, load_audio
from translit import _AR_RE as ARABIC_RE
from translit import to_roman

# roman=True : model writes Roman Hinglish itself (English words stay English) -> language "en"
# roman=False: model writes Devanagari (language "hi") -> romanized by translit.py. Devanagari costs ~3x more
#              decoder tokens than Latin, so these presets decode shorter chunks ("chunk") to avoid cut-off text.
# hf_id      : preset needs a one-time local conversion:  python convert_model.py <hf_id>
# name       : human-readable description shown in the UI / --help (so "fast" etc. say which model they really are)
# lang       : Whisper language code override (default: "en" for roman presets, "hi" otherwise)
# hf_subfolder: the CTranslate2 files live in this subfolder of the Hugging Face repo in "model" (downloaded on first use)
# extra      : additional faster-whisper decoding options for this preset
PRESETS = {
    "swift":       dict(name="Hindi2Hinglish Swift (Whisper base, fine-tuned) -> Roman directly",
                        model="MSpider3/whisper-hindi2hinglish-swift-ct2", beam_size=3, roman=True),
    "prime":       dict(name="Hindi2Hinglish Prime (Whisper large-v3, fine-tuned) -> Roman directly",
                        model="Hub84/faster-whisper-hinglish-prime", beam_size=3, roman=True),
    "apex":        dict(name="Hindi2Hinglish Apex (~800M, fine-tuned) -> Roman directly; needs one-time convert",
                        model="models/Whisper-Hindi2Hinglish-Apex-ct2", beam_size=3, roman=True,
                        hf_id="Oriserve/Whisper-Hindi2Hinglish-Apex"),
    "hindi-small": dict(name="Whisper small, fine-tuned on Hindi (Collabora) -> Devanagari->Roman; needs one-time convert",
                        model="models/whisper-small-hindi-ct2", beam_size=3, roman=False, chunk=15,
                        hf_id="collabora/whisper-small-hindi"),
    "sukhan":      dict(name="Urdu Sukhan ASR v1 (Whisper large-v3-turbo, fine-tuned for Urdu + Urdu/English) -> Urdu script->Roman",
                        model="Loudness-ai/urdu-sukhan-asr-v1", hf_subfolder="ctranslate2",
                        beam_size=5, roman=False, lang="ur", extra=dict(suppress_blank=False)),
    "fast":        dict(name="OpenAI Whisper small (generic, not Hindi-tuned) -> Devanagari->Roman",
                        model="small",          beam_size=3, roman=False, chunk=15),
    "balanced":    dict(name="OpenAI Whisper large-v3-turbo (generic, not Hindi-tuned) -> Devanagari->Roman",
                        model="large-v3-turbo", beam_size=2, roman=False, chunk=15),
    "best":        dict(name="OpenAI Whisper large-v3 (generic, not Hindi-tuned) -> Devanagari->Roman",
                        model="large-v3",       beam_size=3, roman=False, chunk=15),
}
DEFAULT_PRESET = "swift"


def preset_label(key: str) -> str:
    """'swift: Hindi2Hinglish Swift (...) [MSpider3/whisper-hindi2hinglish-swift-ct2]' for menus and --help."""
    cfg = PRESETS[key]
    src = cfg.get("hf_id") or (cfg["model"] if "/" in cfg["model"] else "")
    return f"{key}: {cfg['name']}" + (f" [{src}]" if src else "")


def _resolve_model(cfg) -> str:
    """Local path or model name for faster-whisper. Downloads a repo subfolder (Urdu Sukhan) on first use."""
    sub = cfg.get("hf_subfolder")
    if not sub:
        return cfg["model"]
    try:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(cfg["model"], allow_patterns=[f"{sub}/*"])
    except Exception as e:
        raise RuntimeError(
            f"Could not download '{cfg['model']}' ({type(e).__name__}: {e}).\n"
            "This model is gated (manual approval, licence CC BY-NC 4.0 = non-commercial use only):\n"
            f"  1) open https://huggingface.co/{cfg['model']} and request access\n"
            "  2) create a READ token at https://huggingface.co/settings/tokens and run:  hf auth login\n"
            "  3) run this again (the ~0.8 GB download happens once)") from e
    return str(Path(repo_dir) / sub)
TUNING_FILE = Path(__file__).with_name("tuning.json")

# Second-pass settings for segments the guard flagged (slower, stricter; only used on a few segments).
RESCUE_KW = dict(
    beam_size=5, temperature=[0.0, 0.2, 0.4, 0.6], repetition_penalty=1.2, no_repeat_ngram_size=3,
    condition_on_previous_text=False, initial_prompt=None, vad_filter=False,
    compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
)


def _supported(fn, kwargs: dict) -> dict:
    """Drop options the installed faster-whisper version doesn't know about."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def load_tuning(preset):
    try:
        return json.loads(TUNING_FILE.read_text()).get(f"{preset}|{os.cpu_count()}", {})
    except Exception:
        return {}


class Transcriber:
    def __init__(self, preset=DEFAULT_PRESET, threads=None, model_dir=None, clean=False,
                 denoise=False, repetition_penalty=1.1, hotwords=None, batch_size=None,
                 beam_size=None, rescue=True):
        from faster_whisper import WhisperModel

        cfg = PRESETS[preset]
        if "hf_id" in cfg and not Path(cfg["model"]).exists():
            raise FileNotFoundError(
                f"Preset '{preset}' needs a one-time local conversion:\n"
                "  pip install ctranslate2 transformers torch\n"
                f"  python convert_model.py {cfg['hf_id']}"
            )
        tuned = load_tuning(preset)
        self.cfg, self.preset = cfg, preset
        self.clean, self.denoise, self.rep_pen = clean, denoise, repetition_penalty
        self.hotwords, self.rescue = hotwords, rescue
        self.batch_size = batch_size or tuned.get("batch_size") or 8
        self.threads = threads or tuned.get("threads") or os.cpu_count() or 4
        self.beam_size = beam_size or cfg["beam_size"]
        self.last_stats = {}
        self.model = WhisperModel(_resolve_model(cfg), device="cpu", compute_type="int8",
                                  cpu_threads=self.threads, download_root=model_dir)
        try:
            from faster_whisper import BatchedInferencePipeline
            self.batched = BatchedInferencePipeline(self.model)
        except ImportError:
            self.batched = None
        self._suppress = None

    # ---- decoding options ----
    def _arabic_suppress_ids(self):
        """Token ids of Perso-Arabic script. Devanagari presets must not drift into Urdu script mid-sentence."""
        if self._suppress is None:
            ids = []
            try:
                tok = self.model.hf_tokenizer
                for i in range(tok.get_vocab_size()):
                    if ARABIC_RE.search(tok.decode([i])):
                        ids.append(i)
            except Exception:
                ids = []
            self._suppress = ids
        return self._suppress

    def _decode_kwargs(self):
        roman = self.cfg["roman"]
        chunk = self.cfg.get("chunk", 30)
        lang = self.cfg.get("lang", "en" if roman else "hi")
        kw = dict(
            language=lang,
            task="transcribe",
            beam_size=self.beam_size,
            repetition_penalty=self.rep_pen,
            no_repeat_ngram_size=3,
            temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=None,
            hotwords=self.hotwords,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, max_speech_duration_s=chunk),
        )
        if chunk != 30:
            kw["chunk_length"] = chunk
        if not roman and lang != "ur":        # Devanagari presets must not drift into Urdu script; Urdu presets WRITE it
            ids = self._arabic_suppress_ids()
            if ids:
                kw["suppress_tokens"] = [-1] + ids
        kw.update(self.cfg.get("extra", {}))
        return kw

    # ---- second-pass helpers ----
    def _decode_slice(self, a):
        kw = {**self._decode_kwargs(), **RESCUE_KW}
        fn = self.model.transcribe       # sequential decoder: full temperature fallback
        try:
            segs, _ = fn(a, **_supported(fn, kw))
            return " ".join(x.text.strip() for x in segs).strip()
        except Exception:
            return ""

    @staticmethod
    def _quietest_point(a):
        """Sample index of the quietest 100 ms in the middle third (a natural place to cut)."""
        n = len(a)
        lo, hi = n // 3, 2 * n // 3
        frame = SR // 50
        k = (hi - lo) // frame
        if k < 5:
            return n // 2
        e = (a[lo: lo + k * frame].reshape(k, frame) ** 2).mean(axis=1)
        e = np.convolve(e, np.ones(5) / 5, mode="same")
        return lo + int(np.argmin(e)) * frame + frame // 2

    def _decode_span(self, a, depth=0, force_split=False):
        """Decode an audio slice; if the text is cut off (token overflow), split at a quiet point and retry."""
        if len(a) < 0.4 * SR:
            return ""
        if force_split and len(a) > 4 * SR:
            cut = self._quietest_point(a)
            return " ".join(t for t in (self._decode_span(a[:cut], depth + 1),
                                        self._decode_span(a[cut:], depth + 1)) if t)
        text = self._decode_slice(a)
        if has_truncation(text) and depth < 2 and len(a) > 4 * SR:
            return self._decode_span(a, depth, force_split=True)
        return collapse_repeats(text.replace("�", "").strip())

    def _slice(self, audio, start, end, pad=0.25):
        return audio[max(0, int((start - pad) * SR)): int((end + pad) * SR)]

    # ---- main entry points ----
    def _decode(self, audio):
        kw = self._decode_kwargs()
        if self.batched is not None:
            try:
                fn = self.batched.transcribe
                raw, _ = fn(audio, batch_size=self.batch_size, **_supported(fn, kw))
                return list(raw)
            except Exception as e:
                print(f"[batched decoding unavailable ({type(e).__name__}); using sequential]", file=sys.stderr)
        fn = self.model.transcribe
        raw, _ = fn(audio, **_supported(fn, kw))
        return list(raw)

    def transcribe(self, audio_path: str):
        """Returns (segments, duration_seconds); stats in self.last_stats.
        Segment: dict(start, end, native, roman, rescued, conf)."""
        audio = load_audio(audio_path)
        return self.transcribe_audio(audio)

    def transcribe_audio(self, audio):
        duration = len(audio) / SR
        if self.clean:
            audio = clean_audio(audio, denoise=self.denoise)
        raw = self._decode(audio)

        stats = dict(total=0, flagged=0, truncated=0, rescued=0, dropped=0)
        segments = []
        for s in raw:
            text = s.text.strip()
            if not text:
                continue
            stats["total"] += 1
            rescued = False
            if has_truncation(text):                      # cut-off text: the "?" / U+FFFD problem
                stats["truncated"] += 1
                if self.rescue:
                    fixed = self._decode_span(self._slice(audio, s.start, s.end), force_split=True)
                    if fixed:
                        text, rescued = fixed, True
                        stats["rescued"] += 1
                text = text.replace("�", "")
            if is_degenerate(to_roman(text), getattr(s, "no_speech_prob", None), getattr(s, "avg_logprob", None)):
                stats["flagged"] += 1
                if not self.rescue:
                    text = collapse_repeats(text)
                else:
                    retry = self._decode_span(self._slice(audio, s.start, s.end))
                    if retry and not is_degenerate(to_roman(retry)):
                        text, rescued = retry, True
                        stats["rescued"] += 1
                    else:
                        stats["dropped"] += 1             # better to omit than to emit a hallucination
                        continue
            native = collapse_repeats(text)
            roman = collapse_repeats(to_roman(native))
            if roman.strip():
                segments.append(dict(start=s.start, end=s.end, native=native, roman=roman, rescued=rescued,
                                     conf=getattr(s, "avg_logprob", None)))
        self.last_stats = stats
        return segments, duration


# ---------- one-time speed tuning ----------
def autotune(preset, audio_path, seconds=45, log=lambda m: print(m, file=sys.stderr)):
    """Time a few thread/batch combinations on the first `seconds` of a real file and remember the fastest."""
    audio = load_audio(audio_path)[: seconds * SR]
    if len(audio) < 20 * SR:
        log("[autotune skipped: need at least 20 s of audio]")
        return {}
    n = os.cpu_count() or 4
    results = []
    for t in sorted({max(2, n // 2), n}):
        tr = Transcriber(preset, threads=t, rescue=False)
        tr.transcribe_audio(audio[: 6 * SR])              # warm-up
        for b in (8, 16):
            tr.batch_size = b
            t0 = time.time()
            tr.transcribe_audio(audio)
            took = time.time() - t0
            results.append((took, t, b))
            log(f"[autotune] threads={t} batch={b}: {len(audio) / SR / took:.1f}x realtime")
        del tr
    took, t, b = min(results)
    try:
        data = json.loads(TUNING_FILE.read_text()) if TUNING_FILE.exists() else {}
    except Exception:
        data = {}
    data[f"{preset}|{n}"] = dict(threads=t, batch_size=b)
    TUNING_FILE.write_text(json.dumps(data, indent=2))
    log(f"[autotune] best: threads={t}, batch={b} (saved to {TUNING_FILE.name})")
    return dict(threads=t, batch_size=b)


# ---------- output writers ----------
def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_outputs(segments, base: Path, formats, native: bool):
    key = "native" if native else "roman"
    p = lambda ext: base.parent / (base.name + ext)   # base.with_suffix() would clip names containing dots
    text = lambda s: f"{s['speaker']}: {s[key]}" if s.get("speaker") else s[key]   # "Person 1: ..." when speakers are on
    if "txt" in formats:
        p(".txt").write_text("\n".join(text(s) for s in segments) + "\n", encoding="utf-8")
    if "srt" in formats:
        lines = []
        for i, s in enumerate(segments, 1):
            lines += [str(i), f"{_ts(s['start'])} --> {_ts(s['end'])}", text(s), ""]
        p(".srt").write_text("\n".join(lines), encoding="utf-8")
    if "json" in formats:
        p(".json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="CPU-only Roman Hindi/Urdu transcriber",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Speech presets (--preset):\n" + "\n".join("  " + preset_label(k) for k in PRESETS))
    ap.add_argument("audio", nargs="+", help="audio/video files (mp3, wav, m4a, mp4, ogg, ...)")
    ap.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET, help="speech model (list below)")
    ap.add_argument("--format", default="txt,json", help="comma list of: txt,srt,json (json lets polish.py reuse confidences)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--native", action="store_true", help="write raw model text instead of Roman")
    ap.add_argument("--autotune", action="store_true", help="one-time: find the fastest threads/batch for this CPU")
    ap.add_argument("--hotwords", default=None, help='space-separated terms to favor, e.g. "online income email"')
    ap.add_argument("--clean", action="store_true", help="light audio cleanup (high-pass + loudness); off by default")
    ap.add_argument("--denoise", action="store_true", help="noise reduction (needs --clean; usually HURTS Whisper)")
    ap.add_argument("--no-rescue", action="store_true", help="don't re-decode loop/cut-off segments")
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--beam-size", type=int, default=None, help="lower = faster (1 = greedy)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--model-dir", default=None)
    # --- local LLM polish pass (needs Ollama) ---
    ap.add_argument("--polish", action="store_true", help="polish grammar/spelling with a local LLM (Ollama) and flag unclear lines")
    ap.add_argument("--llm-model", default=None, help="Ollama model name (default: gemma4:e4b; gemma4:e2b is faster)")
    ap.add_argument("--polish-level", choices=["conservative", "standard", "maximum"], default="standard",
                    help="conservative = spelling/grammar only; standard = also repair clearly mis-transcribed words; "
                         "maximum = fix every detail (two review passes, smaller chunks, always every sentence; slowest)")
    ap.add_argument("--polish-mode", choices=["uncertain", "all"], default="all",
                    help="all (default) = every line is reviewed; uncertain = faster, only lines Whisper was unsure about "
                         "(skips the rest; ignored for --polish-level maximum)")
    ap.add_argument("--low-conf", type=float, default=-0.5, help="lines with avg log-prob below this count as uncertain")
    ap.add_argument("--context-file", default="context_tmc.txt", help="background text for the LLM (names, jargon)")
    ap.add_argument("--no-summary", action="store_true", help="skip the auto meeting-summary step")
    ap.add_argument("--window", type=int, default=None, help="sentences per LLM request (default 4; 3 for maximum)")
    ap.add_argument("--context-minutes", type=float, default=5.0,
                    help="the LLM's topic summary is built from the first N minutes only (default 5)")
    ap.add_argument("--lexicon", default=None,
                    help="public word list(s), comma separated, used as spelling hints for the LLM (see lexicon.py)")
    # --- speaker identification (Person 1, Person 2, ...) ---
    from speakers import DEFAULT_DIARIZER, DIARIZERS, diarizer_label
    ap.add_argument("--speakers", action="store_true", help="label who is talking: Person 1, Person 2, ...")
    ap.add_argument("--diarizer", choices=list(DIARIZERS), default=DEFAULT_DIARIZER,
                    help="speaker model: " + "; ".join(diarizer_label(k) for k in DIARIZERS))
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="exact number of people in the recording, if you know it (most accurate)")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--no-split-speakers", action="store_true",
                    help="never split a transcript line at a speaker change (line goes to whoever talks most)")
    args = ap.parse_args()

    formats = {f.strip() for f in args.format.split(",")}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    polisher = None
    if args.polish:
        from polish import DEFAULT_MODEL, Polisher
        polisher = Polisher(model=args.llm_model or DEFAULT_MODEL, level=args.polish_level, window=args.window,
                            context_file=args.context_file, summary=not args.no_summary,
                            mode=args.polish_mode, low_conf=args.low_conf,
                            context_minutes=args.context_minutes, lexicon=args.lexicon)
        polisher.check()   # fail early with install instructions instead of after a long transcription

    if args.speakers:
        from speakers import check_backend
        try:
            check_backend(args.diarizer)   # fail early with install/login instructions, not after the transcription
        except RuntimeError as e:
            sys.exit(str(e))

    if args.autotune:
        autotune(args.preset, args.audio[0])

    t0 = time.time()
    tr = Transcriber(args.preset, args.threads, args.model_dir, clean=args.clean or args.denoise,
                     denoise=args.denoise, repetition_penalty=args.repetition_penalty,
                     hotwords=args.hotwords, batch_size=args.batch_size, beam_size=args.beam_size,
                     rescue=not args.no_rescue)
    print(f"[{preset_label(args.preset)}]", file=sys.stderr)
    print(f"[preset '{args.preset}' loaded in {time.time() - t0:.1f}s | threads {tr.threads}, batch {tr.batch_size}, "
          f"beam {tr.beam_size}]", file=sys.stderr)

    for path in args.audio:
        t1 = time.time()
        audio = load_audio(path)
        segments, duration = tr.transcribe_audio(audio)
        took = time.time() - t1
        st = tr.last_stats
        print(f"[{Path(path).name}: {duration:.0f}s audio in {took:.1f}s = {duration / max(took, 1e-6):.1f}x realtime | "
              f"segments {st['total']}, cut-off {st['truncated']}, loops {st['flagged']}, "
              f"rescued {st['rescued']}, dropped {st['dropped']}]", file=sys.stderr)
        if args.speakers:
            from speakers import assign_speakers, run_diarization, speaker_summary_text
            t3 = time.time()
            try:
                turns = run_diarization(audio, args.diarizer, args.num_speakers, args.min_speakers, args.max_speakers)
                segments = assign_speakers(segments, turns, split=not args.no_split_speakers)
                print(f"[speakers ({args.diarizer}): {speaker_summary_text(segments)} | "
                      f"{time.time() - t3:.0f}s | {len(segments)} lines]", file=sys.stderr)
            except RuntimeError as e:        # keep the transcript even if diarization fails
                print(f"[speaker identification failed, continuing without it: {e}]", file=sys.stderr)
        del audio
        for s in segments:
            line = s["native"] if args.native else s["roman"]
            print(f"{s['speaker']}: {line}" if s.get("speaker") else line)
        base = out_dir / Path(path).stem
        write_outputs(segments, base, formats, args.native)
        if polisher:
            from polish import write_polish_outputs
            t2 = time.time()
            results = polisher.polish(segments)
            write_polish_outputs(results, base)
            st = polisher.stats
            print(f"[polish ({args.polish_level}): {st['fixed']} fixed, {st['flagged']} flagged "
                  f"({st['unchecked']} not reviewed) of {st['lines']} sentences "
                  f"({st['sent']} sent to the LLM) in {time.time() - t2:.0f}s -> {base.name}.polished.txt / .review.html]",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
