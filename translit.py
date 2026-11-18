"""Devanagari -> Roman Hindi/Urdu ("Hinglish"-style) transliteration.

Why Devanagari? Spoken Hindi and Urdu are the same language (Hindustani).
Whisper's Hindi decoding is its strongest for this language, and Devanagari
writes every vowel explicitly, so we can romanize it deterministically
(Urdu script omits short vowels, which makes Roman output a guessing game).
Urdu-origin words come out with nukta letters (ज़ ख़ क़ फ़ ग़) which we map
to z, kh, q, f, gh.

Pure rule-based: runs in microseconds, no model, no network.
"""
import re
import unicodedata

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "w",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
NUKTA = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
IND_V = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऋ": "ri",
}
MATRA = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ृ": "ri",
}
VIRAMA, NUKTA_MARK = "्", "़"
NASALS = "ंँ"
LABIALS = ("p", "b", "m")

_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_PUNCT = {"।": ".", "॥": "."}

# Very common words where the casual Roman spelling differs from the strict rules.
_LEXICON_RAW = {
    "है": "hai", "हैं": "hain", "मैं": "main", "में": "mein", "नहीं": "nahi",
    "नही": "nahi", "और": "aur", "को": "ko", "का": "ka", "की": "ki", "के": "ke",
    "से": "se", "पर": "par", "यह": "yeh", "ये": "ye", "वो": "wo", "वह": "woh",
    "क्या": "kya", "कि": "ke", "कुछ": "kuch", "भी": "bhi", "तो": "to",
    "ही": "hi", "हो": "ho", "था": "tha", "थी": "thi", "थे": "the",
    "कर": "kar", "मुझे": "mujhe", "तुम": "tum", "आप": "aap", "हम": "hum",
    "यहाँ": "yahan", "वहाँ": "wahan", "कहाँ": "kahan", "क्यों": "kyun",
    "अच्छा": "acha", "अच्छी": "achi", "अच्छे": "ache", "बहुत": "bohat",
    "ठीक": "theek", "कैसे": "kaise", "कैसा": "kaisa", "कैसी": "kaisi",
    "हूँ": "hoon", "हूं": "hoon", "तुम्हें": "tumhein", "उन्हें": "unhein",
    "उसे": "usse", "इसे": "isse", "लिए": "liye", "साथ": "saath",
    "एक": "ek", "दो": "do", "जी": "ji", "हाँ": "haan", "ना": "na",
}
LEXICON = {unicodedata.normalize("NFC", k): v for k, v in _LEXICON_RAW.items()}

_WORD_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")

# characters that show up as "?" / boxes: U+FFFD (a cut-off multi-byte character) and invisible marks
_ARTIFACTS = re.compile("[\ufffd\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

# ---- fallback for stray Urdu (Perso-Arabic) script that Whisper sometimes mixes into Devanagari output
_AR_RE = re.compile("[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+")
_AR_DIAC = re.compile("[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_AR_MAP = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh",
    "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh",
    "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k",
    "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "ہ": "h", "ۃ": "h", "ە": "h", "ه": "h",
}
_AR_LEX = {
    "ہیں": "hain", "ہے": "hai", "ہو": "ho", "ہوں": "hoon", "کی": "ki", "کے": "ke", "کا": "ka", "کو": "ko",
    "سے": "se", "میں": "mein", "اور": "aur", "یہ": "yeh", "وہ": "woh", "تو": "to", "بھی": "bhi",
    "نہیں": "nahi", "کر": "kar", "آپ": "aap", "ہم": "hum", "جو": "jo", "پر": "par", "سکتے": "sakte",
    "سکتا": "sakta", "سکتی": "sakti", "کرنا": "karna", "کریں": "karein", "ہی": "hi", "ایک": "ek",
}


def _romanize_arabic_word(w: str) -> str:
    w = _AR_DIAC.sub("", w)
    if w in _AR_LEX:
        return _AR_LEX[w]
    toks, n = [], len(w)              # [roman, kind] kind: C consonant / V vowel
    for i, ch in enumerate(w):
        prev_c = bool(toks) and toks[-1][1] == "C"
        if ch == "ا":
            toks.append(["a" if i in (0, n - 1) else "aa", "V"])
        elif ch == "آ":
            toks.append(["aa", "V"])
        elif ch == "و":
            toks.append(["o", "V"] if (i and prev_c) else ["w", "C"])
        elif ch in "یيئ":
            toks.append(["i", "V"] if (i and prev_c) else ["y", "C"])
        elif ch in "ےۓ":
            toks.append(["e", "V"])
        elif ch == "ؤ":
            toks.append(["o", "V"])
        elif ch == "ع":
            toks.append(["a", "V"])
        elif ch == "ھ":
            if prev_c and not toks[-1][0].endswith("h"):
                toks[-1][0] += "h"
            else:
                toks.append(["h", "C"])
        elif ch == "ء":
            continue
        elif ch in _AR_MAP:
            toks.append([_AR_MAP[ch], "C"])
        else:
            toks.append([ch, "V"])
    if toks and all(k == "C" for _, k in toks) and len(toks) == 2:
        return toks[0][0] + "a" + toks[1][0]
    out = []
    for j, (r, k) in enumerate(toks):
        out.append(r)
        if k == "C" and j + 1 < len(toks) and toks[j + 1][1] == "C":
            if j + 2 < len(toks) and toks[j + 2][1] == "C":
                out.append("a")                       # sa|k|te  -> sakte
            elif j == 0 and j + 2 < len(toks):
                out.append("i")                       # initial cluster: ktaab -> kitaab
    return "".join(out)


def roman_key(word: str) -> str:
    """Phonetic-ish key so spelling variants compare equal: zindagi/zindagee, kaise/kese, theek/thik."""
    w = word.lower()
    for a, b in (("chh", "ch"), ("ph", "f"), ("ee", "i"), ("ii", "i"), ("ie", "i"), ("aa", "a"),
                 ("oo", "u"), ("ou", "u"), ("ai", "e"), ("ei", "e"), ("ay", "e"), ("ae", "e"),
                 ("au", "o"), ("w", "v"), ("q", "k"), ("x", "ks"), ("y", "i")):
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)


def _parse(word: str):
    units = []  # each: dict(c, v, nasal, tail); v: None=inherent a, ""=killed, str=explicit
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in CONS:
            c = CONS[ch]
            if i + 1 < len(word) and word[i + 1] == NUKTA_MARK:
                c = NUKTA.get(ch, c)
                i += 1
            units.append({"c": c, "v": None, "nasal": False, "tail": ""})
        elif ch == VIRAMA:
            if units:
                units[-1]["v"] = ""
        elif ch in MATRA:
            if units:
                units[-1]["v"] = MATRA[ch]
        elif ch in NASALS:
            if units:
                units[-1]["nasal"] = True
        elif ch == "ः":
            if units:
                units[-1]["tail"] = "h"
        elif ch in IND_V:
            units.append({"c": "", "v": IND_V[ch], "nasal": False, "tail": ""})
        elif ch == NUKTA_MARK:
            pass
        else:
            units.append({"c": ch, "v": "", "nasal": False, "tail": ""})
        i += 1
    return units


def _romanize_word(word: str) -> str:
    if word in LEXICON:
        return LEXICON[word]
    word = word.replace("ज्ञ", "ग्य")
    units = _parse(word)
    n = len(units)
    if n == 0:
        return ""
    deleted = [False] * n
    for i, u in enumerate(units):
        if u["v"] is not None or not u["c"]:
            continue
        if i == n - 1:
            # word-final inherent schwa is silent (multi-unit words only)
            if n > 1:
                deleted[i] = True
            continue
        if i == 0:
            continue
        prev, nxt = units[i - 1], units[i + 1]
        if (
            nxt["v"] not in (None, "")
            and prev["v"] != ""
            and not prev["nasal"]
            and not deleted[i - 1]
        ):
            deleted[i] = True  # medial schwa deletion: kam(a)laa -> kamlaa
    out = []
    for i, u in enumerate(units):
        v = "" if deleted[i] or u["v"] == "" else ("a" if u["v"] is None else u["v"])
        s = u["c"] + v
        if u["nasal"]:
            nxt_c = units[i + 1]["c"] if i + 1 < n else ""
            s += "m" if nxt_c.startswith(LABIALS) else "n"
        out.append(s + u["tail"])
    r = "".join(out)
    # casual Roman spelling: word-final long vowels are written short (hamaaraa -> hamara)
    if len(r) > 2:
        if r.endswith("aa"):
            r = r[:-1]
        elif r.endswith("ee"):
            r = r[:-2] + "i"
    return r


def to_roman(text: str) -> str:
    """Romanize any Devanagari in `text`; Latin text and punctuation pass through."""
    text = _ARTIFACTS.sub("", unicodedata.normalize("NFC", text)).translate(_DIGITS)
    for k, v in _PUNCT.items():
        text = text.replace(k, v)
    text = _WORD_RE.sub(lambda m: _romanize_word(m.group(0)), text)
    text = _AR_RE.sub(lambda m: _romanize_arabic_word(m.group(0)), text)
    text = text.replace("\u060c", ",").replace("\u061f", "?").replace("\u06d4", ".")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


if __name__ == "__main__":
    import sys
    print(to_roman(" ".join(sys.argv[1:]) or sys.stdin.read()))
