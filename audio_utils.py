"""Audio cleanup + repetition-loop removal (numpy only; noisereduce optional)."""
import re

import numpy as np

SR = 16000


def load_audio(path: str) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 (uses PyAV bundled with faster-whisper)."""
    from faster_whisper import decode_audio
    return decode_audio(path, sampling_rate=SR)


def _highpass(x: np.ndarray, cutoff: float = 80.0, block: int = 1 << 20) -> np.ndarray:
    """Remove rumble/DC below `cutoff` Hz (FFT filter, processed in blocks)."""
    out = np.empty_like(x)
    for i in range(0, len(x), block):
        seg = x[i:i + block]
        spec = np.fft.rfft(seg)
        spec[np.fft.rfftfreq(len(seg), 1.0 / SR) < cutoff] = 0
        out[i:i + block] = np.fft.irfft(spec, n=len(seg)).astype(np.float32)
    return out


def _normalize(x: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 20.0) -> np.ndarray:
    """Bring speech loudness to a consistent level (measured on non-silent frames only)."""
    frame = SR // 50
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    voiced = rms[rms > 10 ** (-50 / 20)]
    if voiced.size == 0:
        return x
    cur = float(np.sqrt((voiced ** 2).mean()))
    gain = min(10 ** (target_dbfs / 20) / cur, 10 ** (max_gain_db / 20))
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


def _denoise(x: np.ndarray) -> np.ndarray:
    try:
        import noisereduce as nr
    except ImportError:
        print("[denoise skipped: pip install noisereduce]")
        return x
    return nr.reduce_noise(y=x, sr=SR, stationary=True, prop_decrease=0.8).astype(np.float32)


def clean_audio(x: np.ndarray, denoise: bool = False) -> np.ndarray:
    x = _highpass(x.astype(np.float32, copy=False))
    if denoise:
        x = _denoise(x)
    return _normalize(x)


# ---- repetition-loop removal: "aap aap aap aap" -> "aap" (3+ identical repeats of a 1-4 word phrase)
_T = r"[^\s,.;:!?\-\u0964]"           # a token character (script-agnostic)
_LOOP = re.compile(
    rf"(?<!{_T})({_T}+(?:\s+{_T}+){{0,3}}?)(?:[\s,.;:!?\-\u0964]+\1(?!{_T})){{2,}}",
    re.IGNORECASE,
)


def collapse_repeats(text: str) -> str:
    return _LOOP.sub(r"\1", text)


# ---- hallucination detection (metric/pattern based, same idea as Whisper's own compression-ratio check)
import zlib


def compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def is_degenerate(text: str, no_speech_prob=None, avg_logprob=None) -> bool:
    """True if a segment looks like a decoding loop / hallucination."""
    words = text.split()
    if len(words) >= 6:
        if len(collapse_repeats(text).split()) < 0.6 * len(words):
            return True                       # most of the text was a repeated loop
        if len(set(w.lower() for w in words)) / len(words) < 0.35:
            return True                       # very low vocabulary diversity
    if len(text) >= 24 and compression_ratio(text) > 2.4:
        return True                           # repetitive text compresses too well
    if no_speech_prob is not None and avg_logprob is not None:
        if no_speech_prob > 0.8 and avg_logprob < -1.5:
            return True                       # model itself says "no speech, low confidence"
    return False


def has_truncation(text: str) -> bool:
    """U+FFFD at the end/inside a segment = the decoder ran out of tokens in the middle of a character."""
    return "\ufffd" in text
