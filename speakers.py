"""Speaker identification ("Person 1", "Person 2", ...) for transcripts. CPU-only.

Pipeline:  audio -> diarization backend -> speaker turns (start, end, label)
           -> assign_speakers(): every transcript segment gets the speaker who talks during it
              (a segment that spans a change of speaker is split at the change)
           -> labels are renamed by order of first appearance: Person 1 is whoever speaks first.

A diarizer clusters the WHOLE recording at once, so a person who talks again after others have spoken
gets the same label as before (it is not a streaming tracker that can lose someone).

Backends (pick with --diarizer / the UI):
  diarize   Silero VAD + WeSpeaker ResNet34-LM embeddings + spectral clustering. About 8x faster than real
            time on CPU (maintainer-reported), Apache-2.0, `pip install diarize`, no account needed.
  pyannote  pyannote/speaker-diarization-community-1. Best-tested accuracy (AMI meetings ~17% DER in pyannote's
            own table), but about 0.5x real time on CPU (1 h of audio ~ 30 min) and gated: needs a free
            Hugging Face token. CC-BY-4.0.
Only one speaker is reported at a time; overlapping speech is not modelled by either backend here.
"""
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000
MIN_PIECE_S = 1.0     # a speaker change shorter than this inside one segment is absorbed (backchannels, noise)
MAX_GAP_S = 1.5       # a segment with no speaker turn on it takes the nearest turn if it is this close

DIARIZERS = {
    "diarize":  "diarize (Silero VAD + WeSpeaker ResNet34-LM + spectral clustering): fast, ~8x realtime on CPU, no token",
    "pyannote": "pyannote community-1 (pyannote/speaker-diarization-community-1): most accurate, ~0.5x realtime on CPU, needs a free Hugging Face token",
}
DEFAULT_DIARIZER = "diarize"
PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"


def diarizer_label(key: str) -> str:
    return f"{key}: {DIARIZERS[key]}"


# ---------------------------------------------------------------- backends
def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _write_wav(audio: np.ndarray, path: Path):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _backend_diarize(audio, kw):
    try:
        from diarize import diarize
    except ImportError as e:
        raise RuntimeError(
            "Speaker identification needs the 'diarize' package:\n"
            "  pip install diarize\n"
            "(it installs PyTorch; to keep it small run first:  "
            "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu)") from e
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "audio.wav"          # mono 16 kHz wav: works for any input format (mp4, m4a, ...)
        _write_wav(audio, path)
        result = diarize(str(path), **kw)
    return [(float(_get(s, "start")), float(_get(s, "end")), str(_get(s, "speaker"))) for s in result.segments]


_PYANNOTE = None


def _load_pyannote():
    global _PYANNOTE
    if _PYANNOTE is None:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError("The pyannote backend needs:  pip install pyannote.audio") from e
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
        try:
            try:
                pipe = Pipeline.from_pretrained(PYANNOTE_REPO, token=token)
            except TypeError:                  # older pyannote.audio releases
                pipe = Pipeline.from_pretrained(PYANNOTE_REPO, use_auth_token=token)
        except Exception as e:
            raise RuntimeError(
                f"Could not load {PYANNOTE_REPO} ({type(e).__name__}: {e}).\n"
                "This model is gated (free):\n"
                f"  1) open https://huggingface.co/{PYANNOTE_REPO} and accept the conditions\n"
                "  2) create a READ token at https://huggingface.co/settings/tokens, then run  hf auth login\n"
                "     (or set the HF_TOKEN environment variable)\n"
                "  3) run this again (the download happens once)") from e
        if pipe is None:
            raise RuntimeError(f"{PYANNOTE_REPO} could not be loaded: accept its conditions and log in "
                               "with a Hugging Face token (see above).")
        torch.set_num_threads(os.cpu_count() or 4)
        _PYANNOTE = pipe
    return _PYANNOTE


def _backend_pyannote(audio, kw):
    import torch
    pipe = _load_pyannote()
    # in-memory waveform: no ffmpeg/torchcodec needed, and any input format works
    out = pipe({"waveform": torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0),
                "sample_rate": SR}, **kw)
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    return [(float(t.start), float(t.end), str(lab)) for t, _, lab in ann.itertracks(yield_label=True)]


def check_backend(backend: str):
    """Fail early (before a long transcription) with install / login instructions."""
    if backend == "diarize":
        try:
            import diarize  # noqa: F401
        except ImportError:
            _backend_diarize(np.zeros(1, dtype=np.float32), {})      # raises the install message
    elif backend == "pyannote":
        _load_pyannote()
    else:
        raise ValueError(f"unknown diarizer '{backend}' (choose from {', '.join(DIARIZERS)})")


def run_diarization(audio, backend=DEFAULT_DIARIZER, num_speakers=None, min_speakers=None, max_speakers=None):
    """audio: 16 kHz mono float32. -> sorted list of (start, end, raw_label). Giving the real number of
    speakers (num_speakers) is the single biggest accuracy lever when you know it."""
    kw = {k: int(v) for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                     max_speakers=max_speakers).items() if v}
    check_backend(backend)
    turns = _backend_diarize(audio, kw) if backend == "diarize" else _backend_pyannote(audio, kw)
    return sorted(turns, key=lambda t: (t[0], t[1]))


# ---------------------------------------------------------------- merging with the transcript
def _pieces(s, e, turns, max_gap, min_piece):
    """-> [[label, start, end], ...] covering [s, e] without gaps; [] if no speaker is on or near the segment."""
    ov = []
    for ts, te, lab in turns:
        if te <= s:
            continue
        if ts >= e:
            break
        ov.append([lab, max(s, ts), min(e, te)])
    if not ov:
        best = None
        for ts, te, lab in turns:
            gap = ts - e if ts >= e else (s - te if te <= s else 0.0)
            if gap <= max_gap and (best is None or gap < best[0]):
                best = (gap, lab)
        return [[best[1], s, e]] if best else []

    ov.sort(key=lambda p: p[1])
    cov = []
    for lab, a, b in ov:
        if cov and a < cov[-1][2]:
            a = cov[-1][2]
        if b - a <= 1e-6:
            continue
        if cov and cov[-1][0] == lab:
            cov[-1][2] = b
        else:
            cov.append([lab, a, b])
    if not cov:
        return []

    while len(cov) > 1:                        # absorb speaker changes that are too short to be real
        short = [k for k, p in enumerate(cov) if p[2] - p[1] < min_piece]
        if not short:
            break
        k = min(short, key=lambda i: cov[i][2] - cov[i][1])
        j = max((x for x in (k - 1, k + 1) if 0 <= x < len(cov)), key=lambda x: cov[x][2] - cov[x][1])
        cov[j][1], cov[j][2] = min(cov[j][1], cov[k][1]), max(cov[j][2], cov[k][2])
        del cov[k]
        merged = []
        for p in cov:
            if merged and merged[-1][0] == p[0]:
                merged[-1][2] = max(merged[-1][2], p[2])
            else:
                merged.append(p)
        cov = merged

    cov[0][1], cov[-1][2] = s, e
    for a, b in zip(cov, cov[1:]):
        a[2] = b[1] = (a[2] + b[1]) / 2        # boundary halfway across any silence between the two turns
    return cov


def _cut_words(text, fracs):
    """Split text into len(fracs) word groups whose lengths follow fracs. None if there are too few words."""
    words = text.split()
    if len(words) < len(fracs):
        return None
    cum = np.cumsum([len(w) + 1 for w in words])
    total, out, idx, acc = float(cum[-1]), [], 0, 0.0
    for p, f in enumerate(fracs):
        acc += f
        remaining = len(fracs) - p - 1
        if remaining == 0:
            j = len(words)
        else:
            target = acc * total
            j = min(range(idx + 1, len(words) - remaining + 1), key=lambda x: abs(cum[x - 1] - target))
        out.append(" ".join(words[idx:j]))
        idx = j
    return out


def assign_speakers(segments, turns, split=True, min_piece=MIN_PIECE_S, max_gap=MAX_GAP_S):
    """Return NEW segment dicts with a 'speaker' ("Person 1", ...). A segment that spans a change of speaker is
    split there (text divided in proportion to the time, snapped to word boundaries) unless split=False, in
    which case it goes to whoever talks most during it. Numbering follows first appearance in the transcript."""
    turns = sorted(turns, key=lambda t: (t[0], t[1]))
    out = []
    for seg in segments:
        if seg.get("start") is None or seg.get("end") is None:
            out.append({**seg, "speaker": None})
            continue
        pieces = _pieces(seg["start"], seg["end"], turns, max_gap, min_piece)
        if not pieces:
            out.append({**seg, "speaker": None})
        elif len(pieces) == 1:
            out.append({**seg, "speaker": pieces[0][0]})
        else:
            done = False
            if split:
                total = sum(p[2] - p[1] for p in pieces)
                fr = [(p[2] - p[1]) / total for p in pieces]
                roman = _cut_words(seg["roman"], fr)
                if roman:
                    native = _cut_words(seg["native"], fr) if "native" in seg else None
                    if native is None and "native" in seg:
                        native = roman                 # raw text too short to divide: show the Roman piece instead
                    for k, (p, r) in enumerate(zip(pieces, roman)):
                        new = {**seg, "start": p[1], "end": p[2], "roman": r, "speaker": p[0]}
                        if native is not None:
                            new["native"] = native[k]
                        out.append(new)
                    done = True
            if not done:
                out.append({**seg, "speaker": max(pieces, key=lambda p: p[2] - p[1])[0]})

    names = {}
    for s in out:
        raw = s["speaker"]
        if raw is None:
            s["speaker"] = "Unknown"
        else:
            s["speaker"] = names.setdefault(raw, f"Person {len(names) + 1}")
    return out


# ---------------------------------------------------------------- helpers for output
def prefixed(speaker, text):
    return f"{speaker}: {text}" if speaker else text


def speaker_summary(segments):
    """-> [(name, seconds, share)] in Person order."""
    secs = {}
    for s in segments:
        if s.get("speaker") and s.get("start") is not None:
            secs[s["speaker"]] = secs.get(s["speaker"], 0.0) + max(0.0, s["end"] - s["start"])
    total = sum(secs.values()) or 1.0
    def order(kv):
        digits = "".join(c for c in kv[0] if c.isdigit())
        return (kv[0] == "Unknown", int(digits) if digits else 0)
    return [(n, t, t / total) for n, t in sorted(secs.items(), key=order)]


def speaker_summary_text(segments):
    rows = speaker_summary(segments)
    return ", ".join(f"{n} {share:.0%} ({secs / 60:.1f} min)" for n, secs, share in rows) or "no speakers found"
