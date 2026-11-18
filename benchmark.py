#!/usr/bin/env python3
"""Measure accuracy + speed of presets/settings on YOUR audio, so choices come from numbers.

    python benchmark.py clip.mp3 reference.txt --presets swift,prime,fast
    python benchmark.py clip.mp3 reference.txt --presets swift --try-clean --hotwords "online income"

reference.txt = what is actually said, typed in Roman (or Devanagari). A 1-3 minute clip is plenty.
Roman Urdu has no standard spelling (zindagi / zindagee / zindagy), so both texts are reduced to a
phonetic key before scoring: 'kaise' == 'kese', 'theek' == 'thik', 'acha' == 'achha'.
"""
import argparse
import re
import time
from pathlib import Path

from translit import roman_key, to_roman


def tokens(text: str):
    text = to_roman(text).lower()
    return [roman_key(w) for w in re.findall(r"[a-z0-9]+", text)]


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(ref: str, hyp: str):
    r, h = tokens(ref), tokens(hyp)
    wer = edit_distance(r, h) / max(1, len(r))
    rc, hc = list("".join(r)), list("".join(h))
    cer = edit_distance(rc, hc) / max(1, len(rc))
    return wer, cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("reference")
    ap.add_argument("--presets", default="swift,fast")
    ap.add_argument("--try-clean", action="store_true", help="also test with light audio cleanup")
    ap.add_argument("--hotwords", default=None)
    ap.add_argument("--out-dir", default="bench_out")
    args = ap.parse_args()

    from transcribe import Transcriber
    ref = Path(args.reference).read_text(encoding="utf-8")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    variants = [(False, None)]
    if args.try_clean:
        variants.append((True, None))
    if args.hotwords:
        variants.append((False, args.hotwords))

    rows = []
    for preset in [p.strip() for p in args.presets.split(",")]:
        for clean, hot in variants:
            label = preset + (" +clean" if clean else "") + (" +hotwords" if hot else "")
            try:
                tr = Transcriber(preset, clean=clean, hotwords=hot)
                t0 = time.time()
                segs, dur = tr.transcribe(args.audio)
                took = time.time() - t0
            except Exception as e:
                rows.append((label, None, None, None, f"failed: {e}"))
                continue
            hyp = "\n".join(s["roman"] for s in segs)
            (out / f"{label.replace(' ', '_')}.txt").write_text(hyp, encoding="utf-8")
            wer, cer = score(ref, hyp)
            st = tr.last_stats
            rows.append((label, wer, cer, dur / max(took, 1e-6),
                         f"flagged {st['flagged']}, rescued {st['rescued']}, dropped {st['dropped']}"))

    print(f"\n{'setting':28} {'WER':>7} {'CER':>7} {'xRT':>6}  notes")
    for label, wer, cer, xrt, note in sorted(rows, key=lambda r: (r[1] is None, r[1] or 0)):
        if wer is None:
            print(f"{label:28} {'-':>7} {'-':>7} {'-':>6}  {note}")
        else:
            print(f"{label:28} {wer:7.1%} {cer:7.1%} {xrt:6.1f}  {note}")
    print(f"\nTranscripts saved in {out}/. Lower WER/CER is better; xRT = times faster than real time.")


if __name__ == "__main__":
    main()
