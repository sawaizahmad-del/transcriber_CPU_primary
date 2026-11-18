#!/usr/bin/env python3
"""One-time converter: Hugging Face Whisper model -> CTranslate2 int8 (for faster-whisper on CPU).

    pip install ctranslate2 transformers torch
    python convert_model.py collabora/whisper-small-hindi
    python convert_model.py Oriserve/Whisper-Hindi2Hinglish-Apex

Output goes to models/<name>-ct2, which the 'hindi-small' / 'apex' presets read.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Hugging Face model id, e.g. collabora/whisper-small-hindi")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"models/{args.model.split('/')[-1]}-ct2"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    base = ["ct2-transformers-converter", "--model", args.model, "--output_dir", out,
            "--quantization", "int8", "--force"]
    r = subprocess.run(base + ["--copy_files", "tokenizer.json", "preprocessor_config.json"])
    if r.returncode != 0:  # some repos lack tokenizer.json; faster-whisper falls back to the stock tokenizer
        print("[retrying without --copy_files]")
        r = subprocess.run(base)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
