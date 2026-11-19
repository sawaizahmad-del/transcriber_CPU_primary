# Roman Hindi/Urdu Transcriber (CPU-only, free)

Speech -> Roman Hindi/Urdu text, e.g. `aap kaise hain? main theek hoon.`

## Setup
    pip install -r requirements.txt
Models download once from Hugging Face (free), then everything runs offline. No ffmpeg needed.

## Use
    python transcribe.py "talk.mp3"                          # default preset: swift
    python transcribe.py talk.mp3 --format txt,srt,json --out-dir out
    python transcribe.py talk.mp3 --hotwords "online income email password"
    python benchmark.py clip.mp3 reference.txt --presets swift,prime,fast   # measure on YOUR audio
    python app.py                                            # optional web UI

The run prints speed (x realtime) and guard stats: `flagged` = segments that looked like loops,
`rescued` = fixed by a second decode, `dropped` = removed because they could not be fixed.

## Presets (all int8, CPU)
| preset      | model                         | output             | notes |
|-------------|-------------------------------|--------------------|-------|
| swift       | Hindi2Hinglish-Swift (base)   | Roman directly     | default; fast; English words stay English |
| prime       | Hindi2Hinglish-Prime (large-v3)| Roman directly    | most accurate Hinglish; slow on CPU; community CT2 upload |
| apex        | Hindi2Hinglish-Apex (~800M)   | Roman directly     | newer; needs `python convert_model.py Oriserve/Whisper-Hindi2Hinglish-Apex` |
| hindi-small | collabora/whisper-small-hindi | Devanagari->Roman  | clean Hindi/Urdu speech; needs `python convert_model.py collabora/whisper-small-hindi` |
| fast/balanced/best | generic Whisper small/turbo/large-v3 | Devanagari->Roman | NOT fine-tuned for Hindi; weak/looping, kept for comparison |

## What improves accuracy here (and why)
1. Fine-tuned Hindi/Hinglish models instead of generic Whisper (the single biggest factor).
2. Anti-loop decoding: repetition penalty, no-repeat 3-grams, temperature fallback, compression-ratio
   retry, no cross-segment context, VAD.
3. Hallucination guard: segments that look like loops are re-decoded with stricter settings, else dropped.
4. No audio denoising by default (studies show it usually hurts Whisper). `--clean` is opt-in; test it
   with `benchmark.py --try-clean`.
5. Optional `--hotwords` for domain terms (helps rare words; can hurt common ones, so off by default).
6. Benchmark script scores with spelling-tolerant matching (zindagi == zindagee).

## Spelling style
Roman Urdu has no standard spelling. Devanagari presets use `translit.py` (edit `_LEXICON_RAW` to taste).


## Local LLM polish pass (optional, free, CPU)
Fixes grammar/spelling in the Roman Urdu text **without rewriting it**, and flags sentences it can't repair.
