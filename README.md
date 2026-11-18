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

One-time setup:
    1) Install Ollama (free) from https://ollama.com and leave it running
    2) ollama pull gemma4:e4b

Use:
    python transcribe.py talk.mp3 --polish                       # transcribe + polish
    python polish.py raw_transcript.txt                          # polish an existing transcript (one line per segment)
    python transcribe.py talk.mp3 --polish --polish-level conservative   # spelling/grammar only
    python transcribe.py talk.mp3 --polish --polish-mode uncertain       # faster: only low-confidence stretches
Outputs: `<name>.polished.txt`, `<name>.review.html` (open in a browser), `<name>.polished.json` (original + final + flags).

How it behaves:
- **Whole paragraphs, never the whole meeting.** The LLM gets one or two complete paragraphs (about 220 words, up to 12
  sentences; `--window` changes it) per request, with no other lines around them, so it keeps each paragraph's context.
  Inside a paragraph every sentence is numbered and needs its own verdict, so nothing can be silently skipped:
  a sentence with no verdict is re-asked, then marked "not reviewed" in yellow. A chunk in which EVERY sentence came back OK
  is asked once more (second-review wording), because that is almost never true for speech from a weak microphone.
  A sentence never sent to the LLM (polish mode 'uncertain') is marked "not sent to the LLM".
- **Context from the opening only.** The LLM's topic summary and "likely terms" are written from the first 5 minutes of the
  recording (`--context-minutes`), plus whatever you put in `context_tmc.txt`.
- **Green** words = changed (hover to see the original). **Yellow** sentences = unclear or the change was too big to apply
  automatically: click one to see the original and the AI suggestion side by side with the changed words highlighted
  (red in the original, green in the suggestion), then press **Accept AI suggestion** or **Keep original**.
  Accepting swaps in the suggestion and keeps the changed words green (hover = the word it replaced); keeping restores the
  original. Both can be undone, and there are "Accept all" / "Keep all originals" buttons.
  **Copy final text** / **Download .txt** (top of the review page) give the transcript that follows your choices;
  `.polished.txt` only contains the automatic fixes.
- The model only proposes per-sentence edits; each edit must pass guardrails (similar enough to the original, same
  numbers, Latin letters, similar length). A rejected rewrite is shown as a suggestion, not applied.
- Optional public word list: `--lexicon file1,file2` (see `lexicon.py`) shows the LLM the closest known spelling of words that
  are not in the list. It is off by default: run `python polish_eval.py --models gemma4:e4b --lexicon ...` and enable it only
  if the A/B result says it helps on your data.

Choosing the model: `python polish_eval.py --models gemma4:e4b,qwen3.5:4b` scores each on over-correction,
fix accuracy and honest flagging, on your own machine. Add cases from your real transcripts to `CASES`.
Other options: `gemma4:e2b` (faster, weaker), `gemma4:26b` (best quality; needs ~16 GB+ free RAM).
Note: Ollama's default downloads are 4-bit; research suggests heavily quantized small models can degrade in
low-resource languages, so if results look poor try a higher-quality tag from the model's page on ollama.com.


## Speed tips (CPU)
1. **One-time autotune:** `python transcribe.py talk.mp3 --autotune` times a few thread/batch combinations on the
   first 45 s of your file and remembers the fastest in `tuning.json` for this PC.
2. **Use `swift`.** Base-sized and writes Roman directly. Devanagari presets (`fast`, `balanced`, ...) cost ~3x more
   decoder tokens per sentence and need shorter chunks, so they are slower and less accurate for this job.
3. `--beam-size 1` is the fastest decoding (greedy); default is 3 (2 for `balanced`).
4. Polish: `--polish-mode uncertain` only sends sentences the speech model was unsure about; use
   `--llm-model gemma4:e2b` for a faster, smaller LLM. LLM replies are cached on disk (`.polish_cache.json`), so
   re-polishing the same transcript is instant. Transcribe once with `--format txt,json`, then
   `python polish.py talk.json` to try different polish settings without re-transcribing.
5. The web UI shows the raw transcript as soon as it is ready and reuses it if you then tick "Polish".

## Troubleshooting
- **Lots of `?` / replacement characters** came from Devanagari output being cut off mid-character (the model ran
  out of decoder tokens on a long chunk). Fixed: shorter chunks for Devanagari presets, automatic re-decoding of
  cut-off segments (counted as "cut-off" in the stats line), and stray Urdu-script letters are romanized.
- **Hard to read review page:** fixed (dark text forced on the highlight boxes in light and dark themes).
