"""Local web UI: python app.py  ->  http://127.0.0.1:7860 (CPU, free, offline after first model download)."""
import os
import time

import gradio as gr

from audio_utils import load_audio
from polish import DEFAULT_MODEL, LEVELS, REVIEW_JS, Polisher, group_paragraphs, render_html
from speakers import (DEFAULT_DIARIZER, DIARIZERS, assign_speakers, diarizer_label, prefixed, run_diarization,
                      speaker_summary_text)
from transcribe import DEFAULT_PRESET, PRESETS, Transcriber, preset_label

# (menu text, Ollama tag). The box also accepts any other tag you have pulled.
LLM_CHOICES = [
    ("gemma4:e2b: Google Gemma 4 E2B (smallest, fastest, weakest)", "gemma4:e2b"),
    ("gemma4:e4b: Google Gemma 4 E4B (default)", "gemma4:e4b"),
    ("qwen3:8b: Alibaba Qwen3 8B (good smaller model)", "qwen3:8b"),
    ("qwen3:14b: Alibaba Qwen3 14B (stronger, may drift into Hindi script)", "qwen3:14b"),
    ("llama3.3:70b: Meta Llama 3.3 70B (best quality, needs ~48 GB RAM, very slow on CPU)", "llama3.3:70b"),
]

_model_cache: dict[str, Transcriber] = {}
_transcript_cache: dict[tuple, tuple] = {}   # (file, size, mtime, preset) -> (segments, duration, seconds)
_turn_cache: dict[tuple, list] = {}          # (file, size, mtime, diarizer, num_speakers) -> speaker turns


def _key(path, preset):
    st = os.stat(path)
    return (path, st.st_size, int(st.st_mtime), preset)


def run(audio_path, preset, identify, diarizer, num_speakers, polish, llm_model, level, mode, lexicon_paths,
        progress=gr.Progress()):
    """Generator: the raw transcript appears as soon as it is ready; speakers and the polished review follow."""
    if not audio_path:
        yield "", "", "", "Upload or record some audio first."
        return

    key = _key(audio_path, preset)
    if key in _transcript_cache:                      # toggling polish/speakers does not re-transcribe
        segments, duration, took = _transcript_cache[key]
        note = f"Transcript reused from this session ({duration:.0f}s of audio)."
    else:
        progress(0.02, desc="Loading speech model...")
        if preset not in _model_cache:
            _model_cache.clear()                      # keep only one model in RAM
            try:
                _model_cache[preset] = Transcriber(preset)
            except (FileNotFoundError, RuntimeError) as e:     # missing conversion / gated download
                yield "", "", "", f"<pre>{e}</pre>"
                return
        progress(0.1, desc="Transcribing...")
        t0 = time.time()
        segments, duration = _model_cache[preset].transcribe(audio_path)
        took = time.time() - t0
        _transcript_cache[key] = (segments, duration, took)
        st = _model_cache[preset].last_stats
        note = (f"Speech model: {preset_label(preset)}. "
                f"Transcribed {duration:.0f}s of audio in {took:.0f}s ({duration / max(took, 1e-6):.1f}x realtime). "
                f"Cut-off segments fixed: {st['truncated']}, loops handled: {st['flagged']}.")

    def plain(segs):
        return "\n".join(prefixed(s.get("speaker"), s["roman"]) for s in segs)

    native = "\n".join(s["native"] for s in segments)
    yield plain(segments), "", native, note + (" Identifying speakers..." if identify else
                                                " Polishing with the local LLM..." if polish else "")

    if identify:
        n_spk = int(num_speakers or 0) or None
        try:
            dkey = (key[0], key[1], key[2], diarizer, n_spk)
            if dkey not in _turn_cache:
                progress(0.3, desc=f"Identifying speakers ({diarizer})...")
                t2 = time.time()
                turns = run_diarization(load_audio(audio_path), diarizer, n_spk)
                _turn_cache.clear()
                _turn_cache[dkey] = turns
                spk_note = f" ({time.time() - t2:.0f}s)"
            else:
                spk_note = " (reused)"
            segments = assign_speakers(segments, _turn_cache[dkey])     # new list: the cached transcript stays untouched
            note += f" Speakers{spk_note}: {speaker_summary_text(segments)}."
        except (RuntimeError, ImportError) as e:
            note += f" Speaker identification failed ({e}); continuing without it."
        yield plain(segments), "", native, note + (" Polishing with the local LLM..." if polish else "")

    if not polish:
        return

    try:
        pol = Polisher(model=llm_model or DEFAULT_MODEL, level=level, mode=mode,
                       context_file="context_tmc.txt", verbose=False,
                       lexicon=(lexicon_paths or "").strip() or None)
        pol.check()
    except (RuntimeError, OSError, ValueError) as e:
        yield plain(segments), f"<pre>{e}</pre>", native, note
        return

    def on_progress(done, total, eta, phase):
        if phase == "summary":
            progress(0.5, desc="LLM: reading the first minutes for context...")
        else:
            progress(0.5 + 0.5 * done / max(total, 1), desc=f"LLM polishing chunk {done}/{total} (about {eta:.0f}s left)")

    t1 = time.time()
    results = pol.polish(segments, on_progress=on_progress)
    st = pol.stats
    missed = f", {st['unchecked']} NOT reviewed (shown in yellow)" if st["unchecked"] else ""
    if st.get("skipped"):
        missed += f", {st['skipped']} not sent to the LLM because 'Lines to polish' is set to 'uncertain' (use 'all')"
    yield ("\n".join(prefixed(r["speaker"], r["text"]) for r in group_paragraphs(results)), render_html(results), native,
           f"{note} Polished ({level}) in {time.time() - t1:.0f}s: {st['fixed']} fixed, {st['flagged']} flagged, "
           f"{st['sent']} of {st['lines']} sentences reviewed by the LLM{missed}. "
           "Yellow sentences: open them to compare and Accept / Keep original.")


# The Accept / Keep-original buttons need a little JavaScript. Gradio ignores <script> tags inside gr.HTML, so it is
# added to the page <head> (Gradio 4/5: Blocks(head=...); newer versions take it in launch(head=...)).
HEAD = f"<script>{REVIEW_JS}</script>"
try:
    _blocks = gr.Blocks(title="Roman Hindi/Urdu Transcriber", head=HEAD)
    _head_in_launch = False
except TypeError:
    _blocks = gr.Blocks(title="Roman Hindi/Urdu Transcriber")
    _head_in_launch = True

with _blocks as demo:
    gr.Markdown("## Roman Hindi/Urdu Transcriber  \nCPU-only · free · runs locally")
    with gr.Row():
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        with gr.Column():
            preset = gr.Dropdown([(preset_label(k), k) for k in PRESETS], value=DEFAULT_PRESET,
                                 label="Speech model (swift recommended)")
            identify = gr.Checkbox(label="Identify speakers (Person 1, Person 2, ...)", value=False)
            diarizer = gr.Dropdown([(diarizer_label(k), k) for k in DIARIZERS], value=DEFAULT_DIARIZER,
                                   label="Speaker model")
            num_speakers = gr.Number(value=0, precision=0, minimum=0,
                                     label="Number of speakers (0 = detect automatically; exact number is more accurate)")
            polish = gr.Checkbox(label="Polish with local LLM (needs Ollama running)", value=False)
            llm_model = gr.Dropdown(LLM_CHOICES, value=DEFAULT_MODEL, allow_custom_value=True,
                                    label="Ollama model (must be pulled first: ollama pull <tag>)")
            level = gr.Dropdown(list(LEVELS), value="standard", label="Polish level",
                                info="conservative = spelling/grammar only · standard = also repairs mis-heard words · "
                                     "maximum = every detail, two review passes (slowest, always checks every line)")
            mode = gr.Dropdown(["all", "uncertain"], value="all",
                               label="Lines to polish (all = every line; uncertain = faster but skips lines the speech "
                                     "model was confident about; ignored for 'maximum')")
    lexicon = gr.Textbox(value="", label="Public word list(s) for spelling hints (optional, comma-separated file paths)",
                         info="e.g. dakshina/ur/lexicons/ur.translit.sampled.train.tsv,english_words.txt. "
                              "Only helps if polish_eval.py shows it does on your data.")
    btn = gr.Button("Transcribe", variant="primary")
    status = gr.Markdown()
    roman = gr.Textbox(label="Roman text (with speakers / polished if enabled)", lines=8)
    review = gr.HTML(label="Review: green = changed, yellow = unclear (open it, then Accept or Keep original)")
    native = gr.Textbox(label="Raw model output", lines=4)
    btn.click(run, [audio, preset, identify, diarizer, num_speakers, polish, llm_model, level, mode, lexicon],
              [roman, review, native, status])

if __name__ == "__main__":
    demo.launch(head=HEAD) if _head_in_launch else demo.launch()
