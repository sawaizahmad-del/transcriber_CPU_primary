#!/usr/bin/env python3
"""Compare local LLMs on a small Roman Urdu polish test, on YOUR machine.

    python polish_eval.py --models gemma4:e4b,qwen3.5:4b
    python polish_eval.py --models gemma4:e2b,gemma4:e4b --level conservative
    python polish_eval.py --models gemma4:e4b --level maximum     # expect it to touch 'clean' lines more (punctuation/caps)
    python polish_eval.py --models gemma4:e4b --lexicon ur.translit.sampled.train.tsv,hi.translit.sampled.train.tsv,english_words.txt
        ^ A/B test of the public word list: every model is run without and with the spelling hints, and the
          script tells you whether turning the hints on actually helped on these cases.

Case kinds:
  clean   already correct -> must stay UNCHANGED and unflagged (tests over-correction)
  fix     has errors      -> should become the target (scored with spelling-tolerant WER)
  garbled unrecoverable   -> must be FLAGGED and left UNCHANGED (tests honesty)
Add your own cases (from real transcripts) to CASES to make the test match your recordings.
"""
import argparse
import time

from benchmark import score
from polish import LEVELS, Polisher

CASES = [
    ("clean",   "kal ki meeting ka agenda bhej diya hai, please check kar lein.", None),
    ("clean",   "main ne client ko email kar di thi lekin abhi tak jawab nahi aaya.", None),
    ("clean",   "yeh feature agle hafte tak deploy ho jayega.", None),
    ("clean",   "budget ke baare mein hum Monday ko baat karenge.", None),
    ("fix",     "mein kal office nahi aa rha hon", "main kal office nahi aa raha hoon"),
    ("fix",     "yeh project ka dedlain agle haftay hai", "yeh project ki deadline agle hafte hai"),
    ("fix",     "hum ne client se mitting rakhi hai kal subha", "hum ne client se meeting rakhi hai kal subah"),
    ("fix",     "aap ko rapoort kab tak bhejna hai", "aap ko report kab tak bhejni hai"),
    ("fix",     "mujay yeh invoys abhi tak nahi mila", "mujhe yeh invoice abhi tak nahi mila"),
    ("fix",     "hamain apna targit achiev karna hai", "hamen apna target achieve karna hai"),
    ("fix",     "aap ka mashwara mujay bohat pasand aya", "aap ka mashwara mujhe bohat pasand aaya"),
    ("fix",     "wo kal subha jaldi aya tha", "woh kal subah jaldi aaya tha"),
    ("fix",     "hum ne manager ko apni rapoort di", "hum ne manager ko apni report di"),
    ("garbled", "aaz ka sthaanpaar zindagi mein jaan hai kab", None),
    ("garbled", "beeeenaawakzaae ki muwaalphun ki sakreeng par waaeelphun", None),
    ("garbled", "yeh pikamma traaskraaibar banna hai aapne yeh", None),
]


def run_one(model, level, context_file, lexicon):
    """-> dict of scores for one model/setting, or None if the model is not available."""
    pol = Polisher(model=model, level=level, window=None, context_file=context_file, summary=False, verbose=False,
                   mode="all", cache_path=None, lexicon=lexicon, merge=False)   # merge=False: each case is its own sentence
    try:
        pol.check()
    except RuntimeError as e:
        return str(e).splitlines()[0]
    t0 = time.time()
    res = pol.polish([dict(start=None, end=None, roman=c[1]) for c in CASES])
    per_line = (time.time() - t0) / len(CASES)
    kept = [r["status"] == "ok" for r, c in zip(res, CASES) if c[0] == "clean"]
    after, before = [], []
    for r, c in zip(res, CASES):
        if c[0] == "fix":
            after.append(score(c[2], r["text"])[0])
            before.append(score(c[2], c[1])[0])
    good = [r["status"] == "flagged" and r["text"] == r["original"] for r, c in zip(res, CASES) if c[0] == "garbled"]
    return dict(kept=sum(kept), n_kept=len(kept), wer=sum(after) / len(after), before=sum(before) / len(before),
                good=sum(good), n_good=len(good), unchecked=pol.stats["unchecked"], spl=per_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma4:e4b")
    ap.add_argument("--level", default="standard", choices=list(LEVELS))
    ap.add_argument("--context-file", default="context_tmc.txt")
    ap.add_argument("--lexicon", default=None,
                    help="public word list(s), comma separated. Each model is then run twice (without / with hints)")
    args = ap.parse_args()

    lexicon = None
    if args.lexicon:
        from lexicon import Lexicon
        lexicon = Lexicon(args.lexicon)
        print(f"[word list: {len(lexicon)} words]")
    variants = [("", None)] + ([(" +lexicon", lexicon)] if lexicon else [])

    print(f"{'model':26} {'clean kept':>10} {'fix WER':>9} {'(before)':>9} {'garbled flagged+kept':>21} "
          f"{'not reviewed':>13} {'sec/line':>9}")
    scores = {}
    for model in [m.strip() for m in args.models.split(",")]:
        for suffix, lex in variants:
            r = run_one(model, args.level, args.context_file, lex)
            label = model + suffix
            if isinstance(r, str):
                print(f"{label:26} skipped: {r}")
                break
            scores[label] = r
            print(f"{label:26} {r['kept']}/{r['n_kept']:>8} {r['wer']:9.0%} {r['before']:9.0%} "
                  f"{r['good']}/{r['n_good']:>19} {r['unchecked']:>13} {r['spl']:9.1f}")
    if lexicon:
        print("\nDoes the public word list help? (on these cases only; add your own real lines to CASES first)")
        for model in [m.strip() for m in args.models.split(",")]:
            a, b = scores.get(model), scores.get(model + " +lexicon")
            if not a or not b:
                continue
            better = b["wer"] < a["wer"] - 0.01
            safe = b["kept"] >= a["kept"] and b["good"] >= a["good"]
            verdict = ("YES: lower fix WER and no new over-corrections -> use --lexicon" if better and safe else
                       "NO: it makes no clear difference -> leave it off" if not better and safe else
                       "NO: it breaks clean or garbled lines -> leave it off")
            print(f"  {model}: fix WER {a['wer']:.0%} -> {b['wer']:.0%}, clean kept {a['kept']} -> {b['kept']}, "
                  f"garbled kept {a['good']} -> {b['good']}  =>  {verdict}")
    print("\nBest model = highest 'clean kept', lowest 'fix WER', highest 'garbled flagged+kept', 0 'not reviewed', "
          "at acceptable sec/line.")


if __name__ == "__main__":
    main()
