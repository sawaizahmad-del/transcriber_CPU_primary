"""Public word lists as spelling hints for the polish pass (optional, offline, no model).

Any plain text list works: one word per line ("word" or "word<TAB or space>count"), or a TSV with a native-script
column, a Latin column and a count column, such as the Roman Hindi/Urdu romanization lexicons of Google's open
Dakshina dataset (hi/lexicons/*.tsv, ur/lexicons/*.tsv). Non-Latin columns are ignored. Add an English word list
too, otherwise ordinary English words look "unknown" and get wrong hints.

How it is used: for each chunk sent to the LLM, words that are NOT in the list (not even under another common
spelling, see translit.roman_key) are matched to the closest known word, and the pairs are shown to the LLM as
hints ("rha -> raha"). The LLM may ignore them; they are never applied on their own.
"""
import difflib
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from translit import roman_key

_WORD = re.compile(r"[a-z]{2,}")
_TOKEN = re.compile(r"[A-Za-z]+")


class Lexicon:
    def __init__(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [p for p in str(paths).split(",") if p.strip()]
        self.count = {}                      # lower-case word -> attestation count
        for p in paths:
            self._load(Path(str(p).strip()))
        if not self.count:
            raise ValueError("the word list(s) contain no usable Latin words")
        self.by_key = {}                     # roman_key -> most common spelling with that key
        for w, c in self.count.items():
            k = roman_key(w)
            if k not in self.by_key or c > self.count[self.by_key[k]]:
                self.by_key[k] = w
        self._keys_by_first = defaultdict(list)
        for k in self.by_key:
            self._keys_by_first[k[0]].append(k)
        self.suggest = lru_cache(maxsize=4096)(self._suggest)

    def _load(self, path: Path):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t") if "\t" in line else line.split()
            count = 1
            if len(cols) > 1 and cols[-1].strip().isdigit():
                count = int(cols.pop().strip())
            for c in cols:
                w = c.strip().lower()
                if _WORD.fullmatch(w):
                    self.count[w] = self.count.get(w, 0) + count

    def __len__(self):
        return len(self.count)

    def known(self, word: str) -> bool:
        w = word.lower()
        return w in self.count or roman_key(w) in self.by_key

    def _suggest(self, word: str):
        k = roman_key(word.lower())
        if len(k) < 3:
            return None
        cand = difflib.get_close_matches(k, self._keys_by_first.get(k[0], ()), n=1, cutoff=0.8)
        return self.by_key[cand[0]] if cand else None

    def hints(self, text: str, limit: int = 8) -> str:
        """'rha -> raha; dedlain -> deadline' for the unknown words of `text` that have a close known word."""
        seen, out = set(), []
        for tok in _TOKEN.findall(text):
            w = tok.lower()
            if len(w) < 3 or w in seen:
                continue
            seen.add(w)
            if self.known(w):
                continue
            s = self.suggest(w)
            if s and s != w:
                out.append(f"{tok} -> {s}")
                if len(out) >= limit:
                    break
        return "; ".join(out)
