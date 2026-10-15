"""JFLEG GLEU (Napoles et al. 2015/2016 "updated" multi-reference GLEU), a line-for-line port of the official
scorer keisks/jfleg eval/gleu.py (repo master ee06ff806a208aba815ac45313f4e750a48330a5; last change to gleu.py
d1a4a52e2d07c0c8f2786775a34e6a68dee5ad50, 2018-06-06; file sha256 b8bf3605b3c23a89...). Same n-gram statistics,
same 500 iterations with one random reference per sentence (random.seed(j * 101), random.randint), same corpus
score = mean over iterations, reported x100. Only the I/O differs: lists of whitespace-tokenised strings instead of
files, and no scipy (the normal 95% interval is mean +- 1.959964 * std, which is what scipy.stats.norm.interval gives).

Reference sampling. The official script draws the reference index with random.randint after random.seed(j * 101).
Python 2 and Python 3 implement randint differently (py2: a + int(random() * n); py3: _randbelow via getrandbits), so
the same script gives slightly different corpus scores under the two interpreters. ``rng="py2"`` (default here)
reproduces the published leaderboard exactly: SOURCE (copy the input) = 40.543 on test and 38.215 on dev (published
40.54 / 38.21); ``rng="py3"`` = running the official file under Python 3 gives 40.474 / 38.197.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import List, Sequence


def ngram_counts(sent: Sequence[str], n: int) -> Counter:
    return Counter([tuple(sent[i:i + n]) for i in range(len(sent) + 1 - n)])


def ngram_diff(a: Counter, b: Counter) -> Counter:
    """n-grams in a but not in b"""
    d = Counter(a)
    for k in set(a) & set(b):
        del d[k]
    return d


class GLEU:
    def __init__(self, sources: List[str], refs: List[List[str]], order: int = 4):
        """sources[i]: source line; refs[k][i]: reference k of line i (whitespace-tokenised strings)."""
        self.order = order
        self.all_s_ngrams = [[ngram_counts(line.split(), n) for n in range(1, order + 1)] for line in sources]
        self.num_refs = len(refs)
        self.refs = [[] for _ in sources]
        self.rlens = [[] for _ in sources]
        for rfile in refs:
            assert len(rfile) == len(sources)
            for i, line in enumerate(rfile):
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

    def load_hypothesis_sentence(self, hyp: List[str]):
        self.hlen = len(hyp)
        self.this_h_ngrams = [ngram_counts(hyp, n) for n in range(1, self.order + 1)]

    def gleu_stats(self, i: int, r_ind: int) -> List[int]:
        hlen, rlen = self.hlen, self.rlens[i][r_ind]
        out = [hlen, rlen]
        for n in range(1, self.order + 1):
            h = self.this_h_ngrams[n - 1]
            s = self.all_s_ngrams[i][n - 1]
            r = ngram_counts(self.refs[i][r_ind], n)
            sd = ngram_diff(s, r)
            out.append(max([sum((h & r).values()) - sum((h & sd).values()), 0]))
            out.append(max([hlen + 1 - n, 0]))
        return out

    @staticmethod
    def gleu(stats, smooth=False) -> float:
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if any(x == 0 for x in stats):
            return 0
        c, r = stats[:2]
        log_prec = sum(math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])) / 4
        return math.exp(min([0, 1 - float(r) / c]) + log_prec)

    def corpus(self, hyps: List[str], num_iterations: int = 500, rng: str = "py2"):
        """Corpus GLEU as the official run_iterations(per_sent=False): returns dict(mean, std, ci) on the 0-1 scale,
        plus the per-iteration scores."""
        hyp = [h.split() for h in hyps]
        assert len(hyp) == len(self.refs)
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            if rng == "py2":  # Python 2 random.randint(0, k-1) == int(random() * k)
                indices.append([int(random.random() * self.num_refs) for _ in range(len(hyp))])
            else:
                indices.append([random.randint(0, self.num_refs - 1) for _ in range(len(hyp))])
        iter_stats = [[0] * (2 * self.order + 2) for _ in range(num_iterations)]
        for i, h in enumerate(hyp):
            self.load_hypothesis_sentence(h)
            by_ref = [None] * self.num_refs
            for j in range(num_iterations):
                ref = indices[j][i]
                st = by_ref[ref]
                if st is None:
                    st = by_ref[ref] = self.gleu_stats(i, ref)
                iter_stats[j] = [a + b for a, b in zip(iter_stats[j], st)]
        scores = [self.gleu(st) for st in iter_stats]
        m = sum(scores) / len(scores)
        sd = math.sqrt(sum((x - m) ** 2 for x in scores) / len(scores))  # np.std (population)
        return dict(mean=m, std=sd, ci=(m - 1.959963984540054 * sd, m + 1.959963984540054 * sd), scores=scores)

    def sentence(self, hyps: List[str]) -> List[float]:
        """Sentence-level GLEU (official --sent): mean over references of the smoothed GLEU."""
        out = []
        for i, h in enumerate(hyps):
            self.load_hypothesis_sentence(h.split())
            out.append(sum(self.gleu(self.gleu_stats(i, r), smooth=True) for r in range(self.num_refs)) / self.num_refs)
        return out
