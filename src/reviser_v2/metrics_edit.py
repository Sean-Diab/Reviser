"""Text-editing metrics for real-edit benchmarks (task 24): SARI (Xu et al., 2016), exact match, and
token-level edit precision/recall against the reference diff.

SARI follows the widely used tensor2tensor / HF ``evaluate`` "sari" implementation: lowercase, sacrebleu 13a
tokenization, n = 1..4, KEEP F1, DELETE precision, ADD F1, averaged over n and then over the three operations,
reported x100. ``tests/test_metrics_edit.py`` pins the HF documentation example (26.9536...).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence

from .trajectories import diff_hunks


def tok13a(line: str) -> str:
    """sacrebleu TokenizerV14International's simpler sibling: the mteval-v13a tokenizer."""
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    line = f" {line} "
    line = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", line)
    line = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", line)
    line = re.sub(r"([\.,])([^0-9])", r" \1 \2", line)
    line = re.sub(r"([0-9])(-)", r"\1 \2 ", line)
    return " ".join(line.split())


def normalize(s: str) -> str:
    return tok13a(s.lower())


def _ngrams(toks: List[str], n: int) -> List[str]:
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _sari_ngram(sgrams, cgrams, rgramslist, numref):
    rgramcounter = Counter(g for rg in rgramslist for g in rg)
    sgramcounter_rep = Counter({g: c * numref for g, c in Counter(sgrams).items()})
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter({g: c * numref for g, c in cgramcounter.items()})
    # KEEP
    keep_rep = sgramcounter_rep & cgramcounter_rep
    keepgood_rep = keep_rep & rgramcounter
    keepall_rep = sgramcounter_rep & rgramcounter
    k1 = sum(keepgood_rep[g] / keep_rep[g] for g in keep_rep)
    k2 = sum(keepgood_rep[g] for g in keep_rep)
    kp = k1 / len(keep_rep) if len(keep_rep) > 0 else 1
    kr = k2 / sum(keepall_rep.values()) if len(keepall_rep) > 0 else 1
    keep = 2 * kp * kr / (kp + kr) if (kp > 0 or kr > 0) else 0
    # DELETION (precision only, as in the reference implementation)
    del_rep = sgramcounter_rep - cgramcounter_rep
    delgood_rep = del_rep - rgramcounter
    d1 = sum(delgood_rep[g] / del_rep[g] for g in del_rep)
    dp = d1 / len(del_rep) if len(del_rep) > 0 else 1
    # ADDITION
    sset = set(sgrams)
    add = set(cgramcounter) - sset
    addgood = add & set(rgramcounter)
    addall = set(rgramcounter) - sset
    ap = len(addgood) / len(add) if len(add) > 0 else 1
    ar = len(addgood) / len(addall) if len(addall) > 0 else 1
    addf = 2 * ap * ar / (ap + ar) if (ap > 0 or ar > 0) else 0
    return keep, dp, addf


def sari_sent(source: str, pred: str, refs: Sequence[str]) -> Dict[str, float]:
    s, c = normalize(source).split(" "), normalize(pred).split(" ")
    rs = [normalize(r).split(" ") for r in refs]
    ks, ds, adds = [], [], []
    for n in range(1, 5):
        k, d, a = _sari_ngram(_ngrams(s, n), _ngrams(c, n), [_ngrams(r, n) for r in rs], len(refs))
        ks.append(k); ds.append(d); adds.append(a)
    keep, dele, addf = sum(ks) / 4, sum(ds) / 4, sum(adds) / 4
    return {"sari": 100 * (keep + dele + addf) / 3, "keep": 100 * keep, "del": 100 * dele, "add": 100 * addf}


def corpus_sari(sources: Sequence[str], preds: Sequence[str], refs: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Mean of sentence SARI (the HF/EASSE-style corpus score is the sentence average)."""
    rows = [sari_sent(s, p, r) for s, p, r in zip(sources, preds, refs)]
    n = max(1, len(rows))
    return {k: sum(r[k] for r in rows) / n for k in ("sari", "keep", "del", "add")}


def _edit_set(before: Sequence[int], after: Sequence[int]):
    """Token-level edit operations as a multiset: ('del', before_pos) and ('ins', before_pos, token)."""
    ops = []
    for i1, i2, j1, j2 in diff_hunks(list(before), list(after)):
        ops += [("del", i) for i in range(i1, i2)]
        ops += [("ins", i1, t) for t in after[j1:j2]]
    return Counter(ops)


def edit_prf(before: Sequence[int], pred: Sequence[int], ref: Sequence[int]) -> Dict[str, float]:
    """Precision/recall of predicted token edits (relative to the draft) against the reference edits."""
    p, r = _edit_set(before, pred), _edit_set(before, ref)
    tp = sum((p & r).values())
    np_, nr = sum(p.values()), sum(r.values())
    return {"tp": tp, "n_pred_edits": np_, "n_ref_edits": nr,
            "precision": tp / np_ if np_ else (1.0 if nr == 0 else 0.0),
            "recall": tp / nr if nr else 1.0}
