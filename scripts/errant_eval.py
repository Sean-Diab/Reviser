"""ERRANT span F0.5 + corpus BLEU + ROUGE-L for evaluate_edits / evaluate_external outputs (task 26 metrics).

Custom extension, labelled as such: the official BEA-2019 / W&I+LOCNESS test protocol is NOT used (its data needs a
licence acceptance that was not given). ERRANT (Bryant et al. 2017) is applied to the CoEdIT validation `gec` intent
and to the IteraTeR `fluency` intent as sentence-level GEC-style references (one reference each).

Metrics per file:
- ERRANT span-based correction, corpus level: TP/FP/FN over (o_start, o_end, correction) edits between source->hyp and
  source->reference (ERRANT's default `errant_compare` span mode). Reports P, R, F0.5 and per-category F0.5 on the
  main error types (operation prefix removed). The copy baseline proposes no edits: its P is undefined (shown as '-').
- Uncertainty: record-level bootstrap (--boot resamples, seed 0) 95% CI of corpus F0.5 per system, and paired bootstrap CI of
  the F0.5 difference vs the FIRST file given (same resampled records for both systems).
- sacrebleu corpus BLEU (13a tokenization) and ROUGE-L F (rouge_score, stemming) vs the single reference. A copy
  baseline row is always included.
Usage (WSL; extras in .pydeps/errant-env, not on the default path):
  PYTHONPATH=.pydeps/errant-env python scripts/errant_eval.py --data coedit --split validation --intent gec \\
      runs/re100-d4-s1/eval/edits_coedit_validation_tgreedy.jsonl ... [--md out.md]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_refs(data, split):
    out = {}
    with open(ROOT / "data" / "real" / f"{data}_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def edit_set(annotator, cache, src, tgt):
    key = (src, tgt)
    if key not in cache:
        o = cache.get(("parse", src)) or annotator.parse(src)
        cache[("parse", src)] = o
        c = annotator.parse(tgt)
        cache[key] = {(e.o_start, e.o_end, e.c_str, e.type) for e in annotator.annotate(o, c) if e.type != "UNK"}
    return cache[key]


def fb(tp, fp, fn, beta=0.5):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--intent", default=None, help="restrict to one intent (e.g. gec, fluency)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--strata", default=None,
                    help="edit-density strata by ERRANT reference edits per record, e.g. '0-1,2-3,4-7,8-999': corpus F0.5 "
                         "per stratum and the paired bootstrap difference vs the first file within each stratum")
    a = ap.parse_args()
    import errant
    import sacrebleu
    from rouge_score import rouge_scorer
    annotator = errant.load("en")
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs = load_refs(a.data, a.split)
    uids = [u for u, r in refs.items() if a.intent is None or r.get("intent") == a.intent]
    cache = {}
    rows = []
    systems = [("baseline:copy", {u: refs[u]["before_text"] for u in uids})]
    for f in a.files:
        preds = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            preds[r["uid"]] = r.get("pred", "")
        missing = [u for u in uids if u not in preds]
        assert not missing, f"{f}: {len(missing)} uids missing"
        systems.append((f, preds))
    for name, preds in systems:
        tp = fp = fn = 0
        per = []
        cat = collections.defaultdict(lambda: [0, 0, 0])
        hyps, rfs, rl = [], [], []
        for u in uids:
            src, ref, hyp = refs[u]["before_text"], refs[u]["after_text"], preds[u]
            H = edit_set(annotator, cache, src, hyp)
            R = edit_set(annotator, cache, src, ref)
            Hk = {e[:3]: e[3] for e in H}
            Rk = {e[:3]: e[3] for e in R}
            per.append((sum(k in Rk for k in Hk), sum(k not in Rk for k in Hk), sum(k not in Hk for k in Rk)))
            for k, t in Hk.items():
                c = t.split(":", 1)[1] if ":" in t else t
                if k in Rk:
                    tp += 1
                    cat[c][0] += 1
                else:
                    fp += 1
                    cat[c][1] += 1
            for k, t in Rk.items():
                if k not in Hk:
                    fn += 1
                    cat[t.split(":", 1)[1] if ":" in t else t][2] += 1
            hyps.append(hyp)
            rfs.append(ref)
            rl.append(rs.score(ref, hyp)["rougeL"].fmeasure)
        p, r, f05 = fb(tp, fp, fn)
        bleu = sacrebleu.corpus_bleu(hyps, [rfs]).score
        top = sorted(cat.items(), key=lambda kv: -(kv[1][0] + kv[1][2]))[:6]
        cats = ", ".join(f"{c} {fb(*v)[2]:.2f} ({v[0] + v[2]})" for c, v in top)
        rows.append([name, len(uids), tp, fp, fn, p, r, f05, bleu, 100 * sum(rl) / len(rl), cats, np.array(per)])
        print(f"{name}: n={len(uids)} P={p:.3f} R={r:.3f} F0.5={f05:.3f} BLEU={bleu:.1f} ROUGE-L={100 * sum(rl) / len(rl):.1f}", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(uids), size=(a.boot, len(uids)))

    def fboot(per):
        t = per[idx].sum(1)
        return np.array([fb(*x)[2] for x in t])
    base = fboot(rows[1][11]) if len(rows) > 1 else None
    for x in rows:
        b = fboot(x[11])
        x.append("%.3f-%.3f" % tuple(np.percentile(b, [2.5, 97.5])))
        d = b - base if base is not None and x is not rows[1] else None
        x.append("-" if d is None else "%+.3f [%+.3f, %+.3f]" % (x[7] - rows[1][7], *np.percentile(d, [2.5, 97.5])))
    import errant as _e
    L = [f"ERRANT {getattr(_e, '__version__', '?')}, sacrebleu {sacrebleu.__version__}; data {a.data}/{a.split}"
         + (f", intent={a.intent}" if a.intent else "") + "; single reference; custom extension (not the BEA-2019 protocol).", "",
         f"Bootstrap: {a.boot} record resamples, seed 0; paired difference vs `{rows[1][0] if len(rows) > 1 else '-'}`.", "",
         "| system | n | TP | FP | FN | P | R | F0.5 | F0.5 95% CI | dF0.5 vs first [95% CI] | BLEU | ROUGE-L | top categories F0.5 (ref+hyp support) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        P = "-" if x[2] + x[3] == 0 else "%.3f" % x[5]
        L.append("| %s | %d | %d | %d | %d | %s | %.3f | %.3f | %s | %s | %.1f | %.1f | %s |"
                 % (x[0], x[1], x[2], x[3], x[4], P, x[6], x[7], x[12], x[13], x[8], x[9], x[10]))
    if a.strata and len(rows) > 1:
        nref = rows[0][11][:, 2]  # copy baseline has no hypothesis edits, so its FN = reference edits per record
        L += ["", f"Edit-density strata (ERRANT reference edits per record): corpus F0.5 per stratum; d = paired bootstrap "
                  f"difference vs `{rows[1][0]}` within the stratum.", "",
              "| ref edits | n | " + " | ".join(x[0] for x in rows[1:]) + " |", "|---|---|" + "---|" * (len(rows) - 1)]
        for st in a.strata.split(","):
            lo, hi = map(int, st.split("-"))
            sel = np.where((nref >= lo) & (nref <= hi))[0]
            if len(sel) == 0:
                continue
            sidx = sel[rng.integers(0, len(sel), size=(a.boot, len(sel)))]
            f0 = fb(*rows[1][11][sel].sum(0))[2]
            b0 = np.array([fb(*t)[2] for t in rows[1][11][sidx].sum(1)])
            cells = ["%.3f" % f0]
            for x in rows[2:]:
                f = fb(*x[11][sel].sum(0))[2]
                d = np.array([fb(*t)[2] for t in x[11][sidx].sum(1)]) - b0
                cells.append("%.3f (d %+.3f [%+.3f, %+.3f])" % (f, f - f0, *np.percentile(d, [2.5, 97.5])))
            L.append(f"| {lo}-{hi} | {len(sel)} | " + " | ".join(cells) + " |")
    out = "\n".join(L) + "\n"
    print(out)
    if a.md:
        Path(a.md).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
