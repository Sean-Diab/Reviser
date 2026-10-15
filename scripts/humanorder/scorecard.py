#!/usr/bin/env python3
"""Per-benchmark win/loss lines for one classical-Reviser arm against July@84M (notes/overnight-plan-2026-10-07.md,
"Benchmarks: all count equally"). Paired over the same prompt ids. Run in WSL /opt/venv after q_compare.sh (and the
judges, if done).

Rows and the direction used for "win":
- evalPPL (paper definition), C4 and DCLM, GPT-2 base and GPT-2 large scorers: lower. From compare.py's paired
  bootstrap (10,000 resamples, seed 0).
- First-64 NLL (GPT-2 base and large): lower. compare.py.
- Stop rate: lower (ending early is the failure this row watches). compare.py.
- Length gap: |output length - human continuation length| per prompt; lower.
- Repetition seq-rep-4 (per text: 1 - unique 4-grams / 4-grams, texts with >= 4 tokens; as scripts/july_text_stats.py):
  lower. Diversity distinct-2 / distinct-3 (pooled unique / total n-grams): higher.
- Moves and back-moves per output token in the model's own decode actions: higher (= uses the cursor more; more is not
  by itself better writing). Deletes per output token: higher.
- Blind LLM judges (scripts/judge_continuations.py analysis.json, length-matched condition, both orders): arm score > 0.5.
A row is "win" / "loss" when its 95% CI excludes no-difference on that side, otherwise "tie".
CIs for per-prompt means: 10,000 paired resamples (seed 0); ratios of sums: 10,000; pooled distinct-n: 1,000 paired
half-samples without replacement (see boot_distinct_diff).

  python scripts/humanorder/scorecard.py --arm q84 [--judge runs/judge/q84_vs_july84/analysis.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = [("C4", "c4.paper.s20260924"), ("DCLM", "dclm.july.s20260924")]
PREFIX = 35
MOVE_START, MOVE_END = 50261, 50281


def load_dec(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["id"]: r for r in rows[1:]}


def ngrams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


def feats(r):
    c = r["canvas_ids"][PREFIX:]
    g4 = ngrams(c, 4)
    back = sum(1 for t in r["actions_new"] if MOVE_START <= t < MOVE_END and (t - MOVE_START) % 2 == 1)
    return dict(cont=c, rep4=(1 - len(set(g4)) / len(g4)) if len(g4) else None, out_len=max(0, r["out_len"]),
                moves=r["counts"]["move"], back=back, deletes=r["counts"]["delete"],
                gap=abs(r["out_len"] - len(r["ref_cont_ids"])))


def boot_mean(x, n=10000, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, x.size, (n, x.size))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_ratio_diff(num_b, den_b, num_a, den_a, n=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(num_a), (n, len(num_a)))
    f = lambda nu, de: nu[idx].sum(1) / np.maximum(1, de[idx].sum(1))
    d = f(num_b, den_b) - f(num_a, den_a)
    pt = num_b.sum() / max(1, den_b.sum()) - num_a.sum() / max(1, den_a.sum())
    return float(pt), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


class Pool:
    """Pooled distinct-n with a paired bootstrap over prompts (n-grams mapped to ints once)."""

    def __init__(self, conts, n):
        voc = {}
        self.arrs = [np.array([voc.setdefault(g, len(voc)) for g in ngrams(c, n)], dtype=np.int64) for c in conts]

    def value(self, idx):
        a = np.concatenate([self.arrs[i] for i in idx])
        return len(np.unique(a)) / max(1, a.size)


def boot_distinct_diff(pb, pa, n_items, n=1000, seed=0):
    """Pooled distinct-n is not a per-prompt mean, and resampling with replacement repeats texts (their n-grams then
    count as repeats). So: half-samples without replacement (paired), whose spread matches a full-size bootstrap
    for mean-like statistics; the CI is the point difference plus the 2.5/97.5 percentiles of the centred
    half-sample differences."""
    rng = np.random.default_rng(seed)
    full = np.arange(n_items)
    d = []
    for _ in range(n):
        idx = rng.choice(n_items, n_items // 2, replace=False)
        d.append(pb.value(idx) - pa.value(idx))
    d = np.array(d) - np.mean(d)
    pt = float(pb.value(full) - pa.value(full))
    return pt, [pt + float(np.percentile(d, 2.5)), pt + float(np.percentile(d, 97.5))]


def verdict(ci, better):
    lo, hi = ci
    good_side = (hi < 0) if better == "lower" else (lo > 0)
    bad_side = (lo > 0) if better == "lower" else (hi < 0)
    return "win" if good_side else ("loss" if bad_side else "tie")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", default="july11656")
    ap.add_argument("--others", nargs="*", default=["ctl11766", "base"])
    ap.add_argument("--dec", default="runs/humanorder/eval/dec")
    ap.add_argument("--base_dec", default="runs/july_control/eval/dec")
    ap.add_argument("--compare_dir", default="runs/humanorder/eval")
    ap.add_argument("--judge", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    arm, ref = a.arm, a.ref
    cols = [ref, arm] + a.others
    rows, details = [], {}

    def row(bench, better, vals, diff, ci, result=None, note=""):
        rows.append(dict(benchmark=bench, better=better, values=vals, diff=diff, ci95=ci,
                         result=result or verdict(ci, better), note=note))

    for set_name, stem in SETS:
        cmp = {sc: json.loads((ROOT / a.compare_dir / f"compare.{stem}.{sc}.json").read_text())
               for sc in ("gpt2_base", "gpt2_large")}
        key = f"{arm} - {ref}"
        for sc, lab in (("gpt2_base", "GPT-2"), ("gpt2_large", "GPT-2 large")):
            c = cmp[sc]
            p = c["pairs"][key]
            vals = {m: c["models"][m]["evalPPL_paper_def"] for m in cols if m in c["models"]}
            row(f"{set_name} evalPPL, {lab}", "lower", vals, p["PRIMARY_paper_def_ppl_diff"]["mean"],
                p["PRIMARY_paper_def_ppl_diff"]["ci95"])
        for sc, lab in (("gpt2_base", "GPT-2"), ("gpt2_large", "GPT-2 large")):
            c = cmp[sc]
            p = c["pairs"][key]
            vals = {m: c["models"][m]["ppl_first64"] for m in cols if m in c["models"]}
            row(f"{set_name} first-64 NLL, {lab} (values: PPL of first 64)", "lower", vals,
                p["first64_nll_diff"]["mean"], p["first64_nll_diff"]["ci95"],
                note=f"n={p['first64_nll_diff']['n']} prompts where both wrote >= 64 tokens")
        c = cmp["gpt2_base"]
        p = c["pairs"][key]
        row(f"{set_name} stop rate", "lower", {m: c["models"][m]["stop_eor_rate"] for m in cols if m in c["models"]},
            p["stop_rate_diff"]["mean"], p["stop_rate_diff"]["ci95"])
        details[f"{set_name}_out_len_diff"] = p["out_len_diff"]
        details[f"{set_name}_out_len"] = {m: c["models"][m]["out_len"] for m in cols if m in c["models"]}

        dec = {m: load_dec(ROOT / (a.base_dec if m == "base" else a.dec) / f"{stem}.{m}.jsonl") for m in cols}
        ids = sorted(set.intersection(*[set(d) for d in dec.values()]))
        F = {m: [feats(dec[m][i]) for i in ids] for m in cols}
        human = [dec[ref][i]["ref_cont_ids"] for i in ids]
        arr = lambda m, k: np.array([f[k] for f in F[m]], float)

        # length gap to the human continuation
        vals = {m: float(arr(m, "gap").mean()) for m in cols}
        mu, ci = boot_mean(arr(arm, "gap") - arr(ref, "gap"))
        row(f"{set_name} length gap to human continuation (tokens)", "lower", vals, mu, ci,
            note=f"human continuation mean {np.mean([len(h) for h in human]):.1f} tokens; output length "
                 + ", ".join(f"{m} {arr(m, 'out_len').mean():.1f}" for m in cols))
        # repetition
        both = [k for k in range(len(ids)) if F[arm][k]["rep4"] is not None and F[ref][k]["rep4"] is not None]
        vals = {m: float(np.mean([f["rep4"] for f in F[m] if f["rep4"] is not None])) for m in cols}
        hrep = [1 - len(set(ngrams(h, 4))) / len(ngrams(h, 4)) for h in human if len(h) >= 4]
        vals["human"] = float(np.mean(hrep))
        mu, ci = boot_mean([F[arm][k]["rep4"] - F[ref][k]["rep4"] for k in both])
        row(f"{set_name} repetition seq-rep-4", "lower", vals, mu, ci, note=f"paired n={len(both)}")
        # diversity
        for n in (2, 3):
            pools = {m: Pool([f["cont"] for f in F[m]], n) for m in cols}
            full = np.arange(len(ids))
            vals = {m: pools[m].value(full) for m in cols}
            vals["human"] = Pool(human, n).value(full)
            mu, ci = boot_distinct_diff(pools[arm], pools[ref], len(ids))
            row(f"{set_name} diversity distinct-{n}", "higher", vals, mu, ci)
        # cursor use
        for k, lab in (("moves", "moves"), ("back", "back-moves"), ("deletes", "deletes")):
            vals = {m: float(arr(m, k).sum() / max(1, arr(m, "out_len").sum())) for m in cols}
            if arr(arm, k).sum() == 0 and arr(ref, k).sum() == 0:
                row(f"{set_name} {lab} per output token", "higher", vals, 0.0, [0.0, 0.0], result="n/a (none)")
                continue
            mu, ci = boot_ratio_diff(arr(arm, k), arr(arm, "out_len"), arr(ref, k), arr(ref, "out_len"))
            row(f"{set_name} {lab} per output token", "higher", vals, mu, ci)
        # prompt kept intact: a guard row, not counted in the tally (notes/overnight-plan-2026-10-07.md, "Keep strong
        # cursor use", added ~10:00Z): share of outputs whose 35-token prompt changed (decode field prefix_intact);
        # fails above 2x July@84M's share on the same prompts.
        chg = {m: np.array([0.0 if dec[m][i].get("prefix_intact", True) else 1.0 for i in ids]) for m in cols}
        vals = {m: float(chg[m].mean()) for m in cols}
        mu, ci = boot_mean(chg[arm] - chg[ref])
        if "july11656" in chg:
            lim = 2 * vals["july11656"]
            res = "guard pass" if vals[arm] <= lim else "guard FAIL"
            note = f"guard: {arm} {vals[arm]:.3f} vs limit {lim:.3f} (2x July@84M); not counted in the tally"
        else:
            res, note = "guard n/a", "July@84M not among the columns; not counted in the tally"
        row(f"{set_name} prompt changed (share of outputs; guard)", "lower", vals, mu, ci, result=res, note=note)
        details[f"{set_name}_back_move_share"] = {
            m: float(arr(m, "back").sum() / max(1, arr(m, "moves").sum())) for m in cols}
        details[f"{set_name}_n_ids"] = len(ids)

    judged = None
    if a.judge and (ROOT / a.judge).exists():
        jd = json.loads((ROOT / a.judge).read_text())
        judged = jd
        for j in ("claude", "codex"):
            d = jd["judges"].get(j, {}).get("lm")
            if not d:
                row(f"Blind judge {j} (length-matched)", "higher", {}, None, None, result="pending")
                continue
            ci = [d["ci95"][0] - 0.5, d["ci95"][1] - 0.5]
            row(f"Blind judge {j} (length-matched, 100 prompts, both orders; values: {arm} score)", "higher",
                {arm: d["rev_score"]}, d["rev_score"] - 0.5, ci,
                note=f"{arm} wins {d['rev_wins']}, July wins {d['ar_wins']}, ties {d['ties']}, sign p={d['sign_p']:.3g};"
                     f" position consistency {d.get('position_consistency', float('nan')):.2f}")
    else:
        for j in ("claude", "codex"):
            row(f"Blind judge {j} (length-matched)", "higher", {}, None, None, result="pending")

    tally = {r: sum(x["result"] == r for x in rows) for r in ("win", "tie", "loss")}
    rep = dict(arm=arm, ref=ref, columns=cols, rows=rows, tally=tally, details=details,
               judge_file=a.judge or None, judge_agreement=(judged or {}).get("judge_agreement"))
    out = Path(a.out or f"{a.compare_dir}/scorecard.{arm}.json")
    (ROOT / out).write_text(json.dumps(rep, indent=1))
    fmt = lambda v: "—" if v is None else (f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}")
    lines = [f"| Benchmark (better) | " + " | ".join(cols + ["human"]) + f" | {arm} − {ref} [95% CI] | Result |",
             "|---|" + "---|" * (len(cols) + 3)]
    for x in rows:
        ci = "" if x["ci95"] is None else f" [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}]"
        lines.append(f"| {x['benchmark']} ({x['better']}) | " + " | ".join(fmt(x["values"].get(m)) for m in cols + ["human"])
                     + f" | {fmt(x['diff'])}{ci} | {x['result']} |")
    lines.append(f"\nTally vs {ref}: {tally['win']} win, {tally['tie']} tie, {tally['loss']} loss "
                 f"({sum(x['result'] == 'pending' for x in rows)} pending).")
    (ROOT / out.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
