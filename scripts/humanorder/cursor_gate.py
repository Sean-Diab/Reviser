"""Gate for the neural cursor model (version N) on the CoAuthor test windows.

Compares the full model (reader + picker), the features-only picker, "always keep typing" (left to right), July's
run-and-jump dials and (if present) the Q agent's fitted dials:

1. Likelihood of the human orders: mean NLL per pick (all / keep / skip / move), jump-target top-1/top-5 (models
   only), bootstrap 95% CI over windows of the NLL difference full - features-only (and full - dials).
2. Orders each method generates on the same texts (temperature 1) vs the human orders: run length mean/median/p90,
   jumps per 100 tokens, back share, jump distance median/p90 and histogram, actions per output token, share of
   inserts at the end of the canvas.

Rule fixed before the results (task brief 2026-10-07): PASS if (a) the full model's NLL per pick is below the
features-only picker's with the bootstrap 95% CI of the difference entirely below 0, and (b) the full model's
generated statistics are closer to human than July's dials, measured as the mean |log(stat / human stat)| over the
nine scalar statistics in STAT_KEYS.

  python scripts/humanorder/cursor_gate.py --full runs/humanorder_n/full_s0 --feat runs/humanorder_n/feat_s0 \
      --out runs/humanorder_n/gate [--q_dials data/humanorder/q_dials_v1.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from orders import order_to_actions, run_lengths  # noqa: E402
from runjump_baseline import JULY_DIALS, load_dials, runjump_logprobs, runjump_order  # noqa: E402

STAT_KEYS = ("run_mean", "run_median", "run_p90", "jumps_per_100", "back_share", "jump_dist_median",
             "jump_dist_p90", "actions_per_token", "end_share")
HIST_EDGES = [1, 2, 3, 5, 9, 17, 33, 65, 10 ** 9]   # |jump| buckets: 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+
HIST_NAMES = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]


def pick_kinds(order):
    """0 keep (next token at the cursor), 1 skip (other token of the cursor's gap), 2 move."""
    present, last, out = [], -1, []
    for j in order:
        pos = bisect.bisect_left(present, j)
        cur = bisect.bisect_right(present, last) if last >= 0 else 0
        out.append(0 if j == last + 1 else (1 if pos == cur else 2))
        present.insert(pos, j)
        last = j
    return out


def end_appends(order):
    present, n = [], 0
    for j in order:
        pos = bisect.bisect_left(present, j)
        n += pos == len(present)
        present.insert(pos, j)
    return n


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else float("nan")


def order_stats(items, orders):
    """Pooled statistics over (item, order) pairs."""
    runs, jumps, toks, acts, ends = [], [], 0, 0, 0
    for it, o in zip(items, orders):
        r, j = run_lengths(o)
        runs += r
        jumps += j
        toks += len(o)
        acts += len(order_to_actions(it["prefix"], it["cont"], o)) - len(it["prefix"]) - 1
        ends += end_appends(o)
    ad = [abs(x) for x in jumps]
    hist_b = np.histogram([abs(x) for x in jumps if x < 0], bins=HIST_EDGES)[0].tolist()
    hist_f = np.histogram([x for x in jumps if x > 0], bins=HIST_EDGES)[0].tolist()
    return {
        "texts": len(orders), "tokens": toks, "runs": len(runs), "jumps": len(jumps),
        "run_mean": statistics.mean(runs), "run_median": statistics.median(runs), "run_p90": pct(runs, 90),
        "jumps_per_100": 100.0 * len(jumps) / toks,
        "back_share": sum(x < 0 for x in jumps) / len(jumps) if jumps else float("nan"),
        "jump_dist_median": statistics.median(ad) if ad else float("nan"),
        "jump_dist_p90": pct(ad, 90) if ad else float("nan"),
        "actions_per_token": acts / toks, "end_share": ends / toks,
        "hist_back": hist_b, "hist_fwd": hist_f,
    }


def distance_to_human(st, hu):
    terms = {}
    for k in STAT_KEYS:
        a, b = st[k], hu[k]
        terms[k] = abs(math.log(a / b)) if (a > 0 and b > 0 and not math.isnan(a)) else float("inf")
    fin = [v for v in terms.values()]
    return sum(fin) / len(fin), terms


def boot_ci(num_a, den_a, num_b=None, den_b=None, n_boot=10000, seed=0):
    """95% CI over windows of sum(num_a)/sum(den_a) [- sum(num_b)/sum(den_b)]."""
    rng = np.random.default_rng(seed)
    W = len(num_a)
    idx = rng.integers(0, W, size=(n_boot, W))
    v = num_a[idx].sum(1) / den_a[idx].sum(1)
    point = num_a.sum() / den_a.sum()
    if num_b is not None:
        v = v - num_b[idx].sum(1) / den_b[idx].sum(1)
        point = point - num_b.sum() / den_b.sum()
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def dials_eval(items, dials):
    """Per-window NLL sums by pick kind under RunJumpPolicy dials."""
    W = len(items)
    r = {k: np.zeros(W) for k in ("nll", "n", "nll_keep", "n_keep", "nll_skip", "n_skip", "nll_move", "n_move",
                                  "impossible")}
    for i, it in enumerate(items):
        lps = runjump_logprobs(it["order"], dials)
        ks = pick_kinds(it["order"])
        for lp, k in zip(lps, ks):
            name = ("keep", "skip", "move")[k]
            r["impossible"][i] += lp == float("-inf")
            r["nll"][i] -= lp
            r["n"][i] += 1
            r[f"nll_{name}"][i] -= lp
            r[f"n_{name}"][i] += 1
    return r


def summ(r):
    s = lambda k: float(r[k].sum())
    out = {"nll_per_pick": s("nll") / s("n")}
    if "impossible" in r:
        out["impossible_picks"] = int(r["impossible"].sum())
        fin = np.isfinite(r["nll"])
        out["nll_per_pick_windows_without_impossible"] = float(r["nll"][fin].sum() / r["n"][fin].sum())
        out["windows_without_impossible"] = int(fin.sum())
    for k in ("keep", "skip", "move"):
        out[f"nll_{k}"] = s(f"nll_{k}") / max(1.0, s(f"n_{k}"))
    out["nll_jump_any"] = (s("nll_skip") + s("nll_move")) / max(1.0, s("n_skip") + s("n_move"))
    return out


def fmt(x, nd=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data/humanorder/coauthor_v1"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--q_dials", default=os.path.join(ROOT, "data/humanorder/q_dials_v1.json"))
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.data, f"windows_{a.split}.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    W = len(items)
    rep = {"split": a.split, "windows": W, "picks": sum(len(it["cont"]) for it in items), "rule": __doc__.split("Rule")[1].split("python")[0].strip()}

    # 1. likelihood of the human orders
    nll = {}
    win = {}
    for name, d in (("full", a.full), ("features_only", a.feat)):
        with open(os.path.join(d, f"eval_{a.split}.json")) as f:
            nll[name] = json.load(f)
        win[name] = dict(np.load(os.path.join(d, f"eval_{a.split}_windows.npz")))
    dial_sets = {"july_dials": JULY_DIALS}
    if a.q_dials and os.path.exists(a.q_dials):
        qd, extra = load_dials(a.q_dials)
        dial_sets["q_dials"] = qd
        rep["q_dials"] = {"path": a.q_dials, "dials": qd}
    for name, dials in dial_sets.items():
        win[name] = dials_eval(items, dials)
        nll[name] = summ(win[name])
    kinds = [pick_kinds(it["order"]) for it in items]
    n_nonkeep = sum(sum(k != 0 for k in ks) for ks in kinds)
    nll["left_to_right"] = {"nll_per_pick": float("inf"), "nll_keep": 0.0, "nll_skip": float("inf"),
                            "nll_move": float("inf"), "note": f"infinite on the {n_nonkeep} non-keep picks"}
    rep["nll"] = nll
    ci = {"full_minus_features_only": boot_ci(win["full"]["nll"], win["full"]["n"],
                                               win["features_only"]["nll"], win["features_only"]["n"])}
    for name in dial_sets:
        if np.isfinite(win[name]["nll"]).all():
            ci[f"full_minus_{name}"] = boot_ci(win["full"]["nll"], win["full"]["n"], win[name]["nll"], win[name]["n"])
    for k in ("move", "skip"):
        ci[f"full_minus_features_only_{k}"] = boot_ci(win["full"][f"nll_{k}"], win["full"][f"n_{k}"],
                                                       win["features_only"][f"nll_{k}"], win["features_only"][f"n_{k}"])
    rep["nll_ci"] = ci

    # 2. generated orders vs human
    stats = {"human": order_stats(items, [it["order"] for it in items])}
    for name, d in (("full", a.full), ("features_only", a.feat)):
        with open(os.path.join(d, f"gen_{a.split}.json")) as f:
            g = json.load(f)
        allo = g["orders"]
        stats[name] = order_stats(items * len(allo), [o for s in allo for o in s])
        stats[name]["gen_seed"], stats[name]["gen_samples"] = g["seed"], g["samples"]
    stats["left_to_right"] = order_stats(items, [list(range(len(it["cont"]))) for it in items])
    for name, dials in dial_sets.items():
        orders = []
        for s in range(a.samples):
            rng = random.Random(a.seed + s)
            orders += [runjump_order(dials, it["prefix"], it["cont"], rng) for it in items]
        stats[name] = order_stats(items * a.samples, orders)
    dist = {}
    for name, st in stats.items():
        if name == "human":
            continue
        dist[name], terms = distance_to_human(st, stats["human"])
        st["log_ratio_terms"] = terms
    rep["gen_stats"] = stats
    rep["distance_to_human"] = dist
    c = ci["full_minus_features_only"]
    pass_a = c["hi"] < 0
    pass_b = dist["full"] < dist["july_dials"]
    rep["verdict"] = {"nll_full_below_features_only_ci": pass_a, "gen_closer_than_july": pass_b,
                      "pass": bool(pass_a and pass_b)}
    with open(os.path.join(a.out, "gate.json"), "w") as f:
        json.dump(rep, f, indent=1)

    # markdown tables
    L = [f"Test windows: {W}, picks: {rep['picks']}.", "", "| Method | NLL/pick | keep | skip | move | move top-1 | move top-5 | skip top-1 |",
         "|---|---|---|---|---|---|---|---|"]
    for name in ["full", "features_only"] + list(dial_sets) + ["left_to_right"]:
        r = nll[name]
        L.append(f"| {name} | {fmt(r['nll_per_pick'], 4)} | {fmt(r['nll_keep'], 4)} | {fmt(r['nll_skip'])} | "
                 f"{fmt(r['nll_move'])} | {fmt(r.get('top1_move', '-'))} | {fmt(r.get('top5_move', '-'))} | "
                 f"{fmt(r.get('top1_skip', '-'))} |")
    for name in dial_sets:
        r = nll[name]
        if r.get("impossible_picks"):
            L.append(f"\n{name}: {r['impossible_picks']} human picks have probability 0 (NLL infinite); NLL/pick over the "
                     f"{r['windows_without_impossible']} windows without such picks = {r['nll_per_pick_windows_without_impossible']:.4f}")
    cal = [n for n in ("full", "features_only") if "share_keep_predicted" in nll[n]]
    if cal:
        L += ["", "| Teacher-forced calibration (share of picks) | keep actual | keep predicted | skip actual | skip predicted | move actual | move predicted |",
              "|---|---|---|---|---|---|---|"]
        for n in cal:
            r = nll[n]
            L.append(f"| {n} | " + " | ".join(f"{r[f'share_{k}_{w}']:.4f}" for k in ("keep", "skip", "move")
                                              for w in ("actual", "predicted")) + " |")
    L += ["", "| Difference (bootstrap 95% CI over windows) | point | lo | hi |", "|---|---|---|---|"]
    for k, v in ci.items():
        L.append(f"| {k} | {v['point']:.4f} | {v['lo']:.4f} | {v['hi']:.4f} |")
    L += ["", "| Orders | run mean | run median | run p90 | jumps/100 | back share | jump med | jump p90 | actions/token | end share | distance |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, st in stats.items():
        L.append(f"| {name} | {fmt(st['run_mean'], 1)} | {fmt(float(st['run_median']), 1)} | {fmt(st['run_p90'], 1)} | "
                 f"{fmt(st['jumps_per_100'], 2)} | {fmt(st['back_share'], 3)} | {fmt(float(st['jump_dist_median']), 1)} | "
                 f"{fmt(st['jump_dist_p90'], 1)} | {fmt(st['actions_per_token'], 4)} | {fmt(st['end_share'], 3)} | "
                 f"{fmt(dist.get(name, 0.0), 3)} |")
    L += ["", "| Jump distance histogram (back / forward share of jumps) | " + " | ".join(HIST_NAMES) + " |",
          "|---|" + "---|" * len(HIST_NAMES)]
    for name, st in stats.items():
        nj = max(1, st["jumps"])
        L.append(f"| {name} back | " + " | ".join(f"{x / nj:.3f}" for x in st["hist_back"]) + " |")
        L.append(f"| {name} fwd | " + " | ".join(f"{x / nj:.3f}" for x in st["hist_fwd"]) + " |")
    L += ["", f"Verdict: {json.dumps(rep['verdict'])}"]
    with open(os.path.join(a.out, "gate.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
