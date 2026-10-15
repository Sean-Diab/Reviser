#!/usr/bin/env python3
"""Q: fit July's run-and-jump dials to human writing-order statistics (CoAuthor windows).

Human orders: data/humanorder/coauthor_v1/windows_{train,dev,test}.jsonl ({prefix (35 ids), cont, order, src}).
Policy orders: July's RunJumpPolicy (experiments/july_control/code/july_impl/gen_runjump.py, unmodified, imported)
run on the SAME windows (same prefix and continuation), or a policy from runjump6.py if a sixth dial is used.

Every statistic is computed from the raw action stream with the same code for humans and policies:
  runs   = maximal groups of inserts between cursor moves (the first run starts at the prefix end)
  jumps  = maximal groups of consecutive move tokens; distance = sum of the move amounts (written tokens)
  actions per output token = (moves + inserts) / continuation tokens (EOR excluded, as in coauthor_v1 summary.json)
  plus experiments/july_control/code/july_impl/traj_stats.aggregate (action mix, end-append share, ...)

  python scripts/humanorder/fit_q_dials.py human                 # human targets (train/dev/test, with/without AI)
  python scripts/humanorder/fit_q_dials.py grid --out runs/humanorder/fit/grid.json [--n_windows 0]
  python scripts/humanorder/fit_q_dials.py eval --dials '{"lambda_cont":..}' [--split test]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "code" / "july_impl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import traj_stats  # noqa: E402  (July, unmodified)
from gen_runjump import RunJumpPolicy  # noqa: E402  (July, unmodified)
from orders import order_to_actions  # noqa: E402

WIN = ROOT / "data" / "humanorder" / "coauthor_v1"
PREFIX = 35
EOR, MOVE_START = 50260, 50261
MOVES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
JULY_DIALS = {"lambda_cont": 0.82, "jump_alpha": 1.5, "p_skip": 0.3, "skip_q": 0.85, "p_far": 0.9}


def load_windows(split: str, drop_ai: bool = False):
    out = []
    for line in open(WIN / f"windows_{split}.jsonl", encoding="utf-8"):
        w = json.loads(line)
        if drop_ai:
            keep = [j for j in range(len(w["cont"])) if w["src"][j] != "a"]
            if not keep:
                continue
            new = {j: k for k, j in enumerate(keep)}
            w = {**w, "cont": [w["cont"][j] for j in keep], "order": [new[j] for j in w["order"] if j in new],
                 "src": "".join(w["src"][j] for j in keep)}
        out.append(w)
    return out


def parse_stream(actions, prefix_len=PREFIX):
    """(runs, jumps, n_moves, n_inserts) from a raw action stream."""
    runs, jumps = [], []
    run = 0
    jump = None
    n_mv = n_ins = 0
    for t in actions[prefix_len:]:
        t = int(t)
        if MOVE_START <= t < MOVE_START + 20:
            off = t - MOVE_START
            amt = MOVES[off // 2] * (-1 if off % 2 else 1)
            n_mv += 1
            if run:
                runs.append(run)
                run = 0
            jump = amt if jump is None else jump + amt
        elif t == EOR:
            break
        else:
            n_ins += 1
            if jump is not None:
                jumps.append(jump)
                jump = None
            run += 1
    if run:
        runs.append(run)
    return runs, jumps, n_mv, n_ins


def stream_stats(streams, full=False):
    runs, jumps = [], []
    n_mv = n_ins = 0
    for a in streams:
        r, j, m, i = parse_stream(a)
        runs += r
        jumps += j
        n_mv += m
        n_ins += i
    rl = np.asarray(runs, float)
    ad = np.abs(np.asarray(jumps, float)) if jumps else np.zeros(1)
    out = {
        "windows": len(streams), "cont_tokens": n_ins,
        "run_len_mean": float(rl.mean()), "run_len_median": float(np.median(rl)),
        "run_len_p90": float(np.percentile(rl, 90)),
        "jumps_per_100_tokens": 100.0 * len(jumps) / max(1, n_ins),
        "back_jump_share": float(np.mean(np.asarray(jumps) < 0)) if jumps else 0.0,
        "jump_dist_median": float(np.median(ad)), "jump_dist_mean": float(ad.mean()),
        "jump_dist_p90": float(np.percentile(ad, 90)),
        "jump_dist_le2": float(np.mean(ad <= 2)), "jump_dist_le8": float(np.mean(ad <= 8)),
        "jump_dist_gt32": float(np.mean(ad > 32)),
        "moves_per_output_token": n_mv / max(1, n_ins),
        "actions_per_output_token": (n_mv + n_ins) / max(1, n_ins),
    }
    if full:
        ts = traj_stats.aggregate([(a, PREFIX) for a in streams])
        out["end_append_share"] = ts["end_append_frac"]
        out["traj_stats"] = ts
    else:   # end-append share only (cheap)
        ea = ni = 0
        for a in streams:
            s = traj_stats.analyze_sequence(a, PREFIX)
            ea += s["end_appends"]
            ni += s["n_insert"]
        out["end_append_share"] = ea / max(1, ni)
    return out


def human_streams(ws):
    return [order_to_actions(w["prefix"], w["cont"], w["order"]) for w in ws]


def make_policy(d: dict, rng: random.Random):
    if "p_fill" in d or "fill_alpha" in d:   # optional sixth dial (runjump6.py, a modified COPY of July's policy)
        from runjump6 import RunJump6Policy
        return RunJump6Policy(rng=rng, **d)
    return RunJumpPolicy(d["lambda_cont"], d["jump_alpha"], d["p_skip"], d["skip_q"], rng, p_far=d["p_far"])


def policy_streams(ws, d: dict, seed: int = 0):
    pol = make_policy(d, random.Random(seed))
    out = []
    for w in ws:
        a = pol.generate(w["prefix"], w["cont"], 100000)
        assert a is not None
        out.append(a)
    return out


# --- fit objective -----------------------------------------------------------------------------------------
LOG_KEYS = ["run_len_mean", "run_len_median", "run_len_p90", "jumps_per_100_tokens", "jump_dist_median",
            "jump_dist_mean"]
SHARE_KEYS = ["back_jump_share", "jump_dist_le2", "jump_dist_le8", "end_append_share"]


def loss(st: dict, tg: dict) -> float:
    l = 0.0
    for k in LOG_KEYS:
        l += math.log(max(st[k], 1e-3) / max(tg[k], 1e-3)) ** 2
    for k in SHARE_KEYS:
        l += ((st[k] - tg[k]) / 0.1) ** 2
    l += ((st["actions_per_output_token"] - tg["actions_per_output_token"]) / 0.02) ** 2
    return l


_G: dict = {}


def _init(split, n_windows, tg):
    ws = load_windows(split)
    if n_windows:
        ws = random.Random(0).sample(ws, min(n_windows, len(ws)))
    _G["ws"], _G["tg"] = ws, tg


def _eval(d):
    st = stream_stats(policy_streams(_G["ws"], d, seed=0))
    return {"dials": d, "loss": loss(st, _G["tg"]), "stats": st}


def human_targets(split="train", drop_ai=False, full=True):
    return stream_stats(human_streams(load_windows(split, drop_ai)), full=full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["human", "grid", "refine", "eval"])
    ap.add_argument("--out", default="")
    ap.add_argument("--n_windows", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--grid", default="", help="json dict of dial -> list (overrides the default grid)")
    ap.add_argument("--start", default="", help="json dials for refine")
    ap.add_argument("--dials", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()

    if a.mode == "human":
        res = {}
        for sp in ("train", "dev", "test"):
            for drop in (False, True):
                res[f"{sp}{'_noAI' if drop else ''}"] = human_targets(sp, drop)
        res["july_dials_on_train"] = stream_stats(policy_streams(load_windows("train"), JULY_DIALS), full=True)
        print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                              if kk != "traj_stats"} for k, v in res.items()}, indent=1))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
        return

    tg = human_targets("train", full=False)
    if a.mode == "eval":
        d = json.loads(a.dials)
        ws = load_windows(a.split)
        sts = [stream_stats(policy_streams(ws, d, seed=s), full=True) for s in range(a.seeds)]
        tgs = human_targets(a.split, full=True)
        res = {"dials": d, "split": a.split, "human": tgs, "policy_seed0": sts[0],
               "loss_vs_split_human": [loss(s, tgs) for s in sts]}
        print(json.dumps({"human": {k: v for k, v in tgs.items() if k != "traj_stats"},
                          "policy": {k: v for k, v in sts[0].items() if k != "traj_stats"},
                          "loss": res["loss_vs_split_human"]}, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return

    if a.mode == "grid":
        g = json.loads(a.grid) if a.grid else {
            "lambda_cont": [0.8, 0.9, 0.95, 0.97, 0.98, 0.99],
            "jump_alpha": [0.0, 0.5, 1.0, 1.5, 2.5],
            "p_skip": [0.0, 0.2, 0.5, 0.8],
            "skip_q": [0.0, 0.3, 0.6, 0.85],
            "p_far": [0.0, 0.05, 0.2, 0.9],
        }
        keys = list(g)
        settings = [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]
    else:   # refine: coordinate-wise neighbourhood around --start
        s0 = json.loads(a.start)
        settings = []
        steps = {"lambda_cont": [-0.01, -0.005, 0.005, 0.01], "jump_alpha": [-0.5, -0.25, 0.25, 0.5],
                 "p_skip": [-0.1, -0.05, 0.05, 0.1], "skip_q": [-0.1, -0.05, 0.05, 0.1],
                 "p_far": [-0.05, -0.02, 0.02, 0.05], "p_fill": [-0.05, -0.02, 0.02, 0.05],
                 "fill_alpha": [-0.5, -0.25, 0.25, 0.5]}
        settings.append(dict(s0))
        for k in s0:
            for dv in steps.get(k, []):
                s = dict(s0)
                s[k] = round(s0[k] + dv, 4)
                lo, hi = (0.0, 0.9999) if k != "jump_alpha" and k != "fill_alpha" else (-2.0, 8.0)
                if lo <= s[k] <= hi:
                    settings.append(s)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.split, a.n_windows, tg)) as pool:
        res = pool.map(_eval, settings, chunksize=4)
    res.sort(key=lambda r: r["loss"])
    print(f"{len(settings)} settings in {time.time()-t0:.0f}s; target: "
          + json.dumps({k: round(tg[k], 3) for k in LOG_KEYS + SHARE_KEYS + ['actions_per_output_token']}))
    for r in res[:10]:
        print(round(r["loss"], 3), json.dumps(r["dials"]),
              json.dumps({k: round(r["stats"][k], 3) for k in LOG_KEYS + SHARE_KEYS + ['actions_per_output_token']}))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"target_train": tg, "n_windows": a.n_windows, "results": res}, indent=1))


if __name__ == "__main__":
    main()
