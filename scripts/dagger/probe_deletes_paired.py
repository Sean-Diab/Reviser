#!/usr/bin/env python3
"""Paired delete probe: P(DELETE) per state for two checkpoints on the same clean states, with a paired bootstrap CI.

The states are built exactly as probe_deletes.probe builds them (shard 133 of the held-back training texts, n 400,
seed 0; never the eval prompts), so each checkpoint's mean reproduces the p_delete_mean in its probe_deletes.json /
cursor_skill.json. Only the model-independent kinds are paired (clean_july = the "false deletes on clean text" number,
clean_ltr). Output: per-state P(DELETE) for both checkpoints, both means, the mean difference A - B and a 95%
percentile bootstrap CI over states (10,000 resamples, seed 0).

  python scripts/dagger/probe_deletes_paired.py --ckpt_a A.pt --name_a g8m2jd --ckpt_b B.pt --name_b imm_m \
      --out runs/dagger/eval/probe/paired_g8m2jd_vs_imm_m.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagger_data as dd  # noqa: E402
import hunk_oracle as ho  # noqa: E402
import probe_deletes as pdl  # noqa: E402
import reviser_lm as rl  # noqa: E402

KINDS = ("clean_july", "clean_ltr")


def clean_states(shard=133, n=400, seed=0):
    """Same draws, in the same order, as probe_deletes.probe (it also draws the junk tokens, so they are drawn here)."""
    rng = random.Random(seed)
    dd._worker_init(None, seed + 1000)
    kinds = {k: ([], []) for k in KINDS}
    for i in range(n):
        x = dd.july_item((shard, i))
        if x.get("skip"):
            continue
        acts = x["actions"]
        t = rng.randint(dd.PREFIX, len(acts) - 1)
        st = ho.replay(acts[:t])
        kinds["clean_july"][0].append([ho.EOI] + acts[:t])
        kinds["clean_july"][1].append(st)
        m = rng.randint(0, len(x["cont"]) - 1)
        ltr = x["prefix"] + x["cont"][:m]
        kinds["clean_ltr"][0].append([ho.EOI] + ltr)
        kinds["clean_ltr"][1].append(ho.State(list(ltr), len(ltr)))
        rng.randrange(256, 50000)  # junk_random token (keeps the rng stream identical to probe())
    return kinds


def p_delete(ckpt, kinds, device, temperature, top_k):
    amp = rl.autocast_dtype(device)
    model = rl.load_model(ckpt, device).eval()
    out = {}
    with torch.no_grad():
        for k, (seqs, states) in kinds.items():
            out[k] = pdl.dist(model, seqs, states, device, amp, temperature, top_k)[:, ho.DELETE].double().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--name_a", required=True)
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--name_b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=133)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    kinds = clean_states(a.shard, a.n, a.seed)
    pa = p_delete(a.ckpt_a, kinds, device, a.temperature, a.top_k)
    pb = p_delete(a.ckpt_b, kinds, device, a.temperature, a.top_k)
    res = {"a": {"name": a.name_a, "ckpt": a.ckpt_a, "ckpt_sha256": rl.sha256(a.ckpt_a)},
           "b": {"name": a.name_b, "ckpt": a.ckpt_b, "ckpt_sha256": rl.sha256(a.ckpt_b)},
           "args": vars(a), "device": str(device), "kinds": {}}
    rs = np.random.default_rng(a.seed)
    for k in KINDS:
        d = pa[k] - pb[k]
        idx = rs.integers(0, len(d), size=(a.boot, len(d)))
        bm = d[idx].mean(1)
        res["kinds"][k] = {"n": int(len(d)), "mean_a": float(pa[k].mean()), "mean_b": float(pb[k].mean()),
                           "diff_a_minus_b": float(d.mean()),
                           "ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))],
                           "share_states_a_higher": float((d > 0).mean()),
                           "p_a": [round(float(v), 7) for v in pa[k]], "p_b": [round(float(v), 7) for v in pb[k]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k in KINDS:
        r = res["kinds"][k]
        print(f"{k}: {a.name_a} {r['mean_a']:.5f}  {a.name_b} {r['mean_b']:.5f}  diff {r['diff_a_minus_b']:+.5f} "
              f"95% CI [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]  share of states A higher {r['share_states_a_higher']:.3f}")


if __name__ == "__main__":
    main()
