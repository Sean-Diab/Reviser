#!/usr/bin/env python3
"""Table rows for cursor_skill.py results plus a paired comparison with a reference checkpoint.

For each NAME, reads DIR/NAME.json (+ DIR/NAME.json.rows.jsonl) and prints one markdown row in the column order of
notes/overnight-dagger-rl-2026-10-07.md section 5b, then the holes-filled difference vs the reference on the states
both scored (per-state filled/holes, mean over states; paired bootstrap, 10,000 resamples, seed 0) and the pooled
share as reported by cursor_skill.py.
  python scripts/eval/cursor_skill_table.py --dir runs/eval/cursor_skill --ref july11656 g8kl20r8 g8m2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        o = json.loads(line)
        d[tuple(o["key"])] = o
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/eval/cursor_skill")
    ap.add_argument("--ref", default="july11656")
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    D = Path(a.dir)
    ref = rows(D / f"{a.ref}.json.rows.jsonl")
    for n in a.names:
        s = json.load(open(D / f"{n}.json", encoding="utf-8"))
        h = s["holes"]
        loc = h["insert_location_share"]
        print(f"| {n} | {h['holes_filled_share']:.3f} | {h['all_filled_share']:.3f} | {h['stop_with_open_holes_share']:.3f} | "
              f"{h['cap_share']:.3f} | {h['moves_per_new_token']:.3f} | {h['back_moves_per_new_token']:.3f} | "
              f"{loc['hole']:.2f} / {loc['end']:.2f} / {loc['inside_text_without_hole']:.2f} | {h['new_tokens_per_state']:.1f} | "
              f"{h['gpt2_ppl_final']:.1f} | {s['ckpt_sha256'][:8]} |")
        rp = D / f"{n}.json.rows.jsonl"
        if not rp.exists():
            print(f"  {n}: no rows file")
            continue
        arm = rows(rp)
        keys = sorted(set(ref) & set(arm))
        x = np.array([arm[k]["filled"] for k in keys], float)
        y = np.array([ref[k]["filled"] for k in keys], float)
        hh = np.array([ref[k]["holes"] for k in keys], float)
        diff_pooled = x.sum() / hh.sum() - y.sum() / hh.sum()
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(keys), (10000, len(keys)))
        bs = (x[idx].sum(1) - y[idx].sum(1)) / hh[idx].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dels = sum(arm[k]["n_delete"] for k in keys)
        print(f"  {n} vs {a.ref}: paired states {len(keys)}, pooled holes filled {x.sum()/hh.sum():.3f} vs {y.sum()/hh.sum():.3f}, "
              f"diff {diff_pooled:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'win' if lo > 0 else ('loss' if hi < 0 else 'tie')}; "
              f"deletes {dels}")


if __name__ == "__main__":
    main()
