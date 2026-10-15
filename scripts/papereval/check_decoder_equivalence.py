#!/usr/bin/env python3
"""Check scripts/papereval/decode_batched.py against rows decoded by the serial July decoder
(experiments/july_control/eval/decode_eval.py --protocol paper, e.g. runs/cont_base/dec/c4.paper.s20260924.rev100.jsonl).

Each serial row's actions are replayed through decode_batch(forced=...) in one batch. For every step we record whether
the serial action is allowed by the batched decoder's legality mask and inside its top-k set (log-prob finite), and
we compare the final canvas from the replay with the serial canvas. A correct batched decoder gives 0 illegal actions,
identical canvases, and (in the same dtype) only rare top-k boundary flips.

  python scripts/papereval/check_decoder_equivalence.py --ckpt runs/july_control/ckpt/reviser_100m_wrapped.pt \
      --serial runs/cont_base/dec/c4.paper.s20260924.rev100.jsonl --n 16 --device cpu --out runs/papereval/equiv.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decode_batched as DB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import reviser_lm as rl
    device = torch.device(a.device)
    model = rl.load_model(a.ckpt, device).eval()
    rows = [json.loads(l) for l in open(a.serial, encoding="utf-8")]
    meta = rows[0].get("_meta")
    rows = [r for r in rows if "_meta" not in r][:a.n]
    prefixes = [r["prefix_ids"] for r in rows]
    forced = [r["actions_new"] for r in rows]
    states, lps = DB.decode_batch(model, prefixes, device=device, max_seq_len=a.max_seq_len, temperature=0.9,
                                  top_k=50, lock_prompt=False, gen=None, forced=forced)
    per = []
    for r, st, lp in zip(rows, states, lps):
        n_inf = sum(1 for x in lp if not math.isfinite(x))
        per.append({"id": r["id"], "n_actions": len(r["actions_new"]), "n_steps_scored": len(lp),
                    "n_outside_topk_or_illegal": n_inf,
                    "mean_logprob_finite": (sum(x for x in lp if math.isfinite(x)) / max(1, len(lp) - n_inf)),
                    "canvas_equal": [int(x) for x in st.canvas] == r["canvas_ids"]})
    summ = {"serial_file": a.serial, "serial_meta_name": (meta or {}).get("name"), "ckpt": a.ckpt,
            "device": str(device), "n_rows": len(per),
            "rows_canvas_equal": sum(p["canvas_equal"] for p in per),
            "total_steps": sum(p["n_steps_scored"] for p in per),
            "total_outside_topk_or_illegal": sum(p["n_outside_topk_or_illegal"] for p in per),
            "rows": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}))


if __name__ == "__main__":
    main()
