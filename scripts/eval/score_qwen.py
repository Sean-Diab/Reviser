#!/usr/bin/env python3
"""Non-GPT-2 scorer for decode files: Qwen2.5-3B-Instruct NLL of each continuation given its prompt.

Same definition as the group-RL reward (scripts/grpo/train_grpo.py RewardLM): GPT-2-decode the prompt
(prefix_ids) and the continuation (canvas_ids[len(prefix_ids):]), re-tokenize both with Qwen, plain text, no chat
template, and take the mean per-token NLL of the continuation tokens. Empty continuations are skipped (counted).
Caveat: arms trained with this model as their RL reward are scored by their own reward model here.

Score (on a GPU box; writes OUT_DIR/<decode file stem>.qwen.jsonl rows {"i", "nll", "n_tok"} + a summary line):
  python scripts/eval/score_qwen.py score --qwen /root/rv2/data/models/Qwen2.5-3B-Instruct --out_dir OUT DEC.jsonl ...
Compare two scored files (paired over prompts both scored; 10,000 bootstrap resamples, seed 0):
  python scripts/eval/score_qwen.py compare REF.qwen.jsonl ARM.qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(a):
    import torch
    sys.path.insert(0, str(HERE.parents[0] / "grpo"))
    sys.path.insert(0, str(HERE.parents[0] / "dagger"))
    sys.path.insert(0, str(HERE.parents[0] / "arrl"))
    from train_grpo import RewardLM
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu"
    rlm = RewardLM(a.qwen, dev, torch.bfloat16 if dev == "cuda" else torch.float32)
    g2 = AutoTokenizer.from_pretrained(a.gpt2_tok)
    rev = Path(a.qwen) / "REVISION.txt"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in a.files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if "_meta" not in r]
        pre = [g2.decode(r["prefix_ids"]) for r in rows]
        con = [g2.decode(r["canvas_ids"][len(r["prefix_ids"]):]) for r in rows]
        nll = rlm.nll(pre, con, bs=a.bs)
        out = out_dir / (Path(path).name.replace(".jsonl", "") + ".qwen.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r, c, v in zip(rows, con, nll):
                f.write(json.dumps({"i": r["i"], "nll": v, "n_tok": len(rlm.tok.encode(c)) if v is not None else 0}) + "\n")
            vs = [v for v in nll if v is not None]
            summ = {"_summary": {"file": Path(path).name, "n": len(rows), "n_scored": len(vs),
                                 "mean_nll": sum(vs) / max(1, len(vs)), "ppl_of_mean_nll": math.exp(sum(vs) / max(1, len(vs))),
                                 "qwen": a.qwen, "qwen_revision": rev.read_text().strip() if rev.exists() else None}}
            f.write(json.dumps(summ) + "\n")
        print(json.dumps(summ))


def load(p):
    d, s = {}, None
    for l in open(p, encoding="utf-8"):
        o = json.loads(l)
        if "_summary" in o:
            s = o["_summary"]
        elif o["nll"] is not None:
            d[o["i"]] = o["nll"]
    return d, s


def compare(a):
    import numpy as np
    ref, _ = load(a.ref)
    arm, _ = load(a.arm)
    ids = sorted(set(ref) & set(arm))
    diff = np.array([arm[i] - ref[i] for i in ids])
    rng = np.random.default_rng(0)
    bs = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res = {"ref": Path(a.ref).name, "arm": Path(a.arm).name, "n_paired": len(ids),
           "ref_mean_nll": float(np.mean([ref[i] for i in ids])), "arm_mean_nll": float(np.mean([arm[i] for i in ids])),
           "diff_arm_minus_ref": float(diff.mean()), "ci95": [float(lo), float(hi)],
           "result": "win" if hi < 0 else ("loss" if lo > 0 else "tie")}
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score")
    s.add_argument("--qwen", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--gpt2_tok", default="gpt2")
    s.add_argument("--bs", type=int, default=8)
    s.add_argument("--device", default="cuda")
    s.add_argument("files", nargs="+")
    c = sp.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("arm")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else compare(a)


if __name__ == "__main__":
    main()
