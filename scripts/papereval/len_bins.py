#!/usr/bin/env python3
"""Skywork win rate vs AR 100M split by the model's own continuation length (helper E, notes/paper-retest-2026-10-08.md).

Reads arena per-example records (scores/arena/X__vs__BASE.MODE.json) and the model's decode (dec/X.jsonl: out_len =
continuation tokens, stopped_eor) and writes, per model and mode, the win rate in length bins and by stop reason.
CPU only.

  python scripts/papereval/len_bins.py --root /root/rv2/E --out /root/rv2/E/scores/arena/len_bins.json \
      c4paper3k.nolock.rev100 c4paper3k.lock.rev100 c4paper3k.lock.g8m2jd_e8 c4paper3k.lock.g8m2jd
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = "c4paper3k.nolock.ar100"
BINS = [(0, 40), (40, 80), (80, 120), (120, 180), (180, 221)]


def lens(dec):
    d = {}
    for line in open(dec, encoding="utf-8"):
        r = json.loads(line)
        if "_meta" in r:
            continue
        d[r["id"]] = (int(r["out_len"]), bool(r["stopped_eor"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--root", default="/root/rv2/E")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.root)
    res = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "bins": BINS, "rows": []}
    for x in a.models:
        L = lens(R / "dec" / f"{x}.jsonl")
        for mode in ("paper", "paper_both", "matched_both"):
            p = R / "scores" / "arena" / f"{x}__vs__{BASE}.{mode}.json"
            if not p.exists():
                continue
            pe = json.loads(p.read_text(encoding="utf-8"))["per_example"]
            sc = {k: v["score"] for k, v in pe.items() if v["score"] is not None}
            row = {"model": x, "mode": mode, "arena_file": p.name, "bins": []}
            for lo, hi in BINS:
                s = [v for k, v in sc.items() if lo <= L[k][0] < hi]
                row["bins"].append({"lo": lo, "hi": hi, "n": len(s), "win": float(np.mean(s)) if s else None})
            for name, flag in (("eor", True), ("budget", False)):
                s = [v for k, v in sc.items() if L[k][1] == flag]
                row[name] = {"n": len(s), "win": float(np.mean(s)) if s else None}
            res["rows"].append(row)
            print(x, mode, " | ".join(f"[{b['lo']},{b['hi']}) n={b['n']} {100 * b['win']:.1f}"
                                      for b in row["bins"] if b["n"]))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
