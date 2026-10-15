#!/usr/bin/env python3
"""MAUVE stability over k-means seeds, from saved features (helper E, notes/paper-retest-2026-10-08.md). CPU only.

score_mauve.py saves each file's BERT-PLL features (OUT_DIR/STEM.mauve_features.npy) and the reference features
(OUT_DIR/ref_features.<sha8>.<precision>.npy). MAUVE quantizes both sets with k-means (num_buckets="auto" = n/10
clusters), and the clustering depends on its seed (mauve default 25). This recomputes MAUVE for every saved feature
file with several seeds and writes mean, std, min and max.

  python scripts/papereval/mauve_seeds.py --dir /root/rv2/E/scores/mauve --ref ref_features.ec08c05a.tf32.npy \
      --seeds 25,1,2,3,4,5,6,7,8,9 --out /root/rv2/E/scores/mauve/mauve_seeds.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mauve
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", default="25,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    qf = np.load(D / a.ref)
    seeds = [int(s) for s in a.seeds.split(",")]
    res = {"ref": a.ref, "seeds": seeds, "mauve_version": getattr(mauve, "__version__", "?"),
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": {}}
    for f in sorted(D.glob("*.mauve_features.npy")):
        pf = np.load(f)
        v = [float(mauve.compute_mauve(p_features=pf, q_features=qf, num_buckets="auto", seed=s,
                                       verbose=False).mauve) for s in seeds]
        stem = f.name[:-len(".mauve_features.npy")]
        res["rows"][stem] = {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                             "min": min(v), "max": max(v)}
        print(f"{stem:45s} mean {np.mean(v):.4f} std {np.std(v, ddof=1):.4f} min {min(v):.4f} max {max(v):.4f}",
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
