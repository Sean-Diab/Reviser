#!/usr/bin/env python3
"""First-epoch shard visit order of RainingBucketedRestTrajDataset for the 134-shard text-matched layout.

Per worker w (2 workers, seed 123): ids = sorted(shards)[w::2]; rng = np.random.default_rng(123 + 1543 w);
order = list(ids); rng.shuffle(order) — the first random draw of that worker (dataset_audit.worker_batches; the order
does not depend on the trajectories). An 84M-token run reads only the first ~10-13 shards of each worker, so data that
is expensive to make (GPU-generated orders, GPT-2 candidates) is generated for the first N visited shards per worker
only; the rest get a cheap fallback, and count_presented.py confirms afterwards that no fallback shard was read.

  python scripts/humanorder/visit_order.py --per_worker 16          # prints the shard list (comma-separated)
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def visit_order(n_shards: int = 134, workers: int = 2, seed: int = 123) -> list[list[int]]:
    pool = list(range(n_shards))
    out = []
    for w in range(workers):
        rng = np.random.default_rng(seed + 1543 * w)
        order = list(pool[w::workers])
        rng.shuffle(order)
        out.append([int(x) for x in order])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=134)
    ap.add_argument("--per_worker", type=int, default=16)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    vo = visit_order(a.n_shards)
    first = sorted({k for w in vo for k in w[:a.per_worker]})
    if a.json:
        print(json.dumps({"per_worker_order": vo, "first": first}))
    else:
        print(",".join(str(k) for k in first))


if __name__ == "__main__":
    main()
