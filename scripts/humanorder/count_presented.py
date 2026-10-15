#!/usr/bin/env python3
"""Count the texts a trainer run presented, by exact replay of RainingBucketedRestTrajDataset's data order.

Uses the July control's verified replay (experiments/july_control/scripts/dataset_audit.py: 2 workers, seed 123,
bucket_size 5, batch 8, grad-accum 4; bit-exact under numpy 2.2.6) up to the first optimizer step whose cumulative
token count reaches --stop_tokens (the trainer's stop_at_tokens_processed rule). Reports steps, tokens, presentations,
distinct texts and shards touched. Run in /opt/venv (numpy 2.2.6).

  python scripts/humanorder/count_presented.py --data_dir data/humanorder/q_textmatched_v1 --stop_tokens 83866148
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "scripts"))
import dataset_audit as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--stop_tokens", type=int, default=83866148)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sd = Path(a.data_dir) / "shards"
    pool = A.shard_ids(sd)
    r = A.replay(sd, pool, stop_tokens=a.stop_tokens, record=True)
    cons = r.pop("consumed")
    n = sum(int(sel.size) for _, sel in cons)
    distinct = len({(s, int(x)) for s, sel in cons for x in sel.tolist()})
    out = {"data_dir": str(a.data_dir), "n_shards": len(pool), "stop_tokens": a.stop_tokens, "steps": r["steps"],
           "tokens_at_stop": r["tokens"], "texts_presented": n, "distinct_texts": distinct,
           "repeat_presentations": n - distinct, "shards_touched": r["shards_touched"],
           "max_epoch_per_worker": r["max_epoch_per_worker"], "mean_tokens_per_text": r["tokens"] / max(1, n)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
