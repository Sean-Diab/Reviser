#!/usr/bin/env python3
"""Which shards an 84M-token run reads, and whether any is a left-to-right filler shard of a build made with
build_variant.py --real_shards (the variant is built only on the shards a run reads; the rest are filler).

Same exact replay as scripts/humanorder/count_presented.py (experiments/july_control/scripts/dataset_audit.py: 2
workers, seed 123, bucket 5, batch 8, accumulation 4, stop at the first optimizer step whose cumulative tokens reach
--stop_tokens), but it lists the shard ids touched (count_presented.py reports only their number) and compares them
with state.json's filler_shards. Exit code 1 if a filler shard is touched.

  python scripts/humanorder/ov_presented_check.py --data_dir /root/rv2/hq/data/ov_j0 --out out/j0/presented_check.json
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
    first = []
    for s, _ in r["consumed"]:
        if int(s) not in first:
            first.append(int(s))
    st = json.loads((Path(a.data_dir) / "state.json").read_text())
    filler = set(st.get("filler_shards") or [])
    bad = sorted(set(first) & filler)
    out = {"data_dir": str(a.data_dir), "steps": r["steps"], "tokens_at_stop": r["tokens"],
           "shards_touched_in_first_use_order": first, "n_shards_touched": len(first),
           "real_shards_rule": st.get("real_shards_rule"), "n_filler_shards": len(filler),
           "filler_shards_touched": bad, "ok": not bad}
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
