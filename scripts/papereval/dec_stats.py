#!/usr/bin/env python3
"""Per-decode statistics for helper E's table (stdlib only): prompt-intact rate, mean continuation length (GPT-2
tokens on the final canvas after the 35-token prompt), EOR-stop and budget-hit rates, action mix per sample.

  python scripts/papereval/dec_stats.py --out_dir /root/rv2/E/scores/decstats /root/rv2/E/dec/*.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def stats(path):
    meta, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
            else:
                rows.append(r)
    n = len(rows)
    c = {k: sum(r["counts"][k] for r in rows) / n for k in ("insert", "delete", "move")}
    by_seed = {}
    for r in rows:
        by_seed.setdefault(str(r.get("seed")), []).append(r["out_len"])
    return {"file": str(path), "name": meta.get("name") if meta else None,
            "lock": bool(meta and meta.get("args", {}).get("lock_prompt")), "n": n,
            "ckpt_sha256": meta.get("ckpt_sha256") if meta else None,
            "prefix_intact_rate": sum(r["prefix_intact"] for r in rows) / n,
            "mean_out_len": sum(r["out_len"] for r in rows) / n,
            "mean_out_len_by_seed": {k: sum(v) / len(v) for k, v in sorted(by_seed.items())},
            "n_empty_out": sum(r["out_len"] <= 0 for r in rows),
            "stopped_eor_rate": sum(r["stopped_eor"] for r in rows) / n,
            "hit_budget_rate": sum(r["hit_budget"] for r in rows) / n,
            "mean_new_actions": sum(r["n_new_actions"] for r in rows) / n,
            "mean_actions_per_sample": c, "decode_meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for p in a.inputs:
        if not Path(p + ".done").exists():
            continue
        out = od / (Path(p).stem + ".decstats.json")
        if out.exists() and out.stat().st_mtime >= Path(p).stat().st_mtime:
            continue
        out.write_text(json.dumps(stats(p), indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
