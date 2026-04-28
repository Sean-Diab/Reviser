#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import base_parser, iter_records, load_json_or_jsonl, load_yaml, write_json


def _compute_from_rollouts(in_path: Path) -> dict:
    rows = list(iter_records(load_json_or_jsonl(in_path)))
    n = 0
    total_actions = 0
    total_moves = 0
    total_inserts = 0
    total_deletes = 0
    move_abs_sum = 0
    for row in rows:
        hist = row.get("action_history")
        if not isinstance(hist, list):
            continue
        n += 1
        for a in hist:
            if isinstance(a, str):
                if a.startswith("MOVE"):
                    total_moves += 1
                elif a.startswith("INSERT"):
                    total_inserts += 1
                elif a.startswith("DELETE"):
                    total_deletes += 1
                total_actions += 1
                continue
            try:
                ai = int(a)
            except Exception:
                continue
            total_actions += 1
            # Reviser action vocabulary convention in this repo:
            # move tokens occupy ids in [50261, 50280] and delete is 50259.
            if ai == 50259:
                total_deletes += 1
            elif 50261 <= ai <= 50280:
                total_moves += 1
                # map to small move magnitudes by index (approx for diagnostics only)
                idx = ai - 50261
                vals = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512]
                if 0 <= idx < len(vals):
                    move_abs_sum += abs(vals[idx])
            elif ai < 50257:
                total_inserts += 1
    return {
        "status": "computed_from_rollouts",
        "n_examples": n,
        "mean_total_actions": round(total_actions / max(1, n), 4),
        "fractions": {
            "move": round(total_moves / max(1, total_actions), 4),
            "insert": round(total_inserts / max(1, total_actions), 4),
            "delete": round(total_deletes / max(1, total_actions), 4),
        },
        "mean_abs_move_distance": round(move_abs_sum / max(1, total_moves), 4),
    }


def main() -> None:
    ap = base_parser("Public trajectory statistics wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = {
        "task": "trajectory_stats",
        "config_name": cfg.get("model", {}).get("name", "unknown"),
        "input": args.input,
        "seed": args.seed,
        "device": args.device,
    }
    in_path = Path(args.input)
    if in_path.exists() and in_path.suffix == ".json":
        obj = json.loads(in_path.read_text(encoding="utf-8"))
        if "results" in obj:
            out["status"] = "normalized_summary"
            out["datasets"] = obj.get("datasets", [])
            out["results"] = obj.get("results", {})
            out["config"] = obj.get("config", {})
        else:
            out.update(_compute_from_rollouts(in_path))
    elif in_path.exists() and in_path.suffix == ".jsonl":
        out.update(_compute_from_rollouts(in_path))
    else:
        raise FileNotFoundError(f"Input file not found: {in_path}")
    write_json(args.output, out)


if __name__ == "__main__":
    main()
