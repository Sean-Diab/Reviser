#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import (
    base_parser,
    iter_records,
    load_json_or_jsonl,
    pick_prompt_text,
    pick_response_text,
    write_json,
)


def _score(row: dict) -> float:
    txt = pick_response_text(row)
    if not txt:
        return -1e9
    toks = txt.split()
    uniq = len(set(toks)) / max(1, len(toks))
    length_bonus = min(1.0, len(toks) / 180.0)
    return 0.7 * uniq + 0.3 * length_bonus


def main() -> None:
    ap = base_parser("Export released qualitative sample metadata.")
    args = ap.parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    payload = load_json_or_jsonl(in_path)
    rows = list(iter_records(payload))
    scored = []
    for i, row in enumerate(rows):
        scored.append(
            {
                "id": row.get("id", i),
                "source": row.get("source", "unknown"),
                "prompt": pick_prompt_text(row),
                "response": pick_response_text(row),
                "score": _score(row),
            }
        )
    scored = [r for r in scored if r["response"]]
    scored.sort(key=lambda x: x["score"], reverse=True)
    keep_ratio = 0.10
    k = max(1, int(math.ceil(len(scored) * keep_ratio)))
    top = scored[:k]
    write_json(
        args.output,
        {
            "status": "ok",
            "task": "export_top_samples",
            "input": str(in_path),
            "seed": args.seed,
            "n_total": len(scored),
            "n_selected": len(top),
            "selection_ratio": keep_ratio,
            "records": top,
        },
    )


if __name__ == "__main__":
    main()
