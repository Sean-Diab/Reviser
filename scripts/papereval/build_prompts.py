#!/usr/bin/env python3
"""Build the paper's 3000-prompt C4 evaluation set (seeds 123/124/125 x 1000) from the historical pool.

Source (read-only): workspaces/4-21/c4_v2_arena_test/c4_segments_v2.json (5000 C4 en validation segments, GPT-2
use_fast=False ids, first 180 tokens; total length 144-216 filter). The paper's run_v2_pipeline.py takes, per seed, the
first 1000 valid segments of the slice starting at {123: 0, 124: 1400, 125: 2800}. The exact 3000 ids it used are
those of workspaces/4-21/all_rollouts_3k/*.json (all model files share them); we take that id list instead of
re-deriving "valid", and assign each id its seed by slice.

Output: data/papereval/c4_paper_3k.json, a list of {id, seed, token_ids (<=180), full_len}; prompt = token_ids[:35],
reference continuation = token_ids[35:]. Also writes the sha256 next to it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HIST = Path("<archive>/workspaces/4-21")
ROOT = Path(__file__).resolve().parents[2]


def main():
    segs = json.loads((HIST / "c4_v2_arena_test" / "c4_segments_v2.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in segs}
    ids = [r["id"] for r in json.loads((HIST / "all_rollouts_3k" / "reviser_100m.json").read_text(encoding="utf-8"))]
    ar_ids = [r["id"] for r in json.loads((HIST / "all_rollouts_3k" / "fineweb_ar_100m.json").read_text(encoding="utf-8"))]
    assert sorted(ids) == sorted(ar_ids) and len(set(ids)) == 3000
    out = []
    for i in sorted(ids):
        k = int(i.rsplit("_", 1)[1])
        seed = 123 if k < 1400 else (124 if k < 2800 else 125)
        s = by_id[i]
        out.append({"id": i, "seed": seed, "token_ids": [int(x) for x in s["token_ids"]], "full_len": s.get("full_len")})
    cnt = {sd: sum(o["seed"] == sd for o in out) for sd in (123, 124, 125)}
    assert cnt == {123: 1000, 124: 1000, 125: 1000}, cnt
    assert all(len(o["token_ids"]) >= 144 for o in out)
    dst = ROOT / "data" / "papereval" / "c4_paper_3k.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out), encoding="utf-8")
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    (dst.parent / "c4_paper_3k.json.sha256").write_text(f"{h}  c4_paper_3k.json\n")
    print(json.dumps({"out": str(dst), "n": len(out), "per_seed": cnt, "sha256": h,
                      "mean_len": sum(len(o["token_ids"]) for o in out) / len(out)}))


if __name__ == "__main__":
    main()
