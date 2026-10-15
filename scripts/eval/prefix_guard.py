#!/usr/bin/env python3
"""Prompt-intact guard (main session rule, 2026-10-07 ~10:03Z): share of outputs whose final text no longer starts with
the unchanged prompt (decoder field `prefix_intact`), plus how much of the prompt was actually removed (greedy test: is
the prompt a subsequence of the final canvas; prompt tokens not matched = removed). An arm fails the guard when the
changed share is above 2x July@84M's on either set (C4 > 0.020 or DCLM > 0.044).
  python scripts/eval/prefix_guard.py runs/dagger/eval/dec/*.jsonl > runs/dagger/eval/prefix_guard/all.json
"""
import json
import sys
from pathlib import Path

LIMIT = {"c4": 0.020, "dclm": 0.044}


def matched(prompt, canvas):
    j = n = 0
    for t in prompt:
        while j < len(canvas) and canvas[j] != t:
            j += 1
        if j < len(canvas):
            n += 1
            j += 1
    return n


def main():
    out = {}
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if "_meta" not in r and "prefix_intact" in r]
        if not rows:
            continue
        ch = sum(not r["prefix_intact"] for r in rows) / len(rows)
        lost = [1 - matched(r["prefix_ids"], r["canvas_ids"]) / max(1, len(r["prefix_ids"])) for r in rows]
        ds = "c4" if Path(path).name.startswith("c4.") else "dclm"
        out[Path(path).name] = {"n": len(rows), "prompt_changed_share": ch,
                                "prompt_tokens_removed_share": sum(lost) / len(lost),
                                "outputs_losing_half_prompt": sum(x >= 0.5 for x in lost) / len(lost),
                                "limit": LIMIT[ds], "pass": ch <= LIMIT[ds]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
