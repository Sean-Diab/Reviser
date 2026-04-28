#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import base_parser, iter_records, load_json_or_jsonl, load_yaml, write_json


def _tokenize_simple(text: str) -> list[str]:
    return [t for t in text.lower().replace("\n", " ").split(" ") if t]


def _jsd_from_counters(a: Counter, b: Counter) -> float:
    vocab = set(a.keys()) | set(b.keys())
    ta = float(sum(a.values()) or 1.0)
    tb = float(sum(b.values()) or 1.0)
    jsd = 0.0
    for t in vocab:
        pa = a.get(t, 0.0) / ta
        pb = b.get(t, 0.0) / tb
        m = 0.5 * (pa + pb)
        if pa > 0:
            jsd += 0.5 * pa * math.log(pa / m + 1e-12)
        if pb > 0:
            jsd += 0.5 * pb * math.log(pb / m + 1e-12)
    return float(jsd)


def _compute_mauve_proxy(in_path: Path) -> dict:
    rows = list(iter_records(load_json_or_jsonl(in_path)))
    by_source: dict[str, Counter] = {}
    by_n: dict[str, int] = {}
    for row in rows:
        src = str(row.get("source", "unknown"))
        txt = str(row.get("response", row.get("text", "")))
        if not txt:
            continue
        by_n[src] = by_n.get(src, 0) + 1
        c = by_source.setdefault(src, Counter())
        c.update(_tokenize_simple(txt))
    sources = sorted(by_source.keys())
    if len(sources) < 2:
        return {
            "status": "insufficient_sources",
            "notes": ["Need at least two sources in input for MAUVE-style comparison."],
            "sources": sources,
        }
    ref = sources[0]
    out_rows = {}
    for src in sources:
        jsd = _jsd_from_counters(by_source[ref], by_source[src]) if src != ref else 0.0
        # Proxy score in [0,1], 1 means identical unigram distributions.
        mauve_proxy = math.exp(-jsd)
        out_rows[src] = {"mauve": round(float(mauve_proxy), 6), "n": int(by_n.get(src, 0))}
    return {
        "status": "computed_proxy",
        "protocol": "Unigram-JSD MAUVE proxy (tokenized whitespace/lowercase)",
        "reference_source": ref,
        "rows": out_rows,
    }


def main() -> None:
    ap = base_parser("Public MAUVE wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = {
        "task": "mauve",
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
            out["protocol"] = obj.get("protocol")
            out["settings"] = obj.get("settings", {})
            out["models"] = obj.get("models", [])
            out["results"] = obj.get("results", {})
        else:
            out.update(_compute_mauve_proxy(in_path))
    elif in_path.exists() and in_path.suffix == ".jsonl":
        out.update(_compute_mauve_proxy(in_path))
    else:
        raise FileNotFoundError(f"Input file not found: {in_path}")
    write_json(args.output, out)


if __name__ == "__main__":
    main()
