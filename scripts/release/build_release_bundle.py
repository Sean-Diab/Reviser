#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import base_parser, repo_root


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = base_parser("Build a compact release manifest with checksums.")
    args = ap.parse_args()
    root = repo_root()
    tracked = [
        root / "paper" / "main.tex",
        root / "paper" / "references.bib",
        root / "results" / "arena" / "ar_benchmark_v3_summary.json",
        root / "results" / "arena" / "reviser_vs_diffusion_multijudge_summary.json",
        root / "results" / "evalppl" / "gpt2base_evalppl_dream3k_summary.json",
        root / "results" / "mauve" / "bert_pseudologlik_mauve_3000.json",
    ]
    payload = {
        "release": "v0.1.0",
        "files": [
            {"path": str(p.relative_to(root)), "sha256": sha256(p)}
            for p in tracked
            if p.exists()
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
