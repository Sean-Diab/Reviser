#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import base_parser, load_yaml, write_json


def main() -> None:
    ap = base_parser("Public AR baseline training entrypoint (intentionally unsupported in this release).")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    write_json(
        args.output,
        {
            "status": "not_implemented_in_public_release",
            "reason": "AR baseline training is intentionally unsupported in this public release.",
            "entrypoint": "scripts/train/train_ar_baseline.py",
            "config_name": cfg.get("model", {}).get("name", "unknown"),
            "seed": args.seed,
            "device": args.device,
        },
    )


if __name__ == "__main__":
    main()
