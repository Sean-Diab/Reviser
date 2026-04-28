#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import base_parser, load_yaml, write_json


def _f_transformer_full(n: int, d: int, dff: int, layers: int) -> float:
    # Dominant matmul proxy (same style used in paper appendix).
    per_layer = 12 * n * d * d + 2 * n * n * d + 4 * n * d * dff
    return float(layers * per_layer)


def _f_vocab(n: int, d: int, vocab: int) -> float:
    return float(n * d * vocab)


def _compute_from_cfg(cfg: dict) -> dict:
    m = cfg.get("model", {})
    d = int(m.get("d_model", 512))
    dff = int(m.get("d_ff", 4 * d))
    L = int(m.get("n_layers", 24))
    V = int(m.get("vocab_size", 50291))
    Tdec = int(cfg.get("decoding", {}).get("total_max_actions", 128))
    n = int(cfg.get("evaluation", {}).get("flops_sequence_len", 128))
    f_step = _f_transformer_full(n=n, d=d, dff=dff, layers=L) + _f_vocab(n=n, d=d, vocab=V)
    f_total = f_step * Tdec
    return {
        "status": "computed_from_config",
        "model": m.get("name", "unknown"),
        "assumptions": {
            "n": n,
            "T_dec": Tdec,
            "vocab": V,
            "formula": "F_total = T_dec * (F_transformer_full + F_vocab)",
        },
        "flops": {
            "per_step": round(f_step, 4),
            "total": round(f_total, 4),
            "total_G": round(f_total / 1e9, 4),
        },
    }


def main() -> None:
    ap = base_parser("Public FLOPs accounting wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    in_path = Path(args.input)
    if in_path.exists() and in_path.suffix == ".json":
        obj = json.loads(in_path.read_text(encoding="utf-8"))
        out = {
            "status": "normalized_summary",
            "task": "flops_accounting",
            "config_name": cfg.get("model", {}).get("name", "unknown"),
            "payload": obj,
        }
    else:
        out = _compute_from_cfg(cfg)
        out["task"] = "flops_accounting"
        out["config_name"] = cfg.get("model", {}).get("name", "unknown")
        out["seed"] = args.seed
        out["device"] = args.device
    write_json(args.output, out)


if __name__ == "__main__":
    main()
