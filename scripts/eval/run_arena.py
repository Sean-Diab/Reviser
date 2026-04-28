#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import (
    base_parser,
    iter_records,
    load_json_or_jsonl,
    load_yaml,
    write_json,
)


def _strip_elo(payload: dict) -> dict:
    if isinstance(payload, dict):
        return {k: _strip_elo(v) for k, v in payload.items() if k != "elo"}
    if isinstance(payload, list):
        return [_strip_elo(x) for x in payload]
    return payload


def _avg_nll(model, tok, device, prompt: str, resp: str) -> float:
    full = (prompt or "") + (resp or "")
    if not full:
        return float("inf")
    enc_full = tok(full, return_tensors="pt")
    enc_prompt = tok(prompt or "", return_tensors="pt")
    input_ids = enc_full["input_ids"].to(device)
    labels = input_ids.clone()
    prefix_len = int(enc_prompt["input_ids"].shape[1]) if prompt else 0
    labels[:, :prefix_len] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    return float(out.loss.item())


def _compute_from_pairs(cfg: dict, in_path: Path, seed: int, device_name: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(iter_records(load_json_or_jsonl(in_path)))
    judge_model_id = str(cfg.get("evaluation", {}).get("judge_model_id", "openai-community/gpt2"))
    dev = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(judge_model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(judge_model_id).to(dev)
    model.eval()

    rng = random.Random(seed)
    wins_a = 0
    wins_b = 0
    valid = 0
    model_a = cfg.get("evaluation", {}).get("model_a_name", "model_a")
    model_b = cfg.get("evaluation", {}).get("model_b_name", "model_b")
    for row in rows:
        prompt = str(row.get("prompt", ""))
        ra = str(row.get("response_a", ""))
        rb = str(row.get("response_b", ""))
        if not (ra and rb):
            continue
        # Randomized side assignment (paper protocol).
        if rng.random() < 0.5:
            side_a, side_b = ra, rb
            map_side = ("a", "b")
        else:
            side_a, side_b = rb, ra
            map_side = ("b", "a")
        nll_a = _avg_nll(model, tok, dev, prompt, side_a)
        nll_b = _avg_nll(model, tok, dev, prompt, side_b)
        winner_side = map_side[0] if nll_a < nll_b else map_side[1]
        if winner_side == "a":
            wins_a += 1
        else:
            wins_b += 1
        valid += 1

    wa = wins_a / max(1, valid)
    wb = wins_b / max(1, valid)
    return {
        "status": "computed_from_pairs",
        "judge": judge_model_id,
        "n_examples": len(rows),
        "n_valid": valid,
        "pairwise_wins": {f"{model_a}>{model_b}": wins_a, f"{model_b}>{model_a}": wins_b},
        "win_rate_1st": {model_a: round(wa, 4), model_b: round(wb, 4)},
        "confidence_interval_95": {
            model_a: [
                round(max(0.0, wa - 1.96 * math.sqrt(max(1e-12, wa * (1 - wa) / max(1, valid)))), 4),
                round(min(1.0, wa + 1.96 * math.sqrt(max(1e-12, wa * (1 - wa) / max(1, valid)))), 4),
            ],
            model_b: [
                round(max(0.0, wb - 1.96 * math.sqrt(max(1e-12, wb * (1 - wb) / max(1, valid)))), 4),
                round(min(1.0, wb + 1.96 * math.sqrt(max(1e-12, wb * (1 - wb) / max(1, valid)))), 4),
            ],
        },
    }


def main() -> None:
    ap = base_parser("Public arena evaluation wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = {
        "task": "arena",
        "config_name": cfg.get("model", {}).get("name", "unknown"),
        "input": args.input,
        "seed": args.seed,
        "device": args.device,
    }
    in_path = Path(args.input)
    if in_path.exists() and in_path.suffix == ".json":
        obj = json.loads(in_path.read_text(encoding="utf-8"))
        if "comparisons" in obj:
            out["status"] = "normalized_summary"
            out["n_comparisons"] = len(obj["comparisons"])
            out["comparisons"] = {k: _strip_elo(v) for k, v in obj["comparisons"].items()}
        else:
            out["status"] = "passthrough_summary"
            out["payload"] = _strip_elo(obj)
    elif in_path.exists() and in_path.suffix == ".jsonl":
        out.update(_compute_from_pairs(cfg, in_path, args.seed, args.device))
    else:
        raise FileNotFoundError(f"Input file not found: {in_path}")
    write_json(args.output, out)


if __name__ == "__main__":
    main()
