#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import (
    base_parser,
    iter_records,
    load_json_or_jsonl,
    load_yaml,
    pick_prompt_text,
    pick_response_text,
    write_json,
)


def _normalize_summary(obj: dict) -> dict:
    return {
        "status": "normalized_summary",
        "ranked": obj.get("ranked", []),
        "by_source": obj.get("by_source", {}),
        "eval_model": obj.get("eval_model"),
        "scorer": obj.get("scorer"),
    }


def _compute_evalppl_from_rollouts(cfg: dict, in_path: Path, seed: int, device: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_cfg = cfg.get("evaluation", {})
    scorer_model = str(eval_cfg.get("scorer_model_id", "openai-community/gpt2"))
    payload = load_json_or_jsonl(in_path)
    rows = list(iter_records(payload))
    max_examples = int(eval_cfg.get("max_examples", 0) or 0)
    if max_examples > 0:
        rows = rows[:max_examples]

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(scorer_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(scorer_model).to(dev)
    model.eval()

    groups: dict[str, list[float]] = {}
    with torch.no_grad():
        for row in rows:
            prompt = pick_prompt_text(row)
            resp = pick_response_text(row)
            if not resp:
                continue
            source = str(row.get("source", cfg.get("model", {}).get("name", "unknown")))
            full = prompt + resp
            enc_full = tok(full, return_tensors="pt")
            enc_prompt = tok(prompt, return_tensors="pt")
            input_ids = enc_full["input_ids"].to(dev)
            labels = input_ids.clone()
            prompt_len = int(enc_prompt["input_ids"].shape[1]) if prompt else 0
            labels[:, :prompt_len] = -100
            out = model(input_ids=input_ids, labels=labels)
            nll = float(out.loss.item())
            ppl = math.exp(nll)
            groups.setdefault(source, []).append(ppl)

    by_source = {}
    ranked = []
    for src, vals in groups.items():
        mean_ppl = float(sum(vals) / max(1, len(vals)))
        by_source[src] = {"mean_ppl": round(mean_ppl, 4), "n_examples": len(vals)}
        ranked.append({"source": src, "mean_ppl": mean_ppl, "n_examples": len(vals)})
    ranked.sort(key=lambda x: x["mean_ppl"])
    return {
        "status": "computed_from_rollouts",
        "seed": seed,
        "eval_model": "gpt2_base",
        "scorer": "continuation_only",
        "scorer_model_id": scorer_model,
        "ranked": ranked,
        "by_source": by_source,
    }


def main() -> None:
    ap = base_parser("Public EvalPPL wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    in_path = Path(args.input)
    out: dict = {
        "task": "evalppl",
        "config_name": cfg.get("model", {}).get("name", "unknown"),
        "input": args.input,
        "seed": args.seed,
        "device": args.device,
    }
    if in_path.exists() and in_path.suffix == ".json":
        obj = json.loads(in_path.read_text(encoding="utf-8"))
        if "ranked" in obj or "by_source" in obj:
            out.update(_normalize_summary(obj))
        else:
            out.update(_compute_evalppl_from_rollouts(cfg, in_path, args.seed, args.device))
    elif in_path.exists() and in_path.suffix == ".jsonl":
        out.update(_compute_evalppl_from_rollouts(cfg, in_path, args.seed, args.device))
    else:
        raise FileNotFoundError(f"Input file not found: {in_path}")
    write_json(args.output, out)


if __name__ == "__main__":
    main()
