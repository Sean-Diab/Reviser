#!/usr/bin/env python3
from __future__ import annotations

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
    pick_prompt_text,
    write_json,
)


def main() -> None:
    ap = base_parser("Public baseline rollout wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    dec_cfg = cfg.get("decoding", {})
    family = str(model_cfg.get("family", "baseline")).lower()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if family != "ar":
        write_json(
            args.output,
            {
                "status": "unsupported_family",
                "family": family,
                "notes": [
                    "This script performs direct generation for AR baselines.",
                    "For diffusion baselines, provide pre-generated outputs and evaluate via scripts/eval/*.",
                ],
            },
        )
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = str(model_cfg.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("Missing model.checkpoint in config for AR baseline rollout.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.eval()

    payload = load_json_or_jsonl(in_path)
    rows = list(iter_records(payload))
    max_samples = int(cfg.get("inference", {}).get("max_samples", 0) or 0)
    if max_samples > 0:
        rows = rows[:max_samples]

    outputs = []
    gen_len = int(dec_cfg.get("max_new_tokens", 145))
    top_k = int(dec_cfg.get("top_k", 50))
    temperature = float(dec_cfg.get("temperature", 0.9))
    for i, row in enumerate(rows):
        prompt = pick_prompt_text(row)
        if not prompt:
            continue
        x = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            y = model.generate(
                **x,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=gen_len,
                pad_token_id=tok.pad_token_id,
            )
        full = tok.decode(y[0], skip_special_tokens=True)
        continuation = full[len(prompt):] if full.startswith(prompt) else full
        outputs.append(
            {
                "id": row.get("id", i),
                "prompt": prompt,
                "response": continuation,
                "full_text": full,
            }
        )

    write_json(
        args.output,
        {
            "status": "ok",
            "task": "baseline_rollout",
            "model": model_cfg.get("name", "baseline"),
            "family": family,
            "seed": args.seed,
            "device": str(device),
            "n_examples": len(outputs),
            "records": outputs,
        },
    )


if __name__ == "__main__":
    main()
