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
    ap = base_parser("Public Reviser rollout wrapper.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    dec_cfg = cfg.get("decoding", {})
    checkpoint = Path(model_cfg.get("checkpoint", ""))
    in_path = Path(args.input)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "Set model.checkpoint in config to a local trained Reviser checkpoint."
        )
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    from reviser.data import CursorTokenizer
    from reviser.decoding.inference_core import _action_distribution, generate_response
    from reviser.model import CursorConfig, CursorTransformer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    mcfg = CursorConfig(
        vocab_size=int(model_cfg.get("vocab_size", 50291)),
        d_model=int(model_cfg.get("d_model", 512)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        d_ff=int(model_cfg.get("d_ff", 2048)),
        n_layers=int(model_cfg.get("n_layers", 24)),
        max_canvas_length=int(cfg.get("data", {}).get("max_canvas_length", 512)),
    )
    model = CursorTransformer(mcfg).to(device)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    tok = CursorTokenizer()

    payload = load_json_or_jsonl(in_path)
    rows = list(iter_records(payload))
    max_samples = int(cfg.get("inference", {}).get("max_samples", 0) or 0)
    if max_samples > 0:
        rows = rows[:max_samples]

    outputs = []
    for i, row in enumerate(rows):
        prompt = pick_prompt_text(row)
        if not prompt:
            continue
        response, st = generate_response(
            model=model,
            tokenizer=tok,
            prompt_text=prompt,
            device=device,
            max_steps=int(dec_cfg.get("total_max_actions", 256)),
            max_canvas_len=int(cfg.get("data", {}).get("max_canvas_length", 512)),
            temperature=float(dec_cfg.get("temperature", 0.9)),
            top_k=int(dec_cfg.get("top_k", 50)),
        )
        outputs.append(
            {
                "id": row.get("id", i),
                "prompt": prompt,
                "response": response,
                "final_canvas": response,
                "action_history": [int(x) for x in st.action_history],
                "action_counts": _action_distribution([int(x) for x in st.action_history]),
            }
        )

    write_json(
        args.output,
        {
            "status": "ok",
            "task": "reviser_rollout",
            "model": model_cfg.get("name", "reviser"),
            "family": "reviser",
            "seed": args.seed,
            "device": str(device),
            "n_examples": len(outputs),
            "records": outputs,
        },
    )


if __name__ == "__main__":
    main()
