#!/usr/bin/env python3
"""Decode one Reviser checkpoint on a fixed prompt set and record per-example stopping / length / action stats.

Uses the historical July decoder `_generate_from_action_prefix_nocanvas` (pretrain_resttraj.py:647-765, unmodified
copy under code/src) and the July model constructor (code/july_impl/eval_compare.py:30-46). Output is one JSONL row
per prompt, so several checkpoints decoded with the same --seed can be compared pairwise (common random numbers).

Prompt sets
  dclm_dev : July eval_compare.dev_prompts() -- final texts of dclm_baseline_resttraj_dev_fixed35_10k_seed125,
             random.Random(5) shuffle per shard, keep len > prefix+20, first n (July used n=100).
  c4       : workspaces/4-21/c4_v2_arena_test/c4_segments_v2.json, items in file order, first n,
             prompt = token_ids[:35] (GPT-2 ids, used directly as INSERT actions).
Protocols
  july  : max_new_actions=256 after the 35-token prefix, fp32, max_seq_len=512 (eval_compare.py:92-96).
  paper : total budget 256 actions incl. prefix, capped at max_seq_length-1=255 of the release config -> 220 new,
          bf16 autocast (run_cursor_inference_eor_bias.py:115-117,125,324). Decoder math otherwise identical.
Both: temperature 0.9, top_k 50, validity masking, EOR stops decoding.
Seeding
  per_example (default): torch.manual_seed(seed + i) before prompt i -> identical RNG stream per prompt across models.
  global      : torch.manual_seed(seed) once (paper style; July eval_compare.py did not seed at all).
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "july_impl"))

from reviser.data import CursorTokenizer  # noqa: E402
from reviser.model import CursorConfig, CursorTransformer  # noqa: E402
from reviser.training.pretrain_resttraj import _generate_from_action_prefix_nocanvas  # noqa: E402
import traj_stats  # noqa: E402  (July code/july_impl/traj_stats.py, unmodified)

EOR, DELETE, MOVE_START, N_MOVES, GPT2_VOCAB = 50260, 50259, 50261, 20, 50257


def load_model(path: str, device) -> CursorTransformer:
    """Same as July eval_compare.load_model (dropout off, rope caches rebuilt)."""
    cfg = CursorConfig(vocab_size=50291, d_model=512, n_heads=8, d_ff=2048, n_layers=24, dropout=0.0,
                       attention_dropout=0.0, positional_encoding="rope", max_seq_length=2048,
                       max_canvas_length=512, tie_embeddings=True)
    model = CursorTransformer(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k: v for k, v in sd.items() if "rope.cos_cached" not in k and "rope.sin_cached" not in k}
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "rope." not in k] + list(res.unexpected_keys)
    assert not bad, f"bad keys: {bad[:5]}"
    return model.to(device).eval()


def dclm_dev_prompts(dev_dir: str, n: int, prefix_len: int, seed: int = 5):
    """Verbatim logic of July eval_compare.dev_prompts (eval_compare.py:49-67)."""
    files = sorted(glob.glob(os.path.join(dev_dir, "shards", "final_tokens_shard*.npy")))
    assert files, f"no final_tokens under {dev_dir}/shards"
    rng = random.Random(seed)
    out = []
    for tf in files:
        of = tf.replace("final_tokens_", "final_offsets_")
        toks = np.load(tf, mmap_mode="r")
        offs = np.load(of, mmap_mode="r")
        idxs = list(range(offs.shape[0]))
        rng.shuffle(idxs)
        for si in idxs:
            s, l = int(offs[si, 0]), int(offs[si, 1])
            y = [int(x) for x in toks[s:s + l]]
            if len(y) > prefix_len + 20:
                out.append({"id": f"{Path(tf).stem}:{si}", "ref": y})
            if len(out) >= n:
                return out
    return out


def c4_prompts(path: str, n: int):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"id": str(it["id"]), "ref": [int(x) for x in it["token_ids"]]} for it in items[:n]]


def action_counts(actions, prefix_len):
    new = actions[prefix_len:]
    c = {"insert": 0, "delete": 0, "move": 0, "eor": 0, "other": 0}
    for t in new:
        t = int(t)
        if t < GPT2_VOCAB:
            c["insert"] += 1
        elif t == DELETE:
            c["delete"] += 1
        elif MOVE_START <= t < MOVE_START + N_MOVES:
            c["move"] += 1
        elif t == EOR:
            c["eor"] += 1
        else:
            c["other"] += 1
    return c


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt_set", choices=["dclm_dev", "c4"], required=True)
    ap.add_argument("--dclm_dir", default="")
    ap.add_argument("--c4_json", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix_len", type=int, default=35)
    ap.add_argument("--protocol", choices=["july", "paper"], default="july")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--seed_mode", choices=["per_example", "global"], default="per_example")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tok = CursorTokenizer()
    prompts = (dclm_dev_prompts(a.dclm_dir, a.n, a.prefix_len) if a.prompt_set == "dclm_dev"
               else c4_prompts(a.c4_json, a.n))
    if a.protocol == "july":
        max_new, max_seq, amp = 256, 512, contextlib.nullcontext
    else:
        max_new, max_seq = 256 - a.prefix_len, 256  # 256 total, capped at max_seq_length-1 = 255 -> 220 new
        amp = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")  # noqa: E731
    model = load_model(a.ckpt, device)
    if a.seed_mode == "global":
        torch.manual_seed(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"name": a.name, "ckpt": str(a.ckpt), "ckpt_sha256": sha256(a.ckpt), "args": vars(a),
            "max_new_actions_requested": max_new, "max_seq_len": max_seq, "n_prompts": len(prompts),
            "torch": torch.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    t0 = time.time()
    rows = []
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for i, p in enumerate(prompts):
            if a.seed_mode == "per_example":
                torch.manual_seed(a.seed + i)
            prefix = p["ref"][:a.prefix_len]
            with amp():
                _, st = _generate_from_action_prefix_nocanvas(
                    model=model, tokenizer=tok, prefix_actions=list(prefix), device=device,
                    max_new_actions=max_new, max_seq_len=max_seq, temperature=a.temperature, top_k=a.top_k)
            actions = [int(x) for x in st.action_history]
            canvas = [int(x) for x in st.canvas]
            n_new = len(actions) - a.prefix_len
            stopped = bool(actions and actions[-1] == EOR)
            row = {"i": i, "id": p["id"], "prefix_ids": prefix, "actions_new": actions[a.prefix_len:],
                   "canvas_ids": canvas, "ref_cont_ids": p["ref"][a.prefix_len:],
                   "stopped_eor": stopped, "hit_budget": (not stopped), "n_new_actions": n_new,
                   "out_len": len(canvas) - a.prefix_len, "prefix_intact": canvas[:a.prefix_len] == prefix,
                   "counts": action_counts(actions, a.prefix_len)}
            f.write(json.dumps(row) + "\n")
            rows.append((actions, a.prefix_len))
            if (i + 1) % 50 == 0:
                print(f"[{a.name}] {i+1}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    stats = traj_stats.aggregate(rows)
    Path(str(out) + ".summary.json").write_text(json.dumps({"_meta": meta, "traj_stats": stats,
                                                            "decode_secs": time.time() - t0}, indent=1))
    print(json.dumps({"name": a.name, "n": len(prompts), "secs": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
