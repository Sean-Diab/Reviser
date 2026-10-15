#!/usr/bin/env python3
"""Batched version of the July / paper Reviser decoder, for the paper-style re-test (notes/paper-retest-2026-10-08.md).

Same per-row computation as experiments/july_control/eval/decode_eval.py --protocol paper, which calls
_generate_from_action_prefix_nocanvas (pretrain_resttraj.py:647-765):
  * history = [EOI] + prompt tokens as INSERT actions (positions 0..m), then sampled actions; no canvas input;
  * KV-cached stepping with bf16 autocast (CursorTransformer.forward_step_cached);
  * legality mask from the replayed CursorState (the July executor, reviser.decoding.inference_core.CursorState):
    moves must stay in [0, len(canvas)], DELETE needs cursor > 0;
  * temperature then top-k with ties at the cutoff kept, softmax, multinomial; EOR stops a row;
  * budget: total actions <= max_seq_len - 1 = 255 -> 220 new actions after a 35-token prompt.
Differences (all deliberate, recorded in every output's _meta):
  * rows are decoded in lockstep batches (all prompts have the same length, so every row is at the same position);
    finished rows are fed EOR and ignored. Numerics of a batched bf16 matmul can differ from batch-1 in the last bit.
  * RNG: one torch.Generator per batch, seeded from (seed, batch index); not the paper's global stream, so samples
    are not bit-identical to any earlier run (same distribution).
  * --lock_prompt: the paper's validity mask "disallow edits to the prompt prefix" (main.tex Alg. 1, sec. 4.4):
    a move may not take the cursor left of the prompt end m, and DELETE needs cursor > m. The cursor starts at m,
    so inserts can never land inside the prompt and the prompt stays byte-identical. (Compared with helper A's
    decode_eval_slice.py --lock_prompt in notes/paper-retest-2026-10-08.md.)
  * --eor_bias B (default 0 = off, the paper decoder): adds B to the EOR logit before temperature and top-k, so the
    model stops earlier by its own choice. Only for the "at rev100's length" test (added 2026-10-08 13:00Z);
    decodes with B != 0 are not the paper setting and use output names *_eorb<B>.

Output: one JSONL per model (all seeds), decode_eval.py row format plus "seed"; a _meta first row; <out>.done.

  python scripts/papereval/decode_batched.py --ckpt X.pt --name g8m2jd --lock_prompt --out_dir runs/papereval/dec
  (prompts default to data/papereval/c4_paper_3k.json; each prompt is decoded with its own seed 123/124/125)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = os.environ.get("REVISER_SRC", str(ROOT / "experiments" / "july_control" / "code" / "src"))
sys.path.insert(0, SRC)
sys.path.insert(0, str(ROOT / "scripts" / "dagger"))

EOI, DELETE, EOR, MOVE_START = 50258, 50259, 50260, 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
AMTS = []
for _a in MOVE_AMOUNTS:
    AMTS += [_a, -_a]          # token MOVE_START + j has amount AMTS[j]
GPT2_VOCAB = 50257


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def action_counts(new_actions):
    c = {"insert": 0, "delete": 0, "move": 0, "eor": 0, "other": 0}
    for t in new_actions:
        t = int(t)
        if t < GPT2_VOCAB:
            c["insert"] += 1
        elif t == DELETE:
            c["delete"] += 1
        elif MOVE_START <= t < MOVE_START + 20:
            c["move"] += 1
        elif t == EOR:
            c["eor"] += 1
        else:
            c["other"] += 1
    return c


def legal_mask(cur: torch.Tensor, ln: torch.Tensor, V: int, lo: int) -> torch.Tensor:
    """(B, V) bool. lo = 0: the July rule. lo = m: the prompt lock (cursor never left of m, DELETE needs cursor > m)."""
    amt = torch.tensor(AMTS, device=cur.device)
    m = torch.ones((cur.shape[0], V), dtype=torch.bool, device=cur.device)
    np_ = cur[:, None] + amt[None, :]
    m[:, MOVE_START:MOVE_START + 20] = (np_ >= lo) & (np_ <= ln[:, None])
    m[:, DELETE] = cur > lo
    return m


def sample_step(logits: torch.Tensor, valid: torch.Tensor, temperature: float, top_k: int, gen) -> torch.Tensor:
    """July _sample_next, batched: mask -> /T -> top-k (ties at the cutoff kept) -> softmax (fp32) -> multinomial.
    The logits stay in the model's output dtype (bf16 under autocast) up to the softmax, as in the serial decoder."""
    x = logits.masked_fill(~valid, float("-inf"))
    if temperature != 1.0:
        x = x / temperature
    if top_k > 0:
        cutoff = torch.topk(x, top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < cutoff, float("-inf"))
    probs = torch.softmax(torch.nan_to_num(x.float(), neginf=-1e9, posinf=1e9), dim=-1)
    return torch.multinomial(probs, 1, generator=gen)[:, 0]


@torch.no_grad()
def decode_batch(model, prefixes, *, device, max_seq_len, temperature, top_k, lock_prompt, gen,
                 forced=None, eor_bias=0.0):
    """Decode equal-length prefixes in lockstep. forced: optional list of action lists to replay instead of sampling
    (for the equivalence test); then returns per-step log-probabilities of the forced actions too."""
    from reviser.decoding.inference_core import CursorState
    B = len(prefixes)
    m = len(prefixes[0])
    assert all(len(p) == m for p in prefixes), "prefixes must have equal length"
    max_new = min(max_seq_len - 1 - m, 10 ** 9)
    states = [CursorState.from_prompt([EOI]) for _ in range(B)]
    past = None
    pos = 0
    tok = torch.full((B,), EOI, dtype=torch.long, device=device)
    logits, past = model.forward_step_cached(token_id=tok, canvas_ids_last=None, canvas_mask_last=None,
                                             past_key_values=past, position_idx=pos)
    pos += 1
    for j in range(m):
        tok = torch.tensor([p[j] for p in prefixes], dtype=torch.long, device=device)
        for b in range(B):
            states[b].execute_action(int(prefixes[b][j]))
        logits, past = model.forward_step_cached(token_id=tok, canvas_ids_last=None, canvas_mask_last=None,
                                                 past_key_values=past, position_idx=pos)
        pos += 1
    alive = [True] * B
    lo = m if lock_prompt else 0
    forced_lp = [[] for _ in range(B)] if forced is not None else None
    for t in range(max_new):
        cur = torch.tensor([s.cursor_pos for s in states], device=device)
        ln = torch.tensor([len(s.canvas) for s in states], device=device)
        valid = legal_mask(cur, ln, logits.shape[-1], lo)
        if eor_bias:
            logits = logits.clone()
            logits[:, EOR] += eor_bias
        if forced is None:
            a = sample_step(logits, valid, temperature, top_k, gen)
        else:
            x = logits.masked_fill(~valid, float("-inf")) / temperature
            cutoff = torch.topk(x, top_k, dim=-1).values[:, -1:]
            x = x.masked_fill(x < cutoff, float("-inf"))
            lp = torch.log_softmax(x.float(), dim=-1)
            a_list = [forced[b][t] if t < len(forced[b]) else EOR for b in range(B)]
            a = torch.tensor(a_list, dtype=torch.long, device=device)
            lpa = lp.gather(1, a[:, None])[:, 0].tolist()
            for b in range(B):
                if alive[b]:
                    forced_lp[b].append(lpa[b])
        a_list = a.tolist()
        for b in range(B):
            if not alive[b]:
                continue
            if a_list[b] == EOR:
                states[b].action_history.append(EOR)
                alive[b] = False
            else:
                states[b].execute_action(int(a_list[b]))
        if not any(alive):
            break
        feed = torch.tensor([a_list[b] if alive[b] else EOR for b in range(B)], dtype=torch.long, device=device)
        logits, past = model.forward_step_cached(token_id=feed, canvas_ids_last=None, canvas_mask_last=None,
                                                 past_key_values=past, position_idx=pos)
        pos += 1
    return states, forced_lp


def load_prompts(path, n=None):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items["items"]
    items = items if n is None else items[:n]
    return [{"id": str(it["id"]), "seed": it.get("seed"), "ref": [int(x) for x in it["token_ids"]]} for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "data" / "papereval" / "c4_paper_3k.json"),
                    help="JSON list of {id, seed, token_ids} (scripts/papereval/build_prompts.py)")
    ap.add_argument("--n", type=int, default=None, help="first n prompts only (smoke tests)")
    ap.add_argument("--prefix_len", type=int, default=35)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_seq_len", type=int, default=256, help="paper: 256 -> 255 actions incl. prompt -> 220 new")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--lock_prompt", action="store_true")
    ap.add_argument("--eor_bias", type=float, default=0.0, help="added to the EOR logit (0 = paper decoder)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="c4paper3k")
    a = ap.parse_args()

    import reviser_lm as rl  # scripts/dagger/reviser_lm.py: July CursorTransformer loader (+ GPT-2 based arms)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    prompts = load_prompts(a.prompts, a.n)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = "lock" if a.lock_prompt else "nolock"
    out = out_dir / f"{a.tag}.{lock}.{a.name}.jsonl"
    if Path(str(out) + ".done").exists():
        print(json.dumps({"skip_done": str(out)}), flush=True)
        return
    model = rl.load_model(a.ckpt, device).eval()
    meta = {"name": a.name, "ckpt": str(a.ckpt), "ckpt_sha256": sha256(a.ckpt), "args": vars(a),
            "prompts_sha256": sha256(a.prompts), "n_prompts": len(prompts),
            "decoder": "scripts/papereval/decode_batched.py", "decoder_sha256": sha256(__file__),
            "max_new_actions": a.max_seq_len - 1 - a.prefix_len, "torch": torch.__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "rng": "per (seed, batch): torch.Generator seeded seed*1000003 + batch_index; batches of --batch prompts "
                   "within each seed group, in file order"}
    t0 = time.time()
    tmp = Path(str(out) + ".tmp")
    seeds = sorted({p["seed"] for p in prompts}, key=lambda x: (x is None, x))
    done = 0
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for seed in seeds:
            group = [(i, p) for i, p in enumerate(prompts) if p["seed"] == seed]
            sd = int(seed) if seed is not None else 0
            for bi, s in enumerate(range(0, len(group), a.batch)):
                chunk = group[s:s + a.batch]
                gen = torch.Generator(device=device)
                gen.manual_seed(sd * 1000003 + bi)
                prefixes = [p["ref"][:a.prefix_len] for _, p in chunk]
                states, _ = decode_batch(model, prefixes, device=device, max_seq_len=a.max_seq_len,
                                         temperature=a.temperature, top_k=a.top_k, lock_prompt=a.lock_prompt, gen=gen,
                                         eor_bias=a.eor_bias)
                for k, ((i, p), st) in enumerate(zip(chunk, states)):
                    actions = [int(x) for x in st.action_history]
                    canvas = [int(x) for x in st.canvas]
                    prefix = prefixes[k]
                    new = actions[a.prefix_len:]
                    stopped = bool(new and new[-1] == EOR)
                    row = {"i": i, "id": p["id"], "seed": seed, "prefix_ids": prefix, "actions_new": new,
                           "canvas_ids": canvas, "ref_cont_ids": p["ref"][a.prefix_len:], "stopped_eor": stopped,
                           "hit_budget": not stopped, "n_new_actions": len(new), "out_len": len(canvas) - a.prefix_len,
                           "prefix_intact": canvas[:a.prefix_len] == prefix, "counts": action_counts(new)}
                    f.write(json.dumps(row) + "\n")
                done += len(chunk)
                print(f"[{a.name}] seed {seed} {done}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    os.replace(tmp, out)
    Path(str(out) + ".done").write_text(json.dumps({"secs": round(time.time() - t0, 1), "utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(json.dumps({"out": str(out), "secs": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
