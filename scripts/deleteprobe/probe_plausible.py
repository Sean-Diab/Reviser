#!/usr/bin/env python3
"""Delete probe with plausible wrong words (helper D, 2026-10-11; notes/delete-behaviour-2026-10-11.md).

Extends scripts/dagger/probe_deletes.py (helper O; that file is NOT modified, the state construction is copied so the
clean_july / clean_ltr / junk_random / junk_repeat states are exactly O's) with new junk kinds on the same held-out
states (shard 133 of data/humanorder/july_texts_v1, the July run-and-jump trajectory cut at a random t >= 35):

  clean_july  : the cut state, no junk                                   (O's kind; control)
  clean_ltr   : prefix + first m continuation tokens typed left to right (O's kind; control)
  junk_random : + one uniform random vocab token at the cursor           (O's kind; control)
  junk_repeat : + the token left of the cursor typed again               (O's kind; control)
  junk_cand   : + a plausible wrong token: sampled from GPT-2 small (T 1.0, top-k 50) given the TRUE left context
                (the reference text up to and including the canvas token left of the cursor), resampled until it is
                not the true next reference token (and not <|endoftext|>)
  junk_near   : + a token of the same text within 4 positions of the true next token, not equal to it (exactly the
                training kind: variants.wrong_token kind "near", near_d 4, as in jd_n05_imm_m.json)
  junk_near3  : the same with near_d 3 (the brief's wording)
The true next token is the reference token right after the canvas token left of the cursor (July never deletes, so
each canvas token has an exact reference index). States whose true next token is inside the prompt or past the end
of the text are left out of junk_cand / junk_near* (counted).

Per kind, under the eval sampler (legality mask with lock 0 as in O's probe -> T 0.9 -> top-k 50 with ties kept):
  P(DELETE) of the next action (mean), DELETE allowed share, DELETE argmax share; and a seeded rollout of up to
  --roll_steps actions (EOR ends a row): whether the junk token itself is removed within k actions (k = 1..steps),
  whether any DELETE happened within k actions. Per-state rows go to <out>.rows.jsonl (states are identical across
  checkpoints, so models can be compared state by state).

States are built once (needs GPT-2 for junk_cand) and saved to --states (JSON); later runs load that file.

  python scripts/deleteprobe/probe_plausible.py --build_states --states runs/eval/deleteprobe/states.json \
      --gpt2 <gpt2 dir>
  python scripts/deleteprobe/probe_plausible.py --states runs/eval/deleteprobe/states.json --ckpt X.pt --name jd608 \
      --out runs/eval/deleteprobe/probe/jd608.json
  300M: python scripts/humanorder/big608_300m/arch_shim.py scripts/deleteprobe/probe_plausible.py ... (same args)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dagger"))
import dagger_data as dd  # noqa: E402
import hunk_oracle as ho  # noqa: E402
import reviser_lm as rl  # noqa: E402

GPT2_EOT = 50256
KINDS = ["clean_july", "clean_ltr", "junk_random", "junk_repeat", "junk_cand", "junk_near", "junk_near3"]
JUNK_KINDS = [k for k in KINDS if k.startswith("junk_")]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tagged_canvas(actions):
    """Canvas as a list of insert-action indices (July never deletes; same executor as ho.State with lock 0)."""
    canvas, cursor = [], 0
    for j, t in enumerate(actions):
        t = int(t)
        if t == ho.EOR:
            break
        if t == ho.DELETE:
            if cursor > 0:
                canvas.pop(cursor - 1)
                cursor -= 1
        elif ho.is_move(t):
            cursor = max(0, min(len(canvas), cursor + ho.move_amount(t)))
        elif ho.is_insert(t):
            canvas.insert(cursor, j)
            cursor += 1
    return canvas, cursor


def near_token(cont, j, d, rng):
    """variants.wrong_token kind 'near' (copied): a token of cont within d positions of j, != the true token, != EOT."""
    true = int(cont[j])
    opts = [int(cont[i]) for i in range(max(0, j - d), min(len(cont), j + d + 1))
            if i != j and int(cont[i]) != true and int(cont[i]) != GPT2_EOT]
    if opts:
        return opts[rng.randrange(len(opts))]
    return None


# ------------------------------------------------------------------------------------------------ states
def build_states(gpt2_dir, shard=133, n=400, seed=0, device="cuda"):
    from transformers import GPT2LMHeadModel
    rng = random.Random(seed)                       # O's RNG stream, consumed in O's order (t, m, random token)
    dd._worker_init(None, seed + 1000)
    rng_near = random.Random(seed * 1_000_003 + 17)
    rng_near3 = random.Random(seed * 1_000_003 + 19)
    gen_cand = torch.Generator().manual_seed(seed * 1_000_003 + 23)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    lm = GPT2LMHeadModel.from_pretrained(gpt2_dir).to(dev).eval()
    rows = []
    skips = {"july_skip": 0, "cursor0": 0, "true_in_prompt": 0, "true_past_end": 0, "near_none": 0, "near3_none": 0}
    for i in range(n):
        x = dd.july_item((shard, i))
        if x.get("skip"):
            skips["july_skip"] += 1
            continue
        acts = x["actions"]
        t = rng.randint(dd.PREFIX, len(acts) - 1)
        st = ho.replay(acts[:t])
        m = rng.randint(0, len(x["cont"]) - 1)
        tok_rand = rng.randrange(256, 50000)
        full = x["prefix"] + x["cont"]
        P = len(x["prefix"])
        row = {"i": i, "t": t, "m": m, "seq": [ho.EOI] + acts[:t], "canvas": st.canvas, "cursor": st.cursor,
               "ltr": x["prefix"] + x["cont"][:m], "junk": {"junk_random": tok_rand}}
        if st.cursor > 0:
            row["junk"]["junk_repeat"] = st.canvas[st.cursor - 1]
            # true next token: reference token right after the canvas token left of the cursor
            fin, _ = tagged_canvas(acts)
            assert [acts[j] for j in fin] == full, "July trajectory does not rebuild the text"
            ref_pos = {tag: k for k, tag in enumerate(fin)}
            can, cur = tagged_canvas(acts[:t])
            assert cur == st.cursor and [acts[j] for j in can] == st.canvas
            r = ref_pos[can[cur - 1]]
            jn = r + 1
            row["ref_left"] = r
            if jn >= len(full):
                skips["true_past_end"] += 1
            elif jn < P:
                skips["true_in_prompt"] += 1
            else:
                true = int(full[jn])
                row["true_next"] = true
                j = jn - P
                w = near_token(x["cont"], j, 4, rng_near)
                if w is None:
                    skips["near_none"] += 1
                else:
                    row["junk"]["junk_near"] = w
                w3 = near_token(x["cont"], j, 3, rng_near3)
                if w3 is None:
                    skips["near3_none"] += 1
                else:
                    row["junk"]["junk_near3"] = w3
                ctx = full[:jn][-1024:]
                with torch.no_grad():
                    lg = lm(input_ids=torch.tensor([ctx], device=dev)).logits[0, -1].float()
                p_full = torch.softmax(lg, -1).cpu()
                vals, inds = torch.topk(lg, 50)
                p50 = torch.softmax(vals, -1).cpu()
                inds = inds.cpu()
                draws, cand = 0, None
                for _ in range(200):
                    draws += 1
                    c = int(inds[torch.multinomial(p50, 1, generator=gen_cand)[0]])
                    if c != true and c != GPT2_EOT:
                        cand = c
                        break
                if cand is None:          # fallback: most likely top-50 token that is neither true nor EOT
                    cand = next(int(c) for c in inds.tolist() if int(c) != true and int(c) != GPT2_EOT)
                    draws = -1
                rank = int((p_full > p_full[cand]).sum())
                row["junk"]["junk_cand"] = cand
                row["cand_info"] = {"p_gpt2": float(p_full[cand]), "p_true": float(p_full[true]), "rank": rank,
                                    "draws": draws, "true_rank": int((p_full > p_full[true]).sum())}
        else:
            skips["cursor0"] += 1
        rows.append(row)
    return {"shard": shard, "n": n, "seed": seed, "gpt2_dir": str(gpt2_dir), "skips": skips, "rows": rows}


def kind_items(S, kind):
    """(state index, seq, State, junk canvas index or None) for one kind."""
    out = []
    for si, r in enumerate(S["rows"]):
        if kind == "clean_july":
            out.append((si, list(r["seq"]), ho.State(r["canvas"], r["cursor"]), None))
        elif kind == "clean_ltr":
            out.append((si, [ho.EOI] + r["ltr"], ho.State(r["ltr"], len(r["ltr"])), None))
        else:
            tok = r["junk"].get(kind)
            if tok is None:
                continue
            st = ho.State(r["canvas"], r["cursor"])
            jpos = st.cursor
            st.step(int(tok))
            assert st.canvas[jpos] == int(tok) and st.cursor == jpos + 1
            out.append((si, list(r["seq"]) + [int(tok)], st, jpos))
    return out


# ------------------------------------------------------------------------------------------------ probe
@torch.no_grad()
def dist(model, seqs, states, device, amp, T, k, bs):
    out = []
    for i in range(0, len(seqs), bs):
        lg = rl.last_logits(model, seqs[i:i + bs], device, amp)
        out.append(rl.sampling_logprobs(lg, rl.valid_mask(states[i:i + bs], device), T, k).exp().cpu())
    return torch.cat(out)


def run_kind(model, items, device, amp, T, k, steps, gen, bs):
    seqs = [list(s) for _, s, _, _ in items]
    states = [st.copy() for _, _, st, _ in items]
    jpos = [jp for _, _, _, jp in items]
    B = len(items)
    rows = [{"state": items[b][0], "acts": [], "removed_at": None, "first_delete_at": None, "eor_at": None}
            for b in range(B)]
    alive = list(range(B))
    for step in range(1, steps + 1):
        if not alive:
            break
        p = dist(model, [seqs[b] for b in alive], [states[b] for b in alive], device, amp, T, k, bs)
        if step == 1:
            for j, b in enumerate(alive):
                rows[b]["p_delete"] = float(p[j, ho.DELETE])
                rows[b]["delete_allowed"] = bool(p[j, ho.DELETE] > 0)
                rows[b]["delete_argmax"] = bool(int(p[j].argmax()) == ho.DELETE)
                rows[b]["p_eor"] = float(p[j, ho.EOR])
        a = torch.multinomial(p, 1, generator=gen)[:, 0].tolist()
        nxt = []
        for j, b in enumerate(alive):
            act = int(a[j])
            rows[b]["acts"].append(act)
            if act == ho.EOR:
                rows[b]["eor_at"] = step
                continue
            st = states[b]
            c = st.cursor
            if act == ho.DELETE:
                if rows[b]["first_delete_at"] is None:
                    rows[b]["first_delete_at"] = step
                if jpos[b] is not None and c > 0:
                    if c - 1 == jpos[b]:
                        jpos[b] = None
                        rows[b]["removed_at"] = step
                    elif c - 1 < jpos[b]:
                        jpos[b] -= 1
            elif ho.is_insert(act):
                if jpos[b] is not None and c <= jpos[b]:
                    jpos[b] += 1
            st.step(act)
            seqs[b].append(act)
            nxt.append(b)
        alive = nxt
    return rows


def summarize(rows, steps, junk):
    n = len(rows)
    s = {"n": n,
         "p_delete_mean": sum(r["p_delete"] for r in rows) / n,
         "delete_allowed_share": sum(r["delete_allowed"] for r in rows) / n,
         "delete_argmax_share": sum(r["delete_argmax"] for r in rows) / n,
         "p_eor_mean": sum(r["p_eor"] for r in rows) / n,
         "any_delete_within": {str(k): sum(1 for r in rows if r["first_delete_at"] is not None
                                           and r["first_delete_at"] <= k) / n for k in range(1, steps + 1)},
         "eor_within": {str(k): sum(1 for r in rows if r["eor_at"] is not None and r["eor_at"] <= k) / n
                        for k in range(1, steps + 1)}}
    if junk:
        s["junk_removed_within"] = {str(k): sum(1 for r in rows if r["removed_at"] is not None
                                                and r["removed_at"] <= k) / n for k in range(1, steps + 1)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--build_states", action="store_true")
    ap.add_argument("--gpt2", default=None, help="GPT-2 small directory (only for --build_states)")
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=133)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roll_seed", type=int, default=1234)
    ap.add_argument("--roll_steps", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.build_states:
        t0 = time.time()
        S = build_states(a.gpt2, a.shard, a.n, a.seed, a.device)
        S["secs"] = round(time.time() - t0, 1)
        S["script_sha256"] = sha256(__file__)
        S["gpt2_safetensors_sha256"] = sha256(Path(a.gpt2) / "model.safetensors")
        Path(a.states).parent.mkdir(parents=True, exist_ok=True)
        Path(a.states).write_text(json.dumps(S))
        ci = [r["cand_info"] for r in S["rows"] if "cand_info" in r]
        print(json.dumps({"states": a.states, "rows": len(S["rows"]), "skips": S["skips"], "secs": S["secs"],
                          "counts": {k: sum(1 for r in S["rows"] if k in r["junk"]) for k in JUNK_KINDS},
                          "cand_p_gpt2_mean": sum(c["p_gpt2"] for c in ci) / max(1, len(ci))}), flush=True)
        if not a.ckpt:
            return
    S = json.loads(Path(a.states).read_text())
    t0 = time.time()
    amp = rl.autocast_dtype(device)
    model = rl.load_model(a.ckpt, device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    res = {"name": a.name, "ckpt": a.ckpt, "ckpt_sha256": sha256(a.ckpt), "n_params": n_params, "args": vars(a),
           "states_sha256": sha256(a.states), "script_sha256": sha256(__file__),
           "files_sha256": {p: sha256(ROOT / p) for p in ["scripts/dagger/reviser_lm.py", "scripts/dagger/hunk_oracle.py",
                                                          "scripts/dagger/dagger_data.py"]},
           "amp": str(amp), "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kinds": {}}
    out_rows = []
    for ki, kind in enumerate(KINDS):
        items = kind_items(S, kind)
        gen = torch.Generator().manual_seed(a.roll_seed * 1000 + ki)
        rows = run_kind(model, items, device, amp, a.temperature, a.top_k, a.roll_steps, gen, a.bs)
        res["kinds"][kind] = summarize(rows, a.roll_steps, kind in JUNK_KINDS)
        if kind == "junk_cand":
            bins = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.01)]
            by = {}
            for lo, hi in bins:
                sel = [r for r in rows if lo <= S["rows"][r["state"]]["cand_info"]["p_gpt2"] < hi]
                if sel:
                    by[f"{lo}-{hi}"] = summarize(sel, a.roll_steps, True)
            res["kinds"]["junk_cand_by_gpt2_prob"] = by
        for r in rows:
            r["kind"] = kind
            out_rows.append(r)
        print(json.dumps({"name": a.name, "kind": kind, "n": len(rows),
                          "p_delete": round(res["kinds"][kind]["p_delete_mean"], 4),
                          "argmax": round(res["kinds"][kind]["delete_argmax_share"], 3),
                          "removed5": res["kinds"][kind].get("junk_removed_within", {}).get("5"),
                          "secs": round(time.time() - t0, 1)}), flush=True)
    res["secs"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    with open(a.out + ".rows.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"out": a.out, "secs": res["secs"]}), flush=True)


if __name__ == "__main__":
    main()
