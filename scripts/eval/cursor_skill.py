#!/usr/bin/env python3
"""Cursor-skill test for classical Reviser checkpoints (shared by the overnight arms, 2026-10-07).

Two parts, both on held-back training texts (shard 133 of data/humanorder/july_texts_v1; never the eval prompts),
both with the eval sampler (legality mask -> T 0.9 -> top-k 50 with ties kept -> softmax):

(a) Delete probe (scripts/dagger/probe_deletes.py): P(DELETE) on 400 states of each kind:
    clean_july / clean_ltr (clean text: should stay low), junk_random (a random token typed at the cursor),
    junk_repeat (the token left of the cursor typed again), own_token (one insert sampled from the model itself,
    also split by its sampling probability).
(b) Hole filling: a July run-and-jump trajectory (July dials) cut after --cut actions (prefix included), so the
    canvas has holes (missing reference stretches between written tokens; the unwritten end is not a hole) and the
    cursor is wherever July left it. Only states with at least one hole are used. The model continues from that state
    for up to --max_new actions (EOR ends). Holes are found exactly: July never deletes, so each written token's place
    in the finished reference gives its reference index. Each new token is classed by where it lands: in a hole, at
    the end (after the last start token), or inside text that had no hole there. Reported:
      - holes_filled_share: share of start holes with >= 1 new token in them at the end
      - all_filled_share: share of states with every hole filled
      - stop_with_open_holes_share: share of states that emitted EOR with >= 1 hole still empty
      - cap_share: share of states that used all --max_new actions without EOR
      - moves / back-moves / deletes per new action and per new token; insert location shares
      - GPT-2 base PPL (July definition: canvas[35:] given the 35-token prefix, exp of the mean per-text mean NLL) of
        the finished canvas, of the start canvas (holes left empty) and of the reference text
Per-state rows go to <out>.rows.jsonl so two checkpoints can be compared state by state.

  python scripts/eval/cursor_skill.py --ckpt runs/humanorder/ckpt/july_runjump_step11656_model.pt \
      --out runs/eval/cursor_skill/july11656.json [--n 400] [--device cuda]
On a rented box (hq_prepare.sh layout): REVISER_SRC=/root/rv2/hq/work/src HF_HOME=/root/rv2/hq/hf_scorers
HF_HUB_OFFLINE=1 /root/rv2/hq/venv/bin/python scripts/eval/cursor_skill.py ...
Needs scripts/dagger/{hunk_oracle,reviser_lm,dagger_data,probe_deletes}.py, scripts/humanorder/{july_texts,fit_q_dials,
orders}.py, experiments/july_control/code/july_impl/{gen_runjump,traj_stats}.py and the July model source (REVISER_SRC).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dagger"))
import dagger_data as dd  # noqa: E402
import hunk_oracle as ho  # noqa: E402
import probe_deletes as pdl  # noqa: E402
import reviser_lm as rl  # noqa: E402

KMAX = 64
AMTS = []
for _a in ho.MOVE_AMOUNTS:
    AMTS += [_a, -_a]


def legal_mask(cur, ln, V, device):
    amt = torch.tensor(AMTS, device=device)
    m = torch.ones((cur.shape[0], V), dtype=torch.bool, device=device)
    np_ = cur[:, None] + amt[None, :]
    m[:, ho.MOVE_START:ho.MOVE_START + ho.N_MOVES] = (np_ >= 0) & (np_ <= ln[:, None])
    m[:, ho.DELETE] = cur > 0
    return m


@torch.no_grad()
def continue_batch(model, seqs, states, max_new, T, top_k, device, gen):
    """KV-cached continuation of equal-length action sequences with the eval sampler. Returns new actions per row
    (EOR included when sampled) and the final states."""
    B, L0 = len(seqs), len(seqs[0])
    assert all(len(s) == L0 for s in seqs)
    states = [s.copy() for s in states]
    past, logits = None, None
    for j in range(L0):
        tok = torch.tensor([s[j] for s in seqs], device=device)
        logits, past = model.forward_step_cached(token_id=tok, canvas_ids_last=None, canvas_mask_last=None,
                                                 past_key_values=past, position_idx=j)
    new = [[] for _ in range(B)]
    alive = [True] * B
    for t in range(max_new):
        cur = torch.tensor([s.cursor for s in states], device=device)
        ln = torch.tensor([len(s.canvas) for s in states], device=device)
        x = logits.float().masked_fill(~legal_mask(cur, ln, logits.shape[-1], device), float("-inf")) / T
        vals, inds = torch.topk(x, KMAX, dim=-1)
        keep = (vals >= vals[:, top_k - 1:top_k]) & torch.isfinite(vals)
        p = torch.softmax(vals.masked_fill(~keep, float("-inf")), dim=-1)
        a = inds.gather(1, torch.multinomial(p, 1, generator=gen))[:, 0]
        al = a.cpu().tolist()
        for b in range(B):
            if not alive[b]:
                continue
            new[b].append(al[b])
            if al[b] == ho.EOR:
                alive[b] = False
            else:
                states[b].step(al[b])
        if not any(alive):
            break
        feed = torch.tensor([al[b] if alive[b] else ho.EOR for b in range(B)], device=device)
        logits, past = model.forward_step_cached(token_id=feed, canvas_ids_last=None, canvas_mask_last=None,
                                                 past_key_values=past, position_idx=L0 + t)
    return new, states


def tagged_replay(actions, tags=None, start=None):
    """Replay with a tag per canvas token. Inserts get tag = action index (or -1 when tags == 'new').
    start: (canvas_tags, cursor) to continue from."""
    canvas, cursor = (list(start[0]), start[1]) if start else ([], 0)
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
            canvas.insert(cursor, -1 if tags == "new" else j)
            cursor += 1
    return canvas, cursor


def hole_state(x, cut):
    """July trajectory cut after `cut` actions -> (seq, State, holes) with holes = [(boundary b, size)] where b is
    the start-canvas index the hole sits before; None if the state has no hole."""
    acts = x["actions"]
    full = x["prefix"] + x["cont"]
    fin, _ = tagged_replay(acts)
    assert len(fin) == len(full), "July trajectory does not rebuild the text"
    ref_pos = {tag: i for i, tag in enumerate(fin)}            # action index -> place in the finished text
    can, _ = tagged_replay(acts[:cut])
    pos = [ref_pos[tag] for tag in can]
    assert pos == sorted(pos)
    holes = [(b, pos[b] - pos[b - 1] - 1) for b in range(1, len(pos)) if pos[b] - pos[b - 1] > 1]
    if not holes:
        return None
    st = ho.replay(acts[:cut])
    return [ho.EOI] + acts[:cut], st, holes


def classify(n_start, cursor0, holes, new_actions):
    """Where the new tokens land and which holes get filled. Start tokens carry tags 0..n_start-1."""
    fin, _ = tagged_replay(new_actions, tags="new", start=(list(range(n_start)), cursor0))
    hole_b = {b for b, _ in holes}
    # region of each new token = index of the next surviving start token to its right (n_start = the end)
    loc = {"hole": 0, "end": 0, "inside": 0}
    filled = set()
    right = n_start
    region_new = []
    for tag in reversed(fin):
        if tag >= 0:
            right = tag
        else:
            region_new.append(right)
    survivors = sorted(t for t in fin if t >= 0)
    surv_set = set(survivors)
    for r in region_new:
        if r == n_start:
            loc["end"] += 1
            continue
        # the new token sits between the previous surviving start token and start token r; every boundary in
        # (prev, r] that is a hole counts as filled (a deleted start token merges neighbouring boundaries)
        prev = max([t for t in survivors if t < r], default=-1)
        bs = [b for b in range(prev + 1, r + 1) if b in hole_b]
        if bs:
            loc["hole"] += 1
            filled.update(bs)
        else:
            loc["inside"] += 1
    deleted_start = n_start - len(surv_set)
    return fin, loc, filled, deleted_start


def gpt2_nll(lm, tok, device, prefix_ids, cont_ids, bs=16):
    """July definition: mean NLL of the re-tokenized continuation given the re-tokenized prefix (last 1024)."""
    out = []
    rows = []
    for p, c in zip(prefix_ids, cont_ids):
        pt, ct = tok.decode(p), tok.decode(c)
        if not ct.strip():
            rows.append(None)
            continue
        pi, ci = tok.encode(pt), tok.encode(ct)
        ids = (pi + ci)[-1024:]
        rows.append((ids, min(len(ci), len(ids) - 1)))
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        real = [r for r in chunk if r is not None]
        res = {}
        if real:
            L = max(len(r[0]) for r in real)
            ids = torch.full((len(real), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(real), L), dtype=torch.long)
            for k, (x, _) in enumerate(real):
                ids[k, :len(x)] = torch.tensor(x)
                att[k, :len(x)] = 1
            ids, att = ids.to(device), att.to(device)
            with torch.no_grad():
                lg = lm(input_ids=ids, attention_mask=att).logits
            for k, (x, nc) in enumerate(real):
                n = len(x)
                res[k] = float(F.cross_entropy(lg[k, n - 1 - nc:n - 1].float(), ids[k, n - nc:n]))
        k = 0
        for r in chunk:
            if r is None:
                out.append(None)
            else:
                out.append(res[k])
                k += 1
    return out


def ppl(nlls):
    v = [x for x in nlls if x is not None]
    return float(math.exp(sum(v) / len(v))) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--shard", type=int, default=133)
    ap.add_argument("--n", type=int, default=400, help="states per kind (delete probe) and hole states")
    ap.add_argument("--cut", type=int, default=120, help="July actions executed before the model takes over "
                                                         "(prefix included)")
    ap.add_argument("--max_new", type=int, default=200, help="cut + max_new stays <= 350 (July training length)")
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--scorer", default="gpt2")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    assert a.cut + a.max_new <= dd.MAX_TRAJ
    t0 = time.time()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = rl.autocast_dtype(device)
    model = rl.load_model(a.ckpt, device).eval()
    res = {"name": a.name or Path(a.ckpt).stem, "ckpt": a.ckpt, "ckpt_sha256": rl.sha256(a.ckpt), "args": vars(a),
           "files_sha256": {f: rl.sha256(ROOT / f) for f in ("scripts/eval/cursor_skill.py",
                                                              "scripts/dagger/probe_deletes.py",
                                                              "scripts/dagger/dagger_data.py")},
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": str(device)}
    if not a.skip_probe:
        res["delete_probe"] = pdl.probe(model, device, amp, a.shard, a.n, a.seed, a.temperature, a.top_k)

    # ---- (b) hole filling
    dd._worker_init(None, a.seed + 2000)
    items = []
    i = 0
    n_no_hole = n_short = 0
    while len(items) < a.n:
        x = dd.july_item((a.shard, i))
        i += 1
        if x.get("skip"):
            continue
        if len(x["actions"]) <= a.cut + 1:
            n_short += 1
            continue
        hs = hole_state(x, a.cut)
        if hs is None:
            n_no_hole += 1
            continue
        items.append((x, *hs))
    gen = torch.Generator(device=device).manual_seed(a.seed + 17)
    rows = []
    for s in range(0, len(items), a.bs):
        chunk = items[s:s + a.bs]
        new, fin_states = continue_batch(model, [c[1] for c in chunk], [c[2] for c in chunk], a.max_new,
                                         a.temperature, a.top_k, device, gen)
        for (x, seq, st, holes), na, fs in zip(chunk, new, fin_states):
            fin, loc, filled, del_start = classify(len(st.canvas), st.cursor, holes, na)
            kinds = {"insert": 0, "move": 0, "back_move": 0, "delete": 0, "eor": 0}
            for t in na:
                k = rl.action_kind(t)
                kinds[k] = kinds.get(k, 0) + 1
                if k == "move" and ho.move_amount(t) < 0:
                    kinds["back_move"] += 1
            eor = bool(na and na[-1] == ho.EOR)
            rows.append({"key": list(x["key"]), "n_start": len(st.canvas), "cursor0": st.cursor,
                         "cursor0_at_end": st.cursor == len(st.canvas), "holes": len(holes),
                         "hole_tokens": sum(h for _, h in holes), "filled": len(filled),
                         "all_filled": len(filled) == len(holes), "eor": eor,
                         "stop_with_open_holes": eor and len(filled) < len(holes), "cap": not eor,
                         "n_new_actions": len(na), "new_tokens": sum(1 for t in fin if t < 0),
                         "deleted_start_tokens": del_start, **{f"n_{k}": v for k, v in kinds.items()},
                         **{f"ins_{k}": v for k, v in loc.items()},
                         "final_canvas": fs.canvas, "start_canvas": st.canvas, "prefix": x["prefix"],
                         "ref": x["prefix"] + x["cont"]})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(a.scorer)
    lm = GPT2LMHeadModel.from_pretrained(a.scorer).to(device).eval()
    P = dd.PREFIX
    nll_fin = gpt2_nll(lm, tok, device, [r["prefix"] for r in rows], [r["final_canvas"][P:] for r in rows])
    nll_start = gpt2_nll(lm, tok, device, [r["prefix"] for r in rows], [r["start_canvas"][P:] for r in rows])
    nll_ref = gpt2_nll(lm, tok, device, [r["prefix"] for r in rows], [r["ref"][P:] for r in rows])
    for r, f_, s_, q_ in zip(rows, nll_fin, nll_start, nll_ref):
        r.update(nll_final=f_, nll_start=s_, nll_ref=q_)
    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
    na = max(1, tot("n_new_actions"))
    nt = max(1, tot("new_tokens"))
    ni = max(1, tot("ins_hole") + tot("ins_end") + tot("ins_inside"))
    res["holes"] = {
        "n_states": n, "texts_read": i, "skipped_no_hole": n_no_hole, "skipped_short": n_short,
        "holes_per_state": tot("holes") / n, "hole_tokens_per_state": tot("hole_tokens") / n,
        "cursor_at_end_share_start": sum(r["cursor0_at_end"] for r in rows) / n,
        "holes_filled_share": tot("filled") / max(1, tot("holes")),
        "all_filled_share": sum(r["all_filled"] for r in rows) / n,
        "stop_rate": sum(r["eor"] for r in rows) / n,
        "stop_with_open_holes_share": sum(r["stop_with_open_holes"] for r in rows) / n,
        "cap_share": sum(r["cap"] for r in rows) / n,
        "new_actions_per_state": na / n, "new_tokens_per_state": tot("new_tokens") / n,
        "moves_per_new_action": tot("n_move") / na, "moves_per_new_token": tot("n_move") / nt,
        "back_moves_per_new_token": tot("n_back_move") / nt, "deletes_per_new_token": tot("n_delete") / nt,
        "deleted_start_tokens_per_state": tot("deleted_start_tokens") / n,
        "insert_location_share": {"hole": tot("ins_hole") / ni, "end": tot("ins_end") / ni,
                                  "inside_text_without_hole": tot("ins_inside") / ni},
        "gpt2_ppl_final": ppl(nll_fin), "gpt2_ppl_start_with_holes": ppl(nll_start), "gpt2_ppl_reference": ppl(nll_ref),
    }
    res["secs"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    with open(str(out) + ".rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    short = {"name": res["name"], "holes": res["holes"]}
    if "delete_probe" in res:
        short["p_delete"] = {k: round(res["delete_probe"][k]["p_delete_mean"], 4)
                             for k in ("clean_july", "clean_ltr", "junk_random", "junk_repeat", "own_token")}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
