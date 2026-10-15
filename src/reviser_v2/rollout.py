"""Batched history-only rollout with KV caching, executor legality masks, and per-example traces.

Two ways to present a supplied draft:
  * ``conditioning``  prompt SEP draft [cursor moves] EDIT, then free actions (D3-style input).
  * ``forced_prefix`` prompt SEP EDIT, then the draft typed as forced Insert actions (+ cursor
                      moves), then free actions. This is the state distribution blank-start models
                      (D0/D1/D2) see when they themselves have inserted junk.
``forced_after`` optionally appends further forced actions (e.g. delayed-correction probes).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from .actions import ActionSpace
from .executor import Canvas
from .trajectories import conditioning_ids


@dataclass
class Request:
    prompt: List[int]
    draft: List[int] = field(default_factory=list)
    seed_cursor: Optional[int] = None
    draft_mode: str = "conditioning"
    forced_after: List[int] = field(default_factory=list)
    max_actions: int = 256


@dataclass
class Trace:
    actions: List[int]
    final: List[int]
    stopped: bool
    illegal_forced: bool
    draft_inst_deleted: List[int]      # draft indices whose instance was deleted
    self_deletes: int                  # deletions of instances inserted during free rollout
    n_ins: int = 0
    n_del: int = 0
    n_move: int = 0
    p_delete_when_legal: List[float] = field(default_factory=list)  # post-mask P(DELETE) per step where legal
    seconds: float = 0.0


def _prepare(space: ActionSpace, r: Request):
    if r.draft_mode == "conditioning":
        cond = conditioning_ids(space, r.prompt, r.draft, r.seed_cursor)
        canvas = Canvas.from_draft(space, r.draft, r.seed_cursor)
        inst_to_draft = {-(i + 1): i for i in range(len(r.draft))}
    elif r.draft_mode == "forced_prefix":
        cond = list(r.prompt) + [space.SEP, space.EDIT] + list(r.draft)
        if r.seed_cursor is not None and r.seed_cursor != len(r.draft):
            cond += space.compile_goto(len(r.draft), r.seed_cursor)
        canvas = Canvas.from_draft(space, [])
        for t in r.draft:
            canvas.apply(t)
        if r.seed_cursor is not None:
            canvas.cursor = r.seed_cursor
        inst_to_draft = {i: i for i in range(len(r.draft))}
    else:
        raise ValueError(r.draft_mode)
    illegal = False
    for a in r.forced_after:
        if not canvas.is_legal(a):
            illegal = True
            break
        canvas.apply(a)
        cond.append(a)
    return cond, canvas, inst_to_draft, illegal


@torch.no_grad()
def rollout(model, space: ActionSpace, reqs: Sequence[Request], device="cuda", greedy=True,
            temperature=1.0, top_p=1.0, generator: Optional[torch.Generator] = None,
            amp_dtype=torch.float16, record_pdel=True, type_greedy=False, stop_bias=0.0,
            insert_only=False) -> List[Trace]:
    """greedy=True: token argmax. type_greedy=True (with greedy): pick the action TYPE (insert / delete / stop / move)
    with the largest summed probability, then the argmax token within it. Token argmax is biased toward types whose
    mass sits on few tokens (moves), which makes random-order restoration models loop on moves.
    stop_bias: added to the STOP logit (after legality masking) at every step; a decode-time prior against over-editing.
    insert_only=True: only content tokens and STOP are allowed (moves and DELETE masked): the AR reference decoder,
    a softmax over content + STOP(=EOS). Without it an AR-trained model may still emit (rare) moves."""
    model.eval()
    B = len(reqs)
    prepped = [_prepare(space, r) for r in reqs]
    conds = [p[0] for p in prepped]
    canvases = [p[1] for p in prepped]
    first_free_iid = [c.next_iid for c in canvases]
    T0 = max(len(c) for c in conds)
    ids = torch.zeros(B, T0, dtype=torch.long)
    key_pad = torch.ones(B, T0, dtype=torch.bool)
    for i, c in enumerate(conds):
        ids[i, T0 - len(c):] = torch.tensor(c)
        key_pad[i, T0 - len(c):] = False
    positions = (torch.cumsum((~key_pad).long(), 1) - 1).clamp(min=0)
    ids, key_pad, positions = ids.to(device), key_pad.to(device), positions.to(device)
    use_amp = device != "cpu" and amp_dtype is not None
    t_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        logits, past = model(ids, positions=positions, key_pad=key_pad if bool(key_pad.any()) else None)
    last = logits[:, -1, :].float()
    pos = positions[:, -1:]
    done = [c.stopped or prepped[i][3] for i, c in enumerate(canvases)]
    actions: List[List[int]] = [[] for _ in range(B)]
    pdel: List[List[float]] = [[] for _ in range(B)]
    self_del = [0] * B
    draft_del: List[List[int]] = [[] for _ in range(B)]
    max_steps = max(r.max_actions for r in reqs)
    nc = space.n_content
    for step in range(max_steps):
        if all(done):
            break
        spec_mask = torch.tensor([c.legal_special_mask() if not done[i] else [False] * (space.vocab_size - nc)
                                  for i, c in enumerate(canvases)], dtype=torch.bool, device=device)
        lg = last.clone()
        if insert_only:
            spec_mask[:, :] = False
            spec_mask[:, space.STOP - nc] = True
        lg[:, nc:] = lg[:, nc:].masked_fill(~spec_mask, float("-inf"))
        if stop_bias:
            lg[:, space.STOP] += stop_bias
        probs = torch.softmax(lg / (1.0 if greedy else temperature), dim=-1)
        if greedy and type_greedy:
            ms = space.move_start
            groups = [(0, nc), (space.DELETE, space.DELETE + 1), (space.STOP, space.STOP + 1), (ms, space.vocab_size)]
            mass = torch.stack([probs[:, a:b].sum(-1) for a, b in groups], -1)
            g = mass.argmax(-1)
            gmask = torch.zeros_like(lg, dtype=torch.bool)
            for k, (a, b) in enumerate(groups):
                gmask[:, a:b] |= (g == k)[:, None]
            choice = lg.masked_fill(~gmask, float("-inf")).argmax(-1)
        elif greedy:
            choice = lg.argmax(-1)
        else:
            if top_p < 1.0:
                sp, si = probs.sort(-1, descending=True)
                keep = sp.cumsum(-1) - sp < top_p
                sp = sp * keep
                choice = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=generator)).squeeze(1)
            else:
                choice = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        choice_l = choice.tolist()
        pd = probs[:, space.DELETE].tolist() if record_pdel else None
        feed = []
        for i in range(B):
            if done[i]:
                feed.append(space.STOP)
                continue
            a = choice_l[i]
            c = canvases[i]
            if record_pdel and spec_mask[i, space.DELETE - nc]:
                pdel[i].append(pd[i])
            kind, iid = c.apply(a)
            actions[i].append(a)
            if kind == "del":
                if iid is not None and iid >= first_free_iid[i]:
                    self_del[i] += 1
                elif iid in prepped[i][2]:
                    draft_del[i].append(prepped[i][2][iid])
            if c.stopped or len(actions[i]) >= reqs[i].max_actions:
                done[i] = True
            feed.append(a)
        if all(done):
            break
        nxt = torch.tensor(feed, device=device).unsqueeze(1)
        pos = pos + 1
        key_pad = torch.cat([key_pad, torch.zeros(B, 1, dtype=torch.bool, device=device)], 1)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits, past = model(nxt, positions=pos, key_pad=key_pad if bool(key_pad.any()) else None, past=past)
        last = logits[:, -1, :].float()
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    out = []
    for i in range(B):
        acts = actions[i]
        out.append(Trace(
            actions=acts, final=list(canvases[i].tokens), stopped=canvases[i].stopped,
            illegal_forced=prepped[i][3], draft_inst_deleted=draft_del[i], self_deletes=self_del[i],
            n_ins=sum(space.is_content(a) for a in acts), n_del=sum(a == space.DELETE for a in acts),
            n_move=sum(space.is_move(a) for a in acts), p_delete_when_legal=pdel[i], seconds=elapsed / B))
    return out
