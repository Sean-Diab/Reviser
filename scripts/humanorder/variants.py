"""Trajectory variants for the overnight data exploration (notes/overnight-data-variants-2026-10-07.md).

Every variant writes the 35-token prefix first, then builds the continuation, then EOR, in the July shard format
(GPT-2 ids = insert at the cursor, 50259 = DELETE the token left of the cursor, 50261.. = power-of-2 moves,
50260 = EOR). Each builder returns (actions, mask): mask[i] = 1 means the training label of action i is ignored
(used for the wrong inserts of the delete arms when loss masking is on; the prefix is masked by the trainer).

Layers
1. Order (which continuation token is written when):
   - `runjump_order`: July's run-and-jump policy (or the 6-dial copy runjump6.py) run on the token INDICES, so the
     random draws are exactly those of the normal build (the policy never looks at token values) and the order can be
     mixed with other sources and decorated with errors. `orders.order_to_actions(prefix, cont, order)` rebuilds the
     policy's own action stream exactly (tested).
   - precomputed orders (N cursor model npz), or a per-row mix of two sources.
2. Errors (`inject_errors`, optional): before writing token j, with probability `rate` a wrong token w is written in
   j's place first. Later w is deleted and the right token typed in its place:
     imm     right away:            w, DELETE, cont[j]
     delay   after k more tokens:   w, cont[..] x k, move back to just after w, DELETE, cont[j], (move on)
     runend  when the run ends:     the writer finishes the run, jumps back to each pending w, fixes it, jumps on
     final   at the very end:       all pending w are fixed after the last token, before EOR
   Because w stands in j's slot, cursor positions are the same as in the clean order; only the repair trips add
   moves. Wrong-token kinds: uniform (any GPT-2 id), unigram (corpus frequency), near (a token of the same text 1-D
   positions away, e.g. a repeated word), cand (a precomputed plausible candidate, e.g. sampled from GPT-2 at that
   position given the true left context). mask=True ignores the loss on w (as D2 in the toy suite).
3. Restoration with junk (`restoration_junk`): the original Reviser recipe (reversed random obfuscation walk:
   delete / move / insert-junk, historical code/src/reviser/data/obfuscator.py semantics: one power-of-2 move token
   per move, uniformly over valid amounts; junk uniform over ids 0..50255), with junk inserts > 0 so the restoration
   writes junk tokens and deletes them later (D1 = full loss, D2 = junk inserts masked).

`check_traj` replays an action stream (insert / delete / move semantics of the trainer's executor), checks it rebuilds
prefix + cont exactly and ends in EOR, and that the masked positions are exactly the doomed inserts (inserted tokens
that are deleted later) when masking is on, and none otherwise.
"""
from __future__ import annotations

import bisect
import random
from dataclasses import dataclass, field

from orders import decompose_move, move_token, run_lengths

GPT2_VOCAB = 50257
EOT = 50256
DELETE = 50259
EOR = 50260
MOVE_START = 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


# ----------------------------------------------------------------------------------------------------------------
# Replay and checks
# ----------------------------------------------------------------------------------------------------------------

def replay_items(actions: list[int]):
    """Replay with item identity. Returns (canvas tokens, doomed action indices, ok_moves).

    Semantics = trainer executor (resttraj_pretrain_collator._build_canvas_rows / build_textmatched.replay):
    insert at cursor, cursor += 1; DELETE pops the token left of the cursor (no-op at 0); moves clamp to [0, len].
    ok_moves is False if any move had to be clamped or a DELETE was a no-op (our generators never do that).
    """
    canvas: list[tuple[int, int]] = []   # (token, action index that inserted it)
    cur = 0
    doomed: set[int] = set()
    ok = True
    for i, t in enumerate(actions):
        t = int(t)
        if t < GPT2_VOCAB:
            canvas.insert(cur, (t, i))
            cur += 1
        elif MOVE_START <= t < MOVE_START + 20:
            off = t - MOVE_START
            new = cur + MOVE_AMOUNTS[off // 2] * (-1 if off % 2 else 1)
            if not 0 <= new <= len(canvas):
                ok = False
            cur = max(0, min(len(canvas), new))
        elif t == DELETE:
            if cur > 0:
                _, ai = canvas.pop(cur - 1)
                doomed.add(ai)
                cur -= 1
            else:
                ok = False
        elif t == EOR:
            break
        else:
            ok = False
    return [t for t, _ in canvas], doomed, ok


def check_traj(prefix: list[int], cont: list[int], actions: list[int], mask: list[int] | None, masked: bool) -> dict:
    """Return a dict of checks; all must be True for a valid trajectory."""
    P = len(prefix)
    canvas, doomed, ok_moves = replay_items(actions)
    mset = {i for i, v in enumerate(mask or []) if v}
    return {
        "replay_equals_text": canvas == list(prefix) + list(cont),
        "eor_end_and_prefix": bool(actions) and actions[-1] == EOR and list(actions[:P]) == list(prefix)
                              and EOR not in actions[:-1],
        "moves_in_range": ok_moves,
        "mask_len": mask is None or len(mask) == len(actions),
        "mask_equals_doomed": (mset == doomed) if masked else (len(mset) == 0),
        "prefix_untouched": all(i >= P for i in doomed),
        "n_doomed": len(doomed),
    }


def action_counts(actions: list[int], prefix_len: int) -> dict:
    c = {"insert": 0, "delete": 0, "move": 0, "back_move": 0}
    for t in actions[prefix_len:]:
        t = int(t)
        if t < GPT2_VOCAB:
            c["insert"] += 1
        elif t == DELETE:
            c["delete"] += 1
        elif MOVE_START <= t < MOVE_START + 20:
            c["move"] += 1
            c["back_move"] += (t - MOVE_START) % 2
    return c


def clean_len(prefix_len: int, order: list[int]) -> int:
    """len(order_to_actions(prefix, cont, order)) without building it."""
    _, jumps = run_lengths(order)
    return prefix_len + len(order) + sum(abs(d) // 512 + bin(abs(d) % 512).count("1") for d in jumps) + 1


# ----------------------------------------------------------------------------------------------------------------
# Orders
# ----------------------------------------------------------------------------------------------------------------

def runjump_order(pol, prefix: list[int], m: int, max_actions: int = 10 ** 6) -> list[int] | None:
    """Order drawn by a run-and-jump policy (gen_runjump.RunJumpPolicy or runjump6.RunJump6Policy).

    The policy is run on the continuation INDICES 0..m-1 instead of the token ids: its random draws depend only on
    positions, so the RNG stream and the order equal those of `pol.generate(prefix, cont, ...)`.
    """
    acts = pol.generate(list(prefix), list(range(m)), max_actions)
    if acts is None:
        return None
    order = [a for a in acts[len(prefix):-1] if a < GPT2_VOCAB]
    assert len(order) == m
    return order


# ----------------------------------------------------------------------------------------------------------------
# Errors: wrong token first, deleted later
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ErrorSpec:
    rate: float = 0.0          # chance per continuation token that a wrong token is written first
    kind: str = "unigram"      # uniform | unigram | near | cand
    when: str = "imm"          # imm | delay | runend | final
    delay_k: int = 3           # delay: repair after this many more tokens
    mask: bool = True          # ignore the loss on the wrong token
    near_d: int = 4            # near: pick from positions j-d..j+d (d >= 1)

    @staticmethod
    def from_dict(d: dict | None) -> "ErrorSpec | None":
        if not d:
            return None
        return ErrorSpec(**d)


@dataclass
class ErrorCounters:
    errors: int = 0
    cand_fallback: int = 0
    near_fallback: int = 0
    rows_thinned: int = 0
    rows_clean_fallback: int = 0
    redraws: int = 0
    extra: dict = field(default_factory=dict)


def wrong_token(spec: ErrorSpec, rng: random.Random, cont: list[int], j: int, cand, unigram, ctr: ErrorCounters) -> int:
    true = int(cont[j])

    def uni():
        if unigram is not None and len(unigram):
            for _ in range(50):
                t = int(unigram[rng.randrange(len(unigram))])
                if t != true and t != EOT:
                    return t
        while True:
            t = rng.randrange(EOT)          # 0..50255 (never <|endoftext|>), as the historical obfuscator
            if t != true:
                return t

    if spec.kind == "uniform":
        while True:
            t = rng.randrange(EOT)
            if t != true:
                return t
    if spec.kind == "unigram":
        return uni()
    if spec.kind == "near":
        d = max(1, int(spec.near_d))
        opts = [int(cont[i]) for i in range(max(0, j - d), min(len(cont), j + d + 1))
                if i != j and int(cont[i]) != true and int(cont[i]) != EOT]
        if opts:
            return opts[rng.randrange(len(opts))]
        ctr.near_fallback += 1
        return uni()
    if spec.kind == "cand":
        if cand is not None:
            t = int(cand[j])
            if t != true and t < EOT:
                return t
        ctr.cand_fallback += 1
        return uni()
    raise ValueError(spec.kind)


def inject_errors(prefix: list[int], cont: list[int], order: list[int], spec: ErrorSpec, rng: random.Random,
                  cand=None, unigram=None, ctr: ErrorCounters | None = None, rate: float | None = None):
    """(actions, mask) for writing cont in `order`, with wrong-token-then-delete events (see module docstring)."""
    ctr = ctr if ctr is not None else ErrorCounters()
    rate = spec.rate if rate is None else rate
    actions = list(int(x) for x in prefix)
    mask = [0] * len(actions)
    present: list[int] = []
    cursor = 0
    pending: list[list[int]] = []       # [j, due step]
    P = len(prefix)

    def emit(t: int, m: int = 0):
        actions.append(int(t))
        mask.append(m)

    def move_to(pos: int):
        nonlocal cursor
        for mt in decompose_move(pos - cursor):
            emit(mt)
        cursor = pos

    def repair(j: int):
        p = bisect.bisect_left(present, j)
        assert present[p] == j
        move_to(p + 1)                 # just right of the wrong token
        emit(DELETE)
        emit(int(cont[j]))             # cursor ends at p + 1 again

    wm = 1 if spec.mask else 0
    for t, j in enumerate(order):
        pos = bisect.bisect_left(present, j)
        if pending:
            if spec.when == "delay":
                due = [e for e in pending if e[1] <= t]
                for e in due:
                    repair(e[0])
                    pending.remove(e)
            elif spec.when == "runend" and pos != cursor:
                for e in pending:
                    repair(e[0])
                pending.clear()
        move_to(pos)
        if rate > 0 and rng.random() < rate:
            w = wrong_token(spec, rng, cont, j, cand, unigram, ctr)
            emit(w, wm)
            present.insert(pos, j)
            cursor = pos + 1
            ctr.errors += 1
            if spec.when == "imm":
                emit(DELETE)
                emit(int(cont[j]))
            else:
                pending.append([j, t + 1 + int(spec.delay_k)])
        else:
            emit(int(cont[j]))
            present.insert(pos, j)
            cursor = pos + 1
    for e in pending:
        repair(e[0])
    emit(EOR)
    assert len(actions) == len(mask) and P <= len(actions)
    return actions, mask


def errors_fit(prefix, cont, order, spec: ErrorSpec, rng: random.Random, max_len: int, retries: int,
               cand=None, unigram=None, ctr: ErrorCounters | None = None):
    """inject_errors redrawn (errors only; the order is fixed) until len <= max_len; then the error rate is halved
    for this row until it fits (counted as rows_thinned); finally no errors (rows_clean_fallback)."""
    ctr = ctr if ctr is not None else ErrorCounters()
    rate = spec.rate
    level = 0
    while True:
        for _ in range(retries + 1):
            before = ctr.errors
            acts, mask = inject_errors(prefix, cont, order, spec, rng, cand, unigram, ctr, rate=rate)
            if len(acts) <= max_len:
                if level > 0:
                    if rate > 0:
                        ctr.rows_thinned += 1
                    else:
                        ctr.rows_clean_fallback += 1
                return acts, mask
            ctr.errors = before
            ctr.redraws += 1
            if rate <= 0:
                raise RuntimeError("clean order does not fit max_len")
        level += 1
        rate = rate / 2 if rate > 0.004 else 0.0


# ----------------------------------------------------------------------------------------------------------------
# Restoration with junk (original Reviser recipe, D1/D2 style)
# ----------------------------------------------------------------------------------------------------------------

def restoration_junk(prefix: list[int], cont: list[int], rng: random.Random, p_del: float, p_move: float,
                     p_ins: float, mask: bool = True, start: str = "end", max_steps: int = 4000):
    """(actions, mask) of the reversed random obfuscation walk over cont (prefix untouched), or None if the walk
    exceeds max_steps. Obfuscation semantics follow code/src/reviser/data/obfuscator.py (historical): DELETE
    invalid at cursor 0 -> renormalise over move/insert; a move is one power-of-2 token chosen uniformly among the
    valid amounts; junk is uniform over 0..50255."""
    assert abs(p_del + p_move + p_ins - 1.0) < 1e-6
    canvas = list(int(x) for x in cont)
    cur = len(canvas) if start == "end" else rng.randint(0, len(canvas))
    steps: list[tuple[str, int]] = []
    while canvas:
        if len(steps) >= max_steps:
            return None
        if cur > 0:
            r = rng.random()
            a = "D" if r < p_del else ("M" if r < p_del + p_move else "I")
        else:
            tot = p_move + p_ins
            r = rng.random()
            a = "M" if r < p_move / tot else "I"
        if a == "D":
            steps.append(("D", canvas.pop(cur - 1)))
            cur -= 1
        elif a == "M":
            valid = []
            for amt in MOVE_AMOUNTS:
                if amt <= len(canvas) - cur:
                    valid.append(amt)
                if amt <= cur:
                    valid.append(-amt)
            if valid:
                amt = valid[rng.randrange(len(valid))]
                steps.append(("M", amt))
                cur += amt
        else:
            tok = rng.randint(0, EOT - 1)
            canvas.insert(cur, tok)
            steps.append(("I", tok))
            cur += 1
    acts = list(int(x) for x in prefix)
    for kind, v in reversed(steps):
        if kind == "I":
            acts.append(DELETE)
        elif kind == "D":
            acts.append(int(v))
        else:
            acts.append(move_token(-v))
    acts.append(EOR)
    m = [0] * len(acts)
    if mask:
        _, doomed, _ = replay_items(acts)
        for i in doomed:
            m[i] = 1
    return acts, m


def restoration_fit(prefix, cont, rng: random.Random, p_del, p_move, p_ins, mask: bool, max_len: int,
                    retries: int, ladder_counts: dict):
    """Fallback ladder for long texts: p_ins, p_ins/2, 0 (each `retries` draws), then left to right.
    The insert share removed from p_ins goes to delete/move in the original 80/20 ratio."""
    base_dm = p_del + p_move
    levels = []
    for pi in (p_ins, p_ins / 2, 0.0):
        if pi not in levels:
            levels.append(pi)
    for level, pi in enumerate(levels):
        pd = (1.0 - pi) * p_del / base_dm
        pm = (1.0 - pi) * p_move / base_dm
        for _ in range(retries + 1):
            r = restoration_junk(prefix, cont, rng, pd, pm, pi, mask=mask)
            if r is not None and len(r[0]) <= max_len:
                ladder_counts[str(level)] = ladder_counts.get(str(level), 0) + 1
                return r
    ladder_counts["l2r"] = ladder_counts.get("l2r", 0) + 1
    acts = list(int(x) for x in prefix) + [int(x) for x in cont] + [EOR]
    return acts, [0] * len(acts)
