"""Trajectory construction in an encoding-independent form, then compilation to action tokens.

Abstract ops (tuples):
  ("ins", tok)   insert a content token at the cursor
  ("del",)       delete the token left of the cursor
  ("goto", pos)  put the cursor at editable boundary ``pos`` (compiled per move encoding)
  ("stop",)

Keeping edit/content order abstract lets the same trajectory be compiled into pow2/dense/absolute
moves, so navigation representations are compared on identical edit orders (task 17).

Generators:
  * ``restoration_ops``  time-reversal of a random obfuscation walk (historical Reviser recipe).
                          Obfuscation Insert(junk) becomes a restoration Delete, so ``p_ins > 0``
                          produces doomed insertions (D1/D2 data). ``obf_forward_only_moves``
                          yields backward-only restoration trajectories (task 18).
  * ``runjump_ops``      port of the July forward run-and-jump policy (task 19). Source:
                          <archive>/workspaces/7-3/impl/gen_runjump.py
                          sha256 76e337c3a00a4a44539869c7b7604e88e85fa91ec9af31c99f10393f00deaceb
  * ``repair_ops``       draft -> target edit script from a token diff (D3 / real revision pairs).
  * ``ltr_ops``          plain left-to-right insertion (AR reduction / AR rewrite supervision).
"""
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .actions import ActionSpace
from .executor import Canvas, doomed_insert_positions, replay

Op = Tuple


# ---------------------------------------------------------------------------
# Compilation and validation
# ---------------------------------------------------------------------------
def compile_ops(space: ActionSpace, ops: Sequence[Op], draft_len: int = 0, cursor: Optional[int] = None) -> List[int]:
    """Compile abstract ops to action token ids, tracking cursor/length to resolve gotos."""
    cur = draft_len if cursor is None else cursor
    length = draft_len
    out: List[int] = []
    for op in ops:
        k = op[0]
        if k == "ins":
            out.append(int(op[1]))
            cur += 1
            length += 1
        elif k == "del":
            if not space.allow_delete:
                raise ValueError("delete op but action space disallows delete")
            out.append(space.DELETE)
            cur -= 1
            length -= 1
        elif k == "goto":
            dst = int(op[1])
            if not 0 <= dst <= length:
                raise ValueError(f"goto {dst} outside [0,{length}]")
            out.extend(space.compile_goto(cur, dst))
            cur = dst
        elif k == "stop":
            out.append(space.STOP)
        else:
            raise ValueError(f"unknown op {op}")
    return out


def validate_actions(space: ActionSpace, actions: Sequence[int], target: Sequence[int], draft: Sequence[int] = (), cursor: Optional[int] = None) -> Tuple[bool, str]:
    """Replay; require exact target and a single terminal STOP."""
    try:
        c = replay(space, actions, draft, cursor)
    except Exception as e:  # IllegalAction or ValueError
        return False, f"illegal: {e}"
    if not c.stopped:
        return False, "no STOP"
    if c.tokens != list(target):
        return False, "final canvas != target"
    return True, "ok"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def ltr_ops(target: Sequence[int]) -> List[Op]:
    return [("ins", t) for t in target] + [("stop",)]


@dataclass
class ObfuscationPolicy:
    """Random obfuscation walk (applied to the clean target) whose reversal is a restoration.

    Probabilities are for obfuscation ops: delete-left (-> restoration Insert), insert junk
    (-> restoration Delete), move (-> restoration move back).
    Historical paper policy: p_del=0.8, p_move=0.2, p_ins=0. Release defaults: 0.70/0.18/0.12.
    """
    p_del: float = 0.8
    p_move: float = 0.2
    p_ins: float = 0.0
    move_len_alpha: float = 1.5      # P(distance d) ~ d^-alpha over reachable distances
    forward_only_moves: bool = False  # obfuscation moves only right => restoration moves only left
    start_cursor: str = "end"         # "end" or "random": obfuscation start = restoration final cursor
    max_steps: int = 100000


def _sample_distance(rng: random.Random, dmax: int, alpha: float) -> int:
    ws = [(d ** -alpha) for d in range(1, dmax + 1)]
    r = rng.random() * sum(ws)
    acc = 0.0
    for d, w in enumerate(ws, start=1):
        acc += w
        if r <= acc:
            return d
    return dmax


def restoration_ops(target: Sequence[int], rng: random.Random, pol: ObfuscationPolicy,
                    junk_sampler: Callable[[random.Random, List[int], int], int]) -> List[Op]:
    """Blank-start restoration trajectory = reversed obfuscation walk from ``target`` to empty."""
    canvas = list(target)
    cur = len(canvas) if pol.start_cursor == "end" else rng.randint(0, len(canvas))
    start_cursor = cur
    rev: List[Op] = []  # restoration ops in reverse order
    steps = 0
    tot = pol.p_del + pol.p_move + pol.p_ins
    while canvas:
        steps += 1
        if steps > pol.max_steps:
            raise RuntimeError("obfuscation walk did not terminate")
        r = rng.random() * tot
        L = len(canvas)
        if r < pol.p_ins:
            op = "ins"
        elif r < pol.p_ins + pol.p_move:
            op = "move"
        else:
            op = "del"
        if op == "del" and cur == 0:
            op = "move"
        if op == "move":
            if pol.forward_only_moves:
                if cur == L:
                    op = "del"  # cur == L > 0 so delete is legal
                else:
                    d = _sample_distance(rng, L - cur, pol.move_len_alpha)
                    rev.append(("goto", cur))
                    cur += d
                    continue
            else:
                dirs = []
                if cur > 0:
                    dirs.append(-1)
                if cur < L:
                    dirs.append(1)
                s = rng.choice(dirs)
                dmax = cur if s < 0 else L - cur
                d = _sample_distance(rng, dmax, pol.move_len_alpha)
                rev.append(("goto", cur))
                cur += s * d
                continue
        if op == "del":
            tok = canvas.pop(cur - 1)
            cur -= 1
            rev.append(("ins", tok))
        elif op == "ins":
            j = junk_sampler(rng, canvas, cur)
            canvas.insert(cur, j)
            cur += 1
            rev.append(("del",))
    ops = list(reversed(rev))
    # restoration starts at empty canvas, cursor 0; gotos in `rev` store the pre-move cursor,
    # which is exactly the restoration's post-move cursor. Final cursor == start_cursor.
    ops.append(("stop",))
    return ops


def uniform_junk(n_content: int, exclude: Sequence[int] = ()) -> Callable:
    ex = set(exclude)

    def f(rng: random.Random, canvas: List[int], cur: int) -> int:
        while True:
            t = rng.randrange(n_content)
            if t not in ex:
                return t
    return f


@dataclass
class RunJumpPolicy:
    """July forward run-and-jump policy (insert + move + stop), ported to abstract ops.

    Tuned July parameters: lambda_cont=0.82, jump_alpha=1.5, p_skip=0.3, skip_q=0.85, p_far=0.9.
    """
    lambda_cont: float = 0.82
    jump_alpha: float = 1.5
    p_skip: float = 0.3
    skip_q: float = 0.85
    p_far: float = 0.9


def runjump_ops(cont: Sequence[int], rng: random.Random, pol: RunJumpPolicy) -> List[Op]:
    m = len(cont)
    ops: List[Op] = []
    present: List[int] = []
    inserted = [False] * m
    n_ins = 0
    cursor = 0

    def gaps():
        out = []
        prev_t = -1
        for pos, t in enumerate(present):
            if t - prev_t > 1:
                out.append((prev_t + 1, t - prev_t - 1, pos))
            prev_t = t
        if m - 1 - prev_t > 0:
            out.append((prev_t + 1, m - 1 - prev_t, len(present)))
        return out

    def cursor_gap_next():
        left_t = present[cursor - 1] if cursor > 0 else -1
        right_t = present[cursor] if cursor < len(present) else m
        nxt = left_t + 1
        if nxt < right_t and nxt < m and not inserted[nxt]:
            return nxt
        return None

    while n_ins < m:
        nxt = cursor_gap_next()
        if nxt is not None and rng.random() < pol.lambda_cont:
            ops.append(("ins", cont[nxt]))
            present.insert(cursor, nxt)
            inserted[nxt] = True
            n_ins += 1
            cursor += 1
            continue
        gs = gaps()
        ws, cursor_gap_i = [], None
        for i, (gt, gl, bpos) in enumerate(gs):
            if bpos == cursor:
                cursor_gap_i = i
            ws.append((1.0 + abs(bpos - cursor)) ** (-pol.jump_alpha))
        r = rng.random() * sum(ws)
        acc, gi = 0.0, 0
        for i, w in enumerate(ws):
            acc += w
            if r <= acc:
                gi = i
                break
        gt, gl, bpos = gs[gi]
        s = 0
        if gt + gl == m and gl > 1 and rng.random() < pol.p_far:
            s = rng.randint(0, gl - 1)
        else:
            if (gi == cursor_gap_i and gl > 1) or (gl > 1 and rng.random() < pol.p_skip):
                s = 1
                while s < gl - 1 and rng.random() < pol.skip_q:
                    s += 1
        if bpos != cursor:
            ops.append(("goto", bpos))
        cursor = bpos
        tidx = gt + s
        ops.append(("ins", cont[tidx]))
        present.insert(cursor, tidx)
        inserted[tidx] = True
        n_ins += 1
        cursor += 1
    ops.append(("stop",))
    assert present == list(range(m))
    return ops


def diff_hunks(draft: Sequence[int], target: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    """Non-equal (i1, i2, j1, j2) hunks: replace draft[i1:i2] by target[j1:j2]."""
    sm = difflib.SequenceMatcher(a=list(draft), b=list(target), autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def repair_ops(draft: Sequence[int], target: Sequence[int], order: str = "rtl",
               start_cursor: Optional[int] = None) -> List[Op]:
    """Edit script turning ``draft`` into ``target``: per hunk goto end, delete-left, insert.

    ``rtl`` processes hunks right-to-left (earlier coordinates stay valid); ``ltr`` processes
    left-to-right with an explicit running offset. Both are declared synthetic orders, not
    recovered human keystrokes.
    """
    hunks = diff_hunks(draft, target)
    ops: List[Op] = []
    if order == "rtl":
        for i1, i2, j1, j2 in reversed(hunks):
            ops.append(("goto", i2))
            ops.extend([("del",)] * (i2 - i1))
            ops.extend(("ins", t) for t in target[j1:j2])
    elif order == "ltr":
        off = 0
        for i1, i2, j1, j2 in hunks:
            ops.append(("goto", i2 + off))
            ops.extend([("del",)] * (i2 - i1))
            ops.extend(("ins", t) for t in target[j1:j2])
            off += (j2 - j1) - (i2 - i1)
    else:
        raise ValueError(order)
    ops.append(("stop",))
    return _drop_redundant_gotos(ops, len(draft), start_cursor)


def _drop_redundant_gotos(ops: List[Op], draft_len: int, start_cursor: Optional[int] = None) -> List[Op]:
    """Remove gotos to the current cursor (compile_ops would emit nothing, but keep ops clean).

    ``start_cursor`` is the episode's initial cursor (default: end of draft); dropping must track it,
    otherwise a needed first goto is removed when a seed cursor is supplied."""
    out = []
    cur, length = (draft_len if start_cursor is None else start_cursor), draft_len
    for op in ops:
        if op[0] == "goto":
            if op[1] == cur:
                continue
            cur = op[1]
        elif op[0] == "ins":
            cur += 1
            length += 1
        elif op[0] == "del":
            cur -= 1
            length -= 1
        out.append(op)
    return out


# ---------------------------------------------------------------------------
# Serialization into a training sequence
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    prompt: List[int]           # protected conditioning (never on the canvas)
    draft: List[int]            # supplied editable draft (may be empty)
    seed_cursor: Optional[int]  # initial cursor inside draft; None = end of draft
    actions: List[int]          # compiled model actions (ends with STOP)
    target: List[int]
    doomed: List[int]           # indices into actions of doomed inserts


def build_episode(space: ActionSpace, prompt: Sequence[int], draft: Sequence[int], target: Sequence[int],
                  ops: Sequence[Op], seed_cursor: Optional[int] = None) -> Episode:
    acts = compile_ops(space, ops, len(draft), seed_cursor)
    ok, why = validate_actions(space, acts, target, draft, seed_cursor)
    if not ok:
        raise ValueError(f"invalid trajectory: {why}")
    doomed = doomed_insert_positions(space, acts, draft, seed_cursor)
    return Episode(list(prompt), list(draft), seed_cursor, acts, list(target), doomed)


def conditioning_ids(space: ActionSpace, prompt: Sequence[int], draft: Sequence[int], seed_cursor: Optional[int] = None) -> List[int]:
    """[prompt] SEP [draft] [seed-cursor moves] EDIT. The draft is seeded as if typed, then the
    cursor is optionally moved; all of this is conditioning (no loss)."""
    ids = list(prompt) + [space.SEP] + list(draft)
    if seed_cursor is not None and seed_cursor != len(draft):
        ids += space.compile_goto(len(draft), seed_cursor)
    ids.append(space.EDIT)
    return ids


def episode_to_sequence(space: ActionSpace, ep: Episode, mask_doomed: bool) -> Tuple[List[int], List[int], List[int]]:
    """Return (input_ids, labels, eligible_labels).

    ``labels[t]`` is the target for position t (next token); conditioning targets are -100.
    ``eligible_labels`` are the labels before the doomed-insert mask, so D1/D2 can share the
    primary denominator (count of eligible action targets).
    """
    cond = conditioning_ids(space, ep.prompt, ep.draft, ep.seed_cursor)
    ids = cond + ep.actions
    n = len(ids)
    labels = [-100] * n
    eligible = [-100] * n
    base = len(cond) - 1  # position whose next token is actions[0]
    doomed = set(ep.doomed) if mask_doomed else set()
    for k, a in enumerate(ep.actions):
        pos = base + k
        eligible[pos] = a
        labels[pos] = -100 if k in doomed else a
    # the final STOP is an input with no successor; its own position predicts nothing
    return ids[:-1], labels[:-1], eligible[:-1]
