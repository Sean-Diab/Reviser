"""July's run-and-jump dials (and the Q agent's six-dial variant) as an order model: sample and score orders exactly.

`RunJumpPolicy` (experiments/july_control/code/july_impl/gen_runjump.py, unchanged) emits an action stream; the order
over continuation tokens is recovered by replaying the actions (each insert gets an event id; the final canvas lists
the event ids in text order). `runjump_logprobs` computes the probability the policy assigns to each pick of a given
order by enumerating the policy's branches (continue / jump -> gap choice -> far anchor / skip length) in the same
order as the code; `tests/test_cursor_model.py` checks it against Monte Carlo frequencies. Dials with p_fill > 0 use
the Q agent's `runjump6.RunJump6Policy` (July's policy + one dial: before a run-continue step, with probability p_fill
jump to a gap other than the cursor's own).
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/july_control/code/july_impl"))
from gen_runjump import EOR_ID, MOVE_AMOUNTS, MOVE_START, RunJumpPolicy  # noqa: E402

JULY_DIALS = {"lambda_cont": 0.82, "jump_alpha": 1.5, "p_skip": 0.3, "skip_q": 0.85, "p_far": 0.9}
DIAL_KEYS = ("lambda_cont", "jump_alpha", "p_skip", "skip_q", "p_far")
KNOWN_KEYS = DIAL_KEYS + ("p_fill",)


def load_dials(path):
    """Dials from a JSON file: either flat {lambda_cont: ...} or nested under 'dials'/'params'."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for k in ("dials", "params", "fitted"):
        if isinstance(d.get(k), dict):
            d = d[k]
            break
    unknown = sorted(k for k in d if k not in KNOWN_KEYS and not isinstance(d[k], (dict, list, str)))
    if unknown:
        raise SystemExit(f"unsupported dials {unknown} in {path}")
    return {k: float(d[k]) for k in KNOWN_KEYS if k in d}, unknown


def actions_to_order(actions: list[int], prefix_len: int) -> list[int]:
    canvas, cursor, ev = [], 0, 0
    for t in actions[prefix_len:]:
        if t == EOR_ID:
            break
        if MOVE_START <= t < MOVE_START + 2 * len(MOVE_AMOUNTS):
            off = t - MOVE_START
            cursor += MOVE_AMOUNTS[off // 2] * (-1 if off % 2 else 1)
        else:
            canvas.insert(cursor, ev)
            cursor += 1
            ev += 1
    order = [0] * len(canvas)
    for k, e in enumerate(canvas):
        order[e] = k
    return order


def runjump_order(dials: dict, prefix: list[int], cont: list[int], rng: random.Random) -> list[int]:
    if dials.get("p_fill", 0.0) > 0.0:
        from runjump6 import RunJump6Policy
        pol = RunJump6Policy(dials["lambda_cont"], dials["jump_alpha"], dials["p_skip"], dials["skip_q"], rng,
                             p_far=dials["p_far"], p_fill=dials["p_fill"])
    else:
        pol = RunJumpPolicy(dials["lambda_cont"], dials["jump_alpha"], dials["p_skip"], dials["skip_q"], rng,
                            p_far=dials["p_far"])
    acts = pol.generate(list(prefix), list(cont), 10 ** 9)
    order = actions_to_order(acts, len(prefix))
    assert sorted(order) == list(range(len(cont)))
    return order


def _skip_prob(s: int, gl: int, q: float) -> float:
    """P(entry offset s | skip branch) for the loop s=1; while s < gl-1 and rand < q: s += 1."""
    if s < 1 or s > gl - 1:
        return 0.0
    if s < gl - 1:
        return q ** (s - 1) * (1 - q)
    return q ** (gl - 2)


def runjump_logprobs(order: list[int], dials: dict) -> list[float]:
    """log P(pick_t | state_t) under the (six-dial) run-and-jump policy for every step of a given order.

    P(j) = (1 - P_fill) * [lam * 1{j = nxt} + (1 - lam) * J(j, own gap allowed)]   (nxt exists)
         + P_fill * J(j, own gap excluded),     P_fill = p_fill if nxt exists and tokens are missing outside the
    cursor's own gap, else 0; without nxt the policy always jumps with the own gap allowed.
    """
    lam, alpha, p_skip, q, p_far = (dials[k] for k in DIAL_KEYS)
    p_fill = dials.get("p_fill", 0.0)
    m = len(order)
    present: list[int] = []
    cursor = 0
    out = []

    def jump_prob(j, gaps, exclude_own):
        ws = [(1.0 + abs(b - cursor)) ** (-alpha) for _, _, b in gaps]
        cg = next((i for i, (_, _, b) in enumerate(gaps) if b == cursor), None)
        if exclude_own and cg is not None:
            ws[cg] = 0.0
        tot = sum(ws)
        gi = next(i for i, (gt, gl, _) in enumerate(gaps) if gt <= j < gt + gl)
        if ws[gi] == 0.0:
            return 0.0
        gt, gl, _ = gaps[gi]
        s = j - gt
        within, rest = 0.0, 1.0
        if gt + gl == m and gl > 1:
            within += p_far / gl
            rest = 1.0 - p_far
        if gi == cg and gl > 1:
            within += rest * _skip_prob(s, gl, q)
        elif gl > 1:
            within += rest * (p_skip * _skip_prob(s, gl, q) + (1 - p_skip) * (s == 0))
        else:
            within += rest * (s == 0)
        return ws[gi] / tot * within

    n_ins = 0
    for j in order:
        left_t = present[cursor - 1] if cursor > 0 else -1
        right_t = present[cursor] if cursor < len(present) else m
        nxt = left_t + 1 if (left_t + 1 < right_t and left_t + 1 < m) else None
        gaps = []
        prev_t = -1
        for pos, t in enumerate(present):
            if t - prev_t > 1:
                gaps.append((prev_t + 1, t - prev_t - 1, pos))
            prev_t = t
        if m - 1 - prev_t > 0:
            gaps.append((prev_t + 1, m - 1 - prev_t, len(present)))
        if nxt is not None:
            pf = p_fill if (p_fill > 0.0 and (m - n_ins) > right_t - left_t - 1) else 0.0
            p = (1.0 - pf) * (lam * (j == nxt) + (1.0 - lam) * jump_prob(j, gaps, False))
            if pf > 0.0:
                p += pf * jump_prob(j, gaps, True)
        else:
            p = jump_prob(j, gaps, False)
        out.append(math.log(p) if p > 0 else float("-inf"))
        pos = bisect.bisect_left(present, j)
        present.insert(pos, j)
        cursor = pos + 1
        n_ins += 1
    return out
