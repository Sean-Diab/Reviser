"""Run-and-jump policy with ONE extra dial (p_fill): a modified COPY of July's RunJumpPolicy.

Source copied: experiments/july_control/code/july_impl/gen_runjump.py :: RunJumpPolicy (July, historical; that file is
NOT edited). Everything below is July's policy line for line, except the block marked "SIXTH DIAL".

Why a sixth dial: with July's five dials a run ends only with probability (1 - lambda_cont) per token, and at that
moment the cursor's own gap always has weight (1+0)^(-jump_alpha) = 1, so most run breaks become an in-place skip
(a new hole, no cursor move) rather than a short jump back to fill a recent hole. The 1,920-setting grid on the
CoAuthor train windows (runs/humanorder/fit/grid1.json) could not get human run lengths (median 11) together with
human short, balanced jumps (median 5, back share 0.53): best fit had run median 2 and back share 0.85.

SIXTH DIAL p_fill: before each run-continue step, if some gap other than the cursor's own gap exists, with
probability p_fill the run is broken and the policy jumps to one of those OTHER gaps (same (1+d)^(-jump_alpha)
distance weights, same entry rule: p_skip / skip_q, p_far for the tail gap). The random number for p_fill is drawn
only when p_fill > 0 and another gap exists, so p_fill = 0 reproduces July's policy draw-for-draw (tested in
scripts/humanorder/runjump6.py --selftest).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "july_control" / "code" / "july_impl"))
from gen_runjump import EOR_ID, decompose_move  # noqa: E402  (July, unmodified helpers)


class RunJump6Policy:
    def __init__(self, lambda_cont: float, jump_alpha: float, p_skip: float, skip_q: float,
                 rng: random.Random, p_far: float = 0.0, p_fill: float = 0.0):
        self.lambda_cont = lambda_cont
        self.jump_alpha = jump_alpha
        self.p_skip = p_skip
        self.skip_q = skip_q
        self.p_far = p_far
        self.p_fill = p_fill
        self.rng = rng

    def generate(self, prefix: list[int], cont: list[int], max_actions: int) -> list[int] | None:
        m = len(cont)
        actions: list[int] = list(prefix)
        present: list[int] = []
        inserted = [False] * m
        n_inserted = 0
        cursor = 0
        budget = max_actions

        def gaps() -> list[tuple[int, int, int]]:
            out = []
            prev_t = -1
            for pos, t in enumerate(present):
                if t - prev_t > 1:
                    out.append((prev_t + 1, t - prev_t - 1, pos))
                prev_t = t
            if m - 1 - prev_t > 0:
                out.append((prev_t + 1, m - 1 - prev_t, len(present)))
            return out

        def cursor_gap_next() -> int | None:
            left_t = present[cursor - 1] if cursor > 0 else -1
            right_t = present[cursor] if cursor < len(present) else m
            nxt = left_t + 1
            if nxt < right_t and nxt < m and not inserted[nxt]:
                return nxt
            return None

        def own_gap_missing() -> int:
            left_t = present[cursor - 1] if cursor > 0 else -1
            right_t = present[cursor] if cursor < len(present) else m
            return right_t - left_t - 1

        def pick(gs, exclude_own: bool):
            ws = []
            cursor_gap_i = None
            for i, (gt, gl, bpos) in enumerate(gs):
                d = abs(bpos - cursor)
                if bpos == cursor:
                    cursor_gap_i = i
                w = (1.0 + d) ** (-self.jump_alpha)
                if exclude_own and bpos == cursor:
                    w = 0.0
                ws.append(w)
            tot = sum(ws)
            r = self.rng.random() * tot
            acc = 0.0
            gi = max(i for i, w in enumerate(ws) if w > 0.0) if exclude_own else 0   # July: default 0
            for i, w in enumerate(ws):
                acc += w
                if r <= acc and w > 0.0:
                    gi = i
                    break
            return gi, cursor_gap_i

        while n_inserted < m:
            if len(actions) + 1 >= budget:
                return None
            nxt = cursor_gap_next()
            # ---- SIXTH DIAL (the only change to July's policy) ----
            fill_jump = False
            if (self.p_fill > 0.0 and nxt is not None and (m - n_inserted) > own_gap_missing()
                    and self.rng.random() < self.p_fill):
                fill_jump = True
            # --------------------------------------------------------
            if not fill_jump and nxt is not None and self.rng.random() < self.lambda_cont:
                actions.append(int(cont[nxt]))
                present.insert(cursor, nxt)
                inserted[nxt] = True
                n_inserted += 1
                cursor += 1
            else:
                gs = gaps()
                if not gs:
                    break
                gi, cursor_gap_i = pick(gs, exclude_own=fill_jump)
                gt, gl, bpos = gs[gi]
                s = 0
                is_tail_gap = (gt + gl == m)
                if is_tail_gap and gl > 1 and self.rng.random() < self.p_far:
                    s = self.rng.randint(0, gl - 1)
                else:
                    force_skip = (gi == cursor_gap_i and gl > 1)
                    if force_skip or (gl > 1 and self.rng.random() < self.p_skip):
                        s = 1
                        while s < gl - 1 and self.rng.random() < self.skip_q:
                            s += 1
                delta = bpos - cursor
                for mt in decompose_move(delta):
                    actions.append(mt)
                    if len(actions) + 1 >= budget:
                        return None
                cursor = bpos
                tidx = gt + s
                actions.append(int(cont[tidx]))
                present.insert(cursor, tidx)
                inserted[tidx] = True
                n_inserted += 1
                cursor += 1

        actions.append(EOR_ID)
        assert len(present) == m and all(present[i] == i for i in range(m)), "executor invariant violated"
        return actions


def _selftest(n: int = 300) -> None:
    from gen_runjump import RunJumpPolicy
    d = {"lambda_cont": 0.82, "jump_alpha": 1.5, "p_skip": 0.3, "skip_q": 0.85, "p_far": 0.9}
    r0, r1 = random.Random(7), random.Random(7)
    a = RunJumpPolicy(d["lambda_cont"], d["jump_alpha"], d["p_skip"], d["skip_q"], r0, p_far=d["p_far"])
    b = RunJump6Policy(rng=r1, p_fill=0.0, **d)
    rr = random.Random(1)
    for _ in range(n):
        m = rr.randint(65, 257)
        cont = [rr.randint(0, 50000) for _ in range(m)]
        pre = [rr.randint(0, 50000) for _ in range(35)]
        assert a.generate(pre, cont, 100000) == b.generate(pre, cont, 100000)
    c = RunJump6Policy(rng=random.Random(3), p_fill=0.3, **d)
    for _ in range(n):
        m = rr.randint(65, 257)
        cont = [rr.randint(0, 50000) for _ in range(m)]
        assert c.generate([1] * 35, cont, 100000) is not None   # internal replay assert
    print(f"selftest ok: p_fill=0 identical to July on {n} texts; p_fill=0.3 valid on {n} texts")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
