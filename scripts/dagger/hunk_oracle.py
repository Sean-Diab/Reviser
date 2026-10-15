"""The user's hunk oracle (Jan-Apr 2026) ported to the classical Reviser's July vocabulary.

Source (read-only, unedited): <archive>/other/transformer_files/dagger/
  diff.py    sha256 c1bb682a2efd6fc92be0e8fa82511b3fafac79a658720d5c7e1e0d3011e8dd15
  oracle.py  sha256 6cae1b3be7f0c66c7ea20165503079667edc29b413512ce825745a5c3483407b
The algorithm and weights are copied as written. Changes in this port (each marked "PORT:" below):
  1. Token ids are the July / release-checkpoint ids (GPT-2 50257 + 34 specials: DELETE 50259, EOR 50260,
     moves 50261.. as +1,-1,+2,-2,...,+512,-512). No dependency on the old `data` package.
  2. The original returned END_OF_RESPONSE as a fallback when it produced no action ("avoid deadlock"). Here EOR is
     only ever returned when canvas == target; an empty action set raises instead (it cannot happen: see tests).
  3. `oracle_labels()` wraps the half-index API with the integer cursor used by the Reviser executor (0..len).
  4. `normalized()` / `oracle_distribution()` give a normalized soft label (the original loss used raw weights).
  5. Optional alignment `OracleConfig(align="lcs")`: a longest-common-subsequence alignment instead of difflib's
     longest-block heuristic (see `_blocks_lcs`); default stays "difflib" (the original).
  6. Optional insert rule `OracleConfig(insert_rule="count")` (see `_weighted_inserts`); default "proportion".

How it works (unchanged):
  * diff canvas vs target into mismatch regions ("hunks") with difflib.SequenceMatcher(autojunk=False)
    (deterministic; equal tokens are aligned by its longest-matching-block rule);
  * canvas == target -> EOR (weight 1), and only then;
  * cursor outside every hunk -> move toward the closest hunk(s) (fewest move tokens = popcount of the distance;
    ties prefer the rightmost landing position); each distinct move token of the greedy decomposition gets weight
    1/len(decomposition);
  * cursor inside a hunk ->
      - inserts of the hunk's missing target tokens, weighted by where the cursor sits in the hunk (tokens up to that
        proportion weight 1, then up to 4 more with decay m; m = 0 for hunks of canvas length 1-4, else 0.5 -- this
        includes pure-insertion hunks, canvas length 0, so at the writing end the next token has weight 1 and the
        next four 0.5, 0.25, 0.125, 0.0625 -- i.e. soft "skip ahead" options);
      - DELETE (weight 1) when the cursor is not at the hunk's left edge and the hunk has canvas tokens (the token
        left of the cursor is then a mismatched canvas token);
      - at the left edge: move-right options that stay in the hunk (largest power weight 1, then x0.3 each), and,
        if no target tokens are missing, the full move to the hunk's right edge (weight 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

# ---- PORT: July / release vocabulary (experiments/july_control/code/src/reviser/data/vocabulary.py) ----
GPT2_VOCAB = 50257
GPT2_EOT = 50256
CURSOR, EOI, DELETE, EOR = 50257, 50258, 50259, 50260
MOVE_START = 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_MOVES = 20
VOCAB_SIZE = 50291


def move_id(amount: int) -> int:
    a = abs(int(amount))
    idx = MOVE_AMOUNTS.index(a)
    return MOVE_START + 2 * idx + (0 if amount > 0 else 1)


def move_amount(tid: int) -> int:
    off = int(tid) - MOVE_START
    if not 0 <= off < N_MOVES:
        raise ValueError(f"not a move token: {tid}")
    a = MOVE_AMOUNTS[off // 2]
    return -a if off % 2 else a


def is_move(tid: int) -> bool:
    return MOVE_START <= int(tid) < MOVE_START + N_MOVES


def is_insert(tid: int) -> bool:
    return 0 <= int(tid) < GPT2_VOCAB and int(tid) != GPT2_EOT


def decompose_move(amount: int) -> List[int]:
    """Greedy largest-first, same sign (vocabulary.decompose_move)."""
    if amount == 0:
        return []
    sign = 1 if amount > 0 else -1
    rem, out = abs(int(amount)), []
    for p in reversed(MOVE_AMOUNTS):
        while rem >= p:
            out.append(move_id(sign * p))
            rem -= p
    return out


class State:
    """Reviser executor (same semantics as the trainer's canvas replay and the decoder's CursorState).

    lock (default 0 = the original executor; notes/promptlock-2026-10-08.md): protect the first `lock` canvas tokens
    (the prompt). valid(): a move must land in [lock, len] (landing exactly on lock is legal) and DELETE needs
    cursor > lock. step() clamps moves to [lock, len] and ignores DELETE at cursor <= lock, so even an unchecked
    action cannot change the prompt. lock = 0 gives exactly the original rules."""

    __slots__ = ("canvas", "cursor", "lock")

    def __init__(self, canvas: Optional[List[int]] = None, cursor: int = 0, lock: int = 0):
        self.canvas = list(canvas or [])
        self.cursor = int(cursor)
        self.lock = int(lock)

    def copy(self) -> "State":
        return State(self.canvas, self.cursor, self.lock)

    def valid(self, tid: int) -> bool:
        tid = int(tid)
        if tid == DELETE:
            return self.cursor > self.lock
        if is_move(tid):
            return self.lock <= self.cursor + move_amount(tid) <= len(self.canvas)
        return is_insert(tid) or tid == EOR

    def step(self, tid: int) -> None:
        tid = int(tid)
        if tid == DELETE:
            if self.cursor > self.lock:
                self.canvas.pop(self.cursor - 1)
                self.cursor -= 1
        elif is_move(tid):
            self.cursor = max(self.lock, min(len(self.canvas), self.cursor + move_amount(tid)))
        elif is_insert(tid):
            self.canvas.insert(self.cursor, tid)
            self.cursor += 1
        # EOR / other: no state change


def replay(actions: Sequence[int]) -> State:
    st = State()
    for t in actions:
        if int(t) == EOR:
            break
        st.step(t)
    return st


# ---------------------------------------------------------------------------------------------------------------
# diff.py (copied)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hunk:
    """Mismatch region; boundaries in half-index space (-0.5, 0.5, ..., len-0.5), i.e. between tokens."""

    canvas_l: float
    canvas_r: float
    target_l: float
    target_r: float

    def canvas_bounds_int(self) -> Tuple[int, int]:
        return int(round(self.canvas_l + 0.5)), int(round(self.canvas_r + 0.5))

    def target_bounds_int(self) -> Tuple[int, int]:
        return int(round(self.target_l + 0.5)), int(round(self.target_r + 0.5))

    def cursor_in_hunk(self, cursor_pos_half: float) -> bool:
        return self.canvas_l <= float(cursor_pos_half) <= self.canvas_r

    @property
    def canvas_len(self) -> int:
        l, r = self.canvas_bounds_int()
        return r - l

    @property
    def target_len(self) -> int:
        l, r = self.target_bounds_int()
        return r - l


def _blocks_difflib(canvas: List[int], target: List[int]) -> List[Tuple[int, int, int]]:
    return [tuple(b) for b in SequenceMatcher(a=canvas, b=target, autojunk=False).get_matching_blocks()]


def _blocks_lcs(canvas: List[int], target: List[int]) -> List[Tuple[int, int, int]]:
    """PORT addition: matching blocks of a longest common subsequence (insert/delete edit distance).

    difflib's longest-block heuristic is deterministic but not minimal: with repeated tokens it can mark a correct
    canvas token as junk even when the canvas is a subsequence of the target (canvas [1,1,2], target [1,2,1,2]:
    difflib matches [1,2] and calls the first 1 junk). Measured on clean July partial states of real text: 0.2% of
    states get such false junk (a "delete a correct token" label). With an LCS alignment a canvas that is a
    subsequence of the target never has junk. Deterministic traceback from the end, preferring a match, then
    dropping a canvas token, then a target token. Common prefix/suffix are matched first (they are part of some LCS).
    """
    import numpy as np

    n0, m0 = len(canvas), len(target)
    p = 0
    while p < n0 and p < m0 and canvas[p] == target[p]:
        p += 1
    s = 0
    while s < n0 - p and s < m0 - p and canvas[n0 - 1 - s] == target[m0 - 1 - s]:
        s += 1
    a, b = canvas[p:n0 - s], target[p:m0 - s]
    n, m = len(a), len(b)
    pairs: List[Tuple[int, int]] = [(i, i) for i in range(p)]
    if n and m:
        bb = np.asarray(b, dtype=np.int64)
        L = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(1, n + 1):
            eq = (bb == a[i - 1]).astype(np.int32)
            c = np.maximum(L[i - 1, 1:], L[i - 1, :-1] + eq)
            L[i, 1:] = np.maximum.accumulate(c)
        Lt = L.tolist()
        i, j, mid = n, m, []
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1] and Lt[i][j] == Lt[i - 1][j - 1] + 1:
                mid.append((p + i - 1, p + j - 1))
                i, j = i - 1, j - 1
            elif Lt[i - 1][j] >= Lt[i][j - 1]:
                i -= 1
            else:
                j -= 1
        pairs.extend(reversed(mid))
    pairs.extend((n0 - s + k, m0 - s + k) for k in range(s))
    blocks: List[Tuple[int, int, int]] = []
    for ai, bi in pairs:
        if blocks and blocks[-1][0] + blocks[-1][2] == ai and blocks[-1][1] + blocks[-1][2] == bi:
            blocks[-1] = (blocks[-1][0], blocks[-1][1], blocks[-1][2] + 1)
        else:
            blocks.append((ai, bi, 1))
    blocks.append((n0, m0, 0))
    return blocks


ALIGN = {"difflib": _blocks_difflib, "lcs": _blocks_lcs}


def compute_hunks(canvas: Sequence[int], target: Sequence[int], align: str = "difflib") -> List[Hunk]:
    hunks: List[Hunk] = []
    pa = pb = 0
    for a_i, b_i, size in ALIGN[align](list(canvas), list(target)):   # ends with (len(a), len(b), 0)
        if a_i > pa or b_i > pb:
            hunks.append(Hunk(pa - 0.5, a_i - 0.5, pb - 0.5, b_i - 0.5))
        pa, pb = a_i + size, b_i + size
    return [h for h in hunks if h.canvas_len > 0 or h.target_len > 0]


# ---------------------------------------------------------------------------------------------------------------
# oracle.py (copied; PORT changes marked)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OracleConfig:
    include_second_best_moves: bool = False
    second_best_weight: float = 0.5
    move_decay: float = 0.3
    max_insert_suggestions: int = 16
    align: str = "difflib"   # PORT: "difflib" (original) or "lcs" (see _blocks_lcs)
    insert_rule: str = "proportion"   # PORT: "proportion" (original) or "count" (see _weighted_inserts)


@dataclass(frozen=True)
class OracleOutput:
    action_ids: List[int]
    weights: List[float]
    kind: str = ""

    def normalized(self) -> "OracleOutput":
        w = [max(0.0, float(x)) for x in self.weights]
        s = sum(w)
        if s <= 0:
            w, s = [1.0] * len(w), float(len(w))
        return OracleOutput(list(self.action_ids), [x / s for x in w], self.kind)

    def as_dict(self) -> Dict[int, float]:
        return dict(zip(self.action_ids, self.weights))


def _half_to_int(p: float) -> int:
    return int(round(float(p) + 0.5))


def _min_steps_to_interval(cursor_half: float, l_half: float, r_half: float) -> Tuple[int, int]:
    c, l, r = _half_to_int(cursor_half), _half_to_int(l_half), _half_to_int(r_half)
    if l > r:
        l, r = r, l
    best_steps, best_pos = 10 ** 9, l
    for pos in range(l, r + 1):
        steps = abs(pos - c).bit_count()
        if steps < best_steps or (steps == best_steps and pos > best_pos):
            best_steps, best_pos = steps, pos
    return best_steps, best_pos


def _moves_to_hunks(hunks: List[Hunk], cursor_half: float, cfg: OracleConfig) -> OracleOutput:
    scored = sorted(((*_min_steps_to_interval(cursor_half, h.canvas_l, h.canvas_r), h) for h in hunks),
                    key=lambda x: x[0])
    best_steps = scored[0][0]
    best = [x for x in scored if x[0] == best_steps]
    second = []
    if cfg.include_second_best_moves:
        for s, _, _ in scored:
            if s > best_steps:
                second = [x for x in scored if x[0] == s]
                break
    tok_w: Dict[int, float] = {}

    def apply(group, scale):
        for _, pos, _h in group:
            ids = decompose_move(pos - _half_to_int(cursor_half))
            if not ids:
                continue
            per = scale / len(ids)
            for t in set(ids):
                tok_w[t] = max(tok_w.get(t, 0.0), per)   # ties: greatest weight per move token

    apply(best, 1.0)
    if second:
        apply(second, cfg.second_best_weight)
    if not tok_w:   # PORT: was an empty output; unreachable (cursor outside every hunk => distance >= 1)
        raise RuntimeError("hunk oracle: no move toward any hunk")
    ids = sorted(tok_w)
    return OracleOutput(ids, [tok_w[t] for t in ids], "moves")


def _weighted_inserts(target_sub: List[int], cursor_half: float, h: Hunk, cfg: OracleConfig) -> Dict[int, float]:
    if not target_sub:
        return {}
    denom = max(1, h.canvas_len)
    rel = max(0, min(denom, _half_to_int(cursor_half) - h.canvas_bounds_int()[0]))
    p = rel / denom
    n = len(target_sub)
    cutoff = max(0, min(n, int(p * n)))
    if cfg.insert_rule == "count":
        # PORT variant: as many weight-1 target tokens as there are hunk canvas tokens left of the cursor. The original
        # proportion rule gives ALL missing target tokens weight 1 when the cursor is at the hunk's right edge; for a
        # junk token at the writing end (hunk target side = the whole unwritten continuation) that is 16 equally
        # weighted inserts of tokens up to 15 places ahead next to one DELETE.
        cutoff = max(0, min(n, rel))
    base_end = min(max(1, cutoff), n)
    extra = min(4, max(0, n - base_end))
    m = 0.0 if 1 <= h.canvas_len <= 4 else 0.5
    out: Dict[int, float] = {}
    for i in range(base_end):
        out[target_sub[i]] = max(out.get(target_sub[i], 0.0), 1.0)
    w = 1.0
    for j in range(extra):
        w *= m
        if w <= 0.0:
            break
        t = target_sub[base_end + j]
        out[t] = max(out.get(t, 0.0), w)
    if len(out) > cfg.max_insert_suggestions:
        out = dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True)[: cfg.max_insert_suggestions])
    return out


def _moves_within_hunk(cursor_half: float, h: Hunk, cfg: OracleConfig) -> Dict[int, float]:
    delta = max(0, h.canvas_bounds_int()[1] - _half_to_int(cursor_half))
    out: Dict[int, float] = {}
    w = 1.0
    for amt in reversed(MOVE_AMOUNTS):
        if amt <= delta:
            t = move_id(amt)
            out[t] = max(out.get(t, 0.0), w)
            w *= cfg.move_decay
            if w <= 0.0:
                break
    return out


def oracle_policy(canvas: Sequence[int], cursor_half: float, target: Sequence[int],
                  cfg: Optional[OracleConfig] = None) -> OracleOutput:
    """Original API: cursor in half-index space (integer cursor c <-> c - 0.5). Raw (unnormalized) weights."""
    cfg = cfg or OracleConfig()
    canvas, target = list(canvas), list(target)
    if canvas == target:
        return OracleOutput([EOR], [1.0], "eor")
    hunks = compute_hunks(canvas, target, cfg.align)
    h = next((x for x in hunks if x.cursor_in_hunk(cursor_half)), None)
    if h is None:
        return _moves_to_hunks(hunks, cursor_half, cfg)
    t_l, t_r = h.target_bounds_int()
    target_sub = target[t_l:t_r]
    has_inserts = len(target_sub) > 0
    at_left = abs(cursor_half - h.canvas_l) < 1e-9
    tok_w: Dict[int, float] = {}
    if has_inserts:
        for t, w in _weighted_inserts(target_sub, cursor_half, h, cfg).items():
            tok_w[t] = max(tok_w.get(t, 0.0), w)
    if not at_left and h.canvas_len > 0:
        tok_w[DELETE] = max(tok_w.get(DELETE, 0.0), 1.0)
    if at_left:
        for t, w in _moves_within_hunk(cursor_half, h, cfg).items():
            tok_w[t] = max(tok_w.get(t, 0.0), w)
        if not has_inserts:
            for t in set(decompose_move(h.canvas_bounds_int()[1] - _half_to_int(cursor_half))):
                tok_w[t] = max(tok_w.get(t, 0.0), 1.0)
    if not tok_w:   # PORT: the original returned EOR here; EOR is now only for canvas == target
        raise RuntimeError(f"hunk oracle: empty action set inside hunk {h}")
    ids = sorted(tok_w)
    return OracleOutput(ids, [tok_w[t] for t in ids], "in_hunk")


def oracle_labels(canvas: Sequence[int], cursor: int, target: Sequence[int],
                  cfg: Optional[OracleConfig] = None) -> OracleOutput:
    """PORT: integer-cursor wrapper (cursor in 0..len(canvas)); raw weights."""
    return oracle_policy(canvas, float(cursor) - 0.5, target, cfg)


def oracle_distribution(canvas: Sequence[int], cursor: int, target: Sequence[int],
                        cfg: Optional[OracleConfig] = None) -> OracleOutput:
    return oracle_labels(canvas, cursor, target, cfg).normalized()


def junk_count(canvas: Sequence[int], target: Sequence[int], align: str = "difflib") -> Tuple[int, int]:
    """(canvas tokens not aligned to the target, target tokens missing from the canvas) under the oracle's diff."""
    hs = compute_hunks(canvas, target, align)
    return sum(h.canvas_len for h in hs), sum(h.target_len for h in hs)


def run_oracle(state: State, target: Sequence[int], rng, max_steps: int = 4000, mode: str = "sample",
               cfg: Optional[OracleConfig] = None):
    """Execute the oracle alone from `state` until it emits EOR or max_steps. mode: 'sample' (normalized weights),
    'argmax' (highest weight; ties -> smallest id). Returns (actions, final_state, emitted_eor)."""
    st = state.copy()
    acts: List[int] = []
    for _ in range(max_steps):
        out = oracle_labels(st.canvas, st.cursor, target, cfg)
        if mode == "argmax":
            best = max(out.weights)
            a = min(t for t, w in zip(out.action_ids, out.weights) if w == best)
        else:
            tot = sum(out.weights)
            r, acc, a = rng.random() * tot, 0.0, out.action_ids[-1]
            for t, w in zip(out.action_ids, out.weights):
                acc += w
                if r < acc:
                    a = t
                    break
        if not st.valid(a):
            raise RuntimeError(f"oracle proposed invalid action {a} at cursor {st.cursor}/{len(st.canvas)}")
        acts.append(a)
        if a == EOR:
            return acts, st, True
        st.step(a)
    return acts, st, False
