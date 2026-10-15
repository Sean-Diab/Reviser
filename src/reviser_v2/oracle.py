"""Exact, alignment-consistent edit oracle for the Reviser v2 action space (task 7).

Replaces the July DAgger oracle (``workspaces/7-3/impl/edit_env.py``: ``OracleRho`` +
``EpisodeState``), which kept its own token "matched/junk" tags that could disagree with what its
policy intended, could not delete, and declared completion from a matched-token *count*. Diagnosis:
``notes/sources/oracle-repair-2026-09-24.md``.

Design
------
* Single source of truth. The oracle is a pure function of the executor state (canvas token values,
  cursor) and the target. Nothing is tagged at insertion time; every query re-aligns the whole canvas
  against the target with an explicit optimal alignment, so a token whose value equals a *later* target
  token is kept or deleted according to what is optimal *now*, never by a stale greedy tag.
* Completion means ``canvas == target``. STOP is proposed iff that holds (and then it is the only
  proposal).
* Cost model: number of actions, STOP included; every move token costs 1 (so a goto costs the number of
  move tokens along a shortest legal move path, which is ``len(space.compile_goto)`` or less, e.g. pow2
  ``+8,-1`` for a distance of 7).
* ``query`` returns the COMPLETE set of optimal next actions (all actions ``a`` with
  ``1 + d(next(s, a)) == d(s)``, ``d`` = true shortest remaining cost in the executor's action graph)
  plus a deterministic canonical choice.

Algorithm
---------
Exact bounded depth-first search (IDA* with a transposition table) over real executor states, guided
by an admissible heuristic ``h`` that is the exact cost in a relaxed action graph where every cursor
move costs 1 and may reach any boundary (restricted to boundaries left of the cursor when
``backward_only``). ``h`` is computed by an alignment DP over (canvas prefix, target prefix, in-gap
flag): cost = #deletes + #inserts + #work-gaps (a work gap is a maximal region between kept tokens
that needs at least one delete/insert; each needs one cursor arrival), minus one when the cursor
already sits at the right end of a work gap. Proof sketch that this lower-bounds every encoding:
kept canvas tokens form a common subsequence with the target; inserts/deletes never carry the cursor
across a kept token (insert stays left of it, delete-left only removes junk), so every work gap except
one containing the cursor at its right end needs >= 1 move, and one move lands in one gap. The
relaxed graph contains every real edge at cost <= 1, so ``h`` is admissible and consistent.

* Absolute moves with enough range (``n_move - 1 >= max canvas length``), not backward-only: ``h`` is
  exact and the search never branches beyond ties.
* Relative moves: ``h`` undercounts multi-token gotos; the search closes the gap exactly.
* An upper bound comes from a greedy policy that always terminates (take an edit that lowers ``h`` by
  one, else walk a shortest move path to the nearest boundary where ``h`` drops). When lower bound ==
  upper bound no search is needed.

Search space: a maximal run of move tokens happens at fixed canvas length, so it is replaced by a
single macro edge goto(d) costing the shortest legal move-token count (BFS over the encoding's move
graph, bounds and ``backward_only`` respected), after which an edit (or STOP) must follow. Successor
heuristics are computed incrementally from prefix/suffix DP tables of the parent canvas in O(m) each;
one expansion costs O(n*m) for the tables (cached per canvas) + O((|set(target)| + n) * m).

Exactness is unconditional whenever the search finishes; the running time is not: it grows with the
number of separated error regions whose multi-token gotos the relaxation undercounts (relative
encodings). ``max_nodes`` (and optionally ``max_seconds``) bound the search; when exceeded, ``query`` returns the greedy action
with ``exact=False`` and ``cost`` = the greedy upper bound. Verified against brute-force BFS in
``tests/test_oracle.py``; measured speed/limits in ``notes/sources/oracle-repair-2026-09-24.md``.

Search restrictions that do not affect exactness: inserts are generated for target values plus ONE
representative non-target value (all non-target values are interchangeable junk; if the
representative is optimal, every non-target value is reported).
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .actions import ActionSpace
from .executor import Canvas

BIG = 1 << 40          # "infinity" inside the DP (sums stay finite ints)
HALF = BIG >> 1
INF = BIG
Key = Tuple[Tuple[int, ...], int]          # (canvas tokens, cursor)
SKey = Tuple[Tuple[int, ...], int, int]    # search state: key + must-edit flag after a goto


class OracleBudgetExceeded(RuntimeError):
    """Exact search exceeded ``max_nodes`` expansions."""


class OracleUnreachable(RuntimeError):
    """The target cannot be reached from this state under the action space."""


@dataclass(frozen=True)
class OracleResult:
    actions: FrozenSet[int]   # optimal next actions (exact=True) or the fallback action (exact=False)
    canonical: Optional[int]  # deterministic choice from ``actions``; None if unreachable
    cost: Optional[int]       # remaining actions incl. STOP: exact if ``exact`` else an upper bound
    lower_bound: int          # admissible heuristic h at this state (INF if provably unreachable)
    exact: bool
    reachable: bool


def is_complete(tokens: Sequence[int], target: Sequence[int]) -> bool:
    """Completion criterion: the canvas IS the intended text (not a matched-token count)."""
    return list(tokens) == list(target)


def _fin(x: int) -> int:
    return x if x < HALF else INF


# ---------------------------------------------------------------------------
# Relaxed-cost DP tables for one canvas
# ---------------------------------------------------------------------------
class _Tables:
    """Alignment DP tables of canvas ``T`` against target ``Y``.

    P0[i][j] / P1[i][j]: min relaxed cost aligning T[:i] with Y[:j], ending after a kept pair (0) or
    inside an open work gap whose +1 arrival cost is already paid (1).
    S0[i][j] / S1[i][j]: the same for suffixes T[i:], Y[j:], entered outside / inside an open gap.
    Q[c][j]: cost of finishing when inside an open gap at (c, j) whose right end must be canvas
    boundary c: only inserts Y[j:j'] remain in it, then T[c] is kept against Y[j'] (or c == n end).
    """

    def __init__(self, T: Tuple[int, ...], Y: Tuple[int, ...], allow_del: bool, bo: bool, ypos: Dict[int, List[int]]):
        self.T, self.Y, self.allow_del, self.bo, self.ypos = T, Y, allow_del, bo, ypos
        self.n, self.m = len(T), len(Y)
        self.done = T == Y
        self.P0, self.P1 = self._prefix()
        self._rb: Dict[int, int] = {}
        if bo:
            k = 0
            while k < self.n and k < self.m and T[self.n - 1 - k] == Y[self.m - 1 - k]:
                k += 1
            self.sfx = k
        else:
            self.S0, self.S1 = self._suffix()
            self.Q = self._q()
            self.R = min(self.P0[self.n][self.m], self.P1[self.n][self.m])
            self._junk: Dict[int, Tuple[int, int]] = {}

    # -- construction ------------------------------------------------------
    def _prefix(self):
        T, Y, n, m, ad = self.T, self.Y, self.n, self.m, self.allow_del
        P0: List[List[int]] = [None] * (n + 1)  # type: ignore
        P1: List[List[int]] = [None] * (n + 1)  # type: ignore
        r0 = [BIG] * (m + 1)
        r1 = [BIG] * (m + 1)
        r0[0] = 0
        for j in range(1, m + 1):
            a, b = r0[j - 1] + 2, r1[j - 1] + 1
            r1[j] = a if a < b else b
        P0[0], P1[0] = r0, r1
        for i in range(1, n + 1):
            t = T[i - 1]
            a0, a1 = P0[i - 1], P1[i - 1]
            b0 = [BIG] * (m + 1)
            b1 = [BIG] * (m + 1)
            if ad:
                x, y = a0[0] + 2, a1[0] + 1
                b1[0] = x if x < y else y
            for j in range(1, m + 1):
                if Y[j - 1] == t:
                    x, y = a0[j - 1], a1[j - 1]
                    b0[j] = x if x < y else y
                best = BIG
                if ad:
                    x, y = a0[j] + 2, a1[j] + 1
                    best = x if x < y else y
                x = b0[j - 1] + 2
                if x < best:
                    best = x
                x = b1[j - 1] + 1
                if x < best:
                    best = x
                b1[j] = best
            P0[i], P1[i] = b0, b1
        return P0, P1

    def _suffix(self):
        T, Y, n, m, ad = self.T, self.Y, self.n, self.m, self.allow_del
        S0: List[List[int]] = [None] * (n + 1)  # type: ignore
        S1: List[List[int]] = [None] * (n + 1)  # type: ignore
        r0 = [BIG] * (m + 1)
        r1 = [BIG] * (m + 1)
        r0[m] = r1[m] = 0
        for j in range(m - 1, -1, -1):
            op = 1 + r1[j + 1]
            r1[j] = op
            r0[j] = 1 + op
        S0[n], S1[n] = r0, r1
        for i in range(n - 1, -1, -1):
            t = T[i]
            d0, d1 = S0[i + 1], S1[i + 1]
            r0 = [BIG] * (m + 1)
            r1 = [BIG] * (m + 1)
            op = (1 + d1[m]) if ad else BIG
            r1[m] = op
            r0[m] = 1 + op
            for j in range(m - 1, -1, -1):
                keep = d0[j + 1] if Y[j] == t else BIG
                op = (1 + d1[j]) if ad else BIG
                x = 1 + r1[j + 1]
                if x < op:
                    op = x
                r1[j] = keep if keep < op else op
                x = 1 + op
                r0[j] = keep if keep < x else x
            S0[i], S1[i] = r0, r1
        return S0, S1

    def _q(self):
        T, Y, n, m = self.T, self.Y, self.n, self.m
        Q: List[List[int]] = [None] * (n + 1)  # type: ignore
        Q[n] = [m - j for j in range(m + 1)]
        for c in range(n):
            t = T[c]
            s = self.S0[c + 1]
            row = [BIG] * (m + 1)
            for j in range(m - 1, -1, -1):
                x = s[j + 1] if Y[j] == t else BIG
                y = 1 + row[j + 1]
                row[j] = x if x < y else y
            Q[c] = row
        return Q

    # -- relaxed cost-to-go (including STOP) --------------------------------
    def _rb_at(self, c: int) -> int:
        """Min relaxed cost of alignments in which the gap ending at boundary c is a work gap."""
        v = self._rb.get(c)
        if v is None:
            p0, p1, q, m = self.P0[c], self.P1[c], self.Q[c], self.m
            v = BIG
            for j in range(m + 1):
                x = p1[j] + q[j]
                if x < v:
                    v = x
                if j < m:
                    x = p0[j] + 2 + q[j + 1]
                    if x < v:
                        v = x
            self._rb[c] = v
        return v

    def _bo_ok(self, c: int) -> bool:
        return self.n - c <= self.sfx

    def h_at(self, c: int) -> int:
        """Relaxed cost for cursor ``c`` on this canvas."""
        if self.done:
            return 1
        if self.bo:
            if not self._bo_ok(c):
                return INF
            j0 = self.m - (self.n - c)
            return _fin(1 + min(self.P0[c][j0], self.P1[c][j0] - 1))
        return _fin(1 + min(self.R, self._rb_at(c) - 1))

    def _junk_at(self, c: int) -> Tuple[int, int]:
        v = self._junk.get(c)
        if v is None:
            if not self.allow_del:
                v = (BIG, BIG)
            else:
                p0, p1, s1, q = self.P0[c], self.P1[c], self.S1[c], self.Q[c]
                jr = jb = BIG
                for j in range(self.m + 1):
                    x, y = p0[j] + 2, p1[j] + 1
                    base = x if x < y else y
                    a = base + s1[j]
                    if a < jr:
                        jr = a
                    a = base + q[j]
                    if a < jb:
                        jb = a
                v = (jr, jb)
            self._junk[c] = v
        return v

    def h_insert(self, c: int, v: int) -> int:
        """Relaxed cost after inserting value ``v`` at cursor ``c`` (new cursor c+1)."""
        Y, m = self.Y, self.m
        if self.bo:
            if not self._bo_ok(c):
                return INF
            j0 = m - (self.n - c)
            p0, p1, ad = self.P0[c], self.P1[c], self.allow_del
            q0 = [BIG] * (j0 + 1)
            q1 = [BIG] * (j0 + 1)
            for j in range(j0 + 1):
                if j >= 1 and Y[j - 1] == v:
                    x, y = p0[j - 1], p1[j - 1]
                    q0[j] = x if x < y else y
                best = BIG
                if ad:
                    x, y = p0[j] + 2, p1[j] + 1
                    best = x if x < y else y
                if j >= 1:
                    x = q0[j - 1] + 2
                    if x < best:
                        best = x
                    x = q1[j - 1] + 1
                    if x < best:
                        best = x
                q1[j] = best
            return _fin(1 + min(q0[j0], q1[j0] - 1))
        r, rb = self._junk_at(c)
        p0, p1, s0, q = self.P0[c], self.P1[c], self.S0[c], self.Q[c]
        for j in self.ypos.get(v, ()):
            x, y = p0[j], p1[j]
            base = x if x < y else y
            a = base + s0[j + 1]
            if a < r:
                r = a
            if j + 1 < m:
                a = base + 2 + q[j + 2]
                if a < rb:
                    rb = a
        return _fin(1 + min(r, rb - 1))

    def h_delete(self, c: int) -> int:
        """Relaxed cost after DELETE at cursor ``c`` (removes T[c-1]; new cursor c-1)."""
        m = self.m
        if self.bo:
            if not self._bo_ok(c):
                return INF
            j0 = m - (self.n - c)
            return _fin(1 + min(self.P0[c - 1][j0], self.P1[c - 1][j0] - 1))
        p0, p1 = self.P0[c - 1], self.P1[c - 1]
        s0, s1, q = self.S0[c], self.S1[c], self.Q[c]
        r = rb = BIG
        for j in range(m + 1):
            a = p0[j] + s0[j]
            if a < r:
                r = a
            a = p1[j] + s1[j]
            if a < r:
                r = a
            a = p1[j] + q[j]
            if a < rb:
                rb = a
            if j < m:
                a = p0[j] + 2 + q[j + 1]
                if a < rb:
                    rb = a
        return _fin(1 + min(r, rb - 1))


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
class EditOracle:
    """Target-specific oracle with caches; reuse one instance for all states of an episode."""

    def __init__(self, space: ActionSpace, target: Sequence[int], max_nodes: int = 3000,
                 max_seconds: Optional[float] = None, max_cached_tables: int = 4096):
        self.space = space
        self.Y: Tuple[int, ...] = tuple(int(t) for t in target)
        for t in self.Y:
            if not space.is_content(t):
                raise ValueError(f"target token {t} is not a content token")
        self.bo = space.backward_only
        self.allow_del = space.allow_delete
        self.ypos: Dict[int, List[int]] = {}
        for j, t in enumerate(self.Y):
            self.ypos.setdefault(t, []).append(j)
        yset = set(self.Y)
        self.values = sorted(yset)
        self.junk_rep = next((t for t in range(space.n_content) if t not in yset), None)
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self._deadline = INF
        self.max_cached_tables = max_cached_tables
        self._tables: Dict[Tuple[int, ...], _Tables] = {}
        self._h: Dict[Key, int] = {}
        self._esucc: Dict[Key, list] = {}
        self._msucc: Dict[SKey, list] = {}
        self._dist: Dict[int, List[List[int]]] = {}
        self._lb: Dict[SKey, int] = {}
        self._ub: Dict[SKey, int] = {}
        self._results: Dict[Key, OracleResult] = {}
        self._moves_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._nodes = 0

    # -- primitives -----------------------------------------------------------
    def tables(self, T: Tuple[int, ...]) -> _Tables:
        tb = self._tables.get(T)
        if tb is None:
            if len(self._tables) >= self.max_cached_tables:
                self._tables.clear()
            tb = _Tables(T, self.Y, self.allow_del, self.bo, self.ypos)
            self._tables[T] = tb
        return tb

    def h(self, key: Key) -> int:
        v = self._h.get(key)
        if v is None:
            v = self.tables(key[0]).h_at(key[1])
            self._h[key] = v
        return v

    def legal_moves(self, L: int, c: int) -> List[Tuple[int, int]]:
        """(move token, destination) pairs legal at canvas length L, cursor c (executor semantics)."""
        k = (L, c)
        out = self._moves_cache.get(k)
        if out is None:
            s = self.space
            out = []
            for i, v in enumerate(s.move_values):
                if s.move_encoding == "absolute":
                    if v <= L and v != c and (v < c if s.backward_only else True):
                        out.append((s.move_start + i, v))
                else:
                    d = c + v
                    if 0 <= d <= L:
                        out.append((s.move_start + i, d))
            self._moves_cache[k] = out
        return out

    def priority(self, a: int):
        """Canonical tie-break: STOP, DELETE, inserts (by id), then moves (largest step, leftward first)."""
        s = self.space
        if a == s.STOP:
            return (0, 0, 0)
        if a == s.DELETE:
            return (1, 0, 0)
        if s.is_content(a):
            return (2, a, 0)
        v = s.move_value(a)
        if s.move_encoding == "absolute":
            return (3, v, 0)
        return (3, -abs(v), 0 if v < 0 else 1)

    def edit_successors(self, key: Key):
        """Sorted [(h', priority, action, next_key)] over DELETE and useful inserts at the cursor."""
        out = self._esucc.get(key)
        if out is not None:
            return out
        T, c = key
        tb = self.tables(T)
        out = []
        hcache = self._h
        if self.allow_del and c > 0:
            k2 = (T[:c - 1] + T[c:], c - 1)
            h2 = tb.h_delete(c)
            hcache.setdefault(k2, h2)
            out.append((h2, self.priority(self.space.DELETE), self.space.DELETE, k2))
        vals = self.values if self.junk_rep is None else self.values + [self.junk_rep]
        for v in vals:
            k2 = (T[:c] + (v,) + T[c:], c + 1)
            h2 = tb.h_insert(c, v)
            hcache.setdefault(k2, h2)
            out.append((h2, self.priority(v), v, k2))
        out.sort(key=lambda x: (x[0], x[1]))
        self._esucc[key] = out
        return out

    def move_dist(self, L: int) -> List[List[int]]:
        """All-pairs shortest move-token counts between boundaries of a length-L canvas (INF if none)."""
        D = self._dist.get(L)
        if D is None:
            D = []
            for src in range(L + 1):
                row = [INF] * (L + 1)
                row[src] = 0
                dq = deque([src])
                while dq:
                    p = dq.popleft()
                    for _, d in self.legal_moves(L, p):
                        if row[d] == INF:
                            row[d] = row[p] + 1
                            dq.append(d)
                D.append(row)
            self._dist[L] = D
        return D

    def _macro(self, k: SKey):
        """Macro successors [(cost, h', priority, next_skey)] of a search state.

        A maximal run of move tokens happens at fixed canvas length, so it can be replaced by a
        shortest move path without loss: free states (flag 0) may take one edit or one goto(d) of cost
        dist(c, d); the state after a goto (flag 1) must take an edit (or STOP) next."""
        out = self._msucc.get(k)
        if out is not None:
            return out
        T, c, flag = k
        out = [(1, h2, p, (T2, c2, 0)) for h2, p, _, (T2, c2) in self.edit_successors((T, c))]
        if flag == 0:
            L = len(T)
            row = self.move_dist(L)[c]
            tb = self.tables(T)
            for d in range(L + 1):
                D = row[d]
                if d != c and D < INF:
                    h2 = tb.h_at(d)
                    if h2 < INF:
                        out.append((D, h2, (3, D, d), (T, d, 1)))
        out.sort(key=lambda x: (x[0] + x[1], x[2]))
        self._msucc[k] = out
        return out

    # -- exact bounded search -----------------------------------------------------
    def _reach(self, k: SKey, budget: int) -> bool:
        """True iff the target (then STOP) is reachable from search state ``k`` within ``budget`` actions."""
        if self._ub.get(k, INF) <= budget:
            return True
        if self._lb.get(k, 0) > budget:
            return False
        T = k[0]
        if self.h((T, k[1])) > budget:
            return False
        if T == self.Y:
            self._ub[k] = 1
            return True
        self._nodes += 1
        if self._nodes > self.max_nodes:
            raise OracleBudgetExceeded(f"exceeded {self.max_nodes} expansions")
        if self._deadline != INF and time.perf_counter() > self._deadline:
            raise OracleBudgetExceeded(f"exceeded {self.max_seconds} s")
        for cost, h2, _, k2 in self._macro(k):
            if cost + h2 > budget:
                break
            if self._reach(k2, budget - cost):
                self._ub[k] = min(self._ub.get(k, INF), cost + self._ub[k2])
                return True
        self._lb[k] = budget + 1
        return False

    # -- greedy upper bound / fallback policy ------------------------------------
    def greedy_action(self, key: Key) -> Tuple[int, Key]:
        """Terminating (not necessarily optimal) policy: an edit lowering h by one, else the first move
        of a shortest move path to the nearest boundary whose h is one lower."""
        T, c = key
        if T == self.Y:
            return self.space.STOP, key
        h = self.h(key)
        if h >= INF:
            raise OracleUnreachable("no path to target from this state")
        edits = [(p, a, k2) for h2, p, a, k2 in self.edit_successors(key) if h2 == h - 1]
        if edits:
            _, a, k2 = min(edits)
            return a, k2
        tb = self.tables(T)
        L = len(T)
        goal = {d for d in range(L + 1) if d != c and tb.h_at(d) == h - 1}
        if not goal:
            raise OracleUnreachable("greedy policy found no productive boundary")
        first: Dict[int, Tuple[int, int]] = {c: (-1, c)}  # boundary -> (first move token, its destination)
        dq = deque([c])
        while dq:
            p = dq.popleft()
            for a, d in sorted(self.legal_moves(L, p), key=lambda x: self.priority(x[0])):
                if d in first:
                    continue
                first[d] = (a, d) if p == c else first[p]
                if d in goal:
                    a0, d0 = first[d]
                    return a0, (T, d0)
                dq.append(d)
        raise OracleUnreachable("no productive boundary reachable by moves")

    def greedy_plan(self, key: Key, max_steps: Optional[int] = None) -> List[int]:
        """Greedy action list from ``key`` to STOP; seeds exact upper bounds along the path."""
        if max_steps is None:
            max_steps = 8 * (len(key[0]) + len(self.Y)) + 64
        path_keys, acts = [key], []
        k = key
        while True:
            a, k2 = self.greedy_action(k)
            acts.append(a)
            if a == self.space.STOP:
                break
            k = k2
            path_keys.append(k)
            if len(acts) > max_steps:
                raise RuntimeError("greedy plan did not terminate (bug)")
        n = len(acts)
        for i, (T, c) in enumerate(path_keys):
            sk = (T, c, 0)
            if self._ub.get(sk, INF) > n - i:
                self._ub[sk] = n - i
        return acts

    # -- public -------------------------------------------------------------------
    def query(self, tokens: Sequence[int], cursor: int) -> OracleResult:
        key: Key = (tuple(tokens), int(cursor))
        res = self._results.get(key)
        if res is not None:
            return res
        res = self._query(key)
        self._results[key] = res
        return res

    def _query(self, key: Key) -> OracleResult:
        s = self.space
        T, c = key
        if not 0 <= c <= len(T):
            raise ValueError("cursor out of bounds")
        if T == self.Y:
            return OracleResult(frozenset([s.STOP]), s.STOP, 1, 1, True, True)
        h0 = self.h(key)
        if h0 >= INF:
            return OracleResult(frozenset(), None, None, INF, True, False)
        try:
            greedy = self.greedy_plan(key)
            upper = len(greedy)
        except OracleUnreachable:
            greedy, upper = None, INF
        old = sys.getrecursionlimit()
        need = 4 * (min(upper, h0 + 4 * (len(T) + len(self.Y)) + 64)) + 1000
        if need > old:
            sys.setrecursionlimit(need)
        self._nodes = 0
        self._deadline = INF if self.max_seconds is None else time.perf_counter() + self.max_seconds
        root = (T, c, 0)
        try:
            V = h0
            while V < upper and not self._reach(root, V):
                V += 1
            if V >= INF:
                return OracleResult(frozenset(), None, None, h0, False, False)
            acts = set()
            for h2, _, a, (T2, c2) in self.edit_successors(key):
                if 1 + h2 > V:
                    break
                if self._reach((T2, c2, 0), V - 1):
                    acts.add(a)
            L = len(T)
            dist = self.move_dist(L)
            row = dist[c]
            tb = self.tables(T)
            dests = [d for d in range(L + 1) if d != c and row[d] < INF and row[d] + tb.h_at(d) <= V
                     and self._reach((T, d, 1), V - row[d])]
            for a, e in self.legal_moves(L, c):
                if any(1 + dist[e][d] == row[d] for d in dests):
                    acts.add(a)
        except OracleBudgetExceeded:
            if greedy is None:
                raise
            return OracleResult(frozenset([greedy[0]]), greedy[0], upper, h0, False, True)
        finally:
            if need > old:
                sys.setrecursionlimit(old)
        if not acts:  # cannot happen when V is an exact finite distance
            raise RuntimeError("oracle found no optimal action (bug)")
        if self.junk_rep is not None and self.junk_rep in acts:
            yset = set(self.Y)
            acts.update(t for t in range(s.n_content) if t not in yset)
        canonical = min(acts, key=self.priority)
        return OracleResult(frozenset(acts), canonical, V, h0, True, True)

    def cost(self, tokens: Sequence[int], cursor: int) -> Optional[int]:
        return self.query(tokens, cursor).cost


# ---------------------------------------------------------------------------
# Convenience API for rollouts and DAgger labels
# ---------------------------------------------------------------------------
def oracle_next_actions(space: ActionSpace, canvas: Canvas, target: Sequence[int],
                        oracle: Optional[EditOracle] = None) -> OracleResult:
    if canvas.stopped:
        raise ValueError("episode already stopped")
    orc = oracle if oracle is not None else EditOracle(space, target)
    return orc.query(canvas.tokens, canvas.cursor)


def oracle_rollout(space: ActionSpace, prompt: Sequence[int], draft: Sequence[int], target: Sequence[int],
                   seed_cursor: Optional[int] = None, oracle: Optional[EditOracle] = None,
                   max_actions: Optional[int] = None) -> List[int]:
    """Pure-oracle action list (canonical choice each step) from ``draft`` to ``target`` + STOP.

    ``prompt`` is protected conditioning and never on the canvas; it is accepted for API parity with
    ``Episode``/``Request``. Every step is an exact optimal action while the search stays within
    ``oracle.max_nodes``; from the first state where it does not, the rest is the greedy plan (valid,
    terminating, not certified optimal). Raises ``OracleUnreachable`` if the target cannot be reached
    (e.g. under ``backward_only`` with an unfixable suffix right of the seed cursor)."""
    orc = oracle if oracle is not None else EditOracle(space, target)
    canvas = Canvas.from_draft(space, draft, seed_cursor)
    limit = max_actions if max_actions is not None else 8 * (len(draft) + len(target)) + 64
    acts: List[int] = []
    while not canvas.stopped:
        r = orc.query(canvas.tokens, canvas.cursor)
        if not r.reachable:
            raise OracleUnreachable(f"target unreachable at cursor {canvas.cursor}")
        if not r.exact:
            # search budget exceeded: finish with the greedy plan, which terminates on its own
            # (mixing inexact and exact steps could otherwise cycle)
            for a in orc.greedy_plan((tuple(canvas.tokens), canvas.cursor)):
                canvas.apply(a)
                acts.append(a)
            break
        canvas.apply(r.canonical)
        acts.append(r.canonical)
        if len(acts) > limit:
            raise RuntimeError("oracle rollout exceeded action limit")
    if canvas.tokens != list(target):
        raise RuntimeError("oracle stopped on a wrong canvas (bug)")
    return acts


def oracle_labels_for_history(space: ActionSpace, draft: Sequence[int], target: Sequence[int],
                              history: Sequence[int], seed_cursor: Optional[int] = None,
                              oracle: Optional[EditOracle] = None) -> List[OracleResult]:
    """Oracle label at every state of an executed (e.g. model-generated) history.

    Returns ``labels`` with ``labels[t]`` = result for the state BEFORE ``history[t]``; if the history
    does not end with STOP, one extra label for the final state is appended. The history is replayed
    through the executor (raises ``IllegalAction`` on an illegal action). A label with
    ``reachable=False`` has no valid supervision (drop it). A model STOP on a wrong canvas is labeled
    by the state's real optimal set (never STOP), so premature stops are supervised."""
    orc = oracle if oracle is not None else EditOracle(space, target)
    canvas = Canvas.from_draft(space, draft, seed_cursor)
    out: List[OracleResult] = []
    for a in history:
        out.append(orc.query(canvas.tokens, canvas.cursor))
        canvas.apply(a)
        if canvas.stopped:
            return out
    out.append(orc.query(canvas.tokens, canvas.cursor))
    return out
