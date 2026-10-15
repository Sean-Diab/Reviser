"""Tests for the repaired edit oracle (task 7). See notes/sources/oracle-repair-2026-09-24.md."""
import itertools
import random
from collections import deque
from pathlib import Path

import pytest

from reviser_v2.actions import ActionSpace
from reviser_v2.executor import Canvas, replay
from reviser_v2.oracle import (
    INF, EditOracle, _Tables, is_complete, oracle_labels_for_history, oracle_rollout,
)
from reviser_v2.tasks import progression as prog
from reviser_v2.trajectories import validate_actions


# ---------------------------------------------------------------------------
# Brute-force reference: exact distances by reverse BFS over ALL capped states
# ---------------------------------------------------------------------------
def _step(space, key, a):
    c = Canvas(space, list(key[0]), list(range(len(key[0]))), key[1])
    if not c.is_legal(a):
        return None
    c.apply(a)
    return tuple(c.tokens), c.cursor


def _bfs_reference(space, target, cap):
    """d[s] = min #actions (incl. STOP) to reach ``target`` and stop, over canvases of length <= cap,
    using EVERY action id in the vocabulary (all content values, DELETE, all moves)."""
    acts = [a for a in range(space.vocab_size) if a not in (space.SEP, space.EDIT, space.STOP)]
    states = [(T, c) for L in range(cap + 1) for T in itertools.product(range(space.n_content), repeat=L)
              for c in range(L + 1)]
    fwd, rev = {}, {s: [] for s in states}
    for s in states:
        lst = []
        for a in acts:
            t = _step(space, s, a)
            if t is not None and len(t[0]) <= cap:
                lst.append((a, t))
                rev[t].append(s)
        fwd[s] = lst
    d, dq = {}, deque()
    for s in states:
        if s[0] == tuple(target):
            d[s] = 1
            dq.append(s)
    while dq:
        s = dq.popleft()
        for p in rev[s]:
            if p not in d:
                d[p] = d[s] + 1
                dq.append(p)
    return d, fwd


def _check_against_bfs(space, target, cap, max_start_len):
    d, fwd = _bfs_reference(space, target, cap)
    orc = EditOracle(space, target, max_nodes=10 ** 6)
    n = 0
    for s, succ in fwd.items():
        if len(s[0]) > max_start_len:
            continue
        n += 1
        r = orc.query(*s)
        true = d.get(s)
        if true is None:
            assert not r.reachable, (space.describe(), target, s, r)
            continue
        assert r.reachable and r.exact and r.cost == true, (space.describe(), target, s, true, r)
        opt = {space.STOP} if s[0] == tuple(target) else {a for a, t in succ if d.get(t) == true - 1}
        assert set(r.actions) == opt, (space.describe(), target, s, sorted(opt), sorted(r.actions))
        assert r.canonical in r.actions and r.lower_bound <= true
    return n


SMALL_SPACES = [
    dict(move_encoding="pow2", n_move=1), dict(move_encoding="pow2", n_move=2), dict(move_encoding="pow2", n_move=3),
    dict(move_encoding="dense", n_move=1), dict(move_encoding="dense", n_move=2),
    dict(move_encoding="absolute", n_move=7), dict(move_encoding="absolute", n_move=3),  # 3: limited range
]


@pytest.mark.parametrize("cfg", SMALL_SPACES, ids=lambda c: f"{c['move_encoding']}{c['n_move']}")
@pytest.mark.parametrize("bo", [False, True], ids=["bi", "bo"])
@pytest.mark.parametrize("allow_delete", [True, False], ids=["del", "nodel"])
def test_exhaustive_bfs_agreement_tiny(cfg, bo, allow_delete):
    """(a) cost, reachability and the FULL optimal next-action set equal brute-force BFS for every
    start state (canvas length <= 3 over a 3-token vocabulary, every cursor) and several targets,
    including repeated-token targets and a target that uses the whole vocabulary (no junk value)."""
    space = ActionSpace(n_content=3, backward_only=bo, allow_delete=allow_delete, **cfg)
    n = 0
    for target in [(), (0,), (1, 1), (0, 1, 2), (2, 0, 2)]:
        n += _check_against_bfs(space, target, cap=6, max_start_len=3)
    assert n > 0


@pytest.mark.parametrize("cfg", [dict(move_encoding="pow2", n_move=3), dict(move_encoding="dense", n_move=2),
                                 dict(move_encoding="absolute", n_move=9)], ids=["pow2", "dense", "abs"])
@pytest.mark.parametrize("bo", [False, True], ids=["bi", "bo"])
def test_bfs_agreement_longer(cfg, bo):
    """(a) longer canvases (<= 4) / targets (<= 4) over a binary vocabulary: many repeated values."""
    space = ActionSpace(n_content=2, backward_only=bo, **cfg)
    for target in [(0, 1, 0, 1), (1, 1, 0), (0, 0, 0, 1)]:
        _check_against_bfs(space, target, cap=8, max_start_len=4)


def test_random_tiny_cases_vs_bfs():
    """(a) random targets/vocabularies/encodings, sampled start states."""
    rng = random.Random(123)
    for _ in range(12):
        enc = rng.choice(["pow2", "dense", "absolute"])
        nm = rng.randint(1, 3) if enc != "absolute" else rng.choice([4, 8])
        space = ActionSpace(n_content=rng.choice([2, 3]), move_encoding=enc, n_move=nm,
                            backward_only=rng.random() < 0.5, allow_delete=rng.random() < 0.8)
        target = tuple(rng.randrange(space.n_content) for _ in range(rng.randint(0, 3)))
        _check_against_bfs(space, target, cap=6, max_start_len=3)


def test_incremental_heuristic_matches_fresh_tables():
    rng = random.Random(5)
    for _ in range(300):
        bo, ad = rng.random() < 0.3, rng.random() < 0.8
        Y = tuple(rng.randrange(4) for _ in range(rng.randint(0, 7)))
        T = tuple(rng.randrange(5) for _ in range(rng.randint(0, 7)))
        ypos = {}
        for j, t in enumerate(Y):
            ypos.setdefault(t, []).append(j)
        tb = _Tables(T, Y, ad, bo, ypos)
        for c in range(len(T) + 1):
            for v in range(5):
                T2 = T[:c] + (v,) + T[c:]
                assert tb.h_insert(c, v) == _Tables(T2, Y, ad, bo, ypos).h_at(c + 1)
            if c > 0:
                T3 = T[:c - 1] + T[c:]
                assert tb.h_delete(c) == _Tables(T3, Y, ad, bo, ypos).h_at(c - 1)


# ---------------------------------------------------------------------------
# (b) pure-oracle rollouts from corrupted drafts
# ---------------------------------------------------------------------------
def _corrupt(rng, target, n_content, n_err):
    """Extras (incl. copies of LATER target tokens placed earlier, and duplicates), missing tokens,
    replacements and adjacent swaps."""
    d = list(target)
    for _ in range(n_err):
        k = rng.choice(["later_copy", "extra", "dup", "missing", "replace", "swap"])
        if k == "later_copy" and target:
            j = rng.randrange(len(target))
            d.insert(rng.randint(0, min(j, len(d))), target[j])
        elif k == "extra":
            d.insert(rng.randint(0, len(d)), rng.randrange(n_content))
        elif k == "dup" and d:
            p = rng.randrange(len(d))
            d.insert(p + 1, d[p])
        elif k == "missing" and d:
            d.pop(rng.randrange(len(d)))
        elif k == "replace" and d:
            d[rng.randrange(len(d))] = rng.randrange(n_content)
        elif k == "swap" and len(d) > 1:
            p = rng.randrange(len(d) - 1)
            d[p], d[p + 1] = d[p + 1], d[p]
    return d


ROLLOUT_SPACES = [
    dict(move_encoding="pow2", n_move=7), dict(move_encoding="pow2", n_move=7, backward_only=True),
    dict(move_encoding="dense", n_move=8), dict(move_encoding="dense", n_move=2, backward_only=True),
    dict(move_encoding="absolute", n_move=64), dict(move_encoding="absolute", n_move=64, backward_only=True),
    dict(move_encoding="pow2", n_move=7, allow_delete=False),
]


def _run_rollout(space, prompt, draft, target, sc):
    orc = EditOracle(space, target)
    r0 = orc.query(draft, len(draft) if sc is None else sc)
    tail = list(draft[len(draft) if sc is None else sc:])
    if space.backward_only or not space.allow_delete:
        reach = True
        if space.backward_only and (len(tail) > len(target) or list(target[len(target) - len(tail):]) != tail):
            reach = False
        if not space.allow_delete:  # draft must be a subsequence of the target
            it = iter(target)
            reach = reach and all(any(x == y for y in it) for x in draft)
        assert r0.reachable == reach
        if not reach:
            return None
    acts = oracle_rollout(space, prompt, draft, target, sc, oracle=orc)
    ok, why = validate_actions(space, acts, target, draft, sc)  # replays through the executor: no illegal action
    assert ok, why
    assert acts.count(space.STOP) == 1 and acts[-1] == space.STOP
    if all(r.exact for r in orc._results.values()):
        assert len(acts) == r0.cost  # every step was optimal
    return acts


@pytest.mark.parametrize("cfg", ROLLOUT_SPACES, ids=lambda c: ActionSpace(n_content=1, **c).describe())
def test_pure_oracle_rollouts_generic_corruptions(cfg):
    rng = random.Random(hash(tuple(sorted(cfg.items()))) % 1000)
    space = ActionSpace(n_content=6, **cfg)
    for _ in range(40):
        target = [rng.randrange(space.n_content) for _ in range(rng.randint(0, 16))]
        draft = _corrupt(rng, target, space.n_content, rng.randint(0, 4))
        sc = None if rng.random() < 0.4 else rng.randint(0, len(draft))
        _run_rollout(space, [0], draft, target, sc)


@pytest.mark.parametrize("cfg", [c for c in ROLLOUT_SPACES[:6] if c["n_move"] != 2],
                         ids=lambda c: ActionSpace(n_content=1, **c).describe())
def test_pure_oracle_rollouts_progression_repairs(cfg):
    spec = prog.ProgSpec()
    space = ActionSpace(n_content=spec.n_content, **cfg)
    rng = random.Random(11)
    for _ in range(25):
        a, d, n = prog.sample_target(spec, rng, "train")
        tgt = prog.progression(spec, a, d, n)
        c = prog.corrupt(spec, tgt, rng, prog.sample_kinds(rng, rng.randint(1, 3)))
        sc = rng.choice([None, c.first_error_end, rng.randint(0, len(c.draft))])
        _run_rollout(space, prog.make_prompt(n), c.draft, tgt, sc)


def test_blank_start_generation_is_left_to_right():
    space = ActionSpace(n_content=10, move_encoding="pow2", n_move=4)
    tgt = [3, 1, 4, 1, 5, 9, 2, 6]
    acts = oracle_rollout(space, [], [], tgt)
    assert acts == tgt + [space.STOP]


# ---------------------------------------------------------------------------
# (c) July regression
# ---------------------------------------------------------------------------
class _JulyState:
    """Minimal re-implementation of the July ``EpisodeState`` bookkeeping (edit_env.py:64-139):
    an insert is tagged matched only if it equals the LEFTMOST missing target token of the cursor's
    gap (:88-96); anything else is permanently junk; there is no delete; ``complete()`` counts
    matched tokens (:138-139)."""

    def __init__(self, cont):
        self.cont, self.tags, self.tokens, self.cursor = list(cont), [], [], 0
        self.matched, self.n_matched = [False] * len(cont), 0

    def _gap_next(self):
        left = next((self.tags[i][1] for i in range(self.cursor - 1, -1, -1) if self.tags[i][0] == "M"), -1)
        right = next((self.tags[i][1] for i in range(self.cursor, len(self.tags)) if self.tags[i][0] == "M"),
                     len(self.cont))
        return next((j for j in range(left + 1, right) if not self.matched[j]), None)

    def insert(self, a):
        nxt = self._gap_next()
        if nxt is not None and self.cont[nxt] == a:
            self.tags.insert(self.cursor, ("M", nxt))
            self.matched[nxt] = True
            self.n_matched += 1
        else:
            self.tags.insert(self.cursor, ("J", a))
        self.tokens.insert(self.cursor, a)
        self.cursor += 1

    def move(self, amt):
        self.cursor = max(0, min(len(self.tokens), self.cursor + amt))

    def complete(self):
        return self.n_matched == len(self.cont)


def test_july_bug_regression_minimal():
    """The July policy may plant a far anchor (a LATER target token) first (edit_env.py:193-211); the
    July executor tags it junk because it is not the leftmost missing token; filling the prefix then
    yields matched-count completion on a wrong canvas. The new oracle has no tags: it re-aligns the
    canvas, keeps the anchor, and only accepts canvas == target."""
    target = [10, 11, 12]
    old = _JulyState(target)
    old.insert(12)                      # far anchor: target[2]
    assert old.tags == [("J", 12)]      # valid token for the oracle, junk for the executor
    old.move(-1)
    for t in (10, 11, 12):
        old.insert(t)
    assert old.complete() and old.tokens == [10, 11, 12, 12]   # old criterion: "done", wrong text
    assert not is_complete(old.tokens, target)

    space = ActionSpace(n_content=20, move_encoding="pow2", n_move=4)
    orc = EditOracle(space, target)
    # the anchor state: 12 is kept (aligned to target[2]); optimal = move left, insert 10, 11, STOP
    r = orc.query([12], 1)
    assert r.exact and r.cost == 4 and r.actions == frozenset([space.move_token(-1)])
    acts = oracle_rollout(space, [], [12], target, seed_cursor=1)
    assert space.DELETE not in acts and replay(space, acts, [12], 1).tokens == target
    # the July "complete" canvas is not complete for the new oracle: STOP is not proposed
    r = orc.query([10, 11, 12, 12], 3)
    assert space.STOP not in r.actions and r.actions == frozenset([space.DELETE]) and r.cost == 2
    r = orc.query([10, 11, 12, 12], 4)
    assert space.STOP not in r.actions and space.DELETE in r.actions
    assert replay(space, oracle_rollout(space, [], [10, 11, 12, 12], target, 3), [10, 11, 12, 12], 3).tokens == target


HISTORICAL = Path("<archive>/workspaces/7-3/impl/edit_env.py")
if not HISTORICAL.exists():
    HISTORICAL = Path("<archive>/workspaces/7-3/impl/edit_env.py")


@pytest.mark.skipif(not HISTORICAL.exists(), reason="historical July source not available (read-only reference)")
def test_july_bug_reproduced_on_historical_source_and_repaired():
    """Read-only replay of the archived July oracle (torch import stripped in memory): pure-oracle
    rollouts claim completion on wrong canvases; the new oracle rejects each of those canvases and
    repairs it exactly."""
    scope = {}
    src = HISTORICAL.read_text(encoding="utf-8").replace("import torch\n", "")
    exec(compile(src, "edit_env.py", "exec"), scope)
    State, Oracle, EOR = scope["EpisodeState"], scope["OracleRho"], scope["EOR_ID"]
    target = list(range(10, 20))
    space = ActionSpace(n_content=32, move_encoding="pow2", n_move=10)
    orc = EditOracle(space, target)
    wrong = 0
    for seed in range(5):
        st, o = State([], target), Oracle(seed=seed)
        for _ in range(200):
            a = o.sample(st)
            if a == EOR:
                break
            st.execute(a)
        canvas = st.final_text_tokens()
        assert st.complete()
        if canvas != target:
            wrong += 1
            r = orc.query(canvas, st.cursor)
            assert space.STOP not in r.actions and r.cost > 1
            fixed = replay(space, oracle_rollout(space, [], canvas, target, st.cursor), canvas, st.cursor)
            assert fixed.tokens == target
    assert wrong == 5  # audit (notes/audits/revision-audit-july-data-2026-09-23.md): all five seeds wrong


# ---------------------------------------------------------------------------
# (d) STOP only on the exact target; DAgger labels on arbitrary histories
# ---------------------------------------------------------------------------
def test_stop_proposed_iff_canvas_equals_target():
    rng = random.Random(9)
    for enc, nm in [("pow2", 5), ("dense", 3), ("absolute", 40)]:
        space = ActionSpace(n_content=5, move_encoding=enc, n_move=nm)
        for _ in range(150):
            target = [rng.randrange(5) for _ in range(rng.randint(0, 8))]
            T = _corrupt(rng, target, 5, rng.randint(0, 3))
            r = EditOracle(space, target).query(T, rng.randint(0, len(T)))
            assert (space.STOP in r.actions) == (T == target)
            if T == target:
                assert r.actions == frozenset([space.STOP]) and r.cost == 1
        # same multiset / same length / same matched count is not enough
        orc = EditOracle(space, [1, 2, 3])
        for T in ([3, 2, 1], [1, 2, 3, 3], [1, 2], [1, 3, 2]):
            assert space.STOP not in orc.query(T, len(T)).actions


def test_labels_for_random_histories_are_consistent():
    """Model-like histories (random legal actions incl. premature STOP): each exact label set is exactly
    the set of actions that lower the exact remaining cost by one."""
    rng = random.Random(21)
    for cfg in [dict(move_encoding="pow2", n_move=4), dict(move_encoding="absolute", n_move=24),
                dict(move_encoding="dense", n_move=3, backward_only=True)]:
        space = ActionSpace(n_content=5, **cfg)
        for _ in range(6):
            target = [rng.randrange(5) for _ in range(rng.randint(1, 8))]
            draft = _corrupt(rng, target, 5, 2)
            cv = Canvas.from_draft(space, draft)
            hist = []
            for _ in range(rng.randint(0, 12)):
                legal = [a for a in range(space.vocab_size) if cv.is_legal(a)]
                a = rng.choice(legal)
                cv.apply(a)
                hist.append(a)
                if cv.stopped:
                    break
            orc = EditOracle(space, target, max_nodes=10 ** 6)
            labels = oracle_labels_for_history(space, draft, target, hist, oracle=orc)
            assert len(labels) == len(hist) + (0 if hist and hist[-1] == space.STOP else 1)
            cv = Canvas.from_draft(space, draft)
            for t, lab in enumerate(labels):
                if lab.reachable:
                    assert lab.exact
                    assert (space.STOP in lab.actions) == (cv.tokens == target)
                    for a in range(space.vocab_size):
                        if a in (space.SEP, space.EDIT) or not cv.is_legal(a):
                            assert a not in lab.actions
                            continue
                        if a == space.STOP:
                            continue
                        nxt = cv.copy()
                        nxt.apply(a)
                        r2 = orc.query(nxt.tokens, nxt.cursor)
                        c2 = r2.cost if r2.reachable else INF
                        assert (a in lab.actions) == (1 + c2 == lab.cost)
                if t < len(hist):
                    cv.apply(hist[t])


def test_budget_fallback_still_reaches_target():
    space = ActionSpace(n_content=8, move_encoding="pow2", n_move=3)
    rng = random.Random(4)
    target = [rng.randrange(8) for _ in range(20)]
    draft = _corrupt(rng, target, 8, 8)
    orc = EditOracle(space, target, max_nodes=0)
    r = orc.query(draft, len(draft))
    assert not r.exact and r.reachable and r.cost >= r.lower_bound
    acts = oracle_rollout(space, [], draft, target, oracle=orc)
    ok, why = validate_actions(space, acts, target, draft)
    assert ok, why
