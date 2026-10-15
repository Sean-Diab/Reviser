"""Toy "progression language" for mechanical deletion/repair diagnostics (tasks 10-15).

A sentence is an arithmetic progression mod M: a, a+d, ..., a+(n-1)d with d in 1..D. The prompt
gives the length n (as token ``n``), so blank-start generation must produce a valid progression of
exactly that length and every corruption below is identifiable from the draft plus n (checked
explicitly for evaluation sets). Values repeat when d*n >= M, so repeated-token cases arise
naturally. Marker tokens (ids >= M) are deliberately "marked" extra symbols.

Content vocabulary: 0..M-1 progression values, M..M+n_markers-1 markers, then length tokens are
reused from the value range (n < M), so n_content = M + n_markers.

Split: targets are assigned to train/dev/test by a hash of (a, d, n) BEFORE deriving corrupted
variants, so all variants of one target stay in one split.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ProgSpec:
    M: int = 60
    D: int = 6
    n_markers: int = 4
    n_min: int = 8
    n_max: int = 40

    @property
    def n_content(self) -> int:
        return self.M + self.n_markers


def progression(spec: ProgSpec, a: int, d: int, n: int) -> List[int]:
    return [(a + i * d) % spec.M for i in range(n)]


def split_of(a: int, d: int, n: int, salt: str = "prog-v1") -> str:
    h = int(hashlib.sha256(f"{salt}:{a}:{d}:{n}".encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 10 else ("dev" if h < 20 else "train")


def sample_target(spec: ProgSpec, rng: random.Random, split: str, n: Optional[int] = None) -> Tuple[int, int, int]:
    while True:
        nn = n if n is not None else rng.randint(spec.n_min, spec.n_max)
        a, d = rng.randrange(spec.M), rng.randint(1, spec.D)
        if split_of(a, d, nn) == split:
            return a, d, nn


def is_valid(spec: ProgSpec, seq: Sequence[int], n: int) -> bool:
    if len(seq) != n or n < 2:
        return len(seq) == n and all(0 <= t < spec.M for t in seq)
    if any(not 0 <= t < spec.M for t in seq):
        return False
    d = (seq[1] - seq[0]) % spec.M
    if not 1 <= d <= spec.D:
        return False
    return all((seq[i + 1] - seq[i]) % spec.M == d for i in range(n - 1))


def lev_banded(a: Sequence[int], b: Sequence[int], k: int) -> int:
    """Levenshtein distance if <= k, else k+1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return k + 1
    INF = k + 1
    prev = {j: j for j in range(0, min(lb, k) + 1)}
    for i in range(1, la + 1):
        cur = {}
        lo, hi = max(0, i - k), min(lb, i + k)
        for j in range(lo, hi + 1):
            best = INF
            if j == 0:
                best = i
            else:
                if j - 1 in prev:
                    best = min(best, prev[j - 1] + (a[i - 1] != b[j - 1]))
                if j - 1 in cur:
                    best = min(best, cur[j - 1] + 1)
            if j in prev:
                best = min(best, prev[j] + 1)
            cur[j] = min(best, INF)
        prev = cur
    return min(prev.get(lb, INF), INF)


def identifiable(spec: ProgSpec, draft: Sequence[int], target: Sequence[int], n: int) -> bool:
    """Target is the unique length-n progression at minimal edit distance from the draft."""
    from collections import Counter
    k = lev_banded(draft, target, 6)
    if k > 6:
        return False
    dc = Counter(draft)
    for a in range(spec.M):
        for d in range(1, spec.D + 1):
            c = progression(spec, a, d, n)
            if c == list(target):
                continue
            cc = Counter(c)
            # multiset lower bound on edit distance prunes most candidates cheaply
            if max(sum((dc - cc).values()), sum((cc - dc).values())) > k:
                continue
            if lev_banded(draft, c, k) <= k:
                return False
    return True


ERROR_TYPES = ("marked_extra", "extra", "duplicate", "missing", "replace")


@dataclass
class Corruption:
    kind: str
    draft: List[int]
    bad_draft_idx: List[int]      # draft positions of erroneous instances (extras / replacements)
    missing_target_idx: List[int]  # target positions absent from the draft
    edge: bool
    first_error_end: int          # draft boundary right after the first erroneous region


def corrupt(spec: ProgSpec, target: List[int], rng: random.Random, kinds: Sequence[str]) -> Corruption:
    """Apply errors in sequence (positions tracked on a list of (token, src) pairs).

    src >= 0: target index; src = -1: inserted error; src = -2: replacement error.
    """
    items: List[Tuple[int, int]] = [(t, i) for i, t in enumerate(target)]
    missing: List[int] = []
    edge = False
    for kind in kinds:
        L = len(items)
        if kind == "marked_extra":
            p = rng.randint(0, L)
            items.insert(p, (spec.M + rng.randrange(spec.n_markers), -1))
        elif kind == "extra":
            p = rng.randint(0, L)
            items.insert(p, (rng.randrange(spec.M), -1))
        elif kind == "duplicate":
            p = rng.randrange(L)
            items.insert(p + 1, (items[p][0], -1))
        elif kind == "missing":
            cands = [i for i, (_, s) in enumerate(items) if s >= 0]
            p = rng.choice(cands)
            missing.append(items[p][1])
            items.pop(p)
        elif kind == "replace":
            cands = [i for i, (_, s) in enumerate(items) if s >= 0]
            p = rng.choice(cands)
            old = items[p][0]
            new = rng.randrange(spec.M)
            while new == old:
                new = rng.randrange(spec.M)
            missing.append(items[p][1])
            items[p] = (new, -2)
        else:
            raise ValueError(kind)
        if p == 0 or p >= len(items) - 1:
            edge = True
    draft = [t for t, _ in items]
    bad = [i for i, (_, s) in enumerate(items) if s < 0]
    # first erroneous region end: right after the first bad instance, or at the missing position
    ends = [i + 1 for i in bad]
    for mt in missing:
        # boundary in draft where target index mt should be: after all items with src < mt
        ends.append(sum(1 for _, s in items if 0 <= s < mt))
    first_end = min(ends) if ends else len(draft)
    return Corruption("+".join(kinds), draft, bad, sorted(missing), edge, first_end)


def sample_kinds(rng: random.Random, n_errors: int, allow_marked: bool = True) -> List[str]:
    pool = ERROR_TYPES if allow_marked else ERROR_TYPES[1:]
    return [rng.choice(pool) for _ in range(n_errors)]


@dataclass
class RepairItem:
    uid: str
    split: str
    a: int
    d: int
    n: int
    prompt: List[int]
    target: List[int]
    draft: List[int]
    kind: str
    n_errors: int
    edge: bool
    bad_draft_idx: List[int]
    missing_target_idx: List[int]
    seed_cursor: Optional[int]   # None = cursor at end of draft (navigation required)
    category: str


def make_prompt(n: int) -> List[int]:
    return [n]


def build_diagnostic_suite(spec: ProgSpec, split: str, seed: int, per_cell: int = 100) -> List[RepairItem]:
    """Held-out repair diagnostics with identifiability-checked targets.

    Cells: clean controls; each single error type (interior/edge mixed, identifiability checked);
    multi-error (2-3); repeated-token regime (d*n >= M); long (n >= 32); and cursor-at-error
    copies of single-error items. Each item is its own target (a,d,n) drawn from ``split``.
    """
    rng = random.Random(f"diag:{split}:{seed}")
    items: List[RepairItem] = []

    def add(cat, kinds, n=None, cursor_at_error=False, need_repeat=False):
        tries = 0
        made = 0
        while made < per_cell:
            tries += 1
            if tries > per_cell * 200:
                raise RuntimeError(f"could not fill cell {cat}")
            a, d, nn = sample_target(spec, rng, split, n)
            if need_repeat and d * nn < spec.M:
                continue
            tgt = progression(spec, a, d, nn)
            if kinds:
                c = corrupt(spec, tgt, rng, kinds)
                if c.draft == tgt or not identifiable(spec, c.draft, tgt, nn):
                    continue
            else:
                c = Corruption("clean", list(tgt), [], [], False, len(tgt))
            sc = c.first_error_end if cursor_at_error else None
            items.append(RepairItem(f"{split}-{cat}-{made}", split, a, d, nn, make_prompt(nn), tgt, c.draft,
                                    c.kind, len(kinds), c.edge, c.bad_draft_idx, c.missing_target_idx, sc, cat))
            made += 1

    add("clean", [])
    for k in ERROR_TYPES:
        add(f"single_{k}", [k])
        add(f"single_{k}_at_cursor", [k], cursor_at_error=True)
    for m in (2, 3):
        made = 0
        tries = 0
        while made < per_cell:
            tries += 1
            if tries > per_cell * 400:
                raise RuntimeError("multi cell")
            kinds = sample_kinds(rng, m, allow_marked=False)
            a, d, nn = sample_target(spec, rng, split)
            if nn < 16:
                continue
            tgt = progression(spec, a, d, nn)
            c = corrupt(spec, tgt, rng, kinds)
            if c.draft == tgt or not identifiable(spec, c.draft, tgt, nn):
                continue
            items.append(RepairItem(f"{split}-multi{m}-{made}", split, a, d, nn, make_prompt(nn), tgt, c.draft,
                                    c.kind, m, c.edge, c.bad_draft_idx, c.missing_target_idx, None, f"multi{m}"))
            made += 1
    add("repeat_tokens_replace", ["replace"], need_repeat=True)
    add("long_extra", ["extra"], n=spec.n_max)
    add("clean_long", [], n=spec.n_max)
    return items
