"""Token-level edit-script (patch) and fill-in-the-middle (FIM) formats for the AR task adaptations (task 30).

Both are ordinary left-to-right GPT-2-id targets typed by an AR model after a read-only prompt; the delimiters
are single GPT-2 tokens that never occur in the IteraTeR/CoEdIT train pools (counted over 498,283 records):

* PATCH_HUNK  25248 " @@"   starts a hunk
* PATCH_SEP   13163 " >>>"  separates a hunk's search text from its replacement
* FIM_SUF      1421 "===="  (prompt) prefix FIM_SUF suffix FIM_MID ; target = the middle
* FIM_MID     16471 ">>>>"

Patch semantics (search/replace applied sequentially, like LLM "search/replace" blocks): a patch is
``(PATCH_HUNK old PATCH_SEP new)*``. With a read pointer p (initially 0) over ``before``, each hunk replaces the
FIRST occurrence of ``old`` at or after p by ``new`` and moves p past it; an empty ``old`` inserts ``new`` at p.
Unchanged text is never retyped, so the target length scales with the edit, not the text (the point of an
explicit edit-script baseline). ``make_patch`` emits minimal left context (then right context, then merged
hunks) so that every hunk is found at the right position; ``apply_patch(before, make_patch(before, after))``
reproduces ``after`` exactly (tests/test_patchfmt.py).
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Sequence, Tuple

PATCH_HUNK, PATCH_SEP = 25248, 13163
FIM_SUF, FIM_MID = 1421, 16471
MAX_CTX, MAX_RCTX = 48, 8  # left / right context tokens tried before merging with the next hunk


def _find(seq: Sequence[int], sub: Sequence[int], start: int) -> int:
    n, m = len(seq), len(sub)
    if m == 0:
        return start
    first = sub[0]
    for i in range(start, n - m + 1):
        if seq[i] == first and list(seq[i:i + m]) == list(sub):
            return i
    return -1


def _blocks(before: Sequence[int], after: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    sm = SequenceMatcher(None, list(before), list(after), autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def make_patch(before: Sequence[int], after: Sequence[int]) -> List[int]:
    before, after = list(before), list(after)
    blocks = _blocks(before, after)
    out: List[int] = []
    p, k = 0, 0
    while k < len(blocks):
        i1, i2, j1, j2 = blocks[k]
        nxt = blocks[k + 1][0] if k + 1 < len(blocks) else len(before)
        hunk = None
        for R in range(0, min(MAX_RCTX, nxt - i2) + 1):          # right context only up to the next block
            for L in range(0, min(MAX_CTX, i1 - p) + 1):
                s, e = i1 - L, i2 + R
                old = before[s:e]
                if (old or s == p) and _find(before, old, p) == s:
                    hunk = (s, e, before[s:i1] + after[j1:j2] + before[i2:e])
                    break
            if hunk:
                break
        if hunk is None:
            if k + 1 < len(blocks):                               # merge with the next block and retry
                blocks[k + 1] = (i1, blocks[k + 1][1], j1, blocks[k + 1][3])
                k += 1
                continue
            hunk = (p, len(before), before[p:i1] + after[j1:j2] + before[i2:])  # rest of the text: found at p
        s, e, new = hunk
        out += [PATCH_HUNK] + before[s:e] + [PATCH_SEP] + new
        p = e
        k += 1
    return out


def parse_patch(patch: Sequence[int]) -> Tuple[List[Tuple[List[int], List[int]]], bool]:
    """-> ([(old, new)], well_formed). Tokens before the first hunk or a hunk without separator: malformed."""
    hunks, ok, cur, in_new = [], True, None, False
    for t in patch:
        if t == PATCH_HUNK:
            if cur is not None:
                ok &= in_new
                if in_new:
                    hunks.append(cur)
            cur, in_new = ([], []), False
        elif cur is None:
            ok = False
        elif t == PATCH_SEP and not in_new:
            in_new = True
        else:
            (cur[1] if in_new else cur[0]).append(t)
    if cur is not None:
        ok &= in_new
        if in_new:
            hunks.append(cur)
    return hunks, ok


def apply_patch(before: Sequence[int], patch: Sequence[int]) -> Tuple[List[int], dict]:
    """Apply sequentially; a hunk whose search text is not found is skipped (counted). -> (text, info)."""
    before = list(before)
    hunks, well = parse_patch(patch)
    out, p, miss = [], 0, 0
    for old, new in hunks:
        s = _find(before, old, p)
        if s < 0:
            miss += 1
            continue
        out += before[p:s] + new
        p = s + len(old)
    out += before[p:]
    return out, dict(well_formed=well, n_hunks=len(hunks), n_missed=miss, applied=well and miss == 0)


def fim_prompt(prefix: Sequence[int], suffix: Sequence[int]) -> List[int]:
    return list(prefix) + [FIM_SUF] + list(suffix) + [FIM_MID]
