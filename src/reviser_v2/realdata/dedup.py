"""Text normalization, duplicate detection, and eval-overlap filtering (pure functions).

Normalization for matching: lowercase + collapse all whitespace runs to one space + strip.
Empty normalized strings (e.g. whole-sentence deletions with an empty after_text) never count as
overlap, otherwise every deletion in train would "match" every deletion in test. A second, aggressive
alnum-only key (>= 20 alphanumerics) additionally catches spacing/punctuation variants.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def key(s: str) -> bytes:
    """Compact 12-byte digest of the normalized text ("" maps to b"")."""
    n = norm_text(s)
    return hashlib.blake2b(n.encode("utf-8"), digest_size=12).digest() if n else b""


def pair_key(before: str, after: str) -> bytes:
    return hashlib.blake2b((norm_text(before) + "\x00" + norm_text(after)).encode("utf-8"), digest_size=12).digest()


def exact_pair_key(before: str, after: str) -> bytes:
    return hashlib.blake2b((before + "\x00" + after).encode("utf-8"), digest_size=12).digest()


def rkeys(r) -> Tuple[bytes, bytes]:
    """(before key, after key) of a record, cached on dict records under ``_k`` (strip before writing)."""
    k = r.get("_k") if isinstance(r, dict) else None
    if k is None:
        k = (key(r["before_text"]), key(r["after_text"]))
        if isinstance(r, dict):
            r["_k"] = k
    return k


def text_keys(records: Iterable[Mapping]) -> Tuple[Set[bytes], Set[bytes]]:
    """(normalized before keys, normalized after keys), empties excluded."""
    b, a = set(), set()
    for r in records:
        kb, ka = rkeys(r)
        if kb:
            b.add(kb)
        if ka:
            a.add(ka)
    return b, a


def overlap_counts(train: Sequence[Mapping], other_before: Set[bytes], other_after: Set[bytes]) -> Dict[str, int]:
    """How many ``train`` records share normalized text with another split.

    by_before: train.before in other.before; by_after: train.after in other.after;
    any_text: train.before or train.after in (other.before | other.after).
    """
    both = other_before | other_after
    nb = na = nany = 0
    for r in train:
        kb, ka = rkeys(r)
        nb += bool(kb) and kb in other_before
        na += bool(ka) and ka in other_after
        nany += (bool(kb) and kb in both) or (bool(ka) and ka in both)
    return {"n": len(train), "by_before": nb, "by_after": na, "any_text": nany}


def within_split_duplicates(records: Sequence[Mapping]) -> Dict[str, int]:
    ex = Counter(exact_pair_key(r["before_text"], r["after_text"]) for r in records)
    nm = Counter(pair_key(r["before_text"], r["after_text"]) for r in records)
    return {"n": len(records), "exact_pair_dups": sum(c - 1 for c in ex.values()),
            "normalized_pair_dups": sum(c - 1 for c in nm.values())}


ALNUM_MIN = 20


def alnum_key(s: str) -> bytes:
    """Aggressive key: lowercase [a-z0-9] only (catches tokenization/punctuation-spacing variants such
    as "paper , we" vs "paper, we"). Strings with < ALNUM_MIN such characters never match."""
    n = re.sub(r"[^a-z0-9]", "", s.lower())
    return hashlib.blake2b(n.encode(), digest_size=12).digest() if len(n) >= ALNUM_MIN else b""


def alnum_keys(records: Iterable[Mapping]) -> Set[bytes]:
    return {alnum_key(r[f]) for r in records for f in ("before_text", "after_text")} - {b""}


def filter_train(train: Sequence[Mapping], eval_text_keys: Set[bytes], eval_doc_ids: Set[str] = frozenset(),
                 use_doc_ids: bool = False, eval_alnum_keys: Set[bytes] = frozenset()
                 ) -> Tuple[List[Mapping], Dict[str, int], List[Tuple[str, Mapping]]]:
    """Drop train records contaminated by any eval split, then within-train normalized pair dups.

    Order of checks (a record is counted under the first reason that fires):
      1. ``eval_text``:  normalized before OR after text equals any eval before/after text.
      2. ``eval_alnum``: (optional) alnum-only key of before OR after equals an eval alnum key.
      3. ``eval_doc``:   (optional) the record's doc_id occurs in an eval split (document leakage).
      4. ``dup_pair``:   normalized (before, after) already kept earlier in this train split.
    Returns (kept, counts, removed as (reason, record)).
    """
    kept: List[Mapping] = []
    gone: List[Tuple[str, Mapping]] = []
    seen: Set[bytes] = set()
    removed: Dict[str, int] = defaultdict(int)
    for r in train:
        kb, ka = rkeys(r)
        why = None
        if (kb and kb in eval_text_keys) or (ka and ka in eval_text_keys):
            why = "eval_text"
        elif eval_alnum_keys and (alnum_key(r["before_text"]) in eval_alnum_keys or alnum_key(r["after_text"]) in eval_alnum_keys):
            why = "eval_alnum"
        elif use_doc_ids and r["doc_id"] in eval_doc_ids:
            why = "eval_doc"
        else:
            pk = pair_key(r["before_text"], r["after_text"])
            if pk in seen:
                why = "dup_pair"
            else:
                seen.add(pk)
        if why:
            removed[why] += 1
            gone.append((why, r))
        else:
            kept.append(r)
    return kept, {"in": len(train), "kept": len(kept), **{f"removed_{k}": v for k, v in removed.items()}}, gone
