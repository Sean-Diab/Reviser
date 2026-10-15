"""Held-out evaluation sets built deterministically from the real-edit dev pool (tasks 23 / 27).

* ``synth_real`` (task 23): realistic, non-neural errors in clean dev texts, one error family per record, so
  ``evaluate_edits.aggregate`` reports each family as its own intent. Families (``RealisticCorruptor``):
    typo_char      keyboard / spelling typo inside a word (adjacent-key substitution, transposition, dropped or
                   doubled letter). The word is re-encoded, so it usually splits into several GPT-2 tokens.
    typo_realword  real-word error: the word token is replaced by another single-token GPT-2 word at
                   Damerau-Levenshtein distance 1, mined from the vocabulary ("from" -> "form", "than" -> "that").
    confusion      confusable pairs (their/there/they're, its/it's, a/an, then/than, affect/effect, ...).
    morph          inflection / agreement: -s, -ed, -ing variants that are GPT-2 word tokens, and is/are,
                   was/were, has/have, does/do, this/these.
    word_order     two adjacent words swapped.
    fw_drop        one function word (the, a, of, to, and, is, ...) dropped.
    fw_dup         one function word duplicated ("the the").
    punct          comma dropped / spurious comma / period -> comma / final period dropped / apostrophe dropped.
    case           a capitalised word lower-cased ("i", sentence start) or a lower-case word capitalised.
    mixed          2-3 errors of distinct random families.
    clean          no error (control; measures over-editing; target == draft).
  Layout: prompt = FIX_INSTR (as ``synth``), draft = corrupted text, target = clean text. AR layout:
  ``ar_prompt(FIX_INSTR, draft)`` (same as the ``synth_ar`` training kind).
* ``infill`` / ``infill_pos`` (task 27): a contiguous word-aligned span of exactly L = 1..12 GPT-2 tokens is
  removed from a clean dev text (L balanced over records, unknown to the model). prompt = INFILL_INSTR
  ("Fill in the missing words"), draft = gapped text, target = clean text. ``infill_pos`` is the same
  records with the initial cursor at the gap (``Request.seed_cursor``; edit layout only).
  Extra per-row metrics: ``infill_metrics``.
* ``infill2`` / ``infill2_pos`` (task 27, multi-gap): two disjoint word-aligned spans of L1, L2 = 1..6 GPT-2 tokens
  (all 36 pairs balanced over records; total 2..12 as in ``infill``) are removed, with >= 3 fixed tokens between
  them; neither position nor lengths are given (same prompt as ``infill``). ``infill2_pos`` is the same records for
  the oracle-position FIM baseline only (the two gaps filled sequentially, left gap first; evaluate_edits).
  Extra per-row metrics: ``multigap_metrics``.

``RealisticCorruptor.corrupt`` is written to be reusable as a training corruption (``synth_repair`` variant).
"""
from __future__ import annotations

import functools
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
INFILL_INSTR = [33762, 287, 262, 4814, 2456]   # GPT-2 ids of "Fill in the missing words"
DEFAULT_SEED = 20260924

FUNCTION_WORDS = ("the", "a", "an", "of", "to", "and", "is", "in", "for", "that", "on", "with", "as", "are",
                  "was", "it", "be", "by")
CONFUSION_SETS = (("their", "there", "they're"), ("its", "it's"), ("your", "you're"), ("then", "than"),
                  ("affect", "effect"), ("to", "too"), ("lose", "loose"), ("whose", "who's"), ("accept", "except"),
                  ("were", "where"), ("a", "an"), ("of", "off"), ("advice", "advise"), ("principal", "principle"),
                  ("weather", "whether"), ("quiet", "quite"), ("lead", "led"), ("site", "sight", "cite"),
                  ("who", "whom"), ("fewer", "less"))
AGREEMENT = (("is", "are"), ("was", "were"), ("has", "have"), ("does", "do"), ("this", "these"))
CONTRACTIONS = ("'t", "'s", "'re", "'ll", "'ve", "'m", "'d")
FAMILIES = ("typo_char", "typo_realword", "confusion", "morph", "word_order", "fw_drop", "fw_dup", "punct", "case")
SYNTH_REAL_INTENTS = FAMILIES + ("mixed", "clean")
N_DEFAULT = {"synth_real": 1100, "infill": 1200, "infill_pos": 1200, "infill2": 1200, "infill2_pos": 1200}
_WORD = re.compile(r" ?[A-Za-z]+")
_KB = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _kb_neighbours() -> Dict[str, str]:
    pos = {ch: (r, c + 0.25 * r + (0.25 if r == 2 else 0)) for r, row in enumerate(_KB) for c, ch in enumerate(row)}
    return {a: "".join(sorted(b for b, (r2, x2) in pos.items() if b != a and abs(r2 - ra) <= 1 and abs(x2 - xa) <= 1.0))
            for a, (ra, xa) in pos.items()}


def load_gpt2_tokenizer(offline: bool = True):
    """The Reviser GPT-2 tokenizer from data/tokenizers/gpt2 (remote boxes have no HF cache). ``offline`` sets
    HF_HUB_OFFLINE=1 if unset; pass False when a hub model (the continuation scorer) is loaded later."""
    import os
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2TokenizerFast
    tdir = ROOT / "data" / "tokenizers" / "gpt2"
    return GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")


@functools.lru_cache(maxsize=4)
def dev_pool(seed: int = DEFAULT_SEED):
    """The real-edit dev pool [(instruction_ids, before_ids, after_ids)] (same construction as ``synth``)."""
    from .realdata.editscript import gpt2_action_space
    from .tasks.realedit import RealEditSource
    return RealEditSource(gpt2_action_space(), {"max_pair_tokens": 256}, [{"kind": "synth_repair", "weight": 1}],
                          seed, "dev").pool


class RealisticCorruptor:
    """Realistic single errors on GPT-2 id sequences (no neural generator, no downloads).

    Ids are split into segments: a segment starts at every token except an ASCII-letter token that continues an
    ASCII-letter token (a word split into BPE pieces). A modified segment is re-encoded on its own with the
    GPT-2 tokenizer; GPT-2's pre-tokeniser never merges across a word boundary, so this equals what tokenising
    the corrupted text would give locally."""

    def __init__(self, tok, max_realword_id: int = 20000, max_morph_id: int = 30000):
        self.tok = tok
        self.words: Dict[str, int] = {}             # lower-case body -> id of the single token "Ġbody"
        for s, i in tok.get_vocab().items():
            if len(s) > 1 and s[0] == "Ġ" and s[1:].isascii() and s[1:].isalpha() and s[1:].islower():
                self.words[s[1:]] = i
        self.max_realword_id, self.max_morph_id = max_realword_id, max_morph_id
        self.kb = _kb_neighbours()
        self._ed1: Dict[str, List[str]] = {}
        self._forms = []                              # (ids, set_idx, member_idx, lead, cap)
        for si, cs in enumerate(CONFUSION_SETS + AGREEMENT):
            for mi, w in enumerate(cs):
                for lead in (" ", ""):
                    for cap in (False, True):
                        ids = tok.encode(lead + (w[0].upper() + w[1:] if cap else w))
                        self._forms.append((ids, si, mi, lead, cap))
        self._forms.sort(key=lambda f: -len(f[0]))
        self._n_conf = len(CONFUSION_SETS)

    # ------------------------------------------------------------------ segmentation
    def tokstrs(self, ids: Sequence[int]) -> List[str]:
        return self.tok.convert_ids_to_tokens(list(ids))

    def segments(self, ids: Sequence[int]) -> List[Tuple[int, int, str]]:
        ts = self.tokstrs(ids)
        segs: List[List] = []
        for i, s in enumerate(ts):
            if segs and s.isascii() and s.isalpha() and ts[i - 1][-1:].isascii() and ts[i - 1][-1:].isalpha():
                segs[-1][1] = i + 1
            else:
                segs.append([i, i + 1])
        return [(a, b, "".join(ts[a:b]).replace("Ġ", " ")) for a, b in segs]

    @staticmethod
    def _boundary_ok(ts: List[str], e: int) -> bool:
        """The segment ending at e is a whole word (not followed by more letters, e.g. non-ASCII byte pieces)."""
        return e == len(ts) or ts[e][0] in "ĠĊĉ" or ts[e][0] in string.punctuation or ts[e][0].isdigit()

    def _words(self, ids, segs, lead_space=True) -> List[int]:
        """Indices of segments that are whole ASCII words (with a leading space if ``lead_space``)."""
        ts = self.tokstrs(ids)
        return [k for k, (a, b, t) in enumerate(segs) if _WORD.fullmatch(t) and (t[0] == " " or not lead_space)
                and self._boundary_ok(ts, b)]

    def _replace(self, ids, a, b, text: str) -> List[int]:
        return list(ids[:a]) + (self.tok.encode(text) if text else []) + list(ids[b:])

    # ------------------------------------------------------------------ families
    def _typo_char(self, ids, segs, rng):
        cand = [k for k in self._words(ids, segs) if len(segs[k][2]) >= 5 and segs[k][2][1:].islower()]
        if not cand:
            return None
        a, b, t = segs[rng.choice(cand)]
        w = t[1:]
        op = rng.choice(["sub", "transpose", "drop", "double"])
        i = rng.randrange(1, len(w))                  # keep the first letter (as most real typos do)
        if op == "sub":
            w = w[:i] + rng.choice(self.kb.get(w[i], "e")) + w[i + 1:]
        elif op == "transpose" and i < len(w) - 1:
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif op == "drop":
            w = w[:i] + w[i + 1:]
        else:
            w = w[:i] + w[i] + w[i:]
        return self._replace(ids, a, b, " " + w)

    def ed1(self, w: str) -> List[str]:
        """Single-token GPT-2 words at Damerau-Levenshtein distance 1 from w (frequent ids only), sorted."""
        if w not in self._ed1:
            L = string.ascii_lowercase
            sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
            v = {l + r[1:] for l, r in sp if r} | {l + r[1] + r[0] + r[2:] for l, r in sp if len(r) > 1} \
                | {l + c + r[1:] for l, r in sp if r for c in L} | {l + c + r for l, r in sp for c in L}
            v.discard(w)
            self._ed1[w] = sorted(x for x in v if len(x) >= 2 and self.words.get(x, 10 ** 9) < self.max_realword_id)
        return self._ed1[w]

    def _typo_realword(self, ids, segs, rng):
        cand = [k for k in self._words(ids, segs) if len(segs[k][2]) >= 4 and segs[k][2][1:] in self.words
                and self.ed1(segs[k][2][1:])]
        if not cand:
            return None
        a, b, t = segs[rng.choice(cand)]
        return self._replace(ids, a, b, " " + rng.choice(self.ed1(t[1:])))

    def _confusion(self, ids, segs, rng, agreement=False):
        starts = {a: k for k, (a, b, t) in enumerate(segs)}
        ends = {b for a, b, t in segs}
        found: Dict[int, list] = {}
        for a in starts:
            for f_ids, si, mi, lead, cap in self._forms:
                if (si >= self._n_conf) != agreement or (lead == "" and a != 0):
                    continue
                e = a + len(f_ids)
                if list(ids[a:e]) == f_ids and e in ends:
                    found.setdefault(si, []).append((a, e, mi, lead, cap))
                    break
        if not found:
            return None
        si = rng.choice(sorted(found))
        a, e, mi, lead, cap = rng.choice(found[si])
        cs = (CONFUSION_SETS + AGREEMENT)[si]
        w = rng.choice([x for j, x in enumerate(cs) if j != mi])
        return self._replace(ids, a, e, lead + (w[0].upper() + w[1:] if cap else w))

    def morph_variants(self, w: str) -> List[str]:
        v = {w + "s", w + "ed", w + "d", w + "ing"}
        if w.endswith("s") and not w.endswith("ss"):
            v.add(w[:-1])
        if w.endswith("ed"):
            v |= {w[:-2], w[:-1], w[:-2] + "ing", w[:-1] + "ing", w[:-2] + "s"}
        if w.endswith("ing"):
            v |= {w[:-3], w[:-3] + "e", w[:-3] + "ed", w[:-3] + "s", w[:-3] + "es"}
        v.discard(w)
        return sorted(x for x in v if len(x) >= 2 and self.words.get(x, 10 ** 9) < self.max_morph_id)

    def _morph(self, ids, segs, rng):
        if rng.random() < 0.3:
            r = self._confusion(ids, segs, rng, agreement=True)
            if r is not None:
                return r
        cand = [k for k in self._words(ids, segs) if len(segs[k][2]) >= 4 and segs[k][2][1:] in self.words
                and self.morph_variants(segs[k][2][1:])]
        if not cand:
            return self._confusion(ids, segs, rng, agreement=True)
        a, b, t = segs[rng.choice(cand)]
        return self._replace(ids, a, b, " " + rng.choice(self.morph_variants(t[1:])))

    def _word_order(self, ids, segs, rng):
        w = set(self._words(ids, segs))
        cand = [k for k in w if k + 1 in w and segs[k][2] != segs[k + 1][2]]
        if not cand:
            return None
        k = rng.choice(sorted(cand))
        (a, _, t1), (_, b, t2) = segs[k], segs[k + 1]
        return list(ids[:a]) + self.tok.encode(t2) + self.tok.encode(t1) + list(ids[b:])

    def _fw(self, ids, segs, rng, dup):
        cand = [k for k in self._words(ids, segs) if segs[k][2][1:] in FUNCTION_WORDS]
        if not cand:
            return None
        a, b, t = segs[rng.choice(cand)]
        return list(ids[:b]) + list(ids[a:b]) + list(ids[b:]) if dup else list(ids[:a]) + list(ids[b:])

    def _punct(self, ids, segs, rng):
        ws = set(self._words(ids, segs))
        opts = []
        for k, (a, b, t) in enumerate(segs):
            if t == ",":
                opts.append(("drop_comma", k))
            if t == "." and k + 1 < len(segs) and k + 1 in ws:
                opts.append(("period_comma", k))
            if t == "." and k == len(segs) - 1:
                opts.append(("drop_final_period", k))
            if t in CONTRACTIONS and k > 0 and _WORD.fullmatch(segs[k - 1][2]):
                opts.append(("drop_apostrophe", k))
            if k in ws and k + 1 in ws:
                opts.append(("add_comma", k))
        if not opts:
            return None
        kinds = sorted({o[0] for o in opts})
        kind = rng.choice(kinds)
        k = rng.choice([kk for o, kk in opts if o == kind])
        a, b, t = segs[k]
        if kind in ("drop_comma", "drop_final_period"):
            return list(ids[:a]) + list(ids[b:])
        if kind == "period_comma":
            return self._replace(ids, a, b, ",")
        if kind == "add_comma":
            return list(ids[:b]) + self.tok.encode(",") + list(ids[b:])
        pa = segs[k - 1][0]
        return self._replace(ids, pa, b, segs[k - 1][2] + t[1:])

    def _case(self, ids, segs, rng):
        cand = self._words(ids, segs, lead_space=False)
        up = [k for k in cand if segs[k][2].strip()[0].isupper()]
        lo = [k for k in cand if segs[k][2][0] == " " and segs[k][2][1:].islower()]
        if not up and not lo:
            return None
        if up and (not lo or rng.random() < 0.6):
            a, b, t = segs[rng.choice(up)]
            return self._replace(ids, a, b, t.lower())
        a, b, t = segs[rng.choice(lo)]
        return self._replace(ids, a, b, " " + t[1].upper() + t[2:])

    # ------------------------------------------------------------------ API
    def apply(self, ids: Sequence[int], family: str, rng: random.Random) -> Optional[List[int]]:
        """One error of ``family``; None if the text offers no site for it (or the result equals the input)."""
        segs = self.segments(ids)
        f = {"typo_char": self._typo_char, "typo_realword": self._typo_realword, "confusion": self._confusion,
             "morph": self._morph, "word_order": self._word_order, "punct": self._punct, "case": self._case,
             "fw_drop": lambda i, s, r: self._fw(i, s, r, False), "fw_dup": lambda i, s, r: self._fw(i, s, r, True)}[family]
        out = f(list(ids), segs, rng)
        return None if out is None or out == list(ids) else out

    def corrupt(self, clean: Sequence[int], rng: random.Random, n_err: int = 1,
                families: Sequence[str] = FAMILIES) -> Tuple[List[int], List[str]]:
        """Up to ``n_err`` errors of distinct families (training-corruption entry point, cf. RealEditSource.corrupt).
        Returns (draft, applied families); fewer errors are applied if the text offers too few sites."""
        d, applied = list(clean), []
        fams = list(families)
        rng.shuffle(fams)
        for fam in fams:
            if len(applied) >= n_err:
                break
            out = self.apply(d, fam, rng)
            if out is not None:
                d, applied = out, applied + [fam]
        return d, applied


@functools.lru_cache(maxsize=2)
def _corruptor(tok):
    return RealisticCorruptor(tok)


def _rec(tok, uid, intent, prompt, draft, clean, **kw):
    return dict(uid=uid, intent=intent, instruction_ids=list(prompt), before_ids=list(draft), after_ids=list(clean),
                before_text=tok.decode(draft), after_text=tok.decode(clean), **kw)


def synth_real_records(n: int = N_DEFAULT["synth_real"], seed: int = DEFAULT_SEED, tok=None, pool=None):
    from .tasks.realedit import FIX_INSTR
    tok = tok or load_gpt2_tokenizer()
    pool = pool if pool is not None else dev_pool()
    corr = _corruptor(tok)
    out = []
    for i in range(n):
        fam = SYNTH_REAL_INTENTS[i % len(SYNTH_REAL_INTENTS)]
        rng = random.Random(f"synth_real:{seed}:{i}")
        for _ in range(200):
            _, b, a = pool[rng.randrange(len(pool))]
            clean = a or b
            if fam == "clean":
                draft, applied = list(clean), []
                break
            if fam == "mixed":
                draft, applied = corr.corrupt(clean, rng, rng.choice([2, 3]))
                if len(applied) >= 2 and draft != list(clean):
                    break
            else:
                draft = corr.apply(clean, fam, rng)
                applied = [fam]
                if draft is not None:
                    break
        else:
            raise RuntimeError(f"no site for {fam} after 200 texts")
        out.append(_rec(tok, f"synth_real/{i}", fam, FIX_INSTR, draft, clean, families=applied))
    return out


def infill_records(n: int = N_DEFAULT["infill"], seed: int = DEFAULT_SEED, tok=None, pool=None, gap_known=False):
    """Word-aligned gap of exactly L = 1 + i % 12 tokens; at least 4 context tokens remain."""
    tok = tok or load_gpt2_tokenizer()
    pool = pool if pool is not None else dev_pool()
    corr = _corruptor(tok)
    out = []
    for i in range(n):
        L = 1 + i % 12
        rng = random.Random(f"infill:{seed}:{i}")
        for _ in range(500):
            _, b, a = pool[rng.randrange(len(pool))]
            clean = list(a or b)
            if len(clean) < L + 4:
                continue
            segs = corr.segments(clean)
            bounds = {s for s, e, t in segs} | {len(clean)}
            words = [s for s, e, t in segs if _WORD.fullmatch(t)]   # the gap must contain at least one word
            starts = [s for s in sorted(bounds) if s + L in bounds and s < len(clean)
                      and any(s <= w < s + L for w in words)]
            if starts:
                s = rng.choice(starts)
                break
        else:
            raise RuntimeError(f"no gap of length {L}")
        lo = 1 + 3 * ((L - 1) // 3)
        out.append(_rec(tok, f"infill/{i}", f"gap{lo:02d}-{lo + 2:02d}", INFILL_INSTR, clean[:s] + clean[s + L:], clean,
                        gap_start=s, gap_len=L, seed_cursor=s if gap_known else None))
    return out


def infill2_records(n: int = N_DEFAULT["infill2"], seed: int = DEFAULT_SEED, tok=None, pool=None):
    """Two word-aligned gaps of L1 = 1 + i % 6 and L2 = 1 + (i // 6) % 6 tokens, >= 3 fixed tokens between them.
    Draft = A + B + C (clean = A + gap1 + B + gap2 + C); ``gap_starts`` are in draft coordinates."""
    tok = tok or load_gpt2_tokenizer()
    pool = pool if pool is not None else dev_pool()
    corr = _corruptor(tok)
    out = []
    for i in range(n):
        L1, L2 = 1 + i % 6, 1 + (i // 6) % 6
        rng = random.Random(f"infill2:{seed}:{i}")
        for _ in range(500):
            _, b, a = pool[rng.randrange(len(pool))]
            clean = list(a or b)
            if len(clean) < L1 + L2 + 6:
                continue
            segs = corr.segments(clean)
            bounds = {s for s, e, t in segs} | {len(clean)}
            words = [s for s, e, t in segs if _WORD.fullmatch(t)]

            def starts(L, lo):
                return [s for s in sorted(bounds) if s >= lo and s + L in bounds and s < len(clean)
                        and any(s <= w < s + L for w in words)]
            pairs = [(s1, s2) for s1 in starts(L1, 0) for s2 in starts(L2, s1 + L1 + 3)]
            if pairs:
                s1, s2 = rng.choice(pairs)
                break
        else:
            raise RuntimeError(f"no gap pair {L1},{L2}")
        A, B, C = clean[:s1], clean[s1 + L1:s2], clean[s2 + L2:]
        tot = L1 + L2
        lo = 2 + 4 * ((tot - 2) // 4)
        out.append(_rec(tok, f"infill2/{i}", f"tot{lo:02d}-{lo + 3:02d}", INFILL_INSTR, A + B + C, clean,
                        gap_starts=[len(A), len(A) + len(B)], gap_lens=[L1, L2]))
    return out


def eval_set_records(data: str, n: int = 0, seed: int = DEFAULT_SEED, tok=None, pool=None):
    n = n or N_DEFAULT[data]
    if data == "synth_real":
        return synth_real_records(n, seed, tok, pool)
    if data in ("infill", "infill_pos"):
        return infill_records(n, seed, tok, pool, gap_known=data == "infill_pos")
    if data in ("infill2", "infill2_pos"):
        return infill2_records(n, seed, tok, pool)
    raise ValueError(data)


def infill_metrics(r: Dict, fin: Sequence[int]) -> Dict:
    """len_err: len(pred) - len(target) (copy baseline: -L); outside_intact: the gapped draft's left and right
    context are preserved verbatim at the ends of the prediction (the model only wrote into the gap)."""
    tgt, s, L = r["after_ids"], r["gap_start"], r["gap_len"]
    suf = len(tgt) - s - L
    fin = list(fin)
    intact = len(fin) >= s + suf and fin[:s] == tgt[:s] and (suf == 0 or fin[len(fin) - suf:] == tgt[s + L:])
    return dict(gap_len=L, len_err=len(fin) - len(tgt), abs_len_err=abs(len(fin) - len(tgt)),
                len_exact=len(fin) == len(tgt), outside_intact=intact)


def multigap_metrics(r: Dict, fin: Sequence[int]) -> Dict:
    """Multi-gap analogue of ``infill_metrics``: len_err vs the target; outside_intact: the fixed segments A, B, C of
    the draft appear in order in the prediction, A as its prefix and C as its suffix (nothing fixed was changed)."""
    tgt, d, (g1, g2) = r["after_ids"], r["before_ids"], r["gap_starts"]
    A, B, C = list(d[:g1]), list(d[g1:g2]), list(d[g2:])
    fin = list(fin)
    intact = False
    if len(fin) >= len(A) + len(B) + len(C) and fin[:len(A)] == A and (not C or fin[len(fin) - len(C):] == C):
        mid = fin[len(A):len(fin) - len(C)]
        intact = any(mid[k:k + len(B)] == B for k in range(len(mid) - len(B) + 1))
    return dict(gap_len=sum(r["gap_lens"]), len_err=len(fin) - len(tgt), abs_len_err=abs(len(fin) - len(tgt)),
                len_exact=len(fin) == len(tgt), outside_intact=intact)
