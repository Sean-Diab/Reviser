"""Real-text episodes over GPT-2 ids (tasks 16/22/23/26): for 100M-scale fine-tuning of a converted historical
Reviser (convert_historical.py) or a matched model.

Pools come from data/real/<file>.jsonl (reviser_v2.realdata.build; eval-overlap filtered train splits). Mixture kinds:

* ``edit``         real revision pair: prompt = instruction ids, draft = before, target = after,
                   ops = repair_ops(before, after, order) (synthetic hunk order; not human keystrokes).
* ``restoration``  blank-canvas text restoration of one side of a pair: prompt = the first ``prefix_frac`` of the
                   tokens, target = the rest; obfuscation policy as in the toy suite (junk tokens, if p_ins > 0,
                   are drawn from the corpus unigram distribution, not uniformly over the vocabulary).
* ``synth_repair`` a clean text (the pair's ``after``) with 0-3 synthetic token errors (replace / delete / insert
                   corpus-random tokens, or swap two adjacent tokens) supplied as a draft; prompt = fixed instruction
                   "Fix the errors in this text"; target = the clean text. Task 23 "realistic errors" variants can be
                   added as further kinds.
* ``model_repair`` (task 23, D5) prompt = the pair's instruction, draft = a trained model's own output for that pair
                   (``files``: pools written by scripts/remote/gen_model_drafts.py), target = the reference after,
                   ops = repair_ops; exact outputs give draft = target (immediate Stop).

Matched autoregressive baseline kinds (tasks 29/30; same pools, same warm start, only the trajectory differs):
* ``edit_ar`` / ``synth_ar``  prompt = instruction + ":" + before/draft (read-only prompt tokens), blank canvas,
                              target typed left to right (``ltr_ops``) then STOP: a full rewrite, no Delete/Move.
* ``ltr``                     text continuation: prompt = prefix, target typed left to right.

Task 30 AR adaptations (same pools; patchfmt.py formats) and the matching Reviser infilling kind (task 27):
* ``edit_patch`` / ``synth_patch``  prompt = instruction + ":" + before/draft; target = the token-level
                              search/replace edit script ``make_patch(before, after)`` typed left to right.
* ``fim``                     prompt = prefix FIM_SUF suffix FIM_MID; target = the removed middle (1-12 tokens,
                              length unknown to the model); gap position known (the ``infill_pos`` oracle condition).
* ``infill``                  Reviser: prompt = INFILL_INSTR, draft = text with the same kind of gap removed
                              (no marker), target = the full text, ops = repair_ops (gap position unknown).
* ``infill_ar``               AR rewriter counterpart: prompt = INFILL_INSTR + ":" + gapped text, target typed LTR.
"""
from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..actions import ActionSpace
from ..patchfmt import fim_prompt, make_patch
from ..trajectories import Episode, ObfuscationPolicy, build_episode, ltr_ops, repair_ops, restoration_ops

ROOT = Path(__file__).resolve().parents[3]
FIX_INSTR = [22743, 262, 8563, 287, 428, 2420]   # GPT-2 ids of "Fix the errors in this text"
COLON = 25                                        # GPT-2 ":" (AR layout: instruction ":" source)
INFILL_INSTR = [33762, 287, 262, 4814, 2456]      # GPT-2 ids of "Fill in the missing words" (= evalsets.py)


def ar_prompt(instr: Sequence[int], source: Sequence[int]) -> List[int]:
    return list(instr) + [COLON] + list(source)
DEFAULT_TRAIN = ["iterater_human_sent_train", "iterater_full_sent_train", "iterater_v2_sent_train", "coedit_train"]
DEFAULT_DEV = ["iterater_human_sent_dev", "coedit_validation"]


def load_pool(files: Sequence[str], max_pair_tokens: int, data_dir: Path = None) -> List[Tuple[List[int], List[int], List[int]]]:
    """[(instruction_ids, before_ids, after_ids)] with len(before)+len(after) <= max_pair_tokens; cached as pickle."""
    data_dir = Path(data_dir or ROOT / "data" / "real")
    out = []
    for name in files:
        cache = data_dir / "cache" / f"{name}_le{max_pair_tokens}.pkl"
        if cache.exists():
            out += pickle.load(open(cache, "rb"))
            continue
        recs = []
        with open(data_dir / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                b, a = r["before_ids"], r["after_ids"]
                if len(b) + len(a) <= max_pair_tokens and (b or a):
                    recs.append((r["instruction_ids"], b, a))
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".tmp{os.getpid()}")
        pickle.dump(recs, open(tmp, "wb"))
        os.replace(tmp, cache)
        out += recs
    return out


class RealEditSource:
    def __init__(self, space: ActionSpace, task: Dict, mix: List[Dict], data_seed: int, split: str = "train"):
        self.space, self.task, self.mix, self.seed, self.split = space, task, mix, data_seed, split
        files = task.get("train_files", DEFAULT_TRAIN) if split == "train" else task.get("dev_files", DEFAULT_DEV)
        self.pool = load_pool(files, task.get("max_pair_tokens", 256))
        assert self.pool, f"empty pool for {files}"
        # corpus-unigram junk: a fixed token reservoir from the pool (deterministic)
        rr = random.Random(f"junk:{data_seed}")
        res = []
        for _ in range(2000):
            _, b, a = self.pool[rr.randrange(len(self.pool))]
            res += a or b
        self.reservoir = res
        # model_repair (task 23, D5): drafts produced by a trained model (scripts/remote/gen_model_drafts.py), same
        # pool format; the dev split reuses the training drafts (dev CE monitoring only)
        self.model_pools = {i: load_pool(m["files"], task.get("max_pair_tokens", 256), m.get("data_dir"))
                            for i, m in enumerate(mix) if m["kind"] == "model_repair"}
        tot = sum(m["weight"] for m in mix)
        acc, self.cum = 0.0, []
        for m in mix:
            acc += m["weight"] / tot
            self.cum.append(acc)

    def junk(self, rng: random.Random) -> int:
        return self.reservoir[rng.randrange(len(self.reservoir))]

    def corrupt(self, clean: List[int], rng: random.Random, n_err: int) -> List[int]:
        d = list(clean)
        for _ in range(n_err):
            k = rng.choice(["replace", "delete", "insert", "swap"])
            if k == "insert" or not d:
                d.insert(rng.randrange(len(d) + 1), self.junk(rng))
            elif k == "delete":
                del d[rng.randrange(len(d))]
            elif k == "replace":
                d[rng.randrange(len(d))] = self.junk(rng)
            elif len(d) >= 2:
                i = rng.randrange(len(d) - 1)
                d[i], d[i + 1] = d[i + 1], d[i]
        return d

    def episode(self, idx: int) -> Tuple[Episode, str]:
        rng = random.Random(f"{self.seed}:{self.split}:{idx}")
        r = rng.random()
        m = next(mm for mm, c in zip(self.mix, self.cum) if r <= c + 1e-12)
        instr, before, after = self.pool[rng.randrange(len(self.pool))]
        kind = m["kind"]
        if kind == "edit":
            # identity examples: the draft is already the target, so the answer is an immediate Stop (no rng draw unless
            # configured, so existing configs keep their exact data streams)
            if "p_identity" in m and rng.random() < m["p_identity"]:
                before = after
            ops = repair_ops(before, after, m.get("order", "rtl"))
            return build_episode(self.space, instr, before, after, ops), kind
        if kind == "edit_ar":
            return build_episode(self.space, ar_prompt(instr, before), [], after, ltr_ops(after)), kind
        if kind in ("restoration", "ltr"):
            text = after if (not before or rng.random() < m.get("p_after", 0.5)) else before
            if not text:
                text = after or before
            k = int(round(m.get("prefix_frac", 0.35) * len(text)))
            prompt, tgt = text[:k], text[k:]
            if kind == "ltr":
                return build_episode(self.space, prompt, [], tgt, ltr_ops(tgt)), kind
            ops = restoration_ops(tgt, rng, ObfuscationPolicy(**m.get("policy", {})), lambda g, canvas, cur: self.junk(g))
            return build_episode(self.space, prompt, [], tgt, ops), kind
        if kind in ("synth_repair", "synth_ar", "synth_patch"):
            clean = after or before
            probs = m.get("n_errors_probs", [0.2, 0.5, 0.2, 0.1])
            ne = rng.choices(range(len(probs)), weights=probs)[0]
            draft = self.corrupt(clean, rng, ne)
            if kind == "synth_ar":
                return build_episode(self.space, ar_prompt(FIX_INSTR, draft), [], clean, ltr_ops(clean)), kind
            if kind == "synth_patch":
                p = make_patch(draft, clean)
                return build_episode(self.space, ar_prompt(FIX_INSTR, draft), [], p, ltr_ops(p)), kind
            ops = repair_ops(draft, clean, m.get("order", "rtl"))
            return build_episode(self.space, FIX_INSTR, draft, clean, ops), kind
        if kind == "model_repair":
            mp = self.model_pools[self.mix.index(m)]
            instr, draft, tgt = mp[rng.randrange(len(mp))]
            return build_episode(self.space, instr, draft, tgt, repair_ops(draft, tgt, m.get("order", "rtl"))), kind
        if kind == "edit_patch":
            p = make_patch(before, after)
            return build_episode(self.space, ar_prompt(instr, before), [], p, ltr_ops(p)), kind
        if kind in ("infill", "infill_ar", "fim"):
            text = after if (not before or rng.random() < m.get("p_after", 0.5)) else before
            text = text or after or before
            lo, hi = m.get("gap_range", [1, 12])
            L = min(rng.randint(lo, hi), max(0, len(text) - 4))  # keep >= 4 context tokens
            s = rng.randint(0, len(text) - L)
            gapped, mid = text[:s] + text[s + L:], text[s:s + L]
            if kind == "fim":
                return build_episode(self.space, fim_prompt(text[:s], text[s + L:]), [], mid, ltr_ops(mid)), kind
            if kind == "infill_ar":
                return build_episode(self.space, ar_prompt(INFILL_INSTR, gapped), [], text, ltr_ops(text)), kind
            return build_episode(self.space, INFILL_INSTR, gapped, text, repair_ops(gapped, text, m.get("order", "rtl"))), kind
        raise ValueError(kind)
