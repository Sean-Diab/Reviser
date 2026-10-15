"""Real-edit episodes in a pretrained backbone's token ids (task 36; model: hf_backbone.HFActionLM).

Task name ``realedit_hf``. Same pools (the GPT-2 real-edit pools re-tokenized by realdata/retokenize.py: same pairs,
same order), same episode sampling and mixture kinds as tasks/realedit.py, except that the two fixed token constants
(the "Fix the errors in this text" instruction and the AR ":" separator) are the backbone's, read from the cache meta.
RealEditSource is reused as is for the kinds without constants (edit, restoration, ltr: the rng draws are identical);
only edit_ar / synth_repair / synth_ar are re-implemented here with the backbone constants. Kinds that embed other
GPT-2 ids (patch / fim / infill formats) are rejected.

Task config keys: tokenizer (HF id or dir), tokenizer_revision, tok_tag (cache tag, e.g. qwen3; also
data/tokenizers/<tok_tag> is preferred when present), n_content (= tokenizer.vocab_size), max_pair_tokens (GPT-2
length filter, as realedit), train_files / dev_files, data_dir (default data/real).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..actions import ActionSpace
from ..trajectories import Episode, build_episode, ltr_ops, repair_ops
from .realedit import DEFAULT_DEV, DEFAULT_TRAIN, RealEditSource

KINDS = ("edit", "restoration", "ltr", "edit_ar", "synth_repair", "synth_ar")


def build_hf(cfg: Dict):
    """train.build() for task realedit_hf -> (task spec, ActionSpace, HFConfig)."""
    from ..hf_backbone import HFConfig
    t = cfg["task"]
    space = ActionSpace(n_content=t["n_content"], **cfg.get("space", {}))
    mcfg = HFConfig(n_content=space.n_content, vocab_size=space.vocab_size, tokenizer=t.get("tokenizer"),
                    tokenizer_revision=t.get("tokenizer_revision"), tok_tag=t.get("tok_tag"), **cfg["model"])
    return t, space, mcfg


class RealEditHFSource(RealEditSource):
    def __init__(self, space: ActionSpace, task: Dict, mix: List[Dict], data_seed: int, split: str = "train"):
        from ..realdata.retokenize import load_pool_hf
        bad = [m["kind"] for m in mix if m["kind"] not in KINDS]
        assert not bad, f"realedit_hf does not support kinds {bad} (they embed GPT-2 token ids)"
        self.space, self.task, self.mix, self.seed, self.split = space, task, mix, data_seed, split
        files = task.get("train_files", DEFAULT_TRAIN) if split == "train" else task.get("dev_files", DEFAULT_DEV)
        self.pool, self.meta = load_pool_hf(files, task["tokenizer"], task["tok_tag"], task.get("max_pair_tokens", 256),
                                            task.get("tokenizer_revision"), task.get("data_dir"))
        assert self.pool, f"empty pool for {files}"
        assert self.meta["n_content"] == space.n_content, f"pool n_content {self.meta['n_content']} != {space.n_content}"
        self.fix_instr, self.colon = list(self.meta["fix_instr"]), self.meta["colon"]
        # corpus-unigram junk reservoir and mixture table: as RealEditSource.__init__
        rr = random.Random(f"junk:{data_seed}")
        res = []
        for _ in range(2000):
            _, b, a = self.pool[rr.randrange(len(self.pool))]
            res += a or b
        self.reservoir = res
        tot = sum(m["weight"] for m in mix)
        acc, self.cum = 0.0, []
        for m in mix:
            acc += m["weight"] / tot
            self.cum.append(acc)

    def episode(self, idx: int) -> Tuple[Episode, str]:
        rng = random.Random(f"{self.seed}:{self.split}:{idx}")  # same draws as RealEditSource.episode
        r = rng.random()
        m = next(mm for mm, c in zip(self.mix, self.cum) if r <= c + 1e-12)
        instr, before, after = self.pool[rng.randrange(len(self.pool))]
        kind = m["kind"]
        if kind == "edit_ar":
            return build_episode(self.space, list(instr) + [self.colon] + list(before), [], after, ltr_ops(after)), kind
        if kind in ("synth_repair", "synth_ar"):
            clean = after or before
            probs = m.get("n_errors_probs", [0.2, 0.5, 0.2, 0.1])
            ne = rng.choices(range(len(probs)), weights=probs)[0]
            draft = self.corrupt(clean, rng, ne)
            if kind == "synth_ar":
                return build_episode(self.space, self.fix_instr + [self.colon] + draft, [], clean, ltr_ops(clean)), kind
            return build_episode(self.space, self.fix_instr, draft, clean, repair_ops(draft, clean, m.get("order", "rtl"))), kind
        return super().episode(idx)  # edit / restoration / ltr: no token constants
