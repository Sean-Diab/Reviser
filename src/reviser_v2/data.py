"""Config-driven episode sources for training.

Example ``idx`` is generated from ``random.Random(f"{data_seed}:{idx}")`` so data order is a pure
function of (data_seed, idx): paired runs (e.g. D1/D2) see identical trajectory instances in the same
order, and resumption only needs the next index.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .actions import ActionSpace
from .trajectories import (
    Episode, ObfuscationPolicy, RunJumpPolicy, build_episode, episode_to_sequence, ltr_ops, repair_ops,
    restoration_ops, runjump_ops, uniform_junk,
)
from .tasks import progression as prog


def make_space(cfg: Dict, n_content: int) -> ActionSpace:
    return ActionSpace(n_content=n_content, **cfg)


class ProgressionSource:
    """Episodes for the toy progression language (see tasks/progression.py)."""

    def __init__(self, space: ActionSpace, spec: prog.ProgSpec, mix: List[Dict], data_seed: int, split: str = "train"):
        self.space, self.spec, self.mix, self.seed, self.split = space, spec, mix, data_seed, split
        tot = sum(m["weight"] for m in mix)
        self.cum = []
        acc = 0.0
        for m in mix:
            acc += m["weight"] / tot
            self.cum.append(acc)

    def episode(self, idx: int) -> Tuple[Episode, str]:
        rng = random.Random(f"{self.seed}:{self.split}:{idx}")
        r = rng.random()
        m = next(mm for mm, c in zip(self.mix, self.cum) if r <= c + 1e-12)
        a, d, n = prog.sample_target(self.spec, rng, self.split)
        tgt = prog.progression(self.spec, a, d, n)
        prompt = prog.make_prompt(n)
        kind = m["kind"]
        if kind == "ltr":
            return build_episode(self.space, prompt, [], tgt, ltr_ops(tgt)), kind
        if kind == "restoration":
            pol = ObfuscationPolicy(**m.get("policy", {}))
            ops = restoration_ops(tgt, rng, pol, uniform_junk(self.spec.n_content))
            return build_episode(self.space, prompt, [], tgt, ops), kind
        if kind == "runjump":
            ops = runjump_ops(tgt, rng, RunJumpPolicy(**m.get("policy", {})))
            return build_episode(self.space, prompt, [], tgt, ops), kind
        if kind == "repair":
            probs = m.get("n_errors_probs", [0.2, 0.5, 0.2, 0.1])  # P(0), P(1), P(2), P(3) errors
            ne = rng.choices(range(len(probs)), weights=probs)[0]
            kinds = prog.sample_kinds(rng, ne, allow_marked=m.get("allow_marked", True))
            c = prog.corrupt(self.spec, tgt, rng, kinds) if ne else prog.Corruption("clean", list(tgt), [], [], False, len(tgt))
            cur_mode = m.get("cursor", "mixed")
            at_err = cur_mode == "at_error" or (cur_mode == "mixed" and rng.random() < 0.5)
            sc = c.first_error_end if (at_err and ne) else None
            ops = repair_ops(c.draft, tgt, m.get("order", "rtl"), start_cursor=sc)
            return build_episode(self.space, prompt, c.draft, tgt, ops, seed_cursor=sc), kind
        raise ValueError(kind)


def collate(space: ActionSpace, episodes: List[Episode], mask_doomed: bool, max_len: int):
    return collate_seqs([episode_to_sequence(space, ep, mask_doomed) for ep in episodes], max_len)


def collate_seqs(seqs, max_len: int):
    """(ids, labels, eligible) lists -> padded tensors; returns ids, labels, elig, n_truncated."""
    L = min(max(len(s[0]) for s in seqs), max_len)
    B = len(seqs)
    ids = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    elig = torch.full((B, L), -100, dtype=torch.long)
    truncated = 0
    for i, (x, y, e) in enumerate(seqs):
        if len(x) > L:
            truncated += 1
        n = min(len(x), L)
        ids[i, :n] = torch.tensor(x[:n])
        labels[i, :n] = torch.tensor(y[:n])
        elig[i, :n] = torch.tensor(e[:n])
    return ids, labels, elig, truncated
