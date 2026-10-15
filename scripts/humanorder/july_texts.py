"""Loader for the shared July texts file `data/humanorder/july_texts_v1/` (built by build_july_texts.py).

Shard K / row i is the same clean text as row i of shard K of the July control's text-matched layout
(`runs/july_control/data/july_textmatched_inverse_v1`): 134 shards, 2,640,539 texts, prefix = 35 GPT-2 ids,
continuation = the rest (65-314 ids). Files per shard: prefix_shardK.npy uint16 (N,35), cont_tokens_shardK.npy
uint16 flat, cont_offsets_shardK.npy int64 (N,2) [start,len], src_index_shardK.npy int32 (N,4)
(src_shard, src_row, rj_shard, rj_row).

    from july_texts import JulyTexts
    jt = JulyTexts()
    jt.n_shards, jt.shard_sizes[k]
    prefix, cont = jt.text(k, i)              # python lists of ints
    for i, prefix, cont in jt.iter_shard(k): ...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "humanorder" / "july_texts_v1"


class JulyTexts:
    def __init__(self, root: str | Path = DEFAULT_DIR, mmap: bool = True):
        self.root = Path(root)
        self.sd = self.root / "shards"
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.n_shards = int(self.manifest["n_shards"])
        self.shard_sizes = [int(r["n"]) for r in self.manifest["shards"]]
        self.mmap = "r" if mmap else None
        self._cache: dict[int, tuple] = {}

    def shard(self, k: int):
        """(prefix (N,35) uint16, cont_tokens uint16 flat, cont_offsets int64 (N,2), src_index int32 (N,4))."""
        if k not in self._cache:
            if len(self._cache) > 2:
                self._cache.clear()
            self._cache[k] = tuple(np.load(self.sd / f"{name}_shard{k:06d}.npy", mmap_mode=self.mmap)
                                   for name in ("prefix", "cont_tokens", "cont_offsets", "src_index"))
        return self._cache[k]

    def text(self, k: int, i: int) -> tuple[list[int], list[int]]:
        p, ct, co, _ = self.shard(k)
        s, n = int(co[i, 0]), int(co[i, 1])
        return [int(x) for x in p[i]], [int(x) for x in ct[s:s + n]]

    def iter_shard(self, k: int):
        p, ct, co, _ = self.shard(k)
        ct = np.asarray(ct)
        for i in range(p.shape[0]):
            s, n = int(co[i, 0]), int(co[i, 1])
            yield i, p[i].astype(np.int64).tolist(), ct[s:s + n].astype(np.int64).tolist()
