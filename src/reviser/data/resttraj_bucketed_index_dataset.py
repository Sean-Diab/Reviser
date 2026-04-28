from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


_BUCKET_RE = re.compile(r"^len_(\d{4})_(\d{4})$")
_SHARD_RE = re.compile(r"rest_(?:tokens|offsets)_shard(\d+)\.npy$")


@dataclass(frozen=True)
class BucketInfo:
    name: str
    len_start: int
    len_end: int
    shard_idx_path: Path
    sample_idx_path: Path


class _ShardCache:
    def __init__(self, shards_dir: Path, cache_size: int) -> None:
        self.shards_dir = shards_dir
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = OrderedDict()

    def get(self, shard_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        shard_idx = int(shard_idx)
        if shard_idx in self.cache:
            value = self.cache.pop(shard_idx)
            self.cache[shard_idx] = value
            return value

        tokens = np.load(self.shards_dir / f"rest_tokens_shard{shard_idx:06d}.npy", mmap_mode="r")
        offsets = np.load(self.shards_dir / f"rest_offsets_shard{shard_idx:06d}.npy", mmap_mode="r")
        prefix_lens = _load_optional_prefix_lengths(self.shards_dir, shard_idx)
        self.cache[shard_idx] = (tokens, offsets, prefix_lens)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[shard_idx]


def _load_optional_prefix_lengths(shards_dir: Path, shard_idx: int) -> Optional[np.ndarray]:
    candidates = (
        shards_dir / f"prefix_len_shard{shard_idx:06d}.npy",
        shards_dir / f"rest_prefix_len_shard{shard_idx:06d}.npy",
    )
    for path in candidates:
        if path.is_file():
            return np.load(path, mmap_mode="r")
    return None


def _discover_buckets(resttraj_dir: Path, *, min_len: int, max_len: int) -> List[BucketInfo]:
    buckets_root = resttraj_dir / "buckets"
    if not buckets_root.is_dir():
        raise FileNotFoundError(f"Missing buckets directory: {buckets_root}")

    out: List[BucketInfo] = []
    for path in sorted(buckets_root.iterdir()):
        if not path.is_dir():
            continue
        match = _BUCKET_RE.match(path.name)
        if not match:
            continue
        len_start, len_end = int(match.group(1)), int(match.group(2))
        if len_end < int(min_len) or len_start > int(max_len):
            continue
        shard_idx_path = path / "shard_idx.npy"
        sample_idx_path = path / "sample_idx.npy"
        if shard_idx_path.is_file() and sample_idx_path.is_file():
            out.append(
                BucketInfo(
                    name=path.name,
                    len_start=len_start,
                    len_end=len_end,
                    shard_idx_path=shard_idx_path,
                    sample_idx_path=sample_idx_path,
                )
            )
    if not out:
        raise FileNotFoundError(f"No usable buckets found in {buckets_root}")
    return out


class BucketedRestTrajIndexDataset(IterableDataset):
    """Iterates restoration-trajectory samples using precomputed bucket index files."""

    def __init__(
        self,
        *,
        resttraj_dir: str,
        batch_size: int,
        seed: int = 123,
        min_len: int = 1,
        max_len: int = 10_000,
        log_every_epoch: bool = False,
        random_bucket_choice: bool = True,
        shard_cache_size: int = 8,
    ) -> None:
        super().__init__()
        self.resttraj_dir = Path(resttraj_dir).expanduser().resolve()
        self.shards_dir = self.resttraj_dir / "shards"
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.log_every_epoch = bool(log_every_epoch)
        self.random_bucket_choice = bool(random_bucket_choice)
        self.shard_cache_size = int(shard_cache_size)
        self.buckets = _discover_buckets(self.resttraj_dir, min_len=self.min_len, max_len=self.max_len)

    def __iter__(self) -> Iterator[List[Dict[str, np.ndarray]]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = np.random.default_rng(self.seed + 9973 * worker_id)
        cache = _ShardCache(self.shards_dir, self.shard_cache_size)
        epoch = 0

        while True:
            epoch += 1
            bucket_order = list(range(len(self.buckets)))
            if self.random_bucket_choice:
                rng.shuffle(bucket_order)

            if self.log_every_epoch and worker_id == 0:
                print(
                    f"[BucketedRestTrajIndexDataset] epoch={epoch} buckets={len(bucket_order)} workers={num_workers}",
                    flush=True,
                )

            for bucket_idx in bucket_order:
                bucket = self.buckets[bucket_idx]
                shard_ids = np.load(bucket.shard_idx_path, mmap_mode="r")
                sample_ids = np.load(bucket.sample_idx_path, mmap_mode="r")
                count = int(sample_ids.shape[0])
                if count <= 0:
                    continue

                positions = np.arange(count, dtype=np.int64)
                rng.shuffle(positions)
                if num_workers > 1:
                    positions = positions[worker_id::num_workers]

                batch: List[Dict[str, np.ndarray]] = []
                for pos in positions.tolist():
                    shard_idx = int(shard_ids[int(pos)])
                    sample_idx = int(sample_ids[int(pos)])
                    tokens, offsets, prefix_lens = cache.get(shard_idx)
                    if sample_idx < 0 or sample_idx >= int(offsets.shape[0]):
                        continue
                    start = int(offsets[sample_idx, 0])
                    length = int(offsets[sample_idx, 1])
                    if length < self.min_len or length > self.max_len:
                        continue
                    seq = np.asarray(tokens[start : start + length], dtype=np.int64)
                    sample: Dict[str, np.ndarray] = {
                        "input_ids": seq,
                        "action_ids": seq,
                        "seq_len": np.int64(length),
                        "shard_idx": np.int64(shard_idx),
                        "sample_idx": np.int64(sample_idx),
                        "bucket_name": np.array(bucket.name),
                    }
                    if prefix_lens is not None and sample_idx < int(prefix_lens.shape[0]):
                        sample["prefix_len"] = np.int64(prefix_lens[sample_idx])
                    batch.append(sample)
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []

                if batch:
                    yield batch
