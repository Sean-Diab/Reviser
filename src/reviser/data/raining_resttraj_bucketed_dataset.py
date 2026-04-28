from __future__ import annotations

import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


_SHARD_FILE_RE = re.compile(r"rest_(?:tokens|offsets)_shard(\d+)\.npy$")
_DONE_IDX_RE = re.compile(r"(\d{1,})")


class _ShardCache:
    def __init__(self, shards_dir: Path, cache_size: int = 2) -> None:
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


def _paired_shards(shards_dir: Path) -> List[int]:
    token_indices = set()
    offset_indices = set()
    for path in shards_dir.glob("rest_tokens_shard*.npy"):
        match = _SHARD_FILE_RE.match(path.name)
        if match:
            token_indices.add(int(match.group(1)))
    for path in shards_dir.glob("rest_offsets_shard*.npy"):
        match = _SHARD_FILE_RE.match(path.name)
        if match:
            offset_indices.add(int(match.group(1)))
    return sorted(token_indices & offset_indices)


def _done_shards(shards_dir: Path) -> Optional[set[int]]:
    done_paths = list(shards_dir.glob("*.done"))
    if not done_paths:
        return None

    out: set[int] = set()
    for path in done_paths:
        matches = _DONE_IDX_RE.findall(path.stem)
        if not matches:
            continue
        out.add(int(matches[-1]))
    return out or None


def _bucket_for_len(length: int, *, min_len: int, max_len: int, bucket_size: int) -> int:
    if length < int(min_len) or length > int(max_len):
        return -1
    return int((int(length) - int(min_len)) // max(1, int(bucket_size)))


class RainingBucketedRestTrajDataset(IterableDataset):
    """Streams completed restoration-trajectory shards while new shards may still be appearing."""

    def __init__(
        self,
        *,
        resttraj_dir: str,
        batch_size: int,
        seed: int = 123,
        min_len: int = 1,
        max_len: int = 10_000,
        bucket_size: int = 5,
        refresh_secs: float = 30.0,
        log_every_epoch: bool = True,
    ) -> None:
        super().__init__()
        self.resttraj_dir = Path(resttraj_dir).expanduser().resolve()
        self.shards_dir = self.resttraj_dir / "shards"
        if not self.shards_dir.is_dir():
            raise FileNotFoundError(f"Missing shards directory: {self.shards_dir}")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.bucket_size = int(bucket_size)
        self.refresh_secs = float(refresh_secs)
        self.log_every_epoch = bool(log_every_epoch)

    def _available_shards(self) -> List[int]:
        paired = _paired_shards(self.shards_dir)
        done = _done_shards(self.shards_dir)
        if done is None:
            return paired
        return [idx for idx in paired if idx in done]

    def __iter__(self) -> Iterator[List[Dict[str, np.ndarray]]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = np.random.default_rng(self.seed + 1543 * worker_id)
        cache = _ShardCache(self.shards_dir)
        epoch = 0
        last_refresh = 0.0
        shard_ids: List[int] = []

        while True:
            now = time.time()
            if (not shard_ids) or (now - last_refresh >= self.refresh_secs):
                shard_ids = self._available_shards()
                last_refresh = now
                if num_workers > 1:
                    shard_ids = shard_ids[worker_id::num_workers]

            if not shard_ids:
                time.sleep(min(max(self.refresh_secs, 1.0), 30.0))
                continue

            epoch += 1
            shard_order = list(shard_ids)
            rng.shuffle(shard_order)
            if self.log_every_epoch and worker_id == 0:
                print(
                    f"[RainingBucketedRestTrajDataset] epoch={epoch} available_shards={len(shard_order)} workers={num_workers}",
                    flush=True,
                )

            for shard_idx in shard_order:
                tokens, offsets, prefix_lens = cache.get(shard_idx)
                lengths = np.asarray(offsets[:, 1], dtype=np.int64)
                valid = (lengths >= self.min_len) & (lengths <= self.max_len)
                if not np.any(valid):
                    continue

                bucket_path = self.shards_dir / f"rest_bucket_id_shard{int(shard_idx):06d}.npy"
                if bucket_path.is_file():
                    bucket_ids = np.load(bucket_path, mmap_mode="r")
                else:
                    bucket_ids = np.asarray(
                        [_bucket_for_len(int(length), min_len=self.min_len, max_len=self.max_len, bucket_size=self.bucket_size) for length in lengths],
                        dtype=np.int64,
                    )

                unique_bucket_ids = [int(x) for x in np.unique(bucket_ids[valid]).tolist() if int(x) >= 0]
                rng.shuffle(unique_bucket_ids)

                for bucket_id in unique_bucket_ids:
                    positions = np.flatnonzero(valid & (bucket_ids == int(bucket_id)))
                    if positions.size == 0:
                        continue
                    rng.shuffle(positions)
                    batch: List[Dict[str, np.ndarray]] = []

                    for sample_idx in positions.tolist():
                        start = int(offsets[int(sample_idx), 0])
                        length = int(offsets[int(sample_idx), 1])
                        seq = np.asarray(tokens[start : start + length], dtype=np.int64)
                        sample: Dict[str, np.ndarray] = {
                            "input_ids": seq,
                            "action_ids": seq,
                            "seq_len": np.int64(length),
                            "shard_idx": np.int64(shard_idx),
                            "sample_idx": np.int64(sample_idx),
                            "bucket_id": np.int64(bucket_id),
                        }
                        if prefix_lens is not None and sample_idx < int(prefix_lens.shape[0]):
                            sample["prefix_len"] = np.int64(prefix_lens[sample_idx])
                        batch.append(sample)
                        if len(batch) >= self.batch_size:
                            yield batch
                            batch = []

                    if batch:
                        yield batch
