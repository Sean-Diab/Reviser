"""
Bucketed, on-disk preprocessed dataset that yields *pre-batched* samples.

Designed for the C4 bucketed shard format produced by:
  /workspace/other_scripts/preprocess_c4_bucketed.py

Sampling behavior (per user spec):
- Maintain buckets by restoration-trajectory length.
- To yield one training batch:
  - choose a random bucket from the set of non-empty buckets
  - draw up to `batch_size` samples from that bucket, without replacement within an epoch
  - if fewer than `batch_size` remain in the bucket, yield the remainder
- When a bucket empties, remove it from the sampling set until all buckets empty.
- When all buckets empty: log epoch boundary, then restart (reshuffle) and continue.

Continuous update:
- The dataset periodically scans for new shard files and adds them to the epoch’s pool.
  This allows a producer process to keep writing shards while training consumes them.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


Bucket = Tuple[int, int]  # (start, end) e.g. (10,14)


def _parse_bucket_dir_name(name: str) -> Optional[Bucket]:
    # Expected: len_010_014
    if not name.startswith("len_"):
        return None
    parts = name.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except Exception:
        return None


def _load_shard_samples(shard_path: Path, *, rng: random.Random) -> List[dict]:
    """
    Load one shard into a list of training sample dicts.
    Converts `canvas_states` -> `canvas_sequences` to match CursorCollator.
    """
    data = np.load(shard_path, allow_pickle=True)
    input_ids = data["input_ids"]
    labels = data["labels"]
    canvas_states = data["canvas_states"]
    prompt_lengths = data["prompt_lengths"]

    n = int(len(input_ids))
    idxs = list(range(n))
    rng.shuffle(idxs)

    out: List[dict] = []
    for i in idxs:
        inp = input_ids[i]
        lab = labels[i]
        canv = canvas_states[i]
        if hasattr(inp, "tolist"):
            inp = inp.tolist()
        if hasattr(lab, "tolist"):
            lab = lab.tolist()
        if hasattr(canv, "tolist"):
            canv = canv.tolist()
        out.append(
            {
                "input_ids": inp,
                "labels": lab,
                "canvas_sequences": canv,  # collator expects this key
                "prompt_length": int(prompt_lengths[i]),
            }
        )
    return out


def _count_samples_in_shard(shard_path: Path) -> int:
    # Cheap-ish count: read only the traj_lens array length (still opens file).
    data = np.load(shard_path, allow_pickle=True)
    if "traj_lens" in data:
        return int(len(data["traj_lens"]))
    if "input_ids" in data:
        return int(len(data["input_ids"]))
    return 0


@dataclass
class _EpochState:
    epoch_idx: int = 0
    batches_yielded: int = 0


class BucketedPreprocessedBatchDataset(IterableDataset):
    """
    IterableDataset that yields *lists of samples* (a pre-batched list).

    Use with DataLoader(batch_size=None, collate_fn=CursorCollator(...)).
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        poll_secs: float = 10.0,
        seed: int = 42,
        log_every_epoch: bool = True,
        min_bucket_len: Optional[int] = None,
        max_bucket_len: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = int(batch_size)
        self.poll_secs = float(poll_secs)
        self.seed = int(seed)
        self.log_every_epoch = bool(log_every_epoch)
        self.min_bucket_len = int(min_bucket_len) if min_bucket_len is not None else None
        self.max_bucket_len = int(max_bucket_len) if max_bucket_len is not None else None

        self.buckets_root = self.data_dir / "buckets"
        if not self.buckets_root.exists():
            raise FileNotFoundError(f"Expected buckets/ under {self.data_dir} (got {self.buckets_root})")

        # Master file inventory: all discovered shards per bucket.
        self._all_shards: Dict[Bucket, List[Path]] = {}
        self._known_files: Set[Path] = set()
        self.total_samples: int = 0  # running estimate as we discover shards

        # Runtime state (rebuilt each epoch in __iter__)
        self._rng = random.Random(self.seed)
        self._last_poll_time = 0.0

        self._initial_scan()

    def _initial_scan(self) -> None:
        self._scan_for_new_shards(force=True)
        print(
            f"[bucketed] Initialized from {self.data_dir} | buckets={len(self._all_shards)} "
            f"| shards={sum(len(v) for v in self._all_shards.values())} "
            f"| est_samples={self.total_samples}",
            flush=True,
        )

    def _scan_for_new_shards(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_poll_time) < self.poll_secs:
            return
        self._last_poll_time = now

        # Discover bucket dirs
        for bucket_dir in self.buckets_root.iterdir():
            if not bucket_dir.is_dir():
                continue
            bucket = _parse_bucket_dir_name(bucket_dir.name)
            if bucket is None:
                continue
            b0, b1 = bucket
            # Optional range filter. Keep buckets that overlap the allowed range.
            # This is used to avoid sampling sequences longer than the model config supports.
            if self.min_bucket_len is not None and int(b1) < int(self.min_bucket_len):
                continue
            if self.max_bucket_len is not None and int(b0) > int(self.max_bucket_len):
                continue
            self._all_shards.setdefault(bucket, [])

            for shard_path in sorted(bucket_dir.glob("*.npz")):
                # Ignore temporary/partial files.
                # Our producer writes `shard_XXXXXX.tmp.npz` then atomically renames to `.npz`.
                if shard_path.suffix != ".npz":
                    continue
                if shard_path.name.endswith(".tmp.npz"):
                    continue
                if shard_path in self._known_files:
                    continue
                self._known_files.add(shard_path)
                self._all_shards[bucket].append(shard_path)
                try:
                    self.total_samples += _count_samples_in_shard(shard_path)
                except Exception:
                    # If counting fails, leave estimate unchanged.
                    pass

    def __iter__(self) -> Iterable[List[dict]]:
        wi = get_worker_info()
        worker_id = int(wi.id) if wi is not None else 0
        num_workers = int(wi.num_workers) if wi is not None else 1

        # Per-worker RNG so workers don't all walk the same path.
        rng = random.Random(self.seed + 1009 * worker_id + int(time.time()) % 1_000_000)
        epoch = _EpochState(epoch_idx=0, batches_yielded=0)

        # Per-epoch: remaining shard queues and sample buffers.
        shard_q: Dict[Bucket, Deque[Path]] = {}
        sample_buf: Dict[Bucket, Deque[dict]] = {}

        def reset_epoch() -> None:
            nonlocal shard_q, sample_buf
            self._scan_for_new_shards(force=True)

            shard_q = {}
            sample_buf = {}
            for b, files in self._all_shards.items():
                if not files:
                    continue
                # Partition shards across workers to avoid duplicated I/O and better prefetch.
                # Stable order first, then assign by index mod num_workers.
                stable = sorted(files, key=lambda p: str(p))
                assigned = [p for i, p in enumerate(stable) if (i % num_workers) == worker_id]
                if not assigned:
                    continue
                files_copy = list(assigned)
                rng.shuffle(files_copy)
                shard_q[b] = deque(files_copy)
                sample_buf[b] = deque()

        def active_buckets() -> List[Bucket]:
            act: List[Bucket] = []
            for b in shard_q.keys():
                if sample_buf[b]:
                    act.append(b)
                elif shard_q[b]:
                    act.append(b)
            return act

        reset_epoch()

        while True:
            # Periodically discover newly created shards (producer/consumer).
            self._scan_for_new_shards(force=False)

            act = active_buckets()
            if not act:
                # All buckets empty: epoch boundary.
                if self.log_every_epoch:
                    print(
                        f"[bucketed] buckets_empty=True | epoch_end={epoch.epoch_idx} | "
                        f"total_samples_left=0 | est_total_samples={self.total_samples}",
                        flush=True,
                    )
                epoch.epoch_idx += 1
                epoch.batches_yielded = 0
                # Refresh file inventory and restart (epoch = re-sweep all known data).
                reset_epoch()
                # If still empty, wait for producer.
                if not active_buckets():
                    time.sleep(max(0.1, self.poll_secs))
                continue

            # Randomly pick a non-empty bucket.
            b = rng.choice(act)

            # Ensure we have some samples buffered for this bucket.
            while len(sample_buf[b]) == 0 and len(shard_q[b]) > 0:
                shard_path = shard_q[b].popleft()
                try:
                    samples = _load_shard_samples(shard_path, rng=rng)
                except Exception as e:
                    print(f"[bucketed] WARNING: failed loading shard {shard_path}: {e}", flush=True)
                    continue
                for s in samples:
                    sample_buf[b].append(s)

            # If still empty, bucket is exhausted this epoch; skip it.
            if len(sample_buf[b]) == 0:
                continue

            take_n = min(self.batch_size, len(sample_buf[b]))
            batch: List[dict] = [sample_buf[b].popleft() for _ in range(take_n)]

            epoch.batches_yielded += 1
            yield batch

