"""
Bucketed, on-disk preprocessed dataset for *oracle-trajectory* data.

This mirrors the sampling behavior of `data/bucketed_preprocessed_dataset.py` but loads
additional per-step oracle supervision fields:
- oracle_action_ids
- oracle_action_weights

Expected shard schema (npz):
  input_ids: object[ N ] where each element is List[int]
  labels: object[ N ] where each element is List[int] (kept for compatibility)
  canvas_states: object[ N ] where each element is List[List[int]]
  target_token_ids: object[ N ] (optional; not needed for training)
  prompt_lengths: int32[ N ]
  traj_lens: int16[ N ]
  oracle_action_ids: object[ N ] where each element is List[List[int]] (len=traj_len)
  oracle_action_weights: object[ N ] where each element is List[List[float]] (len=traj_len; each sums to 1)
"""

from __future__ import annotations

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
    if not name.startswith("len_"):
        return None
    parts = name.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except Exception:
        return None


def _count_samples_in_shard(shard_path: Path) -> int:
    data = np.load(shard_path, allow_pickle=True)
    if "traj_lens" in data:
        return int(len(data["traj_lens"]))
    if "input_ids" in data:
        return int(len(data["input_ids"]))
    return 0


def _load_shard_samples(shard_path: Path, *, rng: random.Random) -> List[dict]:
    data = np.load(shard_path, allow_pickle=True)
    input_ids = data["input_ids"]
    labels = data.get("labels")
    canvas_states = data.get("canvas_states")
    prompt_lengths = data.get("prompt_lengths")
    traj_lens = data.get("traj_lens")
    oracle_action_ids = data.get("oracle_action_ids")
    oracle_action_weights = data.get("oracle_action_weights")
    oracle_ids_padded = data.get("oracle_ids_padded")
    oracle_wts_padded = data.get("oracle_wts_padded")

    if canvas_states is None or prompt_lengths is None or traj_lens is None:
        raise KeyError(f"Missing required keys in shard: {shard_path}")
    if oracle_action_ids is None or oracle_action_weights is None:
        raise KeyError(f"Missing oracle supervision keys in shard: {shard_path}")

    n = int(len(input_ids))
    idxs = list(range(n))
    rng.shuffle(idxs)

    out: List[dict] = []
    for i in idxs:
        inp = input_ids[i]
        if hasattr(inp, "tolist"):
            inp = inp.tolist()
        lab = labels[i] if labels is not None else [-100] * len(inp)
        if hasattr(lab, "tolist"):
            lab = lab.tolist()
        canv = canvas_states[i]
        if hasattr(canv, "tolist"):
            canv = canv.tolist()
        o_ids = oracle_action_ids[i]
        if hasattr(o_ids, "tolist"):
            o_ids = o_ids.tolist()
        o_w = oracle_action_weights[i]
        if hasattr(o_w, "tolist"):
            o_w = o_w.tolist()
        o_ids_p = oracle_ids_padded[i] if oracle_ids_padded is not None else None
        if hasattr(o_ids_p, "tolist"):
            o_ids_p = o_ids_p.tolist()
        o_w_p = oracle_wts_padded[i] if oracle_wts_padded is not None else None
        if hasattr(o_w_p, "tolist"):
            o_w_p = o_w_p.tolist()

        out.append(
            {
                "input_ids": inp,
                "labels": lab,
                "canvas_sequences": canv,  # collator expects this key
                "prompt_length": int(prompt_lengths[i]),
                "traj_len": int(traj_lens[i]),
                "oracle_action_ids": o_ids,
                "oracle_action_weights": o_w,
                "oracle_ids_padded": o_ids_p,
                "oracle_wts_padded": o_w_p,
            }
        )

    return out


@dataclass
class _EpochState:
    epoch_idx: int = 0
    batches_yielded: int = 0


class OracleBucketedPreprocessedBatchDataset(IterableDataset):
    """
    IterableDataset that yields *lists of samples* (a pre-batched list) from bucketed oracle data.

    Use with DataLoader(batch_size=None, collate_fn=...).
    """

    def __init__(self, data_dir: str, batch_size: int, poll_secs: float = 10.0, seed: int = 42):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = int(batch_size)
        self.poll_secs = float(poll_secs)
        self.seed = int(seed)

        self.buckets_root = self.data_dir / "buckets"
        if not self.buckets_root.exists():
            raise FileNotFoundError(f"Expected buckets/ under {self.data_dir} (got {self.buckets_root})")

        self._all_shards: Dict[Bucket, List[Path]] = {}
        self._known_files: Set[Path] = set()
        self.total_samples: int = 0
        self._last_poll_time = 0.0

        self._scan_for_new_shards(force=True)
        print(
            f"[oracle_bucketed] Initialized from {self.data_dir} | buckets={len(self._all_shards)} "
            f"| shards={sum(len(v) for v in self._all_shards.values())} | est_samples={self.total_samples}",
            flush=True,
        )

    def _scan_for_new_shards(self, *, force: bool) -> None:
        now = time.time()
        if not force and (now - self._last_poll_time) < self.poll_secs:
            return
        self._last_poll_time = now

        for bucket_dir in self.buckets_root.iterdir():
            if not bucket_dir.is_dir():
                continue
            bucket = _parse_bucket_dir_name(bucket_dir.name)
            if bucket is None:
                continue
            self._all_shards.setdefault(bucket, [])
            for shard_path in sorted(bucket_dir.glob("*.npz")):
                if shard_path.name.endswith(".tmp.npz"):
                    continue
                if shard_path in self._known_files:
                    continue
                self._known_files.add(shard_path)
                self._all_shards[bucket].append(shard_path)
                try:
                    self.total_samples += _count_samples_in_shard(shard_path)
                except Exception:
                    pass

    def __iter__(self) -> Iterable[List[dict]]:
        wi = get_worker_info()
        worker_id = int(wi.id) if wi is not None else 0
        num_workers = int(wi.num_workers) if wi is not None else 1

        rng = random.Random(self.seed + 1009 * worker_id + int(time.time()) % 1_000_000)
        epoch = _EpochState(epoch_idx=0, batches_yielded=0)
        last_empty_log_time = 0.0
        empty_log_every_secs = max(30.0, float(self.poll_secs))

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
                if sample_buf[b] or shard_q[b]:
                    act.append(b)
            return act

        reset_epoch()

        while True:
            self._scan_for_new_shards(force=False)
            act = active_buckets()
            if not act:
                now = time.time()
                # Only let one worker emit this message to avoid log spam (each worker has its own loop).
                if worker_id == 0 and (now - last_empty_log_time) >= empty_log_every_secs:
                    last_empty_log_time = now
                    print(
                        f"[oracle_bucketed] buckets_empty=True | epoch_end={epoch.epoch_idx} | total_samples_left=0 | "
                        f"est_total_samples={self.total_samples}",
                        flush=True,
                    )
                epoch.epoch_idx += 1
                epoch.batches_yielded = 0
                reset_epoch()
                if not active_buckets():
                    time.sleep(max(0.1, self.poll_secs))
                continue

            b = rng.choice(act)
            # Ensure we have enough samples buffered to form a full batch when possible.
            # This matters a lot when the producer writes small shards (streaming mode).
            while len(sample_buf[b]) < self.batch_size and len(shard_q[b]) > 0:
                shard_path = shard_q[b].popleft()
                try:
                    samples = _load_shard_samples(shard_path, rng=rng)
                except Exception as e:
                    print(f"[oracle_bucketed] WARNING: failed loading shard {shard_path}: {e}", flush=True)
                    continue
                for s in samples:
                    sample_buf[b].append(s)

            if len(sample_buf[b]) == 0:
                continue

            take_n = min(self.batch_size, len(sample_buf[b]))
            batch = [sample_buf[b].popleft() for _ in range(take_n)]
            epoch.batches_yielded += 1
            yield batch

