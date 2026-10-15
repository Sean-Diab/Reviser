#!/usr/bin/env python3
"""
Pretrain CursorTransformer on FineWeb restoration trajectories (action-token LM).

Data source:
  - Restoration trajectories are stored as offset+token shards in:
      <resttraj_dir>/shards/rest_tokens_shard*.npy
      <resttraj_dir>/shards/rest_offsets_shard*.npy
  - Buckets are index-only files in:
      <resttraj_dir>/buckets/len_XXXX_YYYY/{shard_idx.npy,sample_idx.npy}

Training:
  - Next-token LM on action tokens: input = [END_OF_INPUT] + actions, labels = shift-next.
  - Batches are drawn from exactly one bucket at a time (length-homogeneous, optimal padding).
  - Sampling is without replacement within an epoch (each index exactly once per epoch).

Logging:
  - train.py-style loss lines: "Epoch X | Step Y | Loss ... | LR ... | Toks/sec ..."
  - periodic inference tests that print insert/move/delete mix (percentages) for generated actions.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from reviser.data import CursorTokenizer  # noqa: E402
from reviser.data.resttraj_bucketed_index_dataset import BucketedRestTrajIndexDataset  # noqa: E402
from reviser.data.resttraj_pretrain_collator import (  # noqa: E402
    RestTrajPretrainCollator,
    infer_prefix_length_from_initial_inserts,
)
from reviser.model import CursorConfig, CursorTransformer  # noqa: E402
from reviser.model.utils import count_parameters  # noqa: E402

from reviser.decoding.inference_core import (  # noqa: E402
    CursorState,
    _action_distribution,
    _fmt_action_distribution,
    generate_from_action_prefix,
)
from reviser.data.vocabulary import SPECIAL_TOKENS, is_move_token, is_insert_token  # noqa: E402
from training_metrics_dashboard import append_metric_row, maybe_update_dashboard  # noqa: E402


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self.streams:
            try:
                n = s.write(data)
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _prefetch_loader_to_cuda(dl: DataLoader, device: torch.device):
    """
    Wrap a DataLoader so that the *next* batch is asynchronously copied to CUDA
    on a separate stream (hides H2D transfer latency; CPU collation still happens in workers).
    """
    if device.type != "cuda":
        # no-op passthrough
        for batch in dl:
            yield batch
        return

    it = iter(dl)
    stream = torch.cuda.Stream()
    next_batch: Optional[Dict[str, Any]] = None

    def preload() -> None:
        nonlocal next_batch
        try:
            b = next(it)
        except StopIteration:
            next_batch = None
            return
        with torch.cuda.stream(stream):
            next_batch = _move_batch_to_device(b, device)

    preload()
    while next_batch is not None:
        torch.cuda.current_stream().wait_stream(stream)
        batch = next_batch
        preload()
        yield batch


def _timed_iter(it):
    """
    Wrap an (infinite) iterator, returning (item, wait_ms) where wait_ms measures
    how long we blocked waiting for the next batch to become available.
    """
    while True:
        t0 = time.perf_counter()
        item = next(it)
        yield item, (time.perf_counter() - t0) * 1e3


def _setup_logging(*, out_dir: Path, log_file: Optional[str]) -> Path:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"pretrain_resttraj_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _TeeStream(sys.__stdout__, f)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.__stderr__, f)  # type: ignore[assignment]
    print(f"[logging] Writing logs to: {log_path}", flush=True)
    return log_path


def _cuda_perf_knobs() -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Coerce common numeric types
    if "training" in cfg:
        tr = cfg["training"]
        for k in ("learning_rate", "min_learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
            if k in tr and tr[k] is not None:
                tr[k] = float(tr[k])
        for k in ("batch_size", "gradient_accumulation", "num_epochs", "num_workers", "prefetch_factor"):
            if k in tr and tr[k] is not None:
                tr[k] = int(tr[k])
    if "model" in cfg:
        m = cfg["model"]
        for k in ("d_model", "n_heads", "d_ff", "n_layers", "max_seq_length", "max_canvas_length"):
            if k in m and m[k] is not None:
                m[k] = int(m[k])
    if "data" in cfg:
        d = cfg["data"]
        for k in ("max_seq_length", "max_canvas_length", "min_traj_len", "max_traj_len", "bucket_size"):
            if k in d and d[k] is not None:
                d[k] = int(d[k])
    return cfg


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _load_dataset_state(resttraj_dir: Path) -> Dict[str, Any]:
    state_path = resttraj_dir / "state.json"
    if not state_path.is_file():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_prefix_cfg(
    cfg_section: Dict[str, Any],
    dataset_state: Dict[str, Any],
    *,
    fixed_key: str,
    infer_key: str,
    max_key: str,
) -> tuple[Optional[int], bool, Optional[int]]:
    fixed_prefix_length = _coerce_optional_int(cfg_section.get(fixed_key))
    infer_prefix_length = bool(cfg_section.get(infer_key, False))
    prefix_length_max = _coerce_optional_int(cfg_section.get(max_key))
    if prefix_length_max is None:
        prefix_length_max = _coerce_optional_int(dataset_state.get("prefix_len_max"))
    return fixed_prefix_length, infer_prefix_length, prefix_length_max


def _resolve_prefix_length_for_actions(
    actions: List[int],
    *,
    fixed_prefix_length: Optional[int],
    infer_prefix_length: bool,
    prefix_length_max: Optional[int],
) -> int:
    if fixed_prefix_length is not None:
        prefix_len = int(fixed_prefix_length)
    elif infer_prefix_length:
        prefix_len = infer_prefix_length_from_initial_inserts(actions, max_prefix_length=prefix_length_max)
    else:
        prefix_len = 0
    if prefix_length_max is not None:
        prefix_len = min(int(prefix_len), int(prefix_length_max))
    return max(0, min(int(prefix_len), len(actions)))


def _build_resttraj_dataset(
    *,
    resttraj_dir: Path,
    dataset_type: str,
    batch_size: int,
    seed: int,
    min_traj_len: int,
    max_traj_len: int,
    bucket_size: int,
    refresh_secs: float,
    random_bucket_choice: bool,
    shard_cache_size: int,
    log_every_epoch: bool,
):
    if dataset_type == "bucketed_index":
        return BucketedRestTrajIndexDataset(
            resttraj_dir=str(resttraj_dir),
            batch_size=int(batch_size),
            seed=int(seed),
            min_len=int(min_traj_len),
            max_len=int(max_traj_len),
            random_bucket_choice=bool(random_bucket_choice),
            shard_cache_size=int(shard_cache_size),
            log_every_epoch=bool(log_every_epoch),
        )
    if dataset_type == "raining_bucketed":
        from data.training_resttraj_bucketed_dataset import RainingBucketedRestTrajDataset

        return RainingBucketedRestTrajDataset(
            resttraj_dir=str(resttraj_dir),
            batch_size=int(batch_size),
            seed=int(seed),
            min_len=int(min_traj_len),
            max_len=int(max_traj_len),
            bucket_size=int(bucket_size),
            refresh_secs=float(refresh_secs),
            log_every_epoch=bool(log_every_epoch),
        )
    raise ValueError(f"Unknown data.dataset_type: {dataset_type!r} (expected 'bucketed_index' or 'raining_bucketed')")


def _build_resttraj_dataloader(
    *,
    dataset,
    collator: RestTrajPretrainCollator,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    dl_kwargs: Dict[str, Any] = dict(
        batch_size=None,
        num_workers=int(num_workers),
        collate_fn=collator,
        pin_memory=bool(pin_memory),
        persistent_workers=(int(num_workers) > 0),
    )
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **dl_kwargs)


def _estimate_eval_batches(*, resttraj_dir: Path, batch_size: int, explicit_max_batches: int) -> int:
    if int(explicit_max_batches) > 0:
        return int(explicit_max_batches)
    state = _load_dataset_state(resttraj_dir)
    accepted_samples = int(state.get("accepted_samples", 0) or 0)
    if accepted_samples <= 0:
        return 1
    return max(1, math.ceil(accepted_samples / max(1, int(batch_size))))


def create_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "ln" in name or "layernorm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": float(cfg["training"]["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = AdamW(
        groups,
        lr=float(cfg["training"]["learning_rate"]),
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=bool(cfg["training"].get("use_fused", False)),
    )
    # PyTorch 2.x LambdaLR with last_epoch>=0 requires initial_lr on each param group.
    for g in opt.param_groups:
        g.setdefault("initial_lr", float(g["lr"]))
    return opt


def create_scheduler(optimizer: AdamW, cfg: Dict[str, Any], *, total_steps: int, last_epoch: int = -1) -> LambdaLR:
    if cfg.get("training", {}).get("warmup_steps") is not None:
        warmup_steps = int(cfg["training"]["warmup_steps"])
    else:
        warmup_steps = int(total_steps * float(cfg["training"].get("warmup_ratio", 0.03)))
    min_lr_ratio = float(cfg["training"]["min_learning_rate"]) / float(cfg["training"]["learning_rate"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    tokens_processed: int = 0


@dataclass
class DevEvalStats:
    mean_loss: float
    perplexity: float
    action_accuracy: float
    label_tokens: int
    batches: int


@dataclass
class InferenceMixAggregate:
    """Mean insert/move/delete percentages over generated actions, averaged across inference examples."""

    avg_insert_pct: float
    avg_move_pct: float
    avg_delete_pct: float
    n_examples: int
    seed_used: int


def _save_checkpoint(
    *,
    ckpt_path: Path,
    model: CursorTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler],
    state: TrainState,
    cfg: Dict[str, Any],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": {
                "epoch": int(state.epoch),
                "step": int(state.step),
                "tokens_processed": int(state.tokens_processed),
            },
            "config": cfg,
            "saved_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        ckpt_path,
    )


def _write_step_markers(*, ckpt_dir: Path, state: TrainState) -> None:
    """
    Write small, human-readable step markers to disk at each checkpoint.

    - `latest_step.txt` is overwritten each checkpoint for convenience.
    - `step_XXXXXXXXXXXX.txt` is immutable (not overwritten) and records each checkpoint step.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = int(state.step)
    epoch = int(state.epoch)
    tokens_processed = int(state.tokens_processed)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = f"utc_time={ts}\nepoch={epoch}\nstep={step}\ntokens_processed={tokens_processed}\n"

    # Overwritten marker.
    (ckpt_dir / "latest_step.txt").write_text(payload, encoding="utf-8")

    # Immutable marker.
    step_path = ckpt_dir / f"step_{step:012d}.txt"
    if not step_path.exists():
        step_path.write_text(payload, encoding="utf-8")


def _load_checkpoint(
    *,
    ckpt_path: Path,
    model: CursorTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler],
) -> TrainState:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    s = ckpt.get("state", {}) or {}
    return TrainState(
        epoch=int(s.get("epoch", 0)),
        step=int(s.get("step", 0)),
        tokens_processed=int(s.get("tokens_processed", 0)),
    )


@torch.no_grad()
def _load_random_resttraj_from_bucket_indices(
    *,
    resttraj_dir: Path,
    rng: int,
    min_bucket_len: int,
    max_bucket_len: int,
) -> tuple[list[int], str]:
    """
    Sample one restoration trajectory from the index buckets (no replacement guarantee; this is for inference only).
    """
    import random as pyrandom

    rr = pyrandom.Random(int(rng))
    buckets_root = resttraj_dir / "buckets"
    shard_root = resttraj_dir / "shards"
    bucket_dirs = []
    for d in buckets_root.iterdir():
        if not d.is_dir() or not d.name.startswith("len_"):
            continue
        parts = d.name.split("_")
        if len(parts) != 3:
            continue
        b0, b1 = int(parts[1]), int(parts[2])
        if b1 < int(min_bucket_len) or b0 > int(max_bucket_len):
            continue
        if (d / "shard_idx.npy").exists() and (d / "sample_idx.npy").exists():
            bucket_dirs.append((d, b0, b1))
    if not bucket_dirs:
        raise RuntimeError("no buckets found for inference sampling")

    bdir, b0, b1 = rr.choice(bucket_dirs)
    sh = np.load(bdir / "shard_idx.npy", mmap_mode="r")
    si = np.load(bdir / "sample_idx.npy", mmap_mode="r")
    if sh.size == 0:
        raise RuntimeError("empty bucket")
    j = rr.randrange(int(sh.size))
    shard_idx = int(sh[j])
    sample_idx = int(si[j])
    offsets = np.load(shard_root / f"rest_offsets_shard{shard_idx:06d}.npy", mmap_mode="r")
    tokens = np.load(shard_root / f"rest_tokens_shard{shard_idx:06d}.npy", mmap_mode="r")
    st = int(offsets[sample_idx, 0])
    ln = int(offsets[sample_idx, 1])
    traj = tokens[st : st + ln].tolist()
    traj = [int(t) for t in traj]
    tag = f"{bdir.name}/shard{shard_idx:06d} idx={sample_idx}"
    return traj, tag


@torch.no_grad()
def _load_random_resttraj_from_shards(
    *,
    resttraj_dir: Path,
    rng: int,
    min_len: int,
    max_len: int,
    max_tries: int = 250,
) -> tuple[list[int], str]:
    """
    Sample one restoration trajectory directly from shard files (no bucket index required).

    This is used for "raining" datasets produced by `compute_fineweb_resttraj_prefix35_ar_then_8020.py`,
    which write:
      resttraj_dir/shards/rest_tokens_shardXXXXXX.npy
      resttraj_dir/shards/rest_offsets_shardXXXXXX.npy
      resttraj_dir/shards/shardXXXXXX.done
    but do NOT create resttraj_dir/buckets/.
    """
    import random as pyrandom

    rr = pyrandom.Random(int(rng))
    shard_root = resttraj_dir / "shards"
    if not shard_root.is_dir():
        raise FileNotFoundError(f"Missing shards dir: {shard_root}")

    # Prefer completed shards ('.done' marker). If none exist, fall back to any offsets file present.
    done_shards: List[int] = []
    for p in shard_root.glob("shard*.done"):
        name = p.name
        if not name.startswith("shard") or not name.endswith(".done"):
            continue
        mid = name[len("shard") : -len(".done")]
        try:
            done_shards.append(int(mid))
        except Exception:
            continue
    done_shards = sorted(set(done_shards))

    if not done_shards:
        for p in shard_root.glob("rest_offsets_shard*.npy"):
            name = p.name
            # rest_offsets_shard000123.npy
            try:
                mid = name[len("rest_offsets_shard") : -len(".npy")]
                done_shards.append(int(mid))
            except Exception:
                continue
        done_shards = sorted(set(done_shards))

    if not done_shards:
        raise RuntimeError(f"No shard files found under: {shard_root}")

    for _ in range(int(max_tries)):
        shard_idx = int(rr.choice(done_shards))
        offsets_path = shard_root / f"rest_offsets_shard{shard_idx:06d}.npy"
        tokens_path = shard_root / f"rest_tokens_shard{shard_idx:06d}.npy"
        if (not offsets_path.exists()) or (not tokens_path.exists()):
            continue
        offsets = np.load(offsets_path, mmap_mode="r")
        tokens = np.load(tokens_path, mmap_mode="r")
        if int(offsets.shape[0]) <= 0:
            continue
        sample_idx = int(rr.randrange(int(offsets.shape[0])))
        st = int(offsets[sample_idx, 0])
        ln = int(offsets[sample_idx, 1])
        if ln <= 0:
            continue
        if int(ln) < int(min_len) or int(ln) > int(max_len):
            continue
        traj = tokens[st : st + ln].tolist()
        traj = [int(t) for t in traj]
        tag = f"shards/shard{shard_idx:06d} idx={sample_idx} len={ln}"
        return traj, tag

    raise RuntimeError(f"Failed to sample a valid trajectory from shards after {int(max_tries)} tries")


def _action_mix_insert_move_counts(actions: List[int]) -> tuple[int, int, int, int]:
    ins = mov = dele = 0
    total = 0
    for a in actions:
        a = int(a)
        if a == int(SPECIAL_TOKENS.end_of_response) or a == int(SPECIAL_TOKENS.end_of_input):
            continue
        total += 1
        if a == int(SPECIAL_TOKENS.delete):
            dele += 1
        elif bool(is_move_token(a)):
            mov += 1
        elif bool(is_insert_token(a)):
            ins += 1
    return ins, mov, dele, total


def _action_mix_insert_move_fractions(actions: List[int]) -> tuple[float, float, float, int]:
    ins, mov, dele, total = _action_mix_insert_move_counts(actions)
    if total <= 0:
        return 0.0, 0.0, 0.0, 0
    return 100.0 * ins / total, 100.0 * mov / total, 100.0 * dele / total, total


def _action_mix_insert_move(actions: List[int]) -> str:
    ins, mov, dele, total = _action_mix_insert_move_counts(actions)
    if total <= 0:
        return "insert=0.00% move=0.00% delete=0.00% (n=0)"
    return f"insert={100.0*ins/total:.2f}% move={100.0*mov/total:.2f}% delete={100.0*dele/total:.2f}% (n={total})"


def _escape_one_line(s: str) -> str:
    # Keep inference logging single-line and grep-friendly.
    return s.replace("\n", "\\n").replace("\r", "\\r")


def _canvas_with_cursor_text(*, tokenizer: CursorTokenizer, canvas: List[int], cursor_pos: int) -> str:
    cp = max(0, min(int(cursor_pos), len(canvas)))
    left = tokenizer.decode([int(t) for t in canvas[:cp]])
    right = tokenizer.decode([int(t) for t in canvas[cp:]])
    return _escape_one_line(left) + "[CURSOR]" + _escape_one_line(right)


def _truncate_mid(s: str, max_chars: int) -> str:
    s = str(s)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    # Keep both beginning and end; middle is usually repetitive/less informative.
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + " … " + s[-tail:]

@torch.no_grad()
def _generate_from_action_prefix_nocanvas(
    *,
    model: CursorTransformer,
    tokenizer: CursorTokenizer,
    prefix_actions: List[int],
    device: torch.device,
    max_new_actions: int,
    temperature: float,
    top_k: int,
    max_seq_len: int,
    greedy: bool = False,
) -> tuple[str, CursorState]:
    """
    Action-prefix completion that matches this run's training setup:
    - **NO true-canvas attention pooling**
    - **RoPE-only history positions** (no learned positional embedding for edit-history)

    Implementation:
    - Model forward never receives canvas_ids/canvas_mask (always None).
    - We still maintain a CursorState by replaying actions to:
        - compute the final decoded canvas
        - apply legality masking for MOVE/DELETE based on current cursor/canvas
    """

    def _mask_invalid_actions(logits: torch.Tensor, state: CursorState) -> torch.Tensor:
        logits = logits.clone()
        # Mask invalid moves for current cursor/canvas bounds
        for amt in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512]:
            try:
                mv = int(tokenizer.get_move_token_id(int(amt)))  # type: ignore[attr-defined]
            except Exception:
                # Fall back to data.vocabulary for move token IDs
                from data.vocabulary import get_move_token_id

                try:
                    mv = int(get_move_token_id(int(amt)))
                except Exception:
                    continue
            if not state.can_move(int(amt)):
                logits[mv] = float("-inf")
        if not state.can_delete():
            logits[int(SPECIAL_TOKENS.delete)] = float("-inf")
        return logits

    def _sample_next(logits: torch.Tensor) -> int:
        # logits: (V,)
        if bool(greedy):
            return int(torch.argmax(logits).item())
        x = logits
        if float(temperature) != 1.0:
            x = x / float(temperature)
        if int(top_k) > 0:
            k = int(top_k)
            topk_vals, _ = torch.topk(x, k)
            cutoff = topk_vals[-1]
            x = torch.where(x < cutoff, torch.full_like(x, float("-inf")), x)
        probs = F.softmax(torch.nan_to_num(x, neginf=-1e9, posinf=1e9), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    eoi = int(SPECIAL_TOKENS.end_of_input)
    eor = int(SPECIAL_TOKENS.end_of_response)

    # Hard cap to avoid exceeding model RoPE cache / max_seq_length.
    max_actions_total = max(1, int(max_seq_len) - 1)
    max_new_actions = min(int(max_new_actions), max(0, max_actions_total - int(len(prefix_actions))))

    model.eval()
    state = CursorState.from_prompt([eoi])

    # Prefer cached stepping if available.
    past_key_values = None
    pos_idx = 0

    def step_cached(tok_id: int) -> torch.Tensor:
        nonlocal past_key_values, pos_idx
        if hasattr(model, "forward_step_cached_with_hidden"):
            logits_next, _hidden_last, past_key_values = model.forward_step_cached_with_hidden(
                token_id=torch.tensor([int(tok_id)], dtype=torch.long, device=device),
                canvas_ids_last=None,
                canvas_mask_last=None,
                past_key_values=past_key_values,
                position_idx=int(pos_idx),
            )
            pos_idx += 1
            return logits_next[0]
        # Fallback: full forward each time (still nocanvas)
        full_ids = state.prompt_tokens + state.action_history + [int(tok_id)]
        # Keep within max_seq_len
        full_ids = full_ids[-int(max_seq_len) :]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), canvas_ids=None, canvas_mask=None, labels=None)
        pos_idx += 1
        return out["logits"][0, -1, :]

    # Prime cache with END_OF_INPUT (does not mutate env state).
    logits = step_cached(eoi)

    # Replay prefix actions: update model cache and env state.
    for a in prefix_actions:
        a = int(a)
        if a == eor:
            state.action_history.append(a)
            break
        logits = step_cached(a)
        state.execute_action(a)

    # Generate
    for _ in range(int(max_new_actions)):
        masked = _mask_invalid_actions(logits, state)
        tok_id = _sample_next(masked)
        if tok_id == eor:
            state.action_history.append(tok_id)
            break
        logits = step_cached(tok_id)
        state.execute_action(tok_id)

    resp_text = tokenizer.decode([int(t) for t in state.canvas]).replace("\r", "")
    return resp_text, state


@torch.no_grad()
def run_inference_test_resttraj_prefix_k(
    *,
    model: CursorTransformer,
    tokenizer: CursorTokenizer,
    resttraj_dir: Path,
    device: torch.device,
    max_canvas_len: int,
    n_examples: int,
    fixed_prefix_length: Optional[int],
    infer_prefix_length: bool,
    prefix_length_max: Optional[int],
    max_new_actions: int,
    seed: Optional[int],
    min_bucket_len: int,
    max_bucket_len: int,
    greedy: bool,
) -> InferenceMixAggregate:
    import os as _os
    import random as pyrandom

    model.eval()
    seed_used = int(seed) if seed is not None else (int.from_bytes(_os.urandom(8), "little") % (2**31 - 1))
    rng = pyrandom.Random(seed_used)
    sum_ins = sum_mov = sum_del = 0.0
    n_mixed = 0
    print(
        "[inference_resttraj_kprefix] "
        f"seed={seed_used} "
        f"fixed_prefix_length={fixed_prefix_length} "
        f"infer_prefix_length={bool(infer_prefix_length)} "
        f"prefix_length_max={prefix_length_max} "
        f"greedy={bool(greedy)} "
        f"gen_max={int(max_new_actions)} "
        f"n_examples={int(n_examples)}",
        flush=True,
    )
    # Prevent logs from exploding (but still show meaningful output).
    max_canvas_log_chars = 1200
    for ex in range(int(n_examples)):
        # The raining dataset format does not create `resttraj_dir/buckets/`.
        # Fall back to shard-based sampling when buckets are missing.
        if (resttraj_dir / "buckets").is_dir():
            actions, tag = _load_random_resttraj_from_bucket_indices(
                resttraj_dir=resttraj_dir,
                rng=rng.randrange(2**31 - 1),
                min_bucket_len=int(min_bucket_len),
                max_bucket_len=int(max_bucket_len),
            )
        else:
            actions, tag = _load_random_resttraj_from_shards(
                resttraj_dir=resttraj_dir,
                rng=rng.randrange(2**31 - 1),
                min_len=int(min_bucket_len),
                max_len=int(max_bucket_len),
            )
        # Remove trailing EOR for prefix sampling
        if actions and int(actions[-1]) == int(SPECIAL_TOKENS.end_of_response):
            actions_wo_eor = actions[:-1]
        else:
            actions_wo_eor = actions
        prefix_len = _resolve_prefix_length_for_actions(
            actions_wo_eor,
            fixed_prefix_length=fixed_prefix_length,
            infer_prefix_length=infer_prefix_length,
            prefix_length_max=prefix_length_max,
        )
        prefix = actions_wo_eor[:prefix_len]

        # IMPORTANT: This training run is configured as "nocanvas" (no attention pooling).
        # The default `generate_from_action_prefix` builds per-step canvas tensors and would
        # inadvertently condition on true-canvas (and its positional embedding tables).
        # So we use a nocanvas generator here that only feeds edit-history tokens.
        resp_text, st = _generate_from_action_prefix_nocanvas(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix,
            device=device,
            max_new_actions=int(max_new_actions),
            temperature=1.0 if bool(greedy) else 0.8,
            top_k=0 if bool(greedy) else 50,
            max_seq_len=int(getattr(getattr(model, "config", None), "max_seq_length", 256)),
            greedy=bool(greedy),
        )

        prefix_replayed = min(len(prefix), len(st.action_history))
        dist_prefix = _action_distribution(st.action_history[:prefix_replayed])
        dist_gen = _action_distribution(st.action_history[prefix_replayed:])
        dist_total = _action_distribution(st.action_history)

        print(f"[inference_resttraj_kprefix] ex={ex} source={tag}", flush=True)
        print(f"[inference_resttraj_kprefix] prefix_len={prefix_len}", flush=True)
        print(f"[inference_resttraj_kprefix] action_dist_prefix: {_fmt_action_distribution(dist_prefix)}", flush=True)
        print(f"[inference_resttraj_kprefix] action_dist_generated: {_fmt_action_distribution(dist_gen)}", flush=True)
        print(f"[inference_resttraj_kprefix] action_dist_total: {_fmt_action_distribution(dist_total)}", flush=True)
        # Explicit insert/move percentages (generated only)
        gen_mix = _action_mix_insert_move(st.action_history[prefix_replayed:])
        print(f"[inference_resttraj_kprefix] mix_generated: {gen_mix}", flush=True)
        ip, mp, dp, _gn = _action_mix_insert_move_fractions(st.action_history[prefix_replayed:])
        if _gn > 0:
            sum_ins += ip
            sum_mov += mp
            sum_del += dp
            n_mixed += 1

        # Log what the model actually produced on the canvas.
        pred_canvas_text = _canvas_with_cursor_text(tokenizer=tokenizer, canvas=st.canvas, cursor_pos=st.cursor_pos)
        pred_canvas_text = _truncate_mid(pred_canvas_text, max_chars=max_canvas_log_chars)
        print(
            f"[inference_resttraj_kprefix] pred_final_canvas: len_tokens={len(st.canvas)} cursor_pos={int(st.cursor_pos)} text={pred_canvas_text}",
            flush=True,
        )

        # Also show the ground-truth final canvas for this sampled trajectory (helps eyeballing correctness).
        gt = CursorState.from_prompt([int(SPECIAL_TOKENS.end_of_input)])
        for a in actions_wo_eor:
            gt.execute_action(int(a))
        gt_text = _canvas_with_cursor_text(tokenizer=tokenizer, canvas=gt.canvas, cursor_pos=gt.cursor_pos)
        gt_text = _truncate_mid(gt_text, max_chars=max_canvas_log_chars)
        match = int(st.canvas == gt.canvas)
        print(
            f"[inference_resttraj_kprefix] gt_final_canvas:   len_tokens={len(gt.canvas)} cursor_pos={int(gt.cursor_pos)} exact_match={match} text={gt_text}",
            flush=True,
        )

        _ = resp_text  # still available for potential future HTML viz

    model.train()
    ne = int(n_examples)
    if ne <= 0 or n_mixed <= 0:
        return InferenceMixAggregate(0.0, 0.0, 0.0, ne, seed_used)
    inv = float(n_mixed)
    return InferenceMixAggregate(
        avg_insert_pct=sum_ins / inv,
        avg_move_pct=sum_mov / inv,
        avg_delete_pct=sum_del / inv,
        n_examples=ne,
        seed_used=seed_used,
    )


@torch.no_grad()
def evaluate_resttraj_dev(
    *,
    model: CursorTransformer,
    dl: DataLoader,
    device: torch.device,
    use_amp: bool,
    dtype: torch.dtype,
    max_batches: int,
) -> DevEvalStats:
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_label_tokens = 0
    total_batches = 0
    total_correct = 0

    for batch in dl:
        batch = _move_batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        canvas_ids = batch.get("canvas_ids", None)
        canvas_mask = batch.get("canvas_mask", None)

        with autocast("cuda", enabled=use_amp, dtype=dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                canvas_ids=canvas_ids,
                canvas_mask=canvas_mask,
                labels=None,
            )
            logits = out["logits"]

        flat_logits = logits.view(-1, model.config.vocab_size)
        flat_labels = labels.view(-1)
        valid = flat_labels != -100
        if valid.any():
            loss_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
            total_loss += float(loss_sum.item())
            total_label_tokens += int(valid.sum().item())
            preds = flat_logits.argmax(dim=-1)
            total_correct += int((preds[valid] == flat_labels[valid]).sum().item())
        total_batches += 1
        if total_batches >= int(max_batches):
            break

    if model_was_training:
        model.train()

    mean_loss = total_loss / max(1, total_label_tokens)
    perplexity = math.exp(min(20.0, float(mean_loss)))
    action_accuracy = float(total_correct) / max(1, total_label_tokens)
    return DevEvalStats(
        mean_loss=float(mean_loss),
        perplexity=float(perplexity),
        action_accuracy=float(action_accuracy),
        label_tokens=int(total_label_tokens),
        batches=int(total_batches),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument(
        "--resume_reset_optimizer",
        action="store_true",
        help="With --resume: load model (+ train state) only; rebuild AdamW from config and a fresh cosine scheduler "
        "aligned to the resumed step (does not load optimizer/scheduler/scaler from the checkpoint).",
    )
    ap.add_argument(
        "--resume_reset_step",
        action="store_true",
        help="With --resume: reset step/epoch/tokens_processed to 0 (load model weights only).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir).expanduser().resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(out_dir=out_dir, log_file=args.log_file)
    print(f"[pretrain_resttraj] output_dir={out_dir}", flush=True)
    print(f"[pretrain_resttraj] loaded_config={args.config}", flush=True)
    print("Loaded config:", flush=True)
    print(yaml.dump(cfg, default_flow_style=False), flush=True)
    try:
        (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    except Exception:
        pass

    tok = CursorTokenizer()
    pad_id = int(tok.pad_token_id)

    resttraj_dir = Path(cfg["data"]["resttraj_dir"]).expanduser().resolve()
    train_dataset_state = _load_dataset_state(resttraj_dir)
    max_seq_len = int(cfg["data"]["max_seq_length"])
    # Note: model max_canvas_length controls positional embedding table sizes.
    # Collator max_canvas_length controls the (B,S,C) tensors we materialize per batch.
    max_canvas_len = int(cfg["data"].get("max_canvas_length", max_seq_len))
    collator_max_canvas_len = int(cfg["data"].get("collator_max_canvas_length", max_canvas_len))
    use_canvas = bool(cfg["data"].get("use_canvas", True))
    min_traj_len = int(cfg["data"].get("min_traj_len", 1))
    # Allow one extra token for END_OF_INPUT we prepend.
    max_traj_len = int(cfg["data"].get("max_traj_len", max_seq_len - 1))

    dataset_type = str(cfg.get("data", {}).get("dataset_type", "bucketed_index"))
    train_fixed_prefix_length, train_infer_prefix_length, train_prefix_length_max = _resolve_prefix_cfg(
        cfg["data"],
        train_dataset_state,
        fixed_key="train_prefix_length",
        infer_key="train_infer_prefix_length",
        max_key="train_prefix_length_max",
    )
    dataset = _build_resttraj_dataset(
        resttraj_dir=resttraj_dir,
        dataset_type=dataset_type,
        batch_size=int(cfg["training"]["batch_size"]),
        seed=int(cfg["data"].get("seed", 123)),
        min_traj_len=int(min_traj_len),
        max_traj_len=int(max_traj_len),
        bucket_size=int(cfg["data"].get("bucket_size", 5)),
        refresh_secs=float(cfg["data"].get("refresh_secs", 30.0)),
        random_bucket_choice=bool(cfg.get("data", {}).get("random_bucket_choice", True)),
        shard_cache_size=int(cfg.get("data", {}).get("shard_cache_size", 8)),
        log_every_epoch=bool(cfg["data"].get("log_every_epoch", True)),
    )
    collator = RestTrajPretrainCollator(
        pad_token_id=pad_id,
        max_seq_length=int(max_seq_len),
        max_canvas_length=int(collator_max_canvas_len),
        use_canvas=bool(use_canvas),
        fixed_prefix_length=train_fixed_prefix_length,
        infer_prefix_length=train_infer_prefix_length,
        prefix_length_max=train_prefix_length_max,
    )
    # Dataset yields pre-batched lists; DataLoader must not auto-batch.
    num_workers = int(cfg.get("training", {}).get("num_workers", 0) or 0)
    prefetch_factor = int(cfg.get("training", {}).get("prefetch_factor", 2) or 2)
    pin_memory = bool(cfg.get("training", {}).get("pin_memory", True))
    dl = _build_resttraj_dataloader(
        dataset=dataset,
        collator=collator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    dev_resttraj_dir_raw = cfg["data"].get("dev_resttraj_dir")
    dev_resttraj_dir = Path(dev_resttraj_dir_raw).expanduser().resolve() if dev_resttraj_dir_raw else None
    dev_dataset_state = _load_dataset_state(dev_resttraj_dir) if dev_resttraj_dir is not None else {}
    dev_dl: Optional[DataLoader] = None
    dev_min_traj_len = int(cfg["data"].get("dev_min_traj_len", min_traj_len))
    dev_max_traj_len = int(cfg["data"].get("dev_max_traj_len", max_traj_len))
    dev_fixed_prefix_length = None
    dev_infer_prefix_length = False
    dev_prefix_length_max = None
    if dev_resttraj_dir is not None:
        dev_dataset_type = str(cfg["data"].get("dev_dataset_type", dataset_type))
        dev_fixed_prefix_length, dev_infer_prefix_length, dev_prefix_length_max = _resolve_prefix_cfg(
            cfg["data"],
            dev_dataset_state,
            fixed_key="dev_prefix_length",
            infer_key="dev_infer_prefix_length",
            max_key="dev_prefix_length_max",
        )
        dev_batch_size = int(cfg["training"].get("dev_batch_size", cfg["training"]["batch_size"]))
        dev_dataset = _build_resttraj_dataset(
            resttraj_dir=dev_resttraj_dir,
            dataset_type=dev_dataset_type,
            batch_size=dev_batch_size,
            seed=int(cfg["data"].get("dev_seed", cfg["data"].get("seed", 123))),
            min_traj_len=dev_min_traj_len,
            max_traj_len=dev_max_traj_len,
            bucket_size=int(cfg["data"].get("dev_bucket_size", cfg["data"].get("bucket_size", 5))),
            refresh_secs=float(cfg["data"].get("dev_refresh_secs", cfg["data"].get("refresh_secs", 30.0))),
            random_bucket_choice=bool(cfg["data"].get("dev_random_bucket_choice", cfg["data"].get("random_bucket_choice", True))),
            shard_cache_size=int(cfg["data"].get("dev_shard_cache_size", cfg["data"].get("shard_cache_size", 8))),
            log_every_epoch=bool(cfg["data"].get("dev_log_every_epoch", False)),
        )
        dev_collator = RestTrajPretrainCollator(
            pad_token_id=pad_id,
            max_seq_length=int(cfg["data"].get("dev_max_seq_length", max_seq_len)),
            max_canvas_length=int(cfg["data"].get("dev_collator_max_canvas_length", collator_max_canvas_len)),
            use_canvas=bool(cfg["data"].get("dev_use_canvas", use_canvas)),
            fixed_prefix_length=dev_fixed_prefix_length,
            infer_prefix_length=dev_infer_prefix_length,
            prefix_length_max=dev_prefix_length_max,
        )
        dev_dl = _build_resttraj_dataloader(
            dataset=dev_dataset,
            collator=dev_collator,
            num_workers=int(cfg["training"].get("dev_num_workers", 0) or 0),
            prefetch_factor=int(cfg["training"].get("dev_prefetch_factor", prefetch_factor) or prefetch_factor),
            pin_memory=pin_memory,
        )

    # Model
    mcfg = CursorConfig(
        vocab_size=int(cfg["model"].get("vocab_size", tok.vocab_size)),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        d_ff=int(cfg["model"]["d_ff"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        attention_dropout=float(cfg["model"].get("attention_dropout", 0.1)),
        positional_encoding=str(cfg["model"].get("positional_encoding", "absolute")),
        canvas_pool_positional_encoding=str(cfg["model"].get("canvas_pool_positional_encoding", "none")),
        max_seq_length=int(cfg["data"]["max_seq_length"]),
        max_canvas_length=int(cfg["data"].get("max_canvas_length", cfg["data"]["max_seq_length"])),
        tie_embeddings=bool(cfg["model"].get("tie_embeddings", True)),
        initializer_range=float(cfg["model"].get("initializer_range", 0.02)),
    )
    model = CursorTransformer(mcfg)
    n_params = count_parameters(model)
    print(f"[pretrain_resttraj] model_params={n_params:,}", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        _cuda_perf_knobs()
    model.to(device)

    optimizer = create_optimizer(model, cfg)
    total_tokens = int(cfg["training"].get("total_tokens", 40_000_000_000))
    tokens_per_step = int(cfg["training"]["batch_size"]) * int(cfg["data"]["max_seq_length"]) * max(
        1, int(cfg["training"].get("gradient_accumulation", 1))
    )
    total_steps = max(1, int(total_tokens // max(1, tokens_per_step)))
    scheduler = create_scheduler(optimizer, cfg, total_steps=total_steps, last_epoch=-1)

    precision = str(cfg.get("hardware", {}).get("precision_type", "bfloat16")).lower()
    use_amp = (device.type == "cuda") and (precision in ("bf16", "bfloat16", "fp16", "float16"))
    dtype = torch.bfloat16 if precision in ("bf16", "bfloat16") else torch.float16
    scaler = None if dtype == torch.bfloat16 else GradScaler(enabled=use_amp)

    state = TrainState(epoch=0, step=0)
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        print(f"[pretrain_resttraj] resuming from {resume_path}", flush=True)
        ckpt = torch.load(str(resume_path), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        s = ckpt.get("state", {}) or {}
        state = TrainState(
            epoch=int(s.get("epoch", 0)),
            step=int(s.get("step", 0)),
            tokens_processed=int(s.get("tokens_processed", 0)),
        )
        if bool(getattr(args, "resume_reset_step", False)):
            print(
                f"[pretrain_resttraj] resume_reset_step: resetting step={state.step} -> 0, "
                f"tokens={state.tokens_processed} -> 0, epoch={state.epoch} -> 0",
                flush=True,
            )
            state = TrainState(epoch=0, step=0, tokens_processed=0)
        if bool(getattr(args, "resume_reset_optimizer", False)):
            print(
                "[pretrain_resttraj] resume_reset_optimizer: fresh AdamW + cosine schedule from config; "
                "scheduler aligned via fast-forward; not loading optimizer/scheduler from checkpoint",
                flush=True,
            )
            optimizer = create_optimizer(model, cfg)
            scheduler = create_scheduler(optimizer, cfg, total_steps=total_steps, last_epoch=-1)
            # Align LR schedule to resumed step without LambdaLR(last_epoch>=0), which breaks with fused AdamW
            # on some PyTorch builds (param_groups lack visible initial_lr at scheduler init).
            n_ff = int(state.step)
            if n_ff > 0:
                t0 = time.time()
                for _ in range(n_ff):
                    scheduler.step()
                print(
                    f"[pretrain_resttraj] scheduler fast-forward: {n_ff} steps in {(time.time()-t0):.2f}s",
                    flush=True,
                )
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
        print(f"[pretrain_resttraj] resumed epoch={state.epoch} step={state.step}", flush=True)

    grad_accum = int(cfg["training"].get("gradient_accumulation", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    log_every = int(cfg["training"].get("log_every_steps", 50))
    # Default to 20 minutes if not specified.
    ckpt_every_secs = int(cfg["training"].get("checkpoint_interval_secs", 1200))
    ckpt_every_tokens = int(cfg["training"].get("checkpoint_interval_tokens", 0) or 0)
    infer_every_secs = int(cfg["training"].get("inference_interval_secs", 600))
    infer_n = int(cfg["training"].get("inference_n_examples", 3))
    infer_genmax = int(cfg["training"].get("inference_max_new_actions", 120))
    infer_greedy = bool(cfg["training"].get("inference_greedy", True))
    dev_eval_enabled = bool(cfg["training"].get("dev_eval_enabled", dev_dl is not None))
    dev_eval_every_secs = int(cfg["training"].get("dev_eval_interval_secs", infer_every_secs))
    dev_eval_max_batches = _estimate_eval_batches(
        resttraj_dir=dev_resttraj_dir if dev_resttraj_dir is not None else resttraj_dir,
        batch_size=int(cfg["training"].get("dev_batch_size", cfg["training"]["batch_size"])),
        explicit_max_batches=int(cfg["training"].get("dev_eval_max_batches", 0)),
    )
    infer_resttraj_dir = dev_resttraj_dir if dev_resttraj_dir is not None else resttraj_dir
    infer_min_traj_len = dev_min_traj_len if dev_resttraj_dir is not None else min_traj_len
    infer_max_traj_len = dev_max_traj_len if dev_resttraj_dir is not None else max_traj_len
    infer_fixed_prefix_length = _coerce_optional_int(cfg["training"].get("inference_fixed_prefix_length"))
    infer_prefix_from_initial_inserts = bool(cfg["training"].get("inference_infer_prefix_length", False))
    infer_prefix_length_max = _coerce_optional_int(cfg["training"].get("inference_prefix_length_max"))
    if infer_fixed_prefix_length is None and not infer_prefix_from_initial_inserts and infer_prefix_length_max is None:
        infer_fixed_prefix_length = train_fixed_prefix_length
        infer_prefix_from_initial_inserts = train_infer_prefix_length
        infer_prefix_length_max = train_prefix_length_max

    metrics_enabled = bool(cfg.get("training", {}).get("metrics_log_enabled", True))
    metrics_dir = Path(cfg.get("training", {}).get("metrics_dir", str(out_dir / "metrics"))).expanduser().resolve()
    metrics_csv = metrics_dir / str(cfg.get("training", {}).get("metrics_csv_name", "training_metrics.csv"))
    metrics_html = metrics_dir / str(cfg.get("training", {}).get("metrics_html_name", "dashboard.html"))
    metrics_refresh_html = bool(cfg.get("training", {}).get("metrics_refresh_html", True))
    metrics_title = str(cfg.get("training", {}).get("metrics_dashboard_title", f"Training metrics — {out_dir.name}"))
    metrics_auto_refresh = int(cfg.get("training", {}).get("metrics_dashboard_auto_refresh_sec", 45))

    def _metrics_refresh() -> None:
        maybe_update_dashboard(
            csv_path=metrics_csv,
            html_path=metrics_html,
            enabled=bool(metrics_enabled and metrics_refresh_html),
            title=metrics_title,
            auto_refresh_sec=metrics_auto_refresh,
        )

    if metrics_enabled:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[metrics] enabled csv={metrics_csv} html={metrics_html} auto_refresh_sec={metrics_auto_refresh}",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    last_log_t = time.time()
    last_ckpt_t = time.time()
    last_ckpt_tokens = int(state.tokens_processed)
    last_infer_t = time.time()
    last_dev_eval_t = time.time()
    window_tokens = 0
    window_steps = 0
    window_loss_sum = 0.0
    window_label_tokens = 0
    accum_input_tokens = 0

    # Timing windows (ms); computed without synchronizing except when logging.
    timing_enabled = bool(cfg.get("training", {}).get("timing_enabled", True))
    timing_every = int(cfg.get("training", {}).get("timing_log_every_steps", log_every) or log_every)
    timing_sample_every = int(cfg.get("training", {}).get("timing_sample_every_steps", 10) or 10)
    tw_data_wait = 0.0
    tw_collate = 0.0
    tw_canvas = 0.0
    tw_fwd = 0.0
    tw_bwd = 0.0
    tw_opt = 0.0
    tw_count = 0
    tw_gpu_count = 0
    tw_infer_ms = 0.0
    tw_ckpt_ms = 0.0

    def save_latest() -> float:
        t0 = time.perf_counter()
        latest_path = ckpt_dir / "checkpoint_latest.pt"
        # Always write/overwrite the "latest" checkpoint.
        _save_checkpoint(
            ckpt_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            cfg=cfg,
        )

        # Optionally also write a non-overwritten snapshot keyed by step.
        write_step_snapshots = bool(cfg.get("training", {}).get("checkpoint_write_step_snapshots", True))
        snapshot_written = None
        if write_step_snapshots:
            snap_path = ckpt_dir / f"checkpoint_step{int(state.step):012d}.pt"
            if not snap_path.exists():
                _save_checkpoint(
                    ckpt_path=snap_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                    cfg=cfg,
                )
                snapshot_written = snap_path

        # Also write small step marker files (one overwritten, one immutable).
        _write_step_markers(ckpt_dir=ckpt_dir, state=state)
        print(
            f"[checkpoint] step={int(state.step)} tokens_processed={int(state.tokens_processed)} "
            f"latest={latest_path} snapshot={snapshot_written if snapshot_written is not None else 'unchanged'}",
            flush=True,
        )
        return (time.perf_counter() - t0) * 1e3

    prefetch_to_device = bool(cfg.get("training", {}).get("prefetch_to_device", True))
    data_iter = _prefetch_loader_to_cuda(dl, device) if (prefetch_to_device and device.type == "cuda") else dl
    timed_batches = _timed_iter(iter(data_iter))
    stop_training = False

    for epoch in range(int(state.epoch), int(cfg["training"].get("num_epochs", 1))):
        state.epoch = int(epoch)
        for batch, data_wait_ms in timed_batches:
            if int(state.step) >= int(total_steps) and not bool(cfg["training"].get("run_forever", False)):
                stop_training = True
                break

            if not (prefetch_to_device and device.type == "cuda"):
                batch = _move_batch_to_device(batch, device)

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            canvas_ids = batch.get("canvas_ids", None)
            canvas_mask = batch.get("canvas_mask", None)

            # Collator timing metadata (computed in workers, CPU-side)
            collate_ms = float(batch.get("collate_ms", torch.tensor([0.0])).item())
            canvas_ms = float(batch.get("canvas_ms", torch.tensor([0.0])).item())

            try:
                fwd0 = fwd1 = bwd0 = bwd1 = opt0 = opt1 = None
                step_after = int(state.step) + 1  # step id after optimizer update
                do_gpu_sample = bool(
                    timing_enabled and device.type == "cuda" and (timing_sample_every > 0) and (step_after % timing_sample_every == 0)
                )
                if do_gpu_sample:
                    fwd0 = torch.cuda.Event(enable_timing=True)
                    fwd1 = torch.cuda.Event(enable_timing=True)
                    bwd0 = torch.cuda.Event(enable_timing=True)
                    bwd1 = torch.cuda.Event(enable_timing=True)
                    opt0 = torch.cuda.Event(enable_timing=True)
                    opt1 = torch.cuda.Event(enable_timing=True)
                    fwd0.record()
                with autocast("cuda", enabled=use_amp, dtype=dtype):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        canvas_ids=canvas_ids,
                        canvas_mask=canvas_mask,
                        labels=None,
                    )
                    logits = out["logits"]
                    flat_labels = labels.view(-1)
                    valid_labels = flat_labels != -100
                    valid_label_tokens = int(valid_labels.sum().item())
                    if valid_label_tokens > 0:
                        loss_fp32 = F.cross_entropy(logits.view(-1, model.config.vocab_size), flat_labels, ignore_index=-100)
                    else:
                        loss_fp32 = logits.new_tensor(0.0)
                if do_gpu_sample and fwd1 is not None:
                    fwd1.record()
            except torch.OutOfMemoryError as e:
                print(f"[oom] forward/loss: epoch={state.epoch} step={state.step} err={e}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            loss = loss_fp32 / max(1, grad_accum)
            try:
                if do_gpu_sample and bwd0 is not None:
                    bwd0.record()
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if do_gpu_sample and bwd1 is not None:
                    bwd1.record()
            except torch.OutOfMemoryError as e:
                print(f"[oom] backward: epoch={state.epoch} step={state.step} err={e}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            window_loss_sum += float(loss_fp32.item()) * float(valid_label_tokens)
            window_label_tokens += int(valid_label_tokens)
            batch_input_tokens = int(attention_mask.sum().item())
            window_tokens += batch_input_tokens
            accum_input_tokens += batch_input_tokens
            window_steps += 1

            do_step = (window_steps % grad_accum) == 0
            if do_step:
                try:
                    if do_gpu_sample and opt0 is not None:
                        opt0.record()
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if do_gpu_sample and opt1 is not None:
                        opt1.record()
                except torch.OutOfMemoryError as e:
                    print(f"[oom] optimizer_step: epoch={state.epoch} step={state.step} err={e}", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state.step += 1
                state.tokens_processed += int(accum_input_tokens)
                accum_input_tokens = 0
                if int(state.step) >= int(total_steps) and not bool(cfg["training"].get("run_forever", False)):
                    stop_training = True

                # Accumulate timing (synchronize only when logging to keep overhead low).
                if timing_enabled:
                    tw_data_wait += float(data_wait_ms)
                    tw_collate += float(collate_ms)
                    tw_canvas += float(canvas_ms)
                    if do_gpu_sample and device.type == "cuda" and fwd0 is not None:
                        torch.cuda.synchronize()
                        tw_fwd += float(fwd0.elapsed_time(fwd1))  # ms
                        tw_bwd += float(bwd0.elapsed_time(bwd1)) if bwd0 is not None else 0.0
                        tw_opt += float(opt0.elapsed_time(opt1)) if opt0 is not None else 0.0
                        tw_gpu_count += 1
                    tw_count += 1

            if do_step and (state.step % log_every) == 0:
                dt = max(1e-6, time.time() - last_log_t)
                toks_per_s = window_tokens / dt
                lr = float(scheduler.get_last_lr()[0])
                avg_loss = window_loss_sum / max(1, window_label_tokens)
                print(
                    f"Epoch {state.epoch} | Step {state.step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Toks/sec: {toks_per_s:,.0f} | LabelToks: {window_label_tokens}",
                    flush=True,
                )
                if metrics_enabled:
                    append_metric_row(
                        csv_path=metrics_csv,
                        row={
                            "timestamp_unix": time.time(),
                            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                            "event": "train_loss",
                            "step": int(state.step),
                            "tokens_processed": int(state.tokens_processed),
                            "train_loss": float(avg_loss),
                            "lr": float(lr),
                            "toks_per_sec": float(toks_per_s),
                            "label_toks": int(window_label_tokens),
                        },
                    )
                    _metrics_refresh()
                last_log_t = time.time()
                window_tokens = 0
                window_loss_sum = 0.0
                window_label_tokens = 0

            if do_step and timing_enabled and (state.step % timing_every) == 0 and tw_count > 0:
                gpu_den = max(1, tw_gpu_count)
                print(
                    "[timing] "
                    f"step={state.step} "
                    f"data_wait_ms={tw_data_wait/tw_count:.1f} "
                    f"collate_ms={tw_collate/tw_count:.1f} "
                    f"canvas_ms={tw_canvas/tw_count:.1f} "
                    f"fwd_ms={tw_fwd/gpu_den:.1f} "
                    f"bwd_ms={tw_bwd/gpu_den:.1f} "
                    f"opt_ms={tw_opt/gpu_den:.1f} "
                    f"gpu_samples={tw_gpu_count}/{tw_count} "
                    f"infer_ms={tw_infer_ms:.0f} "
                    f"ckpt_ms={tw_ckpt_ms:.0f}",
                    flush=True,
                )
                tw_data_wait = tw_collate = tw_canvas = tw_fwd = tw_bwd = tw_opt = 0.0
                tw_infer_ms = 0.0
                tw_ckpt_ms = 0.0
                tw_count = 0
                tw_gpu_count = 0

            if do_step:
                should_ckpt = False
                if ckpt_every_tokens > 0 and (int(state.tokens_processed) - int(last_ckpt_tokens)) >= ckpt_every_tokens:
                    should_ckpt = True
                elif ckpt_every_tokens <= 0 and ckpt_every_secs > 0 and (time.time() - last_ckpt_t) >= ckpt_every_secs:
                    should_ckpt = True
                if should_ckpt:
                    tw_ckpt_ms += float(save_latest())
                    last_ckpt_t = time.time()
                    last_ckpt_tokens = int(state.tokens_processed)

            if do_step and dev_eval_enabled and dev_dl is not None and dev_eval_every_secs > 0 and (time.time() - last_dev_eval_t) >= dev_eval_every_secs:
                print("\n" + "=" * 80, flush=True)
                print(f"Running dev evaluation at step {state.step}...", flush=True)
                print("=" * 80, flush=True)
                try:
                    t0 = time.perf_counter()
                    dev_stats = evaluate_resttraj_dev(
                        model=model,
                        dl=dev_dl,
                        device=device,
                        use_amp=use_amp,
                        dtype=dtype,
                        max_batches=int(dev_eval_max_batches),
                    )
                    print(
                        f"[dev_eval] step={state.step} batches={dev_stats.batches} label_tokens={dev_stats.label_tokens} "
                        f"mean_loss={dev_stats.mean_loss:.6f} ppl={dev_stats.perplexity:.3f} "
                        f"action_acc={100.0 * dev_stats.action_accuracy:.2f}%",
                        flush=True,
                    )
                    if metrics_enabled:
                        append_metric_row(
                            csv_path=metrics_csv,
                            row={
                                "timestamp_unix": time.time(),
                                "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                                "event": "dev_eval",
                                "step": int(state.step),
                                "tokens_processed": int(state.tokens_processed),
                                "dev_loss": float(dev_stats.mean_loss),
                                "dev_ppl": float(dev_stats.perplexity),
                                "dev_action_acc": float(dev_stats.action_accuracy),
                            },
                        )
                        _metrics_refresh()
                    tw_infer_ms += (time.perf_counter() - t0) * 1e3
                except Exception as e:
                    print(f"Dev evaluation failed: {e}", flush=True)
                last_dev_eval_t = time.time()
                model.train()

            if do_step and infer_every_secs > 0 and (time.time() - last_infer_t) >= infer_every_secs:
                print("\n" + "=" * 80, flush=True)
                print(f"Running inference test at step {state.step}...", flush=True)
                print("=" * 80, flush=True)
                try:
                    t0 = time.perf_counter()
                    mix = run_inference_test_resttraj_prefix_k(
                        model=model,
                        tokenizer=tok,
                        resttraj_dir=infer_resttraj_dir,
                        device=device,
                        max_canvas_len=int(max_canvas_len),
                        n_examples=int(infer_n),
                        fixed_prefix_length=infer_fixed_prefix_length,
                        infer_prefix_length=infer_prefix_from_initial_inserts,
                        prefix_length_max=infer_prefix_length_max,
                        max_new_actions=int(infer_genmax),
                        seed=None,
                        min_bucket_len=int(infer_min_traj_len),
                        max_bucket_len=int(infer_max_traj_len),
                        greedy=bool(infer_greedy),
                    )
                    if metrics_enabled and mix.n_examples > 0:
                        append_metric_row(
                            csv_path=metrics_csv,
                            row={
                                "timestamp_unix": time.time(),
                                "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                                "event": "inference",
                                "step": int(state.step),
                                "tokens_processed": int(state.tokens_processed),
                                "infer_insert_pct": float(mix.avg_insert_pct),
                                "infer_move_pct": float(mix.avg_move_pct),
                                "infer_delete_pct": float(mix.avg_delete_pct),
                                "infer_seed": int(mix.seed_used),
                            },
                        )
                        _metrics_refresh()
                    tw_infer_ms += (time.perf_counter() - t0) * 1e3
                except Exception as e:
                    print(f"Inference test failed: {e}", flush=True)
                last_infer_t = time.time()
                model.train()
            if stop_training:
                break
        if stop_training:
            break

    if int(state.step) > 0 and int(state.tokens_processed) != int(last_ckpt_tokens):
        tw_ckpt_ms += float(save_latest())

    print(f"[pretrain_resttraj] finished at epoch={state.epoch} step={state.step} total_steps={total_steps}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError as e:
        print(f"[oom] {e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
