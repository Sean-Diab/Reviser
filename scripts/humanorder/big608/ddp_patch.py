#!/usr/bin/env python3
"""Add optional multi-GPU (DDP) training and exact data-order resume to the July-control trainer
(pretrain_resttraj.patched.py, or a later patch of it such as the prompt-lock trainer).

big608 run, helper B, 2026-10-08 (notes/big608-2026-10-08.md). Applied as anchored text edits: each anchor must occur
exactly once in the input, so a changed trainer fails loudly instead of being patched in the wrong place.

What it changes (nothing when WORLD_SIZE is unset/1 and BIG_SKIP_GLOBAL_BATCHES is unset/0, except an inert collate
wrapper and a cheap "skip placeholder" check):
  * torchrun (WORLD_SIZE > 1): NCCL process group, one GPU per rank, model wrapped in DDP for the training forward only
    (checkpoints, inference tests and dev eval use the unwrapped model on rank 0).
  * Same global batch and same data order as one GPU: every rank builds the identical batch stream (same dataset, seed
    and num_workers) and keeps global micro-batch g only if g % WORLD == RANK. Single GPU step s uses g = 4s..4s+3
    (gradient_accumulation 4); with WORLD ranks each rank accumulates 4/WORLD micro-batches and DDP averages the
    gradients over ranks, so each optimizer step sees the same 32 sequences with the same per-micro-batch loss
    weighting (loss / 4 summed over the 4 micro-batches). The skipped batches are replaced by a tiny placeholder inside
    the DataLoader worker before collation.
  * tokens_processed is summed over ranks every step (identical on all ranks, so stop/snapshot decisions agree);
    logged loss / label tokens / tokens per second are summed over ranks at each log step.
  * Rank 0 alone writes checkpoints, metrics in out_dir, inference tests and dev evals; ranks > 0 log to
    out_dir/ddp_rankR/.
  * torch_seed + RANK seeds each rank's dropout RNG (rank 0 = the single-GPU seed).
  * BIG_SKIP_GLOBAL_BATCHES=N skips the first N global micro-batches of the stream (resume after a crash at step S:
    N = 4*S, with --resume CKPT and without --resume_reset_*): the data order then continues exactly where it stopped.

Usage: python ddp_patch.py IN.py OUT.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EDITS = [
    # 1. imports + skip-collate wrapper (module level, picklable)
    ("from torch.utils.data import DataLoader\n",
     "from torch.utils.data import DataLoader\n"
     "import torch.distributed as dist  # [big608 ddp patch]\n"
     "from contextlib import nullcontext  # [big608 ddp patch]\n"
     "\n"
     "\n"
     "class _BigSkipCollate:\n"
     "    \"\"\"[big608 ddp patch] Collate only the global micro-batches this rank trains on (g % world == rank, g >= skip).\n"
     "\n"
     "    The DataLoader hands IterableDataset items out round-robin over its workers, in order, so worker w's k-th item\n"
     "    is global item g = k * num_workers + w. Other items become a tiny placeholder (no collation).\n"
     "    \"\"\"\n"
     "\n"
     "    def __init__(self, inner, world: int, rank: int, skip: int) -> None:\n"
     "        self.inner, self.world, self.rank, self.skip = inner, int(world), int(rank), int(skip)\n"
     "        self._k = 0\n"
     "\n"
     "    def __call__(self, batch):\n"
     "        from torch.utils.data import get_worker_info\n"
     "        wi = get_worker_info()\n"
     "        w, nw = (0, 1) if wi is None else (int(wi.id), int(wi.num_workers))\n"
     "        g = self._k * nw + w\n"
     "        self._k += 1\n"
     "        if g < self.skip or (g % self.world) != self.rank:\n"
     "            return {\"_big_skip\": True}\n"
     "        return self.inner(batch)\n"),
    # 2. DDP init right after argument parsing
    ("    args = ap.parse_args()\n",
     "    args = ap.parse_args()\n"
     "    # [big608 ddp patch] torchrun environment\n"
     "    _ddp_world = int(os.environ.get(\"WORLD_SIZE\", \"1\"))\n"
     "    _ddp_rank = int(os.environ.get(\"RANK\", \"0\"))\n"
     "    _ddp_local = int(os.environ.get(\"LOCAL_RANK\", \"0\"))\n"
     "    _ddp = _ddp_world > 1\n"
     "    _big_skip = int(os.environ.get(\"BIG_SKIP_GLOBAL_BATCHES\", \"0\") or 0)\n"
     "    if _ddp:\n"
     "        torch.cuda.set_device(_ddp_local)\n"
     "        dist.init_process_group(backend=\"nccl\")\n"
     "        args.device = f\"cuda:{_ddp_local}\"\n"),
    # 3. per-rank output dir for ranks > 0 (logs, config copy, metrics); checkpoints are rank 0 only anyway
    ("    out_dir = Path(args.output_dir).expanduser().resolve()\n",
     "    out_dir = Path(args.output_dir).expanduser().resolve()\n"
     "    if _ddp and _ddp_rank > 0:  # [big608 ddp patch]\n"
     "        out_dir = out_dir / f\"ddp_rank{_ddp_rank}\"\n"),
    # 4. per-rank dropout seed (rank 0 keeps the single-GPU seed)
    ("        torch.manual_seed(int(_torch_seed))\n"
     "        if torch.cuda.is_available():\n"
     "            torch.cuda.manual_seed_all(int(_torch_seed))\n",
     "        torch.manual_seed(int(_torch_seed) + _ddp_rank)  # [big608 ddp patch] + rank\n"
     "        if torch.cuda.is_available():\n"
     "            torch.cuda.manual_seed_all(int(_torch_seed) + _ddp_rank)\n"),
    # 5. rank filter / resume skip around the training collator
    ("    dl = _build_resttraj_dataloader(\n"
     "        dataset=dataset,\n"
     "        collator=collator,\n",
     "    print(f\"[big608 ddp] world={_ddp_world} rank={_ddp_rank} local_rank={_ddp_local} \"\n"
     "          f\"skip_global_batches={_big_skip}\", flush=True)\n"
     "    dl = _build_resttraj_dataloader(\n"
     "        dataset=dataset,\n"
     "        collator=_BigSkipCollate(collator, _ddp_world, _ddp_rank, _big_skip),  # [big608 ddp patch]\n"),
    # 6. local accumulation = global accumulation / world
    ("    grad_accum = int(cfg[\"training\"].get(\"gradient_accumulation\", 1))\n",
     "    grad_accum = int(cfg[\"training\"].get(\"gradient_accumulation\", 1))\n"
     "    if _ddp:  # [big608 ddp patch] same global batch: each rank accumulates grad_accum / world micro-batches\n"
     "        assert grad_accum % _ddp_world == 0, (grad_accum, _ddp_world)\n"
     "        grad_accum = grad_accum // _ddp_world\n"
     "        print(f\"[big608 ddp] local gradient_accumulation={grad_accum} x world {_ddp_world}\", flush=True)\n"),
    # 7. DDP wrapper for the training forward only
    ("    model.train()\n"
     "    optimizer.zero_grad(set_to_none=True)\n"
     "\n"
     "    last_log_t = time.time()\n",
     "    model.train()\n"
     "    optimizer.zero_grad(set_to_none=True)\n"
     "    fwd_model = model  # [big608 ddp patch]\n"
     "    if _ddp:\n"
     "        fwd_model = torch.nn.parallel.DistributedDataParallel(\n"
     "            model, device_ids=[_ddp_local], output_device=_ddp_local, broadcast_buffers=False,\n"
     "            find_unused_parameters=bool(int(os.environ.get(\"BIG_DDP_FIND_UNUSED\", \"1\"))))\n"
     "\n"
     "    last_log_t = time.time()\n"),
    # 8. rank-0-only checkpoint writes
    ("    def save_latest(force_snapshot: bool = False) -> float:\n",
     "    def save_latest(force_snapshot: bool = False) -> float:\n"
     "        if _ddp and _ddp_rank != 0:  # [big608 ddp patch] rank 0 writes every checkpoint\n"
     "            return 0.0\n"),
    # 9. drop placeholders before anything else in the loop
    ("        for batch, data_wait_ms in timed_batches:\n",
     "        for batch, data_wait_ms in timed_batches:\n"
     "            if isinstance(batch, dict) and batch.get(\"_big_skip\"):  # [big608 ddp patch]\n"
     "                continue\n"),
    # 10. training forward through DDP; gradient sync only on the last local micro-batch of a step
    ("                with autocast(\"cuda\", enabled=use_amp, dtype=dtype):\n"
     "                    out = model(\n"
     "                        input_ids=input_ids,\n"
     "                        attention_mask=attention_mask,\n"
     "                        canvas_ids=canvas_ids,\n"
     "                        canvas_mask=canvas_mask,\n"
     "                        labels=None,\n"
     "                    )\n",
     "                if _ddp:  # [big608 ddp patch]\n"
     "                    fwd_model.require_backward_grad_sync = ((window_steps + 1) % grad_accum) == 0\n"
     "                with autocast(\"cuda\", enabled=use_amp, dtype=dtype):\n"
     "                    out = fwd_model(\n"
     "                        input_ids=input_ids,\n"
     "                        attention_mask=attention_mask,\n"
     "                        canvas_ids=canvas_ids,\n"
     "                        canvas_mask=canvas_mask,\n"
     "                        labels=None,\n"
     "                    )\n"),
    # 11. tokens_processed summed over ranks
    ("                state.tokens_processed += int(accum_input_tokens)\n",
     "                if _ddp:  # [big608 ddp patch] global input tokens of this step\n"
     "                    _tp = torch.tensor([int(accum_input_tokens)], dtype=torch.long, device=device)\n"
     "                    dist.all_reduce(_tp)\n"
     "                    state.tokens_processed += int(_tp.item())\n"
     "                else:\n"
     "                    state.tokens_processed += int(accum_input_tokens)\n"),
    # 12. log-window sums over ranks
    ("            if do_step and (state.step % log_every) == 0:\n"
     "                dt = max(1e-6, time.time() - last_log_t)\n",
     "            if do_step and (state.step % log_every) == 0:\n"
     "                if _ddp:  # [big608 ddp patch] global loss / label tokens / input tokens of the window\n"
     "                    _w = torch.tensor([float(window_loss_sum), float(window_label_tokens), float(window_tokens)],\n"
     "                                      dtype=torch.float64, device=device)\n"
     "                    dist.all_reduce(_w)\n"
     "                    window_loss_sum, window_label_tokens, window_tokens = (\n"
     "                        float(_w[0].item()), int(round(_w[1].item())), int(round(_w[2].item())))\n"
     "                dt = max(1e-6, time.time() - last_log_t)\n"),
    # 13. dev eval and inference tests on rank 0 only
    ("            if do_step and dev_eval_enabled and dev_dl is not None",
     "            if do_step and (not _ddp or _ddp_rank == 0) and dev_eval_enabled and dev_dl is not None"),
    ("            if do_step and infer_every_secs > 0 and",
     "            if do_step and (not _ddp or _ddp_rank == 0) and infer_every_secs > 0 and"),
    # 14. teardown
    ("    print(f\"[pretrain_resttraj] finished at epoch={state.epoch} step={state.step} total_steps={total_steps}\", flush=True)\n",
     "    print(f\"[pretrain_resttraj] finished at epoch={state.epoch} step={state.step} total_steps={total_steps}\", flush=True)\n"
     "    if _ddp:  # [big608 ddp patch]\n"
     "        dist.barrier()\n"
     "        dist.destroy_process_group()\n"),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"edit {i}: anchor found {n} times (need exactly 1): {old[:80]!r}")
        text = text.replace(old, new)
    dst.write_text(text, encoding="utf-8", newline="\n")
    print(f"in  {sha(raw)}  {src}")
    print(f"out {sha(dst.read_bytes())}  {dst}")
    print(f"patcher {sha(Path(__file__).read_bytes())}  {Path(__file__).name}  edits={len(EDITS)}")


if __name__ == "__main__":
    main()
