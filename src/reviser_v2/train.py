"""Config-driven training with reproducible manifests, frozen code snapshots, and exact resume.

Usage: python -m reviser_v2.train --config configs/toy/d1.yaml [--run-dir runs/NAME] [--set key=value ...]

Loss normalization (``loss_norm``):
  * ``eligible`` (primary for the D1/D2 A/B): sum of CE over retained labels divided by the number
    of *eligible* action targets in the batch, i.e. before the doomed-insert mask. D1 and D2 therefore
    share the denominator and shared labels keep equal weight; D2 simply drops doomed-insert terms.
  * ``retained``: mean over retained labels (sensitivity analysis only).

Multi-GPU (DDP; notes/edit-1b-build-2026-10-08.md): ``torchrun --nproc_per_node N -m reviser_v2.train ...``. The global
batch (``optim.batch_size`` episodes, same fixed schedule) is split into contiguous slices, one per rank; each rank
builds only its own slice. The loss denominator is the global eligible-label count (all-reduce), each rank backpropagates
world x its CE sum / that count and DDP averages, so the update equals the single-process update up to numerics; the
clip, scaler and optimizer act on the synced gradients. Micro-batches keep the size batch_size / accum on every rank.
Rank 0 alone writes the manifest, logs, heartbeat and checkpoints and runs the dev eval. The checkpoint holds the global
step / next_idx and full optimizer state, so a run resumes with any number of processes. Without torchrun (WORLD_SIZE
unset or 1) nothing changes. ``ckpt.keep: {name: step}`` also writes a weights-only copy at that step to
``<run>/keep/<name>/ckpt/latest.pt`` (evaluate_edits / clean_draft_discrim accept ``<run>/keep/<name>`` as a run dir).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml

from .actions import ActionSpace
from .data import ProgressionSource, collate, collate_seqs
from .trajectories import episode_to_sequence
from .model import ActionLM, ModelConfig
from .tasks import progression as prog

ROOT = Path(__file__).resolve().parents[2]


def deep_set(d: Dict, dotted: str, value):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = yaml.safe_load(value)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict:
    if os.environ.get("RV2_BUNDLE_COMMIT"):  # remote runs unpack a `git archive` of a committed tree
        # (or a working-tree tarball: RV2_BUNDLE_SOURCE / RV2_BUNDLE_DIRTY=1 say so; per-file sha256 in code_sha256)
        return {"commit": os.environ["RV2_BUNDLE_COMMIT"], "source": os.environ.get("RV2_BUNDLE_SOURCE", "git archive bundle"),
                "dirty": os.environ.get("RV2_BUNDLE_DIRTY", "0") == "1", "bundle_sha256": os.environ.get("RV2_BUNDLE_SHA256")}

    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # git may be unavailable inside some environments
            return f"unavailable: {e}"
    diff = run("diff", "HEAD", "--", "src", "configs", "scripts")
    return {"commit": run("rev-parse", "HEAD"), "dirty_src_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "dirty": bool(diff)}


def snapshot_code(run_dir: Path) -> Dict[str, str]:
    dst = run_dir / "code_snapshot" / "reviser_v2"
    if dst.exists():
        shutil.rmtree(dst)
    src = ROOT / "src" / "reviser_v2"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return {str(p.relative_to(dst)): sha256_file(p) for p in sorted(dst.rglob("*.py"))}


def build(cfg: Dict):
    if cfg["task"]["name"] == "stargraph":
        from .tasks import stargraph as sgt
        spec = sgt.spec_from_task_cfg(cfg["task"])
    elif cfg["task"]["name"] == "realedit":
        from .realdata.editscript import gpt2_action_space
        spec = cfg["task"]
        space = gpt2_action_space()
        assert all(getattr(space, k) == v for k, v in cfg.get("space", {}).items()), "realedit uses gpt2_action_space()"
        if cfg.get("init"):  # architecture comes from the warm-start checkpoint
            ck = torch.load(cfg["init"]["ckpt"], map_location="cpu", weights_only=False, mmap=True)
            assert ck["action_space"] == space.to_dict(), "init checkpoint action space differs"
            return spec, space, ModelConfig(**ck["model_config"])
        return spec, space, ModelConfig(vocab_size=space.vocab_size, **cfg["model"])
    elif cfg["task"]["name"] == "realedit_hf":  # pretrained HF backbone (hf_backbone.py; task 36)
        from .tasks.realedit_hf import build_hf
        return build_hf(cfg)
    elif cfg["task"]["name"] == "editmix":  # Reviser-Edit v2 mix (tasks/editmix.py) with 5 input-only task tokens
        from .realdata.editscript import gpt2_action_space
        from .tasktokens import N_TASKS
        spec = cfg["task"]
        space = gpt2_action_space()
        if cfg.get("init"):  # architecture comes from the warm-start checkpoint
            ck = torch.load(cfg["init"]["ckpt"], map_location="cpu", weights_only=False, mmap=True)
            assert ck["action_space"] == space.to_dict(), "init checkpoint action space differs"
            mc = dict(ck["model_config"])
            del ck
        else:
            mc = dict(vocab_size=space.vocab_size, **cfg["model"])
        assert (mc.get("n_out") or mc["vocab_size"]) == space.vocab_size, "editmix needs a GPT-2 action-space model"
        if spec.get("task_tokens", True):  # extra embedding rows after the action vocab; logits stay action-vocab wide
            mc.update(vocab_size=space.vocab_size + N_TASKS, n_out=space.vocab_size)
        return spec, space, ModelConfig(**mc)
    else:
        spec = prog.ProgSpec(**cfg["task"].get("spec", {}))
    space = ActionSpace(n_content=spec.n_content, **cfg["space"])
    mcfg = ModelConfig(vocab_size=space.vocab_size, **cfg["model"])
    return spec, space, mcfg


def make_sources(cfg: Dict, space: ActionSpace, spec):
    mix, ds = cfg["data"]["mix"], cfg["run"]["data_seed"]
    if cfg["task"]["name"] == "stargraph":
        from .tasks import stargraph as sgt
        return sgt.StarGraphSource(space, spec, mix, ds, "train"), sgt.StarGraphSource(space, spec, mix, ds, "dev")
    if cfg["task"]["name"] == "realedit":
        from .tasks.realedit import RealEditSource
        return RealEditSource(space, spec, mix, ds, "train"), RealEditSource(space, spec, mix, ds + 7919, "dev")
    if cfg["task"]["name"] == "editmix":
        from .tasks.editmix import EditMixSource
        return EditMixSource(space, spec, mix, ds, "train"), EditMixSource(space, spec, mix, ds + 7919, "dev")
    if cfg["task"]["name"] == "realedit_hf":
        from .tasks.realedit_hf import RealEditHFSource
        return RealEditHFSource(space, spec, mix, ds, "train"), RealEditHFSource(space, spec, mix, ds + 7919, "dev")
    return ProgressionSource(space, spec, mix, ds, "train"), ProgressionSource(space, spec, mix, ds + 7919, "dev")


def dist_init(device: str):
    """(rank, world, device). Single process unless launched by torchrun with WORLD_SIZE > 1 (NCCL on CUDA, gloo on CPU)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, device
    import datetime
    import torch.distributed as dist
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if device.startswith("cuda"):
        torch.cuda.set_device(local)
        device, backend = f"cuda:{local}", "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=int(os.environ.get("RV2_DIST_TIMEOUT_MIN", "60"))))
    return rank, world, device


class LossModule(torch.nn.Module):
    """Summed CE of one micro-batch (the computation of the original loop, wrapped so DDP sees one forward)."""

    def __init__(self, model, sparse_logits: bool):
        super().__init__()
        self.model, self.sparse_logits = model, sparse_logits

    def forward(self, cids, clab):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cids.device.type != "cpu"):
            if self.sparse_logits:  # identical loss; logits only where a label exists (GPT-2-size vocab would not fit)
                _, _, h = self.model(cids, return_hidden=True, skip_logits=True)
                lm = clab != -100
                logits = self.model.logits_from_hidden(h[lm])
                lab = clab[lm]
            else:
                logits, _ = self.model(cids)
                lab = clab.reshape(-1)
        logits = logits.float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), lab, ignore_index=-100, reduction="sum")


def lr_at(step, cfg):
    o = cfg["optim"]
    if step < o["warmup"]:
        return o["lr"] * (step + 1) / o["warmup"]
    frac = (step - o["warmup"]) / max(1, o["steps"] - o["warmup"])
    return o["lr"] * (o.get("min_lr_frac", 0.1) + (1 - o.get("min_lr_frac", 0.1)) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


def class_counts(space: ActionSpace, labels: torch.Tensor) -> Dict[str, int]:
    v = labels[labels != -100]
    return {"ins": int((v < space.n_content).sum()), "del": int((v == space.DELETE).sum()),
            "stop": int((v == space.STOP).sum()), "move": int((v >= space.move_start).sum())}


@torch.no_grad()
def dev_eval(model, space, src_dev, cfg, device, n_batches=4):
    model.eval()
    bs = cfg["optim"]["batch_size"]
    tot, n, correct = 0.0, 0, 0
    per = {k: [0, 0] for k in ["ins", "del", "move", "stop"]}
    for b in range(n_batches):
        eps = [src_dev.episode(b * bs + i)[0] for i in range(bs)]
        ids, labels, elig, _ = collate(space, eps, cfg["data"].get("mask_doomed", False), cfg["data"]["max_len"])
        ids, elig = ids.to(device), elig.to(device)
        m = elig != -100
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device != "cpu"):
            if cfg["data"].get("sparse_logits", False):  # (N, V) logits at eligible positions only
                _, _, h = model(ids, return_hidden=True, skip_logits=True)
                lg = model.logits_from_hidden(h[m])
            else:
                logits, _ = model(ids)
                lg = logits[m]
        lg, tgt = lg.float(), elig[m]
        tot += F.cross_entropy(lg, tgt, reduction="sum").item()
        n += int(m.sum())
        pred = lg.argmax(-1)
        correct += int((pred == tgt).sum())
        for k, sel in [("ins", tgt < space.n_content), ("del", tgt == space.DELETE),
                       ("move", tgt >= space.move_start), ("stop", tgt == space.STOP)]:
            per[k][0] += int((pred[sel] == tgt[sel]).sum())
            per[k][1] += int(sel.sum())
    model.train()
    return {"dev_action_ce": tot / max(1, n), "dev_acc": correct / max(1, n),
            **{f"dev_acc_{k}": (c / t if t else None) for k, (c, t) in per.items()}, "dev_n": n}


def main(argv=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # read at first CUDA allocation
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", action="extend", default=[])  # repeated --set flags accumulate
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0, help="episode-generation worker processes (does not change data order)")
    ap.add_argument("--stop-at-step", type=int, default=None, help="checkpoint and exit at this step (simulated interruption / bounded session)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        deep_set(cfg, k, v)
    run_dir = Path(args.run_dir or (ROOT / "runs" / cfg["run"]["name"]))
    rank, world, device = dist_init(args.device)
    is0 = rank == 0
    if world > 1:
        import torch.distributed as dist
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt" / "latest.pt"
    resuming = ckpt_path.exists()
    if resuming:
        saved_cfg = json.load(open(run_dir / "manifest.json"))["config"]
        if saved_cfg != cfg:
            raise SystemExit("config differs from the run's manifest; refusing to resume into a different run")
    spec, space, mcfg = build(cfg)
    seed = cfg["run"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if getattr(mcfg, "kind", None) == "hf":  # HFConfig (task realedit_hf)
        from .hf_backbone import HFActionLM
        model = HFActionLM(mcfg).to(device)
    else:
        model = ActionLM(mcfg).to(device)
    init_info = None
    if cfg.get("init") and not resuming:  # warm start (weights only; fresh optimizer/schedule)
        ip = cfg["init"]["ckpt"]
        from .ckpt_hash import state_dict_sha256
        sd = torch.load(ip, map_location=device, weights_only=False)["model"]
        h = th = None
        if is0:  # DDP broadcasts rank 0's weights at construction, so only rank 0 checks the hashes
            h = sha256_file(Path(ip))
            th = state_dict_sha256(sd)
            # weights_sha256 (canonical tensor hash) identifies a checkpoint rebuilt elsewhere; file sha256 is exact-file only
            if cfg["init"].get("weights_sha256") and cfg["init"]["weights_sha256"] != th:
                raise SystemExit(f"init checkpoint weights_sha256 mismatch: {th}")
            if not cfg["init"].get("weights_sha256") and cfg["init"].get("sha256") and cfg["init"]["sha256"] != h:
                raise SystemExit(f"init checkpoint sha256 mismatch: {h}")
        n_new = model.embed.weight.shape[0] - sd["embed.weight"].shape[0]
        tt_info = None
        if n_new:  # editmix task tokens: new input rows = mean of the GPT-2 pieces of each category name
            from .tasktokens import N_TASKS, TASKS, default_init_rows, expand_embedding
            assert cfg["task"]["name"] == "editmix" and n_new == N_TASKS, f"init has {n_new} fewer embedding rows than the model"
            sd = expand_embedding(sd, n_new, default_init_rows())
            tt_info = {"tasks": list(TASKS), "first_id": int(sd["embed.weight"].shape[0] - n_new),
                       "init_rows": default_init_rows(), "init": "mean of GPT-2 embedding rows of ' '+name"}
        model.load_state_dict(sd)
        del sd
        init_info = {"ckpt": ip, "sha256": h, "weights_sha256": th, "task_tokens": tt_info}
    o = cfg["optim"]
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": o.get("weight_decay", 0.1)},
                             {"params": no_decay, "weight_decay": 0.0}], lr=o["lr"], betas=tuple(o.get("betas", [0.9, 0.95])))
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    src, src_dev = make_sources(cfg, space, spec)
    step, next_idx = 0, 0
    counters = {"processed_tokens": 0, "eligible_labels": 0, "retained_labels": 0, "doomed_masked": 0,
                "episodes": 0, "truncated_episodes": 0, "label_ins": 0, "label_del": 0, "label_move": 0, "label_stop": 0}
    if resuming:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, next_idx, counters = ck["step"], ck["next_idx"], ck["counters"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device != "cpu" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        del ck
        if is0:
            print(f"[resume] step={step} next_idx={next_idx} world={world}", flush=True)
    elif is0:
        manifest = {
            "run_name": cfg["run"]["name"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": cfg, "action_space": space.to_dict(), "vocab_size": space.vocab_size,
            "model_config": mcfg.to_dict(), "params_total": model.num_params(),
            "params_non_embedding": model.num_params(non_embedding=True),
            "git": git_info(), "code_sha256": snapshot_code(run_dir),
            "command": " ".join([sys.executable, "-m", "reviser_v2.train"] + (argv if argv is not None else sys.argv[1:])),
            "env": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform(),
                    "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None},
            "init": init_info,
            "data_mix": src.describe() if hasattr(src, "describe") else None,
            "resume_command": f"python -m reviser_v2.train --config {args.config} --run-dir {run_dir}",
        }
        if world > 1:
            manifest["dist"] = {"world_size": world, "backend": dist.get_backend(),
                                "note": "global batch split into contiguous per-rank slices; see segments.jsonl"}
        json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=2)
    bs, max_len = o["batch_size"], cfg["data"]["max_len"]
    mask_doomed = cfg["data"].get("mask_doomed", False)
    norm = cfg["data"].get("loss_norm", "eligible")
    sparse_logits = cfg["data"].get("sparse_logits", False)
    ev = cfg.get("eval", {})
    keep = {int(s): str(name) for name, s in (cfg["ckpt"].get("keep") or {}).items()}  # weights-only copies (pass ends)
    accum = o.get("accum", 1)  # micro-batches per step (same data and loss; for models whose full batch does not fit)
    assert bs >= world, f"batch {bs} smaller than world size {world}"
    lo, hi = rank * bs // world, (rank + 1) * bs // world  # this rank's slice of every global batch
    # micro-batches on this rank: the single-process count, or enough to keep the micro-batch size bs / accum
    accum_local = accum if world == 1 else max(1, -(-(hi - lo) // max(1, bs // accum)))
    loss_mod = LossModule(model, sparse_logits)
    fwd = loss_mod
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        fwd = DDP(loss_mod, device_ids=[torch.cuda.current_device()] if device != "cpu" else None,
                  broadcast_buffers=False, bucket_cap_mb=int(o.get("ddp_bucket_mb", 25)))
        if o.get("ddp_comm_hook"):  # optional (default off): all-reduce gradients in 16 bit to halve PCIe traffic
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
            fwd.register_comm_hook(None, {"fp16": dh.fp16_compress_hook, "bf16": dh.bf16_compress_hook}[o["ddp_comm_hook"]])
    mlog = open(run_dir / "metrics.jsonl", "a") if is0 else None
    if is0 and (world > 1 or (run_dir / "segments.jsonl").exists()):
        with open(run_dir / "segments.jsonl", "a") as f:
            f.write(json.dumps({"start_step": step, "next_idx": next_idx, "world_size": world, "accum_local": accum_local,
                                "ddp_comm_hook": o.get("ddp_comm_hook"),
                                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "command": " ".join([sys.executable, "-m", "reviser_v2.train"] +
                                                    (argv if argv is not None else sys.argv[1:]))}) + "\n")
    t_last = time.time()
    model.train()
    batches = iter_batches(src, space, bs, mask_doomed, max_len, next_idx, o["steps"] - step, args.workers, lo=lo, hi=hi)
    while step < o["steps"]:
        ids, labels, elig, trunc = next(batches)
        next_idx += bs
        ids, labels = ids.to(device), labels.to(device)
        n_elig = int((elig != -100).sum())
        n_ret = int((labels != -100).sum())
        cc = class_counts(space, labels)
        n_tok = int(ids.numel())
        if world > 1:  # global counts (the loss denominator is the global eligible-label count)
            st = torch.tensor([n_elig, n_ret, n_tok, trunc, cc["ins"], cc["del"], cc["stop"], cc["move"]],
                              dtype=torch.int64, device=device)
            dist.all_reduce(st)
            n_elig, n_ret, n_tok, trunc, ci_, cd_, cs_, cm_ = (int(x) for x in st.tolist())
            cc = {"ins": ci_, "del": cd_, "stop": cs_, "move": cm_}
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        denom = n_elig if norm == "eligible" else max(1, n_ret)
        opt.zero_grad(set_to_none=True)
        ce_sum = torch.zeros((), device=device)
        comp_local = 0
        nl = hi - lo
        if o.get("trim_micro", False):  # opt-in (task 37, long code episodes): rows sorted by labelled length, each
            # micro-batch cut after its last labelled position. Same loss and gradient (a sum over rows; causal
            # attention: trailing columns never affect earlier positions), less padding compute.
            need = torch.where(labels != -100, torch.arange(1, labels.size(1) + 1, device=labels.device), 0).max(1).values
            order = torch.argsort(need, descending=True, stable=True)
        for ci in range(accum_local):
            sl = slice(ci * nl // accum_local, (ci + 1) * nl // accum_local) if accum_local > 1 else slice(None)
            if o.get("trim_micro", False):
                rows = order[sl]
                n = max(1, int(need[rows].max()))
                cids, clab = ids[rows, :n], labels[rows, :n]
                comp_local += cids.numel()
            else:
                cids, clab = ids[sl], labels[sl]
            # DDP: gradients are all-reduced (averaged) only on the last micro-batch
            sync = contextlib.nullcontext() if world == 1 or ci == accum_local - 1 else fwd.no_sync()
            with sync:
                c_ce = fwd(cids, clab)
                if world == 1:
                    scaler.scale(c_ce / max(1, denom)).backward()
                else:  # DDP averages over ranks: scale by world so the sum over ranks / denom is the global loss
                    scaler.scale(c_ce * (world / max(1, denom))).backward()
            ce_sum = ce_sum + c_ce.detach()
        if world > 1:  # global CE sum (logging) and computed-token count
            red = torch.stack([ce_sum.double(), torch.tensor(float(comp_local), dtype=torch.float64, device=device)])
            dist.all_reduce(red)
            ce_sum, comp_local = red[0].float(), int(round(red[1].item()))
        if o.get("trim_micro", False):
            counters["computed_tokens"] = counters.get("computed_tokens", 0) + comp_local
        loss = ce_sum / max(1, denom)
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("grad_clip", 1.0))
        if hasattr(model, "delete_row_grad"):  # HFActionLM: DELETE is a new row outside the backbone embedding
            del_row_grad = model.delete_row_grad(space)
        else:
            del_row_grad = float(model.embed.weight.grad[space.DELETE].norm()) if model.embed.weight.grad is not None else 0.0
        scaler.step(opt)
        scaler.update()
        step += 1
        counters["processed_tokens"] += n_tok
        counters["eligible_labels"] += n_elig
        counters["retained_labels"] += n_ret
        counters["doomed_masked"] += n_elig - n_ret
        counters["episodes"] += bs
        counters["truncated_episodes"] += trunc
        for k in cc:
            counters[f"label_{k}"] += cc[k]
        if not math.isfinite(loss.item()) and is0:
            print(f"[warn] non-finite loss at step {step}", flush=True)
        if is0 and (step % cfg["log"]["every"] == 0 or step == 1):
            rec = {"step": step, "loss": loss.item(), "ce_per_retained": ce_sum.item() / max(1, n_ret), "lr": lr_at(step - 1, cfg),
                   "grad_norm": float(gnorm), "delete_row_grad_norm": del_row_grad, "scale": scaler.get_scale(),
                   "batch_labels": cc, "sec_per_step": (time.time() - t_last) / cfg["log"]["every"], **counters}
            if world > 1:
                rec["world_size"] = world
            t_last = time.time()
            if ev.get("every") and (step % ev["every"] == 0 or step == o["steps"]):
                rec.update(dev_eval(model, space, src_dev, cfg, device))
            mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
            print(json.dumps({k: rec[k] for k in rec if k in ("step", "loss", "lr", "grad_norm", "dev_action_ce", "dev_acc", "dev_acc_del")}), flush=True)
            json.dump({"step": step, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(), "loss": loss.item()},
                      open(run_dir / "heartbeat.json", "w"))
        stopping = args.stop_at_step is not None and step >= args.stop_at_step  # (>=: a resume past the stop step)
        if is0 and (step % cfg["ckpt"]["every"] == 0 or step == o["steps"] or stopping):
            save(run_dir, model, opt, scaler, step, next_idx, counters, mcfg, space)
        if is0 and step in keep:
            save_weights(run_dir / "keep" / keep[step], model, step, next_idx, counters, mcfg, space, keep[step])
        if stopping:
            if is0:
                print(f"[stop-at-step] exiting at step {step}", flush=True)
            if world > 1:
                dist.barrier()
                dist.destroy_process_group()
            return
    if is0:
        json.dump({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step, "counters": counters},
                  open(run_dir / "done.json", "w"), indent=2)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


_GEN = {}


def _gen_seqs(k):
    """Worker: pure-Python episode generation for batch k (no torch ops in the child)."""
    src, space, bs, mask_doomed, start, lo, hi = _GEN["a"]
    return [episode_to_sequence(space, src.episode(start + k * bs + i)[0], mask_doomed) for i in range(lo, hi)]


def iter_batches(src, space, bs, mask_doomed, max_len, start, n, workers, window=16, lo=0, hi=None):
    """Batch k = collate(episodes start + k*bs + lo ... start + k*bs + hi - 1), a pure function of k, in order
    (lo, hi = this DDP rank's slice of the global batch; default the whole batch).

    workers > 0 forks a pool (pipes only; drvfs has no Unix sockets) whose children run only
    Python episode generation; tensors are built in the parent. Prefetch is bounded by ``window``.
    """
    _GEN["a"] = (src, space, bs, mask_doomed, start, lo, bs if hi is None else hi)
    n = max(0, n)
    if workers <= 0:
        for k in range(n):
            yield collate_seqs(_gen_seqs(k), max_len)
        return
    import collections
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        pend = collections.deque()
        nxt = 0
        while nxt < n or pend:
            while nxt < n and len(pend) < window:
                pend.append(pool.apply_async(_gen_seqs, (nxt,)))
                nxt += 1
            yield collate_seqs(pend.popleft().get(timeout=600), max_len)


def save(run_dir, model, opt, scaler, step, next_idx, counters, mcfg, space):
    d = run_dir / "ckpt"
    d.mkdir(exist_ok=True)
    tmp = d / "latest.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "next_idx": next_idx, "counters": counters, "model_config": mcfg.to_dict(), "action_space": space.to_dict(),
                "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}, tmp)
    os.replace(tmp, d / "latest.pt")


def save_weights(keep_dir: Path, model, step, next_idx, counters, mcfg, space, name):
    """Weights-only copy (no optimizer) at <keep_dir>/ckpt/latest.pt, loadable by evaluate_edits as a run dir."""
    d = keep_dir / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "latest.pt.tmp"
    torch.save({"model": model.state_dict(), "step": step, "next_idx": next_idx, "counters": dict(counters),
                "model_config": mcfg.to_dict(), "action_space": space.to_dict(), "kept_as": name,
                "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, tmp)
    os.replace(tmp, d / "latest.pt")
    json.dump({"name": name, "step": step, "next_idx": next_idx, "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "note": "weights-only checkpoint written by reviser_v2.train (ckpt.keep)"}, open(keep_dir / "keep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
