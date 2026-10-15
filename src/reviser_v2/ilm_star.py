"""Insertion Language Model (ILM) baseline on our Star Graph generator (tasks 31/32).

Re-implementation of Patel et al., arXiv 2505.05755, following the released code (dhruvdcoder/ILM @ 6cc27f1;
provenance, hashes and line references in notes/sources/ilm-code-6cc27f1.md) and its Star-hard training script
(scripts/train_ilm_stopping_vstar_medium.sh, model_type=ilm_stopping):

* Sequence: [CLS] prompt [BOS] path, where the path is the forward flattened-edge target (our `edge_pairs`).
* Training (collator `DefaultILMCollator` + `ILMWithStoppingClassificationLossWithMaskedCE`): draw k ~ U{0..L} path
  tokens, drop them (the model never sees them; positions are renumbered), and train a *joint* softmax over
  (gap, token), where the target at gap i is (#dropped tokens of each type whose nearest kept left neighbour is i) / k.
  Position 0 (CLS) is excluded; padding is masked; prefix gaps are not masked in training (use_constraint=false).
  A binary stop head on the CLS output learns "nothing is missing" (k == 0). Loss = joint CE + stop CE.
* Model (`RotaryTransformerILMModelWithStoppingClassification`, `rotary_transformer_small`): bidirectional pre-LN
  rotary transformer 12L / 768d / 12 heads, ReLU FFN 3072, dropout 0.1, no final LayerNorm before the vocab head
  (`final_layer_without_normalization=true`); stop head = LN -> (d -> 2d, tanh, -> d) residual -> linear(2).
* Optimisation: AdamW lr 1e-4, weight decay 0.01 (not on biases/norms), 500 linear warmup then constant, batch 64,
  50k steps, bf16 autocast.
* Decoding (`ILMPredictorWithStoppingClassification`, top-1/top-1): stop when P(stop) >= 0.9; otherwise pick the gap
  by argmax of logsumexp over tokens, then the token by argmax; only node labels may be inserted and only after BOS or
  after a path token (prefix gaps suppressed).

Deviations (recorded in the run manifest): our graph generator and train/test pools (the same ones the Reviser star2
runs use), gradient clipping at 1.0, and an insertion cap (a run that hits it counts as not stopped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reviser_v2.tasks import stargraph as sgt


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def specials(spec: sgt.StarSpec) -> Dict[str, int]:
    v = spec.n_content  # node labels, EDGE_SEP, QUERY
    return dict(CLS=v, BOS=v + 1, MASK=v + 2, PAD=v + 3, vocab=v + 4)


def prefix_ids(spec: sgt.StarSpec, g: sgt.StarGraph) -> List[int]:
    s = specials(spec)
    return [s["CLS"]] + sgt.prompt_tokens(spec, g) + [s["BOS"]]


# ---------------------------------------------------------------------------
# Collation (ILM training noise)
# ---------------------------------------------------------------------------
def make_example(prefix: Sequence[int], path: Sequence[int], drop: Sequence[int]):
    """Kept sequence and (gap_index, token) pairs for the dropped path positions `drop`."""
    dropset = set(drop)
    kept, gaps = list(prefix), []
    for i, t in enumerate(path):
        if i in dropset:
            gaps.append((len(kept) - 1, t))   # nearest kept left neighbour (BOS at the latest)
        else:
            kept.append(t)
    return kept, gaps


def collate(spec: sgt.StarSpec, items: Sequence[Tuple[List[int], List[int]]], rng: np.random.Generator):
    s = specials(spec)
    rows = []
    for prefix, path in items:
        L = len(path)
        k = int(rng.integers(0, L + 1))
        drop = rng.permutation(L)[:k].tolist()
        rows.append(make_example(prefix, path, drop) + (k,))
    T = max(len(r[0]) for r in rows)
    B = len(rows)
    x = torch.full((B, T), s["PAD"], dtype=torch.long)
    pad = torch.ones((B, T), dtype=torch.bool)
    tgt = torch.zeros((B, T, s["vocab"]), dtype=torch.float32)
    ks = torch.zeros(B, dtype=torch.long)
    for b, (kept, gaps, k) in enumerate(rows):
        x[b, :len(kept)] = torch.tensor(kept)
        pad[b, :len(kept)] = False
        for gi, t in gaps:
            tgt[b, gi, t] += 1.0 / k
        ks[b] = k
    return x, pad, tgt, ks


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); pos: (T,)."""
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, h: int, ff: int, p: float):
        super().__init__()
        self.h, self.p = h, p
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, keymask, pos):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keymask)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.norm2(x)))


class ILM(nn.Module):
    def __init__(self, vocab: int, d: int = 768, layers: int = 12, heads: int = 12, ff: int = 3072, p: float = 0.1):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ff=ff, p=p)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, p) for _ in range(layers))
        self.out = nn.Linear(d, vocab)
        self.stop_norm = nn.LayerNorm(d)
        self.stop_mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.Tanh(), nn.Linear(2 * d, d))
        self.stop_drop = nn.Dropout(p)
        self.stop_lin = nn.Linear(d, 2, bias=False)

    def forward(self, x, pad):
        keymask = (~pad)[:, None, None, :]
        pos = torch.arange(x.shape[1], device=x.device)
        hdn = self.emb(x)
        for blk in self.blocks:
            hdn = blk(hdn, keymask, pos)
        c = hdn[:, 0]
        c = c + self.stop_drop(self.stop_mlp(self.stop_norm(c)))
        return self.out(hdn), self.stop_lin(c)   # stop class 0 = stop, 1 = continue (ILM convention)


def ilm_loss(model: ILM, x, pad, tgt, ks):
    logits, stop_logits = model(x, pad)
    logits = logits.float()[:, 1:].masked_fill(pad[:, 1:, None], torch.finfo(torch.float32).min)
    ce = F.cross_entropy(logits.reshape(x.shape[0], -1), tgt[:, 1:].reshape(x.shape[0], -1), reduction="none")
    st = F.cross_entropy(stop_logits.float(), (ks > 0).long(), reduction="none")
    return (ce + st).mean(), ce.detach(), st.detach()


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(model: ILM, spec: sgt.StarSpec, graphs: Sequence[sgt.StarGraph], device: str, batch: int = 256,
           threshold: float = 0.9, max_insert: int = 40):
    """Greedy ILM decoding. Returns (path tokens, stopped, n_insertions) per graph."""
    s = specials(spec)
    model.eval()
    out = []
    for i0 in range(0, len(graphs), batch):
        gs = graphs[i0:i0 + batch]
        pre = [prefix_ids(spec, g) for g in gs]
        seqs = [list(p) for p in pre]
        stopped = [False] * len(gs)
        done = [False] * len(gs)
        for _ in range(max_insert + 1):
            act = [j for j in range(len(gs)) if not done[j]]
            if not act:
                break
            T = max(len(seqs[j]) for j in act)
            x = torch.full((len(act), T), s["PAD"], dtype=torch.long)
            pad = torch.ones((len(act), T), dtype=torch.bool)
            allow = torch.zeros((len(act), T), dtype=torch.bool)
            for r, j in enumerate(act):
                x[r, :len(seqs[j])] = torch.tensor(seqs[j])
                pad[r, :len(seqs[j])] = False
                allow[r, len(pre[j]) - 1:len(seqs[j])] = True   # after BOS or after a path token
            x, pad, allow = x.to(device), pad.to(device), allow.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, stop_logits = model(x, pad)
            logits = logits.float()
            logits[:, :, spec.n_nodes:] = -math.inf
            logits = logits.masked_fill(~allow[:, :, None], -math.inf)
            p_stop = torch.softmax(stop_logits.float(), -1)[:, 0]
            gap = torch.logsumexp(logits, -1).argmax(-1)
            tok = logits[torch.arange(len(act), device=device), gap].argmax(-1)
            for r, j in enumerate(act):
                n_ins = len(seqs[j]) - len(pre[j])
                if p_stop[r].item() >= threshold:
                    stopped[j] = done[j] = True
                elif n_ins >= max_insert:
                    done[j] = True        # cap reached without a stop decision: stopped stays False
                else:
                    seqs[j].insert(int(gap[r]) + 1, int(tok[r]))
        for j, g in enumerate(gs):
            out.append((seqs[j][len(pre[j]):], stopped[j], len(seqs[j]) - len(pre[j])))
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def evaluate(model, spec, device, out_dir: Path, tag: str, limit: int = 0, batch: int = 256):
    sets = {"test": list(sgt.eval_graphs(spec, "test"))}
    for v in ("gen_relabeled", "gen_longer_path"):
        sets[v] = sgt.generalization_graphs(spec, v, n=1000)
    rows, summ = [], {}
    t0 = time.time()
    for name, gs in sets.items():
        gs = gs[:limit] if limit else gs
        rs = []
        for g, (toks, stopped, n_ins) in zip(gs, decode(model, spec, gs, device, batch=batch)):
            r = sgt.score(spec, g, toks, False, stopped)
            r.update(set=name, n_actions=n_ins, final=toks)
            rs.append(r)
        agg = sgt.aggregate(rs)
        agg["exact_ci"] = wilson(sum(r["exact"] for r in rs), len(rs))
        agg["mean_actions"] = sum(r["n_actions"] for r in rs) / max(1, len(rs))
        summ[name] = agg
        rows += rs
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()
                                if k != "exact_ci"}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"star_test_{tag}_var.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(decoding="ILM greedy (gap by logsumexp, then token), stop at P(stop)>=0.9, max 40 insertions",
                eval_set_digest=sgt.eval_set_digest(spec, "test"), eval_seconds=time.time() - t0)
    (out_dir / f"star_test_{tag}_var.summary.json").write_text(json.dumps(dict(meta=meta, summary=summ), indent=1))
    return summ


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return ""


def save(path: Path, obj):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args):
    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.steps:                      # smoke tests only; recorded in the manifest (argv)
        cfg["optim"]["steps"] = args.steps
    spec = sgt.spec_from_task_cfg(cfg["task"])
    s = specials(spec)
    run = Path(args.run_dir)
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    seed, data_seed = int(cfg["run"]["seed"]), int(cfg["run"]["data_seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = args.device
    o, m = cfg["optim"], cfg["model"]
    model = ILM(s["vocab"], m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], m["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "norm" not in n]
    nodecay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "norm" not in n)]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=o["weight_decay"]), dict(params=nodecay, weight_decay=0.0)],
                            lr=o["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / o["warmup"]))
    t0 = time.time()
    pool = [sgt.train_graph(spec, data_seed, j) for j in range(spec.n_train)]
    items = [(prefix_ids(spec, g), sgt.target_tokens(spec, g)) for g in pool]
    pool_digest = hashlib.sha256("".join(g.digest() for g in pool).encode()).hexdigest()
    print(f"pool {len(pool)} graphs in {time.time() - t0:.0f}s digest {pool_digest[:16]}", flush=True)
    step, order, ptr = 0, rng.permutation(len(items)), 0
    ck = run / "ckpt" / "latest.pt"
    if ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, order, ptr = st["step"], np.array(st["order"]), st["ptr"]
        rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"resumed at step {step}", flush=True)
    manifest = dict(run=str(run), config=cfg, config_path=str(args.config), commit=git_commit(), params_total=n_params,
                    spec=spec.to_dict(), train_pool_digest=pool_digest, argv=sys.argv, torch=torch.__version__,
                    device=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
                    method="ILM (arXiv 2505.05755) re-implementation; see src/reviser_v2/ilm_star.py docstring")
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    mf = open(run / "metrics.jsonl", "a")
    model.train()
    tl = time.time()
    while step < o["steps"]:
        idx = []
        for _ in range(o["batch_size"]):
            if ptr >= len(order):
                order, ptr = rng.permutation(len(items)), 0
            idx.append(int(order[ptr])); ptr += 1
        x, pad, tgt, ks = (t.to(device) for t in collate(spec, [items[i] for i in idx], rng))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, ce, stl = ilm_loss(model, x, pad, tgt, ks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
        opt.step(); sched.step(); step += 1
        if step % cfg["log"]["every"] == 0:
            rec = dict(step=step, loss=loss.item(), ce=ce.mean().item(), stop_ce=stl.mean().item(), gnorm=float(gn),
                       lr=sched.get_last_lr()[0], sps=cfg["log"]["every"] / (time.time() - tl))
            tl = time.time()
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if step % (cfg["log"]["every"] * 20) == 0:
                print(json.dumps(rec), flush=True)
        if step % cfg["ckpt"]["every"] == 0 or step == o["steps"]:
            save(ck, dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                          order=order.tolist(), ptr=ptr, np_rng=rng.bit_generator.state,
                          torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state() if device.startswith("cuda") else None,
                          model_cfg=model.cfg, spec=spec.to_dict()))
    summ = evaluate(model, spec, device, run / "eval", "ilm", limit=args.eval_limit)
    (run / "done.json").write_text(json.dumps(dict(step=step, seconds=time.time() - t0,
                                                   exact={k: v["exact"] for k, v in summ.items()})))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--run_dir", required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--steps", type=int, default=0, help="override optim.steps (smoke tests)")
    t.add_argument("--eval_limit", type=int, default=0, help="evaluate only the first N graphs per set (smoke tests)")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(a)


if __name__ == "__main__":
    main()
