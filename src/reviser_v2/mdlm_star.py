"""Masked diffusion (MDLM) baseline on our Star Graph generator (tasks 31/32).

MDLM (Sahoo et al., arXiv 2406.07524): absorbing-state masked diffusion with the SUBS parameterisation and the
log-linear schedule alpha_t = 1 - t, trained on the continuous-time NELBO. It is conditioned on the graph prompt,
which is never masked.

* Sequence: [CLS] prompt [BOS] c_1 .. c_W, with a fixed canvas of W = 40 slots (ILM's insertion cap).
  The canvas holds the forward flattened-edge target (`edge_pairs`) followed by END tokens.
  Rows are right-padded with PAD (attention padding only; PAD is never a target).
* Training: per example t ~ U(eps, 1) with antithetic batch sampling (MDLM `_sample_t`). Each canvas slot is masked
  independently with probability t. The loss is (1/t) * sum over masked slots of -log p(x_slot), divided by W.
  SUBS: the MASK logit is -inf, and unmasked slots carry over (no loss). There is no time conditioning (MDLM's
  log-linear SUBS model is time-independent).
* Model: the backbone of the ILM baseline (`ilm_star.Block`: bidirectional pre-LN rotary, 12L/768d/12H, ReLU FFN
  3072, dropout 0.1) plus a vocabulary head. No stop head: length is the position of the first END.
* Optimisation: identical to the ILM baseline (AdamW lr 1e-4, wd 0.01, 500 warmup then constant, batch 64, 50k steps,
  bf16 autocast, grad clip 1.0) on the same 50k-graph training pool.
* Decoding (predictions restricted to node labels and END, as ILM restricts insertions to node labels):
  - `ancestral` (primary; MDLM's sampler with T = W steps, t_i = 1 - i/T): at each step each masked slot is revealed
    with probability (t - s)/t, and a revealed slot takes the argmax token. Reveals use a per-graph seeded RNG.
    The last step reveals everything. A step in which nothing is revealed skips the network (MDLM `ddpm_cache`).
  - `confidence` (secondary; MaskGIT/LLaDA-style): reveal the single most confident masked slot per step, W steps.
  - Output: canvas tokens before the first END. `stopped` is true when every slot after the first END is END, so
    stray tokens after END count as not stopped (not exact).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reviser_v2.tasks import stargraph as sgt
from reviser_v2.ilm_star import Block, git_commit, save, wilson

WIDTH = 40
EPS = 1e-3


def specials(spec: sgt.StarSpec) -> Dict[str, int]:
    v = spec.n_content  # node labels, EDGE_SEP, QUERY
    return dict(CLS=v, BOS=v + 1, MASK=v + 2, PAD=v + 3, END=v + 4, vocab=v + 5)


def prefix_ids(spec: sgt.StarSpec, g: sgt.StarGraph) -> List[int]:
    s = specials(spec)
    return [s["CLS"]] + sgt.prompt_tokens(spec, g) + [s["BOS"]]


def canvas_ids(spec: sgt.StarSpec, g: sgt.StarGraph, width: int = WIDTH) -> List[int]:
    t = sgt.target_tokens(spec, g)
    if len(t) > width:
        raise ValueError(f"target length {len(t)} exceeds the canvas width {width}")
    return t + [specials(spec)["END"]] * (width - len(t))


class MDLM(nn.Module):
    def __init__(self, vocab: int, d: int = 768, layers: int = 12, heads: int = 12, ff: int = 3072, p: float = 0.1):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ff=ff, p=p)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, p) for _ in range(layers))
        self.out = nn.Linear(d, vocab)

    def forward(self, x, pad):
        keymask = (~pad)[:, None, None, :]
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h, keymask, pos)
        return self.out(h)


def batch_rows(spec, pres: Sequence[List[int]], canv: Sequence[List[int]]):
    """(x, pad, canvas_mask) with the canvas right after each prefix."""
    s = specials(spec)
    T = max(len(p) + len(c) for p, c in zip(pres, canv))
    B = len(pres)
    x = torch.full((B, T), s["PAD"], dtype=torch.long)
    pad = torch.ones((B, T), dtype=torch.bool)
    cm = torch.zeros((B, T), dtype=torch.bool)
    for b, (p, c) in enumerate(zip(pres, canv)):
        x[b, :len(p) + len(c)] = torch.tensor(list(p) + list(c))
        pad[b, :len(p) + len(c)] = False
        cm[b, len(p):len(p) + len(c)] = True
    return x, pad, cm


def collate(spec, items, rng: np.random.Generator):
    s = specials(spec)
    B = len(items)
    t = (rng.random() + np.arange(B) / B) % 1.0          # antithetic (low-discrepancy) t, MDLM _sample_t
    t = (1 - EPS) * t + EPS
    x0, pad, cm = batch_rows(spec, [p for p, _ in items], [c for _, c in items])
    u = torch.from_numpy(rng.random(x0.shape)).float()
    tt = torch.from_numpy(t).float()
    masked = cm & (u < tt[:, None])
    xt = x0.masked_fill(masked, s["MASK"])
    return xt, pad, x0, masked, tt


def subs_logits(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    logits = logits.float()
    logits[..., mask_id] = -math.inf
    return logits


def mdlm_loss(model, spec, xt, pad, x0, masked, tt, width: int = WIDTH):
    logits = subs_logits(model(xt, pad), specials(spec)["MASK"])
    nll = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")          # (B, T)
    nll = (nll * masked).sum(1) / tt / width
    return nll.mean(), (nll * tt * width / masked.sum(1).clamp(min=1)).detach()


@torch.no_grad()
def decode(model, spec, graphs, device, mode: str = "ancestral", steps: int = WIDTH, batch: int = 256,
           width: int = WIDTH, seed: int = 0):
    """Returns (tokens before the first END, stopped, network calls) per graph."""
    s = specials(spec)
    allowed = torch.full((s["vocab"],), -math.inf, device=device)
    allowed[:spec.n_nodes] = 0.0
    allowed[s["END"]] = 0.0
    model.eval()
    out = []
    for i0 in range(0, len(graphs), batch):
        gs = graphs[i0:i0 + batch]
        pres = [prefix_ids(spec, g) for g in gs]
        x, pad, cm = batch_rows(spec, pres, [[s["MASK"]] * width] * len(gs))
        x, pad, cm = x.to(device), pad.to(device), cm.to(device)
        gen = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + i0)
        calls = torch.zeros(len(gs), dtype=torch.long)
        for i in range(steps):
            still = (x == s["MASK"]) & cm
            if not still.any():
                break
            if mode == "ancestral":
                t, sn = 1 - i / steps, 1 - (i + 1) / steps
                reveal = still & (torch.rand(x.shape, generator=gen).to(device) < (t - sn) / t)
                if i == steps - 1:
                    reveal = still
                rows = reveal.any(1)
                if not rows.any():
                    continue                                   # ddpm_cache: nothing changes, no network call
            else:
                rows = still.any(1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(x[rows], pad[rows])
            logits = subs_logits(logits, s["MASK"]) + allowed
            conf, tok = torch.softmax(logits, -1).max(-1)
            calls[rows.cpu()] += 1
            if mode == "ancestral":
                r = reveal[rows]
                xr = x[rows]
                xr[r] = tok[r]
                x[rows] = xr
            else:
                st = still[rows]
                conf = conf.masked_fill(~st, -1.0)
                j = conf.argmax(1)
                xr = x[rows]
                ar = torch.arange(xr.shape[0], device=device)
                xr[ar, j] = tok[ar, j]
                x[rows] = xr
        for b, p in enumerate(pres):
            c = x[b, len(p):len(p) + width].tolist()
            k = c.index(s["END"]) if s["END"] in c else width
            stopped = s["END"] in c and all(v == s["END"] for v in c[k:])
            out.append((c[:k], stopped, int(calls[b])))
    return out


def evaluate(model, spec, device, out_dir: Path, tag: str, mode: str, limit: int = 0, batch: int = 256):
    sets = {"test": list(sgt.eval_graphs(spec, "test"))}
    for v in ("gen_relabeled", "gen_longer_path"):
        sets[v] = sgt.generalization_graphs(spec, v, n=1000)
    rows, summ = [], {}
    t0 = time.time()
    for name, gs in sets.items():
        gs = gs[:limit] if limit else gs
        rs = []
        for g, (toks, stopped, n_calls) in zip(gs, decode(model, spec, gs, device, mode=mode, batch=batch)):
            r = sgt.score(spec, g, toks, False, stopped)
            r.update(set=name, n_actions=n_calls, final=toks)
            rs.append(r)
        agg = sgt.aggregate(rs)
        agg["exact_ci"] = wilson(sum(r["exact"] for r in rs), len(rs))
        agg["mean_actions"] = sum(r["n_actions"] for r in rs) / max(1, len(rs))
        summ[name] = agg
        rows += rs
        print(mode, name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()
                                      if k != "exact_ci"}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"star_test_{tag}_var.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    desc = {"ancestral": f"MDLM ancestral sampler, T = {WIDTH}, argmax tokens, seeded reveals, ddpm_cache",
            "confidence": f"most-confident slot per step, {WIDTH} steps"}[mode]
    meta = dict(decoding=desc + "; node labels and END only; canvas width 40",
                eval_set_digest=sgt.eval_set_digest(spec, "test"), eval_seconds=time.time() - t0)
    (out_dir / f"star_test_{tag}_var.summary.json").write_text(json.dumps(dict(meta=meta, summary=summ), indent=1))
    return summ


def build(cfg, spec, device):
    m = cfg["model"]
    return MDLM(specials(spec)["vocab"], m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], m["dropout"]).to(device)


def train(args):
    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.steps:                      # smoke tests only; recorded in the manifest (argv)
        cfg["optim"]["steps"] = args.steps
    spec = sgt.spec_from_task_cfg(cfg["task"])
    run = Path(args.run_dir)
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    seed, data_seed = int(cfg["run"]["seed"]), int(cfg["run"]["data_seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = args.device
    o = cfg["optim"]
    model = build(cfg, spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "norm" not in n]
    nodecay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "norm" not in n)]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=o["weight_decay"]), dict(params=nodecay, weight_decay=0.0)],
                            lr=o["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / o["warmup"]))
    t0 = time.time()
    pool = [sgt.train_graph(spec, data_seed, j) for j in range(spec.n_train)]
    items = [(prefix_ids(spec, g), canvas_ids(spec, g)) for g in pool]
    pool_digest = hashlib.sha256("".join(g.digest() for g in pool).encode()).hexdigest()
    print(f"pool {len(pool)} graphs in {time.time() - t0:.0f}s digest {pool_digest[:16]}", flush=True)
    step, order, ptr = 0, rng.permutation(len(items)), 0
    ck = run / "ckpt" / "latest.pt"
    if ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, order, ptr = st["step"], np.array(st["order"]), st["ptr"]
        # map_location moves the saved ByteTensor RNG states to the device; set_rng_state needs them on the CPU.
        rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"resumed at step {step}", flush=True)
    manifest = dict(run=str(run), config=cfg, config_path=str(args.config), commit=git_commit(), params_total=n_params,
                    spec=spec.to_dict(), train_pool_digest=pool_digest, argv=sys.argv, torch=torch.__version__,
                    device=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu", canvas_width=WIDTH,
                    method="MDLM (arXiv 2406.07524) SUBS / log-linear; see src/reviser_v2/mdlm_star.py docstring")
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
        xt, pad, x0, masked, tt = (a.to(device) for a in collate(spec, [items[i] for i in idx], rng))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, per_tok = mdlm_loss(model, spec, xt, pad, x0, masked, tt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
        opt.step(); sched.step(); step += 1
        if step % cfg["log"]["every"] == 0:
            rec = dict(step=step, loss=loss.item(), nll_per_masked=per_tok.mean().item(), gnorm=float(gn),
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
    summ = {m: evaluate(model, spec, device, run / "eval", f"mdlm_{m}", m, limit=args.eval_limit)
            for m in ("ancestral", "confidence")}
    (run / "done.json").write_text(json.dumps(dict(step=step, seconds=time.time() - t0,
                                                   exact={m: {k: v["exact"] for k, v in sm.items()}
                                                          for m, sm in summ.items()})))


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
