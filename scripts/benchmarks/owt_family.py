"""Panel D of task 41: measured inference FLOPs, latency and sample quality of the released OpenWebText diffusion family
(AR, MDLM, SEDD, BD3-LM; Arriola et al. 2025, ``kuleshov-group/bd3lms``) with their official samplers.

All five checkpoints share one backbone (12 layers, d = 768, 12 heads, d_ff = 4d, RoPE, GPT-2 tokenizer, vocabulary
50,257 + [MASK] = 50,258, untied output head with bias, OpenWebText without EOS injection). This file

* re-implements the released backbone (``modeling_bd3lm.py`` at the pinned revisions in
  ``experiments/measured_flops/external_models.json``) as a plain ``nn.Module`` that loads the released ``model.safetensors``
  unchanged (strict key match), adds an exact query-trimmed KV cache, and is checked against the released module
  (``--validate``: logits of full and cached forwards against the released ``DITBackbone`` forward);
* ports the official sampling loops of ``bd3lms/diffusion.py`` (commit 1c3e8f4) line by line: ``_ar_sampler`` (KV
  cache, Gumbel-max over nucleus-filtered probabilities), ``_semi_ar_sampler`` + ``_ddpm_caching_update`` (MDLM: one
  block of the full length; BD3-LM: blocks of b with a KV cache and one extra "store" pass per finished block;
  ``first_hitting`` default on, i.e. one token revealed per network call), ``_analytic_sampler`` (SEDD, T steps + final
  denoiser), including ``_check_stop_conds`` rejection (unigram entropy of the last 256 tokens < 4 -> resample; the
  rejected attempts' FLOPs are charged to the sample);
* runs every sample twice with identical RNG state: once under ``FlopCounter`` (execution-based FLOP accounting) and
  once un-instrumented for latency (``torch.cuda.synchronize``), checking the token sequences agree;
* scores quality as the reference implementation does: GPT-2 Large generative perplexity after decode/re-tokenise
  (first token unscored) and the unigram entropy of each sample's token ids.

Execution variants that change FLOPs but not samples (same RNG stream, outputs checked identical):
  ``--impl official``: what the released code executes -- MDLM/SEDD/BD3-LM project every position of the network input
      to the vocabulary; the released BD3-LM KV-cache path recomputes attention for every cached query (it keeps q in
      the cache and slices the last block afterwards) and computes (discarded) logits in the store pass.
  ``--impl lean``: the same samples with queries trimmed to the new block, no logits in the store pass, and the head
      applied only to still-masked positions (MDLM/BD3-LM; the SEDD analytic update uses every position's score).

Usage (WSL): python scripts/benchmarks/owt_family.py --method ar --n 128 --samples 32 --out runs/measured_flops/panel_d/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
from flopcount import FlopCounter  # noqa: E402

HUB = ROOT / "data" / "hf_cache" / "hub"
PIN = json.loads((ROOT / "experiments" / "measured_flops" / "external_models.json").read_text(encoding="utf-8"))
V = 50258
MASK = 50257
BOS = 50256  # GPT-2 <|endoftext|>; the reference samplers set x[:, 0] = tokenizer.bos_token_id
NEG_INF = -1000000.0
EPS_NOISE = 1e-3  # LogLinearNoise(eps=1e-3)
SIGMA_MAX = -math.log1p(-(1 - EPS_NOISE))

REPOS = {"ar": "ar-noeos-owt", "mdlm": "mdlm-owt-noeos", "sedd": "sedd-noeos-owt",
         "bd3lm4": "bd3lm-owt-block_size4", "bd3lm16": "bd3lm-owt-block_size16"}


def snapshot(key: str) -> Path:
    repo = REPOS[key]
    d = HUB / f"models--kuleshov-group--{repo}" / "snapshots"
    snaps = sorted(d.iterdir())
    assert len(snaps) == 1, snaps
    assert snaps[0].name == PIN["owt_family"][repo]["revision"], (snaps[0].name, repo)
    return snaps[0]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------
# backbone re-implementation (parameter names identical to the released DITBackbone)
# ------------------------------------------------------------------------------------------------------------------
class _Lin(nn.Module):
    def __init__(self, i, o, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(o, i))
        self.bias = nn.Parameter(torch.empty(o)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class _Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.layer_norm(x.float(), [x.shape[-1]]) * self.weight[None, None, :]


class _Block(nn.Module):
    def __init__(self, d, adaln, cond=128):
        super().__init__()
        self.norm1, self.norm2 = _Norm(d), _Norm(d)
        self.attn_qkv = _Lin(d, 3 * d, bias=False)
        self.attn_out = _Lin(d, d, bias=False)
        self.mlp = nn.Sequential(_Lin(d, 4 * d), nn.GELU(approximate="tanh"), _Lin(4 * d, d))
        if adaln:
            self.adaLN_modulation = _Lin(cond, 6 * d)


class _Sigma(nn.Module):
    def __init__(self, cond=128, freq=256):
        super().__init__()
        self.mlp = nn.Sequential(_Lin(freq, cond), nn.SiLU(), _Lin(cond, cond))
        self.freq = freq

    def forward(self, t):
        half = self.freq // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class _Final(nn.Module):
    def __init__(self, d, adaln, cond=128):
        super().__init__()
        self.norm_final = _Norm(d)
        self.linear = _Lin(d, V)
        if adaln:
            self.adaLN_modulation = _Lin(cond, 2 * d)


class _Embed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(V, d))


class _Rotary(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh)))


def _rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class OWTNet(nn.Module):
    """forward(ids, sigma, pos0, cache, store, attn_mask, causal, head_rows, full_query, skip_head)

    cache: list over layers of dicts {k, v[, q]} with shape (B, H, S, 64) after RoPE; new tokens take positions
    pos0 .. pos0+T-1. With a cache the queries are the new tokens only (``full_query=False``) or every cached query
    followed by the new ones (``full_query=True``: the released BD3-LM cache path), of which the last T are kept.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d, self.h = cfg["hidden_dim"], cfg["n_heads"]
        self.d, self.dh = d, d // self.h
        self.adaln = cfg.get("adaln", True)
        self.causal = bool(cfg.get("causal", False))
        self.time_cond = bool(cfg.get("time_conditioning", False))
        self.vocab_embed = _Embed(d)
        if self.adaln:
            self.sigma_map = _Sigma(cfg["cond_dim"])
        self.rotary_emb = _Rotary(self.dh)
        self.blocks = nn.ModuleList(_Block(d, self.adaln, cfg["cond_dim"]) for _ in range(cfg["n_blocks"]))
        self.output_layer = _Final(d, self.adaln, cfg["cond_dim"])

    def _rope(self, pos):
        freqs = torch.einsum("i,j->ij", pos.float(), self.rotary_emb.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(self, ids, sigma=None, pos0=0, cache=None, store=False, attn_mask=None, causal=False,
                head_rows=None, full_query=False, skip_head=False):
        B, T = ids.shape
        x = self.vocab_embed.embedding[ids]
        c = None
        if self.adaln:
            s = sigma if (self.time_cond and sigma is not None) else torch.zeros(B, device=ids.device)
            c = F.silu(self.sigma_map(s))
        cos, sin = self._rope(torch.arange(pos0, pos0 + T, device=ids.device))
        new_cache = [] if store else None
        for i, blk in enumerate(self.blocks):
            if self.adaln:
                sh1, sc1, g1, sh2, sc2, g2 = blk.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            hx = blk.norm1(x)
            if self.adaln:
                hx = hx * (1 + sc1) + sh1
            qkv = blk.attn_qkv(hx).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            if cache is not None:
                kk = torch.cat((cache[i]["k"], k), dim=2)
                vv = torch.cat((cache[i]["v"], v), dim=2)
                qq = torch.cat((cache[i]["q"], q), dim=2) if full_query else q
            else:
                kk, vv, qq = k, v, q
            a = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=attn_mask,
                                               is_causal=bool(causal and cache is None and T > 1),
                                               scale=1 / math.sqrt(self.dh))
            if a.shape[2] != T:
                a = a[:, :, -T:]
            if store:
                new_cache.append(dict(k=kk, v=vv, q=torch.cat((cache[i]["q"], q), dim=2) if (full_query and cache)
                                      else (q if full_query else None)))
            a = blk.attn_out(a.transpose(1, 2).reshape(B, T, self.d))
            x = x + (g1 * a if self.adaln else a)
            hm = blk.norm2(x)
            if self.adaln:
                hm = hm * (1 + sc2) + sh2
            m = blk.mlp(hm)
            x = x + (g2 * m if self.adaln else m)
        if skip_head:
            return None, new_cache
        fl = self.output_layer
        hf = fl.norm_final(x)
        if self.adaln:
            shf, scf = fl.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            hf = hf * (1 + scf) + shf
        if head_rows is None:
            return fl.linear(hf), new_cache
        logits = torch.zeros(B, T, V, device=x.device, dtype=hf.dtype)
        logits[head_rows] = fl.linear(hf[head_rows])
        return logits, new_cache


def load_net(key: str, device: str) -> tuple:
    snap = snapshot(key)
    cfg = json.loads((snap / "config.json").read_text())
    import safetensors.torch as st
    sd = st.load_file(str(snap / "model.safetensors"))
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if "sigma_map.mlp.0.weight" not in sd:
        cfg["adaln"] = False
    else:
        cfg.setdefault("adaln", True)
    net = OWTNet(cfg)
    missing, unexpected = net.load_state_dict(sd, strict=True), None
    net.eval().to(device)
    ident = dict(repo="kuleshov-group/" + REPOS[key], revision=snap.name,
                 model_safetensors_sha256=sha256(snap / "model.safetensors"),
                 modeling_sha256=sha256(snap / "modeling_bd3lm.py"), config=cfg)
    return net, cfg, ident


# ------------------------------------------------------------------------------------------------------------------
# official sampling loops (bd3lms/diffusion.py @ 1c3e8f4), ported
# ------------------------------------------------------------------------------------------------------------------
def sample_categorical(p):
    gumbel_norm = (1e-10 - (torch.rand_like(p) + 1e-10).log())
    return (p / gumbel_norm).argmax(dim=-1)


def nucleus(p_x0, p, block):
    if p == 1.0:
        return p_x0
    p_ = p_x0[:, -block:].clone()
    sp, si = p_.sort(dim=-1, descending=True)
    cm = sp.cumsum(dim=-1)
    nm = cm <= p
    nm[..., 0] = 1
    sp = sp * nm
    p_.scatter_(-1, si, sp * nm)
    p_ /= p_.sum(-1, keepdim=True)
    p_x0[:, -block:] = p_
    return p_x0


def entropy_ok(x):
    _, counts = torch.unique(x[:, -256:], return_counts=True, sorted=False)
    ent = torch.special.entr(counts.float() / counts.sum()).sum()
    return bool(ent >= 4)


def sigma_from_p(p):
    return torch.clamp(-torch.log(1 - p), max=SIGMA_MAX)


class Run:
    """Holds a network and a FlopCounter (or None) and tags calls by stage."""

    def __init__(self, net, fc, impl):
        self.net, self.fc, self.impl = net, fc, impl
        self.calls = 0
        self.query_positions = 0

    def __call__(self, stage, *a, **kw):
        self.calls += 1
        if self.fc is None:
            return self.net(*a, **kw)
        with self.fc.stage(stage):
            return self.net(*a, **kw)


def subs_param(logits, xt):
    logits[:, :, MASK] += NEG_INF
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    um = xt != MASK
    logits[um] = NEG_INF
    logits[um, xt[um]] = 0
    return logits


def ar_sampler(run, n, p_nuc, dev):
    """_ar_sampler with sampling.kv_cache=true; one cached forward per generated token."""
    x = torch.zeros((1, n), dtype=torch.long, device=dev)
    x[:, 0] = BOS
    cache = None
    for i in range(n - 1):
        noise = torch.distributions.Gumbel(0, 1).sample((1, V)).to(dev)
        logits, cache = run("decode", x[:, i:i + 1], pos0=i, cache=cache, store=True)
        logits[:, :, MASK] = NEG_INF
        nl = logits.log_softmax(-1)[:, -1:].to(torch.float64).exp()
        nl = nucleus(nl, p_nuc, 1).log()
        y = (nl[:, -1] + noise).argmax(-1)
        if (i + 1) > 256 and not entropy_ok(x[:, :i + 1]):
            return None
        x[:, i + 1] = y
    return x


def ddpm_caching_update(run, x_win, t, dt, p_x0, cfg, pos0, cache, dev):
    """_ddpm_caching_update; x_win is the current block (BD3-LM: last block, MDLM: whole sequence)."""
    block = cfg["block"]
    mc_t, mc_s = t, t - dt
    sig_t = sigma_from_p(mc_t)
    mask_prob = (mc_s / mc_t)[:, None]
    lean = run.impl == "lean"
    if p_x0 is None:
        rows = (x_win == MASK) if (lean and cfg["lean_head"]) else None
        logits, _ = run("denoise", x_win, sigma=sig_t.squeeze(-1), pos0=pos0, cache=cache,
                        attn_mask=cfg.get("attn_mask"), head_rows=rows,
                        full_query=(cache is not None and not lean))
        p_x0 = subs_param(logits, x_win).to(torch.float64)[:, -block:].exp()
        p_x0 = nucleus(p_x0, cfg["nucleus"], block)
    if cfg["first_hitting"]:
        x_block = sample_categorical(p_x0)
        num_masked = (x_win[:, -block:] == MASK).sum(-1)
        ind = torch.randint(0, int(num_masked.item()), (x_block.shape[0],))
        ind = (x_win[:, -block:] == MASK).nonzero()[ind, 1]
        m = (torch.arange(block, device=dev) == ind[:, None]).to(x_block.dtype)
        x_block = x_block * m + x_win[:, -block:] * (1 - m)
    else:
        q_xs = p_x0 * (1 - mask_prob)
        q_xs[:, :, MASK] = mask_prob.squeeze(-1)
        x_block = sample_categorical(q_xs)
    copy_flag = (x_win[:, -block:] != MASK).to(x_win.dtype)
    x_block = copy_flag * x_win[:, -block:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x_win[:, :-block], x_block), dim=-1)
    new_cache = cache
    if cfg["kv_cache"] and MASK not in x_block:
        _, new_cache = run("store", x_block, sigma=sig_t.squeeze(-1), pos0=pos0, cache=cache, store=True,
                           skip_head=lean, full_query=not lean)
    if not torch.equal(x_new, x_win):
        return None, x_new, new_cache
    return p_x0, x_new, new_cache


def semi_ar_sampler(run, n, cfg, dev):
    """_semi_ar_sampler: MDLM uses one block of length n (block_size = model length); BD3-LM blocks of b with KV cache."""
    b = cfg["block"]
    T = cfg["T"]
    cache = None
    x_accum = None
    for stride in range(n // b):
        blk = torch.full((1, b), MASK, dtype=torch.long, device=dev)
        if stride == 0:
            blk[:, 0] = BOS
            x_accum = blk
        else:
            x_accum = torch.cat((x_accum, blk), dim=1)
        pos0 = stride * b if cfg["kv_cache"] else 0
        dt = 1 / T
        p_cache = None
        timesteps = torch.linspace(1, 0, T, device=dev)
        t = 1.0
        for i in range(T):
            win = x_accum[:, -b:] if cfg["kv_cache"] else x_accum
            if MASK not in win:
                break
            if cfg["first_hitting"]:
                u = np.random.rand()
                num_masked = int((win == MASK).sum(-1).item())
                t *= u ** (1 / num_masked)
                tt = torch.full((1, 1), t, device=dev)
            else:
                tt = timesteps[i] * torch.ones((1, 1), device=dev)
            if cfg.get("no_cache_p"):
                p_cache = None
            p_cache, x_next, cache = ddpm_caching_update(run, win, tt, dt, p_cache, cfg, pos0,
                                                         cache if cfg["kv_cache"] else None, dev)
            if cfg["kv_cache"]:
                x_accum = torch.cat((x_accum[:, :-b], x_next), dim=1)
            else:
                x_accum = x_next
        if x_accum.shape[1] > 256 and not entropy_ok(x_accum):
            return None
    return x_accum


def analytic_sampler(run, n, cfg, dev, eps=1e-5):
    """_analytic_sampler (SEDD): num_steps score passes + one denoiser pass, every pass over the full sequence."""
    T = cfg["T"]
    x = torch.full((1, n), MASK, dtype=torch.long, device=dev)
    x[:, 0] = BOS
    timesteps = torch.linspace(1, eps, T + 1, device=dev)
    dt = (1 - eps) / T

    def get_score(x, sigma):
        logits, _ = run("denoise", x, sigma=sigma.reshape(-1))
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(logits.dtype)
        logits = logits - esigm1_log[:, None, None].reshape(-1, 1, 1) - np.log(logits.shape[-1] - 1)
        logits = torch.scatter(logits, -1, x[..., None], torch.zeros_like(logits[..., :1]))
        mo = logits.to(torch.float64)
        if cfg["nucleus"] == 1.0:
            return mo.exp()
        mo = mo - mo.logsumexp(-1, keepdim=True)
        return nucleus(mo.exp(), cfg["nucleus"], n)

    def stag(score, dsigma):
        score = score.clone()
        extra = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., MASK] += extra
        return score

    def transp(i, sigma):
        sigma = sigma.view(*sigma.shape, *((1,) * (i[..., None].dim() - sigma.dim())))
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=V)
        edge += torch.where(i == MASK, 1 - torch.exp(-sigma).squeeze(-1), 0)[..., None]
        return edge

    for i in range(T):
        t = timesteps[i] * torch.ones(1, 1, device=dev)
        s_t, s_s = sigma_from_p(t), sigma_from_p(t - dt)
        ds = s_t - s_s
        score = get_score(x, s_t)
        x = sample_categorical(stag(score, ds) * transp(x, ds))
    t = timesteps[-1] * torch.ones(1, 1, device=dev)
    sg = sigma_from_p(t)
    score = get_score(x, sg)
    probs = stag(score, sg) * transp(x, sg)
    probs[..., MASK] = 0
    x = sample_categorical(probs)
    if not entropy_ok(x):
        return None
    return x


METHODS = {
    # name: (checkpoint, sampler, config); nucleus values are the official gen_ppl scripts' settings
    "ar": ("ar", "ar", dict(nucleus=0.9)),
    "mdlm_fh": ("mdlm", "semi_ar", dict(nucleus=0.9, first_hitting=True, T=5000, kv_cache=False, lean_head=True)),
    "mdlm_ddpm_cache_T1000": ("mdlm", "semi_ar", dict(nucleus=0.9, first_hitting=False, T=1000, kv_cache=False,
                                                      lean_head=True)),
    "mdlm_ddpm_nocache_T1000": ("mdlm", "semi_ar", dict(nucleus=0.9, first_hitting=False, T=1000, kv_cache=False,
                                                        lean_head=True, no_cache_p=True)),
    "sedd_T32": ("sedd", "analytic", dict(nucleus=0.99, T=32)),
    "sedd_T128": ("sedd", "analytic", dict(nucleus=0.99, T=128)),
    "sedd_T1000": ("sedd", "analytic", dict(nucleus=0.99, T=1000)),
    "sedd_T2048": ("sedd", "analytic", dict(nucleus=0.99, T=2048)),
    "bd3lm_b4": ("bd3lm4", "semi_ar", dict(nucleus=0.9, first_hitting=True, T=5000, kv_cache=True, block=4,
                                           lean_head=True)),
    "bd3lm_b16": ("bd3lm16", "semi_ar", dict(nucleus=0.9, first_hitting=True, T=5000, kv_cache=True, block=16,
                                             lean_head=True)),
}


def generate(run, method, n, dev, max_tries=10):
    ck, kind, cfg = METHODS[method]
    cfg = dict(cfg)
    tries = 0
    while True:
        tries += 1
        if kind == "ar":
            x = ar_sampler(run, n, cfg["nucleus"], dev)
        elif kind == "analytic":
            x = analytic_sampler(run, n, cfg, dev)
        else:
            cfg.setdefault("block", n)
            if cfg["block"] == n:
                cfg["kv_cache"] = False
            if not cfg["kv_cache"] and cfg["block"] < n:
                raise ValueError("block < n needs the KV cache")
            x = semi_ar_sampler(run, n, cfg, dev)
        if x is not None:
            return x, tries
        if tries > max_tries:
            raise RuntimeError("Sampling failed.")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def gen_ppl(texts, device, max_length):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = sorted((HUB / "models--gpt2-large" / "snapshots").iterdir())[0]
    tok = AutoTokenizer.from_pretrained(str(snap))
    lm = AutoModelForCausalLM.from_pretrained(str(snap)).eval().to(device)
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)
            logits = lm(ids).logits[:, :-1].float()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
            out.append(dict(nll_sum=float(nll), n_tok=int(ids.shape[1] - 1)))
    del lm
    return out, dict(scorer="gpt2-large", revision=snap.name, model_sha256=sha256(snap / "model.safetensors"))


def boot_ci(vals, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return [float(v.mean()), float(v.mean())] if len(v) else [None, None]
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(iters)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--impl", default="official", choices=["official", "lean"])
    ap.add_argument("--count-samples", type=int, default=None, help="count only the first k samples (default all)")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = METHODS[a.method][0]
    net, cfg, ident = load_net(ck, dev)
    kcount = a.samples if a.count_samples is None else a.count_samples
    rows = []
    t_start = time.time()
    with torch.no_grad():
        if not a.no_time:  # warm-up (discarded)
            seed_all(10 ** 6 + a.seed)
            generate(Run(net, None, a.impl), a.method, a.n, dev)
        for i in range(a.samples):
            s = a.seed * 1000 + i
            rec = dict(sample=i, seed=s)
            xc = None
            if i < kcount:
                fc = FlopCounter(vocab_sizes=[V])
                run = Run(net, fc, a.impl)
                seed_all(s)
                with fc:
                    xc, tries = generate(run, a.method, a.n, dev)
                su = fc.summary()
                tot = su["total"]
                rec.update(counted=True, tries_counted=tries, net_calls=run.calls,
                           flops_matmul=tot["matmul"], flops_total=tot["total"], flops_linear=tot["linear"],
                           flops_head=tot["head"], flops_attn=tot["attn"], flops_attn_dense=tot["attn_dense"],
                           flops_elementwise=tot["elementwise"], head_positions=tot["head_positions"],
                           attn_pairs=tot["attn_pairs"],
                           per_stage={k: dict(matmul=v["matmul"], head=v["head"], attn=v["attn"],
                                              head_positions=v["head_positions"]) for k, v in su["per_stage"].items()},
                           calls_per_stage={k: sum(c for n_, c in v.items() if n_.startswith("head:")) for k, v in
                                            su["calls"].items()},
                           uncovered=su["uncovered"])
            if not a.no_time:
                run = Run(net, None, a.impl)
                seed_all(s)
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                xt, tries_t = generate(run, a.method, a.n, dev)
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                rec.update(latency_s=time.perf_counter() - t0, tries_timed=tries_t, net_calls_timed=run.calls,
                           peak_mem_bytes=torch.cuda.max_memory_allocated() if dev.startswith("cuda") else None)
                if xc is not None:
                    rec["same_tokens_counted_vs_timed"] = bool(torch.equal(xc, xt))
                x = xt
            else:
                x = xc
            ids = x[0].tolist()
            rec["mask_left"] = int(sum(1 for t in ids if t == MASK))
            rec["tokens"] = ids
            rec["unigram_entropy"] = float(torch.special.entr(torch.unique(x, return_counts=True)[1].float()
                                                              / x.numel()).sum())
            rows.append(rec)
            print(f"  {a.method} n={a.n} {i + 1}/{a.samples} "
                  f"{rec.get('flops_matmul', 0) / 1e9:.1f} GFLOP {rec.get('latency_s', 0):.2f}s "
                  f"tries={rec.get('tries_counted', rec.get('tries_timed'))}", flush=True)
    del net
    torch.cuda.empty_cache() if dev.startswith("cuda") else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(sorted((HUB / "models--gpt2" / "snapshots").iterdir())[0]))
    texts = [tok.decode([t for t in r["tokens"] if t != MASK]) for r in rows]
    q, scorer = gen_ppl(texts, dev, max_length=a.n)
    for r, qq in zip(rows, q):
        r.update(qq)
    cnt = [r for r in rows if r.get("counted")]
    agg = dict(n_samples=len(rows), n_counted=len(cnt),
               gen_ppl=float(math.exp(sum(r["nll_sum"] for r in rows) / sum(r["n_tok"] for r in rows))),
               unigram_entropy=float(np.mean([r["unigram_entropy"] for r in rows])),
               unigram_entropy_ci95=boot_ci([r["unigram_entropy"] for r in rows]),
               mask_left_total=sum(r["mask_left"] for r in rows))
    # bootstrap CI of the gen-ppl over samples (token-weighted)
    rng = np.random.default_rng(0)
    nl = np.array([r["nll_sum"] for r in rows]); nt = np.array([r["n_tok"] for r in rows])
    bs = [math.exp(nl[ix].sum() / nt[ix].sum()) for ix in (rng.integers(0, len(rows), len(rows)) for _ in range(2000))]
    agg["gen_ppl_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    if cnt:
        for k in ("flops_matmul", "flops_total", "flops_linear", "flops_head", "flops_attn", "flops_attn_dense",
                  "head_positions", "net_calls", "tries_counted"):
            vals = [r[k] for r in cnt]
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_min"] = float(np.min(vals))
            agg[k + "_max"] = float(np.max(vals))
        agg["flops_matmul_ci95"] = boot_ci([r["flops_matmul"] for r in cnt])
        agg["uncovered_ops"] = sorted({k for r in cnt for k in r["uncovered"]})
    if not a.no_time:
        lat = [r["latency_s"] for r in rows]
        agg.update(latency_mean=float(np.mean(lat)), latency_p50=float(np.percentile(lat, 50)),
                   latency_p95=float(np.percentile(lat, 95)), latency_ci95=boot_ci(lat),
                   peak_mem_bytes=max(r["peak_mem_bytes"] or 0 for r in rows),
                   same_tokens_counted_vs_timed=sum(1 for r in cnt if r.get("same_tokens_counted_vs_timed")))
    ck_, kind, mcfg = METHODS[a.method]
    sys.path.insert(0, str(ROOT / "src"))
    from reviser_v2 import bench_cost as bc
    out = dict(task=41, panel="D", method=a.method, sampler=kind, sampler_config=mcfg, impl=a.impl, n=a.n,
               samples=a.samples, seed=a.seed, checkpoint=ident, scorer=scorer,
               reference_code="kuleshov-group/bd3lms @ 1c3e8f4 (diffusion.py samplers, ported)",
               harness_sha256=sha256(Path(__file__)), flopcount_sha256=sha256(ROOT / "scripts/benchmarks/flopcount.py"),
               precision="fp32 (TF32 off), batch 1", environment=bc.environment(dev, "fp32"),
               wall_seconds=time.time() - t_start, aggregate=agg,
               command=" ".join(sys.argv))
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(str(op) + ".rows.jsonl", "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(str(op) + ".json", "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: agg[k] for k in agg if not isinstance(agg[k], list) or k.endswith("ci95")}))


if __name__ == "__main__":
    main()
