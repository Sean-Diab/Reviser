"""Panel S (task 41): structural probe of every inference-FLOP row of the original manuscript.

What this is: untrained reference networks at the manuscript's dimensions execute each row's decoding schedule as a
*forced trace* (lengths, pass counts and head positions are fixed by the schedule, not by model outputs) under
``FlopCounter``. It audits the *assumptions* of each closed-form row (prompt/prefill, which positions get a vocabulary
head, early exits, encoder and cross-attention, padding, dense vs triangular attention). It produces no quality numbers
and says nothing about how many iterations a trained model needs; trained measurements are Panels E, C, D and L.

Each row has up to two variants:
  * ``paper``: the schedule exactly as the manuscript states it (integer lengths; the manuscript's fractional surrogate
    lengths such as n0 = 102.4 are rounded, which the closed-form check below applies identically). The measured
    dominant-matmul MACs must equal the manuscript formula evaluated at the same integer lengths (``closed_form_check``);
    this validates that the probe executes the stated schedule.
  * ``faithful``: the operator structure of the released decoder, read from the official code
    (``notes/sources/nar-decoder-structure.md``): e.g. LevT's full-vocabulary head on every canvas position, Mask-Predict's
    length beam of 5, Diffusion-LM's L = 64 canvas with per-step rounding against a V x k embedding table, Dream/LLaDA's
    full logits at every step, LLaDA's semi-AR trunk over the whole canvas.

Execution: by default on the ``meta`` device (shapes only, no arithmetic, no memory), so native 7B/8B/32B architectures
run in seconds. Attention calls the fused CPU flash kernel op directly (``_scaled_dot_product_flash_attention_for_cpu``)
so causal calls are recognised as causal on both devices. ``--device cpu`` executes the same code with real (random)
tensors; ``--crosscheck`` runs a subset on both devices and asserts identical counts.

Units: FLOPs (1 MAC = 2 FLOPs) as counted; ``macs_matmul`` = dominant-matmul FLOPs / 2, the manuscript's unit.
Output: ``runs/measured_flops/panel_s/probe.json`` (+ ``.md`` table).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
from flopcount import FlopCounter  # noqa: E402

aten = torch.ops.aten
G = 1e9

# ------------------------------------------------------------------------------------------------ architectures
REF = dict(d=768, L=12, h=12, hkv=12, dff=3072, V=50000, gated=False)          # manuscript 100M-tier reference
QWEN7B = dict(d=3584, L=28, h=28, hkv=4, dff=18944, V=152064, gated=True)       # Qwen2.5-7B / Dream-7B (config.json)
QWEN32B = dict(d=5120, L=64, h=40, hkv=8, dff=27648, V=152064, gated=True)      # Qwen2.5-32B / SDLM-32B
LLAMA3_8B = dict(d=4096, L=32, h=32, hkv=8, dff=14336, V=128256, gated=True)    # published config values only (gated)
LLADA8B = dict(d=4096, L=32, h=32, hkv=32, dff=12288, V=126464, gated=True)     # LLaDA-8B-Base config.json
N = 128
KMAX = 16
V_ACT = REF["V"] + 2 * KMAX + 1  # Reviser action vocabulary |V_c| + |M| + 1 (moves +-1..K_max, stop)


def attn(q, k, v, causal=False, mask=None):
    """q (B,h,T,dh), k/v (B,hkv,S,dh). GQA by repeating K/V heads (a copy, no arithmetic)."""
    if k.shape[1] != q.shape[1]:
        r = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    if mask is not None:
        return aten._scaled_dot_product_flash_attention_for_cpu(q, k, v, 0.0, False, attn_mask=mask)[0]
    return aten._scaled_dot_product_flash_attention_for_cpu(q, k, v, 0.0, causal)[0]


class Layer(nn.Module):
    def __init__(s, a, xattn=False, posattn=False):
        super().__init__()
        d, dkv = a["d"], a["d"] // a["h"] * a["hkv"]
        s.h, s.hkv, s.dh = a["h"], a["hkv"], a["d"] // a["h"]
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.q, s.k, s.v, s.o = nn.Linear(d, d, False), nn.Linear(d, dkv, False), nn.Linear(d, dkv, False), nn.Linear(d, d, False)
        s.gated = a["gated"]
        if s.gated:
            s.gate, s.up, s.down = nn.Linear(d, a["dff"], False), nn.Linear(d, a["dff"], False), nn.Linear(a["dff"], d, False)
        else:
            s.up, s.down = nn.Linear(d, a["dff"], False), nn.Linear(a["dff"], d, False)
        s.xattn = xattn
        if xattn:
            s.nx = nn.LayerNorm(d)
            s.xq, s.xk, s.xv, s.xo = (nn.Linear(d, d, False) for _ in range(4))
        s.posattn = posattn
        if posattn:
            s.np_ = nn.LayerNorm(d)
            s.pq, s.pk, s.pv, s.po = (nn.Linear(d, d, False) for _ in range(4))

    def heads(s, x, n):
        return x.view(x.shape[0], x.shape[1], n, s.dh).transpose(1, 2)

    def merge(s, x):
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

    def forward(s, x, causal=False, mask=None, cache=None, store=False, enc=None, pos=None, xkv=None):
        hN = s.n1(x)
        q, k, v = s.heads(s.q(hN), s.h), s.heads(s.k(hN), s.hkv), s.heads(s.v(hN), s.hkv)
        if cache is not None and "k" in cache:
            k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
        if store:
            cache["k"], cache["v"] = k, v
        x = x + s.o(s.merge(attn(q, k, v, causal=causal, mask=mask)))
        if s.posattn:  # Gu et al. positional attention: queries/keys from positions, values from states
            hp = s.np_(x)
            x = x + s.po(s.merge(attn(s.heads(s.pq(pos), s.h), s.heads(s.pk(pos), s.h), s.heads(s.pv(hp), s.h))))
        if s.xattn:
            hx = s.nx(x)
            if xkv is not None and "xk" in xkv:  # AR MT decoding caches the source K/V once
                ek, ev = xkv["xk"], xkv["xv"]
            else:
                ek, ev = s.heads(s.xk(enc), s.h), s.heads(s.xv(enc), s.h)
                if xkv is not None:
                    xkv["xk"], xkv["xv"] = ek, ev
            x = x + s.xo(s.merge(attn(s.heads(s.xq(hx), s.h), ek, ev)))
        h2 = s.n2(x)
        m = F.silu(s.gate(h2)) * s.up(h2) if s.gated else F.gelu(s.up(h2))
        return x + s.down(m)


class Stack(nn.Module):
    def __init__(s, a, xattn=False, posattn=False, head=True, L=None):
        super().__init__()
        s.a = a
        s.emb = nn.Embedding(a["V"], a["d"])
        s.layers = nn.ModuleList(Layer(a, xattn, posattn) for _ in range(L or a["L"]))
        s.nf = nn.LayerNorm(a["d"])
        s.head = nn.Linear(a["d"], a["V"], False) if head else None

    def forward(s, ids, causal=False, mask=None, cache=None, store=False, enc=None, pos=None, xkv=None, depth=None):
        x = s.emb(ids)
        for i, lay in enumerate(s.layers[:depth]):
            x = lay(x, causal, mask, None if cache is None else cache[i], store, enc, pos,
                    None if xkv is None else xkv[i])
        return s.nf(x)

    def logits(s, h, rows=None):
        """rows: None = every position; int k = the last k positions; slice/list = those positions."""
        if rows is not None:
            h = h[:, -rows:] if isinstance(rows, int) else h[:, rows]
        return s.head(h)


def ids(dev, b, t):
    return torch.zeros(b, t, dtype=torch.long, device=dev)


def make(a, dev, **kw):
    with torch.device(dev):
        m = Stack(a, **kw)
    return m.eval()


# ------------------------------------------------------------------------------------------------ schedules
# Each schedule: fn(dev, fc) -> dict(schedule description). Runs under fc; uses fc.stage for pass types.

def ar_decode(m, dev, fc, n, prompt=0, head_prompt="last", name="decode"):
    """KV-cached AR: optional prefill of `prompt` tokens (head on the last or on every prompt position), then
    single-token steps until n new tokens exist. With prompt = 0 the first step is the BOS token."""
    cache = [dict() for _ in m.layers]
    steps = n
    if prompt:
        with fc.stage("prefill"):
            h = m(ids(dev, 1, prompt), causal=True, cache=cache, store=True)
            m.logits(h, None if head_prompt == "all" else 1)
        steps = n - 1  # the prefill produced the first new token
    with fc.stage(name):
        for _ in range(steps):
            h = m(ids(dev, 1, 1), causal=True, cache=cache, store=True)
            m.logits(h)
    return dict(prompt=prompt, calls=steps + (1 if prompt else 0))


def full_pass(m, dev, fc, stage, t, head_rows="all", b=1, enc=None, depth=None, mask=None):
    with fc.stage(stage):
        h = m(ids(dev, b, t), enc=enc, depth=depth, mask=mask)
        if head_rows == "all":
            m.logits(h)
        elif head_rows:
            m.logits(h, slice(0, head_rows))
    return h


def encoder(dev, fc, a, s_len, b=1):
    e = make(a, dev, head=False)
    with fc.stage("encoder"):
        return e(ids(dev, b, s_len))


# ---- rows ------------------------------------------------------------------------------------------------------
def S_ar(dev, fc):
    return ar_decode(make(REF, dev), dev, fc, N)


def S_ar_n(n):
    def f(dev, fc):
        return ar_decode(make(REF, dev), dev, fc, n)
    return f


def S_ar_mt(dev, fc):
    """MT AR denominator for encoder-decoder rows: encoder once on n_src = 128, KV-cached causal decoder with
    cross-attention (source K/V computed once and cached, as fairseq's incremental decoding does), head per step."""
    enc = encoder(dev, fc, REF, N)
    m = make(REF, dev, xattn=True)
    cache, xkv = [dict() for _ in m.layers], [dict() for _ in m.layers]
    with fc.stage("decode"):
        for _ in range(N):
            h = m(ids(dev, 1, 1), causal=True, cache=cache, store=True, enc=enc, xkv=xkv)
            m.logits(h)
    return dict(n_src=N, source_kv_cached=True)


def S_reviser(p):
    def f(dev, fc):
        a = dict(REF, V=V_ACT)
        n_eff = round(N / (1 - p))
        ar_decode(make(a, dev), dev, fc, n_eff)
        return dict(p_move=p, n_eff=n_eff, n_eff_exact=N / (1 - p), head_width=V_ACT)
    return f


def levt_paper_iter(m, dev, fc, depth_dp):
    n0, n1, mm = round(0.8 * N), N, round(0.4 * N)
    d = m.a["d"]
    with torch.device(dev):
        hdel, hplh = nn.Linear(d, 2, False), nn.Linear(2 * d, KMAX + 1, False)
    with fc.stage("del"):
        h = m(ids(dev, 1, n0), depth=depth_dp)
        hdel(h)
    with fc.stage("plh"):
        h = m(ids(dev, 1, n0), depth=depth_dp)
        hplh(torch.cat([h, h[:, :1]], 1).repeat(1, 1, 2))  # n0 + 1 slots x 2d, as the manuscript's head term
    full_pass(m, dev, fc, "tok", n1, head_rows=mm)


def S_levt_paper(R, f_ee):
    def f(dev, fc):
        m = make(REF, dev)
        depth = int(REF["L"] * f_ee)
        for _ in range(R):
            levt_paper_iter(m, dev, fc, depth)
        return dict(R=R, f_del=f_ee, f_plh=f_ee, n0=round(0.8 * N), m=round(0.4 * N), encoder=False)
    return f


def S_levt_faithful(R):
    """fairseq LevT (levenshtein_transformer.py @3d262bb): encoder once; step 0 = placeholder pass on <s></s> (len 2)
    + token pass on the new canvas (n + 2); iterations 1..R-2 = deletion + placeholder + token passes on the full canvas
    (placeholder pass at the padded pre-deletion width); last iteration = the adaptive "no change" check (deletion +
    placeholder, no token pass). Heads: d->2 at every position, 2d->256 on the T-1 adjacent slots, d->|V| on *every*
    canvas position of the token pass. Cross-attention K/V recomputed each pass. Reference widths (d768, 12+12 layers)."""
    def f(dev, fc):
        enc = encoder(dev, fc, REF, N)
        m = make(REF, dev, xattn=True)
        d = REF["d"]
        with torch.device(dev):
            hdel, hplh = nn.Linear(d, 2, False), nn.Linear(2 * d, 256, False)
        T = N + 2

        def plh(t):
            with fc.stage("plh"):
                h = m(ids(dev, 1, t), enc=enc)
                hplh(torch.cat([h[:, :-1], h[:, 1:]], -1))

        def dele(t):
            with fc.stage("del"):
                hdel(m(ids(dev, 1, t), enc=enc))

        plh(2)
        full_pass(m, dev, fc, "tok", T, "all", enc=enc)
        for _ in range(R - 2):
            dele(T); plh(T)
            full_pass(m, dev, fc, "tok", T, "all", enc=enc)
        dele(T); plh(T)
        return dict(R=R, passes=2 + 3 * (R - 2) + 2, canvas=T, n_src=N, encoder=True,
                    note="step 0 from <s></s>; last iteration is the adaptive no-change check")
    return f


def S_inst_paper(serial):
    def f(dev, fc):
        m = make(REF, dev)
        ts = list(range(1, N + 1)) if serial else [2 ** i for i in range(8)]
        for t in ts:
            with fc.stage("pass"):
                h = m(ids(dev, 1, t))
                m.logits(torch.cat([h, h[:, :1]], 1))  # n_t + 1 slots
        return dict(n_t=ts if not serial else "1..128", encoder=False)
    return f


def S_inst_faithful(dev, fc):
    """fairseq InsT: encoder once; canvas <s> content </s>; one pass per iteration over the canvas; slot features
    pool_out(cat(h[:-1], h[1:])) 2d->d then d->|V| on every slot; every slot inserts in parallel, so content grows
    0,1,3,7,...,127 then 128 (capped), then a confirming pass where every slot predicts <pad>."""
    enc = encoder(dev, fc, REF, N)
    m = make(REF, dev, xattn=True)
    with torch.device(dev):
        pool = nn.Linear(2 * REF["d"], REF["d"], True)
    content, trace = 0, []
    while True:
        t = content + 2
        trace.append(t)
        with fc.stage("pass"):
            h = m(ids(dev, 1, t), enc=enc)
            m.logits(pool(torch.cat([h[:, :-1], h[:, 1:]], -1)))
        if content >= N:
            break
        content = min(N, 2 * content + 1)
    return dict(canvas_lengths=trace, encoder=True)


def S_nat_paper(dev, fc):
    enc = encoder(dev, fc, REF, N)
    with torch.device(dev):
        fert = nn.Linear(REF["d"], 50, False)
    with fc.stage("fertility"):
        fert(enc)
    full_pass(make(REF, dev, xattn=True), dev, fc, "decoder", N, "all", enc=enc)
    return dict(n_src=N, K_fert=50)


def S_nat_faithful(npd, teacher):
    """salesforce/nonauto-nmt (Gu et al. 2018): encoder once; fertility head d->50 (bias) on every source position;
    decoder with positional attention and cross-attention on B*N rows (noisy parallel decoding, N fertility samples);
    head on every target position. With teacher rescoring: an AR teacher encoder once + one teacher-forced causal
    decoder pass over all N candidates with a full-vocabulary softmax at every position. Reference widths."""
    def f(dev, fc):
        enc = encoder(dev, fc, REF, N)
        with torch.device(dev):
            fert = nn.Linear(REF["d"], 50, True)
            pos = torch.zeros(npd, N, REF["d"])
        with fc.stage("fertility"):
            fert(enc)
        m = make(REF, dev, xattn=True, posattn=True)
        encN = enc.expand(npd, -1, -1)
        with fc.stage("decoder"):
            m.logits(m(ids(dev, npd, N), enc=encN, pos=pos))
        if teacher:
            tenc = encoder(dev, fc, REF, N)
            t = make(REF, dev, xattn=True)
            with fc.stage("teacher"):
                t.logits(t(ids(dev, npd, N), causal=True, enc=tenc.expand(npd, -1, -1)))
        return dict(npd_samples=npd, teacher_rescoring=teacher, positional_attention=True)
    return f


MP_T = 10


def mp_masks(n=N, T=MP_T):
    return [n] + [math.floor(n * (T - t) / T) for t in range(1, T)]


def S_cmlm_paper(dev, fc):
    enc = encoder(dev, fc, REF, N)
    m = make(REF, dev, xattn=True)
    ms = mp_masks()
    for mt in ms:
        full_pass(m, dev, fc, "decoder", N, mt, enc=enc)
    return dict(T=MP_T, masked=ms, sum_masked=sum(ms), length_candidates=1)


def S_cmlm_fairseq(dev, fc):
    """fairseq cmlm_transformer: canvas <s> + L^ + </s> = 130; T = max_iter + 1 = 10 passes; head at every position;
    length head d->256 on the mean-pooled encoder output once; length beam 1 (default)."""
    enc = encoder(dev, fc, REF, N)
    with torch.device(dev):
        lh = nn.Linear(REF["d"], 256, False)
    with fc.stage("length"):
        lh(enc.mean(1))
    m = make(REF, dev, xattn=True)
    for _ in range(MP_T):
        full_pass(m, dev, fc, "decoder", N + 2, "all", enc=enc)
    return dict(T=MP_T, canvas=N + 2, length_beam=1)


def S_maskpredict_official(dev, fc):
    """Mask-Predict official (generate_cmlm.py @11c7535): encoder on S + 1 (LENGTH token), length head d->10000 at
    that position; top-5 length candidates (here 126..130) batched as 5 rows padded to 130 (key padding masked);
    exactly T = 10 passes; head and softmax at every position of every row; best candidate by mean log-prob."""
    enc = encoder(dev, fc, REF, N + 1)
    with torch.device(dev):
        lh = nn.Linear(REF["d"], 10000, False)
        lens = [128, 129, 127, 130, 126]
        W = max(lens)
        mask = torch.zeros(len(lens), 1, W, W, device=dev)
    with fc.stage("length"):
        lh(enc[:, 0])
    m = make(REF, dev, xattn=True)
    encB = enc.expand(len(lens), -1, -1)
    for _ in range(MP_T):
        full_pass(m, dev, fc, "decoder", W, "all", b=len(lens), enc=encB, mask=mask)
    return dict(T=MP_T, length_beam=5, candidate_lengths=lens, padded_to=W, source_len=N + 1)


def S_diffusion_paper(T, head=True):
    def f(dev, fc):
        m = make(REF, dev)
        for _ in range(T):
            full_pass(m, dev, fc, "denoise", N, "all" if head else None)
        return dict(T=T, full_logits_every_step=head)
    return f


def S_difflm_paper(T, gamma):
    def f(dev, fc):
        m = make(REF, dev)
        n_round = 1 + round(gamma * (T - 1))
        for i in range(T):
            h = full_pass(m, dev, fc, "denoise", N, None)
            if i >= T - n_round:
                with fc.stage("round"):
                    m.logits(h)
        return dict(T=T, gamma=gamma, roundings=n_round)
    return f


def S_difflm_official(T, Lc, k, V):
    """Diffusion-LM (improved-diffusion @759889d, TransformerNetModel2): per step a timestep MLP (128->512->768), input
    up-projection k->768->768 (tanh), BERT-base trunk (12 x 768) over the L = image_size^2 canvas, down-projection
    768->768->k, and (clamp='clamp', every step) nearest-neighbour rounding of x0-hat against the V x k embedding
    table (||E||^2 + ||x||^2 - 2 E x^T, top-1); after the loop one tied lm_head k->V on the final sample."""
    def f(dev, fc):
        a = dict(REF, V=V)
        m = make(a, dev, head=False)
        with torch.device(dev):
            t1, t2 = nn.Linear(128, 512), nn.Linear(512, 768)
            u1, u2 = nn.Linear(k, 768), nn.Linear(768, 768)
            d1, d2 = nn.Linear(768, 768), nn.Linear(768, k)
            E = torch.zeros(V, k)
            x = torch.zeros(1, Lc, k)
            temb = torch.zeros(1, 128)
        for _ in range(T):
            with fc.stage("denoise"):
                te = t2(F.silu(t1(temb)))
                hdn = u2(torch.tanh(u1(x))) + te[:, None]
                for lay in m.layers:
                    hdn = lay(hdn)
                x0 = d2(torch.tanh(d1(hdn)))
            with fc.stage("round"):
                xf = x0.reshape(-1, k)
                dist = (E * E).sum(-1)[None] + (xf * xf).sum(-1)[:, None] - 2 * (xf @ E.t())
                dist.topk(1, dim=-1, largest=False)
        with fc.stage("final_head"):
            x.reshape(-1, k) @ E.t()
        return dict(T=T, canvas=Lc, in_channel=k, vocab=V, clamp_every_step=True)
    return f


def S_large_ar(a, n=N, prompt=0):
    def f(dev, fc):
        return ar_decode(make(a, dev), dev, fc, n, prompt=prompt, head_prompt="last")
    return f


def S_masked_diffusion_large(a, T, head, prompt=0, n=N):
    """Dream/LLaDA-style: T full passes over prompt + n; head on the masked positions under a linear schedule
    ('masked', the manuscript's (T+1)/2 * n) or on every position ('all', what both released decoders execute)."""
    def f(dev, fc):
        m = make(a, dev)
        per = n / T
        for t in range(T):
            masked = n - round(per * t)
            with fc.stage("denoise"):
                h = m(ids(dev, 1, prompt + n))
                if head == "all":
                    m.logits(h)
                else:
                    m.logits(h, masked)
        return dict(T=T, prompt=prompt, head=head)
    return f


def S_llada_semi(a, b, Tb, faithful, prompt=0, n=N):
    """Paper: trunk over min(jb, n) positions, head on the masked block positions (linear schedule). Released
    generate.py: trunk over prompt + the whole n-token canvas (future blocks still masked) every step, full logits."""
    def f(dev, fc):
        m = make(a, dev)
        for j in range(1, n // b + 1):
            for t in range(Tb):
                with fc.stage("denoise"):
                    if faithful:
                        m.logits(m(ids(dev, 1, prompt + n)))
                    else:
                        h = m(ids(dev, 1, min(j * b, n)))
                        m.logits(h, b - round(b / Tb * t))
        return dict(block=b, T_b=Tb, faithful=faithful, prompt=prompt)
    return f


def S_sdlm(a, D=4, kbar=2, n=N):
    """SDLM (paper schedule, forced kbar): each pass proposes D positions against the cached prefix (causal within
    the block), head on D positions, commits kbar; S = ceil(n / kbar) passes. Stated schedule only (the released
    decoder was not read)."""
    def f(dev, fc):
        m = make(a, dev)
        cache = [dict() for _ in m.layers]
        S = math.ceil(n / kbar)
        with fc.stage("decode"):
            for s in range(S):
                # the committed kbar tokens enter the cache; the block of D queries attends prefix + causal block
                m.logits(m(ids(dev, 1, D), causal=True, cache=cache, store=False))
                for c in cache:  # the kbar committed positions' K/V are reused from the proposal pass (no matmul)
                    kk = torch.zeros(1, a["hkv"], kbar, a["d"] // a["h"], device=dev)
                    c["k"] = kk if "k" not in c else torch.cat([c["k"], kk], 2)
                    c["v"] = c["k"]
        return dict(D=D, kbar=kbar, passes=S, note="paper schedule; committed tokens' K/V reused from the proposal pass")
    return f


# ------------------------------------------------------------------------------------------------ closed forms (MAC)
def cf_full(n, a=REF):
    d, L = a["d"], a["L"]
    return L * (12 * n * d * d + 2 * n * n * d)


def cf_causal(n, a=REF):
    d, L = a["d"], a["L"]
    return L * (12 * n * d * d + n * n * d + n * d)


def cf_vocab(m, a=REF):
    return m * a["d"] * a["V"]


def cf_xattn(n, ns, a=REF):
    d = a["d"]
    return a["L"] * (2 * n * d * d + 2 * ns * d * d + 2 * n * ns * d)


def closed_forms():
    n0, mm = round(0.8 * N), round(0.4 * N)
    d = REF["d"]
    levt_iter = lambda f: (f * cf_full(n0) + f * cf_full(n0) + cf_full(N) + n0 * d * 2 + (n0 + 1) * 2 * d * (KMAX + 1)
                           + cf_vocab(mm))
    ms = mp_masks()
    out = {
        "ar": cf_causal(N) + cf_vocab(N),
        "sedd_T32": 32 * (cf_full(N) + cf_vocab(N)),
        "mdlm_T1000": 1000 * (cf_full(N) + cf_vocab(N)),
        "difflm_T200_g0": 200 * cf_full(N) + cf_vocab(N),
        "difflm_T200_g1": 200 * cf_full(N) + 200 * cf_vocab(N),
        "difflm_T2000_g0": 2000 * cf_full(N) + cf_vocab(N),
        "inst_balanced": sum(cf_full(t) + cf_vocab(t + 1) for t in [2 ** i for i in range(8)]),
        "inst_serial": sum(cf_full(t) + cf_vocab(t + 1) for t in range(1, N + 1)),
        "nat": cf_full(N) + N * d * 50 + cf_full(N) + cf_xattn(N, N) + cf_vocab(N),
        "cmlm": cf_full(N) + sum(cf_full(N) + cf_xattn(N, N) + cf_vocab(m) for m in ms),
    }
    for R in (5, 10):
        out[f"levt_R{R}_noEE"] = R * levt_iter(1.0)
        # half depth: the manuscript scales the whole trunk term by f
        out[f"levt_R{R}_EE0.5"] = R * levt_iter(0.5)
    for p in (0.20, 0.28, 0.33):
        ne = round(N / (1 - p))
        out[f"reviser_p{p:.2f}"] = cf_causal(ne) + ne * d * V_ACT
    return out


# values printed in the manuscript (G MAC; experiments/measured_flops/coverage.md), for the old-vs-probe table
PRINTED = {"ar": 15.939, "reviser_p0.20": 19.9709, "reviser_p0.28": 22.2191, "reviser_p0.33": 23.9006,
           "levt_R5_noEE": 154.6186, "levt_R10_noEE": 309.2373, "levt_R5_EE0.5": 110.1657, "levt_R10_EE0.5": 220.3315,
           "inst_balanced": 32.1602, "inst_serial": 1036.2024, "nat": 31.1932, "cmlm": 189.0486,
           "difflm_T200_g0": 2239.6404, "difflm_T200_g1": 3217.7652, "difflm_T2000_g0": 22352.1669,
           "sedd_T32": 514.8424, "sedd_T2048": 32949.9154, "mdlm_T1000": 16088.8259, "qwen32b_ar": 4099.3869,
           "qwen7b_ar": 906.6536, "llama3_8b_ar": 962.7605, "sdlm32b_D4_k2": 8198.9416, "dream7b_T10": 8768.9307,
           "dream7b_T20": 17502.9816, "llada8b_T256": 238317.9374, "llada8b_semi_b32_Tb16": 36426.5572}


# ------------------------------------------------------------------------------------------------ registry
def rows():
    R = []
    add = lambda key, fn, variant, mrow, denom, note="": R.append(dict(key=key, fn=fn, variant=variant,
                                                                      manuscript_row=mrow, denominator=denom, note=note))
    add("ar", S_ar, "paper", "1", "ar")
    add("ar_mt", S_ar_mt, "faithful", "denominator for encoder-decoder rows", "ar_mt")
    add("ar_n64", S_ar_n(64), "faithful", "denominator for the official Diffusion-LM canvas (L = 64)", "ar_n64")
    for p in (0.20, 0.28, 0.33):
        add(f"reviser_p{p:.2f}", S_reviser(p), "paper", "2", "ar")
    for R_ in (5, 10):
        add(f"levt_R{R_}_noEE", S_levt_paper(R_, 1.0), "paper", "3a" if R_ == 5 else "3b", "ar")
        add(f"levt_R{R_}_EE0.5", S_levt_paper(R_, 0.5), "paper", "3c" if R_ == 5 else "3d", "ar")
        add(f"levt_faithful_R{R_}", S_levt_faithful(R_), "faithful", "3a/3b", "ar_mt")
    add("inst_balanced", S_inst_paper(False), "paper", "4a", "ar")
    add("inst_serial", S_inst_paper(True), "paper", "4b", "ar")
    add("inst_faithful", S_inst_faithful, "faithful", "4a", "ar_mt")
    add("nat", S_nat_paper, "paper", "5", "ar", "manuscript divides by the decoder-only AR")
    add("nat_faithful_npd1", S_nat_faithful(1, False), "faithful", "5", "ar_mt")
    add("nat_faithful_npd10_teacher", S_nat_faithful(10, True), "faithful", "5", "ar_mt")
    add("cmlm", S_cmlm_paper, "paper", "6", "ar", "manuscript divides by the decoder-only AR")
    add("cmlm_fairseq", S_cmlm_fairseq, "faithful", "6", "ar_mt")
    add("maskpredict_official", S_maskpredict_official, "faithful", "6", "ar_mt")
    add("difflm_T200_g0", S_difflm_paper(200, 0), "paper", "7a", "ar")
    add("difflm_T200_g1", S_difflm_paper(200, 1), "paper", "7b", "ar")
    add("difflm_T2000_g0", S_difflm_paper(2000, 0), "paper", "7c", "ar")
    add("difflm_official_e2e", S_difflm_official(2000, 64, 16, 821), "faithful", "7", "ar_n64",
        "official README setting: T = 2000, L = 64, E2E vocabulary 821, k = 16")
    add("difflm_official_roc", S_difflm_official(2000, 64, 128, 11043), "faithful", "7", "ar_n64",
        "ROCStories: V = 11043, k = 128")
    add("difflm_structure_n128", S_difflm_official(2000, N, 128, REF["V"]), "faithful", "7", "ar",
        "official operator structure at the manuscript's n = 128, |V| = 50k, k = 128")
    add("sedd_T32", S_diffusion_paper(32), "paper", "8a", "ar")
    add("sedd_T2048", S_diffusion_paper(2048), "paper", "8b", "ar")
    add("mdlm_T1000", S_diffusion_paper(1000), "paper", "9/10", "ar", "D3PM row identical")
    add("qwen7b_ar", S_large_ar(QWEN7B), "paper", "12a", "qwen7b_ar")
    add("qwen32b_ar", S_large_ar(QWEN32B), "paper", "11a", "qwen32b_ar")
    add("llama3_8b_ar", S_large_ar(LLAMA3_8B), "paper", "13a", "llama3_8b_ar", "published config values; not downloaded")
    add("sdlm32b_D4_k2", S_sdlm(QWEN32B), "paper", "11b", "qwen32b_ar")
    for T in (10, 20, 128):
        if T != 128:
            add(f"dream7b_T{T}", S_masked_diffusion_large(QWEN7B, T, "masked"), "paper", "12b" if T == 10 else "12c", "qwen7b_ar")
        add(f"dream7b_T{T}_fulllogits", S_masked_diffusion_large(QWEN7B, T, "all"), "faithful", "12b/12c", "qwen7b_ar")
    add("llada8b_T256", S_masked_diffusion_large(LLADA8B, 256, "masked"), "paper", "13b", "llama3_8b_ar")
    add("llada8b_T256_fulllogits", S_masked_diffusion_large(LLADA8B, 256, "all"), "faithful", "13b", "llama3_8b_ar")
    add("llada8b_T128_fulllogits", S_masked_diffusion_large(LLADA8B, 128, "all"), "faithful", "13b",
        "llama3_8b_ar", "released default steps = gen_length")
    add("llada8b_semi_b32_Tb16", S_llada_semi(LLADA8B, 32, 16, False), "paper", "13c", "llama3_8b_ar")
    add("llada8b_semi_b32_Tb16_released", S_llada_semi(LLADA8B, 32, 16, True), "faithful", "13c", "llama3_8b_ar")
    # prompt-conditioned structure (GSM8K-like prompt P = 900): what the no-prompt closed forms omit
    P = 900
    add("qwen7b_ar_P900", S_large_ar(QWEN7B, prompt=P), "faithful", "12a (+prompt)", "qwen7b_ar_P900")
    add("dream7b_T128_P900", S_masked_diffusion_large(QWEN7B, 128, "all", prompt=P), "faithful", "12c (+prompt)",
        "qwen7b_ar_P900")
    add("llada8b_T128_P900", S_masked_diffusion_large(LLADA8B, 128, "all", prompt=P), "faithful", "13b (+prompt)",
        "qwen7b_ar_P900", "different-family AR denominator")
    add("llada8b_semi_b32_Tb16_P900", S_llada_semi(LLADA8B, 32, 16, True, prompt=P), "faithful", "13c (+prompt)",
        "qwen7b_ar_P900", "different-family AR denominator")
    return R


def run_one(r, dev, vocab):
    fc = FlopCounter(vocab_sizes=vocab)
    torch.manual_seed(0)
    with torch.no_grad(), fc:
        sched = r["fn"](dev, fc)
    s = fc.summary()
    return sched, s


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="meta")
    ap.add_argument("--only", default="", help="comma-separated keys")
    ap.add_argument("--crosscheck", action="store_true", help="also run a small subset on CPU and compare counts")
    ap.add_argument("--out", default=str(ROOT / "runs" / "measured_flops" / "panel_s" / "probe.json"))
    a = ap.parse_args(argv)
    vocab = {REF["V"], V_ACT, QWEN7B["V"], LLAMA3_8B["V"], LLADA8B["V"], 821, 11043}
    sel = set(a.only.split(",")) if a.only else None
    cf = closed_forms()
    res = {}
    for r in rows():
        if sel and r["key"] not in sel:
            continue
        sched, s = run_one(r, a.device, vocab)
        t = s["total"]
        e = dict(variant=r["variant"], manuscript_row=r["manuscript_row"], denominator=r["denominator"],
                 note=r["note"], schedule=sched, totals=t,
                 per_stage={k: {kk: v[kk] for kk in ("macs_matmul", "linear", "head", "attn", "attn_dense",
                                                     "other_matmul", "elementwise", "head_positions")}
                            for k, v in s["per_stage"].items()},
                 uncovered=s["uncovered"])
        if r["key"] in PRINTED:
            e["manuscript_printed_gmac"] = PRINTED[r["key"]]
            e["measured_over_printed"] = t["macs_matmul"] / G / PRINTED[r["key"]]
        if r["key"] in cf:
            e["closed_form_macs"] = cf[r["key"]]
            e["closed_form_rel_diff"] = (t["macs_matmul"] - cf[r["key"]]) / cf[r["key"]]
        res[r["key"]] = e
        print(f"{r['key']:34s} {t['macs_matmul'] / G:12.3f} GMAC  head_pos={t['head_positions']:>8}"
              + (f"  cf_rel={e['closed_form_rel_diff']:+.2e}" if "closed_form_rel_diff" in e else ""), flush=True)
    for k, e in res.items():
        den = res.get(e["denominator"])
        if den:
            e["ratio"] = e["totals"]["macs_matmul"] / den["totals"]["macs_matmul"]
            e["ratio_total_incl_elementwise"] = e["totals"]["total"] / den["totals"]["total"]
    cross = None
    if a.crosscheck:
        cross = {}
        for r in rows():
            if r["key"] not in ("ar", "ar_mt", "reviser_p0.20", "inst_balanced", "cmlm", "levt_faithful_R5",
                                "maskpredict_official", "nat_faithful_npd10_teacher"):
                continue
            _, sm = run_one(r, "meta", vocab)
            _, sc = run_one(r, "cpu", vocab)
            keys = ("linear", "head", "attn", "attn_dense", "other_matmul", "elementwise", "head_positions")
            diff = {k: sc["total"][k] - sm["total"][k] for k in keys}
            cross[r["key"]] = dict(identical=all(v == 0 for v in diff.values()), diff=diff,
                                   uncovered_cpu=sc["uncovered"], uncovered_meta=sm["uncovered"])
            print("crosscheck", r["key"], cross[r["key"]]["identical"], flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    out = dict(panel="S (structural probe)", label="probe", device=a.device, torch=torch.__version__,
               git_commit=commit, flopcount_sha256=sha256(ROOT / "scripts/benchmarks/flopcount.py"),
               probe_sha256=sha256(Path(__file__)), reference=REF, n=N, K_max=KMAX, V_action=V_ACT,
               architectures=dict(qwen7b=QWEN7B, qwen32b=QWEN32B, llama3_8b=LLAMA3_8B, llada8b=LLADA8B),
               crosscheck_meta_vs_cpu=cross, rows=res)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
