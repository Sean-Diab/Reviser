"""Neural cursor model (version N): learns the order in which people write the tokens of a known final text.

Design (user, 2026-10-07; `notes/human-order-plan-2026-10-07.md` section 2). Number the continuation tokens 0..n-1.
Each step the model picks which unwritten token to insert next (a pointer over the text, not a vocabulary
prediction). Token j always lands between its nearest written neighbours, so every order is valid; the cursor move
follows from the pick (`orders.order_to_actions`).

- Reader: distilroberta-base reads prefix + continuation once (GPT-2 ids mapped to RoBERTa ids through the shared
  byte-level BPE strings; a learned segment embedding marks prefix vs continuation) -> one vector per token.
- Picker: small pre-LN decoder (causal self-attention over its own steps, cross-attention to the reader vectors).
  Step input = projected reader vector of the token picked last (the token left of the cursor; the last prefix
  token at step 0) + embeddings of that pick's kind and move distance + step position + tokens remaining.
- Score of each unwritten token j = q . k_j + feature terms. Features are computed by code each step (see
  `step_features`): pick kind (next token at the cursor / skip inside the cursor's gap / move elsewhere), signed
  distance from the cursor in written tokens, j's gap size, j's offset from both gap ends, j in the tail gap. Each
  feature bucket has an embedding dotted with q plus a scalar bias, so feature bonuses depend on the picker state.
  Written tokens are masked.
- Features-only ablation (`--use_reader 0`): same picker and features, no reader (no content keys, no
  cross-attention, no content in the step input).

Pick kinds: KEEP = the next token at the cursor (no move, no hole); SKIP = another token of the cursor's gap (no move
token now, leaves a hole that needs a move later); MOVE = a token outside the cursor's gap (needs move tokens; these
are the "jumps" counted by `orders.run_lengths`).

Commands (run from the repo root; WSL or the rented box):
  python scripts/humanorder/cursor_model.py vocab
  python scripts/humanorder/cursor_model.py train --use_reader 1 --out runs/humanorder_n/full_s0
  python scripts/humanorder/cursor_model.py eval --ckpt runs/humanorder_n/full_s0/best.pt --split test
  python scripts/humanorder/cursor_model.py gen --ckpt ... --split test --samples 4 --seed 1
  python scripts/humanorder/cursor_model.py gen-dir --ckpt ... --texts_dir DIR --out FILE.npz --seed 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from orders import order_to_actions, run_lengths  # noqa: E402,F401

KEEP, SKIP, MOVE = 0, 1, 2
START_KIND = 3
BOUNDS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]   # bucketize: 0 | 1 | 2 | 3-4 | 5-8 | ... | 129-256 | >256
FEATURES = (("kind", 3), ("dist", 21), ("gsize", 10), ("goff", 11), ("groff", 11), ("tail", 2))
DEFAULT_READER = os.path.join(ROOT, "data/models/distilroberta-base")
DEFAULT_MAP = os.path.join(ROOT, "data/humanorder/gpt2_to_roberta.npy")
DEFAULT_DATA = os.path.join(ROOT, "data/humanorder/coauthor_v1")
RB_BOS, RB_PAD, RB_EOS = 0, 1, 2


# ----------------------------------------------------------------------------------------------------------------
# Features (one implementation for training [B,T,N] and generation [B,N])
# ----------------------------------------------------------------------------------------------------------------

def step_features(written: torch.Tensor, last: torch.Tensor, n_len: torch.Tensor):
    """Per-candidate features for one or many steps.

    written: bool [..., N] (token already on the canvas); last: long [...] index of the token left of the cursor
    (the last pick; -1 = cursor at the prefix end); n_len: long [...] continuation length (j >= n_len is padding).
    Returns (feats: dict name -> long [..., N], avail: bool [..., N], d: long [..., N] signed distance in written
    tokens between j's insertion point and the cursor; d == 0 exactly for KEEP/SKIP).
    """
    N = written.shape[-1]
    dev = written.device
    ar = torch.arange(N, device=dev)
    nl = n_len.unsqueeze(-1)
    valid = ar < nl
    w = written & valid
    avail = valid & ~written
    neg1 = torch.full(w.shape[:-1] + (1,), -1, dtype=torch.long, device=dev)
    widx = torch.where(w, ar, -1)
    lw = torch.cat([neg1, torch.cummax(widx, -1).values[..., :-1]], -1)              # last written index < j
    widx_r = torch.where(w, ar, nl)
    rw_in = torch.flip(torch.cummin(torch.flip(widx_r, [-1]), -1).values, [-1])
    rw = torch.cat([rw_in[..., 1:], nl.expand(neg1.shape)], -1)                      # next written index > j (or n)
    wl = w.long()
    cs = wl.cumsum(-1)
    pos = cs - wl                                                                      # written tokens before j
    L = last.unsqueeze(-1)
    cur = torch.where(L >= 0, cs.gather(-1, L.clamp(min=0)), torch.zeros_like(L))    # cursor = written tokens <= last
    d = pos - cur
    kind = torch.where(ar == L + 1, KEEP, torch.where(lw == L, SKIP, MOVE))
    b = torch.tensor(BOUNDS, device=dev)
    dist = torch.bucketize(d.abs(), b) + 10 * (d < 0).long()
    gsize = torch.bucketize((rw - lw - 1).clamp(min=1), b[1:])
    goff = torch.bucketize((ar - lw - 1).clamp(min=0), b)
    groff = torch.bucketize((rw - ar - 1).clamp(min=0), b)
    tail = (rw == nl).long()
    feats = {"kind": kind, "dist": dist, "gsize": gsize, "goff": goff, "groff": groff, "tail": tail}
    return feats, avail, d


def teacher_state(order: torch.Tensor, n_len: torch.Tensor):
    """written [B,T,N] and last [B,T] for every step of the given orders (order padded with -1 past n_len)."""
    B, T = order.shape
    dev = order.device
    steps = torch.arange(T, device=dev).expand(B, T)
    rank = torch.full((B, T + 1), T + 1, dtype=torch.long, device=dev)    # column T absorbs the padding
    rank.scatter_(1, torch.where(order >= 0, order, T), steps)
    written = rank[:, :T].unsqueeze(1) < torch.arange(T, device=dev).view(1, T, 1)
    last = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=dev), order[:, :-1]], 1)
    return written, last


def ref_step_features(order_prefix: list[int], n: int) -> list[dict]:
    """Slow pure-Python reference: features of every candidate after the picks in order_prefix (for tests)."""
    present = sorted(order_prefix)
    last = order_prefix[-1] if order_prefix else -1
    import bisect
    cur = bisect.bisect_right(present, last) if last >= 0 else 0
    out = []
    for j in range(n):
        if j in present:
            out.append(None)
            continue
        p = bisect.bisect_left(present, j)
        lw = present[p - 1] if p > 0 else -1
        rw = present[p] if p < len(present) else n
        d = p - cur
        kind = KEEP if j == last + 1 else (SKIP if lw == last else MOVE)
        out.append({"kind": kind, "d": d, "gs": rw - lw - 1, "off": j - lw - 1, "roff": rw - j - 1, "tail": int(rw == n)})
    return out


# ----------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------

class Attn(nn.Module):
    def __init__(self, d, h, p, self_attn):
        super().__init__()
        self.h, self.dh, self.p = h, d // h, p
        if self_attn:
            self.qkv = nn.Linear(d, 3 * d)
        else:
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def merge(self, x):
        B, H, T, dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * dh)

    def forward_self(self, x, cache=None, t=None):
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = self.split(q), self.split(k), self.split(v)
        if cache is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        else:
            if cache.get("k") is None:
                shape = (k.shape[0], k.shape[1], cache["N"], k.shape[3])
                cache["k"] = torch.zeros(shape, dtype=k.dtype, device=k.device)
                cache["v"] = torch.zeros(shape, dtype=v.dtype, device=v.device)
            cache["k"][:, :, t:t + 1] = k
            cache["v"][:, :, t:t + 1] = v
            o = F.scaled_dot_product_attention(q, cache["k"][:, :, :t + 1], cache["v"][:, :, :t + 1])
        return self.o(self.merge(o))

    def mem_kv(self, mem):
        k, v = self.kv(mem).chunk(2, -1)
        return self.split(k), self.split(v)

    def forward_cross(self, x, k, v, mask):
        q = self.split(self.q(x))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.o(self.merge(o))


class Block(nn.Module):
    def __init__(self, d, h, ff, p, cross):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = Attn(d, h, p, True)
        self.cross = cross
        if cross:
            self.ln2 = nn.LayerNorm(d)
            self.ca = Attn(d, h, p, False)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, mem_kv=None, mem_mask=None, cache=None, t=None):
        x = x + self.drop(self.sa.forward_self(self.ln1(x), cache, t))
        if self.cross:
            x = x + self.drop(self.ca.forward_cross(self.ln2(x), mem_kv[0], mem_kv[1], mem_mask))
        return x + self.drop(self.ff(self.ln3(x)))


class CursorModel(nn.Module):
    def __init__(self, use_reader=True, reader_path=DEFAULT_READER, reader_config=None, d=256, layers=2, heads=4,
                 ff=1024, dropout=0.1, max_steps=512):
        super().__init__()
        self.use_reader, self.d = bool(use_reader), d
        self.keep_bias = 0.0   # calibration only (not trained, not saved): added to the score of the next token at the cursor
        if self.use_reader:
            from transformers import RobertaConfig, RobertaModel
            if reader_config is not None:
                self.reader = RobertaModel(RobertaConfig(**reader_config), add_pooling_layer=False)
            else:
                self.reader = RobertaModel.from_pretrained(reader_path, add_pooling_layer=False)
            H = self.reader.config.hidden_size
            self.seg = nn.Embedding(2, H)
            nn.init.zeros_(self.seg.weight)
            self.in_proj = nn.Linear(H, d)
            self.key_proj = nn.Linear(H, d)
            self.mem_proj = nn.Linear(H, d)
        self.start = nn.Parameter(torch.zeros(d))
        self.e_kind = nn.Embedding(4, d)
        self.e_dist = nn.Embedding(21, d)
        self.e_pos = nn.Embedding(max_steps, d)
        self.e_rem = nn.Embedding(11, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout, self.use_reader) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.feat_emb = nn.ModuleDict({k: nn.Embedding(nb, d) for k, nb in FEATURES})
        self.feat_bias = nn.ParameterDict({k: nn.Parameter(torch.zeros(nb)) for k, nb in FEATURES})
        for k, _ in FEATURES:
            nn.init.normal_(self.feat_emb[k].weight, std=0.02)
        for emb in (self.e_kind, self.e_dist, self.e_pos, self.e_rem):
            nn.init.normal_(emb.weight, std=0.02)

    def picker_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("reader.")]

    # --- reader ---
    def read(self, ids, seg, attn):
        """ids/seg/attn [B,L] -> reader vectors [B,L,H]."""
        emb = self.reader.embeddings.word_embeddings(ids) + self.seg(seg)
        pos = (torch.arange(ids.shape[1], device=ids.device) + self.reader.config.pad_token_id + 1).expand_as(ids)
        return self.reader(inputs_embeds=emb, attention_mask=attn, position_ids=pos).last_hidden_state

    def encode(self, batch):
        """Returns ctx: keys [B,N,d] (or None), cont vectors [B,N,H], start vectors [B,H], per-block cross K/V."""
        if not self.use_reader:
            return {"K": None, "hc": None, "h0": None, "mem": [None] * len(self.blocks), "mask": None}
        Hs = self.read(batch["ids"], batch["seg"], batch["attn"])
        hc = Hs.gather(1, batch["cont_pos"].unsqueeze(-1).expand(-1, -1, Hs.shape[-1]))
        h0 = Hs.gather(1, batch["start_pos"].view(-1, 1, 1).expand(-1, 1, Hs.shape[-1])).squeeze(1)
        M = self.mem_proj(Hs)
        mem = [blk.ca.mem_kv(M) for blk in self.blocks]
        mask = batch["attn"].bool()[:, None, None, :]
        return {"K": self.key_proj(hc), "hc": hc, "h0": h0, "mem": mem, "mask": mask}

    def step_input(self, ctx, prev_idx, prev_kind, prev_dist, t_idx, n_len):
        """prev_idx [B,T] (-1 = start), prev_kind/prev_dist [B,T], t_idx [B,T] -> [B,T,d]."""
        x = self.e_kind(prev_kind) + self.e_dist(prev_dist) + self.e_pos(t_idx)
        rem = torch.bucketize((n_len.unsqueeze(-1) - t_idx).clamp(min=0), torch.tensor(BOUNDS, device=t_idx.device))
        x = x + self.e_rem(rem)
        if self.use_reader:
            hc = ctx["hc"]
            v = hc.gather(1, prev_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, hc.shape[-1]))
            v = torch.where((prev_idx >= 0).unsqueeze(-1), v, ctx["h0"].unsqueeze(1).to(v.dtype))
            x = x + self.in_proj(v)
        else:
            x = x + torch.where((prev_idx >= 0).unsqueeze(-1), torch.zeros_like(x), self.start.to(x.dtype))
        return x

    def scores(self, q, ctx, feats, avail):
        """q [B,T,d], feats [B,T,N] -> float32 logits [B,T,N] with unavailable tokens at -inf."""
        scale = 1.0 / math.sqrt(self.d)
        q = q.float()
        if ctx["K"] is not None:
            logit = torch.einsum("btd,bnd->btn", q, ctx["K"].float()) * scale
        else:
            logit = torch.zeros(avail.shape, device=q.device)
        for k, _ in FEATURES:
            A = q @ self.feat_emb[k].weight.float().t() * scale + self.feat_bias[k].float()
            logit = logit + A.gather(-1, feats[k])
        if self.keep_bias:
            logit = logit + self.keep_bias * (feats["kind"] == KEEP).float()
        return logit.masked_fill(~avail, float("-inf"))

    def forward(self, batch):
        """Teacher forcing over batch['order'] [B,N]. Returns logits [B,T,N], feats, avail."""
        order, n_len = batch["order"], batch["n_len"]
        B, T = order.shape
        ctx = self.encode(batch)
        written, last = teacher_state(order, n_len)
        feats, avail, _ = step_features(written, last, n_len.unsqueeze(1).expand(B, T))
        pad_step = (order < 0).unsqueeze(-1) & (torch.arange(T, device=order.device) == 0)
        avail = avail | pad_step                       # keeps padding rows finite; they carry no loss
        lab = order.clamp(min=0).unsqueeze(-1)
        kind_lab = feats["kind"].gather(-1, lab).squeeze(-1)
        dist_lab = feats["dist"].gather(-1, lab).squeeze(-1)
        st = torch.full((B, 1), START_KIND, dtype=torch.long, device=order.device)
        prev_kind = torch.cat([st, kind_lab[:, :-1]], 1)
        prev_dist = torch.cat([torch.zeros_like(st), dist_lab[:, :-1]], 1)
        t_idx = torch.arange(T, device=order.device).expand(B, T)
        x = self.step_input(ctx, last, prev_kind, prev_dist, t_idx, n_len)
        for blk, mem in zip(self.blocks, ctx["mem"]):
            x = blk(x, mem, ctx["mask"])
        q = self.q_proj(self.ln_f(x))
        return self.scores(q, ctx, feats, avail), feats, avail

    @torch.no_grad()
    def generate(self, batch, temperature=1.0, generator=None, forced=None, return_logits=False):
        """Sample one order per text with a KV cache. forced [B,N] replays given orders (for tests)."""
        n_len = batch["n_len"]
        B = n_len.shape[0]
        N = int(n_len.max())
        dev = n_len.device
        ctx = self.encode(batch)
        caches = [{"N": N, "k": None, "v": None} for _ in self.blocks]
        written = torch.zeros(B, N, dtype=torch.bool, device=dev)
        last = torch.full((B,), -1, dtype=torch.long, device=dev)
        prev_kind = torch.full((B,), START_KIND, dtype=torch.long, device=dev)
        prev_dist = torch.zeros(B, dtype=torch.long, device=dev)
        out = torch.full((B, N), -1, dtype=torch.long, device=dev)
        ar_b = torch.arange(B, device=dev)
        logits_all = [] if return_logits else None
        for t in range(N):
            t_idx = torch.full((B, 1), t, dtype=torch.long, device=dev)
            x = self.step_input(ctx, last.unsqueeze(1), prev_kind.unsqueeze(1), prev_dist.unsqueeze(1), t_idx, n_len)
            for blk, mem, cache in zip(self.blocks, ctx["mem"], caches):
                x = blk(x, mem, ctx["mask"], cache, t)
            q = self.q_proj(self.ln_f(x))
            feats, avail, _ = step_features(written, last, n_len)
            active = t < n_len
            avail = avail | (~active).unsqueeze(1) & (torch.arange(N, device=dev) == 0)
            logit = self.scores(q, ctx, {k: v.unsqueeze(1) for k, v in feats.items()}, avail.unsqueeze(1)).squeeze(1)
            if return_logits:
                logits_all.append(logit)
            if forced is not None:
                pick = forced[:, t].clamp(min=0)
            else:
                u = torch.rand(logit.shape, device=dev, generator=generator).clamp_(1e-20, 1.0 - 1e-7)
                pick = (logit / temperature - torch.log(-torch.log(u))).argmax(-1)
            pick = torch.where(active, pick, torch.zeros_like(pick))
            out[:, t] = torch.where(active, pick, out[:, t])
            prev_kind = feats["kind"][ar_b, pick]
            prev_dist = feats["dist"][ar_b, pick]
            written[ar_b, pick] = written[ar_b, pick] | active
            last = torch.where(active, pick, last)
        if return_logits:
            return out, torch.stack(logits_all, 1)
        return out


# ----------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------

def load_windows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def collate(items, vmap, device, with_order=True):
    """items: dicts with prefix, cont (GPT-2 ids) and optionally order."""
    B = len(items)
    P = [len(it["prefix"]) for it in items]
    n = [len(it["cont"]) for it in items]
    L = max(p + m for p, m in zip(P, n)) + 2
    N = max(n)
    ids = np.full((B, L), RB_PAD, dtype=np.int64)
    seg = np.zeros((B, L), dtype=np.int64)
    attn = np.zeros((B, L), dtype=np.int64)
    cont_pos = np.zeros((B, N), dtype=np.int64)
    order = np.full((B, N), -1, dtype=np.int64)
    for i, it in enumerate(items):
        p, m = P[i], n[i]
        seq = [RB_BOS] + [int(vmap[t]) for t in it["prefix"]] + [int(vmap[t]) for t in it["cont"]] + [RB_EOS]
        ids[i, :len(seq)] = seq
        attn[i, :len(seq)] = 1
        seg[i, 1 + p:2 + p + m] = 1
        cont_pos[i, :m] = np.arange(1 + p, 1 + p + m)
        cont_pos[i, m:] = 1 + p
        if with_order:
            order[i, :m] = it["order"]
    t = lambda a: torch.from_numpy(a).to(device)
    return {"ids": t(ids), "seg": t(seg), "attn": t(attn), "cont_pos": t(cont_pos),
            "start_pos": t(np.asarray(P, dtype=np.int64)), "n_len": t(np.asarray(n, dtype=np.int64)), "order": t(order)}


def batches_by_length(items, bs, rng=None):
    idx = sorted(range(len(items)), key=lambda i: len(items[i]["cont"]))
    bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if rng is not None:
        rng.shuffle(bl)
    return bl


def load_vmap(path=DEFAULT_MAP):
    return np.load(path)


# ----------------------------------------------------------------------------------------------------------------
# Heartbeat (remote watchdog: JSON with time_utc, rewritten every 30 s while the process lives)
# ----------------------------------------------------------------------------------------------------------------

def start_heartbeat(path, job):
    if not path:
        return
    def loop():
        while True:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": job,
                           "pid": os.getpid()}, f)
            os.replace(tmp, path)
            time.sleep(30)
    threading.Thread(target=loop, daemon=True).start()


# ----------------------------------------------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------------------------------------------

def build_model(margs, device):
    m = CursorModel(use_reader=margs["use_reader"], reader_path=margs.get("reader", DEFAULT_READER),
                    reader_config=margs.get("reader_config"), d=margs["d"], layers=margs["layers"],
                    heads=margs["heads"], ff=margs["ff"], dropout=margs["dropout"])
    return m.to(device)


def load_ckpt(path, device, keep_bias=0.0):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    margs = dict(ck["model_args"])
    if margs.get("use_reader") and margs.get("reader_config") is None:
        margs["reader_config"] = ck["reader_config"]       # weights come from the checkpoint, not from disk
    m = build_model(margs, device)
    m.load_state_dict(ck["state"])
    m.eval()
    m.keep_bias = float(keep_bias)
    return m, ck


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def evaluate(model, items, vmap, device, bs=16, amp=True):
    """Per-window sums: nll (all/keep/skip/move), counts, top-1/top-5 hits for skip and move picks."""
    model.eval()
    W = len(items)
    res = {k: np.zeros(W) for k in ("nll", "n", "nll_keep", "n_keep", "nll_skip", "n_skip", "nll_move", "n_move",
                                     "top1_skip", "top5_skip", "top1_move", "top5_move", "top1_all",
                                     "pred_keep", "pred_skip", "pred_move")}
    for bidx in batches_by_length(items, bs):
        batch = collate([items[i] for i in bidx], vmap, device)
        with autocast(device, amp):
            logits, feats, avail = model(batch)
        order = batch["order"]
        valid = order >= 0
        lab = order.clamp(min=0).unsqueeze(-1)
        logp = torch.log_softmax(logits, -1)
        lp = logp.gather(-1, lab).squeeze(-1)
        lab_logit = logits.gather(-1, lab)
        rank = (logits > lab_logit).sum(-1)
        kind = feats["kind"].gather(-1, lab).squeeze(-1)
        nll = torch.where(valid, -lp, torch.zeros_like(lp))
        probs = logp.exp()
        pk = {kv: torch.where(valid, (probs * (feats["kind"] == kv)).sum(-1), torch.zeros_like(lp))
              for kv in (KEEP, SKIP, MOVE)}   # predicted probability mass of each pick kind per step
        for bi, wi in enumerate(bidx):
            v = valid[bi]
            res["nll"][wi] = nll[bi][v].sum().item()
            res["n"][wi] = v.sum().item()
            res["top1_all"][wi] = (rank[bi][v] == 0).sum().item()
            for kname, kv in (("keep", KEEP), ("skip", SKIP), ("move", MOVE)):
                res[f"pred_{kname}"][wi] = pk[kv][bi].sum().item()
            for kname, kv in (("keep", KEEP), ("skip", SKIP), ("move", MOVE)):
                mk = v & (kind[bi] == kv)
                res[f"nll_{kname}"][wi] = nll[bi][mk].sum().item()
                res[f"n_{kname}"][wi] = mk.sum().item()
                if kname != "keep":
                    res[f"top1_{kname}"][wi] = (rank[bi][mk] == 0).sum().item()
                    res[f"top5_{kname}"][wi] = (rank[bi][mk] < 5).sum().item()
    return res


def summarize_eval(res):
    s = lambda k: float(res[k].sum())
    out = {"windows": int(len(res["n"])), "picks": int(s("n")), "nll_per_pick": s("nll") / s("n"),
           "top1_all": s("top1_all") / s("n")}
    for k in ("keep", "skip", "move"):
        out[f"picks_{k}"] = int(s(f"n_{k}"))
        out[f"nll_{k}"] = s(f"nll_{k}") / max(1.0, s(f"n_{k}"))
    for k in ("skip", "move"):
        out[f"top1_{k}"] = s(f"top1_{k}") / max(1.0, s(f"n_{k}"))
        out[f"top5_{k}"] = s(f"top5_{k}") / max(1.0, s(f"n_{k}"))
    jn = s("n_skip") + s("n_move")
    out["nll_jump_any"] = (s("nll_skip") + s("nll_move")) / max(1.0, jn)
    if "pred_keep" in res:   # calibration: expected vs actual share of each pick kind under teacher forcing
        for k in ("keep", "skip", "move"):
            out[f"share_{k}_actual"] = s(f"n_{k}") / s("n")
            out[f"share_{k}_predicted"] = s(f"pred_{k}") / s("n")
    return out


def cmd_train(a):
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    start_heartbeat(a.heartbeat, f"train {a.out}")
    vmap = load_vmap(a.vocab_map)
    train = load_windows(os.path.join(a.data, "windows_train.jsonl"))
    dev = load_windows(os.path.join(a.data, "windows_dev.jsonl"))
    if a.limit:
        train, dev = train[:a.limit], dev[:max(8, a.limit // 8)]
    margs = {"use_reader": bool(a.use_reader), "reader": a.reader, "d": a.d, "layers": a.layers, "heads": a.heads,
             "ff": a.ff, "dropout": a.dropout,
             "reader_config": json.loads(a.reader_config) if a.reader_config else None}
    model = build_model(margs, device)
    n_reader = sum(p.numel() for n, p in model.named_parameters() if n.startswith("reader."))
    n_picker = sum(p.numel() for p in model.picker_params())
    groups = [{"params": model.picker_params(), "lr": a.lr}]
    if model.use_reader:
        groups.append({"params": [p for n, p in model.named_parameters() if n.startswith("reader.")], "lr": a.lr_reader})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train) / a.bs)
    total = steps_per_epoch * a.epochs
    warm = min(a.warmup, total // 10 + 1)
    base = [g["lr"] for g in opt.param_groups]
    rng = random.Random(a.seed)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    meta = {"args": {k: v for k, v in vars(a).items() if k != "f"}, "model_args": margs, "params_reader": n_reader, "params_picker": n_picker,
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__, "train_windows": len(train), "dev_windows": len(dev),
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except Exception:
        pass
    print(json.dumps(meta), flush=True)
    log.write(json.dumps({"meta": meta}) + "\n")
    best, bad, step, t0 = float("inf"), 0, 0, time.time()
    for ep in range(a.epochs):
        model.train()
        tl, tn = 0.0, 0
        for bidx in batches_by_length(train, a.bs, rng):
            frac = step / max(1, total)
            mult = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * mult
            batch = collate([train[i] for i in bidx], vmap, device)
            with autocast(device, a.amp):
                logits, _, _ = model(batch)
            valid = batch["order"] >= 0
            lp = torch.log_softmax(logits, -1).gather(-1, batch["order"].clamp(min=0).unsqueeze(-1)).squeeze(-1)
            loss = -torch.where(valid, lp, torch.zeros_like(lp)).sum() / valid.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            tl += loss.item() * valid.sum().item()
            tn += valid.sum().item()
            step += 1
            if step % a.log_every == 0:
                rec = {"step": step, "epoch": ep, "train_nll": tl / tn, "gnorm": float(gn), "lr_mult": mult,
                       "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        dv = summarize_eval(evaluate(model, dev, vmap, device, a.bs, a.amp))
        rec = {"epoch": ep, "step": step, "train_nll_epoch": tl / max(1, tn), "dev": dv,
               "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if dv["nll_per_pick"] < best - 1e-4:
            best, bad = dv["nll_per_pick"], 0
            reader_cfg = model.reader.config.to_dict() if model.use_reader else None
            if reader_cfg is not None:
                reader_cfg = {k: v for k, v in reader_cfg.items() if k in (
                    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                    "hidden_act", "hidden_dropout_prob", "attention_probs_dropout_prob", "max_position_embeddings",
                    "type_vocab_size", "initializer_range", "layer_norm_eps", "pad_token_id", "bos_token_id",
                    "eos_token_id")}
            torch.save({"model_args": margs, "reader_config": reader_cfg, "state": model.state_dict(), "epoch": ep,
                        "step": step, "dev": dv, "meta": meta}, os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print(json.dumps({"early_stop_epoch": ep}), flush=True)
                log.write(json.dumps({"early_stop_epoch": ep, "best_dev_nll": best}) + "\n")
                break
    log.write(json.dumps({"done": True, "best_dev_nll": best, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    log.close()


def cmd_eval(a):
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    start_heartbeat(a.heartbeat, f"eval {a.ckpt}")
    model, ck = load_ckpt(a.ckpt, device, a.keep_bias)
    vmap = load_vmap(a.vocab_map)
    items = load_windows(os.path.join(a.data, f"windows_{a.split}.jsonl"))
    res = evaluate(model, items, vmap, device, a.bs, a.amp)
    summ = summarize_eval(res)
    summ.update({"ckpt": a.ckpt, "split": a.split, "best_epoch": ck.get("epoch")})
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"eval_{a.split}")
    np.savez(out + "_windows.npz", **res, sid=np.array([it["sid"] for it in items]),
             w=np.array([it["w"] for it in items]))
    with open(out + ".json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


def gen_orders(model, items, vmap, device, bs, seed, temperature=1.0, amp=True):
    """List of sampled orders (one per item), deterministic for fixed seed, batches and device."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = [None] * len(items)
    for bidx in batches_by_length(items, bs):
        batch = collate([items[i] for i in bidx], vmap, device, with_order=False)
        with autocast(device, amp):
            o = model.generate(batch, temperature=temperature, generator=g)
        o = o.cpu().numpy()
        for bi, wi in enumerate(bidx):
            out[wi] = o[bi, :len(items[wi]["cont"])].tolist()
    return out


def cmd_gen(a):
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    start_heartbeat(a.heartbeat, f"gen {a.ckpt}")
    model, _ = load_ckpt(a.ckpt, device, a.keep_bias)
    vmap = load_vmap(a.vocab_map)
    items = load_windows(os.path.join(a.data, f"windows_{a.split}.jsonl"))
    allo = []
    t0 = time.time()
    for s in range(a.samples):
        allo.append(gen_orders(model, items, vmap, device, a.bs, a.seed + s, a.temperature, a.amp))
    dt = time.time() - t0
    for o, it in zip(allo[0], items):
        assert sorted(o) == list(range(len(it["cont"])))
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"gen_{a.split}.json")
    with open(out, "w") as f:
        json.dump({"ckpt": a.ckpt, "split": a.split, "seed": a.seed, "samples": a.samples,
                   "temperature": a.temperature, "seconds": dt, "orders": allo}, f)
    print(json.dumps({"out": out, "seconds": round(dt, 2), "texts": len(items) * a.samples}))


# ----------------------------------------------------------------------------------------------------------------
# Generation at scale over the shared July texts
# ----------------------------------------------------------------------------------------------------------------

def cmd_gen_july(a):
    """Orders for the shared July texts (data/humanorder/july_texts_v1), one file per shard.

    Output <out>/shards/orders_shardK.npy: uint16 flat, aligned with cont_tokens_shardK (row i's order occupies
    cont_offsets[i] = [start, len] and is a permutation of 0..len-1: the human-like insertion order of the
    continuation tokens). <out>/shards/orders_shardK.json: seed, timing, sha256, quick statistics. Shard K is sampled
    with torch.Generator seed (seed * 1000 + K), temperature a.temperature.
    """
    from july_texts import JulyTexts
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    start_heartbeat(a.heartbeat, f"gen-july {a.out}")
    model, ck = load_ckpt(a.ckpt, device, a.keep_bias)
    vmap = load_vmap(a.vocab_map)
    jt = JulyTexts(a.texts_dir)
    sd = os.path.join(a.out, "shards")
    os.makedirs(sd, exist_ok=True)
    end = a.shard_end if a.shard_end > 0 else jt.n_shards
    for k in range(a.shard_start, end):
        fo = os.path.join(sd, f"orders_shard{k:06d}.npy")
        if os.path.exists(fo.replace(".npy", ".json")):
            continue
        t0 = time.time()
        p, ct, co, _ = jt.shard(k)
        ct = np.asarray(ct)
        items = [{"prefix": p[i].astype(np.int64).tolist(),
                  "cont": ct[int(co[i, 0]):int(co[i, 0]) + int(co[i, 1])].astype(np.int64).tolist()}
                 for i in range(p.shape[0])]
        orders = gen_orders(model, items, vmap, device, a.bs, a.seed * 1000 + k, a.temperature, a.amp)
        # the trainer skips trajectories longer than max_len actions: redraw those rows (reported, never dropped)
        # rows still too long after a.retries redraws get the left-to-right order when that fits (texts close to the
        # limit, where any jump costs too many move tokens); rows too long even left to right keep their sampled order
        # (the trainer skips that text under every order, including the left-to-right control)
        redrawn, attempts, fallback_l2r, overlong_any_order = 0, 0, 0, 0
        bad = [i for i, o in enumerate(orders) if n_actions(len(items[i]["prefix"]), o) > a.max_len]
        redrawn = len(bad)
        while bad:
            attempts += 1
            if attempts > a.retries:
                for i in bad:
                    l2r = list(range(len(items[i]["cont"])))
                    if n_actions(len(items[i]["prefix"]), l2r) <= a.max_len:
                        orders[i] = l2r
                        fallback_l2r += 1
                    else:
                        overlong_any_order += 1
                print(f"shard {k}: {fallback_l2r} rows set left to right after {a.retries} redraws, "
                      f"{overlong_any_order} rows too long under any order", flush=True)
                break
            new = gen_orders(model, [items[i] for i in bad], vmap, device, a.bs,
                             (a.seed * 1000 + k) * 100 + attempts, a.temperature, a.amp)
            for i, o in zip(bad, new):
                orders[i] = o
            bad = [i for i in bad if n_actions(len(items[i]["prefix"]), orders[i]) > a.max_len]
        flat = np.zeros(len(ct), dtype=np.uint16)
        runs_n = jumps_n = back_n = 0
        for i, o in enumerate(orders):
            s, n = int(co[i, 0]), int(co[i, 1])
            assert sorted(o) == list(range(n)), (k, i)
            flat[s:s + n] = o
            if i < 2000:
                r, j = run_lengths(o)
                runs_n += len(r)
                jumps_n += len(j)
                back_n += sum(x < 0 for x in j)
        np.save(fo, flat)
        import hashlib
        h = hashlib.sha256(open(fo, "rb").read()).hexdigest()
        tok2000 = int(co[:2000, 1].sum())
        meta = {"shard": k, "texts": len(items), "cont_tokens": int(len(ct)), "seed": a.seed * 1000 + k,
                "temperature": a.temperature, "keep_bias": a.keep_bias, "redrawn_overlong": redrawn,
                "redraw_rounds": attempts,
                "fallback_left_to_right": fallback_l2r, "overlong_any_order": overlong_any_order,
                "max_len": a.max_len, "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "seconds": round(time.time() - t0, 2),
                "sha256": h, "first2000": {"jumps_per_100": 100.0 * jumps_n / tok2000,
                                           "back_share": back_n / max(1, jumps_n), "runs": runs_n},
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(fo.replace(".npy", ".json"), "w") as f:
            json.dump(meta, f)
        print(json.dumps(meta), flush=True)


def cmd_calibrate(a):
    """Find the keep-typing bias b (added to the score of the next token at the cursor) that gives a target jump rate
    (jumps per 100 continuation tokens) on one July shard's first n_texts texts. Bisection with the same sampling seed
    at every b; writes the search trace and full statistics (pooled run/jump stats + newline breakdown) at b = 0 and
    at the chosen b."""
    from july_texts import JulyTexts
    from cursor_gate import order_stats
    from july_order_stats import newline_breakdown
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    start_heartbeat(a.heartbeat, f"calibrate {a.ckpt}")
    model, ck = load_ckpt(a.ckpt, device, 0.0)
    vmap = load_vmap(a.vocab_map)
    jt = JulyTexts(a.texts_dir)
    p, ct, co, _ = jt.shard(a.shard)
    ct = np.asarray(ct)
    n = min(a.n_texts, p.shape[0])
    items = [{"prefix": p[i].astype(np.int64).tolist(),
              "cont": ct[int(co[i, 0]):int(co[i, 0]) + int(co[i, 1])].astype(np.int64).tolist()} for i in range(n)]
    tokens = sum(len(it["cont"]) for it in items)

    def run(b):
        model.keep_bias = float(b)
        orders = gen_orders(model, items, vmap, device, a.bs, a.seed, a.temperature, a.amp)
        jumps = sum(len(run_lengths(o)[1]) for o in orders)
        return 100.0 * jumps / tokens, orders

    trace = []
    r0, o0 = run(0.0)
    trace.append({"keep_bias": 0.0, "jumps_per_100": r0})
    lo, hi = a.lo, a.hi
    rhi, _ = run(hi)
    trace.append({"keep_bias": hi, "jumps_per_100": rhi})
    if rhi > a.target:
        raise SystemExit(f"jump rate {rhi:.3f} at keep_bias {hi} still above target {a.target}; raise --hi")
    for _ in range(a.iters):
        mid = 0.5 * (lo + hi)
        rm, _ = run(mid)
        trace.append({"keep_bias": mid, "jumps_per_100": rm})
        print(json.dumps(trace[-1]), flush=True)
        if rm > a.target:
            lo = mid
        else:
            hi = mid
    best = round(0.5 * (lo + hi), 3)
    rb, ob = run(best)
    trace.append({"keep_bias": best, "jumps_per_100": rb})
    stats = {}
    for name, os_ in (("keep_bias_0", o0), ("keep_bias_chosen", ob)):
        st = order_stats(items, os_)
        st.update(newline_breakdown(items, os_))
        stats[name] = st
    rep_ = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "shard": a.shard, "texts": n, "cont_tokens": tokens,
            "seed": a.seed, "temperature": a.temperature, "target_jumps_per_100": a.target, "keep_bias": best,
            "trace": trace, "stats": stats, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out, "w") as f:
        json.dump(rep_, f, indent=1)
    print(json.dumps({k: v for k, v in rep_.items() if k not in ("stats",)}))


def n_actions(prefix_len: int, order) -> int:
    """len(order_to_actions(prefix, cont, order)) without building it: prefix + inserts + move tokens + EOR."""
    _, jumps = run_lengths(order)
    return prefix_len + len(order) + sum(abs(d) // 512 + bin(abs(d) % 512).count("1") for d in jumps) + 1


def cmd_pack(a):
    """Pack per-shard orders into the agreed npz (build_textmatched.py docstring): key order_shard%06d int16 flat
    aligned with cont_tokens_shardK, plus key meta (JSON string)."""
    sd = os.path.join(a.src, "shards")
    files = sorted(glob.glob(os.path.join(sd, "orders_shard*.json")))
    arrs, metas = {}, []
    for fj in files:
        m = json.load(open(fj))
        k = m["shard"]
        if a.max_shard and k >= a.max_shard:
            continue
        o = np.load(fj.replace(".json", ".npy"))
        arrs[f"order_shard{k:06d}"] = o.astype(np.int16)
        metas.append(m)
    meta = {"generator": "cursor_model.py gen-july (N cursor model)", "ckpt": metas[0]["ckpt"] if metas else None,
            "seed_rule": "torch.Generator seed = seed * 1000 + shard", "temperature": metas[0]["temperature"] if metas else None,
            "keep_bias": metas[0].get("keep_bias", 0.0) if metas else None,
            "shards": [m["shard"] for m in metas], "texts": sum(m["texts"] for m in metas),
            "cont_tokens": sum(m["cont_tokens"] for m in metas),
            "redrawn_overlong": sum(m.get("redrawn_overlong", 0) for m in metas),
            "fallback_left_to_right": sum(m.get("fallback_left_to_right", 0) for m in metas),
            "overlong_any_order": sum(m.get("overlong_any_order", 0) for m in metas), "per_shard": metas,
            "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    np.savez(a.out, meta=np.array(json.dumps(meta)), **arrs)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_shard"}))


def cmd_bench(a):
    """Texts/s of batched generation on random-length windows drawn from a split (repeated)."""
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    start_heartbeat(a.heartbeat, "bench")
    model, _ = load_ckpt(a.ckpt, device, a.keep_bias)
    vmap = load_vmap(a.vocab_map)
    items = load_windows(os.path.join(a.data, f"windows_{a.split}.jsonl"))
    items = (items * (a.n // len(items) + 1))[:a.n]
    gen_orders(model, items[:a.bs], vmap, device, a.bs, 0, 1.0, a.amp)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    gen_orders(model, items, vmap, device, a.bs, 0, 1.0, a.amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    toks = sum(len(it["cont"]) for it in items)
    print(json.dumps({"texts": len(items), "bs": a.bs, "seconds": round(dt, 2), "texts_per_s": round(len(items) / dt, 1),
                      "cont_tokens_per_s": round(toks / dt, 1)}))


# ----------------------------------------------------------------------------------------------------------------
# Vocabulary map
# ----------------------------------------------------------------------------------------------------------------

def cmd_vocab(a):
    gv = json.load(open(os.path.join(ROOT, "data/tokenizers/gpt2/vocab.json"), encoding="utf-8"))
    rv = json.load(open(os.path.join(a.reader, "vocab.json"), encoding="utf-8"))
    inv = {i: s for s, i in gv.items()}
    vmap = np.full(len(inv), 3, dtype=np.int32)          # <unk> fallback
    missing = []
    for i in range(len(inv)):
        if inv[i] in rv:
            vmap[i] = rv[inv[i]]
        else:
            missing.append(i)
    rep = {"gpt2_vocab": len(gv), "roberta_vocab": len(rv), "missing": missing,
           "injective": len(set(vmap.tolist())) == len(vmap)}
    used = set()
    for sp in ("train", "dev", "test"):
        for it in load_windows(os.path.join(a.data, f"windows_{sp}.jsonl")):
            used.update(it["prefix"])
            used.update(it["cont"])
    rep["data_ids"] = len(used)
    rep["data_ids_missing"] = sorted(i for i in used if i in set(missing))
    # end-to-end check: GPT-2 tokenization mapped through the table == RoBERTa tokenization of the same text
    from transformers import AutoTokenizer
    gt = AutoTokenizer.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    rt = AutoTokenizer.from_pretrained(a.reader)
    n_ok = n_tot = 0
    bad = []
    for sp in ("dev",):
        for it in load_windows(os.path.join(a.data, f"windows_{sp}.jsonl")):
            text = gt.decode(it["prefix"] + it["cont"])
            g_ids = gt(text, add_special_tokens=False)["input_ids"]
            r_ids = rt(text, add_special_tokens=False)["input_ids"]
            n_tot += 1
            if [int(vmap[t]) for t in g_ids] == r_ids:
                n_ok += 1
            elif len(bad) < 3:
                bad.append(it["sid"])
    rep["roundtrip_texts"] = n_tot
    rep["roundtrip_identical"] = n_ok
    rep["roundtrip_bad_examples"] = bad
    np.save(a.out, vmap)
    rep["out"] = a.out
    with open(a.out.replace(".npy", ".json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "missing"} | {"n_missing": len(missing)}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default=DEFAULT_DATA)
        p.add_argument("--vocab_map", default=DEFAULT_MAP)
        p.add_argument("--device", default="")
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--amp", type=int, default=1)
        p.add_argument("--heartbeat", default="")
        p.add_argument("--keep_bias", type=float, default=0.0,
                       help="calibration: add this to the score of the next token at the cursor (0 = trained model)")

    p = sp.add_parser("vocab")
    p.add_argument("--reader", default=DEFAULT_READER)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=DEFAULT_MAP)
    p.set_defaults(f=cmd_vocab)
    p = sp.add_parser("train")
    common(p)
    p.add_argument("--use_reader", type=int, default=1)
    p.add_argument("--reader", default=DEFAULT_READER)
    p.add_argument("--reader_config", default="", help="JSON RobertaConfig kwargs: random tiny reader (tests)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_reader", type=float, default=3e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log_every", type=int, default=25)
    p.set_defaults(f=cmd_train)
    p = sp.add_parser("eval")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="")
    p.set_defaults(f=cmd_eval)
    p = sp.add_parser("gen")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default="")
    p.set_defaults(f=cmd_gen)
    p = sp.add_parser("gen-july")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--texts_dir", default=os.path.join(ROOT, "data/humanorder/july_texts_v1"))
    p.add_argument("--out", default=os.path.join(ROOT, "data/humanorder/orders_cursor_v1"))
    p.add_argument("--shard_start", type=int, default=0)
    p.add_argument("--shard_end", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=350)
    p.add_argument("--retries", type=int, default=20)
    p.set_defaults(f=cmd_gen_july)
    p = sp.add_parser("calibrate")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--texts_dir", default=os.path.join(ROOT, "data/humanorder/july_texts_v1"))
    p.add_argument("--shard", type=int, default=120)
    p.add_argument("--n_texts", type=int, default=5000)
    p.add_argument("--target", type=float, default=2.157, help="human jumps per 100 tokens (CoAuthor test)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lo", type=float, default=0.0)
    p.add_argument("--hi", type=float, default=4.0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_calibrate)
    p = sp.add_parser("pack")
    p.add_argument("--src", default=os.path.join(ROOT, "data/humanorder/orders_cursor_v1"))
    p.add_argument("--out", default=os.path.join(ROOT, "data/humanorder/orders_cursor_v1.npz"))
    p.add_argument("--max_shard", type=int, default=0)
    p.set_defaults(f=cmd_pack)
    p = sp.add_parser("bench")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=4096)
    p.set_defaults(f=cmd_bench)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
