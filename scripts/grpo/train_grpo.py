#!/usr/bin/env python3
"""Group RL (GRPO-style) for the classical Reviser, from July@84M, on training-text prompts.

Per step: P prompts (the 35-token prefixes of training texts, shards 100-132 by default; never the eval prompts),
G samples each, sampled exactly like the eval decoder (legality mask -> temperature -> top-k with ties -> softmax,
up to --max_new actions, EOR ends; KV-cached, bf16 on CUDA). Reward of a sample (higher is better):
    r = -NLL_reward_lm(continuation | prefix)  -  w_rep * rep4  -  w_short * max(0, 1 - n_new / min_len)
  * NLL_reward_lm: mean per-token NLL of the decoded continuation (canvas[35:], GPT-2 decode) given the decoded prefix,
    under a different, larger LM (default Qwen2.5-3B-Instruct, plain text, no chat template), as score_ppl.py's
    "july" definition does with GPT-2 large. Not the GPT-2 benchmark scorers. Empty continuation -> --empty_reward.
  * rep4: share of repeated 4-grams in the continuation tokens (1 - distinct/total).
  * short: penalty for stopping (EOR) early; the action cap bounds running long.
  * pooled diversity (--w_div > 0): + w_div * novel2, novel2 = (distinct bigrams of the continuation that occur in no
    sample of the previous --div_window steps) / (bigrams of the continuation): the sample's own contribution to a
    pooled distinct-2 (the benchmark's diversity rows pool n-grams over all prompts).
  * cursor guard (--w_move > 0): - w_move * max(0, move_floor - moves_i / new_tokens_i), so samples that move the
    cursor less than the floor lose reward (keeps RL from trading cursor use away for left-to-right writing).
    --move_floor < 0: the floor is the start model's own rate, measured on the first step's samples (before any update).
    --move_metric useful counts only useful moves (scripts/arrl/move_audit.py: a jump with a non-zero net shift
    followed by a kept mid-text insert or a DELETE of a model-written token; fewest move actions for that shift).
    --move_metric gen (main session 2026-10-07 ~11:45Z) counts a move only if it lands the cursor inside the generated
    text: a run of consecutive move actions counts (all its moves) only if, after its last move, no prompt token lies
    at or right of the cursor (= cursor position >= 35 while the prompt is unchanged). Moves into the prompt, and the
    edits made there, earn no move credit. gen_moves_per_new_token is logged for every metric.
  * delete floor (--w_del > 0; move + delete arms E/F): - w_del * max(0, del_floor - deletes_i / new_tokens_i), no
    bonus above the floor. --del_metric effective_own (default) counts only deletes that change the text and remove a
    model-written token (move_audit: deleting X and retyping the same X right away is cancelled; prompt deletes never
    count); effective counts prompt deletes too; all counts every DELETE.
  * prompt kept intact (--w_prompt > 0; main session 2026-10-07 ~11:00Z): - w_prompt * 1[the final canvas does not
    start with the unchanged prompt] (the decoder's prefix_intact, the prompt-intact guard of the report's section 6a).
Prompt lock (--lock_prompt, default off; notes/promptlock-2026-10-08.md): the legality mask also forbids moves that
land left of the prompt end (landing exactly on it stays legal) and DELETE at cursor <= prompt length, in sampling,
forced exploration and the stored allowed sets that policy_terms trains on; the prompt then never changes.
Forced exploration (AR + RL cursor test, --explore_p > 0): at each generated action, with probability
explore_p * max(0, 1 - (step - 1) / (explore_frac * steps)), the sampled action is replaced by a uniformly random
legal move of at most --explore_max_amt; the forced move is added to the stored allowed set, so it is trained on like
any sampled action (no importance weight; a deliberate bias while exploring). --explore_del_p (same fade): with that
probability (disjoint from the forced move draw) the action is replaced by DELETE, only when the token left of the
cursor was written by the model in this sample (never a prompt token). With --explore_del_p 0 the random stream is
the same as before this option existed.
Advantage: (r - group mean) / (group std + 1e-4) (--adv_norm std) or r - group mean (--adv_norm none).
Loss (one on-policy update per batch, so no ratio is needed):
    sum_tokens[-A * log pi(a|s)] + kl_coef * sum_tokens[KL(pi(.|s) || pi_ref(.|s))], divided by the batch's number of
    generated actions. log pi and the KL use the SAME truncated distribution the sample came from: the allowed set
    (legal and in the top-k with ties) is stored at sampling time, the log-softmax is over that set at temperature T
    (fixes the July 4 probe's mismatch between a masked sampler and unmasked log-probs). The KL is exact over that set,
    against the frozen start model restricted to the same set. Dropout 0.

  python scripts/grpo/train_grpo.py --init runs/humanorder/ckpt/july_runjump_step11656_model.pt \
      --reward_lm data/models/Qwen2.5-3B-Instruct --out_dir runs/grpo/g8_kl05 --steps 300 --P 16 --G 8 --kl_coef 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dagger"))
sys.path.insert(0, str(ROOT / "scripts" / "humanorder"))
sys.path.insert(0, str(ROOT / "scripts" / "arrl"))
import hunk_oracle as ho  # noqa: E402
import reviser_lm as rl  # noqa: E402
import move_audit  # noqa: E402

KMAX = 64
AMTS = []
for _a in ho.MOVE_AMOUNTS:
    AMTS += [_a, -_a]


def parse_shards(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def legal_mask(cur, ln, V, device, plen=None):
    """(B, V) bool legality mask from cursor / canvas length tensors (same rule as the eval decoder).

    plen (B,) prompt lengths (--lock_prompt; notes/promptlock-2026-10-08.md): moves must also land at >= plen (exactly
    plen stays legal) and DELETE needs cursor > plen, so the prompt can never change. None = the original rule."""
    amt = torch.tensor(AMTS, device=device)
    m = torch.ones((cur.shape[0], V), dtype=torch.bool, device=device)
    np_ = cur[:, None] + amt[None, :]
    if plen is None:
        m[:, ho.MOVE_START:ho.MOVE_START + ho.N_MOVES] = (np_ >= 0) & (np_ <= ln[:, None])
        m[:, ho.DELETE] = cur > 0
    else:
        plen = plen.to(device)
        m[:, ho.MOVE_START:ho.MOVE_START + ho.N_MOVES] = (np_ >= plen[:, None]) & (np_ <= ln[:, None])
        m[:, ho.DELETE] = cur > plen
    return m


@torch.no_grad()
def generate(model, prefixes, G, max_new, T, top_k, device, gen, explore_p=0.0, explore_max_amt=32, explore_del_p=0.0,
             lock_prompt=False):
    """Returns per-sample dicts (actions, canvas, ...) and tensors idx (B, n, KMAX) int, slot (B, n) long.

    lock_prompt: the legality mask also protects the prompt (legal_mask plen); the stored allowed sets idx then exclude
    the forbidden actions, so policy_terms trains on the same locked distribution the samples came from."""
    ex_ids = torch.tensor([ho.MOVE_START + i for i, amt in enumerate(AMTS) if abs(amt) <= explore_max_amt],
                          device=device)
    n_forced = torch.zeros(len(prefixes) * G, dtype=torch.long)
    n_forced_del = torch.zeros(len(prefixes) * G, dtype=torch.long)
    B = len(prefixes) * G
    states = [ho.State(list(p), len(p), len(p) if lock_prompt else 0) for p in prefixes for _ in range(G)]
    own = [[False] * len(p) for p in prefixes for _ in range(G)]     # canvas token written by the model?
    seqs0 = [[ho.EOI] + list(p) for p in prefixes for _ in range(G)]
    L0 = len(seqs0[0])
    assert all(len(s) == L0 for s in seqs0), "prefixes must have equal length"
    plen = torch.tensor([len(p) for p in prefixes for _ in range(G)], device=device) if lock_prompt else None
    past = None
    logits = None
    for j in range(L0):
        tok = torch.tensor([s[j] for s in seqs0], device=device)
        logits, past = model.forward_step_cached(token_id=tok, canvas_ids_last=None, canvas_mask_last=None,
                                                 past_key_values=past, position_idx=j)
    acts = torch.full((B, max_new), -1, dtype=torch.long)
    idx_all = torch.full((B, max_new, KMAX), -1, dtype=torch.int32)
    slot_all = torch.zeros((B, max_new), dtype=torch.long)
    lp_all = torch.zeros((B, max_new))
    alive = torch.ones(B, dtype=torch.bool)
    tie_overflow = 0
    for t in range(max_new):
        cur = torch.tensor([s.cursor for s in states], device=device)
        ln = torch.tensor([len(s.canvas) for s in states], device=device)
        x = logits.float().masked_fill(~legal_mask(cur, ln, logits.shape[-1], device, plen), float("-inf")) / T
        vals, inds = torch.topk(x, KMAX, dim=-1)
        cutoff = vals[:, top_k - 1:top_k]
        keep = (vals >= cutoff) & torch.isfinite(vals)
        tie_overflow += int((keep[:, -1]).sum())
        force = None
        if explore_p > 0 or explore_del_p > 0:
            lm_ = legal_mask(cur, ln, logits.shape[-1], device, plen)[:, ex_ids]
            u = torch.rand(B, device=device, generator=gen)
            fmove = (u < explore_p) & lm_.any(1)
            force = fmove
            fdel = None
            if explore_del_p > 0:
                own_left = torch.tensor([s_.cursor > 0 and own[b_][s_.cursor - 1] for b_, s_ in enumerate(states)],
                                        device=device)
                fdel = ~fmove & (u >= explore_p) & (u < explore_p + explore_del_p) & own_left
                force = fmove | fdel
            if force.any():
                fm_all = torch.full((B,), ho.DELETE, dtype=torch.long, device=device)
                if fmove.any():
                    fm_all[fmove] = ex_ids[torch.multinomial(lm_[fmove].float(), 1, generator=gen)[:, 0]]
                rows_ = force.nonzero()[:, 0]
                fm = fm_all[rows_]
                hit = (inds[rows_] == fm[:, None]) & keep[rows_]
                miss = ~hit.any(1)
                r2 = rows_[miss]
                inds[r2, KMAX - 1] = fm[miss]
                vals[r2, KMAX - 1] = x[r2, fm[miss]]
                keep[r2, KMAX - 1] = True
        p = torch.softmax(vals.masked_fill(~keep, float("-inf")), dim=-1)
        sl = torch.multinomial(p, 1, generator=gen)[:, 0]
        if force is not None and force.any():
            sl[rows_] = ((inds[rows_] == fm[:, None]) & keep[rows_]).float().argmax(1)
            n_forced += (fmove.cpu() & alive).long()
            if fdel is not None:
                n_forced_del += (fdel.cpu() & alive).long()
        a = inds.gather(1, sl[:, None])[:, 0]
        a_cpu, sl_cpu = a.cpu(), sl.cpu()
        lps_cpu = p.gather(1, sl[:, None])[:, 0].clamp_min(1e-30).log().cpu()
        inds_cpu = inds.masked_fill(~keep, -1).to(torch.int32).cpu()
        acts[:, t] = torch.where(alive, a_cpu, torch.full_like(a_cpu, -1))
        slot_all[:, t] = sl_cpu
        lp_all[:, t] = lps_cpu
        idx_all[:, t] = inds_cpu
        a_list = a_cpu.tolist()
        for b in alive.nonzero()[:, 0].tolist():
            if a_list[b] == ho.EOR:
                alive[b] = False
            else:
                c0 = states[b].cursor
                if a_list[b] == ho.DELETE:
                    if c0 > 0:
                        own[b].pop(c0 - 1)
                elif ho.is_insert(a_list[b]):
                    own[b].insert(c0, True)
                states[b].step(a_list[b])
        if not alive.any():
            acts, idx_all, slot_all, lp_all = acts[:, :t + 1], idx_all[:, :t + 1], slot_all[:, :t + 1], lp_all[:, :t + 1]
            break
        feed = torch.where(alive.to(device), a, torch.full_like(a, ho.EOR))
        logits, past = model.forward_step_cached(token_id=feed, canvas_ids_last=None, canvas_mask_last=None,
                                                 past_key_values=past, position_idx=L0 + t)
    samples = []
    for b in range(B):
        ab = [int(v) for v in acts[b] if v >= 0]
        samples.append({"prompt": b // G, "actions": ab, "canvas": states[b].canvas, "eor": bool(ab and ab[-1] == ho.EOR),
                        "forced": int(n_forced[b]), "forced_del": int(n_forced_del[b])})
    return samples, acts, idx_all, slot_all, lp_all, tie_overflow


def gen_text_moves(prefix, actions):
    """Moves credited by --move_metric gen: each run of consecutive move actions counts in full only if, after its last
    move, no prompt token (one not written by the model) lies at or right of the cursor. Replays the actions with the
    same executor and own-token bookkeeping as generate()."""
    st = ho.State(list(prefix), len(prefix))
    own = [False] * len(prefix)
    n = run = 0
    for x in list(actions) + [None]:
        if x is not None and ho.is_move(x):
            st.step(x)
            run += 1
            continue
        if run:
            if all(own[st.cursor:]):
                n += run
            run = 0
        if x is None or x == ho.EOR:
            break
        c0 = st.cursor
        if x == ho.DELETE:
            if c0 > 0:
                own.pop(c0 - 1)
        elif ho.is_insert(x):
            own.insert(c0, True)
        st.step(x)
    return n


def rep4(toks):
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


class RewardLM:
    def __init__(self, path, device, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        import transformers
        kw = {"dtype": dtype} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": dtype}
        self.lm = AutoModelForCausalLM.from_pretrained(path, **kw).to(device).eval()
        self.device = device

    @torch.no_grad()
    def nll(self, prefix_texts, cont_texts, bs=8, max_len=1024):
        out = [None] * len(prefix_texts)
        rows = []
        for i, (p, c) in enumerate(zip(prefix_texts, cont_texts)):
            if not c.strip():
                continue
            pi, ci = self.tok.encode(p), self.tok.encode(c)
            ids = (pi + ci)[-max_len:]
            rows.append((i, ids, min(len(ci), len(ids) - 1)))
        for s in range(0, len(rows), bs):
            chunk = rows[s:s + bs]
            L = max(len(r[1]) for r in chunk)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, (_, x, _) in enumerate(chunk):
                ids[k, :len(x)] = torch.tensor(x)
                att[k, :len(x)] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            lg = self.lm(input_ids=ids, attention_mask=att).logits       # (b, L, 152k) in bf16: score row by row
            for k, (i, x, nc) in enumerate(chunk):
                n = len(x)
                out[i] = float(F.cross_entropy(lg[k, n - 1 - nc:n - 1].float(), ids[k, n - nc:n]))
            del lg
        return out


def policy_terms(model, ref, ids, att, tgt_pos, acts, idx, slot, T, amp, kl_on, full=False):
    """Per generated position: log pi(a|s) and KL(pi || ref) over the stored allowed set. Shapes (B, n)."""
    with torch.autocast(device_type=ids.device.type, dtype=amp or torch.float32, enabled=amp is not None):
        logits = model(input_ids=ids, attention_mask=att)["logits"]
    B, n = acts.shape
    rows = torch.arange(B, device=ids.device)[:, None]
    lg = logits[rows, tgt_pos]                                         # (B, n, V)
    safe = idx.clamp_min(0).long()
    valid = idx >= 0
    g = (lg.gather(-1, safe).float() / T).masked_fill(~valid, -1e4)   # finite: no 0 * inf in the KL backward
    lsm = torch.log_softmax(g, -1)
    lp = lsm.gather(-1, slot[..., None])[..., 0]
    if full:   # best-of-group mode: plain full-vocabulary cross-entropy, as in supervised training
        lf = lg.float()
        lp = lf.gather(-1, acts.clamp_min(0)[..., None])[..., 0] - torch.logsumexp(lf, -1)
        del lf
    kl = torch.zeros_like(lp)
    if kl_on:
        with torch.no_grad(), torch.autocast(device_type=ids.device.type, dtype=amp or torch.float32,
                                             enabled=amp is not None):
            rlg = ref(input_ids=ids, attention_mask=att)["logits"][rows, tgt_pos]
        rg = (rlg.gather(-1, safe).float() / T).masked_fill(~valid, -1e4)
        rlsm = torch.log_softmax(rg, -1)
        p = lsm.exp()
        kl = (p * (lsm - rlsm)).masked_fill(~valid, 0.0).sum(-1)
    return lp, kl


def reward_var_shares(samples, P, G):
    """Within-group reward variance and the part each reward term carries (main session request, 2026-10-07 09:45Z).

    Pooled over the P groups of one step (samples are contiguous per group, as in Rt.view(P, G)):
      varshare_<term> = sum_g Var_g(term) / sum_g Var_g(reward)
      covshare_<term> = sum_g Cov_g(term, reward) / sum_g Var_g(reward)   (the covshares of all terms add up to 1)
    plus mv_tok_std_in_group / del_tok_std_in_group (mean within-group std of the guard metrics).
    A term whose covshare is near 0 cannot steer the update, whatever its mean size.
    """
    out, den = {}, 0.0
    names = list(samples[0]["r_terms"])
    num = {k: 0.0 for k in names}
    cov = {k: 0.0 for k in names}
    sd = {"mv_tok": 0.0, "del_tok": 0.0}
    for g in range(P):
        grp = samples[g * G:(g + 1) * G]
        r = [x["reward"] for x in grp]
        rm = sum(r) / len(r)
        den += sum((v - rm) ** 2 for v in r)
        for k in names:
            t = [x["r_terms"][k] for x in grp]
            tm = sum(t) / len(t)
            num[k] += sum((v - tm) ** 2 for v in t)
            cov[k] += sum((v - tm) * (w - rm) for v, w in zip(t, r))
        for k in sd:
            t = [x[k] for x in grp]
            tm = sum(t) / len(t)
            sd[k] += (sum((v - tm) ** 2 for v in t) / max(1, len(t) - 1)) ** 0.5 / P
    for k in names:
        out[f"varshare_{k}"] = num[k] / den if den > 0 else None
        out[f"covshare_{k}"] = cov[k] / den if den > 0 else None
    out["mv_tok_std_in_group"] = sd["mv_tok"]
    out["del_tok_std_in_group"] = sd["del_tok"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--reward_lm", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--P", type=int, default=16, help="prompts per step")
    ap.add_argument("--G", type=int, default=8, help="samples per prompt")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=221, help="C4 paper protocol: 221 new actions incl. EOR")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--w_rep", type=float, default=2.0)
    ap.add_argument("--w_short", type=float, default=1.0)
    ap.add_argument("--min_len", type=int, default=150)
    ap.add_argument("--empty_reward", type=float, default=-10.0)
    ap.add_argument("--w_div", type=float, default=0.0, help="pooled-diversity bonus weight (0 = off)")
    ap.add_argument("--div_window", type=int, default=20, help="steps of past samples in the bigram pool")
    ap.add_argument("--w_move", type=float, default=0.0, help="cursor guard weight (0 = off)")
    ap.add_argument("--move_floor", type=float, default=-1.0,
                    help="moves per new token below which the guard penalises; < 0: start model's step-1 rate")
    ap.add_argument("--move_metric", choices=["all", "useful", "gen"], default="all")
    ap.add_argument("--explore_p", type=float, default=0.0, help="forced random move probability per action (start)")
    ap.add_argument("--explore_frac", type=float, default=0.3, help="share of steps over which explore_p fades to 0")
    ap.add_argument("--explore_max_amt", type=int, default=32)
    ap.add_argument("--explore_del_p", type=float, default=0.0, help="forced DELETE probability per action (start)")
    ap.add_argument("--w_del", type=float, default=0.0, help="delete floor weight (0 = off)")
    ap.add_argument("--del_floor", type=float, default=0.01, help="deletes per new token below which w_del penalises")
    ap.add_argument("--del_metric", choices=["effective_own", "effective", "all"], default="effective_own")
    ap.add_argument("--w_prompt", type=float, default=0.0, help="penalty for a changed prompt (0 = off)")
    ap.add_argument("--lock_prompt", action="store_true",
                    help="legality mask forbids moves left of the prompt end and DELETE at cursor <= prompt length, "
                         "so the prompt never changes (sampling, forced exploration and the trained allowed sets); "
                         "--w_prompt is then redundant (prefix_changed is always 0)")
    ap.add_argument("--adv_norm", choices=["std", "none"], default="std")
    ap.add_argument("--mode", choices=["grpo", "best"], default="grpo",
                    help="best: supervised training on the top sample of each group (full-vocab CE)")
    ap.add_argument("--shards", default="100-132")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpt2_tok", default="gpt2")
    ap.add_argument("--save_at", default="")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--sample_every", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_minutes", type=float, default=0)
    a = ap.parse_args()

    from transformers import GPT2TokenizerFast
    from july_texts import JulyTexts
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = rl.autocast_dtype(device)
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed + 17)
    gtok = GPT2TokenizerFast.from_pretrained(a.gpt2_tok)

    model = rl.load_model(a.init, device, dropout=0.0).train()
    ref = rl.load_model(a.init, device, dropout=0.0).eval() if a.kl_coef > 0 else None
    if ref is not None:
        for p in ref.parameters():
            p.requires_grad_(False)
    rlm = RewardLM(a.reward_lm, device, torch.bfloat16 if device.type == "cuda" else torch.float32)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)))

    jt = JulyTexts()
    keys = [(k, i) for k in parse_shards(a.shards) for i in range(jt.shard_sizes[k])]
    random.Random(a.seed).shuffle(keys)
    keys = keys[:a.steps * a.P]
    prov = {"args": vars(a), "init_sha256": rl.sha256(a.init), "device": str(device), "amp": str(amp),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__,
            "first_keys": keys[:5], "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files_sha256": {"train_grpo.py": rl.sha256(Path(__file__)),
                             "reviser_lm.py": rl.sha256(ROOT / "scripts/dagger/reviser_lm.py"),
                             "hunk_oracle.py": rl.sha256(ROOT / "scripts/dagger/hunk_oracle.py")}}
    rev = Path(a.reward_lm) / "REVISION.txt"
    if rev.exists():
        prov["reward_lm_revision"] = rev.read_text().strip()
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    smp = open(out / "samples.jsonl", "a", encoding="utf-8")
    save_at = {int(x) for x in a.save_at.split(",") if x}
    t0 = time.time()
    n_trained = 0
    step = 0
    move_floor = None
    div_pool, div_hist = Counter(), deque()
    for step in range(1, a.steps + 1):
        ks = keys[(step - 1) * a.P: step * a.P]
        prefixes = [jt.text(k, i)[0] for k, i in ks]
        tg = time.time()
        ex_p = a.explore_p * max(0.0, 1.0 - (step - 1) / max(1.0, a.explore_frac * a.steps))
        samples, acts, idx, slot, lp_samp, tie_over = generate(model, prefixes, a.G, a.max_new, a.temperature, a.top_k, device,
                                                      gen, ex_p, a.explore_max_amt,
                                                      a.explore_del_p * max(0.0, 1.0 - (step - 1) / max(1.0, a.explore_frac * a.steps)),
                                                      lock_prompt=a.lock_prompt)
        model.train()
        tr = time.time()
        ptxt = [gtok.decode(prefixes[s["prompt"]]) for s in samples]
        ctxt = [gtok.decode(s["canvas"][len(prefixes[s["prompt"]]):]) for s in samples]
        nll = rlm.nll(ptxt, ctxt)
        R = []
        for s in samples:
            s["n_move"] = sum(1 for x in s["actions"] if ho.is_move(x))
            s["n_back"] = sum(1 for x in s["actions"] if ho.is_move(x) and ho.move_amount(x) < 0)
            s["audit"] = move_audit.audit(prefixes[s["prompt"]], s["actions"])
            s["n_gen_move"] = gen_text_moves(prefixes[s["prompt"]], s["actions"])
            if a.move_metric == "useful":
                s["n_move_metric"] = s["audit"]["useful_moves"]
            elif a.move_metric == "gen":
                s["n_move_metric"] = s["n_gen_move"]
            else:
                s["n_move_metric"] = s["n_move"]
        if move_floor is None:
            move_floor = a.move_floor if a.move_floor >= 0 else (
                sum(s["n_move_metric"] for s in samples) / max(1, sum(len(s["canvas"]) - len(prefixes[s["prompt"]])
                                                                for s in samples)))
            prov["move_floor"] = move_floor
            (out / "provenance.json").write_text(json.dumps(prov, indent=1))
        for s, v in zip(samples, nll):
            cont = s["canvas"][len(prefixes[s["prompt"]]):]
            s["n_new"] = len(cont)
            s["mv_tok"] = s["n_move_metric"] / max(1, len(cont))
            s["move_pen"] = a.w_move * max(0.0, move_floor - s["mv_tok"])
            au = s["audit"]
            s["n_del_metric"] = {"effective_own": au["deletes_effective_own"], "effective": au["deletes_effective"],
                                 "all": au["deletes"]}[a.del_metric]
            s["del_tok"] = s["n_del_metric"] / max(1, len(cont))
            s["del_pen"] = a.w_del * max(0.0, a.del_floor - s["del_tok"]) if a.w_del > 0 else 0.0
            s["rep4"] = rep4(cont)
            big = set(zip(cont, cont[1:]))
            s["novel2"] = sum(1 for g in big if g not in div_pool) / max(1, len(cont) - 1)
            s["nll"] = v
            s["prefix_changed"] = s["canvas"][:len(prefixes[s["prompt"]])] != list(prefixes[s["prompt"]])
            base = -v if v is not None else a.empty_reward
            r = base - a.w_rep * s["rep4"] - a.w_short * max(0.0, 1.0 - len(cont) / a.min_len) - s["move_pen"] - s["del_pen"]
            r += a.w_div * s["novel2"]
            r -= a.w_prompt * float(s["prefix_changed"])
            s["reward"] = r
            s["r_terms"] = {"quality": base, "rep": -a.w_rep * s["rep4"],
                            "short": -a.w_short * max(0.0, 1.0 - len(cont) / a.min_len), "move": -s["move_pen"],
                            "del": -s["del_pen"], "div": a.w_div * s["novel2"],
                            "prompt": -a.w_prompt * float(s["prefix_changed"])}
            R.append(r)
        step_big = Counter()
        for s in samples:
            cont = s["canvas"][len(prefixes[s["prompt"]]):]
            step_big.update(set(zip(cont, cont[1:])))
        div_hist.append(step_big)
        div_pool.update(step_big)
        if len(div_hist) > a.div_window:
            div_pool.subtract(div_hist.popleft())
            div_pool = +div_pool
        Rt = torch.tensor(R).view(a.P, a.G)
        adv = Rt - Rt.mean(1, keepdim=True)
        if a.adv_norm == "std":
            adv = adv / (Rt.std(1, keepdim=True) + 1e-4)
        adv = adv.view(-1)
        if a.mode == "best":   # best-of-group fine-tuning: weight 1 on each group's top sample, 0 elsewhere
            adv = torch.zeros(a.P, a.G)
            adv[torch.arange(a.P), Rt.argmax(1)] = 1.0
            adv = adv.view(-1)
        tt = time.time()
        # training forward: input = [EOI] + prefix + actions[:-1]; action t is predicted at position L0 - 1 + t
        L0 = len(prefixes[0]) + 1
        n_act = (acts >= 0).sum(1)
        n_tot = int(n_act.sum())
        n_best = max(1, int((n_act * (adv > 0)).sum())) if a.mode == "best" else n_tot
        tot = {"pg": 0.0, "kl": 0.0, "lp": 0.0, "lp_absdiff": 0.0}
        for i in range(0, len(samples), a.micro):
            sl = slice(i, i + a.micro)
            seqs = [[ho.EOI] + list(prefixes[s["prompt"]]) + s["actions"][:-1] if s["actions"] else
                    [ho.EOI] + list(prefixes[s["prompt"]]) for s in samples[sl]]
            ids, att = rl.pad_batch(seqs, device)
            n = int(n_act[sl].max())
            pos = (L0 - 1 + torch.arange(n, device=device))[None, :].expand(ids.shape[0], n)
            pos = pos.clamp_max(ids.shape[1] - 1)
            m = (acts[sl, :n] >= 0).to(device)
            lp, kl = policy_terms(model, ref, ids, att, pos, acts[sl, :n].to(device), idx[sl, :n].to(device),
                                  slot[sl, :n].to(device), a.temperature, amp, a.kl_coef > 0, full=a.mode == "best")
            A = adv[sl].to(device)[:, None]
            pg = -(A * lp).masked_fill(~m, 0.0).sum()
            klsum = kl.masked_fill(~m, 0.0).sum()
            loss = (pg + a.kl_coef * klsum) / (n_best if a.mode == "best" else n_tot)
            loss.backward()
            tot["pg"] += float(pg)
            tot["kl"] += float(klsum)
            tot["lp"] += float(lp.masked_fill(~m, 0.0).sum())
            tot["lp_absdiff"] += float((lp.detach() - lp_samp[sl, :n].to(device)).abs().masked_fill(~m, 0.0).sum())
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        n_trained += n_tot
        kinds = {"insert": 0, "move": 0, "back_move": 0, "delete": 0, "eor": 0}
        for s in samples:
            for x in s["actions"]:
                k = rl.action_kind(x)
                kinds[k] = kinds.get(k, 0) + 1
                if k == "move" and ho.move_amount(x) < 0:
                    kinds["back_move"] += 1
        if step % a.log_every == 0:
            nv = [s["nll"] for s in samples if s["nll"] is not None]
            vs = reward_var_shares(samples, a.P, a.G)
            rec = {"step": step, "secs": round(time.time() - t0, 1), "gen_s": round(tr - tg, 1),
                   "reward_s": round(tt - tr, 1), "train_s": round(time.time() - tt, 1), "lr": sched.get_last_lr()[0],
                   "reward_mean": float(Rt.mean()), "reward_std_in_group": float(Rt.std(1).mean()),
                   "rlm_nll_mean": sum(nv) / max(1, len(nv)), "rep4_mean": sum(s["rep4"] for s in samples) / len(samples),
                   "novel2_mean": sum(s["novel2"] for s in samples) / len(samples),
                   "n_new_mean": sum(s["n_new"] for s in samples) / len(samples),
                   "moves_per_new_token": sum(s["n_move"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   "back_moves_per_new_token": sum(s["n_back"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   "gen_moves_per_new_token": sum(s["n_gen_move"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   "metric_moves_per_new_token": sum(s["n_move_metric"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   "move_floor": move_floor, "move_pen_mean": sum(s["move_pen"] for s in samples) / len(samples),
                   "explore_p": ex_p, "forced_per_new_token": sum(s["forced"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   **{f"audit_{k}": v for k, v in move_audit.summarize([s["audit"] for s in samples]).items()
                      if k in ("useful_moves_per_new_token", "share_moves_cancelling", "share_moves_dead_end",
                               "share_moves_then_end_insert", "share_moves_then_kept_mid_edit",
                               "mid_inserts_per_new_token", "deletes_per_new_token", "effective_deletes_per_new_token",
                               "effective_own_deletes_per_new_token", "share_deletes_effective", "share_deletes_own",
                               "share_deletes_prompt", "samples_with_deletes")},
                   "del_metric_per_new_token": sum(s["n_del_metric"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   "del_pen_mean": sum(s["del_pen"] for s in samples) / len(samples),
                   "below_del_floor_share": sum(s["del_tok"] < a.del_floor for s in samples) / len(samples),
                   "forced_del_per_new_token": sum(s["forced_del"] for s in samples) / max(1, sum(s["n_new"] for s in samples)),
                   "below_floor_share": sum(s["mv_tok"] < move_floor for s in samples) / len(samples),
                   "stop_rate": sum(s["eor"] for s in samples) / len(samples),
                   "prefix_changed": sum(s["prefix_changed"] for s in samples) / len(samples),
                   "kl_per_action": tot["kl"] / n_tot, "logp_per_action": tot["lp"] / n_tot, "logp_train_vs_sampler_absdiff": tot["lp_absdiff"] / n_tot, "grad_norm": gn,
                   "actions": n_tot, "tie_overflow": tie_over, **vs,
                   **{f"share_{k}": v / max(1, n_tot) for k, v in kinds.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % a.sample_every == 0 or step == 1:
            for s, c in list(zip(samples, ctxt))[:a.G]:
                smp.write(json.dumps({"step": step, "reward": s["reward"], "nll": s["nll"], "rep4": s["rep4"],
                                      "n_new": s["n_new"], "eor": s["eor"], "prefix": ptxt[0], "cont": c}) + "\n")
            smp.flush()
        if step in save_at:
            rl.save_model(model, out / f"model_step{step:06d}.pt", state={"step": step}, source=str(a.init))
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print(f"max_minutes reached at step {step}", flush=True)
            break
    final = out / f"model_step{step:06d}.pt"
    rl.save_model(model, final, state={"step": step, "actions_trained": n_trained}, source=str(a.init))
    done = {"final": str(final), "final_sha256": rl.sha256(final), "steps": step, "actions_trained": n_trained,
            "secs": round(time.time() - t0, 1), "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "done.json").write_text(json.dumps(done, indent=1))
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
