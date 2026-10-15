#!/usr/bin/env python3
"""What do the self-correction models delete in free writing? (helper D, 2026-10-11; notes/delete-behaviour-2026-10-11.md)

Input: the saved C4 continuation decodes of scripts/papereval/decode_batched.py (3,000 prompts, 35-token prompt,
T 0.9, top-k 50, <= 220 new actions; prompt-locked for the Reviser models, not locked for AR). Each record has
prefix_ids, actions_new and canvas_ids.

Stage "replay" (CPU; also run inside the other stages): replays actions_new from the prompt with the decoder's
executor semantics (reviser.decoding.inference_core.CursorState.execute_action: a move shifts the cursor and must
stay in [0, len]; DELETE removes the token left of the cursor and needs cursor > 0; an insert is any id in
[0, 50257) except 50256 and goes at the cursor; any other id, e.g. EOR, changes nothing) and asserts that the replay
gives canvas_ids for every record. Each canvas token carries a tag: -1 for prompt tokens, else the index of the
action that inserted it. For every DELETE it records the deleted token, its tag/age, whether it repeats a nearby
token, what the next action is (replacement token when the next action is an insert, which lands at the same spot).

Stage "score" (GPU if given): GPT-2 small next-token probability of each deleted token and of its replacement given
the canvas left of it at deletion time (same left context for both), and of every new token of every final text
given its left context in the final canvas (baseline for "typical kept token"). Saved per model as JSON.

Stage "report": all statistics, paired bootstrap CIs over prompts (1,000 resamples), JSON + markdown tables.

  python scripts/deleteprobe/free_deletes.py --stage score --gpt2 <dir> --out_dir runs/eval/deleteprobe/free
  python scripts/deleteprobe/free_deletes.py --stage report --gpt2 <dir> --out_dir runs/eval/deleteprobe/free
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import string
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 35
GPT2_VOCAB, GPT2_EOT = 50257, 50256
EOI, DELETE, EOR, MOVE_START = 50258, 50259, 50260, 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

DECODES = {
    "july608": "runs/papereval/dec/c4paper3k.lock.july608.jsonl",
    "jd608": "runs/papereval/dec/c4paper3k.lock.jd608.jsonl",
    "jd608_g8m2_s1": "runs/papereval/dec/c4paper3k.lock.jd608_g8m2_s1.jsonl",
    "jd608_g8m2_s2": "runs/papereval/dec/c4paper3k.lock.jd608_g8m2_s2.jsonl",
    "jd608_g8m2_s3": "runs/papereval/dec/c4paper3k.lock.jd608_g8m2_s3.jsonl",
    "rev100": "runs/papereval/dec/c4paper3k.lock.rev100.jsonl",
    "ar100": "runs/papereval/dec/c4paper3k.nolock.ar100.jsonl",
    "rev300": "runs/big608_300m/dec/c4paper3k.lock.rev300.jsonl",
    "jd608_300m": "runs/big608_300m/dec/c4paper3k.lock.jd608_300m.jsonl",
    "jd608_300m_g8m2_s3": "runs/big608_300m/dec/c4paper3k.lock.jd608_300m_g8m2_s3.jsonl",
    "ar300": "runs/big608_300m/dec/c4paper3k.nolock.ar300.jsonl",
    # training-recipe reference (scripts/deleteprobe/train_ref.py; shard-133 texts, not the C4 prompts: never paired)
    "train_ref": "runs/eval/deleteprobe/free/train_ref.jsonl",
}
PAIRS = [("jd608", "july608"), ("jd608_g8m2_s1", "jd608"), ("jd608_g8m2_s2", "jd608"), ("jd608_g8m2_s3", "jd608"),
         ("jd608", "rev100"), ("jd608_300m", "rev300"), ("jd608_300m_g8m2_s3", "jd608_300m"),
         ("jd608_300m", "jd608")]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def is_move(t):
    return MOVE_START <= t < MOVE_START + 20


def move_amount(t):
    off = t - MOVE_START
    a = MOVE_AMOUNTS[off // 2]
    return -a if off % 2 else a


def is_insert(t):
    return 0 <= t < GPT2_VOCAB and t != GPT2_EOT


def kind(t):
    if t == EOR:
        return "eor"
    if t == DELETE:
        return "delete"
    if is_move(t):
        return "move"
    if is_insert(t):
        return "insert"
    return "other"


def load(path):
    with open(ROOT / path, encoding="utf-8") as f:
        meta = json.loads(f.readline())["_meta"]
        rows = [json.loads(l) for l in f]
    return meta, rows


def replay(rec):
    """Tagged replay of one record (CursorState semantics). Returns (final tokens, final tags, delete events)."""
    prefix = [int(x) for x in rec["prefix_ids"]]
    toks, tags = list(prefix), [-1] * len(prefix)
    cursor = len(prefix)
    events = []
    acts = [int(a) for a in rec["actions_new"]]
    for s, a in enumerate(acts):
        if a == EOR:
            break
        if is_move(a):
            nc = cursor + move_amount(a)
            assert 0 <= nc <= len(toks), "illegal move in a saved decode"
            cursor = nc
        elif a == DELETE:
            assert cursor > 0, "illegal DELETE in a saved decode"
            pos = cursor - 1
            tok, tag = toks[pos], tags[pos]
            nxt = acts[s + 1] if s + 1 < len(acts) else None
            ev = {"s": s, "pos": pos, "tok": tok, "tag": tag, "age": (s - tag) if tag >= 0 else None,
                  "prev_kind": kind(acts[s - 1]) if s > 0 else "prompt",
                  "just_typed": s > 0 and tag == s - 1,
                  "rep1": pos > 0 and toks[pos - 1] == tok,
                  "rep3": tok in toks[max(0, pos - 3):pos],
                  "near4": tok in toks[max(0, pos - 4):pos] or tok in toks[pos + 1:pos + 5],
                  "right_len": len(toks) - cursor,
                  "next_kind": kind(nxt) if nxt is not None else "end",
                  "repl": nxt if (nxt is not None and is_insert(nxt)) else None,
                  "ctx": toks[:pos], "word_ctx": toks[max(0, pos - 12):pos + 1]}
            ev["repl_same"] = ev["repl"] is not None and ev["repl"] == tok
            events.append(ev)
            toks.pop(pos)
            tags.pop(pos)
            cursor -= 1
        elif is_insert(a):
            toks.insert(cursor, a)
            tags.insert(cursor, s)
            cursor += 1
        # else: EOT / other ids change nothing (CursorState.execute_action appends them to the history only)
    assert toks == [int(x) for x in rec["canvas_ids"]], f"replay mismatch for record {rec['i']}"
    return toks, tags, events


# ------------------------------------------------------------------------------------------------ text stats
_PUNCT = string.punctuation + "“”‘’—–…"


def norm_words(text):
    out = []
    for w in text.split():
        w = w.strip(_PUNCT).lower()
        if w:
            out.append(w)
    return out


def text_counts(new, tok):
    """(adjacent dup tokens, token pairs, adjacent dup words, word pairs, repeated 4-grams, 4-grams)."""
    dt = sum(1 for i in range(1, len(new)) if new[i] == new[i - 1])
    pt = max(0, len(new) - 1)
    ws = norm_words(tok.decode(new)) if new else []
    dw = sum(1 for i in range(1, len(ws)) if ws[i] == ws[i - 1])
    pw = max(0, len(ws) - 1)
    g = [tuple(new[i:i + 4]) for i in range(len(new) - 3)]
    return dt, pt, dw, pw, len(g) - len(set(g)), len(g)


def word_repeat(ev, tok):
    """The word that ends with the deleted token equals the word before it (lowercase, punctuation stripped)."""
    ws = norm_words(tok.decode(ev["word_ctx"]))
    return len(ws) >= 2 and ws[-1] == ws[-2] and tok.decode([ev["tok"]]).strip(_PUNCT + " \n\t") != ""


# ------------------------------------------------------------------------------------------------ GPT-2 scoring
def gpt2_probs(lm, contexts, targets, device, bs=64):
    """P_gpt2(target | context) for each pair (contexts are GPT-2 id lists; last 1023 kept)."""
    import torch
    out = np.zeros(len(contexts), dtype=np.float64)
    order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        cs = [contexts[i][-1023:] for i in idx]
        L = max(len(c) for c in cs)
        ids = torch.full((len(cs), L), GPT2_EOT, dtype=torch.long)
        for k, c in enumerate(cs):
            ids[k, :len(c)] = torch.tensor(c)
        last = torch.tensor([len(c) - 1 for c in cs], device=device)
        with torch.no_grad():        # lm head on the last real position only (right padding; causal, so inert)
            h = lm.transformer(input_ids=ids.to(device)).last_hidden_state
            lg = lm.lm_head(h[torch.arange(len(cs), device=device), last]).float()
        lp = torch.log_softmax(lg, -1)
        tg = torch.tensor([targets[i] for i in idx], device=device)
        v = lp.gather(1, tg[:, None])[:, 0].exp().double().cpu().numpy()
        out[idx] = v
    return out


def gpt2_text_logprobs(lm, texts, device, bs=8):
    """Per-text arrays of log P_gpt2(token_j | tokens_<j) for j >= PREFIX (new tokens), whole canvas as context."""
    import torch
    res = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        cs = [texts[i][:1024] for i in idx]
        L = max(len(c) for c in cs)
        ids = torch.full((len(cs), L), GPT2_EOT, dtype=torch.long)
        for k, c in enumerate(cs):
            ids[k, :len(c)] = torch.tensor(c)
        ids = ids.to(device)
        with torch.no_grad():
            lp = torch.log_softmax(lm(input_ids=ids).logits.float(), -1)
        g = lp[:, :-1].gather(2, ids[:, 1:, None])[:, :, 0].cpu().numpy()
        for k, i in enumerate(idx):
            n = len(cs[k])
            res[i] = g[k, PREFIX - 1:n - 1].astype(np.float64)       # predictions of tokens PREFIX..n-1
    return res


# ------------------------------------------------------------------------------------------------ bootstrap
def boot_idx(n, B=1000, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def ratio_ci(num, den, I):
    est = num.sum() / max(den.sum(), 1e-12)
    bs = num[I].sum(1) / np.maximum(den[I].sum(1), 1e-12)
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ratio_diff_ci(na, da, nb, db, I):
    est = na.sum() / max(da.sum(), 1e-12) - nb.sum() / max(db.sum(), 1e-12)
    bs = na[I].sum(1) / np.maximum(da[I].sum(1), 1e-12) - nb[I].sum(1) / np.maximum(db[I].sum(1), 1e-12)
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "score", "report"], required=True)
    ap.add_argument("--gpt2", required=True, help="GPT-2 small directory (tokenizer; model for --stage score)")
    ap.add_argument("--out_dir", default="runs/eval/deleteprobe/free")
    ap.add_argument("--models", default=",".join(DECODES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N records per model (smoke tests only)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.gpt2)
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in a.models.split(",") if n]
    t0 = time.time()
    data = {}
    for n in names:
        meta, rows = load(DECODES[n])
        assert len(rows) == 3000 and [r["i"] for r in rows] == list(range(3000)), n
        rows = rows[:a.limit] if a.limit else rows
        rep = [replay(r) for r in rows]
        data[n] = (meta, rows, rep)
        nd = sum(len(e) for _, _, e in rep)
        print(json.dumps({"model": n, "replay_ok": len(rows), "deletes": nd, "secs": round(time.time() - t0, 1)}),
              flush=True)
    if a.stage == "replay":
        return

    if a.stage == "score":
        import torch
        from transformers import GPT2LMHeadModel
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        lm = GPT2LMHeadModel.from_pretrained(a.gpt2).to(dev).eval()
        for n in names:
            meta, rows, rep = data[n]
            t1 = time.time()
            evs = [(ri, e) for ri, (_, _, es) in enumerate(rep) for e in es]
            ctx = [e["ctx"] for _, e in evs]
            p_del = gpt2_probs(lm, ctx, [e["tok"] for _, e in evs], dev) if evs else np.zeros(0)
            wr = [(k, e) for k, (_, e) in enumerate(evs) if e["repl"] is not None]
            p_rep = np.full(len(evs), np.nan)
            if wr:
                p_rep[[k for k, _ in wr]] = gpt2_probs(lm, [e["ctx"] for _, e in wr], [e["repl"] for _, e in wr], dev)
            lps = gpt2_text_logprobs(lm, [t for t, _, _ in rep], dev)
            res = {"model": n, "gpt2": a.gpt2, "gpt2_safetensors_sha256": sha256(Path(a.gpt2) / "model.safetensors"),
                   "device": str(dev), "events": [[ri, e["s"], float(p_del[k]),
                                                   None if math.isnan(p_rep[k]) else float(p_rep[k])]
                                                  for k, (ri, e) in enumerate(evs)],
                   "text_logp": [[round(float(x), 5) for x in v] for v in lps], "secs": round(time.time() - t1, 1)}
            (out / f"{n}.gpt2.json").write_text(json.dumps(res))
            print(json.dumps({"model": n, "scored_deletes": len(evs), "with_repl": len(wr), "secs": res["secs"]}),
                  flush=True)
        return

    # ---------------------------------------------------------------- report
    N = a.limit or 3000
    I = boot_idx(N, a.boot, a.boot_seed)
    R = {"decodes": {n: {"path": DECODES[n], "sha256": sha256(ROOT / DECODES[n]), "decoder": data[n][0]["decoder"],
                         "decoder_sha256": data[n][0]["decoder_sha256"], "ckpt_sha256": data[n][0]["ckpt_sha256"],
                         "lock_prompt": data[n][0]["args"]["lock_prompt"]} for n in names},
         "script_sha256": sha256(__file__), "boot": a.boot, "boot_seed": a.boot_seed, "models": {}, "pairs": {}}
    per = {}
    for n in names:
        meta, rows, rep = data[n]
        new = [t[PREFIX:] for t, _, _ in rep]
        nd = np.array([len(es) for _, _, es in rep], dtype=float)
        nt = np.array([len(x) for x in new], dtype=float)
        na = np.array([len([x for x in r["actions_new"] if x != EOR]) for r in rows], dtype=float)
        tc = np.array([text_counts(x, tok) for x in new], dtype=float)          # (N, 6)
        m = {"n_prompts": N, "deletes": int(nd.sum()), "new_tokens_mean": float(nt.mean()),
             "deletes_per_new_token": ratio_ci(nd, nt, I), "deletes_per_new_action": ratio_ci(nd, na, I),
             "outputs_with_delete": ratio_ci((nd > 0).astype(float), np.ones(N), I),
             "prompt_intact_all": all(t[:PREFIX] == [int(x) for x in r["prefix_ids"]] for (t, _, _), r in
                                      zip(rep, rows)),
             "adj_dup_token": ratio_ci(tc[:, 0], tc[:, 1], I), "adj_dup_word": ratio_ci(tc[:, 2], tc[:, 3], I),
             "rep4": ratio_ci(tc[:, 4], tc[:, 5], I)}
        evs = [(ri, e) for ri, (_, _, es) in enumerate(rep) for e in es]
        if evs:
            ri_arr = np.array([ri for ri, _ in evs])
            # per-prompt sums for cluster bootstrap of event shares
            def share(pred):
                num = np.bincount(ri_arr, weights=np.array([float(pred(e)) for _, e in evs]), minlength=N)
                den = np.bincount(ri_arr, minlength=N).astype(float)
                return ratio_ci(num, den, I)
            ages = [e["age"] for _, e in evs if e["age"] is not None]
            m["events"] = {
                "n": len(evs),
                "prompt_token_deleted": int(sum(e["tag"] < 0 for _, e in evs)),
                "just_typed": share(lambda e: e["just_typed"]),
                "age_le_2": share(lambda e: e["age"] is not None and e["age"] <= 2),
                "age_gt_5": share(lambda e: e["age"] is not None and e["age"] > 5),
                "age_median": float(np.median(ages)) if ages else None,
                "age_mean": float(np.mean(ages)) if ages else None,
                "cursor_at_end": share(lambda e: e["right_len"] == 0),
                "rep1": share(lambda e: e["rep1"]), "rep3": share(lambda e: e["rep3"]),
                "near4": share(lambda e: e["near4"]),
                "word_repeat": share(lambda e: word_repeat(e, tok)),
                "next_insert": share(lambda e: e["next_kind"] == "insert"),
                "next_delete": share(lambda e: e["next_kind"] == "delete"),
                "next_move": share(lambda e: e["next_kind"] == "move"),
                "next_eor_or_end": share(lambda e: e["next_kind"] in ("eor", "end")),
                "repl_same_of_with_repl": None,
                "prev_kind": {k: sum(1 for _, e in evs if e["prev_kind"] == k) for k in
                              ("insert", "delete", "move", "prompt")},
            }
            wr = [(ri, e) for ri, e in evs if e["repl"] is not None]
            if wr:
                num = np.bincount([ri for ri, _ in wr], weights=[float(e["repl_same"]) for _, e in wr], minlength=N)
                den = np.bincount([ri for ri, _ in wr], minlength=N).astype(float)
                m["events"]["repl_same_of_with_repl"] = ratio_ci(num, den, I)
            # deletion runs (consecutive DELETE actions)
            runs = []
            for _, _, es in rep:
                ss = [e["s"] for e in es]
                k = 0
                while k < len(ss):
                    j = k
                    while j + 1 < len(ss) and ss[j + 1] == ss[j] + 1:
                        j += 1
                    runs.append(j - k + 1)
                    k = j + 1
            m["events"]["runs"] = len(runs)
            m["events"]["run_len_1_share"] = float(np.mean(np.array(runs) == 1))
            m["events"]["run_len_mean"] = float(np.mean(runs))
            from collections import Counter
            m["events"]["top_deleted"] = [[tok.decode([t]), c] for t, c in
                                          Counter(e["tok"] for _, e in evs).most_common(15)]
            m["events"]["top_repeat_pairs_nonrep1"] = [[tok.decode([t]), c] for t, c in Counter(
                e["tok"] for _, e in evs if not e["rep1"]).most_common(15)]
        # GPT-2 scores
        gp = out / f"{n}.gpt2.json"
        if gp.exists():
            G = json.loads(gp.read_text())
            allp = np.concatenate([np.exp(np.array(v)) for v in G["text_logp"] if v]) if G["text_logp"] else None
            m["gpt2_kept_new_tokens"] = {"n": int(allp.size), "p_median": float(np.median(allp)),
                                         "p_mean": float(allp.mean()),
                                         "share_p_lt_0.01": float((allp < 0.01).mean())}
            if evs:
                ev_by = {(ri, s): (pd, pr) for ri, s, pd, pr in G["events"]}
                pd = np.array([ev_by[(ri, e["s"])][0] for ri, e in evs])
                pr = np.array([np.nan if ev_by[(ri, e["s"])][1] is None else ev_by[(ri, e["s"])][1] for ri, e in evs])
                rep1 = np.array([e["rep1"] for _, e in evs])
                g = {"p_deleted_median": float(np.median(pd)), "p_deleted_mean": float(pd.mean()),
                     "share_deleted_p_lt_0.01": float((pd < 0.01).mean())}
                ok = ~np.isnan(pr)
                if ok.any():
                    ria = ri_arr[ok]
                    lr = np.log10(np.maximum(pr[ok], 1e-12)) - np.log10(np.maximum(pd[ok], 1e-12))
                    better = (pr[ok] > pd[ok]).astype(float)
                    cnt = np.bincount(ria, minlength=N).astype(float)
                    g.update({"n_with_repl": int(ok.sum()),
                              "p_repl_median": float(np.median(pr[ok])),
                              "p_deleted_median_with_repl": float(np.median(pd[ok])),
                              "share_repl_more_likely": ratio_ci(np.bincount(ria, weights=better, minlength=N), cnt, I),
                              "mean_log10_ratio_repl_over_deleted": ratio_ci(np.bincount(ria, weights=lr, minlength=N),
                                                                             cnt, I),
                              "median_log10_ratio": float(np.median(lr))})
                    for lab, sel in (("rep1", rep1[ok]), ("not_rep1", ~rep1[ok])):
                        if sel.any():
                            g[lab] = {"n": int(sel.sum()), "p_deleted_median": float(np.median(pd[ok][sel])),
                                      "p_repl_median": float(np.median(pr[ok][sel])),
                                      "share_repl_more_likely": float(better[sel].mean()),
                                      "median_log10_ratio": float(np.median(lr[sel]))}
                m["gpt2_deletes"] = g
        R["models"][n] = m
        per[n] = (new, nd, nt, tc)
        print(json.dumps({"model": n, "report_done": True, "secs": round(time.time() - t0, 1)}), flush=True)

    # paired comparisons of final texts
    for A, B in PAIRS:
        if A not in per or B not in per:
            continue
        newA, _, ntA, tcA = per[A]
        newB, _, ntB, tcB = per[B]
        P = {"len_mean": [float(ntA.mean()), float(ntB.mean())]}
        for lab, i, j in (("adj_dup_token", 0, 1), ("adj_dup_word", 2, 3), ("rep4", 4, 5)):
            P[lab] = ratio_diff_ci(tcA[:, i], tcA[:, j], tcB[:, i], tcB[:, j], I)
        # matched length: both cut to min new length per prompt
        cA, cB = [], []
        for x, y in zip(newA, newB):
            L = min(len(x), len(y))
            cA.append(text_counts(x[:L], tok))
            cB.append(text_counts(y[:L], tok))
        cA, cB = np.array(cA, float), np.array(cB, float)
        P["matched_len_mean"] = float(np.mean([min(len(x), len(y)) for x, y in zip(newA, newB)]))
        for lab, i, j in (("adj_dup_token", 0, 1), ("adj_dup_word", 2, 3), ("rep4", 4, 5)):
            P["matched_" + lab] = {"A": ratio_ci(cA[:, i], cA[:, j], I), "B": ratio_ci(cB[:, i], cB[:, j], I),
                                   "diff": ratio_diff_ci(cA[:, i], cA[:, j], cB[:, i], cB[:, j], I)}
        R["pairs"][f"{A}_vs_{B}"] = P
    R["secs"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({"report": str(out / "report.json"), "secs": R["secs"]}), flush=True)


if __name__ == "__main__":
    main()
