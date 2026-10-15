#!/usr/bin/env python3
"""Checks for promptlock_patch_v1 (training-time legality mask with the 35-token prompt locked).

Subcommands
  scan     --data_dir D [--shards 0,1]
           Pure numpy, no torch: replays EVERY row of every shard of a built trajectory dir (build_variant.py layout)
           with the trainer's executor semantics and counts gold actions that the prompt-lock mask would forbid
           (MOVE to < P or > len, DELETE at cursor <= P), split into supervised labels (t >= P, loss mask 0) and all
           actions after the prompt. Fast enough for the full 134-shard build on the box. Exit 1 if any supervised
           label is illegal.
  collate  --data_dir D --tree SRC [--batches N]
           Runs the patched dataset + collator (legality_state=True) and the trainer's _promptlock_mask_logits on
           random logits. Checks the collator's cursor/len against an independent CursorState replay, that no label
           is illegal, that every masked column is exactly the columns a prompt-locked CursorState forbids, and that
           with legality_state=False the batch equals the lossmask_patch_v1 collator's batch.
  model    --data_dir D --tree SRC --ckpt WRAPPED [--batches N]
           Real logits from the wrapped release model (CPU, eval): masked CE (trainer function) == reference CE over
           the legal set built position by position with a prompt-locked CursorState, masked CE <= unmasked CE, all
           finite; reports the probability mass the mask removes.

SRC is a promptlock trainer tree (make_tree.sh output). Run with PYTHONPATH=SRC for collate/model (the script also
inserts SRC itself). Everything prints one JSON object on the last line.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DELETE, EOR, MOVE_START, N_MOVES, GPT2_VOCAB, EOT = 50259, 50260, 50261, 20, 50257, 50256
MOVE_AMTS = np.array([(1 << (k // 2)) * (-1 if k % 2 else 1) for k in range(N_MOVES)], np.int64)  # by id offset


def _shards(sd: Path, only: str):
    ks = sorted(int(p.name[len("rest_tokens_shard"):-4]) for p in sd.glob("rest_tokens_shard*.npy"))
    if only:
        want = {int(x) for x in only.split(",")}
        ks = [k for k in ks if k in want]
    return ks


def scan(data_dir: str, only: str = "") -> dict:
    sd = Path(data_dir) / "shards"
    tot = {"rows": 0, "actions": 0, "supervised_labels": 0, "illegal_supervised": 0, "illegal_after_prompt": 0,
           "illegal_in_prompt_region": 0, "moves": 0, "moves_to_exactly_P": 0, "deletes": 0, "deletes_at_P_plus_1": 0,
           "min_cursor_after_prompt": None, "clamped_or_noop": 0, "masked_cols_sum": 0}
    examples = []
    t0 = time.time()
    for k in _shards(sd, only):
        toks = np.load(sd / f"rest_tokens_shard{k:06d}.npy", mmap_mode="r")
        offs = np.load(sd / f"rest_offsets_shard{k:06d}.npy")
        pp = sd / f"prefix_len_shard{k:06d}.npy"
        P = np.load(pp).astype(np.int64) if pp.is_file() else np.zeros(len(offs), np.int64)
        lm_p = sd / f"rest_lossmask_shard{k:06d}.npy"
        lm = np.load(lm_p, mmap_mode="r") if lm_p.is_file() else None
        n = len(offs)
        L = int(offs[:, 1].max())
        A = np.full((n, L), -1, np.int64)
        M = np.zeros((n, L), np.uint8)
        for i, (s, ln) in enumerate(offs):
            A[i, :ln] = toks[s:s + ln]
            if lm is not None:
                M[i, :ln] = lm[s:s + ln]
        cur = np.zeros(n, np.int64)
        cl = np.zeros(n, np.int64)
        mincur = None
        for t in range(L):
            a = A[:, t]
            live = a >= 0
            after = live & (t >= P)
            is_del = a == DELETE
            is_mv = (a >= MOVE_START) & (a < MOVE_START + N_MOVES)
            amt = np.where(is_mv, MOVE_AMTS[np.clip(a - MOVE_START, 0, N_MOVES - 1)], 0)
            new = cur + amt
            ill = (is_del & (cur <= P)) | (is_mv & ((new < P) | (new > cl)))
            sup = after & (M[:, t] == 0)
            tot["supervised_labels"] += int(sup.sum())
            tot["illegal_supervised"] += int((ill & sup).sum())
            tot["illegal_after_prompt"] += int((ill & after).sum())
            tot["illegal_in_prompt_region"] += int((ill & live & (t < P)).sum())
            tot["moves"] += int((is_mv & after).sum())
            tot["moves_to_exactly_P"] += int((is_mv & after & (new == P)).sum())
            tot["deletes"] += int((is_del & after).sum())
            tot["deletes_at_P_plus_1"] += int((is_del & after & (cur == P + 1)).sum())
            # masked columns at this state (DELETE + 20 moves), summed over supervised positions
            newall = cur[:, None] + MOVE_AMTS[None, :]
            ncols = ((newall < P[:, None]) | (newall > cl[:, None])).sum(1) + (cur <= P)
            tot["masked_cols_sum"] += int(ncols[sup].sum())
            bad = np.flatnonzero(ill & after)
            for i in bad[: max(0, 5 - len(examples))]:
                examples.append({"shard": k, "row": int(i), "t": t, "action": int(a[i]), "cursor": int(cur[i]),
                                 "len": int(cl[i]), "P": int(P[i])})
            # executor update (resttraj_pretrain_collator._build_canvas_rows semantics)
            ins = live & (a < GPT2_VOCAB) & (a != EOT)
            tot["clamped_or_noop"] += int(((is_mv & ((new < 0) | (new > cl))) | (is_del & (cur == 0))).sum())
            dok = is_del & (cur > 0)
            cur = np.where(ins, cur + 1, cur)
            cl = np.where(ins, cl + 1, cl)
            cur = np.where(dok, cur - 1, cur)
            cl = np.where(dok, cl - 1, cl)
            cur = np.where(is_mv, np.clip(new, 0, cl), cur)
            # cursor after this action, for rows already past the prompt
            post = live & (t >= P)
            if post.any():
                rel = cur[post] - P[post]
                mincur = int(rel.min()) if mincur is None else min(mincur, int(rel.min()))
        tot["rows"] += n
        tot["actions"] += int(offs[:, 1].sum())
        if mincur is not None:
            prev = tot["min_cursor_after_prompt"]
            tot["min_cursor_after_prompt"] = mincur if prev is None else min(prev, mincur)
    tot["min_cursor_after_prompt_note"] = "minimum of (cursor - P) after any action past the prompt; 0 = at P"
    tot["mean_masked_cols_per_label"] = tot["masked_cols_sum"] / max(1, tot["supervised_labels"])
    tot["examples"] = examples
    tot["secs"] = round(time.time() - t0, 1)
    tot["data_dir"] = str(data_dir)
    tot["ok"] = tot["illegal_supervised"] == 0
    return tot


# ----------------------------------------------------------------------------------------------------------------
# torch parts (need a promptlock trainer tree on sys.path)
# ----------------------------------------------------------------------------------------------------------------

def _locked_state_cls(P: int):
    from reviser.decoding.inference_core import CursorState

    class Locked(CursorState):
        def can_move(self, amount: int) -> bool:
            new = self.cursor_pos + amount
            return P <= new <= len(self.canvas)

        def can_delete(self) -> bool:
            return self.cursor_pos > P
    return Locked


def _ref_illegal_cols(actions, P: int, seq_len: int):
    """Per position t: (cursor, len, set of illegal ids) from a prompt-locked CursorState replay (independent of
    the collator/trainer code). Prompt actions are replayed with the plain executor (they are inserts)."""
    Locked = _locked_state_cls(P)
    st = Locked.from_prompt([0])
    out = []
    for t in range(seq_len):
        ill = set()
        for k in range(N_MOVES):
            if not st.can_move(int(MOVE_AMTS[k])):
                ill.add(MOVE_START + k)
        if not st.can_delete():
            ill.add(DELETE)
        out.append((st.cursor_pos, len(st.canvas), ill))
        a = int(actions[t])
        # advance with the trainer's executor (clamped moves / no-op delete); for legal actions this equals
        # CursorState.execute_action, which would instead refuse an out-of-range move
        if a == DELETE:
            if st.cursor_pos > 0:
                st.canvas.pop(st.cursor_pos - 1)
                st.cursor_pos -= 1
        elif MOVE_START <= a < MOVE_START + N_MOVES:
            st.cursor_pos = max(0, min(len(st.canvas), st.cursor_pos + int(MOVE_AMTS[a - MOVE_START])))
        elif 0 <= a < GPT2_VOCAB and a != EOT:
            st.canvas.insert(st.cursor_pos, a)
            st.cursor_pos += 1
    return out


def _setup_tree(tree: str):
    sys.path.insert(0, str(Path(tree).resolve()))
    import reviser.training.pretrain_resttraj as T  # noqa: E402
    import reviser.data.resttraj_pretrain_collator as C  # noqa: E402
    import reviser.data.raining_resttraj_bucketed_dataset as D  # noqa: E402
    for m in (T, C, D):
        assert str(Path(m.__file__).resolve()).startswith(str(Path(tree).resolve())), m.__file__
    assert hasattr(C, "cursor_state_before_actions") and hasattr(T, "_promptlock_mask_logits"), "not a promptlock tree"
    return T, C, D


def _batches(D, data_dir, n, batch_size=4):
    ds = D.RainingBucketedRestTrajDataset(resttraj_dir=str(data_dir), batch_size=batch_size, seed=123, min_len=100,
                                          max_len=350, bucket_size=5, refresh_secs=60, log_every_epoch=False)
    out = []
    for b in ds:
        out.append(b)
        if len(out) >= n:
            break
    return out


def collate(data_dir: str, tree: str, n_batches: int) -> dict:
    import importlib.util
    import torch
    T, C, D = _setup_tree(tree)
    root = Path(__file__).resolve().parents[3]
    # the lossmask collator uses a relative import (.vocabulary): load it as a reviser.data submodule
    spec = importlib.util.spec_from_file_location(
        "reviser.data._lm_collator", root / "scripts" / "humanorder" / "lossmask_patch_v1" / "resttraj_pretrain_collator.py")
    LM = importlib.util.module_from_spec(spec)
    sys.modules["reviser.data._lm_collator"] = LM
    spec.loader.exec_module(LM)
    kw = dict(pad_token_id=50256, max_seq_length=512, max_canvas_length=512, use_canvas=False)
    c_on = C.RestTrajPretrainCollator(**kw, legality_state=True)
    c_off = C.RestTrajPretrainCollator(**kw)
    c_lm = LM.RestTrajPretrainCollator(**kw)
    g = torch.Generator().manual_seed(0)
    res = {"batches": 0, "rows": 0, "labels": 0, "positions_checked": 0, "masked_cells": 0, "illegal_labels": 0,
           "off_equals_lossmask_v1": True, "state_matches_reference": True, "mask_matches_reference": True}
    for b in _batches(D, data_dir, n_batches):
        on, off, lm = c_on(b), c_off(b), c_lm(b)
        assert set(off) == set(lm), (set(off), set(lm))
        for key in ("input_ids", "labels", "attention_mask", "prefix_lens"):
            res["off_equals_lossmask_v1"] &= bool(torch.equal(off[key], lm[key])) and bool(torch.equal(on[key], lm[key]))
        assert "legal_cursor" in on and "legal_cursor" not in off
        B, S = on["labels"].shape
        logits = torch.randn(B, S, 50291, generator=g)
        masked, n_chk, n_bad = T._promptlock_mask_logits(logits, on["labels"], on, where="check")
        res["illegal_labels"] += n_bad
        res["labels"] += n_chk
        isinf = torch.isinf(masked)
        res["masked_cells"] += int(isinf.sum())
        assert not isinf[..., :DELETE].any() and not isinf[..., EOR].any() and not isinf[..., MOVE_START + N_MOVES:].any()
        lab = on["labels"]
        assert bool(torch.isfinite(masked.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)[lab != -100]).all())
        for r, s in enumerate(b):
            acts = [int(x) for x in np.asarray(s["action_ids"])][:512]
            P = int(on["prefix_lens"][r])
            ref = _ref_illegal_cols(acts, P, len(acts))
            for t, (c, ln, ill) in enumerate(ref):
                if int(on["legal_cursor"][r, t]) != c or int(on["legal_canvas_len"][r, t]) != ln:
                    res["state_matches_reference"] = False
                got = {DELETE + j for j in torch.nonzero(isinf[r, t, DELETE:MOVE_START + N_MOVES]).flatten().tolist()}
                if got != ill:
                    res["mask_matches_reference"] = False
                res["positions_checked"] += 1
            res["rows"] += 1
        res["batches"] += 1
    res["ok"] = (res["illegal_labels"] == 0 and res["off_equals_lossmask_v1"] and res["state_matches_reference"]
                 and res["mask_matches_reference"] and res["batches"] == n_batches)
    return res


def model_check(data_dir: str, tree: str, ckpt: str, n_batches: int, batch_size: int) -> dict:
    import torch
    import torch.nn.functional as F
    T, C, D = _setup_tree(tree)
    from reviser.model import CursorConfig, CursorTransformer
    torch.manual_seed(0)
    cfg = CursorConfig(vocab_size=50291, d_model=512, n_heads=8, d_ff=2048, n_layers=24, dropout=0.0,
                       attention_dropout=0.0, positional_encoding="rope", max_seq_length=512, max_canvas_length=512,
                       tie_embeddings=True)
    model = CursorTransformer(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    col = C.RestTrajPretrainCollator(pad_token_id=50256, max_seq_length=512, max_canvas_length=512, use_canvas=False,
                                     legality_state=True)
    agg = {"batches": 0, "labels": 0, "ce_unmasked_sum": 0.0, "ce_masked_sum": 0.0, "ce_ref_sum": 0.0,
           "max_abs_masked_minus_ref": 0.0, "removed_mass_sum": 0.0, "removed_mass_max": 0.0,
           "per_label_masked_le_unmasked": True, "all_finite": True,
           "move_del_labels": 0, "ce_unmasked_move_del_sum": 0.0, "ce_masked_move_del_sum": 0.0}
    t0 = time.time()
    with torch.no_grad():
        for b in _batches(D, data_dir, n_batches, batch_size=batch_size):
            batch = col(b)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], canvas_ids=None,
                        canvas_mask=None, labels=None)
            logits = out["logits"].float()
            labels = batch["labels"]
            masked, n_chk, n_bad = T._promptlock_mask_logits(logits, labels, batch, where="model_check")
            assert n_bad == 0
            V = logits.shape[-1]
            valid = labels != -100
            ce_u = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none").view(labels.shape)
            ce_m = F.cross_entropy(masked.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none").view(labels.shape)
            lse = torch.logsumexp(logits, -1)
            for r, s in enumerate(b):
                acts = [int(x) for x in np.asarray(s["action_ids"])][:512]
                P = int(batch["prefix_lens"][r])
                ref = _ref_illegal_cols(acts, P, len(acts))
                for t, (_c, _l, ill) in enumerate(ref):
                    if not bool(valid[r, t]):
                        continue
                    row = logits[r, t].clone()
                    if ill:
                        idx = torch.tensor(sorted(ill))
                        removed = float(torch.exp(torch.logsumexp(row[idx], 0) - lse[r, t]))
                        row[idx] = float("-inf")
                    else:
                        removed = 0.0
                    ref_ce = float(torch.logsumexp(row, 0) - row[int(labels[r, t])])
                    agg["ce_ref_sum"] += ref_ce
                    agg["max_abs_masked_minus_ref"] = max(agg["max_abs_masked_minus_ref"], abs(ref_ce - float(ce_m[r, t])))
                    agg["removed_mass_sum"] += removed
                    agg["removed_mass_max"] = max(agg["removed_mass_max"], removed)
                    lab = int(labels[r, t])
                    if lab == DELETE or MOVE_START <= lab < MOVE_START + N_MOVES:
                        agg["move_del_labels"] += 1
                        agg["ce_unmasked_move_del_sum"] += float(ce_u[r, t])
                        agg["ce_masked_move_del_sum"] += float(ce_m[r, t])
            agg["per_label_masked_le_unmasked"] &= bool((ce_m[valid] <= ce_u[valid] + 1e-5).all())
            agg["all_finite"] &= bool(torch.isfinite(ce_m[valid]).all())
            agg["labels"] += int(valid.sum())
            agg["ce_unmasked_sum"] += float(ce_u[valid].sum())
            agg["ce_masked_sum"] += float(ce_m[valid].sum())
            agg["batches"] += 1
    n = max(1, agg["labels"])
    nmd = max(1, agg["move_del_labels"])
    res = {"batches": agg["batches"], "labels": agg["labels"], "ce_unmasked": agg["ce_unmasked_sum"] / n,
           "ce_masked": agg["ce_masked_sum"] / n, "ce_ref_locked_cursorstate": agg["ce_ref_sum"] / n,
           "max_abs_masked_minus_ref": agg["max_abs_masked_minus_ref"],
           "mean_removed_prob_mass": agg["removed_mass_sum"] / n, "max_removed_prob_mass": agg["removed_mass_max"],
           "move_delete_labels": agg["move_del_labels"],
           "ce_unmasked_on_move_delete_labels": agg["ce_unmasked_move_del_sum"] / nmd,
           "ce_masked_on_move_delete_labels": agg["ce_masked_move_del_sum"] / nmd,
           "per_label_masked_le_unmasked": agg["per_label_masked_le_unmasked"], "all_finite": agg["all_finite"],
           "ckpt": ckpt, "secs": round(time.time() - t0, 1)}
    res["ok"] = bool(res["all_finite"] and res["per_label_masked_le_unmasked"] and res["max_abs_masked_minus_ref"] < 1e-3
                     and res["batches"] == n_batches)
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("scan")
    a1.add_argument("--data_dir", required=True)
    a1.add_argument("--shards", default="")
    a2 = sub.add_parser("collate")
    a2.add_argument("--data_dir", required=True)
    a2.add_argument("--tree", required=True)
    a2.add_argument("--batches", type=int, default=20)
    a3 = sub.add_parser("model")
    a3.add_argument("--data_dir", required=True)
    a3.add_argument("--tree", required=True)
    a3.add_argument("--ckpt", required=True)
    a3.add_argument("--batches", type=int, default=3)
    a3.add_argument("--batch_size", type=int, default=4)
    for p in (a1, a2, a3):
        p.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "scan":
        res = scan(a.data_dir, a.shards)
    elif a.cmd == "collate":
        res = collate(a.data_dir, a.tree, a.batches)
    else:
        res = model_check(a.data_dir, a.tree, a.ckpt, a.batches, a.batch_size)
    res["cmd"] = a.cmd
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
