"""Real-edit benchmark evaluation (tasks 24/26): IteraTeR / CoEdIT records from data/real/*.jsonl
(built by reviser_v2.realdata.build; GPT-2 ids + gpt2_action_space).

The model sees the instruction as the prompt and ``before_text`` as a supplied draft (``conditioning`` layout,
rollout.py), then emits edit actions; the final canvas is decoded with the GPT-2 tokenizer. Baselines without a
model: ``copy`` (output = draft; the standard "no edit" floor) and ``reference`` (sanity: must give EM 1).

Metrics: SARI (metrics_edit, pinned to the HF reference example), exact match on token ids, token-edit
precision/recall vs the reference diff, changed-rate, stop rate, action counts; overall and per intent, with
Wilson CIs on EM.

Usage:
  python -m reviser_v2.evaluate_edits --data iterater_human_sent --split test --baseline copy
  python -m reviser_v2.evaluate_edits --data coedit --split validation --ckpt path/to/ckpt.pt --type-greedy --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from .evaluate_toy import wilson
from .metrics_edit import edit_prf, sari_sent

ROOT = Path(__file__).resolve().parents[2]


def load_records(data: str, split: str, limit: int = 0, max_before: int = 0):
    recs = []
    with open(ROOT / "data" / "real" / f"{data}_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if max_before and len(r["before_ids"]) > max_before:
                continue
            recs.append(r)
            if limit and len(recs) >= limit:
                break
    return recs


def synth_records(kind: str, n: int = 1000, seed: int = 20260924, tok=None):
    """Deterministic synthetic-error dev sets from the dev pool (tasks 16/23):
    ``synth``        conditioning layout: prompt = "Fix the errors in this text", draft = corrupted text.
    ``synth_forced`` own-history layout: prompt = first 35% of the clean text, the corrupted remainder is typed as
                     forced Insert actions (draft_mode forced_prefix), the model must delete/fix and stop.
    Errors: 0-3 of replace/delete/insert (corpus-random tokens) / adjacent swap; intent = synth<k>."""
    import random
    from .realdata.editscript import gpt2_action_space
    from .tasks.realedit import FIX_INSTR, RealEditSource
    src = RealEditSource(gpt2_action_space(), {"max_pair_tokens": 256}, [{"kind": "synth_repair", "weight": 1}], seed, "dev")
    out = []
    for i in range(n):
        rng = random.Random(f"{kind}:{seed}:{i}")
        _, b, a = src.pool[rng.randrange(len(src.pool))]
        clean = a or b
        ne = i % 4
        if kind == "synth_forced":
            k = int(round(0.35 * len(clean)))
            prompt, clean = clean[:k], clean[k:]
        else:
            prompt = FIX_INSTR
        draft = src.corrupt(clean, rng, ne)
        out.append(dict(uid=f"{kind}/{i}", intent=f"synth{ne}", instruction_ids=prompt, before_ids=draft, after_ids=clean,
                        before_text=tok.decode(draft), after_text=tok.decode(clean)))
    return out


def load_model(ckpt: str, device: str):
    from .actions import ActionSpace
    from .model import ActionLM, ModelConfig
    p = Path(ckpt)
    if p.is_dir():
        p = p / "ckpt" / "latest.pt"
    from .hf_backbone import HFActionLM, is_hf_checkpoint
    if is_hf_checkpoint(p):  # pretrained HF backbone (task 36): base re-loaded + hash-checked, then trained tensors
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        return HFActionLM.from_checkpoint(ck, device).eval(), ActionSpace.from_dict(ck["action_space"]), ck
    ck = torch.load(p, map_location=device, weights_only=False)
    space = ActionSpace.from_dict(ck["action_space"])
    model = ActionLM(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), space, ck


def predict(recs, ckpt, device, batch, temperature, type_greedy, sample_seed, max_extra, layout="edit", draft_mode="conditioning",
            stop_bias=0.0, insert_only=False):
    from .rollout import Request, rollout
    model, space, ck = load_model(ckpt, device)
    greedy = temperature <= 0
    gen = None if greedy else torch.Generator(device=device).manual_seed(sample_seed)
    if layout in ("ar", "patch"):  # AR baselines: read-only "instruction : before" prompt, blank canvas; ar types the
        from .tasks.realedit import ar_prompt  # full rewrite, patch a search/replace edit script (patchfmt, task 30)
        reqs = [Request(prompt=ar_prompt(r["instruction_ids"], r["before_ids"]), max_actions=2 * len(r["before_ids"]) + max_extra)
                for r in recs]
    elif layout == "fim":  # AR FIM (task 30): prefix FIM_SUF suffix FIM_MID -> middle; needs the gap position (oracle)
        from .patchfmt import fim_prompt
        reqs = [Request(prompt=fim_prompt(r["before_ids"][:r["gap_start"]], r["before_ids"][r["gap_start"]:]), max_actions=max_extra)
                for r in recs]
    else:
        reqs = [Request(prompt=list(r["instruction_ids"]), draft=list(r["before_ids"]), draft_mode=draft_mode,
                        seed_cursor=r.get("seed_cursor"),  # infill_pos: cursor starts at the gap
                        max_actions=3 * len(r["before_ids"]) + max_extra) for r in recs]
    traces = []
    for i in range(0, len(reqs), batch):
        traces += rollout(model, space, reqs[i:i + batch], device=device, greedy=greedy,
                          temperature=1.0 if greedy else temperature, generator=gen, type_greedy=type_greedy,
                          stop_bias=stop_bias, insert_only=insert_only)
    return traces, dict(ckpt=str(ckpt), ckpt_step=ck.get("step"), space=space.describe(), layout=layout, draft_mode=draft_mode)


def score(recs, finals, tok, extra=None):
    rows = []
    for i, (r, fin) in enumerate(zip(recs, finals)):
        pred_text = tok.decode(fin) if fin is not None else ""
        s = sari_sent(r["before_text"], pred_text, [r["after_text"]])
        e = edit_prf(r["before_ids"], fin, r["after_ids"])
        row = dict(uid=r["uid"], intent=r["intent"], exact=fin == r["after_ids"], changed=fin != r["before_ids"],
                   ref_changed=r["after_ids"] != r["before_ids"], pred=pred_text, **s,
                   edit_p=e["precision"], edit_r=e["recall"], n_pred_edits=e["n_pred_edits"], n_ref_edits=e["n_ref_edits"])
        if extra:
            row.update(extra[i])
        if "gap_starts" in r:  # multi-gap infill (evalsets.infill2_records)
            from .evalsets import multigap_metrics
            row.update(multigap_metrics(r, fin))
        elif "gap_len" in r:  # infill sets (evalsets.py): span-length error, context preserved
            from .evalsets import infill_metrics
            row.update(infill_metrics(r, fin))
        rows.append(row)
    return rows


def aggregate(rows):
    def agg(rs):
        n = len(rs)
        out = dict(n=n, exact=sum(r["exact"] for r in rs) / n, exact_ci=wilson(sum(r["exact"] for r in rs), n),
                   changed=sum(r["changed"] for r in rs) / n)
        for k in ("sari", "keep", "del", "add", "edit_p", "edit_r"):
            out[k] = sum(r[k] for r in rs) / n
        for k in ("stopped", "n_actions", "n_del", "n_move", "n_ins", "len_err", "abs_len_err", "len_exact", "outside_intact",
                  "patch_well_formed", "patch_applied", "patch_hunks"):
            if k in rs[0]:
                out[k] = sum(r[k] for r in rs) / n
        return out
    by = defaultdict(list)
    for r in rows:
        by[r["intent"]].append(r)
    return {"ALL": agg(rows), **{f"intent/{k}": agg(v) for k, v in sorted(by.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="iterater_human_sent, iterater_v2_sent, coedit, or dev-pool sets: synth, "
                    "synth_forced, synth_real (realistic errors), infill, infill_pos (gap position known), infill2 (two gaps), "
                    "infill2_pos (two gaps, oracle positions; fim only) (use --split dev)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", choices=["copy", "reference"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-before", type=int, default=0, help="skip drafts longer than this many tokens")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-extra", type=int, default=64)
    ap.add_argument("--layout", choices=["edit", "ar", "patch", "fim"], default="edit",
                    help="edit: before as editable draft; ar: rewrite from prompt; patch: AR edit script applied to before; "
                         "fim: AR fill-in-the-middle at the known gap (infill_pos only: oracle position)")
    ap.add_argument("--out-dir", default=None, help="default: <run/ckpt dir>/eval or runs/edit_baselines")
    ap.add_argument("--tag", default="")
    ap.add_argument("--stop-bias", type=float, default=0.0, help="added to the STOP logit each step (edit layout); name suffix _sb<x>")
    ap.add_argument("--insert-only", action="store_true",
                    help="mask moves and DELETE (rollout insert_only): the exact AR decoder for AR-layout baselines; name suffix _io")
    ap.add_argument("--task-tokens", choices=["auto", "on", "off"], default="auto",
                    help="Reviser-Edit v2 (tasktokens.py): prepend [TASK(intent)] to each instruction; auto = iff the checkpoint "
                         "has task-token rows (no effect on older checkpoints)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    assert bool(a.ckpt) != bool(a.baseline), "give exactly one of --ckpt / --baseline"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2TokenizerFast
    tdir = ROOT / "data" / "tokenizers" / "gpt2"  # local copy of the HF gpt2 tokenizer files (remote boxes have no HF cache)
    tok = GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")
    draft_mode = "conditioning"
    if a.data in ("synth", "synth_forced"):
        recs = synth_records(a.data, a.limit or 1000, tok=tok)
        draft_mode = "forced_prefix" if a.data == "synth_forced" else "conditioning"
        assert a.layout == "edit" or a.data == "synth", "AR layout only for supplied-draft sets"
    elif a.data in ("synth_real", "infill", "infill_pos", "infill2", "infill2_pos"):  # evalsets.py (tasks 23/27)
        from .evalsets import eval_set_records
        recs = eval_set_records(a.data, a.limit, tok=tok)
        assert not a.data.startswith("infill2") or (a.data == "infill2_pos") == (a.layout == "fim"), \
            "infill2_pos is the fim (oracle positions) set; infill2 is for the edit/ar layouts"
        assert a.layout in ("edit", "fim") or a.data != "infill_pos", "infill_pos (gap position given) is edit/fim-layout only"
        assert a.layout != "fim" or a.data in ("infill_pos", "infill2_pos"), "fim needs the gap position: use --data infill_pos"
    else:
        recs = load_records(a.data, a.split, a.limit, a.max_before)
    from .tasktokens import apply_task_tokens, resolve_task_tokens
    tt_vocab = resolve_task_tokens(a.task_tokens, a.ckpt)
    if tt_vocab is not None:
        recs = apply_task_tokens(recs, tt_vocab, a.data)
    if a.ckpt:  # HF-backbone checkpoint (task 36): records mapped to backbone ids, decoded with its tokenizer
        from .hf_backbone import eval_prepare
        recs, tok = eval_prepare(a.ckpt, recs, tok, a.layout)
    t0 = time.time()
    if a.baseline:
        finals = [list(r["before_ids"]) if a.baseline == "copy" else list(r["after_ids"]) for r in recs]
        rows, meta = score(recs, finals, tok), dict(baseline=a.baseline)
        dec = a.baseline
        od = Path(a.out_dir) if a.out_dir else ROOT / "runs" / "edit_baselines"
    elif a.layout == "fim" and a.data == "infill2_pos":  # two gaps: single-gap FIM applied sequentially, left gap
        # first; the first pass sees the (unmarked) second gap inside its suffix. Oracle positions for both gaps.
        cur = [list(r["before_ids"]) for r in recs]
        extra = [dict(stopped=True, n_actions=0, n_del=0, n_move=0, n_ins=0) for _ in recs]
        fills = [0] * len(recs)
        for k in (0, 1):
            pos = [r["gap_starts"][k] + fills[i] for i, r in enumerate(recs)]
            sub = [dict(before_ids=c, gap_start=q) for c, q in zip(cur, pos)]
            traces, meta = predict(sub, a.ckpt, a.device, a.batch, a.temperature, a.type_greedy, a.sample_seed, a.max_extra,
                                   "fim", draft_mode, stop_bias=a.stop_bias, insert_only=a.insert_only)
            for i, (t, q) in enumerate(zip(traces, pos)):
                cur[i] = cur[i][:q] + list(t.final) + cur[i][q:]
                fills[i] += len(t.final)
                e = extra[i]
                e.update(stopped=e["stopped"] and t.stopped, n_actions=e["n_actions"] + len(t.actions),
                         n_del=e["n_del"] + t.n_del, n_move=e["n_move"] + t.n_move, n_ins=e["n_ins"] + t.n_ins)
        meta.update(fim_multigap="sequential, left gap first")
        rows = score(recs, cur, tok, extra)
        dec = ("tgreedy" if a.type_greedy else "greedy") if a.temperature <= 0 else f"T{a.temperature}s{a.sample_seed}"
        p = Path(a.ckpt)
        od = Path(a.out_dir) if a.out_dir else (p if p.is_dir() else p.parent) / "eval"
    else:
        traces, meta = predict(recs, a.ckpt, a.device, a.batch, a.temperature, a.type_greedy, a.sample_seed, a.max_extra, a.layout, draft_mode,
                               stop_bias=a.stop_bias, insert_only=a.insert_only)
        extra = [dict(stopped=t.stopped, n_actions=len(t.actions), n_del=t.n_del, n_move=t.n_move, n_ins=t.n_ins) for t in traces]
        finals = [t.final for t in traces]
        if a.layout == "patch":  # the generated text is an edit script: apply it to the draft
            from .patchfmt import apply_patch
            for i, r in enumerate(recs):
                finals[i], info = apply_patch(r["before_ids"], finals[i])
                extra[i].update(patch_well_formed=info["well_formed"], patch_applied=info["applied"], patch_hunks=info["n_hunks"])
        elif a.layout == "fim":  # the generated text is the middle
            finals = [r["before_ids"][:r["gap_start"]] + f + r["before_ids"][r["gap_start"]:] for r, f in zip(recs, finals)]
        rows = score(recs, finals, tok, extra)
        dec = ("tgreedy" if a.type_greedy else "greedy") if a.temperature <= 0 else f"T{a.temperature}s{a.sample_seed}"
        p = Path(a.ckpt)
        od = Path(a.out_dir) if a.out_dir else (p if p.is_dir() else p.parent) / "eval"
    summ = aggregate(rows)
    meta.update(data=a.data, split=a.split, n=len(recs), limit=a.limit, max_before=a.max_before, decoding=dec, stop_bias=a.stop_bias,
                insert_only=a.insert_only, task_tokens=tt_vocab is not None,
                seconds=time.time() - t0)
    od.mkdir(parents=True, exist_ok=True)
    name = f"edits_{a.data}_{a.split}_{dec}" + ("" if a.layout == "edit" else f"_{a.layout}") + (f"_lim{a.limit}" if a.limit else "") + (f"_sb{a.stop_bias:g}" if a.stop_bias else "") + ("_io" if a.insert_only else "") + (f"_{a.tag}" if a.tag else "")
    with open(od / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"meta": meta, "summary": summ}, open(od / f"{name}.summary.json", "w"), indent=1)
    s = summ["ALL"]
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items() if k != "exact_ci"}))


if __name__ == "__main__":
    main()
