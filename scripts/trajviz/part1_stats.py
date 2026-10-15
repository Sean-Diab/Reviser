#!/usr/bin/env python3
"""Part 1 of the trajectory check (writing from a prompt): cursor, move and delete statistics for the paper models,
plus 10 viewer examples per model (notes/trajectory-check-2026-10-09.md).

Inputs are the existing prompt-locked paper-eval decodes listed in configs/trajviz/models.json (3000 C4 prompts,
1000 per seed 123/124/125, T 0.9, top-k 50, 220 new actions; scripts/papereval/decode_batched.py). Each decode's _meta
ckpt_sha256 is checked against the config. Every row is replayed (prompt + actions) and must reproduce its saved canvas.

Also:
  * the original paper's trajectory table (exact port in trajstats.paper_analyze / paper_table), validated first on the
    paper's own stored rollouts (read-only: ../workspaces/4-21/c4_v2_arena_test/results_{100m,300m}/seed*/
    cursor_*_unbounded_chunks/cursor_chunk_*/inference_examples.txt), against the paper's own analyze_trajectory()
    and against the published table;
  * prompt edits in the same models' unlocked decodes (decode_unlocked), to show what the lock prevents.

Viewer examples: the same 10 prompts for every model, drawn with random.Random(20261009) from the 3000 (never chosen by
quality).

  wsl.exe -d Reviser-Ubuntu -- /opt/venv/bin/python scripts/trajviz/part1_stats.py
Outputs: runs/trajviz/part1/{stats.json, examples.json, paper_validation.json}
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "trajviz"))
import trajstats as T  # noqa: E402

CFG = ROOT / "configs" / "trajviz" / "models.json"
OUT = ROOT / "runs" / "trajviz" / "part1"
PAPER_DIR = ROOT.parent / "workspaces" / "4-21" / "c4_v2_arena_test"
LO = 35
EX_SEED, N_EX = 20261009, 10
PUBLISHED = {  # notes/sources/original-paper-2026-09-24/main.tex, Table tab:trajectory_stats
    "100m": dict(actions_per_token=1.51, insert_frac=0.67, move_frac=0.33, mean_move_dist=4.92,
                 mass_1_2_4_8=[40.5, 23.3, 15.8, 10.0], frac_backward_moves=0.655, mean_insert_rel=0.41,
                 frac_end_append=0.039, backward_revision_share=1.0),
    "300m": dict(actions_per_token=1.40, insert_frac=0.72, move_frac=0.28, mean_move_dist=6.05,
                 mass_1_2_4_8=[37.4, 22.7, 15.5, 11.1], frac_backward_moves=0.645, mean_insert_rel=0.42,
                 frac_end_append=0.031, backward_revision_share=1.0),
}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_decode(path):
    meta, rows = None, []
    for line in open(ROOT / path, encoding="utf-8"):
        r = json.loads(line)
        if "_meta" in r:
            meta = r["_meta"]
        else:
            rows.append(r)
    return meta, rows


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-12
    return a == b


def validate_paper():
    """Our paper_analyze vs the paper's own analyze_trajectory on the paper's stored rollouts, and the seed-averaged
    table vs the published numbers."""
    sys.dont_write_bytecode = True  # do not leave __pycache__ in the read-only reference tree
    src = PAPER_DIR / "analyze_reviser_diagnostics.py"
    spec = importlib.util.spec_from_file_location("paper_diag_ref", src)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    out = {"reference_script": str(src.relative_to(ROOT.parent)), "reference_sha256": sha256_file(src), "sizes": {}}
    for size in ("100m", "300m"):
        by_seed, n_cmp, n_bad = {}, 0, 0
        for seed in (123, 124, 125):
            txts = sorted((PAPER_DIR / f"results_{size}" / f"seed{seed}" / f"cursor_{size}_unbounded_chunks").glob(
                "cursor_chunk_*/inference_examples.txt"))
            seqs = []
            for t in txts:
                seqs += ref.parse_inference_examples_txt(str(t))
            by_seed[str(seed)] = seqs
            for sq in seqs:
                a, b = ref.analyze_trajectory(sq), T.paper_analyze(sq)
                n_cmp += 1
                if any(not _same(a[k], b[k]) if not isinstance(a[k], list) else
                       (len(a[k]) != len(b[k]) or any(not _same(x, y) for x, y in zip(a[k], b[k]))) for k in a):
                    n_bad += 1
        tab = T.paper_table(by_seed)
        avg = tab["avg"]
        # our own measures on the paper's rollouts (prompt already stripped, canvas starts empty, unlocked decode)
        ag = T.aggregate([T.sample_stats([], T.ops_from_july(sq), 0, 0) for v in by_seed.values() for sq in v])
        extra = {k: ag[k] for k in ("n", "mean_out_len", "moves_per_token", "mean_jump_abs", "median_jump_abs",
                                    "jump_target_bins", "edit_bins", "edits_not_at_end", "samples_state_3plus",
                                    "samples_state_5plus", "move_pingpong_per_100_moves", "samples_move_run_8plus",
                                    "deletes_per_token")}
        out["sizes"][size] = dict(
            n_examples={k: len(v) for k, v in by_seed.items()}, compared=n_cmp, mismatched=n_bad,
            ours=dict(actions_per_token=avg["actions_per_token"], insert_frac=avg["insert_frac"], move_frac=avg["move_frac"],
                      mean_move_dist=avg["mean_move_dist"],
                      mass_1_2_4_8=[100 * avg["move_dist_mass"][str(a)] for a in (1, 2, 4, 8)],
                      frac_backward_moves=avg["frac_backward_moves"], mean_insert_rel=avg["mean_insert_rel"],
                      frac_end_append=avg["frac_end_append"], backward_revision_share=avg["backward_revision_share"]),
            published=PUBLISHED[size], our_measures_on_paper_rollouts=extra)
        print("paper validation", size, json.dumps(out["sizes"][size]), flush=True)
    return out


def unlocked_check(path):
    meta, rows = read_decode(path)
    n = len(rows)
    not_intact = sum(not r["prefix_intact"] for r in rows)
    pdel = smp_pdel = mv_in = smp_mv_in = 0
    for r in rows:
        st = T.sample_stats(r["prefix_ids"], T.ops_from_july(r["actions_new"]), LO, LO)
        pdel += st["del_in_prompt"]
        smp_pdel += st["del_in_prompt"] > 0
        mv_in += st["moves_into_prompt"]
        smp_mv_in += st["moves_into_prompt"] > 0
    return dict(decode=path, n=n, ckpt_sha256=meta.get("ckpt_sha256"), prompt_changed_rate=not_intact / n,
                prompt_token_deletes_per_sample=pdel / n, samples_deleting_prompt_tokens=smp_pdel / n,
                samples_moving_into_prompt=smp_mv_in / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--skip-unlocked", action="store_true")
    ap.add_argument("--validation-only", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.skip_validation:
        (OUT / "paper_validation.json").write_text(json.dumps(validate_paper(), indent=1))
    if a.validation_only:
        return

    sys.path.insert(0, str(ROOT / "scripts" / "viz"))
    from text_deletion_traces import piece  # same token-to-text rule as the existing viewer
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))

    stats, examples, ex_idx, ex_ids = {}, {}, None, None
    for m in cfg["part1"]:
        if a.models and m["key"] not in a.models:
            continue
        t0 = time.time()
        meta, rows = read_decode(m["decode"])
        sha_ok = meta.get("ckpt_sha256") == m["ckpt_sha256"]
        if not sha_ok:
            raise SystemExit(f"{m['key']}: decode _meta ckpt_sha256 {meta.get('ckpt_sha256')} != config {m['ckpt_sha256']}")
        S, replay_ok, out_len_ok, by_seed = [], 0, 0, {}
        for r in rows:
            ops = T.ops_from_july(r["actions_new"])
            fin = T.final_tokens(r["prefix_ids"], ops, LO)
            replay_ok += fin == r["canvas_ids"]
            st = T.sample_stats(r["prefix_ids"], ops, LO, LO)
            out_len_ok += st["out_len"] == r["out_len"]
            st["hit_budget"] = r["hit_budget"]
            st["prefix_intact"] = r["prefix_intact"]
            S.append(st)
            by_seed.setdefault(str(r["seed"]), []).append(r["actions_new"])
        agg = T.aggregate(S)
        agg["hit_budget_rate"] = sum(s["hit_budget"] for s in S) / len(S)
        agg["prefix_intact_rate"] = sum(s["prefix_intact"] for s in S) / len(S)
        ptab = T.paper_table(by_seed)
        info = dict(key=m["key"], label=m["label"], kind=m["kind"], note=m["note"], decode=m["decode"],
                    decode_sha256=sha256_file(ROOT / m["decode"]), ckpt=m["ckpt"], ckpt_sha256=m["ckpt_sha256"],
                    meta_sha_ok=sha_ok, decode_meta={k: meta.get(k) for k in ("args", "prompts_sha256", "decoder",
                                                                              "decoder_sha256", "max_new_actions", "gpu", "torch", "rng")},
                    n=len(rows), seeds=sorted(by_seed), replay_ok=replay_ok, out_len_ok=out_len_ok, agg=agg,
                    paper=ptab["avg"], paper_per_seed=ptab["per_seed"])
        if m.get("decode_unlocked") and not a.skip_unlocked:
            info["unlocked"] = unlocked_check(m["decode_unlocked"])
        stats[m["key"]] = info
        print(m["key"], f"n={len(rows)} replay_ok={replay_ok} out_len_ok={out_len_ok} {time.time()-t0:.0f}s", flush=True)

        # viewer examples: same prompt positions for every model
        if ex_idx is None:
            ex_idx = sorted(random.Random(EX_SEED).sample(range(len(rows)), N_EX))
            ex_ids = [rows[i]["id"] for i in ex_idx]
        if [rows[i]["id"] for i in ex_idx] != ex_ids:
            raise SystemExit(f"{m['key']}: prompt order differs from the first model")
        exs = []
        for i in ex_idx:
            r = rows[i]
            s = S[i]
            ops = []
            for op in T.ops_from_july(r["actions_new"]):
                ops.append(["i", piece(tok, op[1])] if op[0] == "i" else list(op))
            exs.append(dict(id=r["id"], seed=r["seed"], prompt=tok.decode(r["prefix_ids"]),
                            ref=tok.decode(r["ref_cont_ids"]), final=tok.decode(r["canvas_ids"][LO:]), ops=ops,
                            stopped=r["stopped_eor"], hit_budget=r["hit_budget"], out_len=r["out_len"],
                            n_ins=s["n_ins"], n_del=s["n_del"], n_move=s["n_move"],
                            self_del=len(s["ages"]), age1=sum(x <= 1 for x in s["ages"])))
        examples[m["key"]] = exs

    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    (OUT / "examples.json").write_text(json.dumps(dict(seed=EX_SEED, index=ex_idx, ids=ex_ids, examples=examples)))
    (OUT / "run_info.json").write_text(json.dumps(dict(seconds=round(time.time() - t_all, 1),
                                                       code_sha256={p: sha256_file(ROOT / "scripts" / "trajviz" / p) for p in ("part1_stats.py", "trajstats.py")},
                                                       utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
    print("done", f"{time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
