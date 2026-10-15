"""Self-revision evaluation (task 39): does a second pass over a draft help or harm?

Two conditions, kept separate as the assignment requires:
* ``own``      round 0 edits the benchmark input; round k >= 1 re-submits the model's OWN round k-1 output as the
               draft (edit / patch layouts: supplied draft; AR layout: "instruction : previous output" prompt, i.e. an
               equal-budget AR re-write of its own output).
* ``supplied`` round 1 starts from the SAME externally supplied drafts for every model (``--drafts`` = a predictions
               JSONL written by evaluate_edits/evaluate_external, e.g. the AR baseline's round-0 outputs), so models
               are compared on identical inputs.
Every round is scored against the ORIGINAL record (source text for SARI, original draft for "changed"/edit P/R), so
rounds are directly comparable with the round-0 benchmark numbers. Transition counts per round: helpful
(wrong -> exact), harmful (exact -> wrong), changed-after-exact, plus mean SARI delta.

Usage:
  python -m reviser_v2.evaluate_selfrevise --ckpt runs/re100-d4-s1 --data synth_real --split dev --rounds 2 --type-greedy
  python -m reviser_v2.evaluate_selfrevise --ckpt runs/re100-d4-s1 --data iterater_human_sent --split test \
      --drafts runs/re100-ar-s1/eval/edits_iterater_human_sent_test_tgreedy_ar.jsonl --rounds 1 --type-greedy
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .evaluate_edits import ROOT, aggregate, load_records, predict, score, synth_records


def _records(a, tok):
    if a.data in ("synth", "synth_forced"):
        assert a.data == "synth", "own-history synth_forced has no supplied-draft second pass"
        return synth_records(a.data, a.limit or 1000, tok=tok)
    if a.data in ("synth_real", "infill"):
        from .evalsets import eval_set_records
        return eval_set_records(a.data, a.limit, tok=tok)
    return load_records(a.data, a.split, a.limit)


def _finals(recs_in, a, layout):
    traces, meta = predict(recs_in, a.ckpt, a.device, a.batch, 0.0, a.type_greedy, 0, a.max_extra, layout, "conditioning")
    fins = [t.final for t in traces]
    if layout == "patch":
        from .patchfmt import apply_patch
        fins = [apply_patch(r["before_ids"], f)[0] for r, f in zip(recs_in, fins)]
    return fins, [len(t.actions) for t in traces], meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2, help="revision rounds after round 0 (own) / after the supplied draft")
    ap.add_argument("--drafts", default=None, help="predictions JSONL (uid, pred) whose outputs are the supplied drafts")
    ap.add_argument("--layout", choices=["edit", "ar", "patch"], default="edit")
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-extra", type=int, default=64)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2TokenizerFast
    tdir = ROOT / "data" / "tokenizers" / "gpt2"
    tok = GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")
    recs = _records(a, tok)
    t0 = time.time()
    rounds = []
    if a.drafts:  # supplied condition: round "S" = the external drafts themselves (scored), then model rounds
        preds = {json.loads(l)["uid"]: json.loads(l)["pred"] for l in open(a.drafts, encoding="utf-8")}
        recs = [r for r in recs if r["uid"] in preds]
        cur = [tok.encode(preds[r["uid"]]) for r in recs]
        rounds.append(("supplied", cur, [0] * len(recs)))
        cond = "supplied"
    else:
        cur, nact, meta = _finals(recs, a, a.layout)
        rounds.append(("r0", cur, nact))
        cond = "own"
    for k in range(1, a.rounds + 1):
        rin = [dict(r, before_ids=list(c), before_text=tok.decode(c)) for r, c in zip(recs, cur)]
        cur, nact, meta = _finals(rin, a, a.layout)
        rounds.append((f"r{k}", cur, nact))
    out, prev_rows = {}, None
    p = Path(a.ckpt)
    od = Path(a.out_dir) if a.out_dir else (p if p.is_dir() else p.parent) / "eval"
    od.mkdir(parents=True, exist_ok=True)
    dec = "tgreedy" if a.type_greedy else "greedy"
    tag = "own" if cond == "own" else "sup-" + Path(a.drafts).stem
    name = f"selfrev_{a.data}_{a.split}_{dec}" + ("" if a.layout == "edit" else f"_{a.layout}") + f"_{tag}" + (f"_lim{a.limit}" if a.limit else "")
    with open(od / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for rname, fins, nact in rounds:
            rows = score(recs, fins, tok, [dict(n_actions=n) for n in nact])
            s = aggregate(rows)["ALL"]
            if prev_rows is not None:
                pe = [r["exact"] for r in prev_rows]
                ce = [r["exact"] for r in rows]
                s["helpful"] = sum((not x) and y for x, y in zip(pe, ce)) / len(rows)
                s["harmful"] = sum(x and (not y) for x, y in zip(pe, ce)) / len(rows)
                s["changed_vs_prev"] = sum(pf != cf for pf, cf in zip(prev_fins, fins)) / len(rows)
                s["sari_delta"] = sum(r["sari"] - q["sari"] for r, q in zip(rows, prev_rows)) / len(rows)
            out[rname] = s
            for r in rows:
                f.write(json.dumps(dict(r, round=rname)) + "\n")
            prev_rows, prev_fins = rows, fins
    meta = dict(ckpt=str(a.ckpt), data=a.data, split=a.split, n=len(recs), condition=cond, drafts=a.drafts, layout=a.layout,
                decoding=dec, rounds=a.rounds, seconds=time.time() - t0)
    json.dump({"meta": meta, "rounds": out}, open(od / f"{name}.summary.json", "w"), indent=1)
    for k, s in out.items():
        print(name, k, json.dumps({x: round(v, 4) for x, v in s.items() if isinstance(v, float)}))


if __name__ == "__main__":
    main()
