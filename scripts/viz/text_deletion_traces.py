"""Record per-action text trajectories of the 100M real-text analogues of the toy deletion recipes D0-D3.

  D0  data/checkpoints/reviser_100m_actionlm.pt            released Reviser 100M (paper recipe: insert/move/stop)
  D1  runs/restore/re100-restonlyu-s1/...                   restoration with junk inserts, every label supervised
  D2  runs/slim/re100-restonly-s1                           same trajectories, junk-insert labels masked
  D3  runs/slim/re100-d3-s1                                 supplied-draft editing + synthetic repair + junk-free restoration

Scenarios (same records for every model; picked by a fixed RNG among short records, never by outcome):
  continue   prompt = first 35% of a dev-pool text, blank canvas, free writing (evaluate_continuation layout);
             nucleus sampling T=0.9 top-p 0.95, fixed seed (the task-28 decoding), budget --budget actions
  synth      supplied draft (conditioning): "Fix the errors in this text" + corrupted draft (evaluate_edits synth)
  forced     own history (forced_prefix): prompt = first 35% of the clean text, corrupted remainder typed as the
             model's own Inserts (evaluate_edits synth_forced)
synth/forced use type-greedy, as the saved benchmark evals. Every trajectory is replayed through the executor and must
reproduce the rollout's final canvas; synth/forced finals are also compared with the saved eval predictions.

Actions are stored as compact ops for the viewer: ["i", piece] insert, ["d"] delete-left, ["m", delta] move, ["s"] stop.
A piece is the token's text, or {"b": [byte, ...]} for a token that holds only part of a multi-byte character.

Usage (WSL, GPU): /opt/venv/bin/python scripts/viz/text_deletion_traces.py --out runs/viz/text_traces.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from reviser_v2.evaluate_edits import load_model, synth_records  # noqa: E402
from reviser_v2.executor import Canvas  # noqa: E402
from reviser_v2.rollout import Request, rollout  # noqa: E402

MODELS = {
    "d0": "data/checkpoints/reviser_100m_actionlm.pt",
    "d1": "runs/restore/re100-restonlyu-s1/runs/re100-restonlyu-s1",
    "d2": "runs/slim/re100-restonly-s1",
    "d3": "runs/slim/re100-d3-s1",
}
SAVED = {  # saved type-greedy benchmark predictions, used only as a fidelity cross-check
    "d0": "runs/zeroshot100m/eval/edits_{s}_dev_tgreedy_lim300.jsonl",
    "d1": "runs/re100-restonlyu-s1/eval/edits_{s}_dev_tgreedy.jsonl",
    "d2": "runs/re100-restonly-s1/eval/edits_{s}_dev_tgreedy.jsonl",
    "d3": "runs/re100-d3-s1/eval/edits_{s}_dev_tgreedy.jsonl",
}


def bad_draft_positions(draft, clean):
    """Draft positions that the minimal diff to the clean text removes or replaces (the errors to delete)."""
    bad = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(a=draft, b=clean, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            bad += range(i1, i2)
    return bad


_BYTE_DEC = None


def piece(tok, t):
    """Decoded text of one GPT-2 token; a token holding part of a multi-byte character is sent as raw bytes {"b": [...]}
    so the viewer can join it with its neighbours instead of showing U+FFFD."""
    global _BYTE_DEC
    s = tok.decode([t])
    if "�" not in s:
        return s
    if _BYTE_DEC is None:  # GPT-2 byte-level alphabet (bytes_to_unicode), inverted
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _BYTE_DEC = {chr(c): b for b, c in zip(bs, cs)}
    return {"b": [_BYTE_DEC[ch] for ch in tok.convert_ids_to_tokens(t)]}


def ops_and_check(space, tok, draft, actions, final):
    c = Canvas.from_draft(space, draft)
    ops = []
    for a in actions:
        if space.is_content(a):
            ops.append(["i", piece(tok, a)])
        elif a == space.DELETE:
            ops.append(["d"])
        elif a == space.STOP:
            ops.append(["s"])
        elif space.is_move(a):
            ops.append(["m", space.move_value(a)])
        else:
            raise SystemExit(f"unexpected action {a}")
        c.apply(a)
    if c.tokens != list(final):
        raise SystemExit("replay mismatch")
    return ops


def pick(rng, cands, k):
    return sorted(rng.sample(cands, min(k, len(cands))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "viz" / "text_traces.json"))
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    rng = random.Random(7)

    # --- record selection (first 300 = the D0 saved-eval range; short so the viewer stays readable)
    synth = synth_records("synth", 300, tok=tok)
    forced = synth_records("synth_forced", 300, tok=tok)
    s_idx = []
    for ne in (1, 2, 3):
        s_idx += pick(rng, [i for i, r in enumerate(synth) if r["intent"] == f"synth{ne}" and len(r["before_ids"]) <= 40], 2)
    s_idx += pick(rng, [i for i, r in enumerate(synth) if r["intent"] == "synth0" and len(r["before_ids"]) <= 40], 1)
    f_idx = []
    for ne in (1, 2, 3):
        f_idx += pick(rng, [i for i, r in enumerate(forced) if r["intent"] == f"synth{ne}"
                            and len(r["before_ids"]) <= 36 and len(r["instruction_ids"]) <= 24], 2)
    f_idx += pick(rng, [i for i, r in enumerate(forced) if r["intent"] == "synth0"
                        and len(r["before_ids"]) <= 36 and len(r["instruction_ids"]) <= 24], 1)
    from reviser_v2.evaluate_continuation import continuation_prompts
    conts = continuation_prompts(300)
    c_idx = pick(rng, [i for i, p in enumerate(conts) if len(p["prompt_ids"]) + len(p["ref_ids"]) <= 64], a.per - 2)

    scen = {
        "continue": [dict(uid=conts[i]["id"], prompt=conts[i]["prompt_ids"], draft=[], clean=conts[i]["ref_ids"], intent="continue")
                     for i in c_idx],
        "synth": [dict(uid=synth[i]["uid"], prompt=synth[i]["instruction_ids"], draft=synth[i]["before_ids"],
                       clean=synth[i]["after_ids"], intent=synth[i]["intent"]) for i in s_idx],
        "forced": [dict(uid=forced[i]["uid"], prompt=forced[i]["instruction_ids"], draft=forced[i]["before_ids"],
                        clean=forced[i]["after_ids"], intent=forced[i]["intent"]) for i in f_idx],
    }
    examples = {s: [dict(uid=r["uid"], intent=r["intent"], prompt=tok.decode(r["prompt"]),
                         draft=[piece(tok, t) for t in r["draft"]], clean=tok.decode(r["clean"]),
                         bad=bad_draft_positions(r["draft"], r["clean"]) if r["draft"] else [], runs={})
                    for r in rs] for s, rs in scen.items()}

    saved = {}
    for m in MODELS:
        for s, key in (("synth", "synth"), ("forced", "synth_forced")):
            p = ROOT / SAVED[m].format(s=key)
            saved[(m, s)] = {json.loads(l)["uid"]: json.loads(l)["pred"] for l in open(p, encoding="utf-8")} if p.exists() else {}

    fidelity = {"compared": 0, "identical": 0, "differs": []}
    n_replayed = 0
    for m, ck in MODELS.items():
        t0 = time.time()
        model, space, meta = load_model(str(ROOT / ck), a.device)
        for s, rs in scen.items():
            if s == "continue":
                reqs = [Request(prompt=list(r["prompt"]), max_actions=a.budget) for r in rs]
                gen = torch.Generator(device=a.device).manual_seed(0)
                tr = rollout(model, space, reqs, device=a.device, greedy=False, temperature=0.9, top_p=0.95, generator=gen)
            else:
                mode = "conditioning" if s == "synth" else "forced_prefix"
                reqs = [Request(prompt=list(r["prompt"]), draft=list(r["draft"]), draft_mode=mode,
                                max_actions=3 * len(r["draft"]) + 64) for r in rs]
                tr = rollout(model, space, reqs, device=a.device, greedy=True, type_greedy=True)
            for r, ex, t in zip(rs, examples[s], tr):
                ops = ops_and_check(space, tok, r["draft"], t.actions, t.final)
                n_replayed += 1
                pred = tok.decode(t.final)
                if s != "continue":
                    ref = saved[(m, s)].get(r["uid"])
                    if ref is not None:
                        fidelity["compared"] += 1
                        if ref == pred:
                            fidelity["identical"] += 1
                        else:
                            fidelity["differs"].append(f"{m}:{r['uid']}")
                ex["runs"][m] = dict(ops=ops, final=pred, stopped=t.stopped, exact=(list(t.final) == list(r["clean"])) if s != "continue" else None,
                                     n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move, self_del=t.self_deletes)
        print(f"{m}: {ck} step {meta.get('step')} done in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    out = dict(meta=dict(models=MODELS, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         decoding=dict(continue_="sampling T=0.9 top-p 0.95 seed 0", synth="type-greedy", forced="type-greedy"),
                         budget=a.budget, replayed=n_replayed, fidelity=fidelity, device=a.device, torch=torch.__version__),
               examples=examples)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"replayed {n_replayed}; saved-pred identical {fidelity['identical']}/{fidelity['compared']}; wrote {a.out}")


if __name__ == "__main__":
    main()
