#!/usr/bin/env python3
"""Record full action traces of the Part 2 (Reviser-Edit) models on fixed eval sets, for the trajectory check
(notes/trajectory-check-2026-10-09.md) and the viewer (docs/demo/paper-model-trajectories.html).

Decoding is the paper-eval setting of src/reviser_v2/evaluate_edits.py: type-greedy (deterministic, no seed),
max_extra 64, task tokens auto, batch 64 (the clean-draft set uses batch 32 as scripts/clean_draft_discrim.py).
Records are built exactly as evaluate_edits builds them (same files, same synthetic seeds). The only difference is
that every action sequence is kept. Each finished file is checked two ways:
  * replay: the actions, executed by reviser_v2.executor on the draft (edit) or blank canvas (ar), give the final text;
  * fidelity: the decoded final text is compared with the saved eval prediction for the same record
    (runs/<run>/eval/edits_<data>_<split>_tgreedy[_ar].jsonl), when that file exists.

Output: runs/trajviz/part2/<model>/<set>.jsonl (+ .done with timing and fidelity). Existing finished files are skipped.

  wsl.exe -d Reviser-Ubuntu -- /opt/venv/bin/python scripts/trajviz/edit_decode.py [--models edit300 ...] [--sets coedit ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

import reviser_v2.evaluate_edits as EE  # noqa: E402
from reviser_v2.evaluate_edits import load_records, predict, synth_records  # noqa: E402
from reviser_v2.executor import replay  # noqa: E402

_CACHE = {}


def cached_load_model(ckpt, device):
    """evaluate_edits.load_model, loaded once per checkpoint, via CPU, keeping only the weights on the GPU (the saved
    optimizer state of a 300M run is 2.4 GB and does not fit next to the model on an 8 GB card). Same weights, same
    model class; predict() only reads ck["step"] from the returned dict."""
    from reviser_v2.actions import ActionSpace
    from reviser_v2.model import ActionLM, ModelConfig
    key = (str(ckpt), str(device))
    if key not in _CACHE:
        _CACHE.clear()
        torch.cuda.empty_cache()
        p = Path(ckpt)
        if p.is_dir():
            p = p / "ckpt" / "latest.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        space = ActionSpace.from_dict(ck["action_space"])
        model = ActionLM(ModelConfig(**ck["model_config"]))
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        _CACHE[key] = (model, space, {"step": ck.get("step"), "model_config": ck["model_config"]})
        del ck
    return _CACHE[key]


EE.load_model = cached_load_model  # predict() looks load_model up in its module at call time

CFG = ROOT / "configs" / "trajviz" / "models.json"
OUT = ROOT / "runs" / "trajviz" / "part2"


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_records(s, tok, ckpt):
    """Records as evaluate_edits.main / clean_draft_discrim.main build them (task tokens auto)."""
    from reviser_v2.tasktokens import apply_task_tokens, resolve_task_tokens
    data, split = s["data"], s["split"]
    draft_mode = "conditioning"
    if data in ("synth", "synth_forced"):
        recs = synth_records(data, 1000, tok=tok)
        draft_mode = "forced_prefix" if data == "synth_forced" else "conditioning"
    elif data in ("synth_real", "infill", "infill_pos", "infill2", "infill2_pos"):
        from reviser_v2.evalsets import eval_set_records
        recs = eval_set_records(data, 0, tok=tok)
    else:
        recs = load_records(data, split)
    if s.get("clean"):  # clean_draft_discrim: changed records only, reference given as the draft
        recs = [r for r in recs if r["before_ids"] != r["after_ids"]]
    tt = resolve_task_tokens("auto", ckpt)
    if tt is not None:
        recs = apply_task_tokens(recs, tt, data)
    if s.get("clean"):
        recs = [dict(r, before_ids=list(r["after_ids"]), before_text=r["after_text"]) for r in recs]
    return recs, draft_mode, tt is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clean-batch", type=int, default=32)
    ap.add_argument("--hash", action="store_true", help="sha256 the checkpoint file (slow on a mounted Windows drive)")
    a = ap.parse_args()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    for m in cfg["part2"]:
        if a.models and m["key"] not in a.models:
            continue
        ck = str(ROOT / m["ckpt"])
        odir = OUT / m["key"]
        odir.mkdir(parents=True, exist_ok=True)
        if a.hash and not (odir / "ckpt.sha256").exists():
            t0 = time.time()
            (odir / "ckpt.sha256").write_text(sha256(ck) + "\n")
            print(m["key"], "sha256", (odir / "ckpt.sha256").read_text().strip(), f"{time.time()-t0:.0f}s", flush=True)
        for s in cfg["part2_sets"]:
            if a.sets and s["key"] not in a.sets:
                continue
            out = odir / f"{s['key']}.jsonl"
            if Path(str(out) + ".done").exists():
                print("skip", out, flush=True)
                continue
            recs, draft_mode, tt = build_records(s, tok, ck)
            bs = a.clean_batch if s.get("clean") else a.batch
            t0 = time.time()
            traces, meta = predict(recs, ck, a.device, bs, 0.0, True, 0, 64, m["layout"], draft_mode)
            secs = time.time() - t0
            space = cached_load_model(ck, a.device)[1]
            saved = {}
            if not s.get("clean") and m.get("saved_eval"):
                sp = ROOT / m["saved_eval"] / (f"edits_{s['data']}_{s['split']}_tgreedy" + ("_ar" if m["layout"] == "ar" else "") + ".jsonl")
                if sp.exists():
                    for line in open(sp, encoding="utf-8"):
                        r = json.loads(line)
                        saved[r["uid"]] = r["pred"]
            n_same = n_cmp = n_replay_ok = 0
            tmp = Path(str(out) + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r, t in zip(recs, traces):
                    start = [] if m["layout"] == "ar" else list(r["before_ids"])
                    try:
                        c = replay(space, t.actions, start)
                        ok = c.tokens == list(t.final)
                    except Exception:
                        ok = False
                    n_replay_ok += ok
                    pred = tok.decode(t.final)
                    same = None
                    if r["uid"] in saved:
                        n_cmp += 1
                        same = saved[r["uid"]] == pred
                        n_same += same
                    f.write(json.dumps(dict(uid=r["uid"], intent=r.get("intent"), task=r.get("task"),
                                            instruction_ids=list(r["instruction_ids"]), before_ids=list(r["before_ids"]),
                                            after_ids=list(r["after_ids"]), actions=list(t.actions), final=list(t.final),
                                            stopped=t.stopped, replay_ok=ok, same_as_saved=same)) + "\n")
            os.replace(tmp, out)
            info = dict(model=m["key"], set=s["key"], n=len(recs), seconds=round(secs, 1), batch=bs, layout=m["layout"],
                        draft_mode=draft_mode, task_tokens=tt, decoding="type-greedy, max_extra 64 (evaluate_edits)",
                        replay_ok=n_replay_ok, saved_compared=n_cmp, saved_identical=n_same,
                        space=space.describe() if space else None, ckpt=m["ckpt"], ckpt_step=meta.get("ckpt_step"),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, torch=torch.__version__,
                        code_sha256=sha256(__file__), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            Path(str(out) + ".done").write_text(json.dumps(info))
            print(json.dumps(info), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
