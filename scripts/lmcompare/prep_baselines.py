#!/usr/bin/env python3
"""Stage the paper's stored baseline texts for the LM comparison (helper M, notes/lm-comparison-2026-10-10.md).

Sources (read-only, never modified): <archive>/workspaces/4-21/all_rollouts_3k/<key>.json, the files the
paper's arena tables were judged from (list of {id, prompt_text, full_text}).
- Byte copies, renamed to the model they hold (the sedd_absorb_{100m,300m} files are one file, louaaron/sedd-medium,
  353M; mdlm_{100m,300m} likewise, kuleshov-group/mdlm-owt, 170M). sha256 of copy == source is asserted.
- SEDD Small (louaaron/sedd-small, 169M): the stored file has 4200 rows, an empty prompt_text and a full_text that is
  the continuation only (workspaces/4-18/sedd_small_arena/run_sedd_small_arena.py judged it that way: Reviser full
  text vs SEDD continuation). Two files are written:
    sedd_small_169m.json             byte copy, as stored (for the reproduction check only);
    sedd_small_169m_withprompt.json  the 3000 paper ids, prompt_text = the paper prompt (GPT-2 decode of the 35
                                     prompt ids, as in scripts/papereval/common.py), full_text = prompt_text + stored
                                     continuation; every other baseline file has this form.
  Checks: the stored continuation equals the 4-18 per-seed cont_text; GPT-2 decode of its cont_token_ids equals it.
- Coverage: every file holds the 3000 ids of data/papereval/c4_paper_3k.json and its prompt_text equals the paper
  prompt for all 3000 (asserted; SEDD Small with the prompt added).
Writes runs/lmcompare/baselines/{*.json, MANIFEST.json, SHA256SUMS}.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "papereval"))
import common as C  # noqa: E402

HIST = Path("<archive>/workspaces/4-21/all_rollouts_3k")
SEDD_SEEDS = Path("<archive>/workspaces/4-18/sedd_small_arena")
OUT = ROOT / "runs" / "lmcompare" / "baselines"

# out name -> (source key, HF id, note)
COPIES = {
    "sedd_absorb_353m": ("sedd_absorb_100m", "louaaron/sedd-medium", "identical bytes to sedd_absorb_300m.json; 128 steps"),
    "mdlm_170m": ("mdlm_100m", "kuleshov-group/mdlm-owt", "identical bytes to mdlm_300m.json; 128 steps"),
    "sedd_small_169m": ("sedd_small_169m", "louaaron/sedd-small", "as stored: 4200 rows, continuation only; 128 steps"),
    "cerebras_111m": ("cerebras_111m", "cerebras/Cerebras-GPT-111M", ""),
    "cerebras_256m": ("cerebras_256m", "cerebras/Cerebras-GPT-256M", ""),
    "pythia_160m": ("pythia_160m", "EleutherAI/pythia-160m", ""),
    "pythia_410m": ("pythia_410m", "EleutherAI/pythia-410m", ""),
    "gpt2_117m": ("gpt2_117m", "openai-community/gpt2", ""),
    "gpt2_medium_345m": ("gpt2_medium_345m", "openai-community/gpt2-medium", ""),
    "opt_125m": ("opt_125m", "facebook/opt-125m", ""),
    "opt_350m": ("opt_350m", "facebook/opt-350m", ""),
    # the paper's own stored Reviser texts (reproduction checks only)
    "reviser_100m": ("reviser_100m", "released Reviser 100M", "paper rollouts, repeated prompt kept as stored"),
    "reviser_300m": ("reviser_300m", "released Reviser 300M", "paper rollouts, repeated prompt kept as stored"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = C.gpt2_tok()
    dec = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)  # noqa: E731
    P = json.loads((ROOT / "data/papereval/c4_paper_3k.json").read_text(encoding="utf-8"))
    ptxt = {p["id"]: dec(p["token_ids"][:35]) for p in P}
    man = {"prompts": "data/papereval/c4_paper_3k.json", "prompts_sha256": sha(ROOT / "data/papereval/c4_paper_3k.json"),
           "builder": "scripts/lmcompare/prep_baselines.py", "builder_sha256": sha(__file__), "files": {}}
    for name, (key, hf, note) in COPIES.items():
        src = HIST / f"{key}.json"
        dst = OUT / f"{name}.json"
        shutil.copyfile(src, dst)
        hs, hd = sha(src), sha(dst)
        assert hs == hd, name
        rows = json.loads(dst.read_text(encoding="utf-8"))
        ids = {r["id"] for r in rows}
        assert set(ptxt) <= ids, (name, len(set(ptxt) - ids))
        n_prompt_eq = sum(r["prompt_text"] == ptxt[r["id"]] for r in rows if r["id"] in ptxt)
        man["files"][name] = {"source": str(src), "sha256": hd, "model": hf, "note": note, "n_rows": len(rows),
                              "n_paper_ids": len(set(ptxt) & ids), "prompt_eq_paper_prompt": n_prompt_eq}
        if name != "sedd_small_169m":
            assert n_prompt_eq == 3000 and len(rows) == 3000, (name, n_prompt_eq, len(rows))
    # SEDD Small with the prompt added
    stored = {r["id"]: r for r in json.loads((HIST / "sedd_small_169m.json").read_text(encoding="utf-8"))}
    seedrows = {}
    for s in (123, 124, 125):
        for r in json.loads((SEDD_SEEDS / f"seed{s}" / "sedd_small_outputs.json").read_text(encoding="utf-8")):
            seedrows[r["id"]] = r
    out, n_eq_seed, n_eq_ids = [], 0, 0
    for i in sorted(ptxt):
        r = stored[i]
        assert r["prompt_text"] == ""
        n_eq_seed += r["full_text"] == seedrows[i]["cont_text"]
        n_eq_ids += dec(seedrows[i]["cont_token_ids"]) == r["full_text"]
        out.append({"id": i, "prompt_text": ptxt[i], "full_text": ptxt[i] + r["full_text"]})
    assert n_eq_seed == 3000, n_eq_seed
    dst = OUT / "sedd_small_169m_withprompt.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    man["files"]["sedd_small_169m_withprompt"] = {
        "source": f"{HIST / 'sedd_small_169m.json'} (+ prompt)", "sha256": sha(dst), "model": "louaaron/sedd-small",
        "note": "3000 paper ids; full_text = paper prompt + stored continuation", "n_rows": len(out),
        "stored_cont_eq_seed_file_cont_text": n_eq_seed, "stored_cont_eq_decode_of_seed_cont_token_ids": n_eq_ids}
    # continuation lengths (GPT-2 tokens, common.load_texts)
    for name in man["files"]:
        if name == "sedd_small_169m":
            continue
        rows, _ = C.load_texts(OUT / f"{name}.json", tok)
        rows = [r for r in rows if r["id"] in ptxt]
        man["files"][name]["mean_cont_tokens"] = sum(len(r["cont_ids"]) for r in rows) / len(rows)
        man["files"][name]["prefix_intact"] = sum(r["prefix_intact"] for r in rows)
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    with open(OUT / "SHA256SUMS", "w", newline="\n") as f:
        for p in sorted(OUT.glob("*.json")):
            if p.name != "MANIFEST.json":
                f.write(f"{sha(p)}  {p.name}\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("sha256", "n_rows", "mean_cont_tokens") if kk in v}
                      for k, v in man["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
