"""First-decision action-type probabilities for an edit model (over-editing diagnostic for finding 22).

For each record, the model sees prompt + supplied draft (conditioning layout, as in evaluate_edits) and we read the
post-legality-mask distribution of the FIRST action, summed by type: Stop / Delete / Insert / Move. A model that has
learned to leave already-acceptable inputs unchanged should put Stop mass there; a constant Stop bias b multiplies the
Stop odds by e^b.
Usage: python scripts/stop_prob_step0.py --ckpt runs/slim/re100-d4-s1 --sets iterater_human_sent:test coedit:validation synth_real:dev [--out J]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reviser_v2.evaluate_edits import load_model, load_records, synth_records  # noqa: E402
from reviser_v2.rollout import Request, _prepare  # noqa: E402


@torch.no_grad()
def first_step(model, space, recs, device, batch=32):
    nc, ms = space.n_content, space.move_start
    out = []
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        prepped = [_prepare(space, Request(prompt=list(r["instruction_ids"]), draft=list(r["before_ids"]))) for r in chunk]
        conds = [p[0] for p in prepped]
        T0 = max(len(c) for c in conds)
        ids = torch.zeros(len(chunk), T0, dtype=torch.long)
        pad = torch.ones(len(chunk), T0, dtype=torch.bool)
        for j, c in enumerate(conds):
            ids[j, T0 - len(c):] = torch.tensor(c)
            pad[j, T0 - len(c):] = False
        pos = (torch.cumsum((~pad).long(), 1) - 1).clamp(min=0)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device != "cpu"):
            logits, _ = model(ids.to(device), positions=pos.to(device), key_pad=pad.to(device) if bool(pad.any()) else None)
        lg = logits[:, -1, :].float()
        spec = torch.tensor([p[1].legal_special_mask() for p in prepped], dtype=torch.bool, device=device)
        lg[:, nc:] = lg[:, nc:].masked_fill(~spec, float("-inf"))
        pr = torch.softmax(lg, -1)
        for j, r in enumerate(chunk):
            out.append(dict(uid=r["uid"], stop=pr[j, space.STOP].item(), delete=pr[j, space.DELETE].item(),
                            insert=pr[j, :nc].sum().item(), move=pr[j, ms:].sum().item(),
                            ref_changed=r["before_ids"] != r["after_ids"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="data:split")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, space, ck = load_model(a.ckpt, dev)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    res = {"ckpt": a.ckpt, "step": ck.get("step"), "sets": {}}
    for s in a.sets:
        data, split = s.split(":")
        if data == "synth":
            recs = synth_records("synth", 1000, tok=tok)  # i % 4 == 0 drafts are clean (0 errors)
        elif data == "synth_real":
            from reviser_v2.evalsets import eval_set_records
            recs = eval_set_records("synth_real", 0, tok=tok)
        else:
            recs = load_records(data, split)
        rows = first_step(model, space, recs, dev)
        st = np.array([r["stop"] for r in rows])
        ch = np.array([r["ref_changed"] for r in rows])
        summ = {"n": len(rows), "ref_unchanged": int((~ch).sum()),
                "stop_mean": float(st.mean()), "stop_median": float(np.median(st)),
                "frac_stop_gt_0.01": float((st > 0.01).mean()), "frac_stop_gt_0.1": float((st > 0.1).mean()),
                "frac_stop_argmax_type": float(np.mean([r["stop"] >= max(r["delete"], r["insert"], r["move"]) for r in rows])),
                "stop_mean_ref_unchanged": float(st[~ch].mean()) if (~ch).any() else None,
                "stop_mean_ref_changed": float(st[ch].mean()) if ch.any() else None,
                "type_mean": {k: float(np.mean([r[k] for r in rows])) for k in ("stop", "delete", "insert", "move")}}
        res["sets"][s] = summ
        print(s, json.dumps(summ), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
