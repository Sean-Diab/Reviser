#!/usr/bin/env python
"""Zero-shot instruction-tuned small LMs as REFERENCE editors (not size- or data-matched) on the Reviser-Edit eval
sets (notes/edit-extras-2026-10-10.md, task 4).

Models (HF, pinned revisions, safetensors): Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct,
HuggingFaceTB/SmolLM2-360M-Instruct.

Sets: those of scripts/remote/editv2_eval.sh (CoEdIT validation, IteraTeR human sentence test, IteraTeR v2 test4k,
synth dev, synth_real dev; synth_forced is an own-history set and is skipped as for the AR baselines), the clean-draft
test of scripts/clean_draft_discrim.py --decode (records with before != after, the reference given as the draft) on
the three real sets, and JFLEG test (raw and detok variants of scripts/editextras/jfleg_eval.py, grammar instruction
"Fix grammatical errors in this sentence").

Prompt (fixed, one per model = the model's own chat template, default system prompt, one user turn):
    "{instruction}:\n\n{text}\n\nReturn only the revised text, with no explanation."
where {instruction} is the record's own instruction (CoEdIT / IteraTeR; synth: "Fix the errors in this text";
synth_real: its record instruction), add_generation_prompt=True. Greedy (do_sample=False, 1 beam), fp16 weights,
batch 32 sorted by prompt length, left padding (fp32 if more than 10% of the first set's outputs are empty or "!!!!",
the fp16-overflow signature; exit 5 then and the launcher reruns with --dtype fp32), max_new_tokens = min(512, 2 * longest input in the batch (model
tokens) + 32).

Post-processing (deterministic, counts recorded per set in the summary meta):
  1. strip; 2. drop a first line that is a preamble ending in ":" (regex PRE_LINE) when more text follows;
  3. drop an inline label prefix (regex PRE_INLINE: "Revised text:", "Corrected sentence:", ...);
  4. remove one pair of wrapping quotes (" or ' or curly) when the input itself is not wrapped in them;
  5. if the input has no blank line but the output does, keep the part before the first blank line (drops trailing
     notes); 6. if the cleaned output equals the stripped input, the prediction is the input exactly (unchanged);
     otherwise the input's own leading/trailing whitespace is put back around it.
The cleaned text is re-tokenised with GPT-2 (as reviser_v2.evaluate_external) and scored by
reviser_v2.evaluate_edits.score / aggregate, so SARI / EM / changed are directly comparable with the model rows.

  /opt/venv/bin/python scripts/editextras/lm_editors.py --model qwen05 [--sets all] [--limit N]
Outputs: runs/edit_extras/lm_editors/<model>/edits_<data>_<split>_greedy.{jsonl,summary.json}, clean_draft.json,
and runs/edit_extras/jfleg/hyp/<model>.<variant>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS = {
    "qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen15": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "smol360": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
}
TEMPLATE = "{instruction}:\n\n{text}\n\nReturn only the revised text, with no explanation."
PRE_LINE = re.compile(r"^(sure|certainly|okay|ok|of course|here|below|the (revised|corrected|edited|improved|rewritten|"
                      r"paraphrased|updated|simplified|neutral|clearer|fixed)|revised|corrected|edited|improved|rewritten|"
                      r"paraphrased|updated|simplified)[^\n]{0,120}:\s*$", re.I)
PRE_INLINE = re.compile(r"^(\*\*)?(the )?(revised|corrected|edited|improved|rewritten|paraphrased|updated|simplified|"
                        r"neutral(ized)?|fixed|final)( (text|sentence|version|paragraph))?:(\*\*)?\s+", re.I)
QUOTES = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]


def clean(out: str, inp: str, counts: dict):
    t = out.strip()
    lines = t.split("\n")
    if len(lines) > 1 and PRE_LINE.match(lines[0].strip()):
        t = "\n".join(lines[1:]).strip()
        counts["pre_line"] = counts.get("pre_line", 0) + 1
    m = PRE_INLINE.match(t)
    if m:
        t = t[m.end():].strip()
        counts["pre_inline"] = counts.get("pre_inline", 0) + 1
    s = inp.strip()
    for a, b in QUOTES:
        if len(t) >= 2 and t.startswith(a) and t.endswith(b) and not (s.startswith(a) and s.endswith(b)):
            t = t[len(a):len(t) - len(b)].strip()
            counts["quotes"] = counts.get("quotes", 0) + 1
            break
    if "\n\n" not in s and "\n\n" in t:
        t = t.split("\n\n")[0].strip()
        counts["blank_cut"] = counts.get("blank_cut", 0) + 1
    if t == s:
        return inp, True
    lead = inp[:len(inp) - len(inp.lstrip())]
    trail = inp[len(inp.rstrip()):]
    return lead + t + trail, False


class Gen:
    def __init__(self, key, batch, device="cuda", dtype="fp16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name, rev = MODELS[key]
        self.name, self.rev, self.batch, self.device = name, rev, batch, device
        self.tk = AutoTokenizer.from_pretrained(name, revision=rev)
        self.tk.padding_side = "left"
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token
        dt = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=dt).to(device).eval()

    def prompt(self, instruction, text):
        msgs = [{"role": "user", "content": TEMPLATE.format(instruction=instruction.strip().rstrip(":."), text=text)}]
        return self.tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def run(self, items):
        """items: [(instruction, text)] -> raw generated strings"""
        ps = [self.prompt(i, t) for i, t in items]
        lens = [len(self.tk(t, add_special_tokens=False).input_ids) for _, t in items]
        order = sorted(range(len(ps)), key=lambda i: len(ps[i]))
        outs = [None] * len(ps)
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            enc = self.tk([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            mx = min(512, 2 * max(lens[i] for i in idx) + 32)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=mx, do_sample=False, num_beams=1,
                                        pad_token_id=self.tk.pad_token_id, temperature=None, top_p=None, top_k=None)
            new = g[:, enc.input_ids.shape[1]:]
            for i, s in zip(idx, self.tk.batch_decode(new, skip_special_tokens=True)):
                outs[i] = s
        return outs


def edit_sets(tok, limit):
    from reviser_v2.evaluate_edits import load_records, synth_records
    from reviser_v2.evalsets import eval_set_records
    out = []
    for data, split in (("coedit", "validation"), ("iterater_human_sent", "test"), ("iterater_v2_sent", "test4k")):
        out.append((data, split, load_records(data, split, limit)))
    out.append(("synth", "dev", synth_records("synth", limit or 1000, tok=tok)))
    out.append(("synth_real", "dev", eval_set_records("synth_real", limit, tok=tok)))
    return out


def instr_of(r, tok):
    return r.get("instruction") or tok.decode(r["instruction_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--sets", default="edit,clean,jfleg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    from transformers import GPT2TokenizerFast
    from reviser_v2.evaluate_edits import aggregate, load_records, score
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    od = Path(a.out) if a.out else ROOT / "runs" / "edit_extras" / "lm_editors" / a.model
    od.mkdir(parents=True, exist_ok=True)
    gen = Gen(a.model, a.batch, a.device, a.dtype)
    base_meta = dict(model=gen.name, revision=gen.rev, dtype=a.dtype, greedy=True, beams=1, batch=a.batch, template=TEMPLATE,
                     chat_template="tokenizer.apply_chat_template(add_generation_prompt=True), default system prompt",
                     max_new="min(512, 2 * longest input in batch + 32)",
                     gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu")
    todo = a.sets.split(",")
    if "edit" in todo:
        for data, split, recs in edit_sets(tok, a.limit):
            t0 = time.time()
            raw = gen.run([(instr_of(r, tok), r["before_text"]) for r in recs])
            bad = sum((not o.strip()) or ("!!!!" in o) for o in raw) / len(raw)
            if bad > 0.1:  # fp16 overflow guard: empty / "!!!!" outputs; the launcher reruns with --dtype fp32
                print(f"FP_GUARD {data} bad outputs {bad:.3f}", flush=True)
                sys.exit(5)
            counts, finals, extra = {}, [], []
            for r, o in zip(recs, raw):
                txt, same = clean(o, r["before_text"], counts)
                finals.append(list(r["before_ids"]) if same else tok.encode(txt))
                extra.append(dict(raw_output=o))
            rows = score(recs, finals, tok, extra)
            summ = aggregate(rows)
            name = f"edits_{data}_{split}_greedy" + (f"_lim{a.limit}" if a.limit else "")
            meta = dict(base_meta, data=data, split=split, n=len(recs), clean_counts=counts, seconds=time.time() - t0)
            with open(od / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            json.dump({"meta": meta, "summary": summ}, open(od / f"{name}.summary.json", "w"), indent=1)
            s = summ["ALL"]
            print(name, f"sari {s['sari']:.2f} exact {s['exact']:.3f} changed {s['changed']:.3f}", counts,
                  f"{meta['seconds']:.0f}s", flush=True)
    if "clean" in todo:
        res = {"meta": dict(base_meta), "sets": {}}
        for data, split in (("iterater_human_sent", "test"), ("iterater_v2_sent", "test4k"), ("coedit", "validation")):
            recs = [r for r in load_records(data, split, a.limit) if r["before_ids"] != r["after_ids"]]
            t0 = time.time()
            raw = gen.run([(instr_of(r, tok), r["after_text"]) for r in recs])
            counts, unch = {}, []
            for r, o in zip(recs, raw):
                txt, same = clean(o, r["after_text"], counts)
                fin = list(r["after_ids"]) if same else tok.encode(txt)
                unch.append(fin == r["after_ids"])
            res["sets"][f"{data}:{split}"] = dict(n=len(recs), decode_unchanged=sum(unch) / len(unch), clean_counts=counts,
                                                  seconds=time.time() - t0)
            with open(od / f"clean_draft_{data}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r, o, u in zip(recs, raw, unch):
                    f.write(json.dumps(dict(uid=r["uid"], unchanged=u, raw_output=o)) + "\n")
            print("clean_draft", data, split, f"{sum(unch) / len(unch):.3f}", counts, flush=True)
        json.dump(res, open(od / "clean_draft.json", "w"), indent=1)
    if "jfleg" in todo:
        jd = ROOT / "runs" / "edit_extras" / "jfleg"
        for v in ("raw", "detok"):
            recs = [json.loads(l) for l in open(jd / f"records_{v}.jsonl", encoding="utf-8")]
            t0 = time.time()
            raw = gen.run([(r["instruction"], r["input"]) for r in recs])
            counts = {}
            hyps = [clean(o, r["input"], counts)[0].replace("\n", " ") for r, o in zip(recs, raw)]
            (jd / "hyp").mkdir(parents=True, exist_ok=True)
            with open(jd / "hyp" / f"{a.model}.{v}.jsonl", "w", encoding="utf-8") as f:
                for r, h, o in zip(recs, hyps, raw):
                    f.write(json.dumps(dict(uid=r["uid"], input=r["input"], hyp=h, raw_output=o)) + "\n")
            json.dump(dict(base_meta, variant=v, clean_counts=counts, seconds=time.time() - t0),
                      open(jd / "hyp" / f"{a.model}.{v}.meta.json", "w"), indent=1)
            print("jfleg", v, counts, flush=True)
    print("LM_EDITORS_DONE", a.model, flush=True)


if __name__ == "__main__":
    main()
