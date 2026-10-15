"""Loaders from raw downloaded files (data/real/raw/, see scripts/data/download_realdata.py) to
normalized text records. Official splits are kept verbatim; nothing is moved between splits here.

Record fields: uid, source, dataset, split, doc_id, instruction, intent, before_text, after_text,
meta. ``before_text``/``after_text`` are the official strings with leading/trailing whitespace
stripped (IteraTeR sentence splitting leaves stray leading spaces); ``meta.stripped`` records whether
stripping changed either side. ``uid = dataset/split/line_index`` points back at the raw line.

IteraTeR instructions are OUR rendering of the intent label (IteraTeR has no instruction text).
CoEdIT ``src`` is ``"<instruction>: <input>"``; it is split at the first ``": "``.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterator, List, Tuple

from .dedup import norm_text

# Rendered instruction per IteraTeR intent label (sentence level).
ITERATER_INSTRUCTIONS = {
    "clarity": "Make this text clearer",
    "fluency": "Improve the fluency of this text",
    "coherence": "Improve the coherence of this text",
    "style": "Improve the style of this text",
    "meaning-changed": "Update the content of this text",
    "others": "Revise this text",
}
_DOC_PHRASES = {
    "clarity": "clarity", "fluency": "fluency", "coherence": "coherence", "style": "style",
    "meaning-changed": "content updates", "others": "other edits",
}
_INTENT_ORDER = ["clarity", "fluency", "coherence", "style", "meaning-changed", "others"]


def doc_instruction(intents: List[str]) -> str:
    """Instruction for a document-level revision carrying several edit intents."""
    ints = [i for i in _INTENT_ORDER if i in set(intents)]
    if not ints:
        return "Revise this text"
    ph = [_DOC_PHRASES[i] for i in ints]
    body = ph[0] if len(ph) == 1 else ", ".join(ph[:-1]) + " and " + ph[-1]
    return f"Revise this text for {body}"


def _jsonl(path: str) -> Iterator[Tuple[int, Dict]]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def _rec(dataset, source, split, idx, doc_id, instruction, intent, before, after, meta) -> Dict:
    b, a = before.strip(), after.strip()
    meta = dict(meta)
    meta["stripped"] = (b != before) or (a != after)
    return {"uid": f"{dataset}/{split}/{idx}", "source": source, "dataset": dataset, "split": split,
            "doc_id": doc_id, "instruction": instruction, "intent": intent,
            "before_text": b, "after_text": a, "meta": meta}


# dataset name -> (raw subdir, source tag, {official split name: file})
ITERATER_SENT = {
    "iterater_human_sent": ("hf_IteraTeR_human_sent", "IteraTeR (human-annotated intents, sentence level)",
                            {"train": "train.json", "dev": "dev.json", "test": "test.json"}),
    "iterater_full_sent": ("hf_IteraTeR_full_sent", "IteraTeR (full, classifier-predicted intents, sentence level)",
                           {"train": "train.json", "dev": "dev.json", "test": "test.json"}),
    "iterater_v2_sent": ("hf_IteraTeR_v2", "IteraTeR_v2 / R3 (classifier-predicted intents, sentence level)",
                         {"train": "sent-level.train.intents.json", "dev": "sent-level.dev.intents.json",
                          "test": "sent-level.test.intents.json", "test4k": "sent-level.test.intents.4K.json"}),
}
ITERATER_DOC = {
    "iterater_human_doc": ("hf_IteraTeR_human_doc", "IteraTeR (human-annotated intents, document level)",
                           {"train": "train.json", "dev": "dev.json", "test": "test.json"}),
    "iterater_full_doc": ("hf_IteraTeR_full_doc", "IteraTeR (full, predicted intents, document level)",
                          {"train": "train.json", "dev": "dev.json", "test": "test.json"}),
}
COEDIT = {"coedit": ("hf_coedit", "CoEdIT public release (grammarly/coedit)",
                     {"train": "train.jsonl", "validation": "validation.jsonl"})}

DATASETS = list(ITERATER_SENT) + list(ITERATER_DOC) + list(COEDIT)
EVAL_SPLITS = {"dev", "test", "test4k", "validation"}


def dataset_splits(name: str) -> List[str]:
    for table in (ITERATER_SENT, ITERATER_DOC, COEDIT):
        if name in table:
            return list(table[name][2])
    raise KeyError(name)


def coedit_group_id(input_text: str) -> str:
    return "coedit:" + hashlib.sha1(norm_text(input_text).encode("utf-8")).hexdigest()[:16]


def load_split(raw_root: str, name: str, split: str) -> List[Dict]:
    if name in ITERATER_SENT:
        sub, src, files = ITERATER_SENT[name]
        out = []
        for i, r in _jsonl(os.path.join(raw_root, sub, files[split])):
            lab = r["labels"]
            meta = {"revision_depth": int(r["revision_depth"])}
            if "confidence" in r:
                meta["intent_confidence"] = float(r["confidence"])
            out.append(_rec(name, src, split, i, str(r["doc_id"]), ITERATER_INSTRUCTIONS[lab], lab,
                            r["before_sent"], r["after_sent"], meta))
        return out
    if name in ITERATER_DOC:
        sub, src, files = ITERATER_DOC[name]
        out = []
        for i, r in _jsonl(os.path.join(raw_root, sub, files[split])):
            ints = sorted({a.get("major_intent") for a in r.get("edit_actions") or [] if a.get("major_intent")},
                          key=lambda x: _INTENT_ORDER.index(x) if x in _INTENT_ORDER else 99)
            meta = {"revision_depth": int(r["revision_depth"]), "n_official_edit_actions": len(r.get("edit_actions") or [])}
            if r.get("domain"):
                meta["domain"] = r["domain"]
            out.append(_rec(name, src, split, i, str(r["doc_id"]), doc_instruction(ints), "+".join(ints) or "none",
                            r["before_revision"], r["after_revision"], meta))
        return out
    if name in COEDIT:
        sub, src, files = COEDIT[name]
        out = []
        for i, r in _jsonl(os.path.join(raw_root, sub, files[split])):
            instr, inp = r["src"].split(": ", 1)
            out.append(_rec(name, src, split, i, coedit_group_id(inp), instr.strip(), r["task"], inp, r["tgt"],
                            {"coedit_id": str(r["_id"])}))
        return out
    raise KeyError(name)
