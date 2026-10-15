#!/usr/bin/env python
"""Build the Reviser-Edit v2 training mix (plan steps 8-10, notes/reviser-edit-data-plan-2026-10-06.md; results in
notes/edit-runs-2026-10-08.md). Output: data/editdata/mix_v2/{pools/*.npz, manifest.json, mixes.json}, read by
src/reviser_v2/tasks/editmix.py. Run in WSL (needs numpy, pyarrow, transformers):

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src /opt/venv/bin/python scripts/editdata/build_mix_v2.py --workers 14

Sources (all GPT-2 ids; new texts encoded with add_special_tokens=False, split_special_tokens=True as realdata/build.py):
  real pairs   data/real/{iterater_human_sent,iterater_full_sent,iterater_v2_sent,coedit,iterater_full_doc,
               iterater_human_doc}_train.jsonl
  LLM drafts   data/editdata/full_v1/{grammar,coherence,polish}.jsonl (draft -> final)
  C4_200M      data/c4_200m/train-0.parquet (input -> output), shard 0 only
  no change    clean-screened C4 "llm"-pool paragraphs (full_v1/screen clean p_yes >= 0.9 and odd false) that are not the
               source of a kept draft; half kept as paragraphs, half cut into sentences
  restoration  C4 "restore"-pool paragraphs (data/editdata/c4_paragraphs.jsonl)

Filters for every pair: drop real rows marked dup_of / overlap_v2_eval, unchanged pairs, pairs outside the kept band
(kept = matching GPT-2 tokens / len(after), difflib autojunk=False, in [0.8, 1.0]) and pairs with len(before) +
len(after) > 480 (so the AR layout fits max_len 512); exact duplicates (before, after) are dropped.

Eval separation: a record is dropped when its normalised text, or any sentence of >= 5 words of it, equals a text or
sentence of any data/real/*_{dev,test,test4k,validation}.jsonl record (every September eval set and the synth dev
pools are built from those files).

Category / level:
  IteraTeR sentence  fluency, clarity (confidence >= 0.9) -> same category; coherence (any confidence) -> coherence;
                     style and lower-confidence fluency / clarity -> polish; meaning-changed / others -> not used
  CoEdIT             gec -> grammar; clarity, simplification -> clarity; coherence -> coherence; neutralize,
                     paraphrase -> polish (tasktokens.COEDIT_CATEGORY)
  IteraTeR documents -> polish paragraphs (rows whose only labels are meaning-changed / others are not used)
  LLM drafts         their category, paragraphs; C4_200M grammar sentences

Splits: 98/1/1 train/dev/test by a sha256 hash of a group key (IteraTeR doc_id, C4 document id, CoEdIT / C4_200M
normalised target text), so related records share a split.

Budget (plan section 1; tokens = before + after, no change = 2 x text, restoration = text; counted with repeats):
restoration 15%, no change 10%, grammar 18%, fluency 15%, clarity 15%, coherence 12%, polish 15%, each category 50/50
paragraphs / sentences except fluency and clarity (sentences only: LLM paragraph drafts for them were not made).
A pool is used at most ``--max-views`` times; within a slot, real (human) pools fill first, then LLM / C4_200M pools,
with equal views inside a priority group. The tokens a slot cannot fill go to C4_200M grammar sentences. Variants:
main; sent_only (paragraph pools removed, the rest scaled to the same total); no_rest (restoration removed, scaled).
"""
from __future__ import annotations

import argparse
import difflib
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reviser_v2.tasktokens import COEDIT_CATEGORY, TASKS  # noqa: E402

REAL_SENT = ["iterater_human_sent_train", "iterater_full_sent_train", "iterater_v2_sent_train", "coedit_train"]
REAL_DOC = ["iterater_full_doc_train", "iterater_human_doc_train"]
SHARES = {"restoration": 0.15, "nochange": 0.10, "grammar": 0.18, "fluency": 0.15, "clarity": 0.15, "coherence": 0.12,
          "polish": 0.15}
SENT_ONLY_CATS = ("fluency", "clarity")  # no paragraph data for these (user-approved departure from 50/50)
# slot -> [(priority group, pool)]; lower group fills first
SLOT_POOLS = {
    ("restoration", None): [(0, "restoration")],
    ("nochange", "para"): [(0, "nochange_para")],
    ("nochange", "sent"): [(0, "nochange_sent")],
    ("grammar", "sent"): [(0, "grammar_sent_coedit"), (1, "grammar_sent_c4200m")],
    ("grammar", "para"): [(0, "grammar_para_llm")],
    ("fluency", "sent"): [(0, "fluency_sent_real")],
    ("clarity", "sent"): [(0, "clarity_sent_real")],
    ("coherence", "sent"): [(0, "coherence_sent_real")],
    ("coherence", "para"): [(0, "coherence_para_llm")],
    ("polish", "sent"): [(0, "polish_sent_real")],
    ("polish", "para"): [(0, "polish_para_iterater"), (0, "polish_para_llm")],
}
SPILL_POOL = "grammar_sent_c4200m"
EVAL_SPLITS = ("dev", "test", "test4k", "validation")

_W = re.compile(r"\W+")
_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def norm(t: str) -> str:
    return _W.sub(" ", t.lower()).strip()


def sentences(t: str):
    return [s.strip() for s in _SENT.split(t) if s.strip()]


def sent_keys(t: str):
    return [k for k in (norm(s) for s in sentences(t)) if len(k.split()) >= 5]


def split_of(key: str) -> str:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32
    return "train" if h < 0.98 else ("dev" if h < 0.99 else "test")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def kept(pair) -> float:
    b, a = pair
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(a))


def in_band(k: float) -> bool:
    return 0.8 <= k <= 1.0


class Tok:
    def __init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import GPT2TokenizerFast
        tdir = ROOT / "data" / "tokenizers" / "gpt2"
        self.tok = GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")

    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), 8192):
            out += self.tok(list(texts[i:i + 8192]), add_special_tokens=False, split_special_tokens=True)["input_ids"]
        return out


class EvalKeys:
    def __init__(self):
        self.whole, self.sent, self.files = set(), set(), []
        for p in sorted(glob.glob(str(ROOT / "data" / "real" / "*.jsonl"))):
            name = Path(p).stem
            if not any(name.endswith("_" + s) for s in EVAL_SPLITS):
                continue
            self.files.append(name)
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    for t in (r["before_text"], r["after_text"]):
                        self.whole.add(norm(t))
                        self.sent.update(sent_keys(t))
        self.whole.discard("")

    def hit(self, *texts) -> bool:
        for t in texts:
            if norm(t) in self.whole or any(k in self.sent for k in sent_keys(t)):
                return True
        return False


class Builder:
    def __init__(self, a):
        self.a = a
        self.pools = defaultdict(list)  # pool -> [dict(uid, split, b, a, instr)]
        self.drops = defaultdict(Counter)  # source -> reason -> count
        self.seen = set()
        self.cat_instr = defaultdict(dict)  # cat -> {tuple(ids): text}
        self.inputs = {}
        self.ek = EvalKeys()
        self.tok = Tok()
        self.pool_meta = {}
        self.mp = mp.get_context("fork").Pool(a.workers) if a.workers > 1 else None

    def kept_many(self, pairs):
        if self.mp is None:
            return [kept(p) for p in pairs]
        return self.mp.map(kept, pairs, chunksize=512)

    def note_input(self, p):
        p = Path(p)
        self.inputs[str(p.relative_to(ROOT))] = dict(bytes=p.stat().st_size, sha256=sha256_file(p))

    def add(self, src, pool, uid, group_key, b, a, instr=None, texts=(), kind="edit", band=None):
        """Common filters; returns True if kept."""
        d = self.drops[src]
        if kind == "edit":
            if b == a:
                d["unchanged"] += 1
                return False
            if len(b) + len(a) > self.a.max_pair:
                d["too_long"] += 1
                return False
            if band is not None and not in_band(band):
                d["out_of_band"] += 1
                return False
        b, a = np.asarray(b, dtype=np.uint16), np.asarray(a, dtype=np.uint16)
        key = hashlib.sha1(b.tobytes() + b"|" + a.tobytes()).digest() + kind.encode()
        if key in self.seen:
            d["duplicate"] += 1
            return False
        if self.ek.hit(*texts):
            d["eval_overlap"] += 1
            return False
        self.seen.add(key)
        self.pools[pool].append(dict(uid=uid, split=split_of(group_key), b=b, a=a, instr=instr))
        d["kept"] += 1
        return True

    # ---------------------------------------------------------------- real pairs
    def real(self):
        conf_min = self.a.conf
        for name in REAL_SENT + REAL_DOC:
            p = ROOT / "data" / "real" / f"{name}.jsonl"
            self.note_input(p)
            rows = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("dup_of") is not None:
                        self.drops[name]["dup_of"] += 1
                        continue
                    if r.get("overlap_v2_eval"):
                        self.drops[name]["overlap_v2_eval"] += 1
                        continue
                    b, a = r["before_ids"], r["after_ids"]
                    if b == a:
                        self.drops[name]["unchanged"] += 1
                        continue
                    if len(b) + len(a) > self.a.max_pair:
                        self.drops[name]["too_long"] += 1
                        continue
                    intent = r["intent"]
                    if name in REAL_DOC:
                        labels = set(intent.split("+"))
                        if intent != "none" and labels <= {"meaning-changed", "others"}:
                            self.drops[name]["content_only"] += 1
                            continue
                        cat, pool = "polish", "polish_para_iterater"
                    elif name.startswith("coedit"):
                        cat = COEDIT_CATEGORY.get(intent)
                        pool = {"grammar": "grammar_sent_coedit"}.get(cat, f"{cat}_sent_real")
                    else:
                        ic = (r.get("meta") or {}).get("intent_confidence", 1.0)
                        if intent in ("fluency", "clarity"):
                            cat = intent if ic >= conf_min else "polish"
                        elif intent == "coherence":
                            cat = "coherence"
                        elif intent == "style":
                            cat = "polish"
                        else:
                            cat = None
                        pool = f"{cat}_sent_real"
                    if cat is None:
                        self.drops[name]["no_category"] += 1
                        continue
                    gk = r["doc_id"] if name.startswith("iterater") else "coedit:" + norm(r["after_text"])
                    rows.append((pool, cat, r["uid"], gk, b, a, r["instruction_ids"], r["instruction"], intent,
                                 r["before_text"], r["after_text"]))
            ks = self.kept_many([(x[4], x[5]) for x in rows])
            for x, k in zip(rows, ks):
                pool, cat, uid, gk, b, a, iids, itext, intent, bt, at = x
                if self.add(name, pool, uid, gk, b, a, instr=tuple(iids), texts=(bt, at), band=k):
                    direct = (name.startswith("coedit") and COEDIT_CATEGORY.get(intent) == cat) or \
                             (name in REAL_SENT and not name.startswith("coedit") and intent == cat) or \
                             (intent == "style" and cat == "polish")
                    if direct:
                        self.cat_instr[cat][tuple(iids)] = itext
            print(f"[real] {name}: {dict(self.drops[name])}", flush=True)

    # ---------------------------------------------------------------- LLM drafts
    def llm(self):
        self.draft_sources = set()
        for cat in ("grammar", "coherence", "polish"):
            p = ROOT / "data" / "editdata" / "full_v1" / f"{cat}.jsonl"
            self.note_input(p)
            recs = [json.loads(l) for l in open(p, encoding="utf-8")]
            self.draft_sources.update(r["source_id"] for r in recs)
            bs = self.tok([r["draft"] for r in recs])
            as_ = self.tok([r["final"] for r in recs])
            ks = self.kept_many(list(zip(bs, as_)))
            src = f"full_v1/{cat}"
            for r, b, a, k in zip(recs, bs, as_, ks):
                doc = "/".join(r["source_id"].split("/")[:2])
                self.add(src, f"{cat}_para_llm", r["id"], doc, b, a, texts=(r["draft"], r["final"]), band=k)
            print(f"[llm] {cat}: {dict(self.drops[src])}", flush=True)

    # ---------------------------------------------------------------- C4_200M
    def c4200m(self):
        import pyarrow.parquet as pq
        p = ROOT / "data" / "c4_200m" / "train-0.parquet"
        self.note_input(p)
        t = pq.read_table(p, columns=["input", "output"])
        ins, outs = t.column("input").to_pylist(), t.column("output").to_pylist()
        if self.a.c4200m_limit:
            ins, outs = ins[:self.a.c4200m_limit], outs[:self.a.c4200m_limit]
        bs, as_ = self.tok(ins), self.tok(outs)
        sel = [i for i in range(len(bs)) if bs[i] != as_[i] and len(bs[i]) + len(as_[i]) <= self.a.max_pair]
        ks = dict(zip(sel, self.kept_many([(bs[i], as_[i]) for i in sel])))
        src = "c4_200m/0"
        for i in range(len(bs)):
            self.add(src, "grammar_sent_c4200m", f"c4_200m/0/{i}", "c4_200m:" + norm(outs[i]), bs[i], as_[i],
                     texts=(ins[i], outs[i]), band=ks.get(i))
        print(f"[c4_200m] {dict(self.drops[src])}", flush=True)

    # ---------------------------------------------------------------- C4 no change + restoration
    def c4(self):
        sdir = ROOT / "data" / "editdata" / "full_v1" / "screen"
        clean = {}
        odd = {}
        for p in sorted(sdir.glob("clean_*.jsonl")):
            self.note_input(p)
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                clean[r["id"]] = r["p_yes"]
        for p in sorted(sdir.glob("odd_*.jsonl")):
            self.note_input(p)
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                odd[r["id"]] = r["odd"]
        ok = {i for i, py in clean.items() if py >= self.a.clean_p and odd.get(i) is False}
        ok -= self.draft_sources
        self.c4_screen = dict(screened=len(clean), pass_both=sum(1 for i, py in clean.items() if py >= self.a.clean_p and odd.get(i) is False),
                              minus_draft_sources=len(ok))
        p = ROOT / "data" / "editdata" / "c4_paragraphs.jsonl"
        self.note_input(p)
        nc, rest = [], []
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["pool"] == "restore":
                rest.append(r)
            elif r["pool"] == "llm" and r["id"] in ok:
                nc.append(r)
        # restoration
        ids = self.tok([r["text"] for r in rest])
        for r, a in zip(rest, ids):
            doc = "/".join(r["id"].split("/")[:2])
            if len(a) > self.a.max_rest:
                self.drops["c4_restore"]["too_long"] += 1
                continue
            self.add("c4_restore", "restoration", r["id"], doc, [], a, texts=(r["text"],), kind="restoration")
        print(f"[restore] {dict(self.drops['c4_restore'])}", flush=True)
        # no change: half paragraphs, half sentences (by a hash of the paragraph id)
        para = [r for r in nc if int(hashlib.sha256(("ncsplit:" + r["id"]).encode()).hexdigest()[:8], 16) % 2 == 0]
        sent_src = [r for r in nc if int(hashlib.sha256(("ncsplit:" + r["id"]).encode()).hexdigest()[:8], 16) % 2 == 1]
        ids = self.tok([r["text"] for r in para])
        for r, a in zip(para, ids):
            doc = "/".join(r["id"].split("/")[:2])
            if 2 * len(a) > self.a.max_pair:
                self.drops["c4_nochange_para"]["too_long"] += 1
                continue
            self.add("c4_nochange_para", "nochange_para", r["id"], doc, [], a, texts=(r["text"],), kind="nochange")
        sents = [(r["id"], j, s) for r in sent_src for j, s in enumerate(sentences(r["text"]))]
        ids = self.tok([s for _, _, s in sents])
        for (pid, j, s), a in zip(sents, ids):
            if not (self.a.min_sent <= len(a) <= self.a.max_sent):
                self.drops["c4_nochange_sent"]["length"] += 1
                continue
            doc = "/".join(pid.split("/")[:2])
            self.add("c4_nochange_sent", "nochange_sent", f"{pid}/s{j}", doc, [], a, texts=(s,), kind="nochange")
        print(f"[nochange] {dict(self.drops['c4_nochange_para'])} {dict(self.drops['c4_nochange_sent'])}", flush=True)

    # ---------------------------------------------------------------- write pools
    def write(self, out: Path):
        pdir = out / "pools"
        pdir.mkdir(parents=True, exist_ok=True)
        meta = {}
        for (cat, level), lst in SLOT_POOLS.items():
            for _, pool in lst:
                kind = "restoration" if cat == "restoration" else ("nochange" if cat == "nochange" else "edit")
                meta[pool] = dict(kind=kind, cat=None if kind != "edit" else cat, level=level)
        for pool, recs in sorted(self.pools.items()):
            m = meta[pool]
            instr_vocab, instr_idx = [], {}
            m["splits"] = {}
            for split in ("train", "dev", "test"):
                rs = [r for r in recs if r["split"] == split]
                la = np.array([len(r["a"]) for r in rs], dtype=np.int64)
                cat_ = lambda xs: np.concatenate(xs).astype(np.uint16) if xs else np.zeros(0, dtype=np.uint16)
                arrays = dict(a_flat=cat_([r["a"] for r in rs]),
                              a_off=np.concatenate([[0], np.cumsum(la)]).astype(np.int64),
                              uids=np.frombuffer("\n".join(r["uid"] for r in rs).encode("utf-8"), dtype=np.uint8))
                lb = np.zeros(len(rs), dtype=np.int64)
                if m["kind"] == "edit":
                    lb = np.array([len(r["b"]) for r in rs], dtype=np.int64)
                    arrays.update(b_flat=cat_([r["b"] for r in rs]),
                                  b_off=np.concatenate([[0], np.cumsum(lb)]).astype(np.int64))
                    if any(r["instr"] is not None for r in rs) or any(r["instr"] is not None for r in recs):
                        ii = []
                        for r in rs:
                            if r["instr"] is None:
                                ii.append(-1)
                                continue
                            if r["instr"] not in instr_idx:
                                instr_idx[r["instr"]] = len(instr_vocab)
                                instr_vocab.append(list(r["instr"]))
                            ii.append(instr_idx[r["instr"]])
                        arrays["instr"] = np.array(ii, dtype=np.int32)
                f = pdir / f"{pool}_{split}.npz"
                np.savez(f, **arrays)
                plan = int(la.sum() + lb.sum()) if m["kind"] == "edit" else int(la.sum()) * (2 if m["kind"] == "nochange" else 1)
                m["splits"][split] = dict(file=str(f.relative_to(out)).replace(os.sep, "/"), sha256=sha256_file(f),
                                          records=len(rs), plan_tokens=plan, text_tokens=int(la.sum() + lb.sum()),
                                          mean_plan_tokens=plan / max(1, len(rs)))
            m["instr_vocab"] = instr_vocab
            print(f"[write] {pool}: " + " ".join(f"{s}={v['records']}/{v['plan_tokens'] / 1e6:.2f}M" for s, v in m["splits"].items()), flush=True)
        self.pool_meta = {k: v for k, v in meta.items() if "splits" in v}
        return self.pool_meta


def allocate(pool_meta, total: float, max_views: float):
    """Main mix: target plan tokens per pool."""
    uniq = {p: m["splits"]["train"]["plan_tokens"] for p, m in pool_meta.items()}
    target, notes = {}, []
    shortfall = 0.0
    for (cat, level), lst in SLOT_POOLS.items():
        share = SHARES[cat] * (1.0 if level is None or cat in SENT_ONLY_CATS else 0.5)
        if cat in SENT_ONLY_CATS and level == "para":
            continue
        need = share * total
        for g in sorted({g for g, _ in lst}):
            ps = [p for gg, p in lst if gg == g and uniq.get(p, 0) > 0]
            if not ps:
                continue
            u = sum(uniq[p] for p in ps)
            take = min(need, max_views * u)
            for p in ps:
                target[p] = target.get(p, 0.0) + take * uniq[p] / u
            need -= take
            if need <= 1e-6:
                break
        if need > 1e-6:
            shortfall += need
            notes.append(f"{cat}/{level}: {need / 1e6:.2f}M short at {max_views} views -> {SPILL_POOL}")
    target[SPILL_POOL] = target.get(SPILL_POOL, 0.0) + shortfall
    if target[SPILL_POOL] > max_views * uniq[SPILL_POOL]:
        notes.append(f"WARNING {SPILL_POOL} at {target[SPILL_POOL] / uniq[SPILL_POOL]:.2f} views")
    return target, notes


def variant(pool_meta, main_target, drop):
    """no_rest: the dropped pools removed, every other pool scaled up by the same factor (same total)."""
    keep = {p: t for p, t in main_target.items() if p not in drop}
    s = sum(main_target.values()) / sum(keep.values())
    return {p: t * s for p, t in keep.items()}


def sent_only_variant(pool_meta, main_target, max_views: float):
    """sent_only: each category keeps its main-mix tokens (restoration and no change unchanged); its paragraph tokens move
    to its own sentence pools up to ``max_views``; what they cannot take goes to C4_200M grammar sentences."""
    uniq = {p: m["splits"]["train"]["plan_tokens"] for p, m in pool_meta.items()}
    out, notes, spill = {}, [], 0.0
    for (cat, level), lst in SLOT_POOLS.items():
        if level == "para":
            continue
        sent = [p for _, p in lst if p in main_target]
        para = [p for _, p in SLOT_POOLS.get((cat, "para"), []) if p in main_target]
        extra = sum(main_target[p] for p in para)
        for p in sent:
            out[p] = main_target[p]
        for g in sorted({g for g, _ in lst}):  # real pools first, as in the main allocation
            ps = [p for gg, p in lst if gg == g and p in sent]
            room = sum(max(0.0, max_views * uniq[p] - out[p]) for p in ps)
            take = min(extra, room)
            for p in ps:
                r = max(0.0, max_views * uniq[p] - out[p])
                out[p] += take * r / room if room else 0.0
            extra -= take
        if extra > 1e-6:
            spill += extra
            notes.append(f"sent_only {cat}: {extra / 1e6:.2f}M beyond {max_views} sentence views -> {SPILL_POOL}")
    out[SPILL_POOL] += spill
    return out, notes


def describe_mix(pool_meta, target, batch: int):
    rows, eps = [], 0.0
    for p, t in sorted(target.items()):
        tr = pool_meta[p]["splits"]["train"]
        e = t / tr["mean_plan_tokens"]
        eps += e
        rows.append(dict(pool=p, target_plan_tokens=t, views=t / tr["plan_tokens"], episodes=e))
    for r in rows:
        r["weight"] = r["episodes"] / eps
    steps = math.ceil(eps / batch)
    by_cat, by_level = Counter(), Counter()
    tot = sum(target.values())
    for p, t in target.items():
        m = pool_meta[p]
        by_cat[m["cat"] or m["kind"]] += t / tot
        by_level[m["level"] or "none"] += t / tot
    return dict(pools=rows, total_plan_tokens=tot, episodes=eps, batch_size=batch, steps=steps,
                schedule_episodes=steps * batch, share_by_category=dict(by_cat), share_by_level=dict(by_level))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/editdata/mix_v2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--total", type=float, default=300e6)
    ap.add_argument("--max-views", type=float, default=4.0)
    ap.add_argument("--max-pair", type=int, default=480)
    ap.add_argument("--max-rest", type=int, default=250)
    ap.add_argument("--min-sent", type=int, default=5)
    ap.add_argument("--max-sent", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--clean-p", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sent-only-max-views", type=float, default=6.0)
    ap.add_argument("--mixes-only", action="store_true", help="recompute mixes.json from an existing manifest")
    ap.add_argument("--c4200m-limit", type=int, default=0, help="first N C4_200M rows only (smoke tests)")
    ap.add_argument("--only", nargs="*", default=None, help="smoke: subset of real, llm, c4200m, c4")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.mixes_only:
        write_mixes(out, json.load(open(out / "manifest.json", encoding="utf-8")), a)
        return
    b = Builder(a)
    print(f"[eval keys] {len(b.ek.whole)} texts, {len(b.ek.sent)} sentences from {len(b.ek.files)} files", flush=True)
    steps = a.only or ["real", "llm", "c4200m", "c4"]
    b.draft_sources = set()
    for s in ["real", "llm", "c4200m", "c4"]:
        if s in steps:
            getattr(b, s)()
    pm = b.write(out)
    cat_instr = {c: [list(k) for k in sorted(b.cat_instr.get(c, {}))] for c in TASKS}
    cat_instr_text = {c: [b.cat_instr[c][k] for k in sorted(b.cat_instr.get(c, {}))] for c in TASKS}
    man = dict(format="reviser-editmix-v2", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               builder=dict(script="scripts/editdata/build_mix_v2.py", sha256=sha256_file(__file__),
                            tasktokens_sha256=sha256_file(ROOT / "src" / "reviser_v2" / "tasktokens.py"), args=vars(a)),
               inputs=b.inputs, eval_files=b.ek.files, eval_keys=dict(texts=len(b.ek.whole), sentences=len(b.ek.sent)),
               drops={k: dict(v) for k, v in b.drops.items()}, c4_screen=getattr(b, "c4_screen", None),
               tasks=list(TASKS), cat_instr=cat_instr, cat_instr_text=cat_instr_text, pools=pm,
               seconds=round(time.time() - t0, 1))
    body = json.dumps(man, sort_keys=True)
    man["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(man, open(out / "manifest.json", "w", encoding="utf-8"), indent=1)
    if set(steps) == {"real", "llm", "c4200m", "c4"} or a.only is None:
        write_mixes(out, man, a)
    print(f"[done] {time.time() - t0:.0f}s manifest sha256 {man['sha256']}", flush=True)


def write_mixes(out: Path, man, a):
    pm = man["pools"]
    target, notes = allocate(pm, a.total, a.max_views)
    so, so_notes = sent_only_variant(pm, target, a.sent_only_max_views)
    mixes = dict(manifest_sha256=man["sha256"], builder_sha256=sha256_file(__file__),
                 created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 max_views=a.max_views, sent_only_max_views=a.sent_only_max_views,
                 shares=SHARES, notes=notes + so_notes,
                 variants={"main": describe_mix(pm, target, a.batch),
                           "sent_only": describe_mix(pm, so, a.batch),
                           "no_rest": describe_mix(pm, variant(pm, target, {"restoration"}), a.batch)})
    if True:
        json.dump(mixes, open(out / "mixes.json", "w", encoding="utf-8"), indent=1)
        for k, v in mixes["variants"].items():
            print(f"[mix] {k}: {v['total_plan_tokens'] / 1e6:.1f}M plan tokens, {v['episodes']:.0f} episodes, {v['steps']} steps; "
                  f"cat {json.dumps({c: round(s, 3) for c, s in v['share_by_category'].items()})} level "
                  f"{json.dumps({c: round(s, 3) for c, s in v['share_by_level'].items()})}", flush=True)
            for r in v["pools"]:
                print(f"    {r['pool']:<24} {r['target_plan_tokens'] / 1e6:7.2f}M views {r['views']:.2f} w {r['weight']:.4f}")
        for n in mixes["notes"]:
            print("[note]", n)


if __name__ == "__main__":
    main()
