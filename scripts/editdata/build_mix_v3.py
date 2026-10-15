#!/usr/bin/env python
"""Build the Reviser-Edit v3 training mix (~1B plan tokens over 3 passes; user decisions 2026-10-08). Output, read by
src/reviser_v2/tasks/editmix.py: <out>/{pools/<pool>_<split>.npz, manifest.json, mixes.json, snapshot/}.

Run in WSL on CPU (fork multiprocessing):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src /opt/venv/bin/python scripts/editdata/build_mix_v3.py --out data/editdata/mix_v3 --workers 14
  ... build_mix_v3.py --report        # seconds: available unique plan tokens per slot / pool vs the per-pass need

Reuses scripts/editdata/build_mix_v2.py (EvalKeys, Tok, split_of, kept, sentences, and its real / llm / c4 stages
unchanged). Same filters as v2 for every source: unchanged pairs, kept band [0.8, 1.0] (GPT-2 tokens, difflib
autojunk=False, recomputed here), len(before) + len(after) <= 480, exact-duplicate removal across all sources (first
source wins, order below), eval-overlap removal (normalised text or any >= 5-word sentence equal to one in
data/real/*_{dev,test,test4k,validation}.jsonl), 98/1/1 train/dev/test by sha256 of a group key (the source document:
C4 'c4en0/80950/1' -> 'c4en0/80950', DCLM 'dclm30b/2027/14556/2' -> 'dclm30b/2027/14556'; CoEdIT / C4_200M: the
normalised target text; IteraTeR: doc_id).

Sources (stage order = duplicate precedence):
  real          v2 real(): IteraTeR / CoEdIT sentence pairs -> grammar_sent_coedit, <cat>_sent_real; IteraTeR docs ->
                polish_para_iterater
  full_v1       v2 llm(): LLM paragraph drafts -> {grammar,coherence,polish}_para_llm
  night_v2      helper C's LLM drafts data/editdata/night_v2/<cat>_<par|sent>.jsonl -> <cat>_<para|sent>_llm. Read from
                a snapshot copy (<out>/snapshot/, sha256 checked against C's night_v2/manifest.json). Only rows whose
                generator is a Qwen2.5-32B model, plus with --allow-haiku rows drafted by Claude Haiku (user decision
                2026-10-09; C's files hold only rows its collect kept after the Qwen review); rows of chunks starting
                '0t72_' dropped. Every pool record keeps its generator: npz array 'gen' (int16 code into the pool's
                manifest 'gen_vocab', -1 = untagged, i.e. not a night_v2 row) and per-split 'generators' counts in the
                manifest, so a run or a release can leave a generator out later.
                The Haiku rows are C's separate files night_v2/haiku_edit/haiku_<cat>_<par|sent>.jsonl (own collect
                manifest haiku_edit/manifest.json, same sha256-checked snapshot; source 'night_v2_haiku/<cat>_<lv>'),
                read only with --allow-haiku, into the same <cat>_<para|sent>_llm pools. Their source paragraphs join
                the handed-out set below (C says they have no kept Qwen record; the C4 / DCLM no-change pools must
                not hold them either).
  c4            v2 c4(): C4 restoration (c4_paragraphs restore pool -> restoration) and clean-screened C4 no-change
                (-> nochange_para / nochange_sent). Excluded as no-change text: sources of full_v1 kept drafts (v2)
                AND every paragraph handed to night_v2 drafting (night_v2/chunks/*.jsonl, incl. withdrawn chunks;
                with --allow-haiku also every source paragraph of the Haiku files).
  synth         unique clean `after` texts of the real SENTENCE pairs kept above (all splits keep their record's split),
                len(before) + len(after) <= 256 (as the September synth_repair pool). Kind 'synth' (a only); the
                training code injects the errors. plan tokens = 2 x len.
  dclm screened night_v2/screen/ok_*.jsonl (7B clean P(yes) >= 0.9, not odd) minus chunk sources -> half paragraphs
                (nochange_para_dclm), half cut into sentences (nochange_sent_dclm), split by hash as v2.
  dclm restore  night_v2/pools/restore_dclm.jsonl (snapshot) -> restoration_dclm (<= 250 tokens as v2).
  dclm raw nc   night_v2/pools/nochange_dclm.jsonl (snapshot; NOT 7B-clean-checked) -> nochange_{para,sent}_dclmraw;
                only processed when the allocation needs them (priority 2).
  C4_200M       data/editdata/c4_200m/grammar_sent_v1.shard<k>.jsonl (helper C's filter) -> grammar_sent_c4200m. Only a
                hash-chosen subset of the first shards, sized to the need, is read, tokenized and filtered.

Allocation per pass (T = total / passes plan tokens; plan tokens: edit = before + after, nochange / synth = 2 x text,
restoration = text): need per slot = share x T (edit categories and no-change 50/50 sent / para). Priority groups fill
in order; inside a group pools first fill to 1 view (small pools whole, then big pools subset), then all go up together
to the per-pass cap max_views_total / passes; then the next group. A slot's shortfall goes to the same category's
other level (up to the cap, in its priority order), then to grammar_sent_c4200m. --cat-max-views-total (e.g.
clarity=6 coherence=6; user decision 2026-10-09: more repeats of the scarce clarity / coherence data rather than
grammar filling the gap) only acts on what is still short after that: the category's pools are raised from the
global cap towards its own cap (own level first, then the other level, priority order) before the rest spills. A pool whose target is below its unique
tokens is SUBSET (hash-chosen records, views 1.0) and only that subset is written as its train split, so the train pool
files are the corpus. Episodes per pass = round(views x records); the pass is padded to a multiple of the batch by
adding episodes to the largest pool.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import build_mix_v2 as v2  # noqa: E402
from build_mix_v2 import norm, sentences, sha256_file, split_of  # noqa: E402

NIGHT = ROOT / "data" / "editdata" / "night_v2"
HAIKU = NIGHT / "haiku_edit"  # C's Qwen-reviewed Haiku drafts (haiku_<cat>_<par|sent>.jsonl + manifest.json)
C4200M_DIR = ROOT / "data" / "editdata" / "c4_200m"
MIX_V2 = ROOT / "data" / "editdata" / "mix_v2"
CATS = ("grammar", "fluency", "clarity", "coherence", "polish")
PLAN_RATIO = {"grammar": 18, "fluency": 15, "clarity": 15, "coherence": 12, "polish": 15}  # v2 plan proportions


def make_shares(synth=0.08, nochange=0.15, restoration=0.15):
    rest = 1.0 - synth - nochange - restoration
    s = {"synth": synth, "nochange": nochange, "restoration": restoration}
    tot = sum(PLAN_RATIO.values())
    s.update({c: rest * PLAN_RATIO[c] / tot for c in CATS})
    return s


SHARES = make_shares()
LEVELED = set(CATS) | {"nochange"}  # 50/50 sent / para
# slot -> [(priority group, pool)]; lower group fills first
SLOT_POOLS = {
    ("restoration", None): [(0, "restoration"), (0, "restoration_dclm")],
    ("synth", None): [(0, "synth")],
    ("nochange", "sent"): [(0, "nochange_sent"), (1, "nochange_sent_dclm"), (2, "nochange_sent_dclmraw")],
    ("nochange", "para"): [(0, "nochange_para"), (1, "nochange_para_dclm"), (2, "nochange_para_dclmraw")],
    ("grammar", "sent"): [(0, "grammar_sent_coedit"), (1, "grammar_sent_c4200m")],
    ("grammar", "para"): [(0, "grammar_para_llm")],
    ("fluency", "sent"): [(0, "fluency_sent_real"), (1, "fluency_sent_llm")],
    ("fluency", "para"): [(0, "fluency_para_llm")],
    ("clarity", "sent"): [(0, "clarity_sent_real"), (1, "clarity_sent_llm")],
    ("clarity", "para"): [(0, "clarity_para_llm")],
    ("coherence", "sent"): [(0, "coherence_sent_real"), (1, "coherence_sent_llm")],
    ("coherence", "para"): [(0, "coherence_para_llm")],
    ("polish", "sent"): [(0, "polish_sent_real"), (1, "polish_sent_llm")],
    ("polish", "para"): [(0, "polish_para_iterater"), (0, "polish_para_llm")],
}
BIG = {"grammar_sent_c4200m", "restoration_dclm", "nochange_para_dclm", "nochange_sent_dclm", "nochange_para_dclmraw",
       "nochange_sent_dclmraw"}
SPILL_POOL = "grammar_sent_c4200m"
ALL_POOLS = {p for lst in SLOT_POOLS.values() for _, p in lst}
REAL_SENT_POOLS = ("grammar_sent_coedit", "fluency_sent_real", "clarity_sent_real", "coherence_sent_real",
                   "polish_sent_real")


def pool_meta(pool):
    for (cat, level), lst in SLOT_POOLS.items():
        if any(p == pool for _, p in lst):
            if cat in ("restoration", "nochange"):
                return dict(kind=cat, cat=None, level=level)
            if cat == "synth":
                return dict(kind="synth", cat="grammar", level=None)
            return dict(kind="edit", cat=cat, level=level)
    raise KeyError(pool)


def hkey(*parts) -> int:
    return int(hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def doc_key(source_id: str) -> str:
    p = source_id.split("/")
    return "/".join(p[:3]) if p[0].startswith("dclm") else "/".join(p[:2])


def ok_generator(g, allow_haiku=False) -> bool:
    g = str(g or "").lower()
    if "replay" in g:
        return False
    return ("qwen2.5-32b" in g and "haiku" not in g) or (allow_haiku and "haiku" in g)


def gen_name(g) -> str:
    """Generator tag as stored in pool records: the model name without its local path."""
    return str(g or "").replace("\\", "/").rstrip("/").split("/")[-1] or "unknown"


# ====================================================================== allocation (pure; shared by --report)
def allocate(U, T, cap, shares=SHARES, slot_pools=SLOT_POOLS, big=BIG, spill_pool=SPILL_POOL, cat_cap=None):
    """U: pool -> available unique train plan tokens. cat_cap: category -> per-pass views cap used only for that
    category's shortfall left after the normal allocation (``cap``, other level); without it the result is the original
    allocation. Returns (target plan tokens per pass per pool, notes, slot rows)."""
    target = {p: 0.0 for lst in slot_pools.values() for _, p in lst}
    cat_cap = dict(cat_cap or {})
    capc = lambda c: max(cap, cat_cap.get(c, cap))  # noqa: E731
    pool_cat = {p: c for (c, _lvl), lst in slot_pools.items() for _, p in lst}
    notes, rows, short = [], [], {}

    def fill(lst, R, capv):
        """Add up to R plan tokens to lst's pools, priority group by group, each pool up to capv x U (proportional to
        its room inside a group). Returns the amount added."""
        added = 0.0
        for g in sorted({g for g, _ in lst}):
            ps = [p for gg, p in lst if gg == g and U.get(p, 0) > 0]
            room = {p: max(0.0, capv * U[p] - target[p]) for p in ps}
            rt = sum(room.values())
            if rt <= 0 or R - added <= 1e-6:
                continue
            take = min(R - added, rt)
            for p in ps:
                target[p] += take * room[p] / rt
            added += take
        return added

    for (cat, level), lst in slot_pools.items():
        need = shares[cat] * T * (0.5 if cat in LEVELED else 1.0)
        R = need
        for g in sorted({g for g, _ in lst}):
            ps = [p for gg, p in lst if gg == g and U.get(p, 0) > 0]
            if not ps or R <= 1e-6:
                continue
            small = [p for p in ps if p not in big]
            bigs = [p for p in ps if p in big]
            us, ub = sum(U[p] for p in small), sum(U[p] for p in bigs)
            if R <= us:
                for p in small:
                    target[p] += R * U[p] / us
                R = 0.0
            elif R <= us + ub:
                for p in small:
                    target[p] += U[p]
                for p in bigs:
                    target[p] += (R - us) * U[p] / ub
                R = 0.0
            else:
                v = min(cap, R / (us + ub))
                for p in ps:
                    target[p] += v * U[p]
                R -= v * (us + ub)
        short[(cat, level)] = R
        rows.append(dict(slot=f"{cat}/{level or '-'}", need=need, short=R))
    # shortfall -> other level of the same category (up to the cap), then (categories in cat_cap) more repeats of
    # the category's own pools up to its cap, own level first, then the other level; the rest -> the spill pool
    other_of = lambda cat, level: (cat, {"sent": "para", "para": "sent"}.get(level))  # noqa: E731
    for (cat, level), R in list(short.items()):
        other = other_of(cat, level)
        if R > 1e-6 and cat in LEVELED and other in slot_pools:
            moved = fill(slot_pools[other], R, cap)
            if moved > 1e-6:
                notes.append(f"{cat}/{level}: {moved / 1e6:.2f}M short -> {cat}/{other[1]} (other level)")
            short[(cat, level)] = R - moved
    for phase in ("own", "other"):
        for (cat, level), R in list(short.items()):
            key = (cat, level) if phase == "own" else other_of(cat, level)
            if R <= 1e-6 or cat not in cat_cap or key not in slot_pools or (phase == "other" and cat not in LEVELED):
                continue
            moved = fill(slot_pools[key], R, capc(cat))
            if moved > 1e-6:
                notes.append(f"{cat}/{level}: {moved / 1e6:.2f}M short -> {cat}/{key[1]} up to {capc(cat):.3f} views "
                             f"per pass (--cat-max-views-total; user decision 2026-10-09)")
            short[(cat, level)] = R - moved
    spill = 0.0
    for (cat, level), R in short.items():
        if R > 1e-6:
            spill += R
            notes.append(f"{cat}/{level or '-'}: {R / 1e6:.2f}M short -> {spill_pool}")
    if spill > 0:
        target[spill_pool] += spill
        if U.get(spill_pool, 0) and target[spill_pool] > capc(pool_cat.get(spill_pool)) * U[spill_pool]:
            notes.append(f"WARNING {spill_pool} at {target[spill_pool] / U[spill_pool]:.2f} views per pass")
    for p in list(target):
        if U.get(p, 0) and target[p] > capc(pool_cat.get(p)) * U[p] * (1 + 1e-9):
            notes.append(f"WARNING {p} over the cap: {target[p] / U[p]:.3f} views per pass")
    return {p: t for p, t in target.items() if t > 0}, notes, rows


def episodes_per_pass(corpus, target, batch):
    """corpus: pool -> (records, unique plan tokens, subset flag). Returns per-pool rows and the pass episode total
    (a multiple of ``batch``; padding goes to the pool with most episodes)."""
    rows = {}
    for p, t in target.items():
        n, u, sub = corpus[p]
        views = 1.0 if sub else t / u
        rows[p] = dict(views=views, episodes=int(round(views * n)), records=n, unique=u)
    tot = sum(r["episodes"] for r in rows.values())
    pad = (-tot) % batch
    if pad:
        big = max(rows, key=lambda p: rows[p]["episodes"])
        rows[big]["episodes"] += pad
        tot += pad
    for r in rows.values():
        r["views"] = r["episodes"] / r["records"] if r["records"] else 0.0
        r["plan_tokens"] = r["views"] * r["unique"]
    return rows, tot


# ====================================================================== worker functions (fork; module globals)
_EK = None
_TOKW = None
_ARGS = {}


def _enc(texts):
    global _TOKW
    if _TOKW is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import GPT2TokenizerFast
        tdir = ROOT / "data" / "tokenizers" / "gpt2"
        _TOKW = GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")
    if not texts:
        return []
    return _TOKW(list(texts), add_special_tokens=False, split_special_tokens=True)["input_ids"]


def _u16(x):
    return np.asarray(x, dtype=np.uint16)


def _prep_pair_rows(rows):
    """rows: [(meta, before_text, after_text)] -> [(meta, b, a, kept or None, eval_hit)]"""
    max_pair = _ARGS["max_pair"]
    bs, as_ = _enc([r[1] for r in rows]), _enc([r[2] for r in rows])
    out = []
    for (meta, bt, at), b, a in zip(rows, bs, as_):
        k = v2.kept((b, a)) if (b != a and len(b) + len(a) <= max_pair) else None
        out.append((meta, _u16(b), _u16(a), k, _EK.hit(bt, at)))
    return out


def _prep_night(lines):
    """Raw night_v2 jsonl lines -> [(status, id, source_id, b, a, kept, hit, generator)]"""
    rows, out = [], []
    for ln in lines:
        try:
            r = json.loads(ln)
        except Exception:
            out.append(("bad_json", None))
            continue
        if not ok_generator(r.get("generator"), _ARGS.get("allow_haiku", False)):
            out.append(("excluded_generator", None))
            continue
        if str(r.get("chunk", "")).startswith("0t72_"):
            out.append(("excluded_chunk_0t72", None))
            continue
        out.append(("ok", len(rows)))
        rows.append(((r["id"], r["source_id"], gen_name(r.get("generator"))), r["draft"], r["final"]))
    prepped = _prep_pair_rows(rows)
    res = []
    for st, j in out:
        if st != "ok":
            res.append((st, None, None, None, None, None, None, None))
        else:
            (uid, sid, gen), b, a, k, hit = prepped[j]
            res.append(("ok", uid, sid, b, a, k, hit, gen))
    return res


def _prep_texts(items):
    """items: [(uid, group, text, mode)] mode 'para' / 'rest' -> [(uid, group, a, hit)];
    mode 'sent' -> one row per sentence (uid/s<j>)."""
    flat = []
    for uid, grp, text, mode in items:
        if mode == "sent":
            flat += [(f"{uid}/s{j}", grp, s) for j, s in enumerate(sentences(text))]
        else:
            flat.append((uid, grp, text))
    ids = _enc([t for _, _, t in flat])
    return [(u, g, _u16(a), _EK.hit(t)) for (u, g, t), a in zip(flat, ids)]


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


# ====================================================================== snapshot of helper C's files
def snapshot_copy(src: Path, dst: Path, expected, tries=6):
    """Copy ``src`` to ``dst`` (one file, short open) and check its sha256 against ``expected()`` (a callable re-reading
    the owner's manifest: the owner may rewrite the file mid-copy). Returns (sha256, attempts)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        exp1 = expected()
        shutil.copyfile(src, dst)
        h = sha256_file(dst)
        if h == exp1 or h == expected():
            return h, i + 1
        print(f"[snapshot] {src.name}: sha256 mismatch (attempt {i + 1}); retrying", flush=True)
        time.sleep(10 + 20 * i)
    raise RuntimeError(f"snapshot of {src} never matched the owner's manifest")


def _load_json_retry(p: Path, tries=10):
    for i in range(tries):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(2 + i)
    raise RuntimeError(f"cannot read {p}")


def read_lines_hashed(p: Path):
    with open(p, "rb") as f:
        data = f.read()
    return data.decode("utf-8").splitlines(), hashlib.sha256(data).hexdigest(), len(data)


def chunk_source_ids():
    """Every paragraph handed to night_v2 drafting (chunks/*.jsonl, withdrawn/*.jsonl). Files are written once."""
    ids, files = set(), {}
    paths = sorted(glob.glob(str(NIGHT / "chunks" / "*.jsonl"))) + sorted(glob.glob(str(NIGHT / "withdrawn" / "**" / "*.jsonl"), recursive=True))
    for p in paths:
        lines, h, nb = read_lines_hashed(Path(p))
        for ln in lines:
            if ln.startswith('{"id": "'):
                ids.add(ln[8:ln.index('"', 8)])
            elif ln.strip():
                ids.add(json.loads(ln)["id"])
        files[str(Path(p).relative_to(ROOT)).replace(os.sep, "/")] = dict(bytes=nb, sha256=h, rows=len(lines))
    return ids, files


# ====================================================================== builder
class BuilderV3(v2.Builder):
    def __init__(self, a):
        global _EK, _ARGS
        self.a = a
        self.pools = defaultdict(list)
        self.drops = defaultdict(Counter)
        self.seen = set()
        self.cat_instr = defaultdict(dict)
        self.inputs = {}
        self.snap = {}
        self.timing = {}
        self.extra = {}
        self.ek = v2.EvalKeys()
        self.tok = v2.Tok()
        self.pool_meta = {}
        self.draft_sources = set()
        _EK, _ARGS = self.ek, dict(max_pair=a.max_pair, allow_haiku=getattr(a, "allow_haiku", False))
        self.mp = mp.get_context("fork").Pool(a.workers)

    # v2 add with an optional precomputed eval hit and an optional fixed split
    def add(self, src, pool, uid, group_key, b, a, instr=None, texts=(), kind="edit", band=None, hit=None, split=None,
            gen=None):
        d = self.drops[src]
        if kind == "edit":
            if len(b) == len(a) and np.array_equal(np.asarray(b), np.asarray(a)):
                d["unchanged"] += 1
                return False
            if len(b) + len(a) > self.a.max_pair:
                d["too_long"] += 1
                return False
            if band is not None and not v2.in_band(band):
                d["out_of_band"] += 1
                return False
        b, a = np.asarray(b, dtype=np.uint16), np.asarray(a, dtype=np.uint16)
        key = hashlib.sha1(b.tobytes() + b"|" + a.tobytes()).digest() + kind.encode()
        if key in self.seen:
            d["duplicate"] += 1
            return False
        if (hit if hit is not None else self.ek.hit(*texts)):
            d["eval_overlap"] += 1
            return False
        self.seen.add(key)
        self.pools[pool].append(dict(uid=uid, split=split or split_of(group_key), b=b, a=a, instr=instr, gen=gen))
        d["kept"] += 1
        return True

    def stage(self, name, fn, *args):
        t = time.time()
        r = fn(*args)
        self.timing[name] = round(time.time() - t, 1)
        print(f"[time] {name}: {self.timing[name]:.1f}s", flush=True)
        return r

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, out: Path):
        sdir = out / "snapshot"
        man_p = NIGHT / "manifest.json"
        man = _load_json_retry(man_p)
        sdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(man_p, sdir / "night_v2_manifest.json")
        self.snap["night_manifest"] = dict(created_utc=man.get("created_utc"), chunks=man.get("chunks"),
                                           generators=man.get("generators"), drop_prefixes=man.get("drop_prefixes"),
                                           sha256=sha256_file(sdir / "night_v2_manifest.json"))
        self.night_files = {}
        for cat in CATS:
            for lv in ("par", "sent"):
                fn = f"{cat}_{lv}.jsonl"
                if fn not in man["files"] or not (NIGHT / fn).exists():
                    continue
                if f"{cat}_{'para' if lv == 'par' else 'sent'}_llm" not in ALL_POOLS:
                    self.extra.setdefault("night_v2_files_not_used", []).append(fn)
                    print(f"[snapshot] {fn}: no slot for it in this mix (not used)", flush=True)
                    continue
                h, n = snapshot_copy(NIGHT / fn, sdir / fn, lambda fn=fn: _load_json_retry(man_p)["files"][fn]["sha256"])
                meta = _load_json_retry(man_p)["files"][fn]
                self.night_files[(cat, lv, "qwen")] = sdir / fn
                self.snap[fn] = dict(sha256=h, attempts=n, bytes=(sdir / fn).stat().st_size,
                                     rows_manifest=meta.get("rows"), tokens_manifest=meta.get("tokens"))
                print(f"[snapshot] {fn}: {self.snap[fn]['bytes'] / 1e6:.0f} MB sha256 {h[:16]} ({n} attempt(s))", flush=True)
        if getattr(self.a, "allow_haiku", False):
            self.snapshot_haiku(sdir)
        pm_p = NIGHT / "pools" / "manifest.json"
        pm = _load_json_retry(pm_p)
        shutil.copyfile(pm_p, sdir / "night_v2_pools_manifest.json")
        self.dclm_files = {}
        for kind in ("restore", "nochange"):
            fn = Path(pm["pools"][kind]["file"]).name
            h, n = snapshot_copy(NIGHT / "pools" / fn, sdir / fn,
                                 lambda kind=kind: _load_json_retry(pm_p)["pools"][kind]["sha256"])
            self.dclm_files[kind] = sdir / fn
            self.snap[f"pools/{fn}"] = dict(sha256=h, attempts=n, rows_manifest=pm["pools"][kind]["rows"],
                                            tokens_manifest=pm["pools"][kind]["tokens"])
            print(f"[snapshot] pools/{fn}: sha256 {h[:16]} ({n} attempt(s))", flush=True)
        for k, v in self.snap.items():
            if k == "haiku_manifest":
                self.inputs["snapshot/haiku_edit_manifest.json (of data/editdata/night_v2/haiku_edit/manifest.json)"] = dict(sha256=v["sha256"])
            elif k != "night_manifest":
                self.inputs[f"snapshot/{k.split('/')[-1]} (of data/editdata/night_v2/{k})"] = dict(sha256=v["sha256"])

    def snapshot_haiku(self, sdir: Path):
        """--allow-haiku: C's haiku_edit/ files (rows kept by its collect after the Qwen review), sha256 vs its manifest."""
        man_p = HAIKU / "manifest.json"
        if not man_p.exists():
            raise RuntimeError(f"--allow-haiku but no {man_p}")
        man = _load_json_retry(man_p)
        shutil.copyfile(man_p, sdir / "haiku_edit_manifest.json")
        self.snap["haiku_manifest"] = dict(collected_utc=man.get("collected_utc"), created_utc=man.get("created_utc"),
                                           generators=man.get("generators"), chunks=man.get("chunks"),
                                           sha256=sha256_file(sdir / "haiku_edit_manifest.json"))
        for fn in sorted(man["files"]):
            stem = fn[len("haiku_"):-len(".jsonl")] if fn.startswith("haiku_") and fn.endswith(".jsonl") else ""
            cat, _, lv = stem.rpartition("_")
            if cat not in CATS or lv not in ("par", "sent") or f"{cat}_{'para' if lv == 'par' else 'sent'}_llm" not in ALL_POOLS:
                self.extra.setdefault("night_v2_files_not_used", []).append(f"haiku_edit/{fn}")
                print(f"[snapshot] haiku_edit/{fn}: no slot for it in this mix (not used)", flush=True)
                continue
            h, n = snapshot_copy(HAIKU / fn, sdir / fn, lambda fn=fn: _load_json_retry(man_p)["files"][fn]["sha256"])
            meta = _load_json_retry(man_p)["files"][fn]
            self.night_files[(cat, lv, "haiku")] = sdir / fn
            k = f"haiku_edit/{fn}"
            self.snap[k] = dict(sha256=h, attempts=n, bytes=(sdir / fn).stat().st_size,
                                rows_manifest=meta.get("rows"), tokens_manifest=meta.get("tokens"))
            print(f"[snapshot] {k}: {self.snap[k]['bytes'] / 1e6:.0f} MB sha256 {h[:16]} ({n} attempt(s))", flush=True)

    def handed_out(self):
        self.handed, files = chunk_source_ids()
        self.inputs.update(files)
        self.extra["night_chunk_sources"] = len(self.handed)
        print(f"[chunks] {len(files)} chunk files, {len(self.handed)} paragraphs handed to drafting", flush=True)
        hk = set()  # Haiku source paragraphs: excluded from no-change / C4 like the chunk paragraphs
        for (cat, lv, st), p in sorted(self.night_files.items()):
            if st != "haiku":
                continue
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        hk.add(json.loads(ln)["source_id"])
        if hk:
            new = hk - self.handed
            self.handed |= hk
            self.extra["haiku_sources"] = dict(paragraphs=len(hk), not_in_chunks=len(new))
            print(f"[chunks] Haiku files: {len(hk)} source paragraphs, {len(new)} not in the chunk files (added)", flush=True)

    # ---------------------------------------------------------------- night_v2 LLM drafts
    def night(self):
        self.night_kept_sources = set()
        for (cat, lv, st), p in sorted(self.night_files.items()):
            level = "para" if lv == "par" else "sent"
            pool, src = f"{cat}_{level}_llm", (f"night_v2/{cat}_{lv}" if st == "qwen" else f"night_v2_haiku/{cat}_{lv}")
            with open(p, encoding="utf-8") as f:
                lines = f.read().splitlines()
            for res in self.mp.imap(_prep_night, _chunks(lines, 2000), chunksize=1):
                for st, uid, sid, b, a, k, hit, gen in res:
                    if st != "ok":
                        self.drops[src][st] += 1
                        continue
                    if self.add(src, pool, uid, doc_key(sid), b, a, band=k, hit=hit, gen=gen):
                        self.night_kept_sources.add(sid)
                        self.drops[src][f"kept_generator:{gen}"] += 1
            print(f"[night] {src}: {dict(self.drops[src])}", flush=True)

    # ---------------------------------------------------------------- C4 (v2) with night_v2 hand-outs excluded
    def c4_v3(self):
        before = set(self.draft_sources)
        self.draft_sources |= self.handed
        self.c4()
        self.c4_screen.update(full_v1_kept_draft_sources=len(before), night_v2_handed_out=len(self.handed),
                              night_v2_kept_draft_sources=len(getattr(self, "night_kept_sources", ())))

    # ---------------------------------------------------------------- synthetic repair clean texts
    def synth(self):
        n = 0
        for pool in REAL_SENT_POOLS:
            for r in self.pools.get(pool, []):
                if len(r["b"]) + len(r["a"]) <= self.a.synth_max_pair:
                    n += 1
                    self.add("synth", "synth", r["uid"], None, [], r["a"], kind="synth", hit=False, split=r["split"])
        print(f"[synth] {n} candidate pairs <= {self.a.synth_max_pair}: {dict(self.drops['synth'])}", flush=True)

    # ---------------------------------------------------------------- text pools (no change / restoration)
    def text_pool(self, rows, src, pool_para, pool_sent=None, pool_rest=None):
        """rows: [(id, text)]. Restoration if pool_rest else half paragraphs / half sentences by v2's ncsplit hash."""
        items = []
        for pid, text in rows:
            grp = doc_key(pid)
            if pool_rest:
                items.append((pid, grp, text, "rest"))
            else:
                odd = int(hashlib.sha256(("ncsplit:" + pid).encode()).hexdigest()[:8], 16) % 2
                items.append((pid, grp, text, "sent" if odd else "para"))
        modes = {pid: m for pid, _, _, m in items}
        for res in self.mp.imap(_prep_texts, _chunks(items, 1000), chunksize=1):
            for uid, grp, a, hit in res:
                mode = modes.get(uid, "sent")
                if mode == "rest":
                    if len(a) > self.a.max_rest:
                        self.drops[src + "_restore"]["too_long"] += 1
                        continue
                    self.add(src + "_restore", pool_rest, uid, grp, [], a, kind="restoration", hit=hit)
                elif mode == "para":
                    if 2 * len(a) > self.a.max_pair:
                        self.drops[src + "_para"]["too_long"] += 1
                        continue
                    self.add(src + "_para", pool_para, uid, grp, [], a, kind="nochange", hit=hit)
                else:
                    if not (self.a.min_sent <= len(a) <= self.a.max_sent):
                        self.drops[src + "_sent"]["length"] += 1
                        continue
                    self.add(src + "_sent", pool_sent, uid, grp, [], a, kind="nochange", hit=hit)

    def dclm_screened(self):
        rows, nfiles, n_ok = [], 0, 0
        for p in sorted(glob.glob(str(NIGHT / "screen" / "ok_*.jsonl"))):
            lines, h, nb = read_lines_hashed(Path(p))
            self.inputs[str(Path(p).relative_to(ROOT)).replace(os.sep, "/")] = dict(bytes=nb, sha256=h)
            nfiles += 1
            for ln in lines:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                n_ok += 1
                if r["id"] not in self.handed:
                    rows.append((r["id"], r["text"]))
        self.extra["dclm_screened"] = dict(ok_files=nfiles, ok_rows=n_ok, never_handed_out=len(rows))
        print(f"[dclm screened] {nfiles} ok files, {n_ok} rows, {len(rows)} never handed out", flush=True)
        self.text_pool(rows, "dclm_screened", "nochange_para_dclm", "nochange_sent_dclm")
        print(f"[dclm screened] para {dict(self.drops['dclm_screened_para'])} sent {dict(self.drops['dclm_screened_sent'])}", flush=True)

    def _read_pool_jsonl(self, p):
        rows = []
        with open(p, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                rows.append((r["id"], r["text"]))
        return rows

    def dclm_restore(self):
        rows = self._read_pool_jsonl(self.dclm_files["restore"])
        self.text_pool(rows, "dclm", None, pool_rest="restoration_dclm")
        print(f"[dclm restore] {len(rows)} rows: {dict(self.drops['dclm_restore'])}", flush=True)

    def dclm_raw_nochange(self):
        rows = self._read_pool_jsonl(self.dclm_files["nochange"])
        self.text_pool(rows, "dclmraw", "nochange_para_dclmraw", "nochange_sent_dclmraw")
        print(f"[dclm raw nochange] {len(rows)} rows: para {dict(self.drops['dclmraw_para'])} sent {dict(self.drops['dclmraw_sent'])}", flush=True)

    # ---------------------------------------------------------------- C4_200M subset
    def c4200m_v3(self, need: float):
        """Fill grammar_sent_c4200m's train split to ``need`` plan tokens with a hash-ordered subset of C's filtered
        grammar_sent_v1 (first shards only, as many as needed with 25% margin)."""
        cm = json.load(open(C4200M_DIR / "manifest.json", encoding="utf-8"))
        per = cm["per_shard"]
        shards, tok = [], 0
        for k in sorted(per, key=int):
            if tok >= 1.25 * need and shards:
                break
            shards.append(int(k))
            tok += per[k]["gpt2_tokens_before_plus_after"]
        cands = []
        for k in shards:
            p = C4200M_DIR / f"grammar_sent_v1.shard{k}.jsonl"
            lines, h, nb = read_lines_hashed(p)
            self.inputs[str(p.relative_to(ROOT)).replace(os.sep, "/")] = dict(bytes=nb, sha256=h)
            for ln in lines:
                r = json.loads(ln)
                cands.append((hkey("v3sel", "grammar_sent_c4200m", r["id"]), r["id"], r["before"], r["after"],
                              r["n_tok_before"] + r["n_tok_after"], r["kept"]))
            del lines
        cands.sort()
        src, pool = "c4_200m_v1", "grammar_sent_c4200m"
        have, i, mism, nproc = 0, 0, 0, 0
        while have < need and i < len(cands):
            est, j = 0, i
            while j < len(cands) and est < 1.06 * (need - have) + 2000:
                est += cands[j][4]
                j += 1
            jobs = [((c[1], c[5], c[3]), c[2], c[3]) for c in cands[i:j]]
            i = j
            for res in self.mp.imap(_prep_pair_rows, _chunks(jobs, 4000), chunksize=1):
                for (uid, ck, at), b, a, k, hit in res:
                    nproc += 1
                    if k is not None and abs(k - ck) > 1e-3:
                        mism += 1
                    if self.add(src, pool, uid, "c4_200m:" + norm(at), b, a, band=k, hit=hit) and                             self.pools[pool][-1]["split"] == "train":
                        have += len(b) + len(a)
        self.extra["c4200m"] = dict(shards=shards, candidates=len(cands), processed=nproc, train_plan_tokens=have,
                                    need=need, kept_share_mismatch_vs_helper_c=mism)
        print(f"[c4_200m] shards {shards}: {len(cands)} candidates, processed {nproc}, train {have / 1e6:.2f}M for need "
              f"{need / 1e6:.2f}M; kept-share mismatches vs C's field: {mism}; {dict(self.drops[src])}", flush=True)
        return dict(records=int(cm["rows_kept"]), plan_tokens=int(cm["gpt2_tokens_before_plus_after"]),
                    note="helper C's filtered grammar_sent_v1, all 10 shards and splits, before the v2 filters (manifest)")

    # ---------------------------------------------------------------- write pools
    def write_v3(self, out: Path, target: dict, available: dict):
        pdir = out / "pools"
        pdir.mkdir(parents=True, exist_ok=True)
        meta = {}
        for pool in sorted(target):
            recs = self.pools.get(pool, [])
            m = pool_meta(pool)
            ptok = (lambda r: len(r["b"]) + len(r["a"])) if m["kind"] == "edit" else \
                ((lambda r: 2 * len(r["a"])) if m["kind"] in ("nochange", "synth") else (lambda r: len(r["a"])))
            by = {s: [r for r in recs if r["split"] == s] for s in ("train", "dev", "test")}
            u = sum(ptok(r) for r in by["train"])
            t = target[pool]
            subset = t < u * (1 - 1e-9)
            if subset:
                order = sorted(by["train"], key=lambda r: hkey("v3sel", pool, r["uid"]))
                sel, cum = [], 0
                for r in order:
                    if cum >= t:
                        break
                    sel.append(r)
                    cum += ptok(r)
                frac = len(sel) / max(1, len(by["train"]))
                by["train"] = sel
                for s in ("dev", "test"):
                    o = sorted(by[s], key=lambda r: hkey("v3sel", pool, r["uid"]))
                    by[s] = o[:min(len(o), max(1000, int(math.ceil(frac * len(o)))))]
            m["available"] = available.get(pool) or dict(records=sum(1 for r in recs if r["split"] == "train"), plan_tokens=u)
            m["subset"] = subset
            instr_vocab, instr_idx = [], {}
            has_gen = any(r.get("gen") is not None for r in recs)
            gen_vocab, gen_idx = [], {}
            m["splits"] = {}
            for split in ("train", "dev", "test"):
                rs = by[split]
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
                    if any(r["instr"] is not None for r in recs):
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
                gstats = {}
                if has_gen:  # generator per record (int16 code into gen_vocab; -1 = untagged, not a night_v2 row)
                    codes = []
                    for r in rs:
                        g = r.get("gen")
                        if g is not None and g not in gen_idx:
                            gen_idx[g] = len(gen_vocab)
                            gen_vocab.append(g)
                        codes.append(gen_idx[g] if g is not None else -1)
                        e = gstats.setdefault(g if g is not None else "untagged", dict(records=0, plan_tokens=0))
                        e["records"] += 1
                        e["plan_tokens"] += ptok(r)
                    arrays["gen"] = np.array(codes, dtype=np.int16)
                f = pdir / f"{pool}_{split}.npz"
                np.savez(f, **arrays)
                if m["kind"] == "edit":
                    plan = int(la.sum() + lb.sum())
                elif m["kind"] in ("nochange", "synth"):
                    plan = 2 * int(la.sum())
                else:
                    plan = int(la.sum())
                m["splits"][split] = dict(file=str(f.relative_to(out)).replace(os.sep, "/"), sha256=sha256_file(f),
                                          records=len(rs), plan_tokens=plan, text_tokens=int(la.sum() + lb.sum()),
                                          mean_plan_tokens=plan / max(1, len(rs)))
                if gstats:
                    m["splits"][split]["generators"] = gstats
            m["instr_vocab"] = instr_vocab
            if has_gen:
                m["gen_vocab"] = gen_vocab
            meta[pool] = m
            print(f"[write] {pool}: avail {m['available']['records']}/{m['available']['plan_tokens'] / 1e6:.2f}M "
                  f"{'SUBSET ' if subset else ''}" + " ".join(f"{s}={v['records']}/{v['plan_tokens'] / 1e6:.2f}M"
                                                           for s, v in m["splits"].items()), flush=True)
            if has_gen:
                gt = m["splits"]["train"].get("generators", {})
                tot = sum(e["plan_tokens"] for e in gt.values()) or 1
                print(f"[generators] {pool} train: " + ", ".join(f"{g} {e['records']} rows / {e['plan_tokens'] / 1e6:.2f}M "
                                                                f"({e['plan_tokens'] / tot:.1%})" for g, e in sorted(gt.items())),
                      flush=True)
        return meta


# ====================================================================== mixes / report helpers
def describe(pool_meta_all, target, args, corpus=None):
    """Per-pass rows for mixes.json. corpus: pool -> (records, unique plan tokens, subset)."""
    corpus = corpus or {p: (pool_meta_all[p]["splits"]["train"]["records"], pool_meta_all[p]["splits"]["train"]["plan_tokens"],
                            pool_meta_all[p].get("subset", False)) for p in target}
    rows, pass_eps = episodes_per_pass(corpus, target, args.batch)
    out, by_cat, by_level = [], Counter(), Counter()
    tot = sum(r["plan_tokens"] for r in rows.values())
    for p in sorted(rows):
        r, m = rows[p], pool_meta(p)
        out.append(dict(pool=p, kind=m["kind"], cat=m["cat"], level=m["level"], records=r["records"],
                        unique_plan_tokens=r["unique"], views_per_pass=r["views"], episodes_per_pass=r["episodes"],
                        plan_tokens_per_pass=r["plan_tokens"], weight=r["episodes"] / pass_eps,
                        subset=bool(corpus[p][2]), target_plan_tokens_per_pass=target[p]))
        by_cat[m["cat"] if m["kind"] == "edit" else m["kind"]] += r["plan_tokens"] / tot
        by_level[m["level"] or "none"] += r["plan_tokens"] / tot
    steps = pass_eps // args.batch
    return dict(passes=args.passes, batch_size=args.batch, pass_episodes=int(pass_eps), pass_steps=int(steps),
                total_steps=int(args.passes * steps), schedule_episodes=int(args.passes * pass_eps),
                total_plan_tokens=float(args.passes * tot), pass_plan_tokens=float(tot), pools=out,
                share_by_category=dict(by_cat), share_by_level=dict(by_level))


def print_table(U, target, rows, notes, args, src=None, mix=None):
    T = args.total / args.passes
    cap = args.max_views_total / args.passes
    print(f"\nper pass: T = {T / 1e6:.1f}M plan tokens; cap {cap:.3f} views per pass ({args.max_views_total:g} over "
          f"{args.passes} passes){cat_cap_text(args)}; M = million plan tokens; avail = unique train plan tokens available")
    mrow = {r["pool"]: r for r in (mix or {}).get("pools", [])}
    print(f"{'slot':<16}{'need':>8} {'pool':<24}{'grp':>3}{'avail':>9}{'take':>8}{'views':>7}{'x' + str(args.passes):>7}  source")
    short = {r["slot"]: r for r in rows}
    for (cat, level), lst in SLOT_POOLS.items():
        slot = f"{cat}/{level or '-'}"
        need = SHARES[cat] * T * (0.5 if cat in LEVELED else 1.0)
        first = True
        for g, p in lst:
            u, t = U.get(p, 0.0), target.get(p, 0.0)
            if p in mrow:
                v, t = mrow[p]["views_per_pass"], mrow[p]["plan_tokens_per_pass"]
            else:
                v = (1.0 if t < u else t / u) if (u and t) else 0.0
            print(f"{slot if first else '':<16}{(f'{need / 1e6:.1f}M') if first else '':>8} {p:<24}{g:>3}{u / 1e6:>8.1f}M"
                  f"{t / 1e6:>7.1f}M{v:>7.2f}{v * args.passes:>7.2f}  {(src or {}).get(p, '')}")
            first = False
        s = short.get(slot, {}).get("short", 0.0)
        if s > 1e-6:
            print(f"{'':<16}{'':>8} -> short {s / 1e6:.2f}M before spill")
    for n in notes:
        print("[note]", n)


def estimate_available(args):
    """--report: available unique train plan tokens per pool without tokenizing (estimates marked in ``src``)."""
    U, src = {}, {}
    man2 = json.load(open(MIX_V2 / "manifest.json", encoding="utf-8"))
    p2 = man2["pools"]
    for p in ("grammar_sent_coedit", "fluency_sent_real", "clarity_sent_real", "coherence_sent_real", "polish_sent_real",
              "polish_para_iterater", "restoration"):
        U[p], src[p] = p2[p]["splits"]["train"]["plan_tokens"], "mix_v2"
    handed, _ = chunk_source_ids()
    if getattr(args, "allow_haiku", False) and (HAIKU / "manifest.json").exists():
        for fn in _load_json_retry(HAIKU / "manifest.json")["files"]:
            with open(HAIKU / fn, encoding="utf-8") as f:
                handed |= {json.loads(ln)["source_id"] for ln in f if ln.strip()}
    for p in ("nochange_para", "nochange_sent"):
        z = np.load(MIX_V2 / p2[p]["splits"]["train"]["file"])
        uids = bytes(z["uids"]).decode("utf-8").split("\n")
        la = np.diff(z["a_off"])
        keep = np.array([("/".join(u.split("/")[:3]) if p.endswith("sent") else u) not in handed for u in uids])
        U[p], src[p] = 2 * int(la[keep].sum()), f"mix_v2 minus night_v2 hand-outs ({int((~keep).sum())} rows removed)"
    seen, tok = set(), 0  # synth from v2's real sentence pools (train)
    for p in REAL_SENT_POOLS:
        z = np.load(MIX_V2 / p2[p]["splits"]["train"]["file"])
        af, ao, bo = z["a_flat"], z["a_off"], z["b_off"]
        la, lb = np.diff(ao), np.diff(bo)
        for j in np.nonzero(la + lb <= args.synth_max_pair)[0]:
            a = af[ao[j]:ao[j + 1]].tobytes()
            if a not in seen:
                seen.add(a)
                tok += 2 * int(la[j])
    U["synth"], src["synth"] = tok, f"mix_v2 real sentence pairs <= {args.synth_max_pair}, {len(seen)} unique"
    nm = _load_json_retry(NIGHT / "manifest.json")
    hm = _load_json_retry(HAIKU / "manifest.json") if getattr(args, "allow_haiku", False) and (HAIKU / "manifest.json").exists() else None
    all_pools = {q for lst in SLOT_POOLS.values() for _, q in lst}
    for cat in CATS:
        for lv, level in (("par", "para"), ("sent", "sent")):
            f = nm["files"].get(f"{cat}_{lv}.jsonl")
            p = f"{cat}_{level}_llm"
            if p not in all_pools:
                continue
            base = p2[p]["splits"]["train"]["plan_tokens"] if p in p2 else 0
            U[p] = base + (0.98 * f["tokens"] if f else 0)
            src[p] = ("mix_v2 full_v1 + " if base else "") + (f"night_v2 {f['rows']} rows x0.98 (C manifest {nm['created_utc']})" if f else "none")
            h = (hm or {}).get("files", {}).get(f"haiku_{cat}_{lv}.jsonl")
            if h:
                U[p] += 0.98 * h["tokens"]
                src[p] += f" + Haiku {h['rows']} rows x0.98"
    n_ok, toks = 0, [0.0, 0.0]  # DCLM screened leftovers (chars / chars_per_token)
    for p in sorted(glob.glob(str(NIGHT / "screen" / "ok_*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                if r["id"] in handed:
                    continue
                n_ok += 1
                odd = int(hashlib.sha256(("ncsplit:" + r["id"]).encode()).hexdigest()[:8], 16) % 2
                toks[odd] += len(r["text"]) / args.chars_per_token
    U["nochange_para_dclm"], U["nochange_sent_dclm"] = 0.98 * 2 * toks[0], 0.98 * 0.95 * 2 * toks[1]
    src["nochange_para_dclm"] = src["nochange_sent_dclm"] = f"{n_ok} screened never-handed-out paragraphs (est. chars/{args.chars_per_token})"
    pm = _load_json_retry(NIGHT / "pools" / "manifest.json")["pools"]
    U["restoration_dclm"], src["restoration_dclm"] = 0.98 * pm["restore"]["tokens"], f"C pools manifest {pm['restore']['rows']} rows x0.98"
    U["nochange_para_dclmraw"] = U["nochange_sent_dclmraw"] = 0.98 * pm["nochange"]["tokens"]
    src["nochange_para_dclmraw"] = src["nochange_sent_dclmraw"] = f"C pools manifest {pm['nochange']['rows']} rows, half each, x0.98"
    cm = json.load(open(C4200M_DIR / "manifest.json", encoding="utf-8"))
    U["grammar_sent_c4200m"], src["grammar_sent_c4200m"] = 0.98 * cm["gpt2_tokens_before_plus_after"], f"C4_200M v1 {cm['rows_kept']} pairs x0.98"
    return U, src


# ====================================================================== main
def cat_caps(args):
    """--cat-max-views-total CAT=X ... -> {category: X / passes} (per-pass views cap of that category's pools, used
    only for its shortfall after the normal allocation)."""
    out = {}
    for item in args.cat_max_views_total or []:
        c, x = item.split("=")
        assert c in SHARES and c not in ("restoration", "synth"), f"unknown or unsupported category {c}"
        out[c] = float(x) / args.passes
    return out


def cat_cap_text(args):
    cc = cat_caps(args)
    return "".join(f"; {c} cap {v:.3f} ({v * args.passes:g} over {args.passes})" for c, v in sorted(cc.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/editdata/mix_v3")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--total", type=float, default=1e9)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-views-total", type=float, default=4.0)
    ap.add_argument("--cat-max-views-total", nargs="*", default=[], metavar="CAT=X",
                    help="per-category views over all passes, used only for that category's shortfall left after "
                         "the normal allocation, before it spills to grammar (e.g. clarity=6 coherence=6)")
    ap.add_argument("--allow-haiku", action="store_true",
                    help="also take night_v2 rows drafted by Claude Haiku (C's files hold only rows kept after its Qwen "
                         "review; user decision 2026-10-09); default: Qwen2.5-32B rows only")
    ap.add_argument("--haiku-dir", default=None,
                    help="with --allow-haiku: folder of haiku_<cat>_<par|sent>.jsonl + manifest.json (default "
                         "data/editdata/night_v2/haiku_edit, C's; another folder only for dry tests)")
    ap.add_argument("--max-pair", type=int, default=480)
    ap.add_argument("--synth-max-pair", type=int, default=256)
    ap.add_argument("--max-rest", type=int, default=250)
    ap.add_argument("--min-sent", type=int, default=5)
    ap.add_argument("--max-sent", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--clean-p", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chars-per-token", type=float, default=4.5, help="--report estimate for screened DCLM text")
    ap.add_argument("--report", action="store_true", help="fast: available vs need per slot / pool, no tokenizing")
    a = ap.parse_args()
    a.c4200m_limit = 0  # v2 attribute (v2's c4200m stage is not run)
    if a.haiku_dir:
        global HAIKU
        HAIKU = Path(a.haiku_dir) if Path(a.haiku_dir).is_absolute() else ROOT / a.haiku_dir
    t0 = time.time()
    T, cap = a.total / a.passes, a.max_views_total / a.passes
    cc = cat_caps(a)
    if a.report:
        U, src = estimate_available(a)
        target, notes, rows = allocate(U, T, cap, cat_cap=cc)
        print_table(U, target, rows, notes, a, src)
        print(f"[report] {time.time() - t0:.0f}s")
        return
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    b = BuilderV3(a)
    print(f"[eval keys] {len(b.ek.whole)} texts, {len(b.ek.sent)} sentences from {len(b.ek.files)} files", flush=True)
    b.stage("snapshot", b.snapshot, out)
    b.stage("chunk_sources", b.handed_out)
    b.stage("real", b.real)
    b.stage("full_v1", b.llm)
    b.stage("night_v2", b.night)
    b.stage("c4", b.c4_v3)
    b.stage("synth", b.synth)
    b.stage("dclm_screened", b.dclm_screened)
    b.stage("dclm_restore", b.dclm_restore)

    def avail_now():
        U = {}
        for p, recs in b.pools.items():
            k = pool_meta(p)["kind"]
            U[p] = sum((len(r["b"]) + len(r["a"])) if k == "edit" else (2 * len(r["a"]) if k in ("nochange", "synth") else len(r["a"]))
                       for r in recs if r["split"] == "train")
        return U

    U = avail_now()
    cm = json.load(open(C4200M_DIR / "manifest.json", encoding="utf-8"))
    U["grammar_sent_c4200m"] = 0.98 * cm["gpt2_tokens_before_plus_after"]
    pm = _load_json_retry(out / "snapshot" / "night_v2_pools_manifest.json")["pools"]
    U["nochange_para_dclmraw"] = U["nochange_sent_dclmraw"] = 0.98 * pm["nochange"]["tokens"]
    target, notes, rows = allocate(U, T, cap, cat_cap=cc)
    if any(p in target for p in ("nochange_para_dclmraw", "nochange_sent_dclmraw")):
        b.stage("dclm_raw_nochange", b.dclm_raw_nochange)
        U.update({p: v for p, v in avail_now().items() if p.endswith("dclmraw")})
        target, notes, rows = allocate(U, T, cap, cat_cap=cc)
    else:
        notes.append("unscreened DCLM no-change (priority 2) not needed: not processed")
        U.pop("nochange_para_dclmraw"), U.pop("nochange_sent_dclmraw")
    avail_c4 = b.stage("c4200m", b.c4200m_v3, target["grammar_sent_c4200m"])
    pmeta = b.stage("write", b.write_v3, out, target, {"grammar_sent_c4200m": avail_c4})
    unused = {p: dict(records=sum(1 for r in recs if r["split"] == "train"), plan_tokens=U.get(p, 0))
              for p, recs in b.pools.items() if p not in target}
    tasks = list(v2.TASKS)
    cat_instr = {c: [list(k) for k in sorted(b.cat_instr.get(c, {}))] for c in tasks}
    cat_instr_text = {c: [b.cat_instr[c][k] for k in sorted(b.cat_instr.get(c, {}))] for c in tasks}
    if not (NIGHT / "grammar_sent.jsonl").exists():
        notes.append("night_v2 grammar_sent.jsonl absent (LLM grammar sentences withdrawn): grammar/sent = CoEdIT + C4_200M")
    sub_notes = [f"{p}: subset {pmeta[p]['splits']['train']['records']} train records of {pmeta[p]['available']['records']} "
                 + ("available (helper C's manifest, all shards and splits)" if p == "grammar_sent_c4200m" else "available")
                 for p in sorted(pmeta) if pmeta[p]["subset"]]
    if b.extra.get("night_v2_files_not_used"):
        notes.append(f"night_v2 files without a slot, not used: {b.extra['night_v2_files_not_used']}")
    excl = [f"{k}: {r} {n}" for k, v in sorted(b.drops.items()) if k.startswith("night_v2")
            for r, n in v.items() if r.startswith("excluded")]
    man = dict(format="reviser-editmix-v3", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               builder=dict(script="scripts/editdata/build_mix_v3.py", sha256=sha256_file(__file__),
                            v2_script_sha256=sha256_file(HERE / "build_mix_v2.py"),
                            tasktokens_sha256=sha256_file(ROOT / "src" / "reviser_v2" / "tasktokens.py"), args=vars(a)),
               inputs=b.inputs, snapshot=b.snap, eval_files=b.ek.files,
               eval_keys=dict(texts=len(b.ek.whole), sentences=len(b.ek.sent)),
               drops={k: dict(v) for k, v in b.drops.items()}, c4_screen=getattr(b, "c4_screen", None), extra=b.extra,
               tasks=tasks, cat_instr=cat_instr, cat_instr_text=cat_instr_text, pools=pmeta, unused_pools=unused,
               timing=b.timing, seconds=round(time.time() - t0, 1))
    body = json.dumps(man, sort_keys=True)
    man["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(man, open(out / "manifest.json", "w", encoding="utf-8"), indent=1)
    main_mix = describe(pmeta, target, a)
    mixes = {"format": "reviser-editmix-v3-mixes", "manifest_sha256": man["sha256"], "builder_sha256": sha256_file(__file__),
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a), "shares": SHARES,
             "max_views_per_pass": cap, "max_views_per_pass_by_category": cc, "notes": notes + sub_notes + excl,
             "generators_train": {p: m["splits"]["train"]["generators"] for p, m in sorted(pmeta.items())
                                  if "generators" in m["splits"]["train"]},
             "variants": {"main": main_mix}}
    json.dump(mixes, open(out / "mixes.json", "w", encoding="utf-8"), indent=1)
    print_table(U, target, rows, notes + sub_notes + excl, a, mix=main_mix)
    v = main_mix
    print(f"[mix] pass: {v['pass_plan_tokens'] / 1e6:.1f}M plan tokens, {v['pass_episodes']} episodes, {v['pass_steps']} steps; "
          f"total {v['total_plan_tokens'] / 1e6:.1f}M, {v['total_steps']} steps")
    print("[mix] cat " + json.dumps({c: round(s, 4) for c, s in v["share_by_category"].items()}) +
          " level " + json.dumps({c: round(s, 4) for c, s in v["share_by_level"].items()}))
    print(f"[done] {time.time() - t0:.0f}s manifest sha256 {man['sha256']}; timings {json.dumps(b.timing)}", flush=True)


if __name__ == "__main__":
    main()
