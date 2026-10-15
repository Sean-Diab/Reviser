"""Reviser-Edit v2 training mix (plan steps 9-10, notes/reviser-edit-data-plan-2026-10-06.md; run note
notes/edit-runs-2026-10-08.md).

Pools are built by scripts/editdata/build_mix_v2.py into ``<mix_dir>/pools/<pool>_<split>.npz`` with a
``<mix_dir>/manifest.json``. Each pool has one category (grammar, fluency, clarity, coherence, polish; or none), one
level (sent / para) and one kind:

* ``edit``         (instruction, before, after) revision pair. Prompt = [TASK_cat] + instruction, where the instruction
                   is the record's own (IteraTeR / CoEdIT) or, for sources without one (LLM drafts, C4_200M), with
                   probability ``p_instr`` an instruction sampled from the category's real instructions, else empty.
* ``nochange``     a clean text given as the draft; target = the same text (answer: immediate Stop). The category is
                   drawn uniformly per episode; the instruction rule is the same as above.
* ``restoration``  D4 blank-canvas restoration of a clean text: prompt = first ``prefix_frac`` of the tokens (no task
                   token), obfuscation policy as configs/realedit100m/d4.yaml (junk from the restoration corpus).

* ``synth``        (v3, notes/edit-1b-build-2026-10-08.md) the September D4 ``synth_repair`` kind on a pool of clean
                   texts: 0-3 synthetic token errors (replace / delete / insert corpus-random tokens, swap two adjacent
                   tokens; P = ``synth.n_errors_probs``, default 0.2/0.5/0.2/0.1, so 20% are clean drafts) supplied as the
                   draft; prompt = [TASK_grammar] + "Fix the errors in this text" (as the synth eval sets); target = clean.

``layout: ar`` gives the matched autoregressive baseline on the SAME episode stream (same schedule, same per-episode
random draws): edit / nochange / synth -> prompt = [TASK] + instruction + ":" + before, blank canvas, target typed left
to right; restoration -> ``ltr`` continuation of the prefix (as configs/realedit300m/ar.yaml).

Schedule. Episode ``idx`` of a split is a pure function of (data_seed, split, idx): a fixed array assigns each index a
pool (multinomial with the mix weights) and a record (concatenated random permutations of the pool), so every record of
a pool is used floor/ceil(views) times, in random order. ``schedule_stats`` reports the realised counts.

Pass mode (train split, ``task.passes: P`` and ``episodes_per_pass`` on every mix entry; v3): pass p (0-based) is a
block of sum(episodes_per_pass) episodes in which pool i appears exactly episodes_per_pass[i] times; positions and
records are reshuffled for every pass (rng seeded by (data_seed, split, 1000 + p)), so each pass shows the same corpus
once (an up-sampled pool's records floor/ceil(views per pass) times) in a new order. The dev split keeps the
multinomial schedule with the ``weight`` values.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..actions import ActionSpace
from ..tasktokens import TASKS, task_id
from ..trajectories import Episode, ObfuscationPolicy, build_episode, ltr_ops, repair_ops, restoration_ops
from .realedit import FIX_INSTR, ar_prompt

ROOT = Path(__file__).resolve().parents[3]
SPLIT_CODE = {"train": 0, "dev": 1, "test": 2}


class Pool:
    """Token-id records of one pool split (uint16 flat arrays + offsets)."""

    def __init__(self, path: Path, meta: Dict):
        z = np.load(path)
        self.a_flat, self.a_off = z["a_flat"], z["a_off"]
        self.has_b = "b_flat" in z.files
        if self.has_b:
            self.b_flat, self.b_off = z["b_flat"], z["b_off"]
        self.instr = z["instr"] if "instr" in z.files else None
        self.n = len(self.a_off) - 1
        self.meta = meta
        self.instr_vocab: List[List[int]] = meta.get("instr_vocab", [])

    def after(self, j: int) -> List[int]:
        return self.a_flat[self.a_off[j]:self.a_off[j + 1]].tolist()

    def before(self, j: int) -> List[int]:
        return self.b_flat[self.b_off[j]:self.b_off[j + 1]].tolist() if self.has_b else self.after(j)

    def instruction(self, j: int) -> Optional[List[int]]:
        if self.instr is None or self.instr[j] < 0:
            return None
        return list(self.instr_vocab[int(self.instr[j])])

    def plan_tokens(self) -> np.ndarray:
        """Per-record tokens in the plan's convention: before + after (edit), 2 x text (nochange), text (restoration)."""
        la = np.diff(self.a_off)
        if self.meta["kind"] == "restoration":
            return la
        if self.meta["kind"] in ("nochange", "synth"):
            return 2 * la
        return la + np.diff(self.b_off)


def load_manifest(mix_dir) -> Dict:
    d = Path(mix_dir)
    if not d.is_absolute():
        d = ROOT / d
    return json.load(open(d / "manifest.json", encoding="utf-8")), d


def build_schedule(weights: Sequence[float], sizes: Sequence[int], n: int, seed: int, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """(pool index per episode, record index per episode) for ``n`` episodes."""
    rng = np.random.default_rng([int(seed), SPLIT_CODE.get(split, 9)])
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    pool_of = rng.choice(len(w), size=n, p=w).astype(np.int16)
    rec_of = np.zeros(n, dtype=np.int64)
    for p, size in enumerate(sizes):
        pos = np.nonzero(pool_of == p)[0]
        if len(pos) == 0:
            continue
        assert size > 0, f"pool {p} has weight but no records"
        reps = -(-len(pos) // size)
        recs = np.concatenate([rng.permutation(size) for _ in range(reps)])[:len(pos)]
        rec_of[pos] = recs
    return pool_of, rec_of


def build_pass_schedule(counts: Sequence[int], sizes: Sequence[int], passes: int, seed: int,
                        split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pass mode: ``passes`` blocks of sum(counts) episodes; in each block pool i appears exactly counts[i] times and its
    records floor/ceil(counts[i] / sizes[i]) times each; every block has its own shuffle."""
    counts = [int(c) for c in counts]
    n_pass = sum(counts)
    base = np.repeat(np.arange(len(counts), dtype=np.int16), counts)
    pool_of = np.empty(passes * n_pass, dtype=np.int16)
    rec_of = np.empty(passes * n_pass, dtype=np.int64)
    for p in range(passes):
        rng = np.random.default_rng([int(seed), SPLIT_CODE.get(split, 9), 1000 + p])
        po = rng.permutation(base)
        ro = np.zeros(n_pass, dtype=np.int64)
        order = np.argsort(po, kind="stable")  # positions grouped by pool, ascending within a pool
        bounds = np.concatenate([[0], np.cumsum(counts)])
        for i, (c, size) in enumerate(zip(counts, sizes)):
            if c == 0:
                continue
            assert size > 0, f"pool {i} has episodes but no records"
            reps = -(-c // size)
            ro[order[bounds[i]:bounds[i + 1]]] = np.concatenate([rng.permutation(size) for _ in range(reps)])[:c]
        pool_of[p * n_pass:(p + 1) * n_pass] = po
        rec_of[p * n_pass:(p + 1) * n_pass] = ro
    return pool_of, rec_of


class EditMixSource:
    def __init__(self, space: ActionSpace, task: Dict, mix: List[Dict], data_seed: int, split: str = "train"):
        self.space, self.task, self.mix, self.seed, self.split = space, task, mix, data_seed, split
        self.layout = task.get("layout", "edit")
        assert self.layout in ("edit", "ar"), self.layout
        man, d = load_manifest(task["mix_dir"])
        self.manifest_sha256 = man.get("sha256")
        if task.get("manifest_sha256") and task["manifest_sha256"] != self.manifest_sha256:
            raise ValueError(f"mix manifest sha256 {self.manifest_sha256} != config {task['manifest_sha256']}")
        self.pools: List[Pool] = []
        for m in mix:
            pm = man["pools"][m["pool"]]
            sp = pm["splits"][split]
            if sp.get("sha256"):  # pool files are content-checked against the mix manifest
                h = hashlib.sha256((d / sp["file"]).read_bytes()).hexdigest()
                if h != sp["sha256"]:
                    raise ValueError(f"{sp['file']}: sha256 {h} != manifest {sp['sha256']}")
            self.pools.append(Pool(d / sp["file"], dict(pm, name=m["pool"])))
        self.cat_instr: Dict[str, List[List[int]]] = man["cat_instr"]
        self.p_instr = float(task.get("p_instr", 0.5))
        rest = task.get("restoration", {})
        self.prefix_frac = float(rest.get("prefix_frac", 0.35))
        self.policy = ObfuscationPolicy(**rest.get("policy", {"p_del": 0.7, "p_move": 0.18, "p_ins": 0.12}))
        self.synth_cfg = dict(task.get("synth") or {})
        self.passes = int(task.get("passes") or 0) if split == "train" else 0
        self.pass_episodes = None
        if self.passes:  # pass mode (v3): exact per-pass counts, reshuffled every pass
            counts = [int(m["episodes_per_pass"]) for m in mix]
            self.pass_episodes = sum(counts)
            self.pool_of, self.rec_of = build_pass_schedule(counts, [p.n for p in self.pools], self.passes, data_seed, split)
            if task.get("schedule_episodes") is not None and int(task["schedule_episodes"]) != len(self.pool_of):
                raise ValueError(f"schedule_episodes {task['schedule_episodes']} != passes x sum(episodes_per_pass) "
                                 f"{len(self.pool_of)}")
        else:
            n = int(task["schedule_episodes"] if split == "train" else task.get("dev_episodes", 4096))
            self.pool_of, self.rec_of = build_schedule([m["weight"] for m in mix], [p.n for p in self.pools], n, data_seed,
                                                       split)
        self._reservoir = None
        self._synth_reservoir = None

    # corpus-unigram junk for restoration Insert(junk) (as tasks/realedit.py: 2000 random records' tokens)
    def junk(self, rng: random.Random) -> int:
        if self._reservoir is None:
            rr = random.Random(f"junk:{self.seed}")
            src = [p for p in self.pools if p.meta["kind"] == "restoration"] or self.pools
            res: List[int] = []
            for _ in range(2000):
                p = src[rr.randrange(len(src))]
                res += p.after(rr.randrange(p.n))
            self._reservoir = res
        return self._reservoir[rng.randrange(len(self._reservoir))]

    # synth: corpus-unigram junk from the synth pool (as RealEditSource: 2000 random clean texts' tokens)
    def synth_junk(self, rng: random.Random) -> int:
        if self._synth_reservoir is None:
            rr = random.Random(f"synthjunk:{self.seed}")
            src = [p for p in self.pools if p.meta["kind"] == "synth" and p.n]
            res: List[int] = []
            for _ in range(2000):
                p = src[rr.randrange(len(src))]
                res += p.after(rr.randrange(p.n))
            self._synth_reservoir = res
        return self._synth_reservoir[rng.randrange(len(self._synth_reservoir))]

    def corrupt(self, clean: List[int], rng: random.Random, n_err: int) -> List[int]:
        """The September synth_repair corruption (tasks/realedit.py RealEditSource.corrupt), synth-pool junk."""
        d = list(clean)
        for _ in range(n_err):
            k = rng.choice(["replace", "delete", "insert", "swap"])
            if k == "insert" or not d:
                d.insert(rng.randrange(len(d) + 1), self.synth_junk(rng))
            elif k == "delete":
                del d[rng.randrange(len(d))]
            elif k == "replace":
                d[rng.randrange(len(d))] = self.synth_junk(rng)
            elif len(d) >= 2:
                i = rng.randrange(len(d) - 1)
                d[i], d[i + 1] = d[i + 1], d[i]
        return d

    def prompt_for(self, cat: str, own: Optional[List[int]], rng: random.Random) -> List[int]:
        if own is None:
            # one draw always (keeps the per-episode random stream identical whatever the outcome)
            u = rng.random()
            pool = self.cat_instr.get(cat) or [[]]
            k = rng.randrange(len(pool))
            own = list(pool[k]) if u < self.p_instr else []
        return [task_id(self.space.vocab_size, cat)] + own

    def lookup(self, idx: int) -> Tuple[Pool, int]:
        i = idx % len(self.pool_of)
        return self.pools[int(self.pool_of[i])], int(self.rec_of[i])

    def episode(self, idx: int) -> Tuple[Episode, str]:
        rng = random.Random(f"{self.seed}:{self.split}:{idx}")
        pool, j = self.lookup(idx)
        kind, meta = pool.meta["kind"], pool.meta
        label = f"{kind}:{meta.get('cat') or '-'}:{meta.get('level') or '-'}"
        sp = self.space
        if kind == "restoration":
            text = pool.after(j)
            k = int(round(self.prefix_frac * len(text)))
            prompt, tgt = text[:k], text[k:]
            if self.layout == "ar":
                return build_episode(sp, prompt, [], tgt, ltr_ops(tgt)), label
            ops = restoration_ops(tgt, rng, self.policy, lambda g, canvas, cur: self.junk(g))
            return build_episode(sp, prompt, [], tgt, ops), label
        if kind == "nochange":
            cat = TASKS[rng.randrange(len(TASKS))]
            after = pool.after(j)
            before = after
            prompt = self.prompt_for(cat, None, rng)
        elif kind == "edit":
            cat = meta["cat"]
            before, after = pool.before(j), pool.after(j)
            prompt = self.prompt_for(cat, pool.instruction(j), rng)
        elif kind == "synth":
            after = pool.after(j)
            probs = self.synth_cfg.get("n_errors_probs", [0.2, 0.5, 0.2, 0.1])
            ne = rng.choices(range(len(probs)), weights=probs)[0]
            before = self.corrupt(after, rng, ne)
            prompt = [task_id(self.space.vocab_size, self.synth_cfg.get("cat", "grammar"))] + list(FIX_INSTR)
        else:
            raise ValueError(kind)
        if self.layout == "ar":
            return build_episode(sp, ar_prompt(prompt, before), [], after, ltr_ops(after)), label
        return build_episode(sp, prompt, before, after, repair_ops(before, after, "rtl")), label

    def describe(self) -> Dict:
        """Recorded in the training manifest: mix manifest identity, pool files and the realised schedule."""
        extra = dict(passes=self.passes, pass_episodes=self.pass_episodes) if self.passes else {}
        return dict(mix_dir=str(self.task["mix_dir"]), manifest_sha256=self.manifest_sha256, layout=self.layout,
                    split=self.split, schedule_episodes=int(len(self.pool_of)), data_seed=self.seed, **extra,
                    pools={p.meta["name"]: dict(file=p.meta["splits"][self.split]["file"],
                                                sha256=p.meta["splits"][self.split].get("sha256"),
                                                kind=p.meta["kind"], cat=p.meta.get("cat"), level=p.meta.get("level"))
                           for p in self.pools},
                    schedule=self.schedule_stats())

    def schedule_stats(self) -> Dict:
        """Realised use of each pool over the schedule: episodes, plan tokens, views, records never / max used."""
        out = {}
        for p_i, pool in enumerate(self.pools):
            sel = self.rec_of[self.pool_of == p_i]
            cnt = np.bincount(sel, minlength=pool.n) if pool.n else np.zeros(0, dtype=np.int64)
            tok = pool.plan_tokens()
            out[pool.meta["name"]] = dict(
                episodes=int(len(sel)), records=int(pool.n), plan_tokens=int((cnt * tok).sum()),
                unique_plan_tokens=int(tok.sum()), views=float((cnt * tok).sum() / max(1, tok.sum())),
                min_uses=int(cnt.min()) if pool.n else 0, max_uses=int(cnt.max()) if pool.n else 0,
                never_used=int((cnt == 0).sum()))
            if self.passes:  # per pass: how often each record of the pool appears in that pass
                per = []
                for p in range(self.passes):
                    s = slice(p * self.pass_episodes, (p + 1) * self.pass_episodes)
                    c = np.bincount(self.rec_of[s][self.pool_of[s] == p_i], minlength=pool.n) if pool.n else np.zeros(0, np.int64)
                    per.append(dict(episodes=int(c.sum()), views=float((c * tok).sum() / max(1, tok.sum())),
                                    min_uses=int(c.min()) if pool.n else 0, max_uses=int(c.max()) if pool.n else 0))
                out[pool.meta["name"]]["per_pass"] = per
        return out
