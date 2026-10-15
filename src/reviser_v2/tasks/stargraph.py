"""Star-graph path-finding benchmark (Bachmann & Nagarajan 2024; ILM paper, Patel et al. 2025).

Protocol source notes: ``notes/sources/stargraph-protocol-2026-09-24.md`` (verified details vs.
assumptions). Summary of what is reproduced here:

* ``symmetric`` generator (ILM "Star easy" = code ``star_small``; BN24 ``G_{d,l}(N)``): the source is
  the centre node; ``degree`` arms of ``pathlength`` nodes each (source included) leave it; the goal
  is the end of one arm. All node labels are distinct, drawn uniformly from ``[0, n_nodes)``.
* ``asymmetric`` generator (ILM "Star medium/hard" = code ``vstar_small`` / ``vstar_medium``):
  ``degree`` directed chains with lengths ~ U{min_pathlength..max_pathlength} (in nodes) all pass
  through one shared centre node at join index ~ U{0..len-min_plan_length-1}; chain 0 is the answer
  (source = its first node, goal = its last node).
* Serialization. ``prompt_format='ilm'``: flattened edge pairs in shuffled order, then source, goal
  (no separators, as ILM ``SimpleSpaceTokenizer``); ``'bn24'``: ``u v | u v | ... / s g`` (BN24
  numeral tokenizer, minus the '=' which our SEP/EDIT replaces). ``target_format='edge_pairs'``:
  the path as flattened edges ``u1 v1 u2 v2 ...`` (ILM, 2*(L-1) tokens); ``'nodes'``: the node list
  (BN24, L tokens).

Token map (content block of the ActionSpace): node label i -> id i for i < n_nodes; EDGE_SEP ('|')
-> n_nodes; QUERY ('/') -> n_nodes + 1. So ``n_content = n_nodes + 2`` for every format (separators
exist in the vocabulary even when a format does not use them). The ActionSpace SEP/EDIT tokens play
the role of ILM's BOS.

Splits. dev/test graph sets are fixed functions of (spec.eval_seed, split, index), drawn from RNG
namespaces disjoint from training; training graphs are drawn from ``data_seed`` and any graph whose
canonical key (edge set, source, goal) appears in the dev or test set is rejected and redrawn, so
train/eval disjointness holds by construction (and is tested). ``n_train`` optionally restricts
training to a finite pool (ILM used 50k fixed training graphs); trajectories are still re-sampled
per episode index.

Trajectory policies (``mix`` kinds):
  * ``ltr``          left-to-right insertion (AR). With ``reverse_target: true`` the canvas target
                     is the reversed serialization (separate reverse-AR model, BN24 "reverse"
                     training / ILM "ARMO" on Star easy); decode with ``canonical_output``.
  * ``rtl_reverse``  (alias ``reverse``) the forward target built right-to-left by one edit model:
                     insert the last token, then ``goto(0)`` + insert before each earlier token.
  * ``restoration``  time-reversed random obfuscation walk (``trajectories.restoration_ops``).

Generalization tests (NOT part of the ILM protocol; clearly separate from dev/test):
``generalization_graphs(spec, 'gen_relabeled' | 'gen_role_holdout' | 'gen_longer_path', ...)``.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..actions import ActionSpace
from ..trajectories import Episode, ObfuscationPolicy, build_episode, ltr_ops, restoration_ops, uniform_junk

Edge = Tuple[int, int]

GENERATORS = ("symmetric", "asymmetric")
PROMPT_FORMATS = ("ilm", "bn24")
TARGET_FORMATS = ("edge_pairs", "nodes")
GEN_VARIANTS = ("gen_relabeled", "gen_role_holdout", "gen_longer_path")


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StarSpec:
    generator: str = "symmetric"
    degree: int = 3
    pathlength: int = 5          # symmetric only: nodes per arm incl. the source
    min_pathlength: int = 3      # asymmetric only: chain length range (nodes)
    max_pathlength: int = 6
    min_plan_length: int = 2     # asymmetric only: >= this many edges from centre to chain end
    n_nodes: int = 20            # |V|: node labels are 0..n_nodes-1
    prompt_format: str = "ilm"
    target_format: str = "edge_pairs"
    n_train: Optional[int] = 50000  # finite training pool (ILM: 50k); None = fresh graph per index
    n_dev: int = 1000
    n_test: int = 5000
    eval_seed: int = 20260924
    role_holdout: Tuple[int, ...] = ()  # labels never used as source/goal in train/dev/test

    def __post_init__(self):
        if self.generator not in GENERATORS:
            raise ValueError(f"unknown generator {self.generator!r}")
        if self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"unknown prompt_format {self.prompt_format!r}")
        if self.target_format not in TARGET_FORMATS:
            raise ValueError(f"unknown target_format {self.target_format!r}")
        object.__setattr__(self, "role_holdout", tuple(int(x) for x in self.role_holdout))
        if any(not 0 <= h < self.n_nodes for h in self.role_holdout):
            raise ValueError("role_holdout labels must be node labels")
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.generator == "symmetric":
            if self.pathlength < 2:
                raise ValueError("pathlength must be >= 2")
        else:
            if not 1 <= self.min_plan_length <= self.min_pathlength - 1 or self.min_pathlength > self.max_pathlength:
                raise ValueError("need 1 <= min_plan_length < min_pathlength <= max_pathlength")
        if self.max_graph_nodes > self.n_nodes:
            raise ValueError(f"graphs need up to {self.max_graph_nodes} distinct labels > n_nodes={self.n_nodes}")
        if len(self.role_holdout) > self.n_nodes - 2:
            raise ValueError("role_holdout leaves too few labels for source and goal")

    # ---- token map ------------------------------------------------------
    @property
    def EDGE_SEP(self) -> int:
        return self.n_nodes

    @property
    def QUERY(self) -> int:
        return self.n_nodes + 1

    @property
    def n_content(self) -> int:
        return self.n_nodes + 2

    # ---- sizes ----------------------------------------------------------
    @property
    def max_path_nodes(self) -> int:
        return self.pathlength if self.generator == "symmetric" else self.max_pathlength

    @property
    def max_graph_nodes(self) -> int:
        if self.generator == "symmetric":
            return 1 + self.degree * (self.pathlength - 1)
        return self.degree * self.max_pathlength - self.degree + 1

    @property
    def max_edges(self) -> int:
        return self.degree * (self.max_path_nodes - 1)

    def target_len(self, path_nodes: int) -> int:
        return 2 * (path_nodes - 1) if self.target_format == "edge_pairs" else path_nodes

    @property
    def max_target_len(self) -> int:
        return self.target_len(self.max_path_nodes)

    @property
    def max_prompt_len(self) -> int:
        e = self.max_edges
        return 2 * e + 2 if self.prompt_format == "ilm" else 3 * e - 1 + 3

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["role_holdout"] = list(self.role_holdout)
        return d


# ILM paper Table 4 / code configs (dhruvdcoder/ILM configs/lightning_train/dataset/*_v2.yaml).
# NOTE medium: the paper text/Table 4 say degree 2, the released config and the paper's own
# Appendix A.3.1 examples use degree 3; we follow the code (see protocol notes).
PRESETS: Dict[str, Dict] = {
    "easy": dict(generator="symmetric", degree=3, pathlength=5, n_nodes=20),
    "medium": dict(generator="asymmetric", degree=3, min_pathlength=3, max_pathlength=6, min_plan_length=2, n_nodes=20),
    "medium_deg2": dict(generator="asymmetric", degree=2, min_pathlength=3, max_pathlength=6, min_plan_length=2, n_nodes=20),
    "hard": dict(generator="asymmetric", degree=5, min_pathlength=6, max_pathlength=12, min_plan_length=5, n_nodes=56),
    # BN24 headline setting G_{2,5}(50) with the numeral format (README command; paper uses N=100).
    "bn24_d2_l5": dict(generator="symmetric", degree=2, pathlength=5, n_nodes=50, prompt_format="bn24",
                       target_format="nodes", n_train=200000, n_test=20000),
}


def make_spec(preset: Optional[str] = None, **overrides) -> StarSpec:
    base = dict(PRESETS[preset]) if preset else {}
    base.update(overrides)
    return StarSpec(**base)


def spec_from_task_cfg(task_cfg: Dict) -> StarSpec:
    """``task: {name: stargraph, preset: hard, spec: {...}}`` -> StarSpec (spec overrides preset)."""
    return make_spec(task_cfg.get("preset"), **dict(task_cfg.get("spec", {}) or {}))


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StarGraph:
    edges: Tuple[Edge, ...]   # shuffled presentation order
    source: int
    goal: int
    path: Tuple[int, ...]     # node list source..goal
    center: int

    @property
    def key(self) -> Tuple[FrozenSet[Edge], int, int]:
        """Canonical identity: edge SET plus query (presentation order ignored)."""
        return frozenset(self.edges), self.source, self.goal

    def digest(self) -> str:
        s = ";".join(f"{u},{v}" for u, v in sorted(self.edges)) + f"/{self.source},{self.goal}"
        return hashlib.sha256(s.encode()).hexdigest()

    def relabel(self, perm: Dict[int, int]) -> "StarGraph":
        f = lambda x: perm.get(x, x)
        return StarGraph(tuple((f(u), f(v)) for u, v in self.edges), f(self.source), f(self.goal),
                         tuple(f(x) for x in self.path), f(self.center))


def _chains_to_graph(rng: random.Random, chains: List[List[int]], center: int) -> StarGraph:
    edges: List[Edge] = [(c[i], c[i + 1]) for c in chains for i in range(len(c) - 1)]
    rng.shuffle(edges)
    ans = chains[0]
    return StarGraph(tuple(edges), ans[0], ans[-1], tuple(ans), center)


def symmetric_graph(rng: random.Random, degree: int, pathlength: int, n_nodes: int,
                    answer_len: Optional[int] = None) -> StarGraph:
    """BN24 ``star_graph`` / ILM ``simple_star_graph`` distribution with distinct uniform labels.

    ``answer_len`` (nodes) overrides only the answer arm (used by the longer-path generalization test).
    """
    la = pathlength if answer_len is None else answer_len
    need = 1 + (degree - 1) * (pathlength - 1) + (la - 1)
    if need > n_nodes:
        raise ValueError(f"needs {need} labels > {n_nodes}")
    labels = rng.sample(range(n_nodes), need)
    src = labels[0]
    chains, j = [], 1
    for a in range(degree):
        L = la if a == 0 else pathlength
        chains.append([src] + labels[j:j + L - 1])
        j += L - 1
    return _chains_to_graph(rng, chains, src)


def asymmetric_graph(rng: random.Random, degree: int, min_pathlength: int, max_pathlength: int,
                     min_plan_length: int, n_nodes: int, answer_len: Optional[int] = None) -> StarGraph:
    """ILM ``asymmetric_variable_armlength_star_graph`` distribution (see module doc)."""
    lens = [rng.randint(min_pathlength, max_pathlength) for _ in range(degree)]
    if answer_len is not None:
        lens[0] = answer_len
    joins = [rng.randrange(L - min_plan_length) for L in lens]
    need = sum(lens) - degree + 1
    if need > n_nodes:
        raise ValueError(f"needs {need} labels > {n_nodes}")
    labels = rng.sample(range(n_nodes), need)
    center = labels[0]
    chains, j = [], 1
    for L, jn in zip(lens, joins):
        c = []
        for k in range(L):
            if k == jn:
                c.append(center)
            else:
                c.append(labels[j])
                j += 1
        chains.append(c)
    return _chains_to_graph(rng, chains, center)


def sample_graph(spec: StarSpec, rng: random.Random, answer_len: Optional[int] = None) -> StarGraph:
    if spec.generator == "symmetric":
        return symmetric_graph(rng, spec.degree, spec.pathlength, spec.n_nodes, answer_len)
    return asymmetric_graph(rng, spec.degree, spec.min_pathlength, spec.max_pathlength, spec.min_plan_length,
                            spec.n_nodes, answer_len)


def _respects_holdout(spec: StarSpec, g: StarGraph) -> bool:
    return not spec.role_holdout or (g.source not in spec.role_holdout and g.goal not in spec.role_holdout)


def _in_dist_graph(spec: StarSpec, rng: random.Random, reject: FrozenSet = frozenset(), max_tries: int = 10000) -> StarGraph:
    for _ in range(max_tries):
        g = sample_graph(spec, rng)
        if _respects_holdout(spec, g) and g.key not in reject:
            return g
    raise RuntimeError("could not sample an admissible graph (label space too small for the reject set?)")


@lru_cache(maxsize=32)
def eval_graphs(spec: StarSpec, split: str) -> Tuple[StarGraph, ...]:
    """Fixed dev/test graphs: graph i is a pure function of (spec, split, i)."""
    n = {"dev": spec.n_dev, "test": spec.n_test}[split]
    return tuple(_in_dist_graph(spec, random.Random(f"star-eval:{spec.eval_seed}:{split}:{i}")) for i in range(n))


@lru_cache(maxsize=32)
def eval_keys(spec: StarSpec) -> FrozenSet:
    return frozenset(g.key for s in ("dev", "test") for g in eval_graphs(spec, s))


def train_graph(spec: StarSpec, data_seed: int, j: int) -> StarGraph:
    """Training graph j (pool index); graphs identical to any dev/test graph are redrawn."""
    return _in_dist_graph(spec, random.Random(f"star-train:{data_seed}:{j}"), eval_keys(spec))


def eval_set_digest(spec: StarSpec, split: str) -> str:
    h = hashlib.sha256()
    for g in eval_graphs(spec, split):
        h.update(g.digest().encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Generalization tests (outside the ILM protocol)
# ---------------------------------------------------------------------------
def generalization_graphs(spec: StarSpec, variant: str, n: int = 1000, extra_len: int = 3) -> List[StarGraph]:
    """GENERALIZATION TEST sets; never mixed into dev/test numbers.

    * ``gen_relabeled``   test graph i under a random label permutation (holdout labels permuted only
                          among themselves). Same topology/query => measures label-invariance on paired
                          items; in-distribution when ``role_holdout`` is empty.
    * ``gen_role_holdout`` in-distribution topology but source and goal relabeled to labels from
                          ``spec.role_holdout`` that never fill those roles in training (OOD binding).
    * ``gen_longer_path`` answer path longer than any training path: answer length (nodes) drawn from
                          [max_path_nodes+1, max_path_nodes+extra_len]; other arms in-distribution.
                          Requires ActionSpace/max_len capacity for the longer target.
    """
    if variant not in GEN_VARIANTS:
        raise ValueError(variant)
    out: List[StarGraph] = []
    test = eval_graphs(spec, "test")
    keys = eval_keys(spec)
    hold = list(spec.role_holdout)
    free = [x for x in range(spec.n_nodes) if x not in spec.role_holdout]
    for i in range(n):
        rng = random.Random(f"star-gen:{spec.eval_seed}:{variant}:{i}")
        if variant == "gen_relabeled":
            g = test[i % len(test)]
            a, b = free[:], hold[:]
            rng.shuffle(a)
            rng.shuffle(b)
            perm = dict(zip(free + hold, a + b))
            out.append(g.relabel(perm))
        elif variant == "gen_role_holdout":
            if len(hold) < 2:
                raise ValueError("gen_role_holdout needs spec.role_holdout with >= 2 labels")
            g = sample_graph(spec, rng)
            h1, h2 = rng.sample(hold, 2)
            g = g.relabel({g.source: h1, h1: g.source})   # transposition source <-> h1
            g = g.relabel({g.goal: h2, h2: g.goal})       # then goal <-> h2 (source stays h1)
            out.append(g)
        else:
            lo = spec.max_path_nodes + 1
            for _ in range(10000):
                try:
                    g = sample_graph(spec, rng, answer_len=rng.randint(lo, lo + extra_len - 1))
                except ValueError:
                    continue  # label budget exceeded for this draw; resample other arms
                if _respects_holdout(spec, g) and g.key not in keys:
                    break
            else:
                raise RuntimeError("longer-path variant infeasible for this label budget")
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def prompt_tokens(spec: StarSpec, g: StarGraph) -> List[int]:
    if spec.prompt_format == "ilm":
        return [x for e in g.edges for x in e] + [g.source, g.goal]
    out: List[int] = []
    for k, (u, v) in enumerate(g.edges):
        if k:
            out.append(spec.EDGE_SEP)
        out += [u, v]
    return out + [spec.QUERY, g.source, g.goal]


def path_to_tokens(spec: StarSpec, path: Sequence[int]) -> List[int]:
    if spec.target_format == "nodes":
        return list(path)
    return [x for i in range(len(path) - 1) for x in (path[i], path[i + 1])]


def target_tokens(spec: StarSpec, g: StarGraph, reverse_target: bool = False) -> List[int]:
    t = path_to_tokens(spec, g.path)
    return t[::-1] if reverse_target else t


def canonical_output(tokens: Sequence[int], reverse_target: bool = False) -> List[int]:
    """Final canvas -> forward-order serialization (undo the reverse-AR target reversal)."""
    return list(tokens)[::-1] if reverse_target else list(tokens)


def tokens_to_path(spec: StarSpec, tokens: Sequence[int]) -> Optional[List[int]]:
    """Forward-order output tokens -> node path, or None if malformed (non-node token, odd length,
    or edge pairs that do not chain)."""
    toks = list(tokens)
    if not toks or any(not 0 <= t < spec.n_nodes for t in toks):
        return None
    if spec.target_format == "nodes":
        return toks
    if len(toks) % 2:
        return None
    pairs = [(toks[i], toks[i + 1]) for i in range(0, len(toks), 2)]
    if any(pairs[i][1] != pairs[i + 1][0] for i in range(len(pairs) - 1)):
        return None
    return [pairs[0][0]] + [v for _, v in pairs]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def is_valid_path(g: StarGraph, path: Optional[Sequence[int]]) -> bool:
    """Starts at source, ends at goal, every consecutive pair is a graph edge."""
    if not path or path[0] != g.source or path[-1] != g.goal:
        return False
    E = set(g.edges)
    return all((path[i], path[i + 1]) in E for i in range(len(path) - 1))


def token_accuracy(pred: Sequence[int], target: Sequence[int]) -> float:
    """Position-aligned accuracy over target positions (ILM ``HammingAccuracy`` analogue; missing
    positions count as wrong, extra predicted tokens are ignored)."""
    if not target:
        return float(len(pred) == 0)
    return sum(1 for i, t in enumerate(target) if i < len(pred) and pred[i] == t) / len(target)


def score(spec: StarSpec, g: StarGraph, final_canvas: Sequence[int], reverse_target: bool = False,
          stopped: bool = True) -> Dict:
    """Per-example metrics for a finished rollout's canvas. ``stopped=False`` (no STOP within the
    action budget) makes ``exact`` and ``valid_path`` false; the other fields still describe the canvas."""
    out = canonical_output(final_canvas, reverse_target)
    tgt = path_to_tokens(spec, g.path)
    path = tokens_to_path(spec, out)
    valid = stopped and is_valid_path(g, path)
    return {
        "exact": stopped and out == tgt,          # whole-path exact match (ILM "sequence acc.")
        "stopped": stopped,
        "token_acc": token_accuracy(out, tgt),    # ILM "token acc." analogue
        "valid_path": valid,
        "well_formed": path is not None,
        "endpoints": bool(path) and path[0] == g.source and path[-1] == g.goal,
        "first_step": bool(path) and len(path) > 1 and len(g.path) > 1 and path[1] == g.path[1],
        "len_pred": len(out),
        "len_target": len(tgt),
    }


def aggregate(rows: Sequence[Dict]) -> Dict:
    n = len(rows)
    keys = ("exact", "token_acc", "valid_path", "well_formed", "endpoints", "first_step", "stopped")
    return {"n": n, **{k: (sum(float(r[k]) for r in rows) / n if n else None) for k in keys}}


# ---------------------------------------------------------------------------
# Trajectories and the episode source
# ---------------------------------------------------------------------------
def rtl_reverse_ops(target: Sequence[int]) -> List[Tuple]:
    """Build ``target`` right-to-left: ins(t[-1]); then (goto 0, ins t[k]) for k = n-2..0; stop.

    After an insert at boundary 0 the cursor sits at 1, so every earlier token needs one goto(0)
    (backward move; legal under ``backward_only``). The final canvas equals ``target``.
    """
    ops: List[Tuple] = []
    for k, t in enumerate(reversed(list(target))):
        if k:
            ops.append(("goto", 0))
        ops.append(("ins", t))
    ops.append(("stop",))
    return ops


KIND_ALIASES = {"reverse": "rtl_reverse", "ar": "ltr"}


def mix_reverse_target(mix: Sequence[Dict]) -> bool:
    """Whether evaluation must reverse the canvas (reverse-AR baseline). All entries must agree."""
    flags = {bool(m.get("reverse_target", False)) for m in mix}
    if len(flags) > 1:
        raise ValueError("mix entries disagree on reverse_target; a model has one output order")
    return flags.pop() if flags else False


def episode_for_graph(space: ActionSpace, spec: StarSpec, g: StarGraph, m: Dict, rng: random.Random) -> Tuple[Episode, str]:
    kind = KIND_ALIASES.get(m["kind"], m["kind"])
    rev = bool(m.get("reverse_target", False))
    if rev and kind != "ltr":
        raise ValueError("reverse_target is only defined for kind 'ltr' (reverse-AR baseline)")
    prompt = prompt_tokens(spec, g)
    tgt = target_tokens(spec, g, rev)
    if kind == "ltr":
        ops = ltr_ops(tgt)
    elif kind == "rtl_reverse":
        ops = rtl_reverse_ops(tgt)
    elif kind == "restoration":
        pol = ObfuscationPolicy(**m.get("policy", {}))
        ops = restoration_ops(tgt, rng, pol, uniform_junk(spec.n_nodes))
    else:
        raise ValueError(f"unknown stargraph kind {m['kind']!r}")
    label = m.get("name") or (kind + ("_revtarget" if rev else ""))
    return build_episode(space, prompt, [], tgt, ops), label


class StarGraphSource:
    """Episodes for the star-graph task; interface mirrors ``data.ProgressionSource``.

    ``split='train'``: graph = training pool item ``idx % n_train`` (or fresh graph idx if n_train is
    None). ``split in ('dev','test')``: graph = fixed eval graph ``idx % n_split``. Mix choice and
    trajectory randomness come from ``random.Random(f"{data_seed}:{split}:{idx}")``.
    """

    def __init__(self, space: ActionSpace, spec: StarSpec, mix: List[Dict], data_seed: int, split: str = "train"):
        if space.n_content != spec.n_content:
            raise ValueError(f"space.n_content={space.n_content} != spec.n_content={spec.n_content}")
        if split not in ("train", "dev", "test"):
            raise ValueError(split)
        self.space, self.spec, self.mix, self.seed, self.split = space, spec, mix, data_seed, split
        self.reverse_target = mix_reverse_target(mix)
        tot = sum(m["weight"] for m in mix)
        self.cum, acc = [], 0.0
        for m in mix:
            acc += m["weight"] / tot
            self.cum.append(acc)

    def graph(self, idx: int) -> StarGraph:
        if self.split == "train":
            j = idx % self.spec.n_train if self.spec.n_train else idx
            return train_graph(self.spec, self.seed, j)
        gs = eval_graphs(self.spec, self.split)
        return gs[idx % len(gs)]

    def item(self, idx: int) -> Tuple[StarGraph, Episode, str]:
        rng = random.Random(f"{self.seed}:{self.split}:{idx}")
        r = rng.random()
        m = next(mm for mm, c in zip(self.mix, self.cum) if r <= c + 1e-12)
        g = self.graph(idx)
        ep, kind = episode_for_graph(self.space, self.spec, g, m, rng)
        return g, ep, kind

    def episode(self, idx: int) -> Tuple[Episode, str]:
        _, ep, kind = self.item(idx)
        return ep, kind
