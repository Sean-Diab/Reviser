"""DAgger data pieces: July teacher trajectories, model excursions with provenance, target re-fitting, oracle labels.

One DAgger example (online, from the current model):
  1. a training text (prefix 35 + continuation) and a July run-and-jump trajectory for it (July dials);
  2. a teacher state s_t = the trajectory cut after t actions (t >= 35; holes, cursor anywhere);
  3. an excursion: the model samples k actions from s_t (eval decoding: T 0.9, top-k 50, legality mask); a sampled
     EOR ends the excursion and is not executed;
  4. a target for labelling (`refit_target`): the reference text, except that model-inserted tokens the model itself
     sampled with probability >= accept_p are accepted into it (they substitute, in order, for the first reference
     tokens of their mismatch region; extra ones are insertions). accept_p > 1 accepts nothing (the naive target:
     every model token that is not in the reference is junk);
  5. labels: the hunk oracle (LCS alignment by default) against that target at each excursion state and at each
     recovery state; recovery executes oracle actions (sampled from its normalized weights) until the canvas has no
     junk or `max_recovery` steps. `label_clean`: label states without junk too ("oracle") or not ("none": only
     states with junk get labels, i.e. the labels teach removing junk and getting to it).
The teacher part (positions 35..t) keeps July's own hard labels, so every DAgger example is also a July example.

Teacher with errors (`errors` in _worker_init; the AR + RL arm F teaching stage on "July-with-delete" data): the July
order of the text is decorated with the data agent's wrong-token-then-DELETE events (scripts/humanorder/variants.py
inject_errors / errors_fit, e.g. configs/humanorder/overnight/jd_n05_imm_m.json "errors"); the error RNG is seeded per
item (err_seed * 1_000_003 + shard * 100_003 + row), so the recipe matches the data agent's shards but the draws are
not byte-identical to them. Masked wrong inserts get no hard label (july_labeled).
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "humanorder"))
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "code" / "july_impl"))

import hunk_oracle as ho  # noqa: E402

JULY_DIALS = {"lambda_cont": 0.82, "jump_alpha": 1.5, "p_skip": 0.3, "skip_q": 0.85, "p_far": 0.9}
MAX_TRAJ = 350
PREFIX = 35


# ------------------------------------------------------------------------------------------------ July teacher
_W: dict = {}


def _worker_init(texts_dir: Optional[str], seed: int, dials: Optional[dict] = None, errors: Optional[dict] = None,
                 err_seed: int = 7):
    from july_texts import JulyTexts
    from fit_q_dials import make_policy
    _W["jt"] = JulyTexts(texts_dir) if texts_dir else JulyTexts()
    _W["make_policy"] = make_policy
    _W["seed"] = seed
    _W["errors"] = None
    if errors:
        import variants
        _W["V"] = variants
        _W["errors"] = variants.ErrorSpec.from_dict(errors)
        _W["err_seed"] = err_seed
    _W["dials"] = dict(dials) if dials else dict(JULY_DIALS)   # e.g. Q's dials (data/humanorder/q_dials_v1.json)


def july_item(key: Tuple[int, int]) -> dict:
    """(shard, row) -> text + July trajectory (RNG seeded by the key, so items are reproducible)."""
    k, i = key
    prefix, cont = _W["jt"].text(k, i)
    pol = _W["make_policy"](_W["dials"], random.Random(_W["seed"] * 1_000_003 + k * 100_003 + i))
    if _W.get("errors") is not None:
        V = _W["V"]
        for _ in range(51):
            order = V.runjump_order(pol, prefix, len(cont))
            if order is not None and V.clean_len(len(prefix), order) <= MAX_TRAJ:
                break
        else:
            return {"key": key, "skip": True}
        rng = random.Random(_W["err_seed"] * 1_000_003 + k * 100_003 + i)
        acts, mask = V.errors_fit(prefix, cont, order, _W["errors"], rng, MAX_TRAJ, 20)
        return {"key": key, "prefix": prefix, "cont": cont, "actions": acts, "mask": mask}
    for _ in range(51):
        acts = pol.generate(prefix, cont, 10 ** 6)
        if acts is not None and len(acts) <= MAX_TRAJ:
            break
    else:
        return {"key": key, "skip": True}
    return {"key": key, "prefix": prefix, "cont": cont, "actions": acts}


def text_keys(shards: List[int], texts_dir: Optional[str] = None) -> List[Tuple[int, int]]:
    from july_texts import JulyTexts
    jt = JulyTexts(texts_dir) if texts_dir else JulyTexts()
    return [(k, i) for k in shards for i in range(jt.shard_sizes[k])]


# ------------------------------------------------------------------------------------------------ excursion state
@dataclass
class Excursion:
    """Executor state + provenance: prov[i] is None for a teacher token, else the sampling probability of the
    model-inserted token at canvas index i."""

    target: List[int]
    teacher: List[int]                 # July actions (incl. EOR)
    t: int                             # cut: teacher actions executed
    state: ho.State
    prov: List[Optional[float]]
    acts: List[int]                    # executed actions so far (teacher part + excursion + recovery)
    exc_actions: List[int] = field(default_factory=list)
    exc_probs: List[float] = field(default_factory=list)
    exc_eor: bool = False
    k: int = 1

    @classmethod
    def from_teacher(cls, target, teacher, t, k):
        st, prov = ho.State(), []
        for a in teacher[:t]:
            _step_prov(st, prov, a, None)
        return cls(target=list(target), teacher=list(teacher), t=t, state=st, prov=prov, acts=list(teacher[:t]), k=k)

    def model_step(self, a: int, p: float):
        self.exc_actions.append(a)
        self.exc_probs.append(p)
        if a == ho.EOR:
            self.exc_eor = True
            return
        _step_prov(self.state, self.prov, a, p)
        self.acts.append(a)

    @property
    def done(self) -> bool:
        return self.exc_eor or len(self.exc_actions) >= self.k


def _step_prov(st: ho.State, prov: list, a: int, p):
    if a == ho.DELETE:
        if st.cursor > 0:
            prov.pop(st.cursor - 1)
    elif ho.is_insert(a):
        prov.insert(st.cursor, p)
    st.step(a)


def refit_target(canvas: List[int], prov: List[Optional[float]], target: List[int], accept_p: float,
                 align: str = "lcs") -> List[int]:
    """Reference with accepted model tokens (prov >= accept_p) substituted into their mismatch regions."""
    out: List[int] = []
    tpos = 0
    for h in ho.compute_hunks(canvas, target, align):
        cl, cr = h.canvas_bounds_int()
        tl, tr = h.target_bounds_int()
        out.extend(target[tpos:tl])
        acc = [canvas[i] for i in range(cl, cr) if prov[i] is not None and prov[i] >= accept_p]
        sub = target[tl:tr]
        out.extend(acc + sub[len(acc):])
        tpos = tr
    out.extend(target[tpos:])
    return out


@dataclass
class Labeled:
    seq: List[int]                         # [EOI] + executed actions
    hard: Dict[int, int]                   # input position -> action id (July teacher labels)
    soft: Dict[int, Tuple[List[int], List[float]]]   # input position -> normalized oracle distribution
    stats: Dict[str, float]


def label_excursion(ex: Excursion, rng: random.Random, accept_p: float, label_clean: str, max_recovery: int,
                    cfg: ho.OracleConfig) -> Labeled:
    """Teacher hard labels on positions 35..t; oracle labels on the excursion + recovery states."""
    hard = {j: ex.teacher[j] for j in range(PREFIX, ex.t + 1)}     # position j predicts action j at state acts[:j]
    tgt2 = refit_target(ex.state.canvas, ex.prov, ex.target, accept_p, cfg.align)
    # replay the excursion states (positions t+1 .. t+len(executed excursion actions))
    st = ho.State()
    for a in ex.acts[:ex.t]:
        st.step(a)
    states: List[ho.State] = []
    for a in ex.acts[ex.t:]:
        st.step(a)
        states.append(st.copy())
    soft: Dict[int, Tuple[List[int], List[float]]] = {}
    n_junk_states = 0

    def lab(s: ho.State, j: int, force: bool = False) -> int:
        """Label the state at input position j (> t). Returns its junk count."""
        junk = ho.junk_count(s.canvas, tgt2, cfg.align)[0]
        if j > ex.t and (force or junk > 0 or label_clean == "oracle"):
            out = ho.oracle_labels(s.canvas, s.cursor, tgt2, cfg).normalized()
            soft[j] = (out.action_ids, out.weights)
        return junk

    acts = list(ex.acts)
    for i, s in enumerate(states):
        n_junk_states += int(lab(s, ex.t + 1 + i) > 0)
    s = (states[-1] if states else _replay(acts)).copy()
    n_rec = 0
    if ex.exc_eor:
        # the model stopped here: always label this state (EOR only if the canvas equals the target); no recovery
        lab(s, len(acts), force=True)
    else:
        junk = ho.junk_count(s.canvas, tgt2, cfg.align)[0]
        while junk > 0 and n_rec < max_recovery:
            out = ho.oracle_labels(s.canvas, s.cursor, tgt2, cfg).normalized()
            a = _sample(out, rng)
            if a == ho.EOR:
                break
            s.step(a)
            acts.append(a)
            n_rec += 1
            junk = lab(s, len(acts))
            n_junk_states += int(junk > 0)
    n_model_ins = sum(1 for a in ex.exc_actions if ho.is_insert(a))
    n_rej = sum(1 for a, p in zip(ex.exc_actions, ex.exc_probs) if ho.is_insert(a) and p < accept_p)
    del_mass = sum(w for ids, ws in soft.values() for i, w in zip(ids, ws) if i == ho.DELETE)
    stats = {"k": len(ex.exc_actions), "exc_eor": float(ex.exc_eor), "model_inserts": n_model_ins,
             "model_inserts_below_accept": n_rej, "recovery_steps": n_rec, "soft_labels": len(soft),
             "junk_states": n_junk_states, "delete_label_mass": del_mass,
             "exc_moves": sum(1 for a in ex.exc_actions if ho.is_move(a)),
             "exc_deletes": sum(1 for a in ex.exc_actions if a == ho.DELETE)}
    return Labeled(seq=[ho.EOI] + acts, hard=hard, soft=soft, stats=stats)


def _replay(acts) -> ho.State:
    s = ho.State()
    for a in acts:
        s.step(a)
    return s


def _sample(out: ho.OracleOutput, rng: random.Random) -> int:
    r, acc = rng.random(), 0.0
    for t, w in zip(out.action_ids, out.weights):
        acc += w
        if r < acc:
            return t
    return out.action_ids[-1]


def july_labeled(item: dict) -> Labeled:
    acts = item["actions"]
    mask = item.get("mask") or [0] * len(acts)
    return Labeled(seq=[ho.EOI] + acts[:-1], hard={j: acts[j] for j in range(PREFIX, len(acts)) if not mask[j]}, soft={},
                   stats={})
