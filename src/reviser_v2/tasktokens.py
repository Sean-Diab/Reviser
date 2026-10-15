"""Reviser-Edit task tokens (plan step 9, notes/reviser-edit-data-plan-2026-10-06.md).

Five task tokens, one per edit category: grammar, fluency, clarity, coherence, polish. They are INPUT-ONLY ids placed
after the action vocabulary: id = space.vocab_size + index (GPT-2 action space: 50281..50285). The model's embedding
matrix gets 5 extra rows, but its output projection stays over the action vocabulary (``ModelConfig.n_out``), so the
model can never emit a task token and every decoder / legality mask keeps its old shape.

Prompt layout of an edit example (read-only conditioning, trajectories.conditioning_ids):
    [TASK_c] + instruction_ids  SEP  draft  EDIT  actions...
``instruction_ids`` is the record's own instruction (IteraTeR / CoEdIT) or empty / a sampled category instruction for
sources without one (tasks/editmix.py). Restoration keeps the D4 prefix prompt and gets no task token.

Evaluation: ``apply_task_tokens`` prepends the token mapped from each record's intent (``intent_category``), so the
September eval records are reused unchanged apart from that one id.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

TASKS = ("grammar", "fluency", "clarity", "coherence", "polish")
N_TASKS = len(TASKS)

# Training-data category of each real-data label (realdata.sources / CoEdIT intents). ``None`` = not used for training
# (content updates are not an editing-quality category). Low-confidence IteraTeR clarity/fluency go to polish
# (plan section 2); that rule lives in the builder (scripts/editdata/build_mix_v2.py), not here.
COEDIT_CATEGORY = {"gec": "grammar", "fluency": "fluency", "clarity": "clarity", "coherence": "coherence",
                   "simplification": "clarity", "neutralize": "polish", "paraphrase": "polish"}
ITERATER_CATEGORY = {"fluency": "fluency", "clarity": "clarity", "coherence": "coherence", "style": "polish",
                     "meaning-changed": None, "others": None}

# Eval-time mapping (every eval record needs some token). Content updates / other edits -> polish (the generic
# category); synthetic and realistic-error sets -> grammar ("Fix the errors in this text").
EVAL_CATEGORY = dict(COEDIT_CATEGORY)
EVAL_CATEGORY.update({k: (v or "polish") for k, v in ITERATER_CATEGORY.items()})

GPT2_NAME_IDS = {  # GPT-2 ids of " " + name (checked in tests/test_editmix.py against the tokenizer)
    "grammar": [23491], "fluency": [6562, 1387], "clarity": [16287], "coherence": [763, 23545], "polish": [25245],
}


def task_id(space_vocab_size: int, name: str) -> int:
    return space_vocab_size + TASKS.index(name)


def eval_category(intent: str, data: str = "") -> Optional[str]:
    """Category for an eval record; None = no task token (prefix-prompt sets such as synth_forced)."""
    if data == "synth_forced":
        return None
    if data in ("synth", "synth_real") or intent.startswith("synth"):
        return "grammar"
    if intent in EVAL_CATEGORY:
        return EVAL_CATEGORY[intent]
    from .evalsets import SYNTH_REAL_INTENTS
    if intent in SYNTH_REAL_INTENTS:
        return "grammar"
    if data.startswith("infill"):
        return None
    return "polish"


def apply_task_tokens(recs: Iterable[Dict], space_vocab_size: int, data: str = "") -> List[Dict]:
    """Copies of ``recs`` with [TASK(intent)] prepended to ``instruction_ids`` (records without a category unchanged).
    Adds ``task`` (category name or None) to each copy."""
    out = []
    for r in recs:
        c = eval_category(r.get("intent", ""), data)
        r2 = dict(r)
        r2["task"] = c
        if c is not None:
            r2["instruction_ids"] = [task_id(space_vocab_size, c)] + list(r["instruction_ids"])
        out.append(r2)
    return out


def expand_embedding(sd: Dict, n_new: int, init_rows: Sequence[Sequence[int]], key: str = "embed.weight") -> Dict:
    """Return a copy of state dict ``sd`` whose embedding has ``n_new`` extra rows; row i = mean of the existing rows
    ``init_rows[i]`` (e.g. the GPT-2 word pieces of the category name). Other tensors are shared, not copied."""
    import torch
    assert len(init_rows) == n_new
    w = sd[key]
    new = torch.stack([w[list(ids)].float().mean(0) for ids in init_rows]).to(w.dtype)
    out = dict(sd)
    out[key] = torch.cat([w, new], 0)
    return out


def default_init_rows() -> List[List[int]]:
    return [GPT2_NAME_IDS[t] for t in TASKS]


def ckpt_task_vocab(ckpt) -> Optional[int]:
    """The action-space vocab size (= first task-token id) if checkpoint ``ckpt`` (file or run dir) has the task-token
    rows (model_config.vocab_size - n_out == N_TASKS), else None. Reads the config only (mmap)."""
    import torch
    from pathlib import Path
    p = Path(ckpt)
    if p.is_dir():
        p = p / "ckpt" / "latest.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    mc = ck.get("model_config") or {}
    if isinstance(mc, dict) and mc.get("n_out") and mc.get("vocab_size", 0) - mc["n_out"] == N_TASKS:
        return int(mc["n_out"])
    return None


def resolve_task_tokens(mode: str, ckpt) -> Optional[int]:
    """--task-tokens auto|on|off -> first task id (space vocab size) or None. ``auto`` = on iff the checkpoint has them."""
    if mode == "off" or not ckpt:
        return None
    v = ckpt_task_vocab(ckpt)
    if mode == "on" and v is None:
        raise SystemExit("--task-tokens on but the checkpoint has no task-token rows")
    return v
