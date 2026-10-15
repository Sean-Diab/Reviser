"""Turn a (before_ids, after_ids) token pair into a replay-validated edit script.

Uses the project's own trajectory API (``trajectories.repair_ops`` / ``compile_ops`` /
``validate_actions``); nothing here re-implements the executor. The before and after texts must be
tokenized separately and completely: a character-level diff spliced into token ids would get BPE
boundaries wrong (e.g. " like" -> " liked" is a different single token, not " like" + "d").

The script's hunk order (right-to-left or left-to-right) is a declared synthetic choice. Before/after
pairs are human revisions; the action order is NOT recovered human keystroke order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from ..actions import ActionSpace
from ..trajectories import compile_ops, diff_hunks, repair_ops, validate_actions

GPT2_EOT = 50256


def gpt2_action_space() -> ActionSpace:
    """Action space used for all real-data validation: GPT-2 content ids + pow2 moves (+-2^k, k<10)."""
    return ActionSpace(n_content=50257, move_encoding="pow2", n_move=10)


@dataclass
class EditStats:
    n_before: int
    n_after: int
    n_hunks: int
    n_del: int          # draft tokens deleted
    n_ins: int          # target tokens inserted
    n_moves: int        # move action tokens in the compiled script
    n_actions: int      # total compiled actions including the final STOP
    unchanged: bool

    def to_dict(self) -> Dict:
        return asdict(self)


class ReplayError(ValueError):
    pass


def edit_script(before: Sequence[int], after: Sequence[int], space: ActionSpace, order: str = "rtl",
                check: bool = True) -> Tuple[List[int], EditStats]:
    """Compile and (by default) replay-validate the edit script turning ``before`` into ``after``.

    The draft cursor starts at the end of ``before`` (the ``compile_ops`` / ``replay`` default).
    Raises ``ReplayError`` if the replayed canvas differs from ``after`` or any action is illegal.
    """
    before, after = list(before), list(after)
    for t in before + after:
        if not space.is_content(t):
            raise ReplayError(f"token {t} outside content range [0,{space.n_content})")
    if space.n_content == 50257 and (GPT2_EOT in before or GPT2_EOT in after):
        raise ReplayError("GPT-2 end-of-text id 50256 inside text")
    ops = repair_ops(before, after, order)
    acts = compile_ops(space, ops, len(before))
    if check:
        ok, why = validate_actions(space, acts, after, before)
        if not ok:
            raise ReplayError(why)
    hunks = diff_hunks(before, after)
    stats = EditStats(
        n_before=len(before), n_after=len(after), n_hunks=len(hunks),
        n_del=sum(i2 - i1 for i1, i2, _, _ in hunks), n_ins=sum(j2 - j1 for _, _, j1, j2 in hunks),
        n_moves=sum(1 for a in acts if space.is_move(a)), n_actions=len(acts), unchanged=before == after,
    )
    assert stats.n_actions == stats.n_del + stats.n_ins + stats.n_moves + 1
    return acts, stats


def check_pair(before: Sequence[int], after: Sequence[int], space: ActionSpace) -> Dict:
    """Validate both synthetic orders; return stats for ``rtl`` plus the ``ltr`` action count."""
    out: Dict = {}
    try:
        _, st = edit_script(before, after, space, "rtl")
        out.update(st.to_dict())
        out["ok_rtl"] = True
    except Exception as e:  # recorded, never silently dropped
        out["ok_rtl"] = False
        out["err_rtl"] = f"{type(e).__name__}: {e}"
    try:
        acts_l, _ = edit_script(before, after, space, "ltr")
        out["ok_ltr"] = True
        out["n_actions_ltr"] = len(acts_l)
    except Exception as e:
        out["ok_ltr"] = False
        out["err_ltr"] = f"{type(e).__name__}: {e}"
    return out
