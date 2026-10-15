"""Deterministic executor for edit actions on the editable canvas.

The canvas holds only the editable region (the supplied draft plus anything inserted). Protected
instruction/prompt tokens live in the history, never on the canvas, so no action can modify them.
Every inserted token is a distinct *instance* (identified by the action index that created it or by
its draft index), which is what the doomed-insertion mask needs: two equal token IDs can have
different fates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .actions import ActionSpace


class IllegalAction(ValueError):
    pass


@dataclass
class Canvas:
    space: ActionSpace
    tokens: List[int] = field(default_factory=list)
    inst: List[int] = field(default_factory=list)  # instance id per canvas token
    cursor: int = 0
    stopped: bool = False
    n_actions: int = 0
    next_iid: int = 0

    @classmethod
    def from_draft(cls, space: ActionSpace, draft: Sequence[int], cursor: Optional[int] = None) -> "Canvas":
        """Seed a supplied draft. Draft instances get ids -1, -2, ... (negative = pre-existing)."""
        c = cls(space=space, tokens=list(draft), inst=[-(i + 1) for i in range(len(draft))])
        c.cursor = len(draft) if cursor is None else cursor
        if not 0 <= c.cursor <= len(draft):
            raise ValueError("initial cursor out of bounds")
        return c

    def copy(self) -> "Canvas":
        return Canvas(self.space, list(self.tokens), list(self.inst), self.cursor, self.stopped, self.n_actions, self.next_iid)

    def __len__(self) -> int:
        return len(self.tokens)

    # ------------------------------------------------------------------
    def is_legal(self, a: int) -> bool:
        s = self.space
        if self.stopped:
            return False
        if s.is_content(a):
            return True
        if a == s.STOP:
            return True
        if a == s.DELETE:
            return s.allow_delete and self.cursor > 0
        if s.is_move(a):
            v = s.move_value(a)
            if s.move_encoding == "absolute":
                if v > len(self.tokens) or v == self.cursor:
                    return False
                return (v < self.cursor) if s.backward_only else True
            dst = self.cursor + v
            return 0 <= dst <= len(self.tokens)
        return False  # SEP / EDIT are never legal actions

    def legal_special_mask(self) -> List[bool]:
        """Legality of ids [n_content, vocab_size) (content inserts are always legal)."""
        s = self.space
        out = [False] * (s.vocab_size - s.n_content)
        if self.stopped:
            return out
        base = s.n_content
        out[s.STOP - base] = True
        out[s.DELETE - base] = s.allow_delete and self.cursor > 0
        L, c = len(self.tokens), self.cursor
        for i, v in enumerate(s.move_values):
            if s.move_encoding == "absolute":
                ok = v <= L and v != c and (v < c if s.backward_only else True)
            else:
                ok = 0 <= c + v <= L
            out[s.move_start - base + i] = ok
        return out

    def legal_mask(self) -> List[bool]:
        s = self.space
        n = s.vocab_size
        mask = [False] * n
        if self.stopped:
            return mask
        for t in range(s.n_content):
            mask[t] = True
        mask[s.STOP] = True
        mask[s.DELETE] = s.allow_delete and self.cursor > 0
        L, c = len(self.tokens), self.cursor
        for i, v in enumerate(s.move_values):
            t = s.move_start + i
            if s.move_encoding == "absolute":
                mask[t] = v <= L and v != c and (v < c if s.backward_only else True)
            else:
                mask[t] = 0 <= c + v <= L
        return mask

    def apply(self, a: int) -> Tuple[str, Optional[int]]:
        """Execute action ``a``. Returns (kind, affected instance id)."""
        if not self.is_legal(a):
            raise IllegalAction(f"illegal action {self.space.token_name(a)} at cursor {self.cursor} len {len(self.tokens)}")
        s = self.space
        self.n_actions += 1
        if s.is_content(a):
            iid = self.next_iid
            self.next_iid += 1
            self.tokens.insert(self.cursor, a)
            self.inst.insert(self.cursor, iid)
            self.cursor += 1
            return "ins", iid
        if a == s.DELETE:
            self.cursor -= 1
            self.tokens.pop(self.cursor)
            iid = self.inst.pop(self.cursor)
            return "del", iid
        if a == s.STOP:
            self.stopped = True
            return "stop", None
        v = s.move_value(a)
        self.cursor = v if s.move_encoding == "absolute" else self.cursor + v
        return "move", None


def replay(space: ActionSpace, actions: Sequence[int], draft: Sequence[int] = (), cursor: Optional[int] = None) -> Canvas:
    """Execute a whole action list; raises IllegalAction on the first illegal action."""
    c = Canvas.from_draft(space, draft, cursor)
    for a in actions:
        if c.stopped:
            raise IllegalAction("action after STOP")
        c.apply(a)
    return c


def doomed_insert_positions(space: ActionSpace, actions: Sequence[int], draft: Sequence[int] = (), cursor: Optional[int] = None) -> List[int]:
    """Indices (into ``actions``) of Insert actions whose token instance is later deleted.

    Uses token-instance identity, not token values. Computed on the complete trajectory, so a later
    truncation cannot hide the deletion (callers must compute this before cropping).
    """
    c = Canvas.from_draft(space, draft, cursor)
    created_at = {}
    doomed = []
    for idx, a in enumerate(actions):
        kind, iid = c.apply(a)
        if kind == "ins":
            created_at[iid] = idx
        elif kind == "del" and iid is not None and iid >= 0:
            doomed.append(created_at[iid])
    return sorted(doomed)
