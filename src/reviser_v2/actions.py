"""Action vocabulary layout.

Token IDs ``[0, n_content)`` are content tokens; emitting one after the edit marker means
Insert(token) at the cursor. Special tokens follow the content block:

* ``SEP``    separates protected instruction/prompt tokens from a supplied draft.
* ``EDIT``   marks the end of conditioning; every later position is a model action.
* ``DELETE`` removes the token immediately LEFT of the cursor (backspace semantics).
* ``STOP``   ends the episode.
* move tokens, whose meaning depends on ``move_encoding``:

  - ``pow2``     relative moves +-2^k, k < n_move (historical Reviser encoding for n_move=10).
  - ``dense``    relative moves +-d, 1 <= d <= n_move.
  - ``absolute`` MOVE_TO(j), 0 <= j <= n_move - 1: put the cursor at editable boundary j
                 (0 = before the first editable token, len = after the last).

``backward_only`` removes forward relative moves from the vocabulary (absolute moves remain in the
vocabulary but only targets left of the cursor are legal).

The layout with ``move_encoding='pow2', n_move=10`` and GPT-2 content matches the historical
release order for DELETE and moves only in spirit, not in exact IDs; use
``historical_gpt2_layout`` helpers in ``compat.py`` to map historical checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

MOVE_ENCODINGS = ("pow2", "dense", "absolute")


@dataclass(frozen=True)
class ActionSpace:
    n_content: int
    move_encoding: str = "pow2"
    n_move: int = 10
    backward_only: bool = False
    allow_delete: bool = True

    def __post_init__(self):
        if self.move_encoding not in MOVE_ENCODINGS:
            raise ValueError(f"unknown move_encoding {self.move_encoding!r}")
        if self.n_move < 1:
            raise ValueError("n_move must be >= 1")

    # ---- fixed specials -------------------------------------------------
    @property
    def SEP(self) -> int:
        return self.n_content

    @property
    def EDIT(self) -> int:
        return self.n_content + 1

    @property
    def DELETE(self) -> int:
        return self.n_content + 2

    @property
    def STOP(self) -> int:
        return self.n_content + 3

    @property
    def move_start(self) -> int:
        return self.n_content + 4

    # ---- moves ------------------------------------------------------------
    @property
    def move_values(self) -> Tuple[int, ...]:
        """Semantic value of each move token, in ID order.

        Relative encodings: signed displacement. Absolute: target boundary index.
        """
        if self.move_encoding == "absolute":
            return tuple(range(self.n_move))
        mags = [2 ** k for k in range(self.n_move)] if self.move_encoding == "pow2" else list(range(1, self.n_move + 1))
        vals: List[int] = []
        for m in mags:
            if not self.backward_only:
                vals.append(m)
            vals.append(-m)
        return tuple(vals)

    @property
    def n_moves_tokens(self) -> int:
        return len(self.move_values)

    @property
    def vocab_size(self) -> int:
        return self.move_start + self.n_moves_tokens

    def is_content(self, t: int) -> bool:
        return 0 <= t < self.n_content

    def is_move(self, t: int) -> bool:
        return self.move_start <= t < self.vocab_size

    def move_value(self, t: int) -> int:
        return self.move_values[t - self.move_start]

    def move_token(self, value: int) -> int:
        try:
            return self.move_start + self.move_values.index(value)
        except ValueError:
            raise ValueError(f"move value {value} not representable in {self.describe()}") from None

    @property
    def max_relative_step(self) -> int:
        if self.move_encoding == "absolute":
            raise ValueError("absolute encoding has no relative step")
        return max(abs(v) for v in self.move_values)

    @property
    def max_position(self) -> int:
        """Largest editable boundary index addressable by a single absolute move (absolute only)."""
        return self.n_move - 1 if self.move_encoding == "absolute" else -1

    def compile_goto(self, cur: int, dst: int) -> List[int]:
        """Move tokens taking the cursor from boundary ``cur`` to ``dst``.

        Relative encodings decompose greedily (largest step first). All intermediate positions lie
        between ``cur`` and ``dst`` so they stay in bounds whenever the endpoints do.
        """
        if cur == dst:
            return []
        if self.move_encoding == "absolute":
            if dst > self.max_position:
                raise ValueError(f"target {dst} exceeds absolute range {self.max_position}")
            if self.backward_only and dst > cur:
                raise ValueError("forward move requested under backward_only")
            return [self.move_token(dst)]
        delta = dst - cur
        if self.backward_only and delta > 0:
            raise ValueError("forward move requested under backward_only")
        sign = 1 if delta > 0 else -1
        rem = abs(delta)
        mags = sorted({abs(v) for v in self.move_values}, reverse=True)
        out: List[int] = []
        for m in mags:
            while rem >= m:
                out.append(self.move_token(sign * m))
                rem -= m
        assert rem == 0
        return out

    def describe(self) -> str:
        bo = "+backonly" if self.backward_only else ""
        nd = "" if self.allow_delete else "+nodelete"
        return f"{self.move_encoding}{self.n_move}{bo}{nd}/content{self.n_content}"

    def token_name(self, t: int) -> str:
        if self.is_content(t):
            return f"i{t}"
        if t == self.SEP:
            return "<SEP>"
        if t == self.EDIT:
            return "<EDIT>"
        if t == self.DELETE:
            return "<DEL>"
        if t == self.STOP:
            return "<STOP>"
        if self.is_move(t):
            v = self.move_value(t)
            return f"<TO{v}>" if self.move_encoding == "absolute" else f"<M{v:+d}>"
        return f"<?{t}>"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ActionSpace":
        return cls(**d)
