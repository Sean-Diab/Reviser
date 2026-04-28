from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class CursorState:
    """
    Minimal Cursor editing environment state:
    - The prompt is immutable and ends with END_OF_INPUT.
    - The editable canvas starts empty.
    - Actions are appended as an autoregressive edit-history sequence.
    """

    prompt_tokens: List[int]
    canvas: List[int]
    cursor_pos: int
    actions: List[int]

    @classmethod
    def from_prompt(cls, prompt_tokens: List[int]) -> "CursorState":
        return cls(prompt_tokens=list(prompt_tokens), canvas=[], cursor_pos=0, actions=[])

    def can_move(self, amt: int) -> bool:
        new_pos = self.cursor_pos + int(amt)
        return 0 <= new_pos <= len(self.canvas)

    def can_delete(self) -> bool:
        return self.cursor_pos > 0

    def execute(self, token_id: int, special_tokens) -> bool:
        """
        Execute one action token. Returns False if invalid.
        `special_tokens` should be the SPECIAL_TOKENS object from `data`.
        """
        if token_id == special_tokens.end_of_response:
            self.actions.append(int(token_id))
            return True

        if token_id == special_tokens.delete:
            if not self.can_delete():
                return False
            self.canvas.pop(self.cursor_pos - 1)
            self.cursor_pos -= 1
            self.actions.append(int(token_id))
            return True

        # Move tokens are represented by special token IDs in the vocab.
        # We import helpers from `data` where needed by callers.
        # Caller should check move validity via masking; still validate here.
        from reviser.data import is_move_token, get_move_amount, is_insert_token

        if is_move_token(token_id):
            amt = int(get_move_amount(token_id))
            if not self.can_move(amt):
                return False
            self.cursor_pos += amt
            self.actions.append(int(token_id))
            return True

        # Insert: any non-control token treated as insert (per Cursor family)
        if is_insert_token(token_id):
            self.canvas.insert(self.cursor_pos, int(token_id))
            self.cursor_pos += 1
            self.actions.append(int(token_id))
            return True

        # Unknown token: still record it, treat as no-op
        self.actions.append(int(token_id))
        return True


def build_canvas_tensors_shifted(
    *,
    prompt_tokens: List[int],
    action_history: List[int],
    device: torch.device,
    max_canvas_len: int,
    special_tokens,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build canvas tensors aligned with "predict-next-token" semantics.

    Sequence fed to model is:
      input_ids = prompt_tokens + action_history

    We want canvas_ids[t] to represent the canvas state BEFORE executing token t+1.
    - For prompt positions before END_OF_INPUT: canvas is []
    - At END_OF_INPUT position: canvas is [CURSOR]
    - After that: canvas is state after executing actions so far, with CURSOR inserted
    """
    from reviser.data import is_move_token, get_move_amount, is_insert_token

    seq_len = len(prompt_tokens) + len(action_history)
    if seq_len <= 0:
        raise ValueError("build_canvas_tensors_shifted: seq_len must be >= 1")

    has_eoi = (len(prompt_tokens) >= 1 and int(prompt_tokens[-1]) == int(special_tokens.end_of_input))
    canvas_rows: List[List[int]] = []

    temp_canvas: List[int] = []
    temp_cursor = 0

    if has_eoi:
        # Prompt positions before END_OF_INPUT: empty canvas
        for _ in range(max(0, len(prompt_tokens) - 1)):
            canvas_rows.append([])
        # END_OF_INPUT position: empty canvas with cursor
        canvas_rows.append([int(special_tokens.cursor)])

        # For each already-executed action, update state then append state-with-cursor
        for a in action_history:
            a = int(a)
            if a == int(special_tokens.delete):
                if temp_cursor > 0:
                    temp_canvas.pop(temp_cursor - 1)
                    temp_cursor -= 1
            elif is_move_token(a):
                amt = int(get_move_amount(a))
                temp_cursor = max(0, min(len(temp_canvas), temp_cursor + amt))
            elif is_insert_token(a):
                temp_canvas.insert(temp_cursor, a)
                temp_cursor += 1
            else:
                # END_OF_RESPONSE or unknown: no-op for state tracking
                pass

            row = temp_canvas[:temp_cursor] + [int(special_tokens.cursor)] + temp_canvas[temp_cursor:]
            canvas_rows.append(row)
    else:
        # No END_OF_INPUT sentinel: treat the whole sequence as action-history tokens.
        # For prompt tokens (if any), canvas is always empty.
        for _ in range(len(prompt_tokens)):
            canvas_rows.append([])

        # For each action token at position t, we want canvas_rows[t] to be the canvas AFTER executing that action
        # (i.e., BEFORE executing the next token), matching "predict-next-token" semantics.
        for a in action_history:
            a = int(a)
            if a == int(special_tokens.delete):
                if temp_cursor > 0:
                    temp_canvas.pop(temp_cursor - 1)
                    temp_cursor -= 1
            elif is_move_token(a):
                amt = int(get_move_amount(a))
                temp_cursor = max(0, min(len(temp_canvas), temp_cursor + amt))
            elif is_insert_token(a):
                temp_canvas.insert(temp_cursor, a)
                temp_cursor += 1
            else:
                # END_OF_RESPONSE or unknown: no-op for state tracking
                pass

            row = temp_canvas[:temp_cursor] + [int(special_tokens.cursor)] + temp_canvas[temp_cursor:]
            canvas_rows.append(row)

    if len(canvas_rows) != seq_len:
        raise RuntimeError(
            f"build_canvas_tensors_shifted: internal error: len(canvas_rows)={len(canvas_rows)} != seq_len={seq_len}"
        )

    # Build padded CPU lists first, then materialize single tensors.
    # This is much faster than allocating per-row tensors on GPU in a Python loop.
    pad_id = 0
    ids_rows: List[List[int]] = []
    mask_rows: List[List[int]] = []
    for row in canvas_rows:
        r = list(row)
        if len(r) > max_canvas_len:
            r = r[:max_canvas_len]
        m = [1] * len(r)
        if len(r) < max_canvas_len:
            r = r + [pad_id] * (max_canvas_len - len(r))
            m = m + [0] * (max_canvas_len - len(m))
        ids_rows.append(r)
        mask_rows.append(m)

    canvas_ids = torch.tensor([ids_rows], dtype=torch.long, device=device)
    canvas_mask = torch.tensor([mask_rows], dtype=torch.long, device=device)
    return canvas_ids, canvas_mask


def canvas_row_with_cursor_tensors(
    *,
    canvas: List[int],
    cursor_pos: int,
    device: torch.device,
    max_canvas_len: int,
    special_tokens,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a single (B=1,T=1) canvas_ids/canvas_mask tensor for the CURRENT state,
    using the already-maintained (canvas, cursor_pos) rather than replaying actions.
    """
    row = list(canvas[: int(cursor_pos)]) + [int(special_tokens.cursor)] + list(canvas[int(cursor_pos) :])
    if len(row) > int(max_canvas_len):
        row = row[: int(max_canvas_len)]
    mask = [1] * len(row)
    if len(row) < int(max_canvas_len):
        row = row + [0] * (int(max_canvas_len) - len(row))
        mask = mask + [0] * (int(max_canvas_len) - len(mask))
    canvas_ids_last = torch.tensor(row, dtype=torch.long, device=device).unsqueeze(0)  # (1,C)
    canvas_mask_last = torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0)  # (1,C)
    return canvas_ids_last, canvas_mask_last

