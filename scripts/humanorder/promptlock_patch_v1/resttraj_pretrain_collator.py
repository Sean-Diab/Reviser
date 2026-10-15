from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .vocabulary import SPECIAL_TOKENS, get_move_amount, is_insert_token, is_move_token


def _to_int_list(values: Any) -> List[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def infer_prefix_length_from_initial_inserts(
    action_ids: List[int],
    *,
    max_prefix_length: Optional[int] = None,
) -> int:
    limit = len(action_ids) if max_prefix_length is None else min(len(action_ids), int(max_prefix_length))
    prefix_len = 0
    for token_id in action_ids[:limit]:
        if is_insert_token(int(token_id)):
            prefix_len += 1
            continue
        break
    return int(prefix_len)


def resolve_prefix_length(
    *,
    sample: Dict[str, Any],
    action_ids: List[int],
    fixed_prefix_length: Optional[int],
    infer_prefix_length: bool,
    prefix_length_max: Optional[int],
) -> int:
    sample_prefix = _to_optional_int(sample.get("prefix_len"))
    if sample_prefix is not None:
        prefix_len = sample_prefix
    elif fixed_prefix_length is not None:
        prefix_len = int(fixed_prefix_length)
    elif infer_prefix_length:
        prefix_len = infer_prefix_length_from_initial_inserts(
            action_ids,
            max_prefix_length=prefix_length_max,
        )
    else:
        prefix_len = 0

    if prefix_length_max is not None:
        prefix_len = min(int(prefix_len), int(prefix_length_max))
    return max(0, min(int(prefix_len), len(action_ids)))


def cursor_state_before_actions(action_ids: List[int]) -> tuple[List[int], List[int]]:
    """[promptlock_patch_v1] (cursor, canvas_len) BEFORE each action (state after action_ids[:t]).

    Executor semantics are exactly those of `_build_canvas_rows` below (and of the July trainer's CursorState for
    legal actions): insert at the cursor (cursor += 1), DELETE pops the token left of the cursor (no-op at 0),
    moves clamp to [0, len]. Position t of the batch predicts label action_ids[t] from this state.
    """
    delete_id = int(SPECIAL_TOKENS.delete)
    cur = 0
    n = 0
    curs: List[int] = []
    lens: List[int] = []
    for token_id in action_ids:
        tid = int(token_id)
        curs.append(cur)
        lens.append(n)
        if tid == delete_id:
            if cur > 0:
                cur -= 1
                n -= 1
        elif is_move_token(tid):
            cur = max(0, min(n, cur + int(get_move_amount(tid))))
        elif is_insert_token(tid):
            n += 1
            cur += 1
    return curs, lens


def _build_canvas_rows(action_ids: List[int], max_canvas_length: int) -> List[List[int]]:
    canvas: List[int] = []
    cursor_pos = 0
    rows: List[List[int]] = [[int(SPECIAL_TOKENS.cursor)]]

    for token_id in action_ids[:-1]:
        tid = int(token_id)
        if tid == int(SPECIAL_TOKENS.delete):
            if cursor_pos > 0:
                canvas.pop(cursor_pos - 1)
                cursor_pos -= 1
        elif is_move_token(tid):
            amt = int(get_move_amount(tid))
            cursor_pos = max(0, min(len(canvas), cursor_pos + amt))
        elif is_insert_token(tid):
            canvas.insert(cursor_pos, tid)
            cursor_pos += 1

        row = canvas[:cursor_pos] + [int(SPECIAL_TOKENS.cursor)] + canvas[cursor_pos:]
        if max_canvas_length > 0 and len(row) > max_canvas_length:
            row = row[:max_canvas_length]
        rows.append(row)

    return rows


class RestTrajPretrainCollator:
    """Builds next-token LM batches for restoration-trajectory supervised training."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        max_seq_length: int,
        max_canvas_length: int,
        use_canvas: bool,
        fixed_prefix_length: Optional[int] = None,
        infer_prefix_length: bool = False,
        prefix_length_max: Optional[int] = None,
        legality_state: bool = False,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.max_seq_length = int(max_seq_length)
        self.max_canvas_length = int(max_canvas_length)
        self.use_canvas = bool(use_canvas)
        self.fixed_prefix_length = None if fixed_prefix_length is None else int(fixed_prefix_length)
        self.infer_prefix_length = bool(infer_prefix_length)
        self.prefix_length_max = None if prefix_length_max is None else int(prefix_length_max)
        # [promptlock_patch_v1] also emit per-position cursor / canvas length (training.legality_mask). Off by
        # default: the batch is then exactly the lossmask_patch_v1 batch (same keys, same tensors).
        self.legality_state = bool(legality_state)

    def __call__(self, batch: List[Dict[str, Any]] | Dict[str, Any]) -> Dict[str, torch.Tensor]:
        collate_t0 = time.perf_counter()
        if isinstance(batch, dict):
            batch = [batch]

        samples = [sample for sample in batch if sample is not None]
        if not samples:
            raise ValueError("Empty batch in RestTrajPretrainCollator")

        processed: List[Dict[str, Any]] = []
        canvas_ms = 0.0
        for sample in samples:
            action_ids = sample.get("action_ids", sample.get("input_ids"))
            if action_ids is None:
                continue
            actions = _to_int_list(action_ids)
            if not actions:
                continue
            actions = actions[: self.max_seq_length]
            if not actions:
                continue

            prefix_len = resolve_prefix_length(
                sample=sample,
                action_ids=actions,
                fixed_prefix_length=self.fixed_prefix_length,
                infer_prefix_length=self.infer_prefix_length,
                prefix_length_max=self.prefix_length_max,
            )

            input_ids = [int(SPECIAL_TOKENS.end_of_input)] + actions[:-1]
            labels = list(actions)
            for idx in range(min(prefix_len, len(labels))):
                labels[idx] = -100
            # [lossmask_patch_v1] optional per-action label mask from the dataset (1 = ignore this label).
            loss_mask = sample.get("loss_mask")
            if loss_mask is not None:
                for idx, flag in enumerate(_to_int_list(loss_mask)[: len(labels)]):
                    if flag:
                        labels[idx] = -100
            record: Dict[str, Any] = {
                "input_ids": input_ids,
                "labels": labels,
                "prefix_len": prefix_len,
                "seq_len": len(input_ids),
            }
            if self.use_canvas:
                canvas_t0 = time.perf_counter()
                record["canvas_rows"] = _build_canvas_rows(actions, max(1, self.max_canvas_length))
                canvas_ms += (time.perf_counter() - canvas_t0) * 1e3
            if self.legality_state:  # [promptlock_patch_v1]
                record["cursor"], record["canvas_len"] = cursor_state_before_actions(actions)
            processed.append(record)

        if not processed:
            raise ValueError("All samples were empty in RestTrajPretrainCollator")

        batch_size = len(processed)
        max_seq_len = max(item["seq_len"] for item in processed)

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        output: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prefix_lens": torch.zeros((batch_size,), dtype=torch.long),
        }
        if self.legality_state:  # [promptlock_patch_v1] padding positions: cursor 0, len 0 (labels are -100 there)
            output["legal_cursor"] = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            output["legal_canvas_len"] = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        if self.use_canvas:
            max_canvas_len = max(
                1,
                min(
                    max(len(row) for item in processed for row in item["canvas_rows"]),
                    max(1, self.max_canvas_length),
                ),
            )
            canvas_ids = torch.zeros((batch_size, max_seq_len, max_canvas_len), dtype=torch.long)
            canvas_mask = torch.zeros((batch_size, max_seq_len, max_canvas_len), dtype=torch.long)
            output["canvas_ids"] = canvas_ids
            output["canvas_mask"] = canvas_mask

        for i, item in enumerate(processed):
            seq_len = int(item["seq_len"])
            output["input_ids"][i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            output["labels"][i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
            output["attention_mask"][i, :seq_len] = 1
            output["prefix_lens"][i] = int(item["prefix_len"])
            if self.legality_state:  # [promptlock_patch_v1]
                output["legal_cursor"][i, :seq_len] = torch.tensor(item["cursor"], dtype=torch.long)
                output["legal_canvas_len"][i, :seq_len] = torch.tensor(item["canvas_len"], dtype=torch.long)

            if self.use_canvas:
                for t, row in enumerate(item["canvas_rows"][:seq_len]):
                    row_len = min(len(row), output["canvas_ids"].size(2))
                    if row_len <= 0:
                        continue
                    output["canvas_ids"][i, t, :row_len] = torch.tensor(row[:row_len], dtype=torch.long)
                    output["canvas_mask"][i, t, :row_len] = 1

        collate_ms = (time.perf_counter() - collate_t0) * 1e3
        output["collate_ms"] = torch.tensor(collate_ms, dtype=torch.float32)
        output["canvas_ms"] = torch.tensor(canvas_ms, dtype=torch.float32)
        return output
