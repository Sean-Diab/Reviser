"""Tests for train_grpo.gen_text_moves (--move_metric gen: only moves that land in the generated text count)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "grpo"))

import train_grpo as tg  # noqa: E402

ho = tg.ho
MV = {ho.move_amount(t): t for t in range(ho.MOVE_START, ho.MOVE_START + ho.N_MOVES)}
P5 = [1, 2, 3, 4, 5]


def test_move_inside_generated_text_counts():
    # canvas 1 2 3 4 5 | 10 11, cursor 7 -> move -1 -> cursor 6 (inside the generated text)
    assert tg.gen_text_moves(P5, [10, 11, MV[-1], 12, ho.EOR]) == 1


def test_move_to_start_of_generated_text_counts():
    # cursor 7 -> move -2 -> cursor 5 = first position after the prompt (position >= prompt length)
    assert tg.gen_text_moves(P5, [10, 11, MV[-2], 12]) == 1


def test_move_into_prompt_does_not_count():
    # cursor 7 -> move -4 -> cursor 3, inside the prompt; the insert there earns nothing either
    assert tg.gen_text_moves(P5, [10, 11, MV[-4], 12, ho.EOR]) == 0


def test_run_counts_by_final_position():
    # run through the prompt and back out: -4 (cursor 3, prompt) then +2 (cursor 5): the whole run counts (2 moves)
    assert tg.gen_text_moves(P5, [10, 11, MV[-4], MV[2], 12]) == 2
    # run ending inside the prompt: neither move counts
    assert tg.gen_text_moves(P5, [10, 11, MV[-1], MV[-2], 12]) == 0


def test_prompt_edits_shift_the_boundary():
    # 1 2 3 4 5 10 | move -4 -> cursor 2 (prompt, 0) | insert 11 -> 1 2 11 3 4 5 10, cursor 3
    # move +2 -> cursor 5: prompt tokens 4 5 still right of it -> 0
    # move +1 -> cursor 6 (before 10, all right of it model-written) -> 1
    acts = [10, MV[-4], 11, MV[2], 12, MV[1], 13]
    assert tg.gen_text_moves(P5, acts) == 1


def test_prompt_delete_moves_the_boundary_left():
    # delete the last prompt token (cursor 5 -> 4), type 10 -> 1 2 3 4 10, cursor 5; move -1 -> cursor 4: right of it
    # only the model's 10 -> counts
    assert tg.gen_text_moves(P5, [ho.DELETE, 10, MV[-1], 11]) == 1


def test_no_moves():
    assert tg.gen_text_moves(P5, [10, 11, 12, ho.EOR]) == 0
    assert tg.gen_text_moves(P5, []) == 0
