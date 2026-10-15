"""CPU tests of the trajectory statistics (scripts/trajviz/trajstats.py) on hand-made trajectories."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trajviz"))
import trajstats as T  # noqa: E402

I = lambda t: ("i", t)  # noqa: E731
D = ("d",)
S = ("s",)
M = lambda k: ("m", k)  # noqa: E731


def test_july_ids():
    assert T.july_op(11) == ("i", 11)
    assert T.july_op(50259) == ("d",)
    assert T.july_op(50260) == ("s",)
    assert T.july_op(50261) == ("m", 1)
    assert T.july_op(50262) == ("m", -1)
    assert T.july_op(50263) == ("m", 2)
    assert T.july_op(50280) == ("m", -512)


def test_pos_bins():
    assert [T.pos_bin(c, 10, 0) for c in (0, 3, 4, 6, 7, 9, 10)] == [0, 0, 1, 1, 2, 2, 3]
    assert T.pos_bin(35, 35, 35) == 3          # empty continuation: at the end
    assert T.pos_bin(36, 45, 35) == 0          # measured inside the editable region only


def test_insert_delete_reinsert():
    # type 1 2 3, jump back 2, type 9 mid-text, delete it at once, type 9 back, stop
    ops = [I(1), I(2), I(3), M(-2), I(9), D, I(9), S]
    s = T.sample_stats([], ops)
    assert (s["n_ins"], s["n_del"], s["n_move"], s["n_actions"], s["n_steps"]) == (5, 1, 1, 7, 8)
    assert s["stopped"] and s["out_len"] == 4
    assert T.final_tokens([], ops) == [1, 9, 2, 3]
    assert s["ages"] == [1] and s["del_pre_existing"] == 0
    assert s["del_same_back"] == 1 and s["del_reinsert5"] == 1
    assert s["del_next"] == {"same token typed back": 1}
    assert s["n_back"] == 1 and s["jumps"] == [-2]
    assert s["jump_bins"] == {1: 1}            # cursor 1 of 3 tokens: middle third
    assert s["edit_bins"] == {3: 3, 1: 3}      # three appends at the end; insert, delete, insert in the middle
    agg = T.aggregate([s])
    assert agg["moves_per_token"] == 1 / 4 and agg["deletes_per_token"] == 1 / 4
    assert agg["delete_age_le1"] == 1.0 and agg["edits_not_at_end"] == 0.5
    assert agg["samples_with_backward_move"] == 1.0 and agg["prompt_token_deletes"] == 0


def test_replace_is_not_churn():
    s = T.sample_stats([], [I(1), I(2), D, I(3), S])
    assert s["del_same_back"] == 0 and s["del_reinsert5"] == 0
    assert s["del_next"] == {"different token typed (replace)": 1}


def test_delete_age_and_prompt_lock_offset():
    # prompt [100, 101] is protected (lo=2); the model types 5 6, moves back, deletes 5 (age 3), then a prompt token
    ops = [I(5), I(6), M(-1), D, D, S]
    s = T.sample_stats([100, 101], ops, lo=2, cursor=2)
    assert T.final_tokens([100, 101], ops, cursor=2) == [100, 6]
    assert s["ages"] == [3] and s["del_pre_existing"] == 1 and s["del_in_prompt"] == 1
    assert s["out_len"] == 0  # 2 tokens left, both counted against the 2-token prompt
    s2 = T.sample_stats([100, 101], [M(-1)], lo=2, cursor=2)
    assert s2["moves_into_prompt"] == 1


def test_loops_and_pingpong():
    s = T.sample_stats([1, 2, 3], [M(-1), M(1)] * 3)
    assert s["max_state_visits"] == 3 and s["move_pingpong"] == 5 and s["max_move_run"] == 6
    agg = T.aggregate([s])
    assert agg["samples_state_3plus"] == 1.0 and agg["samples_move_run_8plus"] == 0.0


def test_clamped_move():
    s = T.sample_stats([1, 2], [M(-4), I(7)])
    assert s["n_clamped"] == 1 and T.final_tokens([1, 2], [M(-4), I(7)]) == [7, 1, 2]


def test_edit_sample_stats():
    draft = [1, 2, 3, 4]
    s = T.edit_sample_stats(draft, [M(-2), D, I(7), S], reference=[1, 7, 3, 4])
    assert s["final"] == [1, 7, 3, 4] and s["changed"] and s["exact"]
    assert s["net_regions"] == 1 and s["ref_regions"] == 1
    assert s["min_token_edits"] == 2 and s["extra_content_actions"] == 0
    assert s["del_pre_existing"] == 1 and s["self_deletes"] == 0
    assert s["first_edit_bin"] == 1           # delete with the cursor at 2 of 4 tokens: middle third
    assert s["back_after_first_edit"] == 0
    two = T.edit_sample_stats(draft, [M(-1), D, M(-2), I(9), S])
    assert two["final"] == [9, 1, 2, 4] and two["back_after_first_edit"] == 1 and two["net_regions"] == 2
    noop = T.edit_sample_stats(draft, [I(5), D, S])
    assert noop["noop_rollout"] and not noop["changed"] and noop["extra_content_actions"] == 2
    clean = T.edit_sample_stats(draft, [S], reference=draft)
    assert not clean["changed"] and not clean["noop_rollout"] and clean["content_actions"] == 0


def test_paper_port_by_hand():
    # inserts 11, 12 at the end, move -1, insert 13 in the middle, EOR (not counted)
    st = T.paper_analyze([11, 12, 50262, 13, 50260, 99])
    assert (st["n_inserts"], st["n_moves"], st["n_deletes"], st["final_canvas_len"]) == (3, 1, 0, 3)
    assert st["steps_to_eor"] == 5
    assert st["insert_rel_positions"] == [0.0, 1.0, 0.5]   # empty canvas counts as 0.0, as in the paper's code
    assert math.isclose(st["actions_per_token"], 4 / 3)
    assert math.isclose(st["frac_end_append"], 2 / 3) and st["frac_backward_moves"] == 1.0
    assert st["has_backward_revision"]
    st2 = T.paper_analyze([50256, 11, 50259, 50260])   # 50256 is not an insert in the paper's code
    assert st2["n_inserts"] == 1 and st2["n_deletes"] == 1 and math.isnan(st2["actions_per_token"])
    tab = T.paper_table({"1": [[11, 12, 50262, 13, 50260]], "2": [[11, 50260]]})
    assert math.isclose(tab["avg"]["actions_per_token"], (4 / 3 + 1) / 2)
    assert tab["avg"]["backward_revision_share"] == 0.5
