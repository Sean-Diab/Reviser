"""order_to_actions must rebuild exactly prefix + cont for any order (replayed with explicit cursor semantics)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "humanorder"))
from orders import EOR_ID, MOVE_AMOUNTS, MOVE_START, order_to_actions, run_lengths  # noqa: E402


def replay(actions, prefix_len):
    canvas, cursor = [], 0
    for t in actions:
        if t == EOR_ID:
            break
        if MOVE_START <= t < MOVE_START + 2 * len(MOVE_AMOUNTS):
            off = t - MOVE_START
            amt = MOVE_AMOUNTS[off // 2] * (-1 if off % 2 else 1)
            cursor += amt
            assert prefix_len <= cursor <= len(canvas), "cursor left the canvas or entered the prefix"
        else:
            canvas.insert(cursor, t)
            cursor += 1
    return canvas


def test_random_orders():
    rng = random.Random(0)
    for _ in range(300):
        p = [rng.randrange(1000) for _ in range(rng.randint(0, 35))]
        c = [rng.randrange(1000) for _ in range(rng.randint(1, 120))]
        order = list(range(len(c)))
        if rng.random() < 0.5:
            rng.shuffle(order)
        acts = order_to_actions(p, c, order)
        assert acts[-1] == EOR_ID
        assert replay(acts, len(p)) == p + c


def test_runs():
    assert run_lengths([0, 1, 2, 3]) == ([4], [])
    # skip token 1, write 2-3, go back to fill 1: canvas [0,2,3] cursor 3 -> pos 1, jump -2
    assert run_lengths([0, 2, 3, 1]) == ([3, 1], [-2])


if __name__ == "__main__":
    test_random_orders(); test_runs(); print("ok")
