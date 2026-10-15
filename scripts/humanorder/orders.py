"""Shared helpers for human-order trajectories: an order over final-text tokens -> Reviser actions.

A trajectory writes the prefix left to right, then inserts every continuation token exactly once in some order,
then EOR. Token j always lands between its nearest already-written neighbours, so any order is valid. The cursor
moves are derived from the order with the same power-of-2 move tokens as July's run-and-jump generator.
"""
from __future__ import annotations

import bisect

EOR_ID = 50260
MOVE_START = 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def move_token(amount: int) -> int:
    idx = MOVE_AMOUNTS.index(abs(amount))
    return MOVE_START + idx * 2 + (1 if amount < 0 else 0)


def decompose_move(amount: int) -> list[int]:
    """Greedy power-of-2 move tokens (identical to gen_runjump.decompose_move)."""
    if amount == 0:
        return []
    sign = 1 if amount > 0 else -1
    remaining, out = abs(amount), []
    for p in reversed(MOVE_AMOUNTS):
        while remaining >= p:
            out.append(move_token(sign * p))
            remaining -= p
    return out


def order_to_actions(prefix: list[int], cont: list[int], order: list[int]) -> list[int]:
    """Raw action stream [prefix inserts] + trajectory + EOR for inserting cont[j] in `order`."""
    assert sorted(order) == list(range(len(cont))), "order must be a permutation of the continuation"
    actions = list(prefix)
    present: list[int] = []   # sorted continuation indices on the canvas
    cursor = 0                # boundary in continuation-canvas coordinates (0..len(present))
    for j in order:
        pos = bisect.bisect_left(present, j)
        actions.extend(decompose_move(pos - cursor))
        actions.append(int(cont[j]))
        present.insert(pos, j)
        cursor = pos + 1
    actions.append(EOR_ID)
    return actions


def run_lengths(order: list[int]) -> tuple[list[int], list[int]]:
    """(insert runs without a cursor move, signed jump distances in written tokens), from the canvas replay.

    The first insert of the continuation starts at the prefix end (cursor 0), as in the action stream.
    """
    present: list[int] = []
    cursor, run, runs, jumps = 0, 0, [], []
    for j in order:
        pos = bisect.bisect_left(present, j)
        if pos != cursor:
            if run:
                runs.append(run)
            jumps.append(pos - cursor)
            run = 0
        run += 1
        present.insert(pos, j)
        cursor = pos + 1
    if run:
        runs.append(run)
    return runs, jumps
