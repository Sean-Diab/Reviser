#!/usr/bin/env python3
"""Are the cursor moves real? Per sample (35-token prefix typed, cursor at its end, then the model's new actions):

- A "jump" is a maximal run of consecutive move actions. Its net shift is where the cursor ends up minus where it was.
- A jump is **useful** when its net shift is not 0 and the next action is an edit in the middle of the text that
  stays: an insert with the cursor before the end of the canvas whose token is still on the final canvas, or a DELETE
  of a token the model wrote itself (not a prompt token). Useful moves of a jump = the fewest move actions that give
  its net shift (len(decompose_move(net))), so padding a jump with cancelling moves earns nothing.
- Cancelling jumps (net 0), jumps followed by EOR / the end of the budget, and jumps followed by an insert at the end
  of the canvas (e.g. returning to the end) are not useful.
- moves_removed_changes_text: replaying the actions without any move gives a different final canvas.

Deletes (move + delete arms E/F of the AR + RL test; DELETE removes the token left of the cursor):
- A "delete run" is a maximal run of consecutive DELETEs. Its removed tokens, in text order, are compared with the
  inserts that follow it immediately: the longest common prefix is **cancelled** (deleting X and retyping the same X
  right away changes nothing). The other deletes of the run are **effective**.
- deletes_own / deletes_prompt: the removed token was written by the model / was a prompt token.
- deletes_effective_own: effective deletes that removed a model-written token (the delete floor's count in
  train_grpo.py --del_metric effective_own; the cancelled ones are the run's last-deleted tokens).
- share_deletes_effective is the "deletes that change the text" measure (a delete undone later by other edits
  still counts as effective; rare).

Used as the "useful move" reward of the AR + RL cursor test (train_grpo.py --move_metric useful) and, as a CLI, on
decode_eval.py-format files:
  python scripts/arrl/move_audit.py runs/dagger/eval/dec/c4.paper.s20260924.july11656.jsonl [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "dagger"))
import hunk_oracle as ho  # noqa: E402


def _replay(prefix, actions, skip_moves=False):
    canvas = [(-1, t) for t in prefix]
    cursor = len(canvas)
    for j, a in enumerate(actions):
        a = int(a)
        if a == ho.EOR:
            break
        if ho.is_move(a):
            if not skip_moves:
                cursor = max(0, min(len(canvas), cursor + ho.move_amount(a)))
        elif a == ho.DELETE:
            if cursor > 0:
                canvas.pop(cursor - 1)
                cursor -= 1
        elif ho.is_insert(a):
            canvas.insert(cursor, (j, a))
            cursor += 1
    return canvas


def audit(prefix, actions):
    """Counts for one sample. actions: the new actions only (EOR, if any, last)."""
    prefix = [int(t) for t in prefix]
    actions = [int(a) for a in actions]
    final = _replay(prefix, actions)
    alive = {tag for tag, _ in final if tag >= 0}
    canvas = [(-1, t) for t in prefix]
    cursor = len(canvas)
    r = {"moves": 0, "jumps": 0, "cancel_jumps": 0, "cancel_moves": 0, "dead_end_moves": 0, "end_insert_moves": 0,
         "useful_jumps": 0, "useful_moves": 0, "useful_raw_moves": 0, "mid_inserts": 0, "mid_inserts_kept": 0,
         "deletes": 0, "deletes_own": 0, "deletes_prompt": 0, "deletes_cancelled": 0, "deletes_effective": 0,
         "deletes_effective_own": 0, "new_tokens": len(final) - len(prefix)}
    j, n = 0, len(actions)
    while j < n:
        a = actions[j]
        if a == ho.EOR:
            break
        if ho.is_move(a):
            k, c0 = j, cursor
            while k < n and ho.is_move(actions[k]):
                cursor = max(0, min(len(canvas), cursor + ho.move_amount(actions[k])))
                k += 1
            m, net = k - j, cursor - c0
            r["moves"] += m
            r["jumps"] += 1
            nxt = actions[k] if k < n else None
            useful = False
            if net == 0:
                r["cancel_jumps"] += 1
                r["cancel_moves"] += m
            elif nxt is None or nxt == ho.EOR:
                r["dead_end_moves"] += m
            elif ho.is_insert(nxt):
                if cursor < len(canvas):
                    useful = k in alive
                else:
                    r["end_insert_moves"] += m
            elif nxt == ho.DELETE:
                useful = cursor > 0 and canvas[cursor - 1][0] >= 0
            if useful:
                r["useful_jumps"] += 1
                r["useful_moves"] += len(ho.decompose_move(net))
                r["useful_raw_moves"] += m
            j = k
            continue
        if a == ho.DELETE:
            k, removed = j, []                      # removed: (tag, token) in deletion order (right to left)
            while k < n and actions[k] == ho.DELETE:
                r["deletes"] += 1
                if cursor > 0:
                    removed.append(canvas.pop(cursor - 1))
                    cursor -= 1
                k += 1
            text_order = removed[::-1]
            lcp = 0
            while (lcp < len(text_order) and k + lcp < n and ho.is_insert(actions[k + lcp])
                   and actions[k + lcp] == text_order[lcp][1]):
                lcp += 1
            r["deletes_own"] += sum(tag >= 0 for tag, _ in removed)
            r["deletes_prompt"] += sum(tag < 0 for tag, _ in removed)
            r["deletes_cancelled"] += lcp
            r["deletes_effective"] += len(removed) - lcp
            r["deletes_effective_own"] += sum(tag >= 0 for tag, _ in text_order[lcp:])
            j = k
            continue
        elif ho.is_insert(a):
            if cursor < len(canvas):
                r["mid_inserts"] += 1
                r["mid_inserts_kept"] += j in alive
            canvas.insert(cursor, (j, a))
            cursor += 1
        j += 1
    r["moves_removed_changes_text"] = [t for _, t in _replay(prefix, actions, skip_moves=True)] != [t for _, t in final]
    return r


def summarize(rows):
    tot = {}
    for r in rows:
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + int(v)
    mv, nt = max(1, tot["moves"]), max(1, tot["new_tokens"])
    with_moves = [r for r in rows if r["moves"] > 0]
    return {"n": len(rows), "moves_per_new_token": tot["moves"] / nt,
            "useful_moves_per_new_token": tot["useful_moves"] / nt,
            "share_moves_cancelling": tot["cancel_moves"] / mv, "share_moves_dead_end": tot["dead_end_moves"] / mv,
            "share_moves_then_end_insert": tot["end_insert_moves"] / mv,
            "share_moves_then_kept_mid_edit": tot["useful_raw_moves"] / mv,
            "jumps": tot["jumps"], "useful_jump_share": tot["useful_jumps"] / max(1, tot["jumps"]),
            "mid_inserts_per_new_token": tot["mid_inserts"] / nt,
            "mid_insert_kept_share": tot["mid_inserts_kept"] / max(1, tot["mid_inserts"]),
            "deletes_per_new_token": tot["deletes"] / nt,
            "effective_deletes_per_new_token": tot["deletes_effective"] / nt,
            "effective_own_deletes_per_new_token": tot["deletes_effective_own"] / nt,
            "share_deletes_effective": tot["deletes_effective"] / max(1, tot["deletes"]),
            "share_deletes_own": tot["deletes_own"] / max(1, tot["deletes"]),
            "share_deletes_prompt": tot["deletes_prompt"] / max(1, tot["deletes"]),
            "samples_with_deletes": sum(r["deletes"] > 0 for r in rows),
            "samples_with_moves": len(with_moves),
            "share_samples_text_changes_without_moves": (sum(r["moves_removed_changes_text"] for r in with_moves)
                                                         / max(1, len(with_moves)))}


def main():
    out = {}
    for path in sys.argv[1:]:
        rows = []
        for line in open(path, encoding="utf-8"):
            o = json.loads(line)
            if "_meta" in o:
                continue
            rows.append(audit(o["prefix_ids"], o["actions_new"]))
        out[Path(path).name] = summarize(rows)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
