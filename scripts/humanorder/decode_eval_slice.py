#!/usr/bin/env python3
"""Decode prompts [start, end) of a prompt set with the July-control decoder, for running one eval in parallel slices.

Imports the July control's eval/decode_eval.py (unmodified: same model loader, prompt sets, protocols and row format)
and repeats its main loop over a slice. Prompt i is still seeded with torch.manual_seed(seed + i) (per_example
seeding), so the merged slices are row-for-row the same computation as one unsliced decode_eval.py run.
merge_decode_slices.py joins the slices into one decode_eval.py-format file (one _meta row, rows 0..n-1).

  python decode_eval_slice.py --eval_dir $CODE/eval --start 0 --end 250 <decode_eval.py args> --out x.part000.jsonl

--lock_prompt (default off; notes/promptlock-2026-10-08.md): the decoder's legality mask also forbids MOVEs that land
left of the prompt (cursor + amt < prefix_len; landing exactly on prefix_len stays legal) and DELETE at
cursor <= prefix_len, so the first prefix_len canvas tokens can never change (prefix_intact is then always true).
It swaps pretrain_resttraj.CursorState for a subclass; nothing else changes (same RNG use per legal action set).
Off: identical code path and output rows; the _meta args gain "lock_prompt" only when it is on.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch


def locked_cursor_state(base, prompt_len: int):
    """Subclass of the July decoder's CursorState whose legality also protects the first `prompt_len` canvas tokens.

    can_move: prompt_len <= cursor + amt <= len(canvas)   (moving exactly to the prompt end is legal)
    can_delete: cursor > prompt_len                       (DELETE removes the token left of the cursor)
    execute_action of the base class consults these, so an illegal move/delete is also refused on replay.
    """
    p = int(prompt_len)

    class LockedCursorState(base):
        PROMPT_LEN = p

        def can_move(self, amount: int) -> bool:
            new_pos = self.cursor_pos + int(amount)
            return self.PROMPT_LEN <= new_pos <= len(self.canvas)

        def can_delete(self) -> bool:
            return self.cursor_pos > self.PROMPT_LEN

    LockedCursorState.__name__ = LockedCursorState.__qualname__ = f"LockedCursorState{p}"
    return LockedCursorState


def install_prompt_lock(prompt_len: int, module=None):
    """Point reviser.training.pretrain_resttraj.CursorState (read at call time by _generate_from_action_prefix_nocanvas
    via CursorState.from_prompt, a classmethod, so the state is the subclass) at the locked subclass.
    module: the trainer module to patch (default: import reviser.training.pretrain_resttraj; tests pass their own).
    Returns the previous class (for tests)."""
    if module is None:
        import reviser.training.pretrain_resttraj as module  # noqa: E402
    PR = module
    prev = PR.CursorState
    base = getattr(prev, "_promptlock_base", prev)
    locked = locked_cursor_state(base, prompt_len)
    locked._promptlock_base = base
    PR.CursorState = locked
    return prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="the July control bundle's eval/ (decode_eval.py)")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt_set", choices=["dclm_dev", "c4"], required=True)
    ap.add_argument("--dclm_dir", default="")
    ap.add_argument("--c4_json", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix_len", type=int, default=35)
    ap.add_argument("--protocol", choices=["july", "paper"], default="july")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--seed_mode", choices=["per_example"], default="per_example")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock_prompt", action="store_true",
                    help="forbid moves left of prefix_len and DELETE at cursor <= prefix_len (prompt never edited)")
    a = ap.parse_args()
    sys.path.insert(0, a.eval_dir)
    import decode_eval as D  # noqa: E402  (July control eval/decode_eval.py, unmodified)
    from reviser.data import CursorTokenizer  # noqa: E402
    from reviser.training.pretrain_resttraj import _generate_from_action_prefix_nocanvas  # noqa: E402
    if a.lock_prompt:
        install_prompt_lock(a.prefix_len)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tok = CursorTokenizer()
    prompts = (D.dclm_dev_prompts(a.dclm_dir, a.n, a.prefix_len) if a.prompt_set == "dclm_dev"
               else D.c4_prompts(a.c4_json, a.n))
    if a.protocol == "july":
        max_new, max_seq, amp = 256, 512, contextlib.nullcontext
    else:
        max_new, max_seq = 256 - a.prefix_len, 256
        amp = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")  # noqa: E731
    model = D.load_model(a.ckpt, device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = {k: v for k, v in vars(a).items() if k not in ("eval_dir",) and not (k == "lock_prompt" and not v)}
    meta = {"name": a.name, "ckpt": str(a.ckpt), "ckpt_sha256": D.sha256(a.ckpt), "args": args,
            "max_new_actions_requested": max_new, "max_seq_len": max_seq, "n_prompts": len(prompts),
            "torch": torch.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "decode_eval_sha256": D.sha256(Path(a.eval_dir) / "decode_eval.py")}
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for i in range(a.start, min(a.end, len(prompts))):
            p = prompts[i]
            torch.manual_seed(a.seed + i)
            prefix = p["ref"][:a.prefix_len]
            with amp():
                _, st = _generate_from_action_prefix_nocanvas(
                    model=model, tokenizer=tok, prefix_actions=list(prefix), device=device,
                    max_new_actions=max_new, max_seq_len=max_seq, temperature=a.temperature, top_k=a.top_k)
            actions = [int(x) for x in st.action_history]
            canvas = [int(x) for x in st.canvas]
            n_new = len(actions) - a.prefix_len
            stopped = bool(actions and actions[-1] == D.EOR)
            row = {"i": i, "id": p["id"], "prefix_ids": prefix, "actions_new": actions[a.prefix_len:],
                   "canvas_ids": canvas, "ref_cont_ids": p["ref"][a.prefix_len:],
                   "stopped_eor": stopped, "hit_budget": (not stopped), "n_new_actions": n_new,
                   "out_len": len(canvas) - a.prefix_len, "prefix_intact": canvas[:a.prefix_len] == prefix,
                   "counts": D.action_counts(actions, a.prefix_len)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1 - a.start) % 25 == 0:
                print(f"[{a.name} {a.start}:{a.end}] {i+1-a.start} {time.time()-t0:.0f}s", flush=True)
    Path(str(out) + ".done").write_text(json.dumps({"secs": time.time() - t0}))
    print(json.dumps({"name": a.name, "start": a.start, "end": a.end, "secs": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
