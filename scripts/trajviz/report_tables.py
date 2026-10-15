#!/usr/bin/env python3
"""Print the markdown tables of notes/trajectory-check-2026-10-09.md from runs/trajviz/part{1,2}/stats.json.

  python scripts/trajviz/report_tables.py > runs/trajviz/report_tables.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pct(v, d=1):
    return "-" if v is None else f"{100 * v:.{d}f}%"


def fx(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def bins(v):
    return "-" if not v or v[0] is None else "/".join(f"{100 * x:.0f}" for x in v)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    s1 = json.loads((ROOT / "runs/trajviz/part1/stats.json").read_text())
    pv = json.loads((ROOT / "runs/trajviz/part1/paper_validation.json").read_text())
    print("### Part 1: where the cursor goes and where edits happen (3000 samples per model)\n")
    print(table(["model", "mean length", "stopped", "moves/token", "mean jump", "median jump", "samples with a move left",
                 "jump targets start/mid/end/at end (%)", "edits not at end", "edit positions start/mid/end/at end (%)"],
                [[v["label"], fx(v["agg"]["mean_out_len"], 1), pct(v["agg"]["stopped_rate"]), fx(v["agg"]["moves_per_token"], 3),
                  fx(v["agg"]["mean_jump_abs"]), v["agg"]["median_jump_abs"] if v["agg"]["median_jump_abs"] is not None else "-",
                  pct(v["agg"]["samples_with_backward_move"]), bins(v["agg"]["jump_target_bins"]), pct(v["agg"]["edits_not_at_end"]),
                  bins(v["agg"]["edit_bins"])] for v in s1.values()]))
    print("\n### Part 1: deletes\n")
    print(table(["model", "deletes/token", "deletes/sample", "samples with a delete", "deleted token typed <=1 action before",
                 "<=3", "<=5", "deleted text not typed in this run", "next action: different token", "next: same token back",
                 "same token back within 5", "prompt tokens deleted"],
                [[v["label"], fx(v["agg"]["deletes_per_token"], 3), fx(v["agg"]["deletes_per_sample"]), pct(v["agg"]["samples_with_delete"]),
                  pct(v["agg"]["delete_age_le1"]), pct(v["agg"]["delete_age_le3"]), pct(v["agg"]["delete_age_le5"]),
                  pct(v["agg"]["delete_of_start_tokens"]), pct(v["agg"]["delete_next"].get("different token typed (replace)")),
                  pct(v["agg"]["delete_next"].get("same token typed back")), pct(v["agg"]["delete_reinsert_within5"]),
                  v["agg"]["prompt_token_deletes"]] for v in s1.values()]))
    print("\n### Part 1: loops, limits and the prompt\n")
    print(table(["model", "hit 220-action limit", "state revisited 3+ times", "5+ times", "moves undone by next move (per 100)",
                 "8+ moves in a row", "end-of-input no-op samples", "prompt intact (locked)", "unlocked: prompt changed",
                 "unlocked: samples deleting prompt tokens"],
                [[v["label"], pct(v["agg"]["hit_budget_rate"]), pct(v["agg"]["samples_state_3plus"]), pct(v["agg"]["samples_state_5plus"]),
                  fx(v["agg"]["move_pingpong_per_100_moves"], 1), pct(v["agg"]["samples_move_run_8plus"]),
                  round(v["agg"]["samples_with_noop_tokens"] * v["n"]), pct(v["agg"]["prefix_intact_rate"]),
                  pct(v["unlocked"]["prompt_changed_rate"]) if v.get("unlocked") else "-",
                  pct(v["unlocked"]["samples_deleting_prompt_tokens"]) if v.get("unlocked") else "-"] for v in s1.values()]))
    print("\n### Part 1: the original paper's trajectory table (per-seed averages)\n")
    rows = []
    for size in ("100m", "300m"):
        for src in ("published", "ours"):
            o = pv["sizes"][size][src]
            rows.append([f"paper Reviser {size.upper()}, {'published' if src == 'published' else 'recomputed on the paper rollouts'}",
                         fx(o["actions_per_token"]), pct(o["insert_frac"]), pct(o["move_frac"]), "-", fx(o["mean_move_dist"]),
                         ", ".join(f"{x:.1f}" for x in o["mass_1_2_4_8"]), pct(o["frac_backward_moves"]),
                         fx(o["mean_insert_rel"]), pct(o["frac_end_append"]), pct(o["backward_revision_share"])])
    for v in s1.values():
        p = v["paper"]
        rows.append([v["label"], fx(p["actions_per_token"]), pct(p["insert_frac"]), pct(p["move_frac"]), pct(p["delete_frac"]),
                     fx(p.get("mean_move_dist")), ", ".join(f"{100 * p['move_dist_mass'][k]:.1f}" for k in ("1", "2", "4", "8")),
                     pct(p["frac_backward_moves"]), fx(p["mean_insert_rel"]), pct(p["frac_end_append"]), pct(p["backward_revision_share"])])
    print(table(["model", "actions/output token", "insert fraction", "move fraction", "delete fraction", "mean move size",
                 "move mass 1/2/4/8 (%)", "backward moves", "mean insert position", "end-appends", "examples with a backward revision"], rows))
    p2 = ROOT / "runs/trajviz/part2/stats.json"
    if not p2.exists():
        return
    s2 = json.loads(p2.read_text())
    sets = []
    for v in s2.values():
        for k in v["sets"]:
            if k not in sets:
                sets.append(k)
    for sk in sets:
        print(f"\n### Part 2: {sk}\n")
        rows = []
        for v in s2.values():
            r = v["sets"].get(sk)
            if not r:
                continue
            t = r.get("traj") or {}
            g = (lambda k, f=pct: f(t.get(k)) if t else "-")
            rows.append([v["label"], r["n"], pct(r["exact"]), pct(r["changed_rate"]), pct(r["stopped_rate"]),
                         ((pct(r["clean_left_alone"]) + f" of {r['n_clean']}"
                           + (f" (saved eval {pct(r['saved_clean_left_alone']['value'])})" if r.get("saved_clean_left_alone") else ""))
                          if r["n_clean"] else "-"),
                         g("mean_net_regions", fx), g("mean_ref_regions", fx), g("mean_content_actions", fx), g("moves_per_sample", fx),
                         bins(t.get("net_region_bins")) if t else "-", bins(t.get("ref_region_bins")) if t else "-",
                         g("deletes_per_sample", fx), g("self_delete_share"), g("delete_reinsert_within5"), g("noop_rollout_rate"),
                         g("extra_content_action_share"), g("samples_with_backward_move"), g("samples_back_after_first_edit"),
                         g("samples_state_3plus"), f"{r['saved_identical']}/{r['saved_compared']}" if r["saved_compared"] else "-"])
        print(table(["model", "n", "exact", "changed draft", "stopped", "clean left alone", "edit regions/draft",
                     "reference regions/draft", "inserts+deletes/draft", "moves/draft", "edit regions start/mid/end/at end (%)",
                     "reference regions start/mid/end/at end (%)", "deletes/draft", "deletes of own typing",
                     "deleted token typed back within 5", "did something, ended unchanged", "wasted inserts/deletes",
                     "samples with a move left", "moved left again after first edit", "state revisited 3+ times",
                     "same final text as saved eval"], rows))


if __name__ == "__main__":
    main()
