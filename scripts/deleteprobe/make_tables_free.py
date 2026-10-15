#!/usr/bin/env python3
"""Markdown tables from runs/eval/deleteprobe/free/report.json (scripts/deleteprobe/free_deletes.py; helper D 2026-10-11).

  python scripts/deleteprobe/make_tables_free.py [report.json] > runs/eval/deleteprobe/free/tables.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/eval/deleteprobe/free/report.json"
R = json.loads(P.read_text())
ORDER = ["july608", "jd608", "jd608_g8m2_s1", "jd608_g8m2_s2", "jd608_g8m2_s3", "rev100", "ar100", "rev300",
         "jd608_300m", "jd608_300m_g8m2_s3", "ar300", "train_ref"]
M = {k: R["models"][k] for k in ORDER if k in R["models"]}


def f(t, d=4, pct=False, pd=1):
    if t is None:
        return "—"
    if pct:
        return f"{100 * t[0]:.{pd}f}% [{100 * t[1]:.{pd}f}, {100 * t[2]:.{pd}f}]"
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


print("## A. How often the models delete (C4 3k, prompt-locked decodes; AR not locked)\n")
print("| Model | New tokens / output | Deletes | Deletes / new token [95% CI] | Deletes / new action | "
      "Outputs with ≥ 1 delete |")
print("|---|---|---|---|---|---|")
for n, m in M.items():
    print(f"| {n} | {m['new_tokens_mean']:.1f} | {m['deletes']} | {f(m['deletes_per_new_token'])} | "
          f"{m['deletes_per_new_action'][0]:.4f} | {f(m['outputs_with_delete'], pct=True)} |")

print("\n## B. What is deleted (share of deletes; 95% CI by prompt-cluster bootstrap)\n")
print("| Model | Deletes | Prompt tokens deleted | Deleted token typed by the action just before | Median age (actions) | "
      "Cursor at canvas end | Next action is an insert | Single-delete runs | Replacement = deleted token |")
print("|---|---|---|---|---|---|---|---|---|")
for n, m in M.items():
    e = m.get("events")
    if not e:
        continue
    print(f"| {n} | {e['n']} | {e['prompt_token_deleted']} | {f(e['just_typed'], pct=True)} | {e['age_median']:.0f} "
          f"(mean {e['age_mean']:.3f}) | {f(e['cursor_at_end'], pct=True)} | {f(e['next_insert'], pct=True)} | "
          f"{100 * e['run_len_1_share']:.2f}% | {f(e['repl_same_of_with_repl'], pct=True)} |")

print("\n## C. Is the deleted token a repeat? (share of deletes)\n")
print("| Model | = token just left of it | in the 3 tokens left of it | within 4 tokens either side | "
      "word repeat (same word as the previous word) |")
print("|---|---|---|---|---|")
for n, m in M.items():
    e = m.get("events")
    if not e:
        continue
    print(f"| {n} | {f(e['rep1'], pct=True)} | {f(e['rep3'], pct=True)} | {f(e['near4'], pct=True)} | "
          f"{f(e['word_repeat'], pct=True)} |")

print("\n## D. GPT-2 small: probability of the deleted token vs the token typed in its place (same left context)\n")
print("| Model | Deletes with an insert next | P(deleted) median | P(replacement) median | Replacement more likely "
      "[CI] | Mean log10 P(repl)/P(del) [CI] | Kept new tokens: P median / share P < 0.01 | Deleted: share P < 0.01 |")
print("|---|---|---|---|---|---|---|---|")
for n, m in M.items():
    g = m.get("gpt2_deletes")
    k = m.get("gpt2_kept_new_tokens")
    if not g:
        if k:
            print(f"| {n} | — | — | — | — | — | {k['p_median']:.3f} / {100 * k['share_p_lt_0.01']:.1f}% | — |")
        continue
    print(f"| {n} | {g.get('n_with_repl', 0)} | {g['p_deleted_median']:.4f} | {g.get('p_repl_median', float('nan')):.4f} | "
          f"{f(g.get('share_repl_more_likely'), pct=True)} | {f(g.get('mean_log10_ratio_repl_over_deleted'), d=2)} | "
          f"{k['p_median']:.3f} / {100 * k['share_p_lt_0.01']:.1f}% | {100 * g['share_deleted_p_lt_0.01']:.1f}% |")
print("\nSplit by whether the deleted token repeats the token just left of it:\n")
print("| Model | rep1: n / P(del) med / P(repl) med / repl more likely | not rep1: n / P(del) med / P(repl) med / "
      "repl more likely |")
print("|---|---|---|")
for n, m in M.items():
    g = m.get("gpt2_deletes")
    if not g or "rep1" not in g:
        continue
    a, b = g["rep1"], g["not_rep1"]
    print(f"| {n} | {a['n']} / {a['p_deleted_median']:.4f} / {a['p_repl_median']:.4f} / "
          f"{100 * a['share_repl_more_likely']:.1f}% | {b['n']} / {b['p_deleted_median']:.4f} / "
          f"{b['p_repl_median']:.4f} / {100 * b['share_repl_more_likely']:.1f}% |")

print("\n## E. Final texts (new tokens only): repetition per model [95% CI]\n")
print("| Model | Adjacent duplicate tokens | Adjacent duplicate words | rep-4 |")
print("|---|---|---|---|")
for n, m in M.items():
    if n == "train_ref":
        continue
    print(f"| {n} | {f(m['adj_dup_token'], pct=True, pd=2)} | {f(m['adj_dup_word'], pct=True, pd=2)} | "
          f"{f(m['rep4'], pct=True, pd=2)} |")

print("\n## F. Paired differences A − B over the same 3,000 prompts (percentage points, 95% paired bootstrap CI)\n")
print("| A vs B | New tokens A / B | Δ adj dup tokens | Δ adj dup words | Δ rep-4 | Matched length (mean) | "
      "Δ adj dup tokens (matched) | Δ adj dup words (matched) | Δ rep-4 (matched) |")
print("|---|---|---|---|---|---|---|---|---|")
for k, p in R["pairs"].items():
    print(f"| {k.replace('_vs_', ' vs ')} | {p['len_mean'][0]:.1f} / {p['len_mean'][1]:.1f} | "
          f"{f(p['adj_dup_token'], pct=True, pd=2)} | {f(p['adj_dup_word'], pct=True, pd=2)} | "
          f"{f(p['rep4'], pct=True, pd=2)} | {p['matched_len_mean']:.1f} | "
          f"{f(p['matched_adj_dup_token']['diff'], pct=True, pd=2)} | "
          f"{f(p['matched_adj_dup_word']['diff'], pct=True, pd=2)} | {f(p['matched_rep4']['diff'], pct=True, pd=2)} |")

print("\n## G. Most deleted tokens (top 10)\n")
for n, m in M.items():
    e = m.get("events")
    if e:
        print(f"- {n}: " + ", ".join(f"{json.dumps(t)} {c}" for t, c in e["top_deleted"][:10]))
