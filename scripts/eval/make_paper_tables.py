#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import base_parser


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}\\%"


def main() -> None:
    ap = base_parser("Build compact paper-facing table assets from released summaries.")
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    arena_diff = _load_json(root / "results" / "arena" / "reviser_vs_diffusion_multijudge_summary.json")
    arena_ar = _load_json(root / "results" / "arena" / "ar_benchmark_v3_summary.json")
    evalppl = _load_json(root / "results" / "evalppl" / "gpt2base_evalppl_dream3k_summary.json")
    mauve = _load_json(root / "results" / "mauve" / "bert_pseudologlik_mauve_3000.json")

    # Arena table rows (AR baseline + diffusion)
    arena_rows = []
    for row in arena_ar.get("comparisons", {}).get("reviser_vs_ar_baseline", []):
        scale = str(row.get("scale", ""))
        wr = row.get("reviser_win_rate")
        br = row.get("baseline_win_rate")
        if wr is None or br is None:
            continue
        arena_rows.append((f"AR Baseline {scale}", float(wr), float(br)))
    for row in arena_diff.get("comparisons", []):
        baseline = str(row.get("baseline", ""))
        size = str(row.get("baseline_size", ""))
        wr = row.get("reviser_win_rate")
        br = row.get("baseline_win_rate")
        if wr is None or br is None:
            continue
        arena_rows.append((f"{baseline} {size}".strip(), float(wr), float(br)))

    # EvalPPL rows
    ppl_rows = []
    for section in ("reviser_vs_ar_baseline", "reviser_vs_sedd_mdlm"):
        for r in evalppl.get("comparisons", {}).get(section, []):
            ppl_rows.append(
                (
                    str(r.get("model", "")),
                    float(r.get("evalppl_gpt2_large", 0.0)),
                    float(r.get("evalppl_dream_7b", 0.0)),
                )
            )

    # MAUVE rows
    mauve_rows = []
    n_mauve = int(mauve.get("settings", {}).get("n_examples", 0))
    for r in mauve.get("results", []):
        mauve_rows.append((str(r.get("model", "")), float(r.get("mauve", 0.0)), n_mauve))

    lines = []
    lines.append("% Auto-generated from released artifacts via scripts/eval/make_paper_tables.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Released arena win rates used in the paper.}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Comparison & Reviser Win Rate & Baseline Win Rate \\\\")
    lines.append("\\midrule")
    for name, wr, br in arena_rows:
        lines.append(f"{name.replace('_', '\\_')} & {_fmt_pct(wr)} & {_fmt_pct(br)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Released evalPPL results used in the paper. Lower is better.}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Model & evalPPL GPT-2 Large & evalPPL Dream 7B \\\\")
    lines.append("\\midrule")
    for src, ppl_gpt2, ppl_dream in ppl_rows:
        lines.append(f"{src.replace('_', '\\_')} & {ppl_gpt2:.4f} & {ppl_dream:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Released MAUVE results used in the paper (BERT pseudo-loglik protocol).}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Model & MAUVE & $n$ \\\\")
    lines.append("\\midrule")
    for src, mv, n in mauve_rows:
        lines.append(f"{src.replace('_', '\\_')} & {mv:.4f} & {n} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
