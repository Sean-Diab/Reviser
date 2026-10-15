"""Blinded pairwise LLM-judge assessment of base-model continuations (task 28): released Reviser 100M (rev100) vs the
paper's AR 100M (ar100), on existing paper-protocol outputs in runs/cont_base/dec/. It uses subscription CLIs only,
with no purchased judge API.

The design was fixed before any verdicts existed:
- **Items.** 100 prompts: 50 from C4 and 50 from itdocs, sampled with seed 20260924.
- **Texts.** The judged text is the full final canvas decoded with GPT-2, prefix included; the 35-token prefix is shown
  separately.
- **Conditions.**
  - `lm` (length-matched, PRIMARY): both canvases are truncated to the prefix plus min(out_len_rev, out_len_ar) new
    tokens. Finding 26 showed that stopping/length drives the metric differences, so this condition removes length.
  - `ag` (as generated): full outputs.
- **Orders.**
  - `lm` is judged twice, in order 1 (random A/B per item) and order 2 (swapped), in separate calls.
  - `ag` is judged once, in order 1.
- **Judges.** `codex` (Codex CLI, ChatGPT login, API-key variables removed, read-only sandbox) and `claude` (Claude
  Code CLI, claude-opus-5-5 via scripts/claude-subscription.ps1, no tools). Neither sees system names.
- **Score.** Per item, the rev100 score is 1 for a win, 0.5 for a tie and 0 for a loss, averaged over orders. Report
  the mean with a 95% bootstrap CI over items (10k resamples), an exact sign test on non-tied items, position
  consistency, and agreement between the two judges.

  python scripts/judge_continuations.py build          # writes runs/cont_judge/{items.json,prompt_<cond>_<order>.md}
  python scripts/judge_continuations.py run codex      # then: run claude (skips finished calls)
  python scripts/judge_continuations.py analyze        # runs/cont_judge/analysis.json + markdown

Second use (JUDGE_CFG=july; task 19 secondary, exploratory, design fixed before any verdict): the same design,
instruction, conditions, seed and analysis applied to the July control outputs in runs/july_control/eval/dec/.
Here "rev" = july84488 (run-and-jump) and "ar" = ctl_tok (inverse-trajectory control). Pools: C4 paper protocol and
DCLM July protocol, 50 prompts each. Outputs go to runs/july_judge/.
  JUDGE_CFG=july python scripts/judge_continuations.py build|run codex|run claude|analyze
"""
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("JUDGE_CFG", "base") == "custom":
    # Overnight arms (2026-10-07): the same design applied to any pair of decoded systems in one directory, e.g.
    # JUDGE_CFG=custom JUDGE_DEC=runs/humanorder/eval/dec JUDGE_OUT=runs/judge/q84_vs_july84 JUDGE_REV=q84
    # JUDGE_AR=july11656. "rev" is the new arm, "ar" the reference arm; pools C4 paper + DCLM July, 50 prompts each.
    DEC = ROOT / os.environ["JUDGE_DEC"]
    OUT = ROOT / os.environ["JUDGE_OUT"]
    SYS = (os.environ["JUDGE_REV"], os.environ["JUDGE_AR"])
    SETS = [("c4", "c4.paper.s20260924"), ("dclm", "dclm.july.s20260924")]
elif os.environ.get("JUDGE_CFG", "base") == "july":
    DEC = ROOT / "runs/july_control/eval/dec"
    OUT = ROOT / "runs/july_judge"
    SYS = ("july84488", "ctl_tok")
    SETS = [("c4", "c4.paper.s20260924"), ("dclm", "dclm.july.s20260924")]
else:
    DEC = ROOT / "runs/cont_base/dec"
    OUT = ROOT / "runs/cont_judge"
    SYS = ("rev100", "ar100")
    SETS = [("c4", "c4.paper.s20260924"), ("itdocs", "itdocs.paper.s20260924")]
EMPTY = ROOT / "tmp/judge_empty_cwd"
SEED = 20260924
PREFIX = 35
TASKS = [("lm", 1), ("lm", 2), ("ag", 1)]

INSTR = """You are judging text continuations. For each item you get a PREFIX (the start of a document) and two
complete texts, A and B, each of which begins with (approximately) that prefix and continues it. Decide which text
is the better continuation of the prefix, judging:
coherence and consistency with the prefix, fluency and grammaticality, and absence of degenerate repetition or
nonsense. Both texts may be cut off mid-sentence; do not penalize an abrupt ending, and do not prefer a text for
being longer or shorter. Answer "tie" only if you truly cannot prefer one.

Do not use any tools, do not read files, do not run commands. Output ONLY one JSON object per line, one per item,
in item order, exactly like: {"item": "J001", "winner": "A"} with winner one of "A", "B", "tie". No other text.

"""


def load(stem, name):
    rows = [json.loads(l) for l in open(DEC / f"{stem}.{name}.jsonl")]
    return {r["id"]: r for r in rows[1:]}


def build():
    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(SEED)
    items = []
    for ds, stem in SETS:
        rev, ar = load(stem, SYS[0]), load(stem, SYS[1])
        ids = sorted(set(rev) & set(ar))
        for i in rng.sample(ids, 50):
            r, a = rev[i], ar[i]
            assert r["prefix_ids"] == a["prefix_ids"]
            m = min(r["out_len"], a["out_len"])
            items.append(dict(id=i, dataset=ds, prefix=enc.decode(r["prefix_ids"]),
                              rev_prefix_intact=r["prefix_intact"], rev_out_len=r["out_len"], ar_out_len=a["out_len"],
                              matched_new=m,
                              text={"ag": {"rev": enc.decode(r["canvas_ids"]), "ar": enc.decode(a["canvas_ids"])},
                                    "lm": {"rev": enc.decode(r["canvas_ids"][:PREFIX + m]),
                                           "ar": enc.decode(a["canvas_ids"][:PREFIX + m])}}))
    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it["item"] = f"J{k:03d}"
        it["rev_is_A_order1"] = rng.random() < 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(dict(seed=SEED, items=items), indent=1), encoding="utf-8")
    for cond, order in TASKS:
        parts = [INSTR]
        for it in items:
            rev_a = it["rev_is_A_order1"] if order == 1 else not it["rev_is_A_order1"]
            t = it["text"][cond]
            a, b = (t["rev"], t["ar"]) if rev_a else (t["ar"], t["rev"])
            parts.append(f"### {it['item']}\nPREFIX:\n{it['prefix']}\n\nA:\n{a}\n\nB:\n{b}\n")
        (OUT / f"prompt_{cond}_{order}.md").write_text("\n".join(parts), encoding="utf-8")
    print(len(items), "items; prompts:", [f"prompt_{c}_{o}.md" for c, o in TASKS])


def run(judge):
    for cond, order in TASKS:
        p = OUT / f"prompt_{cond}_{order}.md"
        out = OUT / f"verdicts_{judge}_{cond}_{order}.txt"
        if out.exists() and out.stat().st_size > 0:
            continue
        EMPTY.mkdir(parents=True, exist_ok=True)  # judge cwd: nothing to read (items.json holds system names)
        with open(p, encoding="utf-8") as f:
            if judge == "claude":
                cmd = ["powershell.exe", "-NoProfile", "-File", str(ROOT / "scripts/claude-subscription.ps1"), "-p",
                       "--model", "claude-opus-5-5", "--no-session-persistence", "--output-format", "json",
                       "--disallowedTools", "Bash,Edit,Write,NotebookEdit,Agent,WebFetch,WebSearch,Read,Grep,Glob"]
                r = subprocess.run(cmd, stdin=f, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
                d = json.loads(r.stdout)
                (OUT / f"meta_{judge}_{cond}_{order}.json").write_text(json.dumps(
                    {k: d.get(k) for k in ("is_error", "modelUsage", "duration_ms", "session_id")}, indent=1))
                if d.get("is_error"):  # e.g. transient OAuth refresh failure: keep no verdict file so a rerun retries
                    (OUT / f"error_{judge}_{cond}_{order}.txt").write_text(str(d.get("result")), encoding="utf-8")
                else:
                    out.write_text(d.get("result", ""), encoding="utf-8")
            else:
                env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")}
                env["PATH"] = env.get("PATH", "") + os.pathsep + r"~\AppData\Roaming\npm"
                last = OUT / f"verdicts_{judge}_{cond}_{order}.txt"
                r = subprocess.run(["~/AppData/Roaming/npm/codex.cmd", "exec", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                                    "--output-last-message", str(last), "-"],
                                   input=f.read(), capture_output=True, text=True, encoding="utf-8", env=env, cwd=EMPTY)
                (OUT / f"log_{judge}_{cond}_{order}.txt").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
        print(judge, cond, order, "rc", r.returncode, "bytes", out.stat().st_size if out.exists() else 0)


def parse(path, n):
    v = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.search(r'\{[^{}]*"item"\s*:\s*"(J\d{3})"[^{}]*"winner"\s*:\s*"(A|B|tie)"[^{}]*\}', line, re.I)
        if m:
            v[m.group(1)] = m.group(2).upper() if m.group(2).lower() != "tie" else "tie"
    return v


def sign_p(k, n):
    if n == 0:
        return None
    c = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * c)


def analyze():
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))["items"]
    rng = random.Random(SEED)
    res = {"n_items": len(items), "judges": {}}
    per = {}
    for judge in ("codex", "claude"):
        jd = {}
        for cond, order in TASKS:
            f = OUT / f"verdicts_{judge}_{cond}_{order}.txt"
            if not f.exists():
                continue
            v = parse(f, len(items))
            sc = {}
            for it in items:
                w = v.get(it["item"])
                if w is None:
                    continue
                rev_a = it["rev_is_A_order1"] if order == 1 else not it["rev_is_A_order1"]
                sc[it["item"]] = 0.5 if w == "tie" else float((w == "A") == rev_a)
            jd[(cond, order)] = sc
        per[judge] = jd
        out = {}
        for cond in ("lm", "ag"):
            orders = [jd[k] for k in jd if k[0] == cond]
            if not orders:
                continue
            common = set.intersection(*[set(o) for o in orders])
            s = {i: sum(o[i] for o in orders) / len(orders) for i in common}
            vals = list(s.values())
            boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(10000))
            wins = sum(x > 0.5 for x in vals)
            losses = sum(x < 0.5 for x in vals)
            d = dict(n=len(vals), parsed_per_order=[len(o) for o in orders], rev_score=sum(vals) / len(vals),
                     ci95=[boots[249], boots[9749]], rev_wins=wins, ar_wins=losses, ties=len(vals) - wins - losses,
                     sign_p=sign_p(wins, wins + losses))
            by = {}
            for ds, _ in SETS:
                vv = [s[it["item"]] for it in items if it["dataset"] == ds and it["item"] in s]
                by[ds] = dict(n=len(vv), rev_score=sum(vv) / len(vv) if vv else None)
            d["by_dataset"] = by
            if len(orders) == 2:
                o1, o2 = orders
                both = set(o1) & set(o2)
                d["position_consistency"] = sum(o1[i] == o2[i] for i in both) / len(both)
                d["first_position_win_rate"] = sum(
                    (o1[i] == 1.0) == it["rev_is_A_order1"] for it in items for i in [it["item"]] if i in o1 and o1[i] != 0.5
                ) / max(1, sum(1 for i in o1 if o1[i] != 0.5))
            out[cond] = d
        res["judges"][judge] = out
    agree = {}
    for cond, order in TASKS:
        a, b = per.get("codex", {}).get((cond, order)), per.get("claude", {}).get((cond, order))
        if a and b:
            c = set(a) & set(b)
            agree[f"{cond}_{order}"] = dict(n=len(c), exact_agreement=sum(a[i] == b[i] for i in c) / len(c))
    res["judge_agreement"] = agree
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        {"build": build, "analyze": analyze}[sys.argv[1]]()
