#!/usr/bin/env python3
"""Claude / Codex CLI judges, model vs AR 100M at matched length (helper E, notes/paper-retest-2026-10-08.md).

Same blinded design and instruction as scripts/judge_continuations.py (task 28), applied to helper E's decodes:
- **Items.** n prompts sampled once from the 3000 (seed 20261008); the same ids are used for every model.
- **Texts.** Both final canvases cut to prefix + min(out_len_x, out_len_ar) new GPT-2 tokens (tiktoken gpt2 decode).
  The 35-token prefix is shown separately.
- **Orders.** Order 1 = random A/B per item (seeded); order 2 = swapped. Each order goes in separate calls of
  --chunk items.
- **Score.** Per item, the model scores 1 for a win, 0.5 for a tie and 0 for a loss, averaged over the two orders.
  We report the mean with a 95% bootstrap CI over items and an exact sign test on non-tied items.
- **Logins.** Subscription only: Claude via scripts/claude-subscription.ps1 (refuses API-key auth) and Codex via
  its ChatGPT login (OPENAI_API_KEY / CODEX_API_KEY removed from the environment).
- **Stops.** On a usage-limit or auth error the judge stops: a STOPPED_<judge>.txt file records it and nothing falls
  back to an API key.

  python scripts/papereval/judge_cli.py build --tag claude100 --n 100 --models g8m2jd,ov_jl90
  python scripts/papereval/judge_cli.py run --tag claude100 --judge claude --chunk 50
  python scripts/papereval/judge_cli.py analyze --tag claude100
Decode files are read from runs/papereval/dec/c4paper3k.{lock.<model>|nolock.ar100}.jsonl
(pulled from /root/rv2/E/dec/).
build --full (added 2026-10-08 14:20Z): both texts are the full final canvases, not cut to a matched length (the
Skywork "paper protocol" texts; both orders as above). Same items, instruction and scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[2]
DEC = ROOT / "runs" / "papereval" / "dec"
OUTR = ROOT / "runs" / "papereval" / "judge"
EMPTY = ROOT / "tmp" / "judge_empty_cwd"
SEED = 20261008
PREFIX = 35
INSTR = """You are judging text continuations. For each item you get a PREFIX (the start of a document) and two
complete texts, A and B, each of which begins with (approximately) that prefix and continues it. Decide which text
is the better continuation of the prefix, judging:
coherence and consistency with the prefix, fluency and grammaticality, and absence of degenerate repetition or
nonsense. Both texts may be cut off mid-sentence; do not penalize an abrupt ending, and do not prefer a text for
being longer or shorter. Answer "tie" only if you truly cannot prefer one.

Do not use any tools, do not read files, do not run commands. Output ONLY one JSON object per line, one per item,
in item order, exactly like: {"item": "J001", "winner": "A"} with winner one of "A", "B", "tie". No other text.

"""
LIMIT_RX = re.compile(r"usage limit|rate limit|limit reached|quota|exceeded|not logged in|login|unauthor|401|403",
                      re.I)


def load(stem):
    rows = [json.loads(l) for l in open(DEC / f"{stem}.jsonl", encoding="utf-8")]
    return {r["id"]: r for r in rows[1:]}


def build(a):
    enc = tiktoken.get_encoding("gpt2")
    out = OUTR / a.tag
    ar = load("c4paper3k.nolock.ar100")
    ids = sorted(ar)
    pick = random.Random(SEED).sample(ids, a.n)
    for m in a.models.split(","):
        x = load(f"c4paper3k.lock.{m}")
        rng = random.Random(f"{SEED}-{m}")
        items = []
        for k, i in enumerate(pick, 1):
            r, b = x[i], ar[i]
            assert r["prefix_ids"] == b["prefix_ids"]
            L = None if a.full else min(r["out_len"], b["out_len"])
            items.append({"item": f"J{k:03d}", "id": i, "matched_new": L,
                          "x_out_len": r["out_len"], "ar_out_len": b["out_len"],
                          "x_is_A_order1": rng.random() < 0.5, "prefix": enc.decode(r["prefix_ids"]),
                          "x": enc.decode(r["canvas_ids"][:PREFIX + L] if L is not None else r["canvas_ids"]),
                          "ar": enc.decode(b["canvas_ids"][:PREFIX + L] if L is not None else b["canvas_ids"])})
        d = out / m
        d.mkdir(parents=True, exist_ok=True)
        (d / "items.json").write_text(json.dumps({"seed": SEED, "model": m, "base": "ar100",
                                                  "texts": "full" if a.full else "matched", "items": items},
                                                 indent=1), encoding="utf-8")
        print(m, len(items), "items ->", d)


def prompts(items, order, chunk):
    for c in range(0, len(items), chunk):
        parts = [INSTR]
        for it in items[c:c + chunk]:
            xa = it["x_is_A_order1"] if order == 1 else not it["x_is_A_order1"]
            A, B = (it["x"], it["ar"]) if xa else (it["ar"], it["x"])
            parts.append(f"### {it['item']}\nPREFIX:\n{it['prefix']}\n\nA:\n{A}\n\nB:\n{B}\n")
        yield c // chunk, "\n".join(parts)


def call(judge, text, d, tag):
    """-> (ok, verdict_text, error_or_None)"""
    EMPTY.mkdir(parents=True, exist_ok=True)
    if judge == "claude":
        cmd = ["powershell.exe", "-NoProfile", "-File", str(ROOT / "scripts/claude-subscription.ps1"), "-p",
               "--model", "claude-opus-5-5", "--no-session-persistence", "--output-format", "json",
               "--disallowedTools", "Bash,Edit,Write,NotebookEdit,Agent,WebFetch,WebSearch,Read,Grep,Glob"]
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", cwd=EMPTY,
                           timeout=1800)
        try:
            j = json.loads(r.stdout)
        except Exception:
            return False, "", f"rc {r.returncode}: {(r.stdout or '')[-400:]} {(r.stderr or '')[-400:]}"
        (d / f"meta_{tag}.json").write_text(json.dumps(
            {k: j.get(k) for k in ("is_error", "modelUsage", "duration_ms", "session_id")}, indent=1))
        if j.get("is_error"):
            return False, "", str(j.get("result"))[:1000]
        return True, j.get("result", ""), None
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")}
    env["PATH"] = env.get("PATH", "") + os.pathsep + r"~\AppData\Roaming\npm"
    last = d / f"last_{tag}.txt"
    r = subprocess.run(["~/AppData/Roaming/npm/codex.cmd", "exec", "--sandbox", "read-only",
                        "--ephemeral", "--skip-git-repo-check", "--output-last-message", str(last), "-"],
                       input=text, capture_output=True, text=True, encoding="utf-8", env=env, cwd=EMPTY, timeout=1800)
    (d / f"log_{tag}.txt").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    v = last.read_text(encoding="utf-8") if last.exists() else ""
    if r.returncode != 0 or not v.strip():
        return False, v, f"rc {r.returncode}: {(r.stderr or r.stdout or '')[-600:]}"
    return True, v, None


def login_ok(judge):
    if judge == "claude":
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                                                                 "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_API_KEY")}
        r = subprocess.run(["claude", "auth", "status", "--json"], capture_output=True, text=True, env=env, shell=True)
        try:
            s = json.loads(r.stdout)
            return s.get("loggedIn") and s.get("authMethod") == "claude.ai" and not s.get("apiKeySource")
        except Exception:
            return False
    r = subprocess.run(["~/AppData/Roaming/npm/codex.cmd", "login", "status"], capture_output=True,
                       text=True)
    return "ChatGPT" in (r.stdout + r.stderr)


def run(a):
    out = OUTR / a.tag
    stop = out / f"STOPPED_{a.judge}.txt"
    if stop.exists():
        print("stopped earlier:", stop.read_text())
        return
    if not login_ok(a.judge):
        stop.write_text("subscription login check failed before the run; no API-key fallback")
        print("login check failed")
        return
    models = a.models.split(",") if a.models else sorted(p.name for p in out.iterdir() if p.is_dir())
    for m in models:
        d = out / m
        items = json.loads((d / "items.json").read_text(encoding="utf-8"))["items"]
        for order in (1, 2):
            for c, text in prompts(items, order, a.chunk):
                tag = f"{a.judge}_o{order}_c{c}_k{a.chunk}"
                vf = d / f"verdicts_{tag}.txt"
                if vf.exists() and vf.stat().st_size > 0:
                    continue
                ok, v, err = call(a.judge, text, d, tag)
                if ok:
                    vf.write_text(v, encoding="utf-8")
                    print(m, tag, "ok", len(parse(v)), "verdicts", flush=True)
                else:
                    (d / f"error_{tag}.txt").write_text(str(err), encoding="utf-8")
                    print(m, tag, "ERROR", str(err)[:300], flush=True)
                    if LIMIT_RX.search(str(err)):
                        stop.write_text(f"{m} {tag}: {str(err)[:800]}")
                        print("judge stopped (limit/auth); recorded in", stop)
                        return


def parse(text):
    v = {}
    for line in text.splitlines():
        mm = re.search(r'\{[^{}]*"item"\s*:\s*"(J\d{3})"[^{}]*"winner"\s*:\s*"(A|B|tie)"[^{}]*\}', line, re.I)
        if mm:
            v[mm.group(1)] = mm.group(2).upper() if mm.group(2).lower() != "tie" else "tie"
    return v


def sign_p(k, n):
    if n == 0:
        return None
    c = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * c)


def analyze(a):
    out = OUTR / a.tag
    res = {}
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        items = json.loads((d / "items.json").read_text(encoding="utf-8"))["items"]
        byj = {}
        for vf in d.glob("verdicts_*.txt"):
            judge, o = vf.stem.split("_")[1], int(vf.stem.split("_")[2][1:])
            byj.setdefault(judge, {}).setdefault(o, {}).update(parse(vf.read_text(encoding="utf-8")))
        for judge, orders in byj.items():
            sc = {}
            for o, v in orders.items():
                for it in items:
                    w = v.get(it["item"])
                    if w is None:
                        continue
                    xa = it["x_is_A_order1"] if o == 1 else not it["x_is_A_order1"]
                    sc.setdefault(it["item"], {})[o] = 0.5 if w == "tie" else float((w == "A") == xa)
            both = {k: (s[1] + s[2]) / 2 for k, s in sc.items() if 1 in s and 2 in s}
            vals = list(both.values())
            if not vals:
                continue
            rng = random.Random(SEED)
            boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(10000))
            wins, losses = sum(x > 0.5 for x in vals), sum(x < 0.5 for x in vals)
            pc = sum(sc[k][1] == sc[k][2] for k in both) / len(both)
            res.setdefault(d.name, {})[judge] = {
                "n_items": len(items), "n_both_orders": len(vals), "x_score": sum(vals) / len(vals),
                "ci95": [boots[249], boots[9749]], "x_wins": wins, "ar_wins": losses, "ties": len(vals) - wins - losses,
                "sign_p": sign_p(wins, wins + losses), "position_consistency": pc,
                "texts": "full" if items[0]["matched_new"] is None else "matched",
                "mean_matched_new": (sum(it["matched_new"] for it in items) / len(items)
                                     if items[0]["matched_new"] is not None else None),
                "mean_x_new": sum(it["x_out_len"] for it in items) / len(items),
                "mean_ar_new": sum(it["ar_out_len"] for it in items) / len(items)}
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "analyze"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", default="")
    ap.add_argument("--judge", choices=["claude", "codex"])
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--full", action="store_true", help="build: full texts instead of matched length")
    a = ap.parse_args()
    {"build": build, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
