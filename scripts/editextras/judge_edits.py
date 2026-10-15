#!/usr/bin/env python
"""Blind pairwise Codex judge of edit outputs (notes/edit-extras-2026-10-10.md, task 2).

Design (fixed before any verdict existed):
- **Items.** 300 records: 150 CoEdIT validation + 150 IteraTeR human sentence test, each stratified by task (record
  intent) with proportional allocation (largest remainder), sampled with random.Random(SEED). The same 300 items are
  used for every pair. Pair "e300_vs_coedit" uses a 150-item sub-sample (75 + 75, same stratified rule, seed SEED+1).
- **Outputs.** The existing type-greedy pass-3 eval outputs (pred field) of F's runs (pulled eval jsonl, untouched);
  CoEdIT-large: runs/external/grammarly__coedit-large greedy outputs (reviser_v2.evaluate_external).
- **Pairs.** e100_vs_ar100, e300_vs_ar300 (300 items each); e300_vs_coedit (150 items, if time allows).
- **Orders.** Order 1: the edit model is A with probability 1/2 per item (seeded); order 2: swapped. The two orders
  are judged in separate Codex calls. Items whose two outputs are identical (after whitespace normalisation) are not
  sent and count as ties (0.5).
- **Calls.** 50 items per call, items in a seeded random order; one Codex job at a time. Codex CLI with the ChatGPT
  subscription login (OPENAI_API_KEY / CODEX_API_KEY removed from the environment), read-only sandbox, ephemeral, empty
  working directory, the CLI's configured model and reasoning effort (recorded from the log header).
  A usage-limit / auth error stops the run (no retries with other credentials).
- **Score.** Per item, edit-model score 1 win / 0.5 tie / 0 loss, averaged over the two orders. Mean with 95% bootstrap
  CI over items (10,000 resamples, seed 0); exact two-sided sign test on items with a non-0.5 score; position
  consistency (same underlying winner, or tie, in both orders) and the share of "A" verdicts; split by dataset and task.

  python scripts/editextras/judge_edits.py build
  python scripts/editextras/judge_edits.py run e300_vs_ar300 e100_vs_ar100 [e300_vs_coedit]
  python scripts/editextras/judge_edits.py analyze
Outputs: runs/edit_extras/judge/{items.json, calls/<pair>_o<order>_c<k>.{md,txt,log}, analysis.json, analysis.md}.
"""
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "edit_extras" / "judge"
EMPTY = ROOT / "runs" / "edit_extras" / "judge_empty_cwd"
CODEX = "~/AppData/Roaming/npm/codex.cmd"
SEED = 20261010
PER_CALL = 50
MAXCH = 2000
B1 = "runs/edit_seed1_100m/runs"
B3 = "runs/edit_seed1_300m/runs"
SETS = [("coedit", "data/real/coedit_validation.jsonl", "coedit_validation"),
        ("iterater", "data/real/iterater_human_sent_test.jsonl", "iterater_human_sent_test")]
SYSTEMS = {
    "edit100": B1 + "/ev3-main100-s1/keep/pass3/eval/edits_{s}_tgreedy.jsonl",
    "ar100": B1 + "/ev3-ar100-s1/keep/pass3/eval/edits_{s}_tgreedy_ar.jsonl",
    "edit300": B3 + "/ev3-main300-s1/keep/pass3/eval/edits_{s}_tgreedy.jsonl",
    "ar300": B3 + "/ev3-ar300-s1/keep/pass3/eval/edits_{s}_tgreedy_ar.jsonl",
    "coedit_large": "runs/external/grammarly__coedit-large/edits_{s}_greedy.jsonl",
}
PAIRS = {"e100_vs_ar100": ("edit100", "ar100", "main"), "e300_vs_ar300": ("edit300", "ar300", "main"),
         "e300_vs_coedit": ("edit300", "coedit_large", "sub")}

INSTR = """You are judging text edits. Each item has an INSTRUCTION, an INPUT text, and two edited versions of the
input, A and B, produced by two different systems. Decide which version better carries out the instruction on the
input while keeping the meaning of the input (except where the instruction itself asks for a change of content).
Judge whether the instruction was followed, whether the meaning was kept, and whether the result is fluent and
grammatical. Do not prefer a version for being longer or shorter, or for changing more or less, except as the
instruction requires. A version may be identical to the input; judge it like any other. Answer "tie" if both are
equally good or equally bad.

Do not use any tools, do not read files, do not run commands. Output ONLY one JSON object per line, one per item,
in item order, exactly like: {"item": "E001", "winner": "A"} with winner one of "A", "B", "tie". No other text.

"""


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(s):
    return " ".join(s.split())


def stratified(recs, n, rng):
    by = defaultdict(list)
    for r in recs:
        by[r["intent"]].append(r)
    tot = len(recs)
    quota = {k: n * len(v) / tot for k, v in by.items()}
    alloc = {k: int(math.floor(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[:n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k in sorted(by):
        out += rng.sample(by[k], alloc[k])
    return out, alloc


def build():
    rng = random.Random(SEED)
    items, alloc, files = [], {}, {}
    for ds, path, stem in SETS:
        recs = [json.loads(l) for l in open(ROOT / path, encoding="utf-8")]
        files[path] = sha(ROOT / path)
        preds = {}
        for s, pat in SYSTEMS.items():
            p = ROOT / pat.format(s=stem)
            files[str(p.relative_to(ROOT))] = sha(p)
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            assert [r["uid"] for r in rows] == [r["uid"] for r in recs], p
            preds[s] = {r["uid"]: r["pred"] for r in rows}
        pick, al = stratified(recs, 150, rng)
        alloc[ds] = al
        for r in pick:
            items.append(dict(uid=r["uid"], dataset=ds, intent=r["intent"], instruction=r["instruction"],
                              input=r["before_text"], reference=r["after_text"],
                              out={s: preds[s][r["uid"]] for s in SYSTEMS}))
    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it["item"] = f"E{k:03d}"
    # sub-sample for edit300 vs CoEdIT-large: 75 + 75, same stratified rule, own seed
    rng2 = random.Random(SEED + 1)
    sub = set()
    for ds, _, _ in SETS:
        pick, al = stratified([it for it in items if it["dataset"] == ds], 75, rng2)
        sub |= {it["item"] for it in pick}
        alloc[ds + "_sub"] = al
    for it in items:
        it["in_sub"] = it["item"] in sub
    # A/B per pair: the first system is A in order 1 with probability 1/2 (independent draw per pair and item)
    for pair in PAIRS:
        for it in items:
            it.setdefault("first_is_A_o1", {})[pair] = rng.random() < 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(dict(seed=SEED, per_call=PER_CALL, alloc=alloc, files_sha256=files,
                                                    prompt_header=INSTR, items=items), indent=1), encoding="utf-8")
    print(len(items), "items", alloc)


def clip(s):
    s = s if s.strip() else "(empty output)"
    return s if len(s) <= MAXCH else s[:MAXCH] + " [... truncated]"


def call_plan(items, pair):
    x, y, which = PAIRS[pair]
    pool = [it for it in items if (which == "main" or it["in_sub"]) and norm(it["out"][x]) != norm(it["out"][y])]
    calls = []
    k = max(1, math.ceil(len(pool) / PER_CALL))  # k nearly equal chunks of at most PER_CALL items
    bounds = [round(i * len(pool) / k) for i in range(k + 1)]
    for order in (1, 2):
        for c in range(k):
            chunk = pool[bounds[c]:bounds[c + 1]]
            parts = [INSTR]
            for it in chunk:
                x_is_a = it["first_is_A_o1"][pair] if order == 1 else not it["first_is_A_o1"][pair]
                a, b = (it["out"][x], it["out"][y]) if x_is_a else (it["out"][y], it["out"][x])
                parts.append(f"### {it['item']}\nINSTRUCTION: {it['instruction']}\n\nINPUT:\n{clip(it['input'])}\n\n"
                             f"A:\n{clip(a)}\n\nB:\n{clip(b)}\n")
            calls.append((f"{pair}_o{order}_c{c + 1}", "\n".join(parts), [it["item"] for it in chunk]))
    return calls


LIMIT_RE = re.compile(r"usage limit|rate limit|quota|exceeded|unauthori[sz]ed|401|log ?in again|not logged in", re.I)


def run(pairs):
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))["items"]
    (OUT / "calls").mkdir(parents=True, exist_ok=True)
    EMPTY.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")}
    for pair in pairs:
        for name, prompt, ids in call_plan(items, pair):
            out = OUT / "calls" / f"{name}.txt"
            if out.exists() and len(parse(out)) == len(ids):
                continue
            (OUT / "calls" / f"{name}.md").write_text(prompt, encoding="utf-8")
            for attempt in (1, 2):
                t0 = time.time()
                r = subprocess.run([CODEX, "exec", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                                    "--output-last-message", str(out), "-"], input=prompt, capture_output=True, text=True,
                                   encoding="utf-8", env=env, cwd=EMPTY)
                log = (r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or "")
                (OUT / "calls" / f"{name}.log").write_text(log, encoding="utf-8")
                got = parse(out) if out.exists() else {}
                print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {name} attempt {attempt} rc {r.returncode} "
                      f"verdicts {len(got)}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
                if r.returncode != 0 and LIMIT_RE.search(r.stderr or ""):
                    print("STOP: Codex usage limit / auth problem; see", OUT / "calls" / f"{name}.log", flush=True)
                    sys.exit(3)
                if len(got) == len(ids) and set(got) == set(ids):
                    break
            else:
                print("STOP: incomplete verdicts after 2 attempts:", name, flush=True)
                sys.exit(4)
    print("RUN_DONE", flush=True)


def parse(path):
    v = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.search(r'\{[^{}]*"item"\s*:\s*"(E\d{3})"[^{}]*"winner"\s*:\s*"(A|B|tie)"[^{}]*\}', line, re.I)
        if m:
            v[m.group(1)] = m.group(2).upper() if m.group(2).lower() != "tie" else "tie"
    return v


def sign_p(k, n):
    if n == 0:
        return None
    c = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * c)


def boot_ci(xs, n=10000, seed=0):
    rng = random.Random(seed)
    N = len(xs)
    ms = sorted(sum(xs[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def analyze():
    meta = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    items = meta["items"]
    res = {}
    models = Counter()
    for pair, (x, y, which) in PAIRS.items():
        plan = call_plan(items, pair)
        verd = {1: {}, 2: {}}
        ok = True
        for name, _, ids in plan:
            p = OUT / "calls" / f"{name}.txt"
            if not p.exists():
                ok = False
                break
            o = int(name.split("_o")[1][0])
            verd[o].update(parse(p))
            lg = OUT / "calls" / f"{name}.log"
            if lg.exists():
                m = re.search(r"^model: (.+)$", lg.read_text(encoding="utf-8"), re.M)
                e = re.search(r"^reasoning effort: (.+)$", lg.read_text(encoding="utf-8"), re.M)
                models[(m.group(1).strip() if m else "?", e.group(1).strip() if e else "?")] += 1
        if not ok:
            continue
        pool = [it for it in items if which == "main" or it["in_sub"]]
        rows = []
        cons = n_judged = n_a = n_v = 0
        for it in pool:
            if norm(it["out"][x]) == norm(it["out"][y]):
                rows.append(dict(item=it["item"], dataset=it["dataset"], intent=it["intent"], score=0.5, identical=True))
                continue
            sc = []
            winners = []
            for o in (1, 2):
                v = verd[o][it["item"]]
                x_is_a = it["first_is_A_o1"][pair] if o == 1 else not it["first_is_A_o1"][pair]
                n_v += 1
                n_a += v == "A"
                w = "tie" if v == "tie" else (x if (v == "A") == x_is_a else y)
                winners.append(w)
                sc.append(1.0 if w == x else 0.5 if w == "tie" else 0.0)
            n_judged += 1
            cons += winners[0] == winners[1]
            rows.append(dict(item=it["item"], dataset=it["dataset"], intent=it["intent"], score=sum(sc) / 2,
                             identical=False, winners=winners))

        def summ(rs):
            xs = [r["score"] for r in rs]
            if not xs:
                return None
            lo, hi = boot_ci(xs)
            wins = sum(r["score"] > 0.5 for r in rs)
            losses = sum(r["score"] < 0.5 for r in rs)
            return dict(n=len(xs), pref=sum(xs) / len(xs), ci95=[lo, hi], wins=wins, losses=losses,
                        ties=len(xs) - wins - losses, sign_p=sign_p(wins, wins + losses))
        split = {}
        for key in sorted({(r["dataset"], r["intent"]) for r in rows}):
            split[f"{key[0]}/{key[1]}"] = summ([r for r in rows if (r["dataset"], r["intent"]) == key])
        for ds in ("coedit", "iterater"):
            split[ds] = summ([r for r in rows if r["dataset"] == ds])
        res[pair] = dict(first=x, second=y, overall=summ(rows), identical=sum(r["identical"] for r in rows),
                         judged=n_judged, position_consistency=cons / n_judged if n_judged else None,
                         share_A=n_a / n_v if n_v else None, split=split, rows=rows)
    res["_judge"] = {f"{m} / {e}": c for (m, e), c in models.items()}
    json.dump(res, open(OUT / "analysis.json", "w"), indent=1)
    L = []
    for pair, r in res.items():
        if pair.startswith("_"):
            continue
        o = r["overall"]
        L.append(f"**{pair}** ({r['first']} vs {r['second']}): preference for {r['first']} {o['pref']:.3f} "
                 f"[{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}], n={o['n']} (wins {o['wins']}, losses {o['losses']}, "
                 f"ties {o['ties']}; sign test p={o['sign_p']:.2g}); identical outputs {r['identical']}; judged {r['judged']}; "
                 f"position consistency {r['position_consistency']:.3f}; share of 'A' verdicts {r['share_A']:.3f}")
        L += ["", "| split | n | preference | 95% CI | wins / losses / ties |", "|---|---|---|---|---|"]
        for k, s in r["split"].items():
            if s:
                L.append(f"| {k} | {s['n']} | {s['pref']:.3f} | [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                         f"{s['wins']} / {s['losses']} / {s['ties']} |")
        L.append("")
    L.append(f"Judge (from the Codex log headers): {res['_judge']}")
    (OUT / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        run(sys.argv[2:])
    elif cmd == "analyze":
        analyze()
