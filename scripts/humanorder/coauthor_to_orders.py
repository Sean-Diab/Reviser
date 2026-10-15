"""CoAuthor keystroke logs -> human writing orders over GPT-2 tokens of the final text, cut into Reviser windows.

Replays every session's Quill text deltas and tags each character with the event that typed it (prompt text and
AI-suggestion text are tagged separately). Text deleted before the end is dropped: only the final text counts. Each
GPT-2 token of the final text gets the time it was completed (the latest event among its characters). Sorting the
tokens by (time, position) gives the human order over final tokens.

Windows match the classical Reviser's data: a 35-token prefix (already written) and a continuation of 65-315 tokens
(sampled per window) whose tokens contain no CoAuthor prompt text. Windows tile each session; the prefix of a window
is the 35 tokens before its continuation. The order inside a window is the human order restricted to its
continuation tokens.

  python scripts/humanorder/coauthor_to_orders.py --zip COAUTHOR.zip --out_dir data/humanorder [--seed 0]

Outputs (gitignored): sessions.jsonl (per session: final tokens, times, sources), windows_{train,dev,test}.jsonl
({sid, w, prefix, cont, order, src}), summary.json. Split 80/10/10 by sha256 of the session id (the zip has no
writer IDs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from orders import order_to_actions, run_lengths  # noqa: E402

PREFIX = 35
CONT_MIN, CONT_MAX = 65, 315


def replay(lines):
    """Return the final document as a list of (char, time, src) with src in {'p','u','a'}."""
    doc = []
    for line in lines:
        e = json.loads(line)
        if e["eventName"] == "system-initialize":
            doc = [(c, -1, "p") for c in (e.get("currentDoc") or "")]
            continue
        d = e.get("textDelta")
        if not d or not isinstance(d, dict) or "ops" not in d:
            continue
        src = "u" if e.get("eventSource") == "user" else "a"
        pos, t = 0, int(e["eventNum"])
        for op in d["ops"]:
            if "retain" in op:
                pos += op["retain"]
            elif "insert" in op:
                s = op["insert"] if isinstance(op["insert"], str) else ""
                doc[pos:pos] = [(c, t, src) for c in s]
                pos += len(s)
            elif "delete" in op:
                del doc[pos:pos + op["delete"]]
    while doc and doc[-1][0].isspace():   # Quill keeps a trailing newline
        doc.pop()
    return doc


def tokenize_with_times(doc, tok):
    text = "".join(c for c, _, _ in doc)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, times, srcs = [], [], []
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        ch = doc[a:b] if b > a else doc[max(0, a - 1):a]
        ts = [t for _, t, _ in ch]
        ss = {s for _, _, s in ch}
        ids.append(int(tid))
        times.append(max(ts) if ts else -1)
        srcs.append("p" if ss == {"p"} else ("a" if "a" in ss else "u"))
    return text, ids, times, srcs


def split_of(sid: str) -> str:
    h = int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else ("dev" if h == 1 else "train")


def windows(ids, times, srcs, rng):
    """Tile the session after its last prompt token into (prefix, cont, order, src) windows."""
    last_p = max([i for i, s in enumerate(srcs) if s == "p"], default=-1)
    start = max(last_p + 1, PREFIX)
    out = []
    while start + CONT_MIN <= len(ids):
        c = min(rng.randint(CONT_MIN, CONT_MAX), len(ids) - start)
        prefix, cont = ids[start - PREFIX:start], ids[start:start + c]
        order = sorted(range(c), key=lambda j: (times[start + j], j))
        out.append({"prefix": prefix, "cont": cont, "order": order, "src": "".join(srcs[start:start + c])})
        start += c
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    z = zipfile.ZipFile(args.zip)
    names = sorted(n for n in z.namelist() if n.endswith(".jsonl"))
    fw = {s: open(os.path.join(args.out_dir, f"windows_{s}.jsonl"), "w", encoding="utf-8")
          for s in ("train", "dev", "test")}
    stats = {s: {"sessions": 0, "windows": 0, "tokens": 0, "ai_tokens": 0, "runs": [], "jumps": [],
                 "actions": 0} for s in fw}
    with open(os.path.join(args.out_dir, "sessions.jsonl"), "w", encoding="utf-8") as fs:
        for n in names:
            sid = os.path.basename(n)[:-len(".jsonl")]
            doc = replay(z.read(n).decode("utf-8").splitlines())
            text, ids, times, srcs = tokenize_with_times(doc, tok)
            sp = split_of(sid)
            fs.write(json.dumps({"sid": sid, "split": sp, "ids": ids, "times": times, "src": "".join(srcs)}) + "\n")
            st = stats[sp]
            st["sessions"] += 1
            for w_i, w in enumerate(windows(ids, times, srcs, rng)):
                fw[sp].write(json.dumps({"sid": sid, "w": w_i, **w}) + "\n")
                runs, jumps = run_lengths(w["order"])
                st["windows"] += 1
                st["tokens"] += len(w["cont"])
                st["ai_tokens"] += w["src"].count("a")
                st["runs"] += runs
                st["jumps"] += jumps
                st["actions"] += len(order_to_actions(w["prefix"], w["cont"], w["order"])) - PREFIX - 1
    for f in fw.values():
        f.close()
    summary = {}
    for s, st in stats.items():
        j = st["jumps"]
        summary[s] = {
            "sessions": st["sessions"], "windows": st["windows"], "cont_tokens": st["tokens"],
            "ai_token_share": round(st["ai_tokens"] / max(1, st["tokens"]), 4),
            "run_len_mean": round(statistics.mean(st["runs"]), 2) if st["runs"] else None,
            "run_len_median": statistics.median(st["runs"]) if st["runs"] else None,
            "jumps_per_100_tokens": round(100 * len(j) / max(1, st["tokens"]), 3),
            "back_jump_share": round(sum(x < 0 for x in j) / max(1, len(j)), 4),
            "jump_dist_median": statistics.median([abs(x) for x in j]) if j else None,
            "actions_per_output_token": round(st["actions"] / max(1, st["tokens"]), 4),
        }
    summary["config"] = {"zip": os.path.abspath(args.zip), "prefix": PREFIX, "cont_range": [CONT_MIN, CONT_MAX],
                         "seed": args.seed, "tokenizer": "data/tokenizers/gpt2"}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
