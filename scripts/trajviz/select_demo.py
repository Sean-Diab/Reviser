#!/usr/bin/env python3
"""Candidate pool for the phone demo videos of Reviser trajectories (notes/demo-videos-2026-10-09.md).

Stage 1 of the selection (the pool). Stages 2 and 3 (10 best, then 3 best) are made by reading the replays this script
writes; the picks and their reasons are recorded in configs/trajviz/demo_picks.json and the note.

Sources and filters: configs/trajviz/demo_videos.json. Every trajectory is replayed with scripts/trajviz/trajstats.py
(the same replay as the trajectory viewer) and must reproduce its stored final text (canvas_ids for the classical
decodes, `final` for the edit decodes); a row that does not is dropped and counted. Hard filters come first, then each
source's passing rows are shuffled with random.Random(seed + source index) and the first `quota` rows form the pool.
Nothing is ranked by quality here.

  python scripts/trajviz/select_demo.py                       # pool from the sources with a path
  python scripts/trajviz/select_demo.py --new300 PATH         # add the new delete+RL 300M decode (uses quota_with_new)

Outputs (runs/demo_videos/selection/, not in Git): pool.json (pool rows with everything the renderer needs),
filter_counts.json, pool_readout.txt (one readable replay per pool row, for stages 2 and 3).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "trajviz"))
import trajstats as T  # noqa: E402

CFG = ROOT / "configs" / "trajviz" / "demo_videos.json"
OUT = ROOT / "runs" / "demo_videos" / "selection"
PREFIX_LEN = 35
EXTRA_OK_CHARS = set("’‘“”–—…éèáóíúñüöä°£")


# ---- tokens -------------------------------------------------------------------------------------------------------
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tok:
    """GPT-2 token -> bytes, from the local tokenizer files (data/tokenizers/gpt2/vocab.json); no network."""

    def __init__(self):
        vocab = json.loads((ROOT / "data" / "tokenizers" / "gpt2" / "vocab.json").read_text(encoding="utf-8"))
        dec = {v: k for k, v in _bytes_to_unicode().items()}
        self.b = {i: bytes(dec[ch] for ch in s) for s, i in vocab.items()}

    def piece(self, t: int) -> bytes:
        return self.b[int(t)]

    def text(self, ids) -> str:
        return b"".join(self.b[int(t)] for t in ids if int(t) < 50257).decode("utf-8", errors="replace")

    def clean_piece(self, t: int):
        """The token as a str when it is a whole, displayable UTF-8 string, else None."""
        try:
            s = self.b[int(t)].decode("utf-8")
        except UnicodeDecodeError:
            return None
        if all((32 <= ord(c) < 127) or c == "\n" or c in EXTRA_OK_CHARS for c in s):
            return s
        return None


# ---- pace (shared with render_video.py) ------------------------------------------------------------------------------
PACE = {
    # seconds per action; inserts right after a jump are slower so the eye can find the new cursor spot
    "classical": dict(intro=2.4, ins=0.15, ins_after_jump=0.34, n_after_jump=2, dele=0.65, move=0.30, run_hold=0.55,
                      stop=0.9, outro=3.6, ins_min=0.125, ins_max=0.2, target=(22.0, 44.0)),
    "edit": dict(intro=3.2, ins=0.34, ins_after_jump=0.45, n_after_jump=1, dele=0.48, move=0.38, run_hold=0.6,
                 stop=1.0, outro=6.0, ins_min=0.3, ins_max=0.6, target=(21.0, 44.0)),
}


def action_durations(ops, kind, ins_s=None):
    p = PACE[kind]
    ins_s = p["ins"] if ins_s is None else ins_s
    out, since_jump = [], 99
    for k, op in enumerate(ops):
        if op[0] == "i":
            d = p["ins_after_jump"] if since_jump < p["n_after_jump"] else ins_s
            since_jump += 1
        elif op[0] == "d":
            d = p["dele"]
            since_jump = 99
        elif op[0] == "m":
            d = p["move"]
            if k + 1 >= len(ops) or ops[k + 1][0] != "m":
                d += p["run_hold"]
                since_jump = 0
        else:  # stop
            d = p["stop"]
        out.append(d)
    return out


def fit_pace(ops, kind):
    """Pick an insert speed (and, for short edit runs, a slow-down) so the video lands inside the target length.
    Returns (durations per action, intro s, outro s, total s)."""
    p = PACE[kind]
    lo, hi = p["target"]
    dur = action_durations(ops, kind)
    total = p["intro"] + sum(dur) + p["outro"]
    if total > hi:  # speed inserts up, no faster than ins_min
        n_plain = sum(1 for d, op in zip(dur, ops) if op[0] == "i" and abs(d - p["ins"]) < 1e-9)
        if n_plain:
            ins_s = max(p["ins_min"], p["ins"] - (total - hi) / n_plain)
            dur = action_durations(ops, kind, ins_s=ins_s)
            total = p["intro"] + sum(dur) + p["outro"]
    elif total < lo:  # slow everything down, at most 1.6x, then lengthen the final hold
        scale = min(1.6, (lo - p["intro"] - p["outro"]) / max(1e-6, sum(dur)))
        dur = [d * scale for d in dur]
        total = p["intro"] + sum(dur) + p["outro"]
    outro = p["outro"] + max(0.0, lo - total) if total < lo else p["outro"]
    total = p["intro"] + sum(dur) + outro
    return dur, p["intro"], outro, total


# ---- per-row analysis --------------------------------------------------------------------------------------------------
def tok_lev(a, b):
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def move_runs(ops):
    """[(start index, end index exclusive, net, list of deltas)] for maximal runs of consecutive moves."""
    runs, i = [], 0
    while i < len(ops):
        if ops[i][0] != "m":
            i += 1
            continue
        j = i
        while j < len(ops) and ops[j][0] == "m":
            j += 1
        ds = [o[1] for o in ops[i:j]]
        runs.append((i, j, sum(ds), ds))
        i = j
    return runs


def analyse(tok: Tok, start, ops, lo, stored_final, kind):
    """Replay + the demo measures. start = prompt (classical) or draft (edit)."""
    ev = T.replay(start, ops, lo)
    fin = T.final_tokens(start, ops)
    s = T.sample_stats(start, ops, lo)
    runs = move_runs(ops)
    back = [r for r in runs if r[2] < 0]
    wander = [r for r in runs if r[2] == 0 or sum(1 for a, b in zip(r[3], r[3][1:]) if (a > 0) != (b > 0)) >= 2]
    mid_ins = sum(1 for e in ev if e["op"] == "i" and not e["at_end"])
    # (text, cursor) states including the start state. Excluded: the stop action (it repeats the last state) and a
    # backspace right after typing (type X, delete X returns to the state before X by design; retyping the same X is
    # caught separately as churn).
    visits = Counter([hash((tuple(start), len(start)))] + [
        e["state"] for i, e in enumerate(ev)
        if e["op"] != "s" and not (e["op"] == "d" and i > 0 and ev[i - 1]["op"] == "i")])
    draft_dels = sum(1 for e in ev if e["op"] == "d" and not e.get("illegal") and e["born"] is None)
    # displayable: prompt/draft and final text use only characters the video fonts have; a token split inside a
    # character (e.g. a curly quote over two tokens) is fine because the renderer joins such pieces, but every deleted
    # token must be whole so its struck-through ghost can be drawn
    full = tok.text(list(start)) + tok.text(fin)
    dels_ok = all(tok.clean_piece(e["tok"]) is not None for e in ev if e["op"] == "d" and not e.get("illegal"))
    pieces_ok = dels_ok and "�" not in full and all(
        (32 <= ord(c) < 127) or c == "\n" or c in EXTRA_OK_CHARS for c in full)
    dur, intro, outro, total = fit_pace(ops, kind)
    text = tok.text(fin)
    return dict(
        replay_ok=fin == list(stored_final), stopped=bool(ops) and ops[-1][0] == "s",
        n_ins=s["n_ins"], n_del=s["n_del"], n_move=s["n_move"], n_actions=len(ops),
        back_jumps=len(back), fwd_jumps=sum(1 for r in runs if r[2] > 0), wander_jumps=len(wander),
        mid_ins=mid_ins, draft_dels=draft_dels, same_back=s["del_same_back"], reinsert5=s["del_reinsert5"],
        max_state_visits=max(visits.values()), pingpong=s["move_pingpong"], pieces_ok=pieces_ok,
        chars=len(text), newlines=text.count("\n"), est_seconds=round(total, 1), jump_sizes=[r[2] for r in runs],
    )


def ops_of(actions):
    ops = []
    for t in actions:
        op = T.july_op(t)
        ops.append(op)
        if op[0] == "s":
            break
    return ops


# ---- readable replay (for reading the pool) ------------------------------------------------------------------------------
def story(tok: Tok, start, ops, lo):
    """Revision events: each cursor jump with its context and what was typed/deleted there, plus every backspace made
    while writing at the end."""
    toks, cur, lines, k = list(start), len(start), [], 0
    vis = lambda ids: tok.text(ids).replace("\n", "\\n")
    while k < len(ops):
        op = ops[k]
        if op[0] == "m":
            j, c0 = k, cur
            while j < len(ops) and ops[j][0] == "m":
                cur = max(0, min(len(toks), cur + ops[j][1]))
                j += 1
            net = cur - c0
            ctx = (vis(toks[max(lo, cur - 7):cur]), vis(toks[cur:cur + 5]))
            typed, deleted, own, m = [], [], 0, j
            while m < len(ops) and ops[m][0] in "id":
                o = ops[m]
                if o[0] == "i":
                    toks.insert(cur, o[1]); cur += 1; typed.append(o[1])
                elif cur > lo:
                    t = toks.pop(cur - 1); cur -= 1
                    if typed:
                        typed.pop(); own += 1
                    else:
                        deleted.insert(0, t)
                m += 1
            tt = vis(typed)
            if len(tt) > 70:
                tt = tt[:70] + "..."
            lines.append(f"  step {k:>3} {('<-' if net < 0 else '->') + str(abs(net)):>5} at \"...{ctx[0]}|{ctx[1]}...\""
                         + (f"  DEL \"{vis(deleted)}\"" if deleted else "") + (f"  (deleted own typing x{own})" if own else "")
                         + (f"  TYPE \"{tt}\"" if typed else ""))
            k = m
            continue
        if op[0] == "i":
            toks.insert(cur, op[1]); cur += 1
        elif op[0] == "d" and cur > lo:
            prev_typed = k > 0 and ops[k - 1][0] == "i"
            nxt = f" -> next \"{vis([ops[k + 1][1]])}\"" if k + 1 < len(ops) and ops[k + 1][0] == "i" else ""
            lines.append(f"  step {k:>3} backspace \"{vis([toks[cur - 1]])}\"" + (" (just typed)" if prev_typed else "") + nxt)
            toks.pop(cur - 1); cur -= 1
        k += 1
    return lines


def marked_final(tok: Tok, start, ops, lo):
    """Final text with tokens that were inserted mid-text (cursor not at the end) wrapped in [[ ]]."""
    toks = [(t, False) for t in start]
    cur = len(toks)
    for op in ops:
        if op[0] == "i":
            toks.insert(cur, (op[1], cur < len(toks))); cur += 1
        elif op[0] == "d" and cur > 0:
            toks.pop(cur - 1); cur -= 1
        elif op[0] == "m":
            cur = max(0, min(len(toks), cur + op[1]))
        elif op[0] == "s":
            break
    out, inside = [], False
    for t, m in toks[lo:]:
        if m and not inside:
            out.append("[["); inside = True
        if not m and inside:
            out.append("]]"); inside = False
        out.append(tok.text([t]))
    if inside:
        out.append("]]")
    return "".join(out)


# ---- sources -------------------------------------------------------------------------------------------------------------
def load_classical(path):
    rows, meta = [], None
    with open(ROOT / path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
                continue
            rows.append(r)
    return meta, rows


def load_edit(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return None, [json.loads(line) for line in f]


def check_classical(tok, r, F):
    ops = ops_of(r["actions_new"])
    prefix = list(r["prefix_ids"])
    a = analyse(tok, prefix, ops, len(prefix), r["canvas_ids"], "classical")
    a["prompt_intact"] = bool(r.get("prefix_intact", True)) and list(r["canvas_ids"][:len(prefix)]) == prefix
    a["new_tokens"] = len(r["canvas_ids"]) - len(prefix)
    reasons = []
    if not a["replay_ok"]:
        reasons.append("replay mismatch")
    if F["stopped_by_itself"] and not (a["stopped"] and r.get("stopped_eor", True) and not r.get("hit_budget", False)):
        reasons.append("did not stop by itself")
    if F["prompt_intact"] and not a["prompt_intact"]:
        reasons.append("prompt changed")
    if a["back_jumps"] < F["min_backward_jumps"] or a["mid_ins"] < F["min_mid_text_inserts"]:
        reasons.append("too little cursor use")
    if a["max_state_visits"] > F["max_state_visits"] or a["pingpong"] > F["max_move_pingpong"] or a["wander_jumps"] > F["max_wander_jumps"]:
        reasons.append("loop or jitter")
    if a["same_back"] > F["max_same_token_back_after_delete"] or a["reinsert5"] > F["max_same_token_back_after_delete"]:
        reasons.append("churn (deleted token typed back)")
    if a["new_tokens"] < F["min_new_tokens"] or a["chars"] > F["max_total_chars"] or a["newlines"] > F["max_newlines"]:
        reasons.append("length does not fit the screen")
    if F["printable_ascii_only"] and not a["pieces_ok"]:
        reasons.append("characters that do not render")
    lo_s, hi_s = F["duration_s"]
    if not lo_s <= a["est_seconds"] <= hi_s:
        reasons.append("video length out of range")
    return ops, a, reasons


def check_edit(tok, r, F):
    ops = ops_of(r["actions"])
    a = analyse(tok, r["before_ids"], ops, 0, r["final"], "edit")
    fin = T.final_tokens(r["before_ids"], ops)
    a["exact"] = fin == list(r["after_ids"])
    a["dist_ref"] = tok_lev(fin, r["after_ids"])
    a["dist_draft_ref"] = tok_lev(r["before_ids"], r["after_ids"])
    es = T.edit_sample_stats(r["before_ids"], ops, r["after_ids"])
    a["extra_content_actions"] = es["extra_content_actions"]
    a["net_regions"] = es["net_regions"]
    a["draft_tokens"] = len(r["before_ids"])
    reasons = []
    if not a["replay_ok"] or not r.get("replay_ok", True):
        reasons.append("replay mismatch")
    if F["stopped_by_itself"] and not (a["stopped"] and r.get("stopped", True)):
        reasons.append("did not stop by itself")
    if F["changed_draft"] and fin == list(r["before_ids"]):
        reasons.append("left the draft unchanged")
    if a["dist_ref"] >= a["dist_draft_ref"]:
        reasons.append("not close to the reference fix")
    if a["extra_content_actions"] > F["max_extra_content_actions"] or a["same_back"] or a["reinsert5"]:
        reasons.append("wasted edits (churn)")
    if a["max_state_visits"] > F["max_state_visits"] or a["pingpong"]:
        reasons.append("loop or jitter")
    if a["back_jumps"] < F["min_backward_jumps"] or a["draft_dels"] < F["min_draft_deletes"]:
        reasons.append("too little cursor use")
    if a["draft_tokens"] > F["max_draft_tokens"]:
        reasons.append("length does not fit the screen")
    if F["printable_ascii_only"] and not a["pieces_ok"]:
        reasons.append("characters that do not render")
    lo_s, hi_s = F["duration_s"]
    if not lo_s <= a["est_seconds"] <= hi_s:
        reasons.append("video length out of range")
    return ops, a, reasons


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new300", default=None, help="path (repo-relative) of the new delete+RL 300M locked decode")
    ap.add_argument("--new300-label", default=None, help="override the new300 label (e.g. for the no-RL decode)")
    ap.add_argument("--new300-short", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    tok = Tok()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with_new = args.new300 is not None
    pool, counts, readout = [], {}, []
    for si, src in enumerate(cfg["sources"]):
        src = dict(src)
        if src["key"] == "new300":
            if not with_new:
                continue
            src["path"] = args.new300
            src["label"] = args.new300_label or src["label"]
            src["short"] = args.new300_short or src["short"]
        if not src.get("path"):
            continue
        quota = src["quota_with_new"] if with_new else src["quota"]
        kind = src["kind"]
        F = cfg["filters"][kind]
        meta, rows = (load_classical if kind == "classical" else load_edit)(src["path"])
        why = Counter()
        passing = []
        for r in rows:
            ops, a, reasons = (check_classical if kind == "classical" else check_edit)(tok, r, F)
            for x in reasons:
                why[x] += 1
            if not reasons:
                passing.append((r, ops, a))
        rng = random.Random(cfg["seed"] + si)
        order = list(range(len(passing)))
        rng.shuffle(order)
        chosen = [passing[i] for i in order[:quota]]
        counts[src["key"]] = dict(label=src["label"], path=src["path"], sha256=sha256_file(ROOT / src["path"]), rows=len(rows),
                                  passing=len(passing), quota=quota, in_pool=len(chosen), failed_by_reason=dict(why),
                                  ckpt_sha256=(meta or {}).get("ckpt_sha256"))
        for r, ops, a in chosen:
            if kind == "classical":
                cid = f"{src['key']}:{r['id']}:s{r['seed']}"
                row = dict(cid=cid, source=src["key"], kind=kind, model_label=src["label"], short=src["short"],
                           item=r["id"], seed=r["seed"], start_ids=list(r["prefix_ids"]), lo=len(r["prefix_ids"]),
                           actions=list(r["actions_new"][:len(ops)]), final_ids=list(r["canvas_ids"]), stats=a)
                start, lo = r["prefix_ids"], len(r["prefix_ids"])
            else:
                cid = f"{src['key']}:{r['uid']}"
                instr = tok.text([t for t in r["instruction_ids"] if t < 50257])
                row = dict(cid=cid, source=src["key"], kind=kind, model_label=src["label"], short=src["short"],
                           item=r["uid"], intent=r["intent"], task=r["task"], instruction=instr,
                           start_ids=list(r["before_ids"]), lo=0, actions=list(r["actions"][:len(ops)]),
                           final_ids=list(r["final"]), reference_ids=list(r["after_ids"]), stats=a)
                start, lo = r["before_ids"], 0
            pool.append(row)
            st = a
            head = (f"=== {cid}  [{src['label']}]  actions {st['n_actions']} (type {st['n_ins']}, move {st['n_move']}, "
                    f"delete {st['n_del']}); back jumps {st['back_jumps']}, mid-text inserts {st['mid_ins']}; "
                    f"est {st['est_seconds']} s")
            readout.append(head)
            if kind == "classical":
                readout.append("PROMPT: " + tok.text(start).replace("\n", "\\n"))
                readout.append("TEXT ([[ ]] = typed after jumping back): " + marked_final(tok, start, ops, lo).replace("\n", "\\n"))
            else:
                readout.append(f"TASK ({r['intent']}): {row['instruction']}")
                readout.append("DRAFT: " + tok.text(start).replace("\n", "\\n"))
                readout.append("FINAL: " + tok.text(row["final_ids"]).replace("\n", "\\n"))
                readout.append("REF:   " + tok.text(row["reference_ids"]).replace("\n", "\\n")
                               + ("   [exact match]" if st["exact"] else f"   [token distance {st['dist_ref']}]"))
            readout.extend(story(tok, start, ops, lo))
            readout.append("")
    info = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), config=str(CFG.relative_to(ROOT)),
                config_sha256=sha256_file(CFG), script_sha256=sha256_file(Path(__file__)),
                trajstats_sha256=sha256_file(ROOT / "scripts" / "trajviz" / "trajstats.py"),
                with_new300=with_new, pool_size=len(pool), sources=counts)
    (out / "pool.json").write_text(json.dumps(dict(info=info, pool=pool)), encoding="utf-8")
    (out / "filter_counts.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (out / "pool_readout.txt").write_text("\n".join(readout), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
