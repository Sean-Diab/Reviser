"""Trajectory statistics for cursor-and-delete text models (stdlib only; notes/trajectory-check-2026-10-09.md).

Every trajectory is first normalized to ops:
    ("i", tok)  insert tok at the cursor, cursor moves right past it
    ("d",)      delete the token left of the cursor (backspace), cursor moves left
    ("m", k)    move the cursor by k (negative = left)
    ("s",)      stop
and replayed on a start canvas with a protected prefix of length ``lo`` (the locked prompt in Part 1; 0 in Part 2,
where the canvas holds only the editable draft). Positions are measured inside the editable region:
rel = (cursor - lo) / (len - lo), so 0 = start of the model's text, 1 = its end.

Three groups of measures:
  * paper_diagnostics(): an exact port of the original Reviser paper's trajectory table code
    (workspaces/4-21/c4_v2_arena_test/analyze_reviser_diagnostics.py: analyze_trajectory + traj_agg, read-only
    reference), so our numbers sit next to the paper's Table "Reviser trajectory diagnostics in decoding".
  * sample_stats() / aggregate(): where moves and edits land, jump sizes, delete ages, churn, loops, prompt safety.
  * edit_sample_stats(): Part 2 extras (net edit regions vs the draft, self-reverting and no-op edits).

Vocabularies: Part 1 uses the July / paper ids (GPT-2 inserts < 50257, DELETE 50259, EOR 50260, moves 50261 + 2i
(+2^i) and 50261 + 2i + 1 (-2^i)); Part 2 uses reviser_v2.actions.ActionSpace (see ops_from_space).
"""
from __future__ import annotations

import difflib
import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---- Part 1 (July / paper) vocabulary -------------------------------------------------------------------------
GPT2_VOCAB = 50257
DELETE_ID, EOR_ID, MOVE_START = 50259, 50260, 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

AGE_BINS = [(1, 1, "1 (deleted right after typing it)"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"),
            (11, 50, "11-50"), (51, 10 ** 9, "over 50")]
POS_BINS = ("start third", "middle third", "end third", "at the very end")


def july_op(t: int):
    t = int(t)
    if 0 <= t < GPT2_VOCAB:
        return ("i", t)
    if t == DELETE_ID:
        return ("d",)
    if t == EOR_ID:
        return ("s",)
    if MOVE_START <= t < MOVE_START + 20:
        off = t - MOVE_START
        amt = MOVE_AMOUNTS[off // 2]
        return ("m", -amt if off % 2 else amt)
    return ("x", t)  # other special (never produced by the masked decoder)


def ops_from_july(actions: Sequence[int]) -> List[tuple]:
    return [july_op(t) for t in actions]


def ops_from_space(space, actions: Sequence[int]) -> List[tuple]:
    out = []
    for a in actions:
        if space.is_content(a):
            out.append(("i", int(a)))
        elif a == space.DELETE:
            out.append(("d",))
        elif a == space.STOP:
            out.append(("s",))
        elif space.is_move(a):
            out.append(("m", int(space.move_value(a))))
        else:
            out.append(("x", int(a)))
    return out


def pos_bin(cur: int, n: int, lo: int) -> int:
    """0/1/2 = start/middle/end third of the editable text, 3 = exactly at its end (or the text is empty)."""
    e = n - lo
    if e <= 0 or cur >= n:
        return 3
    r = (cur - lo) / e
    return 0 if r < 1 / 3 else (1 if r < 2 / 3 else 2)


# ---- replay -----------------------------------------------------------------------------------------------------
def replay(start: Sequence[int], ops: Sequence[tuple], lo: int = 0, cursor: Optional[int] = None) -> List[dict]:
    """Execute ops; one event dict per op with the state before it. Start tokens get birth = None (prompt/draft).
    Moves are clamped to [0, len] like the executors (the decoders mask illegal moves, so clamping never fires there;
    it is counted as 'clamped')."""
    toks = list(start)
    birth: List[Optional[int]] = [None] * len(toks)
    cur = len(toks) if cursor is None else cursor
    ev = []
    for k, op in enumerate(ops):
        n = len(toks)
        e = {"k": k, "op": op[0], "cur": cur, "n": n}
        if op[0] == "i":
            e.update(tok=op[1], bin=pos_bin(cur, n, lo), at_end=cur >= n, rel=(cur - lo) / (n - lo) if n > lo else 1.0)
            toks.insert(cur, op[1])
            birth.insert(cur, k)
            cur += 1
        elif op[0] == "d":
            if cur <= 0:
                e.update(illegal=True)
            else:
                e.update(bin=pos_bin(cur, n, lo), at_end=cur >= n, rel=(cur - lo) / (n - lo) if n > lo else 1.0,
                         tok=toks[cur - 1], born=birth[cur - 1], in_prompt=cur - 1 < lo)
                toks.pop(cur - 1)
                birth.pop(cur - 1)
                cur -= 1
        elif op[0] == "m":
            to = cur + op[1]
            clamped = not 0 <= to <= n
            to = max(0, min(n, to))
            e.update(delta=op[1], to=to, clamped=clamped, into_prompt=to < lo)
            cur = to
        e["state"] = hash((tuple(toks), cur))
        e["cur_after"], e["n_after"] = cur, len(toks)
        ev.append(e)
        if op[0] == "s":
            break
    return ev


def final_tokens(start: Sequence[int], ops: Sequence[tuple], cursor: Optional[int] = None) -> List[int]:
    toks = list(start)
    cur = len(toks) if cursor is None else cursor
    for op in ops:
        if op[0] == "i":
            toks.insert(cur, op[1])
            cur += 1
        elif op[0] == "d" and cur > 0:
            toks.pop(cur - 1)
            cur -= 1
        elif op[0] == "m":
            cur = max(0, min(len(toks), cur + op[1]))
        elif op[0] == "s":
            break
    return toks


def jumps(ev: Sequence[dict]) -> List[Tuple[int, int, int, int]]:
    """Consecutive moves merged into one jump: (net displacement, target cursor, text length, index of first move)."""
    out, i = [], 0
    while i < len(ev):
        if ev[i]["op"] != "m":
            i += 1
            continue
        j, net = i, 0
        while j < len(ev) and ev[j]["op"] == "m":
            net += ev[j]["delta"]
            j += 1
        out.append((net, ev[j - 1]["to"], ev[j - 1]["n"], i))
        i = j
    return out


# ---- per-sample stats ---------------------------------------------------------------------------------------------
def sample_stats(start: Sequence[int], ops: Sequence[tuple], lo: int = 0, cursor: Optional[int] = None) -> Dict:
    ev = replay(start, ops, lo, cursor)
    n_ins = sum(e["op"] == "i" for e in ev)
    n_del = sum(e["op"] == "d" and not e.get("illegal") for e in ev)
    n_move = sum(e["op"] == "m" for e in ev)
    stopped = bool(ev) and ev[-1]["op"] == "s"
    out_len = (ev[-1]["n_after"] if ev else len(start)) - lo
    move_d = [abs(e["delta"]) for e in ev if e["op"] == "m"]
    js = jumps(ev)
    jump_bins = Counter(pos_bin(to, n, lo) for _, to, n, _ in js)
    edit_bins = Counter(e["bin"] for e in ev if e["op"] in ("i", "d") and "bin" in e)
    ins_bins = Counter(e["bin"] for e in ev if e["op"] == "i")
    del_bins = Counter(e["bin"] for e in ev if e["op"] == "d" and "bin" in e)
    # deletes: age since the deleted token was typed, what happens next, prompt safety
    ages, pre_existing, prompt_dels = [], 0, 0
    nxt = Counter()
    same_back = reinsert5 = 0
    for idx, e in enumerate(ev):
        if e["op"] != "d" or e.get("illegal"):
            continue
        if e["in_prompt"]:
            prompt_dels += 1
        if e["born"] is None:
            pre_existing += 1
        else:
            ages.append(e["k"] - e["born"])
        p = e["cur"] - 1  # boundary where the token was
        if any(f["op"] == "i" and f["cur"] == p and f["tok"] == e["tok"] for f in ev[idx + 1: idx + 6]):
            reinsert5 += 1  # insert -> delete -> same token typed back at the same place within 5 actions
        f = ev[idx + 1] if idx + 1 < len(ev) else None
        if f is None:
            nxt["end of budget"] += 1
        elif f["op"] == "i":
            if f["tok"] == e["tok"]:
                nxt["same token typed back"] += 1
                same_back += 1
            else:
                nxt["different token typed (replace)"] += 1
        elif f["op"] == "d":
            nxt["another delete"] += 1
        elif f["op"] == "m":
            nxt["cursor moves away"] += 1
        else:
            nxt["stop"] += 1
    # loops: the same (text, cursor) state seen 3+ times
    visits = Counter(e["state"] for e in ev)
    max_visits = max(visits.values()) if visits else 0
    ping = sum(1 for a, b in zip(ev, ev[1:]) if a["op"] == "m" and b["op"] == "m" and a["delta"] == -b["delta"])
    run = best = 0
    for e in ev:
        run = run + 1 if e["op"] == "m" else 0
        best = max(best, run)
    return dict(
        n_actions=n_ins + n_del + n_move, n_steps=len(ev), max_move_run=best, n_ins=n_ins,
        n_other=sum(e["op"] == "x" for e in ev), n_del=n_del, n_move=n_move, stopped=stopped, out_len=out_len,
        move_d=move_d, n_back=sum(e["op"] == "m" and e["delta"] < 0 for e in ev),
        jumps=[j[0] for j in js], jump_bins=dict(jump_bins), edit_bins=dict(edit_bins), ins_bins=dict(ins_bins),
        del_bins=dict(del_bins), n_edit=sum(edit_bins.values()),
        ages=ages, del_pre_existing=pre_existing, del_in_prompt=prompt_dels, del_next=dict(nxt), del_same_back=same_back, del_reinsert5=reinsert5,
        max_state_visits=max_visits, move_pingpong=ping, n_clamped=sum(bool(e.get("clamped")) for e in ev),
        moves_into_prompt=sum(bool(e.get("into_prompt")) for e in ev),
        illegal_deletes=sum(bool(e.get("illegal")) for e in ev),
        ins_rel=[e["rel"] for e in ev if e["op"] == "i"],
    )


def edit_sample_stats(draft: Sequence[int], ops: Sequence[tuple], reference: Optional[Sequence[int]] = None) -> Dict:
    """Part 2 extras on top of sample_stats (cursor starts at the end of the draft, lo = 0)."""
    s = sample_stats(draft, ops, 0, None)
    fin = final_tokens(draft, ops)
    sm = difflib.SequenceMatcher(a=list(draft), b=fin, autojunk=False)
    regions = [o for o in sm.get_opcodes() if o[0] != "equal"]
    n = max(1, len(draft))
    s["final"] = fin
    s["changed"] = fin != list(draft)
    s["net_regions"] = len(regions)
    s["net_region_bins"] = dict(Counter(pos_bin(i1, len(draft), 0) if i1 < len(draft) else 3 for _, i1, _, _, _ in regions))
    s["content_actions"] = s["n_ins"] + s["n_del"]
    s["self_deletes"] = len(s["ages"])            # deleted tokens the model typed in this run
    s["noop_rollout"] = s["content_actions"] > 0 and not s["changed"]  # typed/deleted something, net result = draft
    # wasted content actions: content actions beyond the minimum token edits needed for the net change
    need = sum(max(i2 - i1, 0) + max(j2 - j1, 0) for _, i1, i2, j1, j2 in regions)
    s["min_token_edits"] = need
    s["extra_content_actions"] = max(0, s["content_actions"] - need)
    if reference is not None:
        rr = [o for o in difflib.SequenceMatcher(a=list(draft), b=list(reference), autojunk=False).get_opcodes() if o[0] != "equal"]
        s["ref_regions"] = len(rr)
        s["ref_region_bins"] = dict(Counter(pos_bin(i1, len(draft), 0) if i1 < len(draft) else 3 for _, i1, _, _, _ in rr))
        s["exact"] = fin == list(reference)
        s["ref_changed"] = list(reference) != list(draft)
    ev = replay(draft, ops, 0, None)
    k0 = next((e["k"] for e in ev if e["op"] in ("i", "d")), None)
    s["first_edit_bin"] = None if k0 is None else ev[k0].get("bin")
    # going back again after editing has started (the cursor starts at the end, so the first trip back is needed)
    s["back_after_first_edit"] = 0 if k0 is None else sum(e["op"] == "m" and e["delta"] < 0 for e in ev[k0:])
    return s


# ---- aggregation --------------------------------------------------------------------------------------------------
def _share(c: Counter, keys=range(4)):
    t = sum(c.values())
    return [c.get(k, 0) / t if t else None for k in keys]


def aggregate(samples: Iterable[Dict]) -> Dict:
    S = list(samples)
    n = len(S)
    tot = Counter()
    jb, eb, ib, db = Counter(), Counter(), Counter(), Counter()
    move_d, jl, ages, nxt = [], [], [], Counter()
    for s in S:
        for k in ("n_actions", "n_ins", "n_del", "n_move", "n_back", "out_len", "n_edit", "del_pre_existing",
                  "del_in_prompt", "del_same_back", "del_reinsert5", "move_pingpong", "n_clamped", "moves_into_prompt", "illegal_deletes"):
            tot[k] += s[k]
        for c, src in ((jb, "jump_bins"), (eb, "edit_bins"), (ib, "ins_bins"), (db, "del_bins")):
            c.update({int(k): v for k, v in s[src].items()})
        move_d += s["move_d"]
        jl += s["jumps"]
        ages += s["ages"]
        nxt.update(s["del_next"])
    L = max(1, tot["out_len"])
    nd = tot["n_del"]
    age_hist = []
    for lo_, hi_, lab in AGE_BINS:
        age_hist.append((lab, sum(lo_ <= a <= hi_ for a in ages) / len(ages) if ages else None))
    na = max(1, len(ages))
    return dict(
        n=n,
        mean_out_len=tot["out_len"] / n if n else None,
        stopped_rate=sum(s["stopped"] for s in S) / n if n else None,
        actions_per_token=tot["n_actions"] / L,
        moves_per_token=tot["n_move"] / L,
        deletes_per_token=tot["n_del"] / L,
        inserts_per_token=tot["n_ins"] / L,
        moves_per_sample=tot["n_move"] / n if n else None,
        deletes_per_sample=tot["n_del"] / n if n else None,
        mean_move_abs=sum(move_d) / len(move_d) if move_d else None,
        jumps_per_sample=len(jl) / n if n else None,
        mean_jump_abs=sum(abs(x) for x in jl) / len(jl) if jl else None,
        median_jump_abs=sorted(abs(x) for x in jl)[len(jl) // 2] if jl else None,
        backward_move_share=tot["n_back"] / tot["n_move"] if tot["n_move"] else None,
        backward_jump_share=sum(x < 0 for x in jl) / len(jl) if jl else None,
        samples_with_backward_move=sum(s["n_back"] > 0 for s in S) / n if n else None,
        samples_with_delete=sum(s["n_del"] > 0 for s in S) / n if n else None,
        jump_target_bins=_share(jb), edit_bins=_share(eb), insert_bins=_share(ib), delete_bins=_share(db),
        edits_not_at_end=(1 - eb.get(3, 0) / sum(eb.values())) if sum(eb.values()) else None,
        inserts_not_at_end=(1 - ib.get(3, 0) / sum(ib.values())) if sum(ib.values()) else None,
        n_deletes=nd,
        delete_age_le1=sum(a <= 1 for a in ages) / na if ages else None,
        delete_age_le3=sum(a <= 3 for a in ages) / na if ages else None,
        delete_age_le5=sum(a <= 5 for a in ages) / na if ages else None,
        delete_age_median=sorted(ages)[len(ages) // 2] if ages else None,
        delete_age_hist=age_hist,
        delete_of_start_tokens=tot["del_pre_existing"] / nd if nd else None,
        delete_next=({k: v / nd for k, v in nxt.most_common()} if nd else {}),
        delete_same_token_back=tot["del_same_back"] / nd if nd else None,
        delete_reinsert_within5=tot["del_reinsert5"] / nd if nd else None,
        prompt_token_deletes=tot["del_in_prompt"],
        moves_into_prompt=tot["moves_into_prompt"],
        clamped_moves=tot["n_clamped"],
        illegal_deletes=tot["illegal_deletes"],
        samples_with_noop_tokens=sum(s["n_other"] > 0 for s in S) / n if n else None,
        empty_output_rate=sum(s["out_len"] <= 0 for s in S) / n if n else None,
        samples_move_run_8plus=sum(s["max_move_run"] >= 8 for s in S) / n if n else None,
        samples_state_3plus=sum(s["max_state_visits"] >= 3 for s in S) / n if n else None,
        samples_state_5plus=sum(s["max_state_visits"] >= 5 for s in S) / n if n else None,
        move_pingpong_per_100_moves=100 * tot["move_pingpong"] / tot["n_move"] if tot["n_move"] else None,
        mean_insert_rel=(sum(r for s in S for r in s["ins_rel"]) / max(1, sum(len(s["ins_rel"]) for s in S))),
    )


def aggregate_edit(samples: Iterable[Dict]) -> Dict:
    S = list(samples)
    a = aggregate(S)
    n = len(S)
    rb, fb = Counter(), Counter()
    for s in S:
        rb.update({int(k): v for k, v in s.get("ref_region_bins", {}).items()})
        fb.update({int(k): v for k, v in s["net_region_bins"].items()})
    nd = sum(s["n_del"] for s in S)
    a.update(
        changed_rate=sum(s["changed"] for s in S) / n if n else None,
        mean_net_regions=sum(s["net_regions"] for s in S) / n if n else None,
        mean_ref_regions=(sum(s.get("ref_regions", 0) for s in S) / n) if n and "ref_regions" in S[0] else None,
        mean_content_actions=sum(s["content_actions"] for s in S) / n if n else None,
        mean_actions=sum(s["n_actions"] for s in S) / n if n else None,
        self_delete_share=sum(s["self_deletes"] for s in S) / nd if nd else None,
        noop_rollout_rate=sum(s["noop_rollout"] for s in S) / n if n else None,
        extra_content_action_share=(sum(s["extra_content_actions"] for s in S) / max(1, sum(s["content_actions"] for s in S))),
        net_region_bins=_share(fb), ref_region_bins=_share(rb) if sum(rb.values()) else None,
        exact=sum(bool(s.get("exact")) for s in S) / n if n and "exact" in S[0] else None,
        first_edit_bins=_share(Counter(s["first_edit_bin"] for s in S if s["first_edit_bin"] is not None)),
        samples_back_after_first_edit=(sum(s["back_after_first_edit"] > 0 for s in S) /
                                       max(1, sum(s["first_edit_bin"] is not None for s in S))),
    )
    return a


# ---- the paper's trajectory table (exact port) -------------------------------------------------------------------
def paper_analyze(action_ids: Sequence[int]) -> Dict:
    """analyze_trajectory() of the original paper's analyze_reviser_diagnostics.py, on the continuation actions only
    (the 35 prompt inserts already stripped; the canvas starts empty, as there). Same constants and quirks: moves are
    clamped, the first insert into an empty canvas has relative position 0.0, EOR ends the scan and is not counted."""
    canvas: List[int] = []
    cur = 0
    n_ins = n_mov = n_del = n_back = n_end = n_mid = 0
    md: List[int] = []
    rel: List[float] = []
    last_ins = None
    went_back = False
    has_rev = False
    steps = 0
    for tid in action_ids:
        tid = int(tid)
        steps += 1
        if tid == EOR_ID:
            break
        if 0 <= tid < GPT2_VOCAB and tid != GPT2_VOCAB - 1:
            if last_ins is not None and went_back:
                has_rev = True
            L = len(canvas)
            rel.append(cur / L if L > 0 else 0.0)
            if cur >= len(canvas):
                n_end += 1
            else:
                n_mid += 1
            canvas.insert(cur, tid)
            cur += 1
            n_ins += 1
            last_ins = cur
            went_back = False
        elif tid == DELETE_ID:
            if cur > 0:
                canvas.pop(cur - 1)
                cur -= 1
            n_del += 1
        elif MOVE_START <= tid < MOVE_START + 20:
            off = tid - MOVE_START
            d = -MOVE_AMOUNTS[off // 2] if off % 2 else MOVE_AMOUNTS[off // 2]
            cur = max(0, min(len(canvas), cur + d))
            md.append(abs(d))
            if d < 0:
                n_back += 1
            n_mov += 1
            if last_ins is not None and cur < last_ins:
                went_back = True
    fl = len(canvas)
    na = n_ins + n_mov + n_del
    nan = float("nan")
    return dict(n_actions=na, n_inserts=n_ins, n_moves=n_mov, n_deletes=n_del, steps_to_eor=steps, final_canvas_len=fl,
                actions_per_token=na / fl if fl > 0 else nan,
                insert_frac=n_ins / na if na else nan, move_frac=n_mov / na if na else nan,
                delete_frac=n_del / na if na else nan, frac_backward_moves=n_back / n_mov if n_mov else 0.0,
                mean_move_dist=sum(md) / len(md) if md else 0.0, move_distances=md, n_end_appends=n_end,
                n_mid_inserts=n_mid, frac_end_append=n_end / n_ins if n_ins else nan,
                has_backward_revision=has_rev, insert_rel_positions=rel)


def paper_agg(stats: Sequence[Dict]) -> Dict:
    """traj_agg() of the paper: per-example means for fractions and actions/token (NaN skipped), pooled move distances
    and insert positions, share of examples with a backward revision."""
    def fmean(k):
        v = [s[k] for s in stats if not (isinstance(s[k], float) and math.isnan(s[k]))]
        return sum(v) / len(v) if v else float("nan")
    md = [d for s in stats for d in s["move_distances"]]
    rel = [r for s in stats for r in s["insert_rel_positions"]]
    hist = {str(a): (sum(1 for d in md if d == a) / len(md) if md else None) for a in MOVE_AMOUNTS}
    return dict(n=len(stats), actions_per_token=fmean("actions_per_token"), insert_frac=fmean("insert_frac"),
                move_frac=fmean("move_frac"), delete_frac=fmean("delete_frac"),
                mean_move_dist=sum(md) / len(md) if md else None, move_dist_mass=hist,
                frac_backward_moves=fmean("frac_backward_moves"),
                mean_insert_rel=sum(rel) / len(rel) if rel else None,
                frac_end_append=fmean("frac_end_append"),
                frac_non_end_append=1 - fmean("frac_end_append"),
                backward_revision_share=sum(1 for s in stats if s["has_backward_revision"]) / len(stats) if stats else None)


def paper_table(rows_by_seed: Dict[str, List[Sequence[int]]]) -> Dict:
    """The paper's table: computed per seed, then averaged across seeds ("Statistic (averaged across seeds)")."""
    per = {sd: paper_agg([paper_analyze(a) for a in acts]) for sd, acts in rows_by_seed.items()}
    keys = [k for k, v in next(iter(per.values())).items() if isinstance(v, (int, float)) and k != "n"]
    avg = {k: sum(p[k] for p in per.values() if p[k] is not None) / len(per) for k in keys}
    avg["move_dist_mass"] = {a: sum((p["move_dist_mass"][a] or 0) for p in per.values()) / len(per)
                             for a in next(iter(per.values()))["move_dist_mass"]}
    avg["n"] = sum(p["n"] for p in per.values())
    return dict(avg=avg, per_seed=per)
