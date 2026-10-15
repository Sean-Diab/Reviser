"""Length of the opening left-to-right stretch (inserts before the first cursor move) in July-order
training shards vs locked model decodes. --ignore-deletes counts net kept tokens up to the first move.
Main session, 2026-10-09."""
import sys, json, statistics as st
IGNORE_DEL = "--ignore-deletes" in sys.argv
sys.path.insert(0, r"experiments/july_control/code/july_impl")
from traj_stats import load_samples, is_insert, is_move, DELETE_ID, EOR_ID

def opening(body):
    n = 0
    for t in body:
        if is_insert(int(t)): n += 1
        elif is_move(int(t)): return n, True
        elif int(t) == DELETE_ID:
            if IGNORE_DEL: n -= 1
            else: return n, True
        elif int(t) == EOR_ID: return n, False
    return n, False

def summarize(name, rows):
    op = [r[0] for r in rows]; out = [r[1] for r in rows]; plen = [r[2] for r in rows]; jumped = [r[3] for r in rows]
    share = [o / x for o, x in zip(op, out) if x > 0]
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    print(f"{name:22s} n={len(rows):5d} prefix med {q(plen,.5):4d} | opening run med {q(op,.5):4d} mean {st.mean(op):6.1f} p90 {q(op,.9):4d} | "
          f"share of output med {q(share,.5):.2f} mean {st.mean(share):.2f} | >=20 tok {sum(o>=20 for o in op)/len(op):.0%} >=50 {sum(o>=50 for o in op)/len(op):.0%} | never jumps {1-sum(jumped)/len(jumped):.0%} | out med {q(out,.5)}")

def train(name, d, n=20000):
    rows = []
    for seq, plen in load_samples(d, n, 512):
        body = seq[plen:]
        o, j = opening(body)
        out = sum(1 for t in body if is_insert(int(t))) - sum(1 for t in body if int(t) == DELETE_ID)
        rows.append((o, out, plen, j))
    summarize(name, rows)
    # by prefix length
    for lo, hi in [(0, 1), (1, 20), (20, 60), (60, 10**9)]:
        sub = [r for r in rows if lo <= r[2] < hi]
        if len(sub) > 50: summarize(f"  prefix [{lo},{hi})", sub)

def dec(name, path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if "actions_new" not in d: continue
        body = d["actions_new"]; plen = len(d["prefix_ids"])
        o, j = opening(body)
        rows.append((o, int(d["out_len"]), plen, j))
    summarize(name, rows)

R = r""
train("TRAIN j0 (July dials)", R + "runs/overnight/check/j0")
train("TRAIN jd_n05_imm_m", R + "runs/overnight/check/jd_n05_imm_m")
for m in ["july608", "jd608", "jd608_g8m2_s3", "rev100"]:
    dec("MODEL " + m, R + f"runs/papereval/dec/c4paper3k.lock.{m}.jsonl")
