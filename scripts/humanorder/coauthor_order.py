"""Replay CoAuthor keystroke logs; measure the order in which the final text's words were written."""
import collections, json, re, statistics, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
files = [n for n in z.namelist() if n.endswith(".jsonl")]
tot = collections.Counter(); runs_all = []; sess_runs = []; jump_back = jump_fwd = 0; jump_dist = []
for n in files:
    doc = []  # list of (char, t, src)
    for line in z.read(n).decode("utf-8").splitlines():
        e = json.loads(line)
        if e["eventName"] == "system-initialize":
            doc = [(c, -1, "prompt") for c in (e.get("currentDoc") or "")]
            continue
        d = e.get("textDelta")
        if not d or not isinstance(d, dict) or "ops" not in d:
            continue
        src = "user" if e.get("eventSource") == "user" else "api"
        pos = 0; t = e["eventNum"]
        for op in d["ops"]:
            if "retain" in op:
                pos += op["retain"]
            elif "insert" in op:
                s = op["insert"] if isinstance(op["insert"], str) else ""
                tot["typed_" + src] += len(s)
                if src == "user":
                    tot["ins_at_end" if all(x[0].isspace() for x in doc[pos:]) else "ins_mid"] += len(s)
                doc[pos:pos] = [(c, t, src) for c in s]; pos += len(s)
            elif "delete" in op:
                k = op["delete"]; tot["deleted"] += sum(1 for x in doc[pos:pos + k] if x[2] != "prompt")
                del doc[pos:pos + k]
    # words of the final doc that were written in-session (not the prompt)
    text = "".join(c for c, _, _ in doc)
    words = []
    for m in re.finditer(r"\S+", text):
        ch = doc[m.start():m.end()]
        if all(s == "prompt" for _, _, s in ch):
            continue
        words.append((max(t for _, t, _ in ch), any(s == "api" for _, _, s in ch), len(words)))
    user_words = [w for w in words if not w[1]]
    tot["final_words"] += len(words); tot["final_words_api"] += len(words) - len(user_words)
    order = sorted(user_words, key=lambda w: (w[0], w[2]))
    pos_rank = {w[2]: i for i, w in enumerate(sorted(user_words, key=lambda w: w[2]))}
    seq = [pos_rank[w[2]] for w in order]
    run = 1; rs = []
    for a, b in zip(seq, seq[1:]):
        if b == a + 1:
            run += 1
        else:
            rs.append(run); run = 1
            if b < a: jump_back += 1
            else: jump_fwd += 1
            jump_dist.append(abs(b - a))
    if seq:
        rs.append(run)
        runs_all += rs; sess_runs.append(len(seq) / len(rs))
typed = tot["typed_user"]
print(f"sessions {len(files)}  final in-session words {tot['final_words']}  from AI suggestions {tot['final_words_api'] / tot['final_words']:.0%}")
print(f"user-typed chars {typed}: typed at the end of the text {tot['ins_at_end'] / typed:.0%}, typed in the middle {tot['ins_mid'] / typed:.0%}")
print(f"typed-then-deleted chars / all typed chars {tot['deleted'] / (typed + tot['typed_api']):.0%}")
print(f"word runs (written in final order without a jump): mean {statistics.mean(runs_all):.1f} words, median {statistics.median(runs_all)}")
print(f"words per run, mean over sessions {statistics.mean(sess_runs):.1f}, median {statistics.median(sess_runs):.1f}")
print(f"jumps: back {jump_back}  forward {jump_fwd}  back share {jump_back / max(1, jump_back + jump_fwd):.0%}  median distance {statistics.median(jump_dist)} words")
print("share of user words written in runs of >= 20 words:", f"{sum(r for r in runs_all if r >= 20) / sum(runs_all):.0%}")
