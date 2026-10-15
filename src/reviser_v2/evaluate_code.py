"""CanItEdit evaluation of HF-backbone action LMs by executing tests (task 37; training data: realdata/prep_code.py).

Benchmark: ``nuprl/CanItEdit`` @ 3c07f38b (105 Python problems; MIT), read from ``data/code/canitedit_test.parquet``
(the byte copy written by prep_code, sha256-checked against its manifest when present) or, failing that, from the
pinned hub revision. Each problem has ``before``, ``after``, ``tests``, a descriptive and a lazy instruction and a
taxonomy ``change_kind`` (adaptive / corrective / perfective). A candidate is correct iff ``candidate + "\\n\\n" +
tests`` runs to completion (exit code 0) in a fresh Python subprocess.

Model inputs are those of evaluate_edits.predict (reused as is): the instruction text and the ``before`` file are
tokenized separately with the checkpoint's backbone tokenizer (the prep_code / retokenize rule), then
  * ``--layout edit`` (Reviser): prompt = instruction ids, before = editable draft (conditioning layout), free edit
    actions until STOP (max 3 * len(before) + max_extra actions);
  * ``--layout ar`` (AR rewrite baseline): prompt = instruction ":" before (read-only), blank canvas, full rewrite
    typed left to right (max 2 * len(before) + max_extra actions).
The final canvas is decoded with the backbone tokenizer (no clean-up of tokenization spaces) and executed. Both
instruction variants are evaluated from one model load (``--instr descriptive lazy``, the default); records are
batched in order of prompt length (order restored in the outputs). Greedy (default) or ``--type-greedy``.

Baselines (no model): ``--baseline copy`` (candidate = before; the "no edit" floor) and ``--baseline reference``
(candidate = after; must pass ~100%: failures there are environment problems such as a missing library).

Execution harness (``run_program``): the program is written to a fresh temporary directory and run with
``sys.executable`` (cwd = that directory, stdin closed, minimal environment: no PYTHONPATH, no credentials, no GPU,
single-threaded BLAS), wall-clock ``--timeout`` (default 10 s; the whole process group is killed on expiry). On
Linux also RLIMIT_AS = ``--mem-mb`` (default 2048; 0 = off), RLIMIT_CPU = timeout + 1 s, RLIMIT_FSIZE 64 MB, no core
files. No network isolation (none is needed by the tests; run on a disposable box). Error classes: ``ok``,
``Timeout``, the exception class of the last traceback line (AssertionError, SyntaxError, NameError, ...),
``Signal<k>`` (e.g. 9 = killed, 24 = CPU limit) or ``Exit<rc>``.

Metrics (overall, per change_kind): pass@1 (greedy: fraction passing; Wilson CI), exact match after whitespace
normalisation (trailing spaces and blank lines dropped, indentation kept), exact match on backbone ids, changed rate
(normalised candidate != normalised before), stop rate, n_actions / n_del / n_move / n_ins, generation and execution
seconds, error-class counts; plus prompt lengths and the same metrics on the subset of problems whose AR sequence
(instruction ":" before SEP EDIT after STOP) fits the training max_len (``--pool-max-len``, 1024): the code_hf models
were trained on sequences <= 1024 tokens, and ~25 of the 105 CanItEdit problems are longer as AR sequences (only ~9 as
oracle D4 episodes), a length-generalisation asymmetry between the layouts.

``--data code_dev`` scores the EditPackFT dev pool (no tests): evaluate_edits.score metrics (SARI on text, EM on ids,
edit P/R) + whitespace-normalised EM.

Outputs: ``<run>/eval/code_canitedit_<instr>_<dec>[_ar][_lim<n>][_<tag>].{jsonl,summary.json}`` (baselines:
runs/code_baselines), ``code_dev_<dec>[_ar]...`` for the dev pool, with a meta block.

Usage:
  python -m reviser_v2.evaluate_code --baseline reference
  python -m reviser_v2.evaluate_code --ckpt runs/qwen17b-code-d4-s1 --type-greedy [--batch 8]
  python -m reviser_v2.evaluate_code --ckpt runs/qwen17b-code-ar-s1 --layout ar
  python -m reviser_v2.evaluate_code --ckpt runs/qwen17b-code-d4-s1 --data code_dev --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .evaluate_toy import wilson

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "data" / "code"
INSTR_COLS = {"descriptive": "instruction_descriptive", "lazy": "instruction_lazy"}


# ---- data --------------------------------------------------------------------------------------------------------
def norm_code(s: str) -> str:
    """Whitespace-normalised code for exact match: trailing spaces and blank lines dropped, indentation kept."""
    return "\n".join(l.rstrip() for l in s.strip().splitlines() if l.strip())


def load_canitedit(path: Optional[str] = None, code_dir: Path = CODE_DIR):
    """(dataframe, info) from ``path``, else data/code/canitedit_test.parquet, else the pinned hub revision."""
    import pandas as pd
    from .realdata.prep_code import CANITEDIT, PARQUET_SHA256, fetch, sha256_file
    p = Path(path) if path else Path(code_dir) / "canitedit_test.parquet"
    if not p.exists():
        if path:
            raise FileNotFoundError(p)
        p = fetch(CANITEDIT)  # hub cache (HF_HUB_OFFLINE=1 needs it cached: run prep_code first)
    sha = sha256_file(p)
    expected = {m["inputs"]["canitedit"]["sha256"] for m in (json.load(open(q)) for q in Path(code_dir).glob("manifest_*.json"))
                if m.get("inputs", {}).get("canitedit")} if Path(code_dir).exists() else set()
    if expected and sha not in expected:
        raise RuntimeError(f"{p}: sha256 {sha} differs from the prep_code manifest(s) {sorted(expected)}")
    if not path and sha != PARQUET_SHA256[CANITEDIT["repo"]]:
        raise RuntimeError(f"{p}: sha256 {sha} != pinned {PARQUET_SHA256[CANITEDIT['repo']]}")
    return pd.read_parquet(p), dict(repo=CANITEDIT["repo"], revision=CANITEDIT["revision"], file=CANITEDIT["file"],
                                    path=str(p), sha256=sha)


def encode(tok, text: str, n_content: int) -> List[int]:
    ids = tok([text], add_special_tokens=False, split_special_tokens=True)["input_ids"][0]
    assert not ids or max(ids) < n_content, f"backbone id {max(ids)} >= n_content {n_content}"
    return list(ids)


def canitedit_records(df, instr: str, tok=None, n_content: int = 0, limit: int = 0) -> List[Dict]:
    """One record per problem for instruction variant ``instr``; *_ids only when a tokenizer is given."""
    out = []
    for r in df.itertuples(index=False):
        tax = r.taxonomy if isinstance(r.taxonomy, dict) else {}
        rec = dict(uid=f"canitedit/{r.full_name}/{instr}", id=int(r.id), name=r.name, intent=tax.get("change_kind", "?"),
                   instr=instr, instruction=getattr(r, INSTR_COLS[instr]), before_text=r.before, after_text=r.after,
                   tests=r.tests)
        if tok is not None:
            for k, t in (("instruction_ids", rec["instruction"]), ("before_ids", r.before), ("after_ids", r.after)):
                rec[k] = encode(tok, t, n_content)
        out.append(rec)
        if limit and len(out) >= limit:
            break
    return out


def code_dev_records(tag: str = "qwen3", max_len: int = 1024, limit: int = 0, code_dir: Path = CODE_DIR) -> List[Dict]:
    """EditPackFT dev pool of prep_code: backbone ids from the cache pickle, texts from the parallel text jsonl."""
    import pickle
    d = pickle.load(open(Path(code_dir) / "cache" / f"editpackft_dev_{tag}_le{max_len}.pkl", "rb"))
    texts = [json.loads(l) for l in open(Path(code_dir) / f"editpackft_dev_{tag}_le{max_len}.text.jsonl", encoding="utf-8")]
    assert len(texts) == len(d["recs"]), "dev pickle and text jsonl differ"
    out = []
    for t, (i, b, a) in zip(texts, d["recs"]):
        out.append(dict(uid=t["uid"], intent="code_dev", instruction=t["instruction"], before_text=t["before"],
                        after_text=t["after"], instruction_ids=list(i), before_ids=list(b), after_ids=list(a)))
        if limit and len(out) >= limit:
            break
    return out, d["meta"]


# ---- execution harness -----------------------------------------------------------------------------------------
def _env(tmp: str) -> Dict[str, str]:
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "LANG", "LC_ALL") if k in os.environ}
    import site  # HOME/APPDATA are not passed: keep the interpreter's user site-packages visible explicitly
    env.update(HOME=tmp, TMPDIR=tmp, TEMP=tmp, TMP=tmp, PYTHONUSERBASE=site.getuserbase(), PYTHONHASHSEED="0",
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    return env


def _preexec(timeout: float, mem_mb: int):
    if os.name != "posix":
        return None

    def f():
        import resource
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        if mem_mb:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


_EXC = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")


def error_class(rc: int, stderr: str) -> str:
    if rc == 0:
        return "ok"
    for line in reversed(stderr.strip().splitlines()):
        m = _EXC.match(line.strip())
        if m and not line.startswith(" ") and ("Error" in m.group(1) or "Exception" in m.group(1) or
                                               m.group(1) in ("KeyboardInterrupt", "SystemExit", "StopIteration")):
            return m.group(1).split(".")[-1]
    return f"Signal{-rc}" if rc < 0 else f"Exit{rc}"


def run_program(src: str, timeout: float = 10.0, mem_mb: int = 2048) -> Dict:
    """Run ``src`` as a Python program in a fresh temp dir; {passed, status, rc, seconds, stderr_tail}."""
    with tempfile.TemporaryDirectory(prefix="rv2code_") as tmp:
        prog = Path(tmp) / "prog.py"
        prog.write_text(src, encoding="utf-8")
        t0 = time.time()
        p = subprocess.Popen([sys.executable, str(prog)], cwd=tmp, env=_env(tmp), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=_preexec(timeout, mem_mb),
                             start_new_session=os.name == "posix")
        try:
            _, err = p.communicate(timeout=timeout)
            rc, status = p.returncode, None
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                import signal
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                p.kill()
            _, err = p.communicate()
            rc, status = p.returncode, "Timeout"
        err = err.decode("utf-8", "replace")
        status = status or error_class(rc, err)
        return dict(passed=rc == 0 and status == "ok", status=status, rc=rc, exec_seconds=round(time.time() - t0, 3),
                    stderr_tail=err[-400:])


def execute_all(programs: List[str], timeout: float, mem_mb: int, workers: int) -> List[Dict]:
    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(lambda s: run_program(s, timeout, mem_mb), programs))


# ---- model -------------------------------------------------------------------------------------------------------
def checkpoint_tokenizer(ckpt):
    """(backbone tokenizer, model_config) of an HF-backbone checkpoint (as hf_backbone.eval_prepare)."""
    from .hf_backbone import is_hf_checkpoint
    from .realdata.retokenize import load_tokenizer
    mc = is_hf_checkpoint(ckpt)
    if mc is None:
        raise SystemExit("evaluate_code needs an HF-backbone checkpoint (model_config.kind == 'hf')")
    return load_tokenizer(mc["tokenizer"] or mc["pretrained"], mc.get("tokenizer_revision"), mc.get("tok_tag")), mc


def generate(recs, ckpt, tok, a):
    """evaluate_edits.predict over ``recs`` in prompt-length order; returns (texts, finals, extra, meta)."""
    from .evaluate_edits import predict
    from .hf_backbone import _Decoder
    if a.layout == "ar":  # predict builds AR prompts with realedit.ar_prompt (GPT-2 ":" id 25)
        from .tasks.realedit import COLON
        assert encode(tok, ":", 1 << 30) == [COLON], "backbone ':' id differs from GPT-2's (AR layout)"
    plen = [len(r["instruction_ids"]) + len(r["before_ids"]) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: (plen[i], i))
    traces, meta = predict([recs[i] for i in order], ckpt, a.device, a.batch, a.temperature, a.type_greedy,
                           a.sample_seed, a.max_extra, a.layout)
    tr = [None] * len(recs)
    for k, i in enumerate(order):
        tr[i] = traces[k]
    dec = _Decoder(tok)
    texts = [dec.decode(t.final) for t in tr]
    extra = [dict(stopped=t.stopped, n_actions=len(t.actions), n_del=t.n_del, n_move=t.n_move, n_ins=t.n_ins,
                  gen_seconds=t.seconds, prompt_len=p) for t, p in zip(tr, plen)]
    return texts, [t.final for t in tr], extra, meta


# ---- scoring -----------------------------------------------------------------------------------------------------
def score_canitedit(recs, texts, finals=None, extra=None, results=None) -> List[Dict]:
    rows = []
    for i, (r, pred) in enumerate(zip(recs, texts)):
        row = dict(uid=r["uid"], id=r["id"], name=r["name"], intent=r["intent"], instr=r["instr"],
                   exact=norm_code(pred) == norm_code(r["after_text"]), changed=norm_code(pred) != norm_code(r["before_text"]),
                   pred=pred)
        if finals is not None and "after_ids" in r:
            row["exact_ids"] = finals[i] == r["after_ids"]
            row["after_len"], row["before_len"] = len(r["after_ids"]), len(r["before_ids"])
            # length of the AR training sequence of this pair (instruction ":" before SEP EDIT after STOP): the same
            # for both layouts, so "fits the training max_len" subsets are identical across D4 and AR models
            row["ar_seq_len"] = len(r["instruction_ids"]) + len(r["before_ids"]) + len(r["after_ids"]) + 3
        if extra:
            row.update(extra[i])
        if results:
            row.update({k: v for k, v in results[i].items()})
        rows.append(row)
    return rows


def aggregate_code(rows, train_max_len: int = 1024) -> Dict:
    def agg(rs):
        n = len(rs)
        out = dict(n=n)
        if "passed" in rs[0]:
            k = sum(r["passed"] for r in rs)
            out.update(pass1=k / n, pass1_ci=wilson(k, n), status=dict(Counter(r["status"] for r in rs).most_common()))
        for key in ("exact", "exact_ids", "exact_ws", "changed", "stopped", "n_actions", "n_del", "n_move", "n_ins",
                    "gen_seconds", "exec_seconds", "prompt_len", "sari", "edit_p", "edit_r"):
            if key in rs[0]:
                out[key] = sum(float(r[key]) for r in rs) / n
        if "prompt_len" in rs[0]:
            out[f"prompt_len_gt{train_max_len}"] = sum(r["prompt_len"] > train_max_len for r in rs) / n
        return out
    by = defaultdict(list)
    for r in rows:
        by[r["intent"]].append(r)
    out = {"ALL": agg(rows), **{f"change_kind/{k}": agg(v) for k, v in sorted(by.items())}}
    if "ar_seq_len" in rows[0]:  # problems whose AR sequence fits the training length (length-generalisation control)
        fit = [r for r in rows if r["ar_seq_len"] <= train_max_len]
        if fit:
            out[f"subset/ar_seq_len_le{train_max_len}"] = agg(fit)
    return out


def _write(od: Path, name: str, rows, summ, meta):
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"meta": meta, "summary": summ}, open(od / f"{name}.summary.json", "w"), indent=1)
    s = summ["ALL"]
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items() if k not in ("pass1_ci", "status")}),
          json.dumps(s.get("status", {})))


def rescore(a):
    """Re-execute saved CanItEdit predictions (same text, new resource limits); scores are recomputed from scratch."""
    df, dinfo = load_canitedit(a.parquet, Path(a.code_dir))
    tag = a.tag or f"mem{a.mem_mb}"
    for f in a.rescore:
        f = Path(f)
        old = [json.loads(l) for l in open(f, encoding="utf-8")]
        assert f.name.endswith(".jsonl"), f
        smeta = json.load(open(f.parent / (f.name[:-len(".jsonl")] + ".summary.json")))["meta"]
        v = old[0]["instr"]
        recs = {r["uid"]: r for r in canitedit_records(df, v)}
        per = [recs[r["uid"]] for r in old]
        t1 = time.time()
        results = execute_all([r["pred"] + "\n\n" + p["tests"] for r, p in zip(old, per)], a.timeout, a.mem_mb, a.exec_workers)
        keep = ("exact_ids", "after_len", "before_len", "ar_seq_len", "stopped", "n_actions", "n_del", "n_move", "n_ins",
                "gen_seconds", "prompt_len", "sari", "edit_p", "edit_r")
        rows = []
        for r, p, res in zip(old, per, results):
            row = score_canitedit([p], [r["pred"]])[0]
            row.update({k: r[k] for k in keep if k in r})
            row.update(res)
            rows.append(row)
        summ = aggregate_code(rows, a.pool_max_len)
        limits = dict(timeout=a.timeout, mem_mb=a.mem_mb if os.name == "posix" else None, exec_workers=a.exec_workers,
                      python=sys.executable, python_version=sys.version.split()[0], platform=sys.platform, executed=True)
        meta = dict(smeta, rescored_from=str(f), rescore_limits=limits, limits=limits, dataset=dinfo,
                    exec_wall_seconds=time.time() - t1)
        _write(f.parent, f"{f.name[:-len('.jsonl')]}_{tag}", rows, summ, meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["canitedit", "code_dev"], default="canitedit")
    ap.add_argument("--instr", nargs="+", choices=list(INSTR_COLS), default=["descriptive", "lazy"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", choices=["copy", "reference"], default=None)
    ap.add_argument("--rescore", nargs="+", default=None,
                    help="re-execute the saved predictions of existing code_canitedit_*.jsonl files under the given "
                         "--timeout/--mem-mb (no generation); writes <name>_<tag or mem{N}>.jsonl next to each input")
    ap.add_argument("--layout", choices=["edit", "ar"], default="edit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8, help="rollout batch (prefill logits: B x T0 x 151,667)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-extra", type=int, default=512, help="action budget beyond 3x (edit) / 2x (ar) len(before)")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per program (wall clock)")
    ap.add_argument("--mem-mb", type=int, default=2048, help="RLIMIT_AS per program on Linux (0 = no limit)")
    ap.add_argument("--exec-workers", type=int, default=4)
    ap.add_argument("--no-exec", action="store_true", help="generate and score text only (no program execution)")
    ap.add_argument("--parquet", default=None, help="CanItEdit parquet (default data/code/canitedit_test.parquet)")
    ap.add_argument("--code-dir", default=str(CODE_DIR))
    ap.add_argument("--pool-tag", default="qwen3")
    ap.add_argument("--pool-max-len", type=int, default=1024, help="prep_code --max-len of the pools = training max_len; "
                    "also the length of the fits-in-training subset reported for CanItEdit")
    ap.add_argument("--out-dir", default=None, help="default: <run/ckpt dir>/eval or runs/code_baselines")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.rescore:
        return rescore(a)
    assert bool(a.ckpt) != bool(a.baseline), "give exactly one of --ckpt / --baseline / --rescore"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    t0 = time.time()
    if a.ckpt:
        p = Path(a.ckpt)
        od = Path(a.out_dir) if a.out_dir else (p if p.is_dir() else p.parent) / "eval"
        dec = ("tgreedy" if a.type_greedy else "greedy") if a.temperature <= 0 else f"T{a.temperature}s{a.sample_seed}"
    else:
        od = Path(a.out_dir) if a.out_dir else ROOT / "runs" / "code_baselines"
        dec = a.baseline
    suffix = ("" if a.layout == "edit" or a.baseline else f"_{a.layout}") + (f"_lim{a.limit}" if a.limit else "") + \
        (f"_{a.tag}" if a.tag else "")
    limits = dict(timeout=a.timeout, mem_mb=a.mem_mb if os.name == "posix" else None, cpu_s=int(a.timeout) + 1 if os.name == "posix" else None,
                  exec_workers=a.exec_workers, python=sys.executable, python_version=sys.version.split()[0], platform=sys.platform,
                  executed=not a.no_exec)

    if a.data == "code_dev":
        assert a.ckpt, "code_dev scores a checkpoint (copy/reference are trivial there)"
        from .evaluate_edits import score
        from .hf_backbone import _Decoder
        tok, mc = checkpoint_tokenizer(a.ckpt)
        recs, pmeta = code_dev_records(a.pool_tag, a.pool_max_len, a.limit, Path(a.code_dir))
        from .realdata.retokenize import vocab_fingerprint
        assert pmeta["n_content"] == mc["n_content"] and pmeta["vocab_fingerprint"] == vocab_fingerprint(tok), \
            "dev pool and checkpoint tokenizer differ"
        texts, finals, extra, meta = generate(recs, a.ckpt, tok, a)
        rows = score(recs, finals, _Decoder(tok), extra)
        for r, rec, t in zip(rows, recs, texts):
            r["exact_ws"] = norm_code(t) == norm_code(rec["after_text"])
        summ = aggregate_code(rows)
        meta.update(data="code_dev", pool=pmeta.get("source"), n=len(recs), limit=a.limit, decoding=dec, batch=a.batch,
                    max_extra=a.max_extra, seconds=time.time() - t0)
        _write(od, f"code_dev_{dec}{suffix}", rows, summ, meta)
        return

    df, dinfo = load_canitedit(a.parquet, Path(a.code_dir))
    if a.baseline:
        per = {v: canitedit_records(df, v, limit=a.limit) for v in a.instr}
        texts = {v: [r["before_text"] if a.baseline == "copy" else r["after_text"] for r in per[v]] for v in a.instr}
        finals = extra = {v: None for v in a.instr}
        meta0 = dict(baseline=a.baseline)
    else:
        tok, mc = checkpoint_tokenizer(a.ckpt)
        per = {v: canitedit_records(df, v, tok, mc["n_content"], a.limit) for v in a.instr}
        allrecs = [r for v in a.instr for r in per[v]]  # one model load for all instruction variants
        tx, fi, ex, meta0 = generate(allrecs, a.ckpt, tok, a)
        texts, finals, extra, k = {}, {}, {}, 0
        for v in a.instr:
            n = len(per[v])
            texts[v], finals[v], extra[v] = tx[k:k + n], fi[k:k + n], ex[k:k + n]
            k += n
        meta0.update(batch=a.batch, max_extra=a.max_extra, temperature=a.temperature, type_greedy=a.type_greedy)
    for v in a.instr:
        t1 = time.time()
        results = None
        if not a.no_exec:
            results = execute_all([t + "\n\n" + r["tests"] for t, r in zip(texts[v], per[v])], a.timeout, a.mem_mb, a.exec_workers)
        rows = score_canitedit(per[v], texts[v], finals[v], extra[v], results)
        summ = aggregate_code(rows, a.pool_max_len)
        meta = dict(meta0, data="canitedit", dataset=dinfo, instr=v, n=len(rows), limit=a.limit, decoding=dec,
                    layout=a.layout if a.ckpt else None, limits=limits, exec_wall_seconds=time.time() - t1,
                    seconds=time.time() - t0)
        _write(od, f"code_canitedit_{v}_{dec}{suffix}", rows, summ, meta)


if __name__ == "__main__":
    main()
