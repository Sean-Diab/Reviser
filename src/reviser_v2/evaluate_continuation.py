"""Continuation evaluation (task 28): free-running text continuation from real-edit dev-pool prefixes, scored by a
frozen scorer LM, with a paired comparison across checkpoints.

Prompts (deterministic, ``continuation_prompts``): distinct dev-pool texts (``after`` side, else ``before``) with
>= --min-len GPT-2 tokens, in a fixed shuffled order (seed); prompt = the first 35% of the tokens (round, as the
``ltr`` / ``restoration`` training kinds), reference = the rest. Every model gets prompt + EMPTY canvas: this is the
``ltr`` layout of the matched AR baseline and the ``restoration`` layout of Reviser D-models (they may also
delete / move on their own canvas; the continuation is the final canvas). Fixed action budget (--budget).

Decoding: sampling (default T=0.9, top-p 0.95) or greedy / type-greedy with --temperature 0. The sampling generator
is re-seeded per batch (seed * 100003 + batch index), so with the same --batch every checkpoint sees the same random
stream per prompt position (common random numbers, like decode_eval.py's per-example seeding).

Scores (frozen scorer, --scorer HF name, default "gpt2" = GPT-2 base; downloaded from the HF hub on first use, so run
this on the remote box; --scorer none skips it). Definitions follow experiments/july_control/eval/score_ppl.py:
  july_nll   per example: mean NLL of the continuation tokens given the prompt, both re-tokenised by the scorer
             (encoded separately), last 1024 tokens; aggregate evalPPL_july = exp(mean over non-empty examples).
  paper_ppl  per example: full text = decode(prompt + continuation), scorer tokens truncated to 512, prompt positions
             masked, PPL = exp(mean NLL); aggregate = arithmetic mean (empty continuation -> 1.0, as in the paper
             code; also reported over non-empty only).
  nll_first{L}  length-controlled: mean NLL of the first L continuation scorer tokens (only if >= L exist).
  ref_july_nll  the same July definition on the reference continuation (anchor).
Differences from score_ppl.py: the prompt is read-only conditioning here (it cannot be edited), so "full text" is
always prompt + continuation (July's canvas contained the prefix); prompts are dev-pool texts, not DCLM/C4 docs;
default decoding is nucleus (top-p) sampling because rollout.py has no top-k.
Text statistics on the continuation GPT-2 ids: distinct_n (n = 1..3; per example |unique n-grams| / |n-grams|, mean over
examples with >= 1 n-gram; plus corpus-level distinct_n over all examples), seq_rep_4 = 1 - distinct_4 (Welleck et al.
2020), rep_32 = fraction of tokens that occur among the previous 32 continuation tokens, stop rate, length (tokens),
empty rate, actions and action mix.

Paired comparison (several --ckpt, or --compare on existing jsonl files): per-prompt differences B - A against the first
model, bootstrap 95% CIs (10,000 resamples, seed 0, as compare.py) on paper_ppl, july_nll, nll_first{L}, distinct_2,
seq_rep_4, rep_32, stopped, out_len.

Usage (remote):
  python -m reviser_v2.evaluate_continuation --ckpt /root/rv2/runs/re100-d3-s1 /root/rv2/runs/re100-ar-s1 --out cmp.json
  python -m reviser_v2.evaluate_continuation --compare A.jsonl B.jsonl --out cmp.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PAIR_KEYS = ("paper_ppl", "july_nll", "distinct_2", "seq_rep_4", "rep_32", "stopped", "out_len")


def continuation_prompts(n: int = 300, seed: int = 20260924, prefix_frac: float = 0.35, min_len: int = 40, pool=None):
    if pool is None:
        from .evalsets import dev_pool
        pool = dev_pool()
    seen, texts = set(), []
    for _, b, a in pool:
        t = tuple(a or b)
        if len(t) >= min_len and t not in seen:
            seen.add(t)
            texts.append(list(t))
    random.Random(f"cont:{seed}").shuffle(texts)
    out = []
    for i, t in enumerate(texts[:n]):
        k = int(round(prefix_frac * len(t)))
        out.append(dict(id=f"cont/{i}", prompt_ids=t[:k], ref_ids=t[k:]))
    return out


# ---------------------------------------------------------------------------------------------------- generation
def generate(model, space, prompts, budget: int, device="cpu", batch: int = 32, temperature: float = 0.9,
             top_p: float = 0.95, type_greedy: bool = False, seed: int = 0) -> List[Dict]:
    from .rollout import Request, rollout
    greedy = temperature <= 0
    out = []
    for bi, i in enumerate(range(0, len(prompts), batch)):
        reqs = [Request(prompt=list(p["prompt_ids"]), max_actions=budget) for p in prompts[i:i + batch]]
        gen = None if greedy else torch.Generator(device=device).manual_seed(seed * 100003 + bi)
        for t in rollout(model, space, reqs, device=device, greedy=greedy, temperature=1.0 if greedy else temperature,
                         top_p=top_p, generator=gen, type_greedy=type_greedy, record_pdel=False):
            out.append(dict(cont_ids=[int(x) for x in t.final], stopped=bool(t.stopped), n_actions=len(t.actions),
                            n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move))
    return out


# ---------------------------------------------------------------------------------------------------- text stats
def _distinct(ids: Sequence[int], n: int) -> Optional[float]:
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(g)) / len(g) if g else None


def text_stats(ids: Sequence[int], window: int = 32) -> Dict:
    d = {f"distinct_{n}": _distinct(ids, n) for n in (1, 2, 3)}
    d4 = _distinct(ids, 4)
    d["seq_rep_4"] = None if d4 is None else 1.0 - d4
    d["rep_32"] = (sum(ids[i] in ids[max(0, i - window):i] for i in range(1, len(ids))) / (len(ids) - 1)
                   if len(ids) > 1 else None)
    return d


# ---------------------------------------------------------------------------------------------------- scoring
class HFScorer:
    """Frozen causal LM scorer (HF hub name or local dir). Interface: encode(text) -> ids; token_nlls(ids) -> (T-1,)."""

    def __init__(self, name: str, device: str, dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = {"fp16": torch.float16, "fp32": torch.float32}.get(dtype, torch.float16 if name == "gpt2" else torch.float32)
        if device == "cpu":
            dt = torch.float32
        self.name, self.device, self.dtype = name, device, str(dt)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name).to(device=device, dtype=dt).eval()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    @torch.no_grad()
    def token_nlls(self, ids: Sequence[int]) -> torch.Tensor:
        x = torch.tensor([list(ids)], device=self.device)
        logits = self.lm(x).logits[0, :-1].float()
        return F.cross_entropy(logits, x[0, 1:], reduction="none").cpu()


def load_scorer(name: str, device: str, dtype: str = "auto"):
    return HFScorer(name, device, dtype)


def score_july(sc, prefix_text: str, cont_text: str, L: int):
    if not cont_text.strip():
        return None, 0, None
    p_ids, c_ids = sc.encode(prefix_text), sc.encode(cont_text)
    if not c_ids:
        return None, 0, None
    ids = (p_ids + c_ids)[-1024:]
    n_cont = min(len(c_ids), len(ids) - 1)
    if n_cont <= 0:
        return None, 0, None
    nll = sc.token_nlls(ids)[-n_cont:]
    return float(nll.mean()), int(n_cont), (float(nll[:L].mean()) if n_cont >= L else None)


def score_paper(sc, prompt_text: str, full_text: str):
    full = sc.encode(full_text)[:512]
    pl = min(len(sc.encode(prompt_text)), len(full))
    if len(full) < 2:
        return 0.0, 0
    valid = sc.token_nlls(full)[max(0, pl - 1):]
    return (float(valid.mean()), int(valid.numel())) if valid.numel() else (0.0, 0)


def score_rows(rows: List[Dict], tok, scorer, first_L: int = 32):
    for r in rows:
        r.update(text_stats(r["cont_ids"]))
        r["out_len"] = len(r["cont_ids"])
        if scorer is None:
            continue
        ptxt = tok.decode(r["prompt_ids"])
        j, jn, jf = score_july(scorer, ptxt, tok.decode(r["cont_ids"]), first_L)
        rj, _, rjf = score_july(scorer, ptxt, tok.decode(r["ref_ids"]), first_L)
        pn, pc = score_paper(scorer, ptxt, tok.decode(list(r["prompt_ids"]) + list(r["cont_ids"])))
        r.update(july_nll=j, july_n=jn, **{f"nll_first{first_L}": jf, f"ref_nll_first{first_L}": rjf},
                 ref_july_nll=rj, paper_nll=pn, paper_n=pc, paper_ppl=float(np.exp(pn)))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(rows: List[Dict], first_L: int = 32) -> Dict:
    ol = np.array([r["out_len"] for r in rows])
    acts = sum(r["n_actions"] for r in rows) or 1
    s = dict(n=len(rows), stop_rate=_mean([float(r["stopped"]) for r in rows]),
             out_len=dict(mean=float(ol.mean()), median=float(np.median(ol)), p10=float(np.percentile(ol, 10)),
                          p90=float(np.percentile(ol, 90))),
             empty_rate=float((ol == 0).mean()), ref_len_mean=_mean([len(r["ref_ids"]) for r in rows]),
             n_actions_mean=_mean([r["n_actions"] for r in rows]),
             action_mix={k: sum(r[f"n_{k}"] for r in rows) / acts for k in ("ins", "del", "move")})
    for k in ("distinct_1", "distinct_2", "distinct_3", "seq_rep_4", "rep_32"):
        s[k] = _mean([r[k] for r in rows])
    for n in (1, 2, 3):
        g = [tuple(r["cont_ids"][i:i + n]) for r in rows for i in range(len(r["cont_ids"]) - n + 1)]
        s[f"corpus_distinct_{n}"] = len(set(g)) / len(g) if g else None
    if "paper_ppl" in rows[0]:
        jn = [r["july_nll"] for r in rows if r["july_nll"] is not None]
        fk = f"nll_first{first_L}"
        first = [r[fk] for r in rows if r.get(fk) is not None]
        rj = [r["ref_july_nll"] for r in rows if r["ref_july_nll"] is not None]
        s.update({"evalPPL_july_def": float(np.exp(np.mean(jn))) if jn else None,
                  "evalPPL_paper_def": _mean([r["paper_ppl"] for r in rows]),
                  "evalPPL_paper_def_nonempty": _mean([r["paper_ppl"] for r in rows if r["paper_n"] > 0]),
                  f"ppl_first{first_L}": float(np.exp(np.mean(first))) if first else None, f"n_first{first_L}": len(first),
                  "ref_evalPPL_july_def": float(np.exp(np.mean(rj))) if rj else None})
    return s


def boot(x, n: int = 10000, seed: int = 0):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (n, x.size))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(x.size)}


def paired(rows_a: List[Dict], rows_b: List[Dict], first_L: int = 32, n_boot: int = 10000) -> Dict:
    """B - A on common prompt ids: per-key bootstrap CI of the mean per-prompt difference (None-valued pairs dropped)."""
    ra, rb = {r["id"]: r for r in rows_a}, {r["id"]: r for r in rows_b}
    ids = sorted(set(ra) & set(rb), key=lambda s: int(s.split("/")[-1]) if s.split("/")[-1].isdigit() else s)
    out = {"n_common": len(ids)}
    for k in PAIR_KEYS + (f"nll_first{first_L}",):
        d = [float(rb[i][k]) - float(ra[i][k]) for i in ids
             if rb[i].get(k) is not None and ra[i].get(k) is not None]
        out[k] = boot(d, n_boot)
    return out


def _load_jsonl(path):
    meta, rows = None, []
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        if "_meta" in o:
            meta = o["_meta"]
        else:
            rows.append(o)
    return meta, rows


def compare(results: List[tuple], first_L: int) -> Dict:
    """results: [(name, rows)]; every later model against the first."""
    (na, ra), rep = results[0], {"reference": results[0][0], "pairs": {}}
    for nb, rb in results[1:]:
        rep["pairs"][f"{nb} - {na}"] = paired(ra, rb, first_L)
    return rep


def _run_name(ckpt: str) -> str:
    p = Path(ckpt)
    if p.is_dir():
        return p.name
    return p.parent.parent.name if p.parent.name == "ckpt" else p.stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", nargs="*", default=[], help="run dirs (ckpt/latest.pt) or .pt files; >1 -> paired comparison")
    ap.add_argument("--compare", nargs="*", default=[], help="existing cont_*.jsonl outputs to compare (first = reference)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260924, help="prompt-set seed")
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--prefix-frac", type=float, default=0.35)
    ap.add_argument("--budget", type=int, default=128, help="max actions per continuation (all action types)")
    ap.add_argument("--temperature", type=float, default=0.9, help="<= 0: greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--scorer", default="gpt2", help="HF causal-LM name or local dir for the frozen scorer; 'none' to skip")
    ap.add_argument("--scorer-dtype", choices=["auto", "fp16", "fp32"], default="auto")
    ap.add_argument("--first-L", type=int, default=32)
    ap.add_argument("--out-dir", default=None, help="shared output dir (files prefixed with the run name); default <run>/eval")
    ap.add_argument("--out", default=None, help="paired-comparison json (default: next to the first output)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    assert bool(a.ckpt) != bool(a.compare), "give --ckpt or --compare"
    if a.compare:
        loaded = [_load_jsonl(p) for p in a.compare]
        rep = compare([(m["name"], r) for m, r in loaded], a.first_L)
        rep["files"] = a.compare
        out = Path(a.out or Path(a.compare[0]).with_name("cont_compare.json"))
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep["pairs"], indent=1))
        return rep
    from .evalsets import load_gpt2_tokenizer
    from .evaluate_edits import load_model
    tok = load_gpt2_tokenizer(offline=False)   # local files; must not force HF offline mode (scorer may download)
    prompts = continuation_prompts(a.n, a.seed, a.prefix_frac, a.min_len)
    scorer = None if a.scorer == "none" else load_scorer(a.scorer, a.device, a.scorer_dtype)
    dec = ("tgreedy" if a.type_greedy else "greedy") if a.temperature <= 0 else f"T{a.temperature}p{a.top_p}s{a.sample_seed}"
    stag = "noscore" if scorer is None else Path(a.scorer).name
    results, paths = [], []
    for ck in a.ckpt:
        t0 = time.time()
        model, space, meta_ck = load_model(ck, a.device)
        gens = generate(model, space, prompts, a.budget, a.device, a.batch, a.temperature, a.top_p, a.type_greedy, a.sample_seed)
        del model
        rows = [dict(id=p["id"], prompt_ids=p["prompt_ids"], ref_ids=p["ref_ids"], **g) for p, g in zip(prompts, gens)]
        score_rows(rows, tok, scorer, a.first_L)
        name = _run_name(ck)
        meta = dict(name=name, ckpt=str(ck), ckpt_step=meta_ck.get("step"), n=len(rows), seed=a.seed, min_len=a.min_len,
                    prefix_frac=a.prefix_frac, budget=a.budget, decoding=dec, temperature=a.temperature, top_p=a.top_p,
                    sample_seed=a.sample_seed, batch=a.batch, scorer=a.scorer,
                    scorer_dtype=None if scorer is None else getattr(scorer, "dtype", None), first_L=a.first_L,
                    seconds=time.time() - t0)
        summ = summarize(rows, a.first_L)
        p = Path(ck)
        od = Path(a.out_dir) if a.out_dir else (p if p.is_dir() else p.parent.parent if p.parent.name == "ckpt" else p.parent) / "eval"
        od.mkdir(parents=True, exist_ok=True)
        fn = od / ((f"{name}." if a.out_dir else "") + f"cont_dev_n{a.n}_{dec}_b{a.budget}_{stag}.jsonl")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        json.dump({"meta": meta, "summary": summ}, open(fn.with_suffix(".summary.json"), "w"), indent=1)
        print(fn.name, name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()
                                         if not isinstance(v, dict)}))
        results.append((name, rows))
        paths.append(str(fn))
    if len(results) > 1:
        rep = compare(results, a.first_L)
        rep["files"] = paths
        out = Path(a.out or Path(paths[0]).with_name(f"cont_compare_{dec}_b{a.budget}_{stag}.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep["pairs"], indent=1))
    return paths


if __name__ == "__main__":
    main()
