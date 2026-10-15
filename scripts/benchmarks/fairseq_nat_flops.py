"""Task 41 rows 3a-3d (LevT) and 6 (CMLM / Mask-Predict): measured-trained FLOPs, realized iterations, BLEU and CPU latency
of released translation checkpoints, run with their own 2019-2020 fairseq forks.

Environment (built by scripts/wsl/fairseq_env.sh; CPU only, torch 1.13.1, /opt/venv untouched). Put the fork on PYTHONPATH:
  PYTHONPATH=tmp/fairseq/constrained-levt tmp/fairseq/env/bin/python scripts/benchmarks/fairseq_nat_flops.py --fork levt ...
  PYTHONPATH=tmp/fairseq/Mask-Predict     tmp/fairseq/env/bin/python scripts/benchmarks/fairseq_nat_flops.py --fork cmlm ...

Per sentence (batch 1, the first --n lines of newstest2014, already tokenized and BPE-segmented as each model expects):
  1. counted pass: the fork's own decoder under FlopCounter (scripts/benchmarks/flopcount.py), stages "encoder" and
     "decoder". LevT uses fairseq's IterativeRefinementGenerator (adaptive stop, eos penalty 0, beam 1, as in the
     constrained-levt README) and records the realized number of decoder iterations. CMLM uses the Mask-Predict authors'
     generate() + MaskPredict strategy, which always runs exactly T iterations over every length candidate.
  2. timed pass: the same decode, uninstrumented (time.perf_counter); it must give the same tokens.
  3. denominator: a same-shape fairseq encoder-decoder AR (`transformer` with the checkpoint's layers, widths, heads,
     embedding sharing and dictionary; randomly initialised, seed 0), forced to emit the NAR's own output tokens + EOS
     with incremental (KV-cached) greedy steps. FLOPs of this execution depend only on source and output lengths, not on
     the weights, so it is the length-matched AR cost on the same sentence (as `ar_mt` in panel S). Its quality is not
     measured; published AR base BLEU is quoted in the results note instead.
BLEU: tokenized BLEU (sacrebleu tokenize='none' on BPE-removed, Moses-tokenized text, as fairseq --remove-bpe and the
papers), with a 1000-sample bootstrap 95% CI over sentences. On 200 sentences it is a sanity check, not a leaderboard number.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flopcount import FlopCounter  # noqa: E402


def self_check():
    """FlopCounter was validated on torch 2.6; re-check the operators this old stack dispatches under torch 1.13."""
    fc = FlopCounter(vocab_sizes=[1000])
    lin, x = torch.nn.Linear(512, 2048), torch.randn(1, 7, 512)
    a, b = torch.randn(8, 7, 64), torch.randn(8, 64, 9)
    head = torch.randn(1000, 512)
    with torch.no_grad(), fc:
        lin(x); torch.bmm(a, b); torch.nn.functional.linear(x, head)
    t = fc.totals()
    exp = dict(linear=2 * 7 * 512 * 2048, attn=2 * 8 * 7 * 64 * 9, head=2 * 7 * 512 * 1000)
    got = {k: t[k] for k in exp}
    assert got == exp, (got, exp)
    return dict(ok=True, expected=exp, uncovered=dict(fc.uncovered))


def load(ck, data, task_name, fork):
    from fairseq import checkpoint_utils, options, tasks
    extra = ['--task', task_name, '--cpu', '-s', 'en', '-t', 'de']
    args = options.parse_args_and_arch(options.get_generation_parser(), [data, '--path', ck] + extra)
    task = tasks.setup_task(args)
    if fork == 'levt':
        models, margs = checkpoint_utils.load_model_ensemble([ck], task=task)
    else:
        from fairseq import utils
        models, margs = utils.load_ensemble_for_inference([ck], task)
    m = models[0].eval()
    return task, m, margs


def build_ar(margs, data, fork, seed=0):
    """Same-shape encoder-decoder AR (fairseq `transformer`), random init."""
    from fairseq import options, tasks
    from fairseq.models.transformer import TransformerModel
    targs = ['--arch', 'transformer', '--task', 'translation', '-s', 'en', '-t', 'de',
             '--encoder-layers', str(margs.encoder_layers), '--decoder-layers', str(margs.decoder_layers),
             '--encoder-embed-dim', str(margs.encoder_embed_dim), '--decoder-embed-dim', str(margs.decoder_embed_dim),
             '--encoder-ffn-embed-dim', str(margs.encoder_ffn_embed_dim),
             '--decoder-ffn-embed-dim', str(margs.decoder_ffn_embed_dim),
             '--encoder-attention-heads', str(margs.encoder_attention_heads),
             '--decoder-attention-heads', str(margs.decoder_attention_heads)]
    if getattr(margs, 'share_all_embeddings', False):
        targs.append('--share-all-embeddings')
    args = options.parse_args_and_arch(options.get_training_parser(), [data] + targs)
    task = tasks.setup_task(args)
    torch.manual_seed(seed)
    return TransformerModel.build_model(args, task).eval()


def ar_forced(ar, src, hyp, eos):
    """Encoder once, then len(hyp)+1 incremental steps (the last predicts EOS); output head on one position per step."""
    lengths = torch.tensor([src.size(1)])
    enc = ar.encoder(src, src_lengths=lengths)
    prev = torch.cat([torch.tensor([[eos]]), hyp.view(1, -1)], 1)
    inc = {}
    for t in range(prev.size(1)):
        ar.decoder(prev[:, :t + 1], encoder_out=enc, incremental_state=inc)
    return prev.size(1)


class Staged:
    """Tag everything run inside model.encoder as stage 'encoder' (the rest of the decode is 'decoder')."""

    def __init__(self, module, fc):
        self.module, self.fc, self.orig = module, fc, module.forward

    def __enter__(self):
        fc, orig = self.fc, self.orig

        def fwd(*a, **k):
            with fc.stage('encoder'):
                return orig(*a, **k)
        self.module.forward = fwd

    def __exit__(self, *e):
        self.module.forward = self.orig


def decode_levt(model, task, src, max_iter):
    from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
    gen = IterativeRefinementGenerator(task.target_dictionary, eos_penalty=0.0, max_iter=max_iter, adaptive=True)
    sample = {'net_input': {'src_tokens': src, 'src_lengths': torch.tensor([src.size(1)])}}
    hypo = gen.generate([model], sample)[0][0]
    toks = hypo['tokens']
    toks = toks[(toks != task.target_dictionary.bos()) & (toks != task.target_dictionary.eos())]
    return toks, dict(iterations=int(hypo['steps']) + 1)


def decode_cmlm(model, task, src, iters, length_beam):
    import generate_cmlm
    from fairseq import strategies

    class A:
        decoding_strategy, decoding_iterations = 'mask_predict', iters
    strat = strategies.setup_strategy(A)
    hyp = generate_cmlm.generate(strat, {'src_tokens': src, 'src_lengths': torch.tensor([src.size(1)])}, [model],
                                 task.target_dictionary, length_beam, None)[0]
    pad = task.target_dictionary.pad()
    toks = hyp[hyp != pad]
    return toks, dict(iterations=iters, length_beam=length_beam, canvas_len=int(hyp.numel()))


def bleu(hyps, refs, n_boot=1000, seed=0):
    import sacrebleu
    score = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none', force=True).score
    rng, n, bs = random.Random(seed), len(hyps), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bs.append(sacrebleu.corpus_bleu([hyps[i] for i in idx], [[refs[i] for i in idx]], tokenize='none', force=True).score)
    bs.sort()
    return dict(bleu=score, ci95=[bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]], n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fork', choices=['levt', 'cmlm'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True, help='directory with dict.en.txt / dict.de.txt')
    ap.add_argument('--src', required=True, help='BPE source lines (a tab and constraints after it are ignored)')
    ap.add_argument('--ref', required=True, help='tokenized reference lines (BPE removed here if present)')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--iters', type=int, required=True, help='LevT: --iter-decode-max-iter; CMLM: T')
    ap.add_argument('--length-beam', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    check = self_check()
    task, model, margs = load(a.ckpt, a.data, 'translation_lev' if a.fork == 'levt' else 'translation_self', a.fork)
    d = task.source_dictionary
    V = len(task.target_dictionary)
    ar = build_ar(margs, a.data, a.fork)
    srcs = [l.rstrip('\n').split('\t')[0] for l in open(a.src, encoding='utf-8')][:a.n]
    refs = [l.rstrip('\n').replace('@@ ', '') for l in open(a.ref, encoding='utf-8')][:a.n]
    rows, hyps = [], []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(srcs):
        src = d.encode_line(s, add_if_not_exist=False, append_eos=True).long().view(1, -1)
        n_unk = int((src == d.unk()).sum())
        dec = (lambda: decode_levt(model, task, src, a.iters)) if a.fork == 'levt' else \
            (lambda: decode_cmlm(model, task, src, a.iters, a.length_beam))
        fc = FlopCounter(vocab_sizes=[V])
        with torch.no_grad(), fc, Staged(model.encoder, fc):
            toks, info = dec()
        s_nar = fc.summary()
        torch.manual_seed(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            toks2, _ = dec()
        lat = time.perf_counter() - t0
        fa = FlopCounter(vocab_sizes=[V])
        with torch.no_grad(), fa, Staged(ar.encoder, fa):
            ar_steps = ar_forced(ar, src, toks, task.target_dictionary.eos())
        s_ar = fa.summary()
        t0 = time.perf_counter()
        with torch.no_grad():
            ar_forced(ar, src, toks, task.target_dictionary.eos())
        lat_ar = time.perf_counter() - t0
        hyp = task.target_dictionary.string(toks, bpe_symbol='@@ ')
        hyps.append(hyp)
        tot, tot_ar = s_nar['total'], s_ar['total']
        rows.append(dict(i=i, src_len=int(src.size(1)), src_unk=n_unk, out_len=int(toks.numel()), **info,
                         same_tokens=bool(torch.equal(toks, toks2)),
                         flops=tot['total'], matmul=tot['matmul'], head=tot['head'], head_positions=tot['head_positions'],
                         enc_matmul=s_nar['per_stage'].get('encoder', {}).get('matmul', 0.0),
                         ar_flops=tot_ar['total'], ar_matmul=tot_ar['matmul'], ar_steps=ar_steps,
                         ar_enc_matmul=s_ar['per_stage'].get('encoder', {}).get('matmul', 0.0),
                         latency_s=lat, ar_latency_s=lat_ar, uncovered=s_nar['uncovered'], ar_uncovered=s_ar['uncovered'],
                         hyp=hyp))
        r = rows[-1]
        print(f"{a.fork} {i + 1}/{len(srcs)} L_src={r['src_len']} L_out={r['out_len']} it={r['iterations']} "
              f"{r['matmul'] / 1e9:.3f} GF ÷AR {r['matmul'] / r['ar_matmul']:.2f} lat {lat:.3f}s ÷AR {lat / lat_ar:.2f} "
              f"same={r['same_tokens']}", flush=True)
    with open(a.out + '.rows.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n = len(rows)

    def ratio_ci(num, den, n_boot=2000, seed=0):
        rng, vals = random.Random(seed), []
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            vals.append(sum(rows[j][num] for j in idx) / sum(rows[j][den] for j in idx))
        vals.sort()
        return [sum(r[num] for r in rows) / sum(r[den] for r in rows), vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]
    its = [r['iterations'] for r in rows]
    summ = dict(fork=a.fork, ckpt=a.ckpt, src=a.src, ref=a.ref, n=n, iters_arg=a.iters,
                length_beam=a.length_beam if a.fork == 'cmlm' else None, threads=a.threads, torch=torch.__version__,
                flopcounter_self_check=check, vocab=V,
                iterations_mean=sum(its) / n, iterations_hist={str(k): its.count(k) for k in sorted(set(its))},
                out_len_mean=sum(r['out_len'] for r in rows) / n, src_len_mean=sum(r['src_len'] for r in rows) / n,
                src_unk_total=sum(r['src_unk'] for r in rows),
                matmul_ratio_vs_ar=ratio_ci('matmul', 'ar_matmul'), total_ratio_vs_ar=ratio_ci('flops', 'ar_flops'),
                decoder_only_matmul_ratio_vs_ar=None,
                latency_ratio_vs_ar=ratio_ci('latency_s', 'ar_latency_s'),
                gmac_per_sentence=sum(r['matmul'] for r in rows) / n / 2e9,
                ar_gmac_per_sentence=sum(r['ar_matmul'] for r in rows) / n / 2e9,
                all_same_tokens=all(r['same_tokens'] for r in rows),
                uncovered={k: sum(r['uncovered'].get(k, 0) for r in rows) for r in rows for k in r['uncovered']},
                bleu=bleu(hyps, refs))
    for r in rows:
        r['dec_matmul'], r['ar_dec_matmul'] = r['matmul'] - r['enc_matmul'], r['ar_matmul'] - r['ar_enc_matmul']
    summ['decoder_only_matmul_ratio_vs_ar'] = ratio_ci('dec_matmul', 'ar_dec_matmul')
    with open(a.out + '.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summ, f, indent=1)
        f.write('\n')
    print(json.dumps({k: summ[k] for k in ('n', 'iterations_mean', 'iterations_hist', 'matmul_ratio_vs_ar',
                                           'decoder_only_matmul_ratio_vs_ar', 'latency_ratio_vs_ar', 'bleu',
                                           'all_same_tokens', 'src_unk_total')}, indent=1))


if __name__ == '__main__':
    main()
