"""Full-newstest2014 BLEU sanity check for the released LevT / Mask-Predict checkpoints (no FLOP counting, batched).

Checks that the checkpoints, forks and segmentations loaded by fairseq_nat_flops.py reproduce the published quality, so a
200-sentence BLEU from that harness can be read against the right reference point. Same decoders and settings, batch
--batch sentences (source left-padded as in fairseq's LanguagePairDataset), CPU.
  PYTHONPATH=tmp/fairseq/constrained-levt tmp/fairseq/env/bin/python scripts/benchmarks/fairseq_nat_bleu.py --fork levt ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairseq_nat_flops import bleu, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fork', choices=['levt', 'cmlm'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--iters', type=int, required=True)
    ap.add_argument('--length-beam', type=int, default=5)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from fairseq.data import data_utils
    task, model, _ = load(a.ckpt, a.data, 'translation_lev' if a.fork == 'levt' else 'translation_self', a.fork)
    d, td = task.source_dictionary, task.target_dictionary
    srcs = [l.rstrip('\n').split('\t')[0] for l in open(a.src, encoding='utf-8')]
    refs = [l.rstrip('\n').replace('@@ ', '') for l in open(a.ref, encoding='utf-8')]
    hyps, iters, t0 = [None] * len(srcs), [None] * len(srcs), time.time()
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i].split()))
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        toks = [d.encode_line(srcs[i], add_if_not_exist=False, append_eos=True).long() for i in idx]
        src = data_utils.collate_tokens(toks, d.pad(), d.eos(), left_pad=True)
        lens = torch.tensor([t.numel() for t in toks])
        with torch.no_grad():
            if a.fork == 'levt':
                from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
                gen = IterativeRefinementGenerator(td, eos_penalty=0.0, max_iter=a.iters, adaptive=True)
                out = gen.generate([model], {'net_input': {'src_tokens': src, 'src_lengths': lens}})
                for j, i in enumerate(idx):
                    h = out[j][0]['tokens']
                    hyps[i] = td.string(h[(h != td.bos()) & (h != td.eos())], bpe_symbol='@@ ')
                    iters[i] = int(out[j][0]['steps']) + 1
            else:
                import generate_cmlm
                from fairseq import strategies

                class A:
                    decoding_strategy, decoding_iterations = 'mask_predict', a.iters
                out = generate_cmlm.generate(strategies.setup_strategy(A), {'src_tokens': src, 'src_lengths': lens},
                                             [model], td, a.length_beam, None)
                for j, i in enumerate(idx):
                    h = out[j]
                    hyps[i] = td.string(h[h != td.pad()], bpe_symbol='@@ ')
                    iters[i] = a.iters
        print(f'{b + len(idx)}/{len(srcs)} {time.time() - t0:.0f}s', flush=True)
    res = dict(fork=a.fork, ckpt=a.ckpt, src=a.src, ref=a.ref, iters_arg=a.iters,
               length_beam=a.length_beam if a.fork == 'cmlm' else None, n=len(srcs), batch=a.batch,
               iterations_mean=sum(iters) / len(iters), seconds=time.time() - t0,
               bleu_full=bleu(hyps, refs, n_boot=200), bleu_first200=bleu(hyps[:200], refs[:200], n_boot=200))
    with open(a.out + '.hyp', 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(hyps) + '\n')
    with open(a.out + '.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(res, f, indent=1)
        f.write('\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
