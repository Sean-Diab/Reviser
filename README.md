# Reviser

Code, configs and model outputs for the paper
**Reviser: Writing and Editing Text with Autoregressive Cursor Actions** (Sean Diab).

Reviser is a decoder-only Transformer that writes and edits text with cursor actions (insert, move, delete,
stop) that a deterministic executor applies to a canvas. It is autoregressive over its own actions, so a step
costs the same as a step of an autoregressive (AR) model.

- Paper: [arXiv:2609.20830](https://arxiv.org/abs/2609.20830). This code is for version 2 of the paper.
- Checkpoints: [huggingface.co/sean-diab/reviser-checkpoints](https://huggingface.co/sean-diab/reviser-checkpoints)
  (the models of this version are under `v2/`; see [Checkpoints](#checkpoints)).
- Step-by-step replays of 70 example trajectories:
  [sean-diab.github.io/reviser_trajectories.html](https://sean-diab.github.io/reviser_trajectories.html)
  (the same page is `samples/example_viewer.html`).
- Live demo: [huggingface.co/spaces/sean-diab/reviser-demo](https://huggingface.co/spaces/sean-diab/reviser-demo).
- Code of the first version of the paper (arXiv v1): the [`v1` branch](https://github.com/Sean-Diab/Reviser/tree/v1).

## What is here

- `samples/` has every output the paper scores: the 3,000 continuations of each model in Table 1, the edit
  outputs of Table 5, and the example replay page. Start with `samples/README.md`. It also shows that the files
  reproduce Table 1 and Table 5.
- `data/papereval/c4_paper_3k.json` holds the 3,000 C4 prompts of Table 1 (35 GPT-2 tokens each).
- Everything else is code and configs, arranged as in the development repository so the imports work.

The prepared training data are not included (size); every script takes local paths, and the scripts that
build the data are included (see the table below). Docstrings often cite internal notes (`notes/...`) and task
numbers. Those notes are the project's lab log and are not included; the docstrings themselves describe what
each script does.

## Checkpoints

All files are in [sean-diab/reviser-checkpoints](https://huggingface.co/sean-diab/reviser-checkpoints).
Each file holds the weights and the small config the code reads (no optimizer state).
`v2/SHA256SUMS` lists their hashes, and `v2/manifest.json` gives for each file the SHA-256 of the training
checkpoint it was made from, which is the `checkpoint_sha256` recorded for the outputs in
`samples/continuation/models.json`.

| File | Model | Paper |
|---|---|---|
| `reviser_100m.pt`, `reviser_300m.pt` | Reviser 100M / 300M of the first version ("released") | Table 1 |
| `v2/continuation/reviser-100m-runjump.pt` | 100M + run-and-jump | Table 1 |
| `v2/continuation/reviser-100m-runjump-delete.pt` | 100M + run-and-jump + Delete | Table 1 |
| `v2/continuation/reviser-100m-runjump-delete-rl-seed{1,2,3}.pt` | 100M + RL, seeds 1, 2, 3 | Table 1 |
| `v2/continuation/reviser-300m-runjump-delete.pt` | 300M + run-and-jump + Delete | Table 1 |
| `v2/continuation/reviser-300m-runjump-delete-rl-seed3.pt` | 300M + RL, seed 3 | Table 1 |
| `v2/continuation/ar-100m.pt`, `ar-300m.pt` | AR 100M / 300M baselines | Table 1 |
| `v2/editing/reviser-edit-100m.pt`, `reviser-edit-300m.pt` | Reviser-Edit 100M / 300M (seed 1) | Table 5 |
| `v2/editing/ar-rewriter-100m.pt`, `ar-rewriter-300m.pt` | AR rewriter 100M / 300M (seed 1) | Table 5 |

The continuation files load with `scripts/dagger/reviser_lm.py` (`load_model`); the 300M ones through
`scripts/humanorder/big608_300m/arch_shim.py` (300M shape). The first-version files are converted for this code by
`experiments/july_control/code/july_impl/wrap_ckpt.py` (100M) and
`scripts/humanorder/big608_300m/wrap_ckpt_300m.py` (300M). The editing files load with
`src/reviser_v2/evaluate_edits.py` (`load_model`).

```bash
python -c "from huggingface_hub import hf_hub_download as d; print(d('sean-diab/reviser-checkpoints', 'v2/continuation/reviser-100m-runjump-delete-rl-seed3.pt', local_dir='checkpoints'))"

# Decode the 3,000 C4 prompts as in Table 1 (T 0.9, top-k 50, at most 220 new actions, prompt locked)
python scripts/papereval/decode_batched.py --ckpt checkpoints/v2/continuation/reviser-100m-runjump-delete-rl-seed3.pt \
  --name rl3 --lock_prompt --out_dir DEC
# 300M models: the same command through the 300M shape shim
python scripts/humanorder/big608_300m/arch_shim.py scripts/papereval/decode_batched.py \
  --ckpt checkpoints/v2/continuation/reviser-300m-runjump-delete-rl-seed3.pt --name rl300 --lock_prompt --out_dir DEC
```

The AR baselines decode with the same command without `--lock_prompt` (they only append tokens). Add `--n 10`
for a quick test; `--device cpu` works without a GPU.

## Layout

There are two code bases, one for each model family in the paper.

**Continuation models (Reviser; Sections 3–7, Tables 1–4, 7 and 8).**
All continued training of the first-version Reviser models uses the trainer in
`experiments/july_control/code/src/reviser` (model: `model/`, data loading: `data/`, trainer:
`training/pretrain_resttraj.py`); `experiments/july_control/code/july_impl/wrap_ckpt.py` and
`scripts/humanorder/big608_300m/wrap_ckpt_300m.py` load the first-version checkpoints into it. The training
runs build their trainer tree from it in three steps.
Each step is a small patch that checks the SHA-256 of its inputs and outputs:

1. `experiments/july_control/patches/pretrain_resttraj.patched.py`: stop at a token count, snapshots,
   seeding (the diff is next to it).
2. `scripts/humanorder/promptlock_patch_v1/` (built with its `make_tree.sh`): the prompt-locked loss,
   on top of `scripts/humanorder/lossmask_patch_v1/` (no loss on the prompt).
3. `scripts/humanorder/big608/ddp_patch.py`: multi-GPU training with the same data order and global
   batch as one GPU, and exact resume.

`scripts/humanorder/big608_300m/` adapts the 300M model (architecture shim, checkpoint wrapper, loss
scaler).

**Editing models (Reviser-Edit and the AR rewriter; Section 8, Table 5).**
`src/reviser_v2/` is a separate, self-contained implementation: action vocabulary (`actions.py`),
executor (`executor.py`), model (`model.py`), training (`train.py`), batched rollout with legality
masks (`rollout.py`), edit evaluation (`evaluate_edits.py`, `metrics_edit.py`) and the training mix
(`tasks/editmix.py`). It also holds code for earlier experiments that are not in the paper (star graph,
code editing, toy tasks).

## Where each result comes from

| Paper | Files |
|---|---|
| Training orders: run-and-jump, Delete (Sec. 4; App. C.1, Algorithm 3) | `scripts/humanorder/variants.py`, `runjump6.py`, `build_variant.py`, `orders.py`, `visit_order.py`; the original run-and-jump policy `experiments/july_control/code/july_impl/gen_runjump.py` |
| 608M-token training runs (Table 1) | `configs/humanorder/big608/train_jd608_pl.yaml` (100M), `configs/humanorder/big608_300m/train_jd608_300m_pl.yaml` (300M) |
| Ablations at 84M tokens (Table 4) | one spec per variant in `configs/humanorder/overnight/*.json`, built by `scripts/humanorder/build_variant.py` |
| RL fine-tuning (App. D) | `scripts/grpo/train_grpo.py` |
| C4 prompts and decoding (Sec. 5) | `scripts/papereval/build_prompts.py`, `scripts/papereval/decode_batched.py` (`--lock_prompt`); prompts in `data/papereval/c4_paper_3k.json` |
| Length, stops, moves, deletes (Table 1) | `scripts/papereval/dec_stats.py` |
| Trajectory statistics (Table 3; its "Backward fill" row is `backward_revision_share`) | `scripts/trajviz/trajstats.py`, `part1_stats.py` (models in `configs/trajviz/models.json`), `scripts/papereval/opening_run.py` (opening run); edit traces: `scripts/trajviz/edit_decode.py`, `edit_stats.py` |
| GPT-2 / GPT-2 Large / Dream-7B evalPPL, MAUVE (Tables 1, 7) | `scripts/papereval/score_ppl.py`, `score_dream.py`, `score_mauve.py`, `mauve_seeds.py` |
| Skywork win rates, matched length (Tables 1, 2) | `scripts/papereval/arena_skywork.py`, `length_control.py`, `scripts/lmcompare/` |
| Claude and Codex judges (App. B) | `scripts/papereval/judge_cli.py`, `judge_paper_items.py`, `scripts/judge_continuations.py` |
| Duplicate outputs and diversity (App. E) | `scripts/papereval/paper_dedup.py`, `diversity_stats.py` |
| What the models delete (App. F, Table 8) | `scripts/dagger/probe_deletes.py`, `probe_deletes_paired.py`, `scripts/deleteprobe/` |
| Gaps filled and P(Delete) (Table 4) | `scripts/eval/cursor_skill.py`, `cursor_skill_table.py` |
| Example pools for the viewer and App. J | `scripts/trajviz/select_paper_examples.py` |
| Edit training data (App. C.2) | `scripts/editdata/` (source pools, LLM drafts, reviewer, Haiku batches, final mix `build_mix_v3.py`), configs in `configs/editdata/` |
| Edit training (Sec. 8) | `configs/realedit_v3/` (seed 1) and `configs/realedit_v3_s2/` (seed 2): `main*.yaml` = Reviser-Edit, `ar*.yaml` = AR rewriter |
| Edit scores (Table 5) | `src/reviser_v2/evaluate_edits.py`, `scripts/errant_eval.py` (ERRANT), `scripts/editextras/jfleg_eval.py`, `jfleg_gleu.py`, `scripts/paired_edits.py` (paired tests), `scripts/editextras/judge_edits.py` (Codex judge), `lm_editors.py` (small instruction-tuned LMs) |
| Figure 1, measured FLOPs (App. A.2) | `scripts/benchmarks/measure_continuation.py` (Reviser vs. AR on C4), `fairseq_nat_flops.py` (LevT, Mask-Predict), `owt_family.py` (SEDD, MDLM), `large_models.py` (Dream, LLaDA), `sdlm_flops.py`, `sdlm_compare.py` (SDLM); counter `flopcount.py` (checked by `validate_flopcount.py`); records and model manifests in `experiments/measured_flops/` (`results.json`) |
| Figure 1, analytic FLOPs (Apps. H, I) | `scripts/benchmarks/analytic_flops_released_settings.py` |
| Measured cost (Sec. 9) | `scripts/speed/bench_speed.py`, `scripts/humanorder/big608_300m/bench_speed300.py`, `scripts/speed/bench_edit_speed.py` |

## Main commands

Paths to data and checkpoints are arguments or config fields; point them at local copies.

```bash
# Continued training of a Reviser continuation model (608M tokens, 4 GPUs). TREE = the patched trainer
# tree built above; the trajectory shards are set by data.resttraj_dir in the config.
PYTHONPATH=TREE python -m torch.distributed.run --standalone --nproc_per_node 4 \
  -m reviser.training.pretrain_resttraj --config configs/humanorder/big608/train_jd608_pl.yaml \
  --output_dir OUT --device cuda --resume RELEASED_100M.pt --resume_reset_step --resume_reset_optimizer

# RL fine-tuning
# (the paper's settings; the script's own defaults differ; seeds 1, 2, 3 at 100M, 3 at 300M)
python scripts/grpo/train_grpo.py --init CKPT --reward_lm QWEN2.5-3B-INSTRUCT_DIR --out_dir OUT \
  --steps 200 --P 16 --G 8 --micro 16 --lr 5e-6 --kl_coef 0.2 --w_rep 8.0 --w_short 1.0 --min_len 150 \
  --w_move 2.0 --move_floor -1 --seed S --save_at 100 --lock_prompt

# Decode the 3,000 C4 prompts (T 0.9, top-k 50, at most 220 new actions, prompt locked)
python scripts/papereval/decode_batched.py --ckpt CKPT --name NAME --lock_prompt --out_dir DEC

# Score
python scripts/papereval/score_ppl.py --scorer gpt2 --out_dir SCORES DEC/*.jsonl
python scripts/papereval/score_mauve.py --out_dir SCORES DEC/*.jsonl
python scripts/papereval/arena_skywork.py --base DEC/ar100.jsonl --out_dir ARENA DEC/rev100.jsonl

# Reviser-Edit: train and evaluate (run from the repository root with PYTHONPATH=src)
python -m reviser_v2.train --config configs/realedit_v3/main100.yaml
python -m reviser_v2.evaluate_edits --data coedit --split validation --ckpt checkpoints/v2/editing/reviser-edit-100m.pt --type-greedy
```

Training runs are fully described by their configs. The RL settings are listed in Appendix D, and each
script documents its flags in `--help`.

## Tests

```bash
PYTHONPATH=src:experiments/july_control/code/src python -m pytest tests -q
```

Unit tests for the executor, trajectory builders, edit metrics, rollout, the training mix, the
prompt-lock patch and the trajectory statistics. On Linux (Python 3.10, CPU) this gives 191 passed and
28 skipped. Skipped tests need files that are not included (the GPT-2 tokenizer copy, the built data pools)
or `tiktoken`. One is Windows-only: the Haiku batch client refuses to run elsewhere.

## Requirements

Python 3.10+, `requirements.txt`. We used torch 2.6.0, transformers 5.13.0 and numpy 2.2.6 on Linux
with CUDA GPUs.

## License

The code is released under the Apache-2.0 license (`LICENSE`). The second-version checkpoints are released
under CC BY-NC 4.0 (non-commercial; see the [model card](https://huggingface.co/sean-diab/reviser-checkpoints)).
The sample outputs are model outputs on inputs from C4, CoEdIT, IteraTeR and JFLEG, which keep their own
licenses (JFLEG: CC-BY-NC-SA-4.0). The first version's code on the `v1` branch keeps its MIT license.

## Citation

```bibtex
@misc{diab2026reviser,
  title         = {Reviser: Writing and Editing Text with Autoregressive Cursor Actions},
  author        = {Sean Diab},
  year          = {2026},
  eprint        = {2609.20830},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.20830}
}
```
