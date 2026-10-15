# Task 41: measured inference FLOPs — configurations and pre-stated plan

Assignment: `notes/claude-measured-flops-assignment.md`. Coverage of every manuscript row: `coverage.md` (this directory).
Harness: `scripts/benchmarks/flopcount.py` (execution-based FLOP accounting; validation
`scripts/benchmarks/validate_flopcount.py`, 61/61 closed-form checks on CPU and the RTX 2070 SUPER,
`runs/measured_flops/validation.json` sha256 72f45b24…). Outputs go to `runs/measured_flops/` (not tracked; hashes in
`experiments/artifacts.md`).

Conventions (fixed before any comparative result was examined):
- 2 FLOPs per multiply-add (primary). The manuscript's unit (1 MAC = 1) is `matmul / 2`; ratios are unaffected
  as long as numerator and denominator use the same convention, which every table here does.
- "Dominant matmul" (`flops_matmul`) = linear + output head + attention (causal kernels charged triangular pairs),
  which is the manuscript's accounting scope; `flops_total` adds the supplemental elementwise estimate.
- A response's FLOPs include every executed stage: prompt/draft prefill, all decode calls, output-head positions
  actually projected, and the selection step outside the model.

## Panel E (editing; practical trained-model comparison) — plan stated 2026-09-25 05:15Z, before any panel run

- **Question.** Measured inference FLOPs, latency and quality of the deletion-capable Reviser editor (D4) against
  its matched AR baselines trained on the same data from the same initialisation (AR rewrite = the model re-types the
  whole output; patch AR = the model writes a search/replace edit script), at 100M and 300M.
- **Checkpoints** (`runs/slim/<run>/ckpt/latest.pt`, step 12000): re100-d4-s1, re100-ar-s1, re100-patch-s1;
  re300-d4-s1, re300-ar-s1, re300-patch-s1. Same tokenizer (GPT-2), architecture and action vocabulary within a scale.
- **Decoding.** Type-greedy (the selected decoding of findings 15/19). AR and patch use `--insert-only` (the exact AR
  decoder; task 29). Budgets as in `evaluate_edits` (edit: 3·|draft|+64 actions; AR/patch: 2·|draft|+64).
- **Data.** 200 records per set, seeded random subset (seed 41) of: synth dev (1000), synth_real dev (1100),
  CoEdIT validation (1712), IteraTeR-human sentence test (364). Held-out sets already used in findings 15/19/23 for
  quality; decoding is not tuned on these subsets.
- **Denominator.** For each set and scale, the AR-rewrite checkpoint of the same scale (primary) and patch AR
  (secondary). Ratios are ratios of per-set mean FLOPs per response and also per-record paired ratios.
- **Measured per record.** FLOPs by stage/category, actions, stop/cap, final length, new/changed tokens, batch-1
  latency (separate un-instrumented run, 3 warm-ups, synchronized), peak memory, EM/SARI. Static batch 16 throughput
  and its FLOPs (including padding and idle rows) on the same records.
- **Expected (not a result).** The manuscript's Reviser row (1.25–1.50× AR) assumes both models produce the same
  n-token output from scratch. Editing breaks that assumption both ways: the draft prefill is shared, and the
  editor emits far fewer actions on sparse edits. No direction is predicted for CoEdIT/IteraTeR, where D4 was
  measured to perform about as many actions as AR (task 35).

## Panel S (structural probe of the manuscript's assumptions)

Untrained, randomly initialised reference networks (n = 128, d = 768, L = 12, d_ff = 4d, |V| = 50,000, K_max = 16)
executing each manuscript row's stated decoding schedule as forced traces. These are **structural probes**, never
quality results. Implementation: `scripts/benchmarks/structural_probe.py` (planned).

## Panel D (OWT diffusion family, trained) and Panel L (appendix large models) — plan stated 2026-09-25 ~05:45Z

Harnesses: `scripts/benchmarks/owt_family.py` (backbone re-implementation checked against the released modules by
`scripts/benchmarks/validate_owt_family.py`: identical logits on CPU, max |diff| 0.0 for all five checkpoints; KV-cache paths
≤ 3e-5; official bd3lms samplers ported line by line) and `scripts/benchmarks/large_models.py` (Qwen2.5-7B / Dream-7B /
LLaDA-8B official decoders). Sample counts, configurations, seeds, denominators and expectations:
`experiments/jobs/flops_remote/README.md` (fixed before any Panel D/L run). Runs on GPU2/GPU3 of the follow-up rental.

## Panel C (open-ended continuation; original insert/move release) — plan stated 2026-09-25 06:30Z, before any panel-C run

- **Question.** In the manuscript's own setting (row 2: base Reviser vs AR, open-ended text), what does the released
  insert/move Reviser 100M cost per generated token, relative to the paper's AR 100M, under the paper's decoder?
- **Checkpoints.** `runs/july_control/ckpt/reviser_100m_wrapped.pt` (release, sha256 54dc8596…) and
  `data/checkpoints/historical/ar_100m_fineweb_step870982_model.pt` (paper AR 100M, sha256 252aff00…; identity
  supported by the task-28 evalPPL reproduction). Same architecture (d512, L24, 8 heads, d_ff 2048, action vocab 50,291).
- **Decoder.** The historical decode path of task 28, unmodified (`experiments/july_control/eval/decode_eval.py`,
  protocol `paper`: 35-token C4 prefix replayed as INSERT actions, ≤220 new actions, T 0.9, top-k 50, bf16 autocast,
  per-example seeds 20260924+i). KV-cached, one token per call, full-vocabulary head at every call.
- **Data.** First 200 prompts of the paper's C4 pool (`c4_segments_v2.json`), the first 200 of task 28's 1000.
- **Harness.** `scripts/benchmarks/measure_continuation.py` via `scripts/wsl/measured_flops_panel_c.sh`, local RTX 2070
  SUPER after panel E. Each prompt is decoded twice (counted; timed without the counter) and both must equal the stored
  task-28 row; quality (paper-definition evalPPL under GPT-2) is joined from task 28 only for identical rows.
- **Metrics.** Primary: dominant-matmul FLOPs per generated output token (ratio of means over prompts), Reviser / AR.
  Secondary: FLOPs per example; AR cost at the Reviser's output length for the same prompt (exact, from the AR's
  per-call log); latency p50/p95 per example and per output token; peak memory. Batch 1 only (the historical decoder has
  no batched path; recorded as not measured).
- **Expected (not a result).** The manuscript's proxy gives Reviser/AR = 1/(1 − p_move); task 28 measured
  p_move = 0.315 on these prompts, i.e. ≈1.46 per output token before the prompt term. The prompt replay (36 calls) is
  shared, but it is amortised over ≈67 Reviser tokens versus 220 AR tokens, so the per-token ratio should exceed 1.46
  somewhat; per example the Reviser should cost less than AR because it stops early (AR never stops within 220).
