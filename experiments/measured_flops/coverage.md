# Task 41 coverage: every inference-FLOP row of the original manuscript

Source: the historical manuscript `<archive>/paper/main.tex` (read-only; the archived copy is
`notes/sources/original-paper-2026-09-24/main.tex`).
- Main-text table `tab:prior_art_summary_main` is at lines 314–340.
- The appendix `app:prior_art_efficiency_flops` runs from line 1196 to 2004. It has three tables: `app:diff_family_table_100m` (1482–1522), `app:diff_family_table_big` (1743–1778) and the refinement summary `app:prior_art_table` (1963–2004).

**Manuscript conventions** (lines 1199–1285):
- Output length n = 128, d = 768, L = 12, d_ff = 4d, |V_c| = 50,000, K_max = 16.
- Only dominant matmuls are counted, with 1 MAC = 1 unit.
- AR uses causal (triangular) attention with n head applications. Every NAR row uses full attention.
- There is no prompt or prefill term for any method. The encoder is counted only for NAT and CMLM.

**Arithmetic audit (2026-09-25).** Every closed form was recomputed in Python from the stated formulas. All old values below reproduce to the printed precision. The formulas are internally consistent, so any discrepancy found by this task lies in their *assumptions*, not in the arithmetic.

## Status vocabulary

| Status | Meaning |
|---|---|
| **measured-trained** | A real trained checkpoint was measured with its supported decoder and its own AR denominator. |
| **probe** | Structural probe: an untrained reference network runs the stated schedule as a forced trace. The FLOPs are measured, but there is no quality result. |
| **blocked** | A concrete blocker is recorded, with next steps. |
| **deferred** | Feasible, but not yet run. |
| **planned** | Queued in this session. |

## Rows

The AR denominator for every 100M-tier row is the paper's AR(128) = 15.939 G MAC. The "Checkpoint / implementation" column says what exists to measure.

| # | Row (source line) | Old F (G MAC) | Old ratio | Stated assumptions | Checkpoint / implementation | Intended experiment | Status |
|---|---|---|---|---|---|---|---|
| 1 | AR baseline (1290–1310; main 322) | 15.939 | 1.00 | KV-cached, causal-triangular attention, n = 128 heads, no prompt | Trained: re100-ar-s1 / re300-ar-s1 (editing, same data as the Reviser editors); paper AR-100M (FineWeb, converted `data/checkpoints/ar_100m_actionlm.pt`); OWT `kuleshov-group/ar-noeos-owt` (12L d768, the exact reference width and depth) | Panels S, E, C, D | measured-trained (panels E, C, D, L) + probe |
| 2 | Reviser, p_move ∈ {0.20, 0.28, 0.33} (1312–1353; main 323) | 19.97–23.90 | 1.253 / 1.394 / 1.4995 | Cost = AR proxy at n_eff = n/(1−p_move); no prompt, no draft; insert/move release | Original insert/move release (`reviser_100m_actionlm.pt`, `reviser_300m_actionlm.pt`); deletion-capable editor re100/re300-d4-s1 | Panel S (forced traces at each p_move); panel C (release, real continuation traces); panel E (D4 editing) | measured-trained (panel C release: 1.308 all stages / 1.474 decode-only, length-matched; panel E D4) + probe |
| 3a | LevT, no early exit, R = 5 (1781–1849, 1982) | 154.62 | 9.7006 | Decoder-side only (no encoder or cross-attention), n0 = 0.8n, m = 0.4n placeholders, 3 trunk passes per iteration | No official checkpoint (fairseq released none); third-party MIT LevT (raymondhs/constrained-levt v1.0, WMT14 En-De, needs a fairseq fork) | Panel S (paper proxy plus a faithful encoder-decoder variant) | probe; **measured-trained** (panel T: third-party LevT, 200 newstest2014 sentences). Adaptive stop after 3.27 iterations on average (cap 5): **4.18×** a length-matched enc-dec AR (decoder-only 5.49×), CPU latency 0.41×, BLEU 21.2 (200) |
| 3b | LevT, no EE, R = 10 | 309.24 | 19.4013 | as 3a | as 3a | as 3a | probe; measured-trained (panel T): cap 10: 3.28 iterations, **4.20×** (decoder-only 5.51×), latency 0.43×, BLEU 21.3 (200) / 24.2 (full test) |
| 3c | LevT, EE f = 0.5, R = 5 | 110.17 | 6.9117 | as 3a, but the delete and placeholder predictors read the trunk at half depth (b = L/2) | as 3a | as 3a | probe only. The released checkpoint runs all 6 decoder layers for every predictor (no early-exit heads), so this half-depth variant is not measured. The measured full-depth adaptive-stop decoder (row 3a, panel T) costs 4.18× at cap 5 |
| 3d | LevT, EE f = 0.5, R = 10 | 220.33 | 13.8234 | as 3c | as 3a | as 3a | probe only (as 3c). Measured full-depth adaptive stop at cap 10 (row 3b, panel T): 4.20× |
| 4a | InsT balanced tree (1851–1873; main 326) | 32.16 | 2.0177 | log2 n + 1 = 8 full passes at n_t = 1, 2, …, 128; full vocabulary head on n_t + 1 slots; no encoder | No released checkpoint (fairseq `insertion_transformer`, no weights) | Panel S | probe (trained blocked) |
| 4b | InsT serial (1875–1884; main 327) | 1036.20 | 65.0105 | 128 full passes at n_t = 1..128 | as 4a | Panel S | probe (upper bound; no released decoder) |
| 5 | One-shot NAT (1886–1912; main 331) | 31.19 | 1.957 | Encoder + fertility head + decoder + cross-attention, n_src = n = 128, K_fert = 50 | No released checkpoint | Panel S (encoder-decoder) | probe (trained blocked) |
| 6 | CMLM / Mask-Predict, T_mp = 10 (1914–1961; main 328) | 189.05 | 11.8607 | Encoder once; 10 decoder passes plus cross-attention; head only on masked positions (Σ m_t = 700); single length candidate | Mask-Predict authors' checkpoints (6+6 layers, d512, CC-BY-NC 4.0, old fairseq fork) | Panel S; panel T | probe; **measured-trained** (panel T: authors' checkpoint, 200 newstest2014 sentences). T = 10, 1 length candidate: **7.05×** a length-matched enc-dec AR (decoder-only 9.68×), CPU latency 0.74×. 5 candidates (authors' default): **36.2×**, latency 1.98×. BLEU 23.0 (200) for both; 26.5 on the full test set (published 27.03) |
| 7a | Diffusion-LM T = 200, γ = 0 (1429–1480; main 329) | 2239.64 | 140.5131 | 200 full passes + one rounding matmul | No released checkpoint (author confirmed, issues #7/#18) | Panel S | probe (trained blocked) |
| 7b | Diffusion-LM T = 200, γ = 1 (clamping every step) | 3217.77 | 201.8799 | + rounding on every step | as 7a | Panel S | probe (trained blocked) |
| 7c | Diffusion-LM T = 2000, γ = 0 | 22352.17 | 1402.356 | 2000 full passes | as 7a (the official decoder's default is clamping on, i.e. γ = 1 at T = 2000; that row is not in the paper) | Panel S (plus the γ = 1 official default) | probe + official default (trained blocked) |
| 8a | SEDD T = 32 (1364–1390) | 514.84 | 32.3008 | T full passes with full logits | `kuleshov-group/sedd-noeos-owt` (12L d768, OWT); Lou's `louaaron/sedd-small` needs flash-attn (not on Turing) | Panel D (analytic sampler at T = 32 and at the recommended setting) | probe; measured-trained (panel D n128: 33.7× AR per try, 45.1× per accepted sample with the official entropy-rejection retries, 1.34 tries) |
| 8b | SEDD T = 2048 | 32949.92 | 2067.2497 | as 8a | as 8a | Panel D | probe; measured-trained (panel D n128: 2086× AR per try vs printed 2067×; 4432× per accepted sample with 2.13 tries) |
| 9 | MDLM T = 1000 (1392–1409) | 16088.83 | 1009.3993 | T full passes with full logits every step | `kuleshov-group/mdlm-owt-noeos` (pure-PyTorch format); `kuleshov-group/mdlm-owt` (flash-attn) | Panel D: `ddpm_cache` (the official default, which skips the network when no token changes), plus `ddpm` without cache | probe; measured-trained (n128: official default first-hitting 129.3×, ddpm_cache 121.3×, ddpm no cache 1011×; n1024: 1125× / 704× / 1098×) |
| 10 | D3PM T = 1000 (1411–1427) | 16088.83 | 1009.3993 | as MDLM | No released text checkpoint (official code is JAX, LM1B/text8 only) | Panel D proxy: absorbing-state ancestral sampling (`ddpm`, no cache) on the MDLM checkpoint, labelled as a D3PM-schedule proxy, not a D3PM model | probe. D3PM itself is **not** measured (no checkpoint). A schedule proxy on the trained MDLM checkpoint is measured: n128 1011× vs printed 1009× |
| 11a | AR Qwen2.5-32B (1620–1621, 1756) | 4099.39 | 1 | GQA + gated MLP, causal, n = 128, no prompt | `Qwen/Qwen2.5-32B` (65 GB) | Probe on the native architecture (meta device); trained run on one A100 80 GB | probe + measured-trained (panel L rows 11a/11b, Vast <id removed> A100-SXM4-80GB: the length-matched denominator of SDLM-32B, 23.9–24.3 TFLOP per GSM8K question; free-running accuracy 0.688 on 32 questions) |
| 11b | SDLM-32B-D4, k̄ = 2 (1630–1670, 1758) | 8198.94 | 2.0000 | D = 4 positions per pass, k̄ = 2 committed, KV cache, head on D positions, no verification | `OpenGVLab/SDLM-32B-D4` @865c33ee (65 GB); `SDLM-3B-D4` @925abb3f (6.8 GB) | Probe (native 32B arch, forced k̄); trained measurement at 32B (A100 80 GB) and 3B (local), each against its length-matched Qwen2.5 AR | probe; **measured-trained at 32B and 3B** (panel L rows 11a/11b, released `SDLM_generate`, 32 GSM8K questions). `prob_conf`: 32B 1.85× (τ = 0.5, k̄ = 3.48) / 2.15× (τ = 0.9, k̄ = 2.59) FLOPs at 0.30× / 0.40× A100 latency; 3B 1.92× / 2.26× at 0.32× / 0.45×. Latencies use eager attention; a 3B sdpa rerun moves them to 0.35× / 0.44× (FLOPs unchanged; 32B not re-timed). The decoder feeds k̄ + 4 positions per call, not D = 4. Verify-based `self_speculative` (released, no KV cache; first 8 questions): 3B 227×, 32B 232× FLOPs |
| 12a | AR Qwen2.5-7B (1624, 1761) | 906.65 | 1 | as 11a | `Qwen/Qwen2.5-7B` (15 GB, bf16) | Probe (native); trained needs a 24 GB GPU (existing Vast budget) | probe + measured-trained (panel L) |
| 12b | Dream-7B T = 10, linear unmasking (1672–1700, 1763) | 8768.93 | 9.6718 | T full passes; head only on masked positions ((T+1)/2 · n) | `Dream-org/Dream-v0-Base-7B` (15 GB, bf16) | Probe; trained on a rented 24 GB GPU | probe + measured-trained (panel L) |
| 12c | Dream-7B T = 20 | 17502.98 | 19.305 | as 12b | as 12b. Note: Dream's own evaluations use steps = new tokens (T = 128 at n = 128) | Probe + trained; also the recommended T = n | probe + measured-trained (panel L, incl. T = 128) |
| 13a | AR LLaMA3-8B (1627, 1768) | 962.76 | 1 | as 11a | `meta-llama/Meta-Llama-3-8B` is **gated** (the assignment forbids gated agreements) | Probe from the published config values only (no gated download) | probe (config values); trained blocked (gated) |
| 13b | LLaDA-8B T = 256 full length (1702–1735, 1770) | 238317.94 | 247.536 | T full passes; head on masked positions only | `GSAI-ML/LLaDA-8B-Base` (16 GB, MIT). The official sampler computes full logits every step and uses steps = gen_length (T = 128 at n = 128) | Probe; trained on a rented 24 GB GPU with the ungated Qwen2.5-7B as a disclosed substitute AR | probe at the printed T = 256. That setting lies outside the released sampler's documented range (steps ≤ gen_length = 128): steps 129–256 would transfer no tokens but still run full passes, which the probe executes (256.3×). Measured-trained only at the released settings, T = 64 (semi-AR) and T = 128 = gen_length (panel L, Qwen2.5-7B substitute denominator). |
| 13c | LLaDA-8B semi-AR b = 32, T_b = 16 | 36426.56 | 37.8355 | blocks of 32, trunk over min(jb, n) | as 13b | as 13b | probe + measured-trained (panel L, Qwen2.5-7B substitute denominator) |

## Panels

- **Panel S (structural probe).** Untrained reference networks at the manuscript's dimensions execute each row's stated schedule. The output is execution-based counts that audit each row's assumptions: prompt/prefill, head positions, early exits, encoder and cross-attention, and dense versus triangular attention. No quality numbers.
- **Panel E (editing, trained).** re100/re300 D4 editor versus the AR-rewrite and patch-AR baselines trained on the same data (`README.md`).
- **Panel C (open-ended continuation, trained).** The original insert/move release versus the paper's AR at 100M on the first 200 C4 prompts (task-28 protocol; all decodes token-identical to task 28). Batch 1 only: the historical decoder has no batched path, so batched continuation throughput is not measured.
- **Panel D (OWT diffusion family, trained).** AR, MDLM, SEDD and BD3-LM from one codebase, with the same backbone (12L, d768, GPT-2 tokenizer, OpenWebText). Generation length 128 (paper) and 1024 (native). Quality is judged by GPT-2 Large perplexity and unigram entropy. Status: measured at both lengths; raw traces `runs/measured_flops/panel_d/`, write-up in `notes/sources/measured-flops-results.md`.

## Blockers recorded so far

- **Translation-family trained rows: LevT and CMLM resolved 2026-09-25 (panel T); InsT and NAT still blocked.**
  - LevT (third-party MIT checkpoint) and Mask-Predict (authors' checkpoint) load and decode with their own 2019–2020 fairseq forks in an isolated CPU venv (`scripts/wsl/fairseq_env.sh`). The earlier "no Python 3.10 wheel" blocker was avoided by building the forks' extensions in place.
  - The AR denominator is an executed same-shape encoder-decoder AR, forced to each NAR output's length (FLOPs do not depend on its weights). A trained matched AR is still missing, so no AR quality is claimed.
  - InsT and one-shot NAT: no released checkpoints (fairseq released none). They stay probe-only.
  - fairseq itself is archived and ships no Python 3.10 wheel.
  - There is no *matched* released AR translation model at the same size and data (6+6 layers, d512, distilled WMT14).
  - A defensible trained comparison would need training a small AR plus CMLM pair on distilled WMT14 En-De. That is a multi-GPU-day job at the published scale and is not affordable within the remaining budget.
  - Next step if resourced: fairseq `cmlm_transformer` and `transformer` base, both trained on `distill_dataset.zip`, with identical layers and width.
- **Diffusion-LM trained rows.** No checkpoint was ever released. Reproducing E2E training takes 200k steps.
- **D3PM text.** No released checkpoint (see row 10 for the proxy).
- **LLaMA3-8B.** Gated.
- **SDLM-32B-D4 and Qwen2.5-32B (rows 11a/11b at the printed scale): resolved 2026-09-25.** This was listed as blocked on an 80 GB GPU, but that was a cost question, not an access blocker. Both repos are public and ungated, and the rental policy (≤ $2/h) allows one 80 GB GPU. They were measured on Vast <id removed> (A100-SXM4-80GB, $1.227/h; `experiments/jobs/sdlm32b/`).
