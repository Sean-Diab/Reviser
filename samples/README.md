# Model outputs

Raw outputs of the models in the paper, so that anyone can read what the models actually wrote.
All files are UTF-8 JSON Lines (one JSON object per line) unless noted.
Texts are decoded from GPT-2 token ids.

The texts are unfiltered web prompts (C4) and model outputs. They may contain errors, made-up facts,
offensive content, or contact details copied from or modelled on web pages.

## 1. Continuation (Table 1)

Folder `continuation/`.

Setup: 3,000 C4 validation prompts in three groups of 1,000. Each group is sampled with its own seed
(123, 124, 125), and each prompt is decoded once per model. The prompt is the first 35 GPT-2 tokens of the
document. Sampling: temperature 0.9, top-k 50, at most 220 new actions. Every Reviser model is decoded
with the prompt lock: the cursor starts at the end of the prompt and cannot move into it, so the prompt
never changes. The AR models only append tokens.

| file prefix | Table 1 row |
|---|---|
| `ar_100m` | AR 100M |
| `reviser_100m_released` | 100M: Reviser, released |
| `reviser_100m_runjump` | 100M: + run-and-jump |
| `reviser_100m_runjump_delete` | 100M: + run-and-jump + Delete |
| `reviser_100m_runjump_delete_rl_s1`, `_s2`, `_s3` | 100M: + RL (seeds 1, 2, 3) |
| `ar_300m` | AR 300M |
| `reviser_300m_released` | 300M: Reviser, released |
| `reviser_300m_runjump_delete` | 300M: + run-and-jump + Delete |
| `reviser_300m_runjump_delete_rl_s3` | 300M: + RL (seed 3) |

Files:

- `<model>.texts.jsonl`: all 3,000 prompts. Fields: `id`, `seed`, `prompt`, `continuation` (the new text
  after the prompt), `stopped` (the model chose STOP; otherwise it hit the 220-action budget),
  `n_actions` (new actions, including STOP), `n_new_tokens`, `n_insert`, `n_move`, `n_delete`, and
  `n_noop` when it is not 0 (see the notation below). The prompt was intact in every output, so no
  `final_text` field was needed.
- `<model>.trajectories.jsonl`: the first 100 prompts (all from the seed-123 group). Fields: `id`, `seed`,
  `prompt`, `actions` (the action sequence after the prompt, in the notation below) and `final_text`
  (prompt plus continuation).
- `c4_reference.texts.jsonl`: the human continuation of each prompt from C4. Fields: `id`, `seed`,
  `prompt`, `reference`, `n_reference_tokens`.
- `models.json`: the decoding settings of each file and the SHA-256 of each checkpoint.

### Action notation (trajectories)

The canvas is a list of GPT-2 tokens with a cursor between two tokens.

- `INS ' the'`: insert the token at the cursor; the cursor moves to just after it. The token text is shown
  as a Python string literal. A token that holds only part of a multi-byte character is shown with
  `\xNN` byte escapes.
- `DEL`: delete the token to the left of the cursor (backspace).
- `MOVE +k` / `MOVE -k`: move the cursor k tokens right or left, k in {1, 2, 4, ..., 512}.
- `STOP`: end the output.
- `NOOP(EOI)`: the model emitted its end-of-input marker. The executor ignores it, so the text does not
  change. This is rare: at most 316 times in all 3,000 outputs of one model. `NOOP(<id>)` is another
  ignored special token (once, in AR 300M).

### Check against Table 1

Measured on these files. Len: mean new tokens. Stop: share of outputs that end with STOP. Mv/tok and
Del/tok: all moves (deletes) divided by all new tokens.

| model | Len | Stop % | Mv/tok | Del/tok | Table 1 (Len / Stop / Mv / Del) |
|---|---|---|---|---|---|
| AR 100M | 220.0 | 0.0 | 0 | 0 | 220.0 / 0.0 / 0 / 0 |
| Reviser 100M, released | 67.8 | 99.5 | 0.465 | 0 | 67.8 / 99.5 / 0.465 / 0 |
| + run-and-jump | 101.1 | 94.3 | 0.188 | 0 | 101.1 / 94.3 / 0.188 / 0 |
| + run-and-jump + Delete | 113.6 | 87.9 | 0.152 | 0.033 | 113.6 / 87.9 / 0.152 / 0.033 |
| + RL, mean of 3 seeds | 142.1 | 74.9 | 0.140 | 0.029 | 142.1 / 74.9 / 0.140 / 0.029 |
| (seeds 1 / 2 / 3) | 144.8 / 142.0 / 139.6 | 71.2 / 76.0 / 77.6 | 0.143 / 0.143 / 0.135 | 0.028 / 0.028 / 0.030 | |
| AR 300M | 220.0 | 0.0 | 0 | 0 | 220.0 / 0.0 / 0 / 0 |
| Reviser 300M, released | 94.7 | 98.3 | 0.381 | 0 | 94.7 / 98.3 / 0.381 / 0 |
| + run-and-jump + Delete | 110.4 | 90.6 | 0.147 | 0.029 | 110.4 / 90.6 / 0.147 / 0.029 |
| + RL, seed 3 | 141.2 | 79.1 | 0.146 | 0.026 | 141.2 / 79.1 / 0.146 / 0.026 |

## 2. Editing (Table 5)

Folder `editing/`. Training seed 1 of each model.

Decoding is greedy over action types: take the action type (insert, move, delete or stop) with the
largest total probability, then its most likely action. Reviser-Edit starts with the input on the
canvas and the cursor at its end, and edits it in place. The AR rewriter writes the output from scratch,
left to right, with inserts only.

- `coedit_validation.jsonl` (1,712 items) and `iterater_human_sent_test.jsonl` (364 items). Fields: `uid`,
  `dataset`, `intent`, `instruction`, `input`, `reference`, the four outputs `reviser_edit_100m`,
  `ar_rewriter_100m`, `reviser_edit_300m`, `ar_rewriter_300m`, and per model: `sari` (sentence SARI with
  the one reference), `exact_match`, `stopped`, and `actions` (counts of actions, inserts, deletes and
  moves, including STOP in `n_actions`).
- `jfleg_test.jsonl` (747 items). Fields: `uid`, `dataset`, `instruction` (the same for all items),
  `input` (the official tokenized source), the four outputs and their action counts. The four reference
  corrections are in the official JFLEG release. The models usually return untokenized text
  ("society." for "society ."); for GLEU the outputs are PTB-tokenized to match the references.

The action sequences of the edit runs were not stored, so only counts are given. The example viewer
(below) replays 40 edit trajectories.

### Check against Table 5

| model | CoEdIT SARI | CoEdIT EM | IteraTeR SARI |
|---|---|---|---|
| Reviser-Edit 100M | 57.6 | 0.125 | 46.1 |
| AR rewriter 100M | 50.4 | 0.055 | 47.1 |
| Reviser-Edit 300M | 59.3 | 0.141 | 46.2 |
| AR rewriter 300M | 54.8 | 0.075 | 49.7 |

These are the means of the per-item scores in the files and equal the Table 5 values.

## 3. Example viewer

`example_viewer.html`: open in any browser. It replays 70 example trajectories one action at a time:
10 each for Reviser 100M + RL (next to AR 100M), Reviser 300M + RL and the released Reviser 100M, and
20 each for Reviser-Edit 100M and Reviser-Edit 300M (next to the AR rewriter). The examples were picked by
hand for readable text and clear cursor use from pools of filtered outputs (Appendix J describes the pools),
so they are not a random sample; the files above contain every output. The examples printed in Appendix J
are marked "Shown in Appendix J". All data and code are inside the page; it makes no network
requests.

## Data

Prompts, inputs and references come from C4 (validation split), CoEdIT (validation), IteraTeR
(human-annotated sentence-level test set) and JFLEG (test), under their licenses.
