# Experiment Protocol

## Continuation Setup

- Prompts are derived from 35-token GPT-2-tokenized prefixes
- Reviser generates a restoration trajectory after the prompt prefix
- Final text is the decoded canvas after the generated action sequence halts

## Arena Evaluation

- Pairwise comparisons use prompt-conditioned candidate responses
- A/B side assignment is randomized per example
- Released summaries report `n_examples`, `n_valid`, win rates, and judge metadata

## Baseline Families

- Reviser
- Autoregressive baselines
- SEDD
- MDLM

## Public Result Policy

- The repo releases compact summaries and representative visualizations
- Very large raw rollout stores and internal debugging logs are excluded
- Paper-facing public summaries may omit internal comparisons or fields that were not retained in the manuscript-facing reporting
