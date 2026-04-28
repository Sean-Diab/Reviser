# Metrics

## Arena

Arena evaluation compares two candidate responses for the same prompt under randomized A/B assignment. Public summaries report:
- `n_examples`
- `n_valid`
- pairwise win counts
- win rate
- judge identifier

## evalPPL

`evalPPL` refers to perplexity-style scoring of generated continuations under a separate scorer model. Lower is better.

Released scorer variants include:
- GPT-2 base continuation-only EvalPPL
- diffusion-style proxy or upper-bound estimates where applicable

## MAUVE

The public repo includes a released BERT pseudo-loglik MAUVE artifact over the shared eligible overlap used in the later paper analysis.

## Why regular PPL is different for Reviser

For randomized restoration trajectories, “regular PPL” over the final text is not the same object as likelihood over the action trajectory. The paper therefore distinguishes:
- trajectory-level PPL over actions after the prefix
- insert-only trajectory PPL
- final-canvas token PPL
