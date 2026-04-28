# Reviser 100M

## Intended Use

Research evaluation of revision-capable text generation via cursor actions.

## Training Data

The 100M model was trained on restoration-trajectory style data derived from FineWeb-style data.

## Limitations

- Sensitive to prompt formatting and action-prefix construction
- Not intended as a safety-tuned deployment model

## Released Evaluation Snapshot

- EvalPPL summary artifact: `results/evalppl/gpt2base_evalppl_dream3k_summary.json`
- Arena summaries: `results/arena/`
