# Reviser 300M

## Intended Use

Research evaluation of larger-scale revision-capable text generation via cursor actions.

## Training Data

The 300M model was trained on restoration trajectories derived from FineWeb-style data.

## Limitations

- Not a general-purpose aligned assistant
- Requires correct action-prefix setup for faithful reproduction

## Released Evaluation Snapshot

- Reviser 300M outperforms Reviser 100M on the released trajectory and EvalPPL artifacts
- See `results/evalppl/`, `results/arena/`, and `results/mauve/`
