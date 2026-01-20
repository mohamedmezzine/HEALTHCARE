# Healthcare RL Project

This repository hosts the data pipeline and offline RL modeling for MIMIC-III.

## Layout
- `data/`: raw CSVs (read-only)
- `artifacts/`: generated outputs (parquet, npz)
- `src/`: core library code
  - `ingest/`: loading + schema checks
  - `features/`: state construction
  - `actions/`: action mapping + discretization
  - `rewards/`: reward functions
  - `datasets/`: replay buffer builders
  - `models/`: DQN / CQL etc.
  - `eval/`: FQE, BC baseline, sanity checks
- `notebooks/`: exploration only
- `tests/`: unit tests that catch silent bugs

## Conventions
- Keep `data/` immutable; all derived outputs go to `artifacts/`.
- Keep notebooks disposable; core logic lives in `src/`.

