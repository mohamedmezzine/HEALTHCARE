# Repository Guidelines
Project: Offline RL for Clinical Decision Support using MIMIC-III Demo

Goal:
Build an offline Deep Q-Network agent that learns treatment policies from historical ICU data.

Data:
- Source: MIMIC-III Clinical Database Demo v1.4
- Format: 26 CSV files with lowercase column names
- Known quirks:
  * De-identified DOBs cause age overflow
  * Irregular timestamps
  * 
  * Multiple event streams per ICU stay

MDP Definition:
- Episode = one ICUSTAY_ID
- State = hourly aggregated vitals + labs + demographics
- Action = discretized drug/dose from INPUTEVENTS
- Reward = survival or clinical improvement proxy
- No use of future information allowed

Engineering rules:
- Windows filesystem
- pandas processing (no PostgreSQL)
- safe datetime parsing with errors="coerce"
- age computed via year/month/day method

Evaluation:
- Behavior cloning baseline
- Offline DQN with conservative penalty
- Fitted Q Evaluation on held-out patients

## Project Structure & Module Organization
- `data/`: raw MIMIC-III CSVs (read-only inputs).
- `artifacts/`: generated outputs (e.g., `artifacts/cleaned/*.parquet`).
- `src/`: core library code organized by domain:
  - `ingest/` (schema checks, CSV loading, cleaning)
  - `features/`, `actions/`, `rewards/`, `datasets/`, `models/`, `eval/`
- `tests/`: unit tests.
- `notebooks/`: exploration only; keep reusable logic in `src/`.

## Build, Test, and Development Commands
- `python -m src.ingest.prepare_data --source data --dest artifacts/cleaned --mode rl_core`
  - Cleans and exports RL-relevant tables.
- `python -m src.ingest.prepare_data --format csv`
  - Writes cleaned outputs as CSV instead of parquet.
- `pytest`
  - Runs unit tests in `tests/`.

## Coding Style & Naming Conventions
- Python 3.10+; follow `black` formatting and `ruff` linting settings in `pyproject.toml`.
- Use 4 spaces for indentation.
- Modules: lowercase with underscores (e.g., `prepare_data.py`).
- Functions: snake_case; classes: PascalCase; constants: UPPER_SNAKE_CASE.

## Testing Guidelines
- Framework: `pytest`.
- Place tests in `tests/` named `test_*.py` (e.g., `tests/test_schema.py`).
- Include tests for schema validation and data preprocessing to catch silent drift.

## Commit & Pull Request Guidelines
- Commit message conventions are not yet standardized in this repository; use clear,
  imperative summaries (e.g., "Add schema validation for MIMIC tables").
- Pull requests should include:
  - a short problem statement,
  - a summary of changes,
  - any relevant run logs (e.g., `pytest` output),
  - notes on data assumptions or preprocessing choices.

## Data Handling Notes
- Do not modify raw files in `data/`.
- All derived artifacts should be written to `artifacts/` and not committed unless requested.
