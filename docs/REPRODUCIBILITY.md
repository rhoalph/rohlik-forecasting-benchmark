# Reproducibility

This repository does not include raw Kaggle data or credentials.

## Data download

Download the Rohlik Sales Forecasting Challenge V2 data from Kaggle and place the files in:

```text
data/raw/
```

Expected files:

- `calendar.csv`
- `inventory.csv`
- `sales_train.csv`
- `sales_test.csv`
- `test_weights.csv`
- `solution.csv`

## Expected project layout

- `eval/` — frozen scoring and backtest logic
- `dataguard/` — frozen leakage controls and cutoff helpers
- `features/` — feature-building code
- `models/` — model code
- `scripts/` — experiment and candidate-generation scripts
- `reports/` — experiment reports and reviews
- `results.csv` — experiment log
- `submissions/` — committed candidate CSVs

## Setup

1. Install dependencies.
2. Download the Kaggle data into `data/raw/`.
3. Confirm that `data/raw/` contains the expected CSV files.

## Validation commands

Run the repository checks with:

```bash
python3 -m pytest -q
python3 -m scripts.validate_stage1
```

## Candidate-generation scripts

The following scripts reproduce the released benchmark steps:

- `python3 -m scripts.run_stage5g_fixed_blend_kaggle_candidate`
  - reproduces the final Stage 5-G candidate file
- `python3 -m scripts.run_stage5e_kaggle_candidate`
  - reproduces the Stage 5-E candidate file

Diagnostic-only scripts include:

- `scripts/run_stage5f_objective_blend_experiments.py`
- `scripts/run_stage5h_supply_chain_category_pressure_experiment.py`
- earlier baseline and ablation scripts under `scripts/`

These scripts are for local validation and audit trails. They do not submit to Kaggle.

## Reproducing the final Stage 5-G candidate

With data access in place:

```bash
python3 -m scripts.run_stage5g_fixed_blend_kaggle_candidate
```

This generates:

```text
submissions/stage5g_fixed_50_50_raw_tweedie_candidate.csv
```

The candidate can then be submitted manually with the Kaggle CLI if desired.

## Limitations

- The repository does not ship raw Kaggle data.
- Benchmark-specific known-future price/discount handling should not be assumed to generalize.
- The public result is a benchmark result, not a production guarantee.

