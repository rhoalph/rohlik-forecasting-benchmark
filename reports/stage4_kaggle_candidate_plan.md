# Stage 4 Kaggle Candidate Plan

Date: 2026-07-05
Status: Candidate preparation only
Scope: final official submission path, not tuning

## Purpose

Prepare one final Kaggle submission candidate using the approved Stage 3 plain LightGBM design without changing the model contract, feature contract, or frozen evaluation layers.

This is candidate preparation, not leaderboard tuning:

- no hyperparameter search
- no feature search
- no ablations
- no ensembles
- no recursive forecasting
- no clipping or rounding
- no changes to `eval/` or `dataguard/`

## Final training and forecast horizon

| Item | Value |
|---|---|
| Final training cutoff | 2024-06-02 |
| Official forecast start | 2024-06-03 |
| Official forecast end | 2024-06-16 |
| Horizon length | 14 days |

The official Kaggle test grid is the shifted request mask from `sales_test.csv` plus `test_weights.csv`, validated against `solution.csv`.

## Training-origin design

The final candidate uses the same twelve-origin style as Stage 3, shifted forward to the official cutoff.

| Origin index | Origin date | Label window |
|---|---|---|
| 1 | 2024-05-19 | 2024-05-20 through 2024-06-02 |
| 2 | 2024-05-05 | 2024-05-06 through 2024-05-19 |
| 3 | 2024-04-21 | 2024-04-22 through 2024-05-05 |
| 4 | 2024-04-07 | 2024-04-08 through 2024-04-21 |
| 5 | 2024-03-24 | 2024-03-25 through 2024-04-07 |
| 6 | 2024-03-10 | 2024-03-11 through 2024-03-24 |
| 7 | 2024-02-25 | 2024-02-26 through 2024-03-10 |
| 8 | 2024-02-11 | 2024-02-12 through 2024-02-25 |
| 9 | 2024-01-28 | 2024-01-29 through 2024-02-11 |
| 10 | 2024-01-14 | 2024-01-15 through 2024-01-28 |
| 11 | 2023-12-31 | 2024-01-01 through 2024-01-14 |
| 12 | 2023-12-17 | 2023-12-18 through 2023-12-31 |

This design keeps every training label historical relative to the official test horizon. No label from 2024-06-03 through 2024-06-16 is used for fitting.

## Feature and data contract

The candidate keeps the exact approved 36-feature Stage 3 matrix.

| Class | Examples | Policy |
|---|---|---|
| Static metadata | `unique_id`, `product_unique_id`, warehouse and category names | Allowed |
| Known future | date features, holiday flags, `sell_price_main`, discount fields | Allowed because they are present in `sales_test.csv` |
| Historical only | lag, rolling, same-weekday, and historical summary features | Allowed only from data on or before the final cutoff |
| Forbidden / excluded | future `sales`, `total_orders`, future `availability`, `weight`, `id`, `sales_hat`, recursive targets | Excluded |

Price and discount fields are treated as Kaggle-known-future benchmark covariates only. That is benchmark-specific and does not imply operational availability outside this competition.

Future `total_orders` and future availability are not used.

## Official grid alignment

The candidate submission is aligned to the official grid as follows:

1. Load `sales_test.csv` and `test_weights.csv`.
2. Build the official grid with `eval.grid.build_official_test_grid`.
3. Validate the grid against `solution.csv` with `eval.grid.validate_solution_template`.
4. Build the final feature batch using the approved 36-feature contract and the final cutoff.
5. Predict on the official grid in the same order as the validated solution template.
6. Emit a submission file with exactly two columns: `id` and `sales_hat`.

The output must have:

- exact row count match with `solution.csv`
- no missing IDs
- no duplicate IDs
- exact order match with `solution.csv`
- numeric, non-null `sales_hat`

## Validation plan

Before any submission, the candidate will be checked for:

- exact row count match with `solution.csv`
- exact ID order match with `solution.csv`
- no null predictions
- no duplicate prediction keys
- no forbidden columns in the feature matrix
- no future `total_orders`
- no future availability
- no target leakage into features
- frozen `eval/` and `dataguard/` unchanged

## Files to be created

- `scripts/run_stage4_kaggle_candidate.py`
- `tests/test_stage4_kaggle_candidate.py`
- `reports/stage4_kaggle_candidate_report.md`
- `submissions/stage4_plain_lgbm_candidate.csv`

## Files that will not be changed

- `eval/`
- `dataguard/`
- Stage 3 model design
- Stage 2 baselines
- Kaggle submission state

## Why this is candidate preparation, not tuning

The candidate uses the already approved model structure, feature contract, and training schedule. It does not search for a better score, alter hyperparameters, or test alternate feature sets. The purpose is to create one submission-ready artifact that is consistent with the frozen Stage 3 result and the official Kaggle request grid.
