# Rohlik Forecasting Benchmark Run — Stage 0 Report

Date: 2026-07-04  
Status: Stage 0 complete; pending human review before Stage 1

## Work completed

The permanent project specification was saved as `rohlik_spec.md`. The requested repository directories were created, all six competition CSV files were inspected, and the initial data contract, risk register, and proposed backtest design were documented.

No models, features, evaluation functions, submissions, or performance claims were created.

## Files created

- `rohlik_spec.md`
- `README.md`
- `requirements.txt`
- `results.csv`
- `reports/data_contract.md`
- `reports/initial_risk_register.md`
- `reports/proposed_backtest_design.md`
- `reports/stage0_report.md`

## Data contract summary

- Training data: 4,007,419 rows from 2020-08-01 through 2024-06-02.
- Test data: 47,021 rows from 2024-06-03 through 2024-06-16.
- Forecast horizon: 14 calendar days.
- Target: `sales`.
- Forecast key: (`unique_id`, `date`).
- Test scope: 3,625 IDs across seven warehouses, using a sparse item-date grid.
- Inventory: 5,432 warehouse-specific IDs representing 2,670 products.
- All train and test IDs join to inventory.
- All train and test rows join to calendar on (`warehouse`, `date`).
- Natural keys and submission IDs contain no duplicates.
- The 52 missing `sales` values occur on exactly the same rows as 52 missing `total_orders` values. These rows are in early Munich and Frankfurt history and must not be silently interpreted as zero.

## Field-availability conclusions

- `sales` and `availability` are historical-only fields.
- Calendar fields are known future when joined by both warehouse and date.
- Inventory fields are static metadata in this competition snapshot.
- Prices and discounts are supplied for the Kaggle forecast period, but their operational availability as scheduled inputs still requires confirmation.
- `total_orders` is supplied for future test dates but is a high-risk field because it may summarize realized future-day demand.
- Test weights are evaluation metadata and should not be used as predictive inputs.

## Key leakage and integrity risks

1. Future sales entering lags, rolling windows, aggregates, encodings, or fallback statistics.
2. Train/test contamination from fitting transforms after concatenation.
3. Rolling calculations including the current row or validation labels.
4. Same-day validation `availability` being used even though it is absent from test.
5. Future `total_orders` being treated as operationally known.
6. Price and discount values being described as scheduled without evidence.
7. Calendar joins omitting warehouse and mixing national calendars.
8. Cached features being reused across incompatible forecast cutoffs.
9. Sparse item-date rows being interpreted as zero demand.
10. Evaluation weights being joined positionally or used as features.
11. Public solution logic being copied without validating its assumptions.
12. Public claims overstating autonomy, production readiness, or comparison with the original competitors.

## Proposed backtest design

Four recent, non-overlapping rolling-origin folds are proposed:

| Fold | Training cutoff | Validation window |
|---|---|---|
| F1 | 2024-05-19 | 2024-05-20 through 2024-06-02 |
| F2 | 2024-05-05 | 2024-05-06 through 2024-05-19 |
| F3 | 2024-04-21 | 2024-04-22 through 2024-05-05 |
| F4 | 2024-04-07 | 2024-04-08 through 2024-04-21 |

Each fold uses only data dated on or before its cutoff for training and target-derived computation. The following 14 days are held out for scoring.

Two validation views are proposed:

1. A Kaggle-mimicking score using the public test (`unique_id`, horizon-day) request mask shifted to each local origin.
2. A robustness score using all observed, non-null labels in each validation window.

The proposal also separates two covariate policies:

- Strict operational policy: exclude future `total_orders`; include future prices or discounts only after their forecast-origin availability is confirmed.
- Kaggle-provided policy: allow covariates supplied in the official test file, but label results as benchmark-specific.

## Open decisions

Human approval is needed on:

1. Whether strict operational validation is the primary reporting track.
2. Whether future `total_orders` should be excluded entirely.
3. Whether price and discount fields are genuinely scheduled known-future inputs.
4. Whether the public test-grid mask or all observed rows should define the primary local score.
5. Whether four recent folds are sufficient or a seasonal fold should be added.
6. The exact official WMAE formula and normalization before freezing evaluation.
7. The explicit exclusion policy for the 52 missing target rows.

## Recommended next step

Review and resolve the open decisions. After approval, proceed to Stage 1: implement and test WMAE, WAPE, bias, time-based splits, and data guards. Freeze `eval/` and `dataguard/` only after human review.
