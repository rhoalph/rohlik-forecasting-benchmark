# Proposed Backtest Design

Status: proposal for human review. No split code or evaluation implementation exists yet.

## Test-period anatomy to reproduce

The final observed training date is Sunday 2024-06-02. Kaggle requests forecasts for Monday 2024-06-03 through Sunday 2024-06-16: a contiguous 14-day horizon.

The test contains 47,021 requested rows for 3,625 IDs. The grid is not a full 3,625 × 14 rectangle: 2,688 IDs have all 14 horizon days and 937 have fewer. A realistic local evaluation should reproduce both the 14-day temporal horizon and, for its primary score, the requested item/horizon-day mask.

## Candidate forecast origins

Use four non-overlapping rolling-origin windows near the end of training:

| Fold | Forecast origin / training cutoff | Validation window | Horizon |
|---|---|---|---:|
| F1 (latest) | 2024-05-19 | 2024-05-20 to 2024-06-02 | 14 days |
| F2 | 2024-05-05 | 2024-05-06 to 2024-05-19 | 14 days |
| F3 | 2024-04-21 | 2024-04-22 to 2024-05-05 | 14 days |
| F4 | 2024-04-07 | 2024-04-08 to 2024-04-21 | 14 days |

For each fold, training may use only records with `date <= cutoff`. The following 14 dates supply labels for evaluation only. The final Kaggle origin (2024-06-02) is not a local validation fold because its labels are hidden.

Four folds balance recency with stability and avoid overlapping validation periods. Stage 1 should report each fold separately plus an aggregate; poor performance on one fold must not be hidden by averaging.

## Validation row selection

Primary benchmark-mimicking score:

1. Convert each test date to `horizon_day` 1–14 relative to 2024-06-02.
2. Retain the resulting (`unique_id`, `horizon_day`) request mask; this uses only the public test grid, never hidden targets.
3. For each local origin, shift the mask to `origin + horizon_day` and score matching rows that have non-null historical `sales` labels.
4. Record coverage and missing predictions explicitly. Do not convert absent rows to zero labels.

Secondary robustness score:

- Score all observed, non-null item-date labels in each 14-day window for IDs with history at the cutoff.
- Report this separately so reliance on the public test-grid selection is visible.

The primary score best matches Kaggle's requested combinations. The secondary score tests whether conclusions generalize beyond that mask.

## Allowed data before each cutoff

- `sales_train.csv` rows dated on or before the fold cutoff.
- Historical targets and covariates from those rows, including only cutoff-safe lags or aggregates in later stages.
- Static `inventory.csv` metadata joined by `unique_id`, with warehouse consistency asserted.
- `test_weights.csv` joined by `unique_id` inside scoring only.
- Configuration and transforms fitted only on the fold's allowed training rows.

Rows within the validation window may supply `sales` only to the scoring function after predictions have been fixed. They may not affect preprocessing, features, fallbacks, model fitting, or calibration.

## Allowed known-future data

Unconditionally allowed:

- Prediction dates and horizon-day position.
- Requested item IDs and warehouse assignment.
- `calendar.csv` fields for validation dates, joined on (`warehouse`, `date`).
- Static inventory metadata.

Two covariate policies should be kept distinct:

1. **Strict operational policy (recommended primary claim):** exclude future `total_orders`; use future price/discount fields only if human review confirms they represent schedules known at the origin. Otherwise exclude them too.
2. **Kaggle-provided policy:** permit `total_orders`, price, and discount values for the forecast window because the official test file supplies them. Results must be labeled benchmark-specific and cannot establish real operational availability.

For local folds, future covariates must be taken from the corresponding historical validation rows only if the selected policy declares that column known future. Their `sales` and `availability` values remain hidden from the prediction pipeline.

## Forbidden data

- Any `sales` observation dated after the fold cutoff during training, feature generation, target encoding, calibration, or fallback construction.
- Same-day or future-window `availability`; it is not supplied in test.
- Future `total_orders` under the strict operational policy.
- Price or discounts under a policy that has not declared them known at forecast origin.
- Aggregates, scalers, encoders, category statistics, or caches fitted using later folds, the full training file, or hidden validation labels.
- Test `sales` (not provided), solution-template zeros, public-solution predictions, or leaderboard feedback used as local labels.
- Weight values as predictors; they are evaluation metadata.

## Required Stage 1 assertions

- Maximum source date for every target-derived input is at or before the fold cutoff.
- Training and validation keys are disjoint and unique.
- Validation dates equal `cutoff + 1` through `cutoff + 14`.
- Calendar and inventory joins do not change row counts.
- No null target is scored.
- Every scored ID has exactly one joined weight.
- Prediction code cannot access the validation target column.
- Cache metadata matches fold cutoff and covariate policy.
- Per-fold row counts, ID counts, date range, and warehouse coverage are logged.

## Reporting

For every baseline/model in later stages, report WMAE, WAPE, bias, runtime, and coverage by fold. Also retain warehouse-level diagnostics. Keep the official WMAE and business-facing WAPE conceptually and numerically separate.

## Open decisions for human approval

1. Is the strict operational policy the primary backtest, with Kaggle-provided covariates as a secondary benchmark track?
2. Can price and discounts be verified as schedules known before each 14-day horizon?
3. Should future `total_orders` be excluded entirely from publishable results because of its likely realized-demand content?
4. Should the primary score use the exact public test request mask, or should all-observed rows be primary and the mask be secondary?
5. Is four recent, non-overlapping folds sufficient, or is a seasonal fold required before freezing evaluation?
