# Stage 2 F1 Naive Baseline Results

Date: 2026-07-04  
Status: F1 complete; F2–F4 not run  
Scope: simple non-model baselines only

## Evaluation contract

| Field | Value |
|---|---|
| Fold | F1 |
| Forecast cutoff | 2024-05-19 |
| Validation window | 2024-05-20 through 2024-06-02 |
| Training rows | 3,960,048 |
| Training IDs | 5,390 |
| Official shifted-grid rows requested | 47,021 |
| Rows with available labels and scored | 44,212 |
| Missing-label rows excluded | 2,809 |
| Scored-row coverage | 94.03% |
| Missing-target training rows excluded | 52 |
| Official weights | Joined by `unique_id` through frozen `eval/` |
| Global training median | 39.48 |

Only `unique_id`, `date`, and float64 `sales` were loaded from training data. The runner did not load `total_orders`, price, discount, availability, inventory, or calendar fields. No future covariates were required by these baselines.

Prediction keys were validated by `eval.score_kaggle_aligned`, which enforces an exact one-to-one match with the available F1 labels and applies official weights by `unique_id`.

## Results

WAPE and bias are shown as ratios. Multiply by 100 for percentage presentation. Runtime is prediction generation plus frozen metric alignment/scoring for that baseline; it excludes shared file loading and split preparation.

| Baseline | WMAE | WAPE | Bias | Runtime (s) | Coverage | Primary fallback rows | Global fallback rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero forecast | 77.27816 | 1.00000 | -1.00000 | 0.182 | 94.03% | 0 | 0 |
| Global median | 58.46590 | 0.81943 | -0.65855 | 0.185 | 94.03% | 0 | 0 |
| Last observed by ID | 41.05448 | 0.40151 | -0.13466 | 0.202 | 94.03% | 0 | 0 |
| Same weekday last week | 37.75263 | 0.36385 | -0.05253 | 1.615 | 94.03% | 23,102 | 0 |
| Trailing 7-day mean by ID | 32.14859 | 0.31139 | 0.02649 | 0.239 | 94.03% | 594 | 0 |
| Trailing 14-day mean by ID | 31.19858 | 0.30211 | 0.03040 | 0.259 | 94.03% | 537 | 0 |
| Median by ID and day of week | 30.71502 | 0.34111 | -0.19291 | 1.262 | 94.03% | 36 | 0 |

These are local F1 diagnostics. They are not Kaggle submission scores and are not performance claims.

## Baseline definitions

### Zero forecast

Predict zero for every scored validation row.

### Global median

Predict the median of every non-null training `sales` value dated on or before the cutoff.

### Last observed by ID

Predict each ID's latest observed `sales` value on or before the cutoff. Use the global median only when the ID has no safe history.

### Same weekday last week

For a validation date, use the same ID's value at `date - 7 days` only when that source date is on or before the forecast cutoff. If the source is absent or crosses the cutoff, use last observed sales and then the global median.

This cutoff rule is material: horizons 8–14 cannot use literal `t-7` actuals because those source dates are inside the validation period. Consequently, 23,102 rows use the fallback. No recursive validation predictions are used.

### Trailing 7-day and 14-day means

For each ID, average observed training rows within the final 7 or 14 calendar days ending at the cutoff. Missing item-date rows are not converted to zero. If an ID has no observation in the window, use last observed sales and then the global median.

### Median by ID and day of week

Compute the median from all non-null pre-cutoff training history grouped by ID and day of week. Use last observed sales and then global median when the group is absent.

## Runtime and memory

| Measurement | Result |
|---|---:|
| Shared data loading, frozen split materialization, and fallback preparation | 15.206 seconds |
| Total wall time for preparation and all seven baselines | 19.533 seconds |
| Peak process RSS | 882.41 MiB |
| Processed data written | No |
| Kaggle submission made | No |

The full-data F1 run fits comfortably in available RAM. CPU and repeated full-history aggregation remain the relevant constraints, but the measured runtime is acceptable for running later folds sequentially.

## Correctness and leakage review

Checks passed:

- Frozen F1 cutoff and validation dates were used.
- Training contains no row after the cutoff.
- Validation labels are physically separate from baseline inputs.
- Every prediction frame exactly matches all 44,212 scored keys.
- Official weights are joined by ID.
- Missing targets are excluded rather than treated as zero.
- `total_orders` is not loaded or used.
- No future sales are used.
- Same-weekday source dates are asserted at or before cutoff.
- All predictions are finite.
- 31 unit tests pass, including new baseline cutoff tests.

Known interpretation concerns:

1. Coverage is 94.03%, not the full Kaggle test grid, because 2,809 shifted historical keys lack labels.
2. The same-weekday baseline becomes a last-observed fallback for most horizon 8–14 rows by design; using actual `t-7` values there would leak validation targets.
3. Rolling means use observed rows, not a dense calendar with absent days set to zero.
4. The ID/day-of-week median uses long history and may represent stale regimes, though it is cutoff-safe.
5. Local F1 scores must not be interpreted using final/private Kaggle score bands.

## F1 gate recommendation

F1 is clean enough to run F2–F4 sequentially on full data:

- All baseline grids and metric calls completed.
- Runtime and memory are within hack-box limits.
- No global fallback was needed for scored rows.
- Leakage-specific tests passed.
- No protected-layer changes were made.

This is a recommendation only. The run stops after F1 as instructed; F2–F4 require the next human decision.
