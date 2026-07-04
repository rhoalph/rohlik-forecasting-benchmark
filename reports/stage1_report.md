# Rohlik Forecasting Benchmark Run — Stage 1 Report

Date: 2026-07-04  
Status: Stage 1 implemented; evaluation and data-guard layers remain unfrozen pending human review

## Implementation completed

Stage 1 now includes:

- Exact Kaggle WMAE calculation.
- Business-facing WAPE and bias metrics.
- Official test-grid validation and per-`unique_id` weight alignment.
- Four rolling-origin, 14-day backtest folds.
- Central field-availability classification.
- Cutoff, validation-window, target-isolation, and key-disjointness guards.
- Unit tests and a full-data split-validation script.

No models or forecasting features were built.

## Metric definitions

Kaggle WMAE:

$$
\mathrm{WMAE} =
\frac{\sum_i w_{u_i}\lvert y_i - \hat{y}_i\rvert}
     {\sum_i w_{u_i}}.
$$

Weights are joined by `unique_id` and repeated over every requested item-date row. Predictions must match label keys exactly; missing, extra, or duplicate keys are rejected.

WAPE:

$$
\mathrm{WAPE} =
\frac{\sum_i \lvert y_i - \hat{y}_i\rvert}
     {\sum_i \lvert y_i\rvert}.
$$

Bias:

$$
\mathrm{Bias} =
\frac{\sum_i (\hat{y}_i-y_i)}
     {\sum_i y_i}.
$$

Positive bias means over-forecasting; negative bias means under-forecasting.

## Adopted covariate policy

- `sell_price_main` and all seven discount fields are treated as known future because Kaggle supplies them in the test set.
- Future `total_orders` is explicitly excluded.
- `availability` is historical only.
- `sales` is isolated as the target.
- Test weights are evaluation only and cannot enter feature frames.
- Unclassified fields fail closed until reviewed and registered.

## Official grid verification

- Official test rows: 47,021.
- Official test IDs: 3,625.
- Horizon days: 1 through 14.
- Every test ID has an official weight.
- `solution.csv` IDs and ordering exactly match the official test grid.

## Backtest results

These are split-coverage results, not model scores.

| Fold | Cutoff | Validation period | Training rows | Scored rows | Coverage |
|---|---|---|---:|---:|---:|
| F1 | 2024-05-19 | 2024-05-20 through 2024-06-02 | 3,960,048 | 44,212 | 94.03% |
| F2 | 2024-05-05 | 2024-05-06 through 2024-05-19 | 3,912,496 | 43,433 | 92.37% |
| F3 | 2024-04-21 | 2024-04-22 through 2024-05-05 | 3,864,436 | 42,794 | 91.01% |
| F4 | 2024-04-07 | 2024-04-08 through 2024-04-21 | 3,816,573 | 42,035 | 89.40% |

The 52 historical rows with missing targets are excluded from training and are never interpreted as zero.

## Verification status

```text
28 unit tests passed
Official grid validation passed
Solution ordering validation passed
Future total_orders exclusion passed
Price and discount inclusion passed
```

## Issues requiring review before freeze

1. Shifted historical grids do not contain labels for every official item/horizon combination. Local scores use the official mask and weights on available rows, but fold coverage ranges from 89.40% to 94.03% rather than 100%.
2. Excluding future `total_orders` intentionally differs from the full set of Kaggle-provided covariates.
3. Price and discounts are valid Kaggle-known-future fields, but equivalent operational availability outside the benchmark is not proven.
4. The four folds cover recent periods only; no seasonal fold is included.
5. Negative discount values remain untransformed. Kaggle describes them as no discount, but transformation belongs to a later logged feature phase.
6. The project is currently untracked inside the parent `/home/tucan` Git repository. A dedicated repository baseline is needed before freezing protected code.
7. Final empirical equivalence with Kaggle scoring requires a scored submission because hidden test labels are unavailable locally.

## Freeze recommendation

Do not freeze `eval/` or `dataguard/` until the coverage policy, metric behavior, covariate policy, fold selection, and Git baseline are explicitly approved. After approval, establish a reviewed freeze commit and add the protected-directory pre-commit guard before Stage 2.
