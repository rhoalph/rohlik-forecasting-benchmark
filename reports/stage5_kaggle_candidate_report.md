# Stage 5 Kaggle Candidate Report

Date: 2026-07-05
Status: Candidate prepared; pending pre-submit audit before Kaggle submission

## Model configuration

| Setting | Value |
|---|---|
| Library | LightGBM 4.6.0 |
| Objective | regression_l1 |
| Target | Raw `sales` |
| Boosting rounds | 300 |
| Learning rate | 0.05 |
| Leaves | 31 |
| Minimum rows per leaf | 100 |
| Clipping / rounding | Zero-clipped, no rounding |
| Recursive prediction | None |
| Target transform | None |

## Relation to Stage 4

- Stage 4 is the frozen audited first submission.
- Stage 5 adds public-solution-informed relative price and discount features on top of the same plain LightGBM structure.
- The final candidate keeps the same official grid alignment, same raw target, and the same frozen evaluation layers.

## Final training design

- Final cutoff: 2024-06-02
- Official forecast window: 2024-06-03 through 2024-06-16
- Twelve cutoff-safe historical origins are used, shifted forward to the official cutoff.
- Training labels remain historical only and end on or before the final cutoff.

## Feature contract

- Stage 4 features retained: 36
- Added relative price/discount features: 10
- Historical price references are computed only from rows available on or before each origin or the final cutoff.
- Official test price/discount values are treated as Kaggle-known-future covariates.
- Future `total_orders`, future availability, weights, sales targets, IDs, and `sales_hat` are excluded from the model matrix.

## Candidate validation

- Submission path: `submissions/stage5_s5b_price_discount_candidate.csv`
- Candidate rows: 47021
- Row count matches `solution.csv`.
- IDs and order match `solution.csv` and the official grid.
- Predictions are numeric and non-null.
- Zero clipping was applied after model prediction; the number of clipped rows is recorded below.

## Prediction summary

| Metric | Value |
|---|---:|
| Minimum | 0.000000 |
| Maximum | 14531.801207 |
| Mean | 113.045521 |
| Median | 42.624098 |
| Negative count before clipping | 29 |
| Negative count after clipping | 0 |
| Clipped rows | 29 |
| Null count | 0 |

## Local validation evidence

- Stage 3 plain all-fold mean WMAE: 20.6963289733
- Stage 5 S5-B local mean WMAE: 20.4635858813
- Stage 5 S5-B local mean bias: -0.0471988703
- S5-B improved every fold over the Stage 3 plain model and is the basis for this candidate.

## Runtime and memory

| Phase | Seconds |
|---|---:|
| Raw load | 9.019 |
| Training-origin feature build | 185.247 |
| Official-grid feature build | 9.527 |
| Model fit | 47.200 |
| Prediction + clipping | 1.564 |
| Total | 254.124 |
| Peak RSS (MiB) | 1877.42 |

## Clipping summary

- Clipping was applied because it is a documented, business-valid post-processing choice.
- Predictions changed by clipping: 29
- The separate S5-A diagnostic showed clipping to be safe and only marginally beneficial.

## Caveats

- This is a public-solution-informed stage, not independent discovery.
- Price/discount covariates are benchmark-specific known-future inputs.
- Local improvement may not transfer exactly to Kaggle hidden scoring.
- Official Kaggle score remains unknown until submission.

## Recommendation

Ready for pre-submit audit before Kaggle submission.
