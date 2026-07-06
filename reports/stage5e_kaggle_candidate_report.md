# Stage 5-E Kaggle Candidate Report

Date: 2026-07-06
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

## Relation to Stage 5 S5-B

- Stage 5 S5-B is the current official Kaggle result.
- Stage 5-E keeps the S5-B price/discount basis and adds stronger cutoff-safe demand, group, and interaction features.
- The final candidate keeps the same official grid alignment, raw target, and frozen scoring layers.

## Local validation evidence

- Stage 5-E local mean WMAE: 19.5424424315
- Stage 5-E local mean WAPE: 0.2027897892
- Stage 5-E local mean bias: -0.0472425016
- Stage 5-E improvement vs S5-B local mean WMAE: 0.9211434498
- Stage 5-E improved every fold over S5-B.

## Final training design

- Final cutoff: 2024-06-02
- Official forecast window: 2024-06-03 through 2024-06-16
- Twelve cutoff-safe historical origins are used to build the training set up to the final cutoff.
- Training labels remain historical only and end on or before the final cutoff.

## 76-feature contract

- Stage 5-E approved feature count: 76
- Stage 5-B features are retained.
- Stage 5-E adds lag, rolling, price/discount dynamics, group-demand, and interaction features.
- Historical references are computed only from data available on or before each origin or the final cutoff.
- Official test price/discount values are treated as Kaggle-known-future covariates.
- Future `total_orders`, future availability, weights, sales targets, IDs, and `sales_hat` are excluded from the model matrix.

## Candidate validation

- Submission path: `submissions/stage5e_stronger_features_candidate.csv`
- Candidate rows: 47021
- Row count matches `solution.csv`.
- IDs and order match `solution.csv` and the official grid.
- Predictions are numeric and non-null.
- Zero clipping was applied after model prediction; the number of clipped rows is recorded below.

## Prediction summary

| Metric | Value |
|---|---:|
| Minimum | 0.000000 |
| Maximum | 15283.618576 |
| Mean | 113.576305 |
| Median | 42.253201 |
| Negative count before clipping | 75 |
| Negative count after clipping | 0 |
| Clipped rows | 75 |
| Null count | 0 |

## Runtime and memory

| Phase | Seconds |
|---|---:|
| Raw load | 7.342 |
| Training-origin feature build | 296.123 |
| Official-grid feature build | 19.323 |
| Model fit | 65.721 |
| Prediction + clipping | 1.742 |
| Total | 391.711 |
| Peak RSS (MiB) | 2197.20 |

## Clipping summary

- Clipping was applied because it is a documented, business-valid post-processing choice.
- Predictions changed by clipping: 75
- The separate S5-A diagnostic showed clipping to be safe and only marginally beneficial.

## Caveats

- This is a public-solution-informed stage, not independent discovery.
- Price/discount covariates are benchmark-specific known-future inputs.
- Local improvement may not transfer exactly to Kaggle hidden scoring.
- Official Kaggle score remains unknown until submission.

## Recommendation

Ready for pre-submit audit before Kaggle submission.
