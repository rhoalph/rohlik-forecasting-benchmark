# Stage 4 Kaggle Candidate Report

Date: 2026-07-05
Status: Candidate prepared; pending human review before Kaggle submission

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
| Clipping / rounding | None |
| Recursive prediction | None |
| Target transform | None |

## Final training design

- Final cutoff: 2024-06-02
- Official forecast window: 2024-06-03 through 2024-06-16
- Twelve cutoff-safe historical origins are used, shifted forward to the official cutoff.
- Training labels remain historical only and end on or before the final cutoff.

## Feature contract

- Approved features: 36
- Static metadata, known-future price/discount/calendar features, and historical-only demand summaries are retained exactly as approved.
- Future `total_orders` and future availability are excluded.

## Submission validation

- Submission path: `submissions/stage4_plain_lgbm_candidate.csv`
- Row count: 47021
- Order matches `solution.csv` and the official grid.
- No missing or duplicate IDs.
- Predictions are numeric and non-null.

## Prediction summary

| Metric | Value |
|---|---:|
| Minimum | -26.200262 |
| Maximum | 14486.535310 |
| Mean | 111.691945 |
| Median | 42.322338 |
| Negative count | 20 |
| Null count | 0 |

## Runtime and memory

| Phase | Seconds |
|---|---:|
| Raw load | 10.016 |
| Training-origin feature build | 135.742 |
| Official-grid feature build | 4.596 |
| Model fit | 43.180 |
| Prediction | 1.797 |
| Total | 196.837 |
| Peak RSS (MiB) | 1875.27 |

## Caveats

- Price/discount covariates are benchmark-specific known-future inputs.
- The local all-fold run triggered the suspicious-improvement threshold on every fold.
- F2 carried elevated negative bias in the diagnostic review.
- The Kaggle score is still unknown until a human approves submission.

## Recommendation

Ready for human review before Kaggle submission.
