# Stage 5-G Fixed Blend Kaggle Candidate Report

Date: 2026-07-06
Status: Candidate prepared; pending pre-submit audit before Kaggle submission

## Model configuration

| Component | Setting |
|---|---|
| Library | LightGBM 4.6.0 |
| Blend weights | 50% raw L1 + 50% Tweedie 1.1 |
| Model A objective | regression_l1 |
| Model B objective | tweedie |
| Tweedie variance power | 1.1 |
| Target | Raw `sales` for both models |
| Clipping / rounding | Zero-clipped after blending, no rounding |
| Recursive prediction | None |
| Target transform | None |

## Why this blend is promoted

- The Stage 5-G adversarial review approved the fixed 50/50 blend.
- Membership is globally fixed, the same weights apply to every fold, and the same rule applies to the official Kaggle test set.
- The fixed 50/50 blend improved the local mean WMAE versus both Stage 5-E raw L1 and Tweedie 1.1.

## Why the fold-specific equal blend is rejected

- The fold-specific equal-blend family changes membership by fold.
- That makes it non-generalizable to the official Kaggle test set.

## Local validation evidence

- Stage 5-E raw L1 local mean WMAE: 19.5424424315
- Tweedie 1.1 local mean WMAE: 19.1888923539
- Fixed 50/50 raw + Tweedie local mean WMAE: 19.0089170845
- Improvement vs Stage 5-E raw L1: 0.5335253470
- Improvement vs Tweedie 1.1: 0.1799752694

## Official comparison context

- Stage 5-E official public WMAE: 21.09367
- Stage 5-E official private WMAE: 20.61497
- This fixed blend is intended as a follow-on candidate from that frozen official result.

## Final training design

- Final cutoff: 2024-06-02
- Official forecast window: 2024-06-03 through 2024-06-16
- Twelve cutoff-safe historical origins are used to build the training set up to the final cutoff.
- Training labels remain historical only and end on or before the final cutoff.

## 76-feature contract

- Stage 5-E approved feature count: 76
- No additional features are introduced.
- Official test price/discount values are treated as Kaggle-known-future covariates.
- Future `total_orders`, future availability, weights, sales targets, IDs, and `sales_hat` are excluded from the model matrix.

## Candidate validation

- Submission path: `submissions/stage5g_fixed_50_50_raw_tweedie_candidate.csv`
- Candidate rows: 47021
- Row count matches `solution.csv`.
- IDs and order match `solution.csv` and the official grid.
- Raw and Tweedie prediction vectors were generated on the same official request frame and checked for identical `unique_id` / `date` order before blending.
- Predictions are numeric and non-null.
- Zero clipping was applied after blending; the number of clipped rows is recorded below.

## Prediction summaries

### Raw L1 model

| Metric | Value |
|---|---:|
| Minimum | -4.843105 |
| Maximum | 15283.618576 |
| Mean | 113.574529 |
| Median | 42.253201 |
| Negative count | 75 |
| Null count | 0 |

### Tweedie 1.1 model

| Metric | Value |
|---|---:|
| Minimum | 1.365962 |
| Maximum | 17811.580781 |
| Mean | 116.901678 |
| Median | 44.395940 |
| Negative count | 0 |
| Null count | 0 |

### Fixed blend

| Metric | Value |
|---|---:|
| Minimum | 0.000000 |
| Maximum | 16547.599679 |
| Mean | 115.238161 |
| Median | 43.321397 |
| Negative count before clipping | 7 |
| Negative count after clipping | 0 |
| Clipped rows | 7 |
| Null count | 0 |

## Runtime and memory

| Phase | Seconds |
|---|---:|
| Raw load | 7.281 |
| Training-origin feature build | 321.476 |
| Official-grid feature build | 21.153 |
| Raw model fit | 71.721 |
| Raw model prediction | 2.386 |
| Tweedie model fit | 72.559 |
| Tweedie model prediction | 2.004 |
| Blend computation | 0.013 |
| Total | 500.353 |
| Peak RSS (MiB) | 2200.28 |

## Clipping summary

- Clipping was applied after blending.
- Predictions changed by clipping: 7

## Caveats

- This is a public-solution-informed stage, not independent discovery.
- Price/discount covariates are benchmark-specific known-future inputs.
- Local improvement may not transfer exactly to Kaggle hidden scoring.
- Kaggle validates the prediction file and hidden score, not the full methodology.

## Recommendation

Ready for pre-submit audit before Kaggle submission.
