# Stage 4 Pre-Submit Audit

Date: 2026-07-05
Status: ready_to_submit

## Candidate

- Candidate file: `submissions/stage4_plain_lgbm_candidate.csv`
- Candidate commit: `5264e0708cc305b25738473025fcee2931417c05`

## Validation checklist

- [x] Repository HEAD matches the committed candidate
- [x] Git status is clean
- [x] Candidate file reads cleanly with pandas
- [x] Row count is exactly 47,021
- [x] Columns are exactly `id, sales_hat`
- [x] IDs exactly match `solution.csv`
- [x] Row order exactly matches `solution.csv`
- [x] No duplicate IDs
- [x] No missing IDs
- [x] No null `sales_hat`
- [x] `sales_hat` is numeric
- [x] Negative prediction count confirmed
- [x] Min/max/mean/median confirmed
- [x] Approved 36-feature design preserved
- [x] No `total_orders` in model features
- [x] No future availability in model features
- [x] No sales target in model features
- [x] No weight in model features
- [x] No `id` or `sales_hat` in model features
- [x] No clipping
- [x] No rounding
- [x] No target transform
- [x] No recursive prediction
- [x] No hyperparameter tuning
- [x] Price/discount used only as Kaggle-known-future covariates
- [x] `eval/` unchanged
- [x] `dataguard/` unchanged

## Prediction summary

| Metric | Value |
|---|---:|
| Rows | 47,021 |
| Negative predictions | 20 |
| Minimum | -26.2002622826 |
| Maximum | 14486.5353101106 |
| Mean | 111.6919451894 |
| Median | 42.3223380539 |
| Null predictions | 0 |

## Caveats to disclose

- Price/discount covariates are benchmark-specific Kaggle-known-future inputs.
- Every local fold triggered the suspicious-improvement threshold.
- F2 carried elevated negative bias in the diagnostic review.
- Official Kaggle score remains unknown until submission.

## Manual Kaggle submission command

```bash
kaggle competitions submit -c rohlik-sales-forecasting-challenge-v2 -f submissions/stage4_plain_lgbm_candidate.csv -m "stage4 candidate submission: approved plain lightgbm"
```

## Recommendation

The candidate is ready for a human decision on submission. No blocking issue was found in the file shape, ordering, frozen validation, or protected-directory checks.
