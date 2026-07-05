# Stage 5 Pre-Submit Audit

Verdict: ready_to_submit

## Candidate reference

- Candidate file: `submissions/stage5_s5b_price_discount_candidate.csv`
- Candidate commit: `6a9d54a`

## Validation checklist

- [x] Repository HEAD matches the committed candidate
- [x] Git status is clean
- [x] Candidate rows match the official solution template
- [x] Candidate columns are exactly `id` and `sales_hat`
- [x] Candidate order matches `solution.csv`
- [x] No duplicate IDs
- [x] No missing IDs
- [x] No null predictions
- [x] Predictions are numeric
- [x] No negative predictions remain after clipping
- [x] Approved S5-B feature policy used
- [x] Raw `sales` target used
- [x] No target transform
- [x] No horizon-specific model
- [x] No ensemble
- [x] No Optuna
- [x] No hyperparameter tuning
- [x] No future `total_orders`
- [x] No future availability leakage
- [x] No `solution.csv` target leakage
- [x] `eval/` unchanged
- [x] `dataguard/` unchanged

## Prediction summary

- Rows: 47,021
- Minimum: 0.0
- Maximum: 14531.801206601447
- Mean: 113.04552120282365
- Median: 42.62409770159663
- Negative predictions after clipping: 0

## Clipping note

- Clipping was applied.
- Affected rows: 29
- No negative predictions remain in the final candidate.
- This is consistent with the earlier S5-A diagnostic, which found clipping to be safe and only marginally beneficial.

## Comparison context

- Stage 4 official public WMAE: 22.37834
- Stage 4 official private WMAE: 21.91884
- S5-B local all-fold mean WMAE: 20.4635858813
- Stage 3 plain local all-fold mean WMAE: 20.6963289733

## Caveats

- This is a public-solution-informed stage.
- Price/discount covariates are benchmark-specific Kaggle-known-future inputs.
- Local improvement may not fully transfer to Kaggle hidden scoring.
- Official Stage 5 score is unknown until submission.

## Manual Kaggle submission command

```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage5_s5b_price_discount_candidate.csv \
  -m "stage5 candidate submission: approved s5b price discount candidate"
```

## Recommendation

Ready for human submission approval. No blocking issue was found in the file shape, ordering, clipping policy, frozen validation, or protected-directory checks.

