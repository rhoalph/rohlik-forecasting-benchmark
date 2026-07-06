# Stage 5-E Pre-Submit Audit

Verdict: ready_to_submit

## Candidate and commit context

- Candidate commit hash: `5b3dc3133eb27e50d99bf94ccf5337577d396fa0`
- Audit commit hash: pending commit of this report
- Candidate file: `submissions/stage5e_stronger_features_candidate.csv`

## Candidate validation summary

- File exists and is readable by pandas.
- Row count: `47,021`
- Columns: exactly `id, sales_hat`
- Duplicate IDs: `0`
- Missing IDs: `0`
- Null predictions: `0`
- Negative predictions after clipping: `0`
- IDs match `solution.csv`: yes
- Order matches `solution.csv`: yes
- Minimum prediction: `0.0`
- Maximum prediction: `15283.618576136105`
- Mean prediction: `113.57630511838808`
- Median prediction: `42.25320084462688`

## Policy confirmation

- Stage 5-E 76-feature contract used: yes
- One global LightGBM model: yes
- Raw sales target: yes
- No horizon-specific routing: yes
- No target transform: yes
- No ensemble: yes
- No Optuna: yes
- No hyperparameter tuning: yes
- No future sales: yes
- No future `total_orders`: yes
- No future availability: yes
- No solution.csv target leakage: yes
- No weights in feature matrix: yes
- No `id` or `sales_hat` in feature matrix: yes
- Price/discount used only as Kaggle-known-future covariates: yes
- Clipping applied after prediction and documented: yes

## Comparison context

- Stage 5 S5-B official public WMAE: `21.99264`
- Stage 5 S5-B official private WMAE: `21.61114`
- Stage 5-E local mean WMAE: `19.5424424315`
- Stage 5-E local mean WAPE: `0.2027897892`
- Stage 5-E local mean bias: `-0.0472425016`
- Stage 5-E improvement vs S5-B local mean WMAE: `0.9211434498`
- S5-B local mean WMAE: `20.4635858813`

## Caveats

- This is public-solution-informed improvement work, not independent discovery.
- Price/discount covariates are benchmark-specific Kaggle-known-future inputs.
- Local improvement may not fully transfer to Kaggle hidden scoring.
- The candidate uses clipping on `75` predictions.

## Manual Kaggle submission command

```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage5e_stronger_features_candidate.csv \
  -m "stage5e candidate submission: approved stronger feature candidate"
```
