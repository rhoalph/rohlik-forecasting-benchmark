# Stage 5-G Fixed 50/50 Raw + Tweedie Pre-Submit Audit

## Verdict

ready_to_submit

## Candidate metadata

- Candidate commit hash: `e5bcee87a0ee9f34c1bdf74f4abe45a8cadaa258`
- Audit commit hash: not committed yet
- Candidate file: `submissions/stage5g_fixed_50_50_raw_tweedie_candidate.csv`

## Candidate validation summary

- Rows: 47,021
- Columns: exactly `id, sales_hat`
- IDs match `data/raw/solution.csv`
- Order matches `data/raw/solution.csv`
- Duplicate IDs: 0
- Missing IDs: 0
- Null predictions: 0
- Numeric predictions: yes
- Negative predictions after clipping: 0
- Clipped rows: 7

Prediction summary:

- Min: `0.0`
- Max: `16547.599678599647`
- Mean: `115.23816080445405`
- Median: `43.32139715386394`

## Raw model prediction summary

- Min: `-4.843105337034841`
- Max: `15283.618576136105`
- Mean: `113.57452859542913`
- Median: `42.25320084462688`
- Negative count: `75`
- Null count: `0`

## Tweedie model prediction summary

- Min: `1.36596206355316`
- Max: `17811.580781063185`
- Mean: `116.90167807309058`
- Median: `44.395939818721736`
- Negative count: `0`
- Null count: `0`

## Fixed blend policy confirmation

- Blend weights are globally fixed at 50% raw L1 and 50% Tweedie 1.1.
- Weights sum to 1.
- Raw and Tweedie predictions are generated on the same official request frame.
- Prediction keys are checked for exact `unique_id` / `date` order before blending.
- No fold-specific membership is used.
- No fold-specific weights are used.
- No validation labels are used to choose blend members in candidate generation.
- No additional model components are used.

## Model and feature policy confirmation

- Stage 5-E 76-feature contract is used.
- One global raw L1 LightGBM model is used.
- One global Tweedie 1.1 LightGBM model is used.
- Both models use raw sales targets.
- No horizon-specific routing.
- No target transform.
- No Optuna.
- No hyperparameter tuning.
- No future sales.
- No future `total_orders`.
- No future availability.
- No solution leakage from `solution.csv`.
- No weights, `id`, or `sales_hat` are used as model features.
- Price / discount fields are used only as Kaggle-known-future covariates.
- Clipping is applied after blending and documented.

## Comparison context

- Stage 5-E official public WMAE: 21.09367
- Stage 5-E official private WMAE: 20.61497
- Stage 5-G fixed 50/50 local mean WMAE: 19.0089170845
- Stage 5-E raw L1 local mean WMAE: 19.5424424315
- Tweedie 1.1 local mean WMAE: 19.1888923539
- Local improvement vs Stage 5-E raw L1: 0.5335253470
- Local improvement vs Tweedie 1.1: 0.1799752694

## Caveats

- Public-solution-informed stage.
- Price / discount benchmark dependency remains.
- Local improvement may not fully transfer to Kaggle.
- Kaggle validates the prediction file and hidden score, not the full methodology.
- This is the final planned scoring candidate before moving to executive insights.

## Exact Kaggle submission command

```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage5g_fixed_50_50_raw_tweedie_candidate.csv \
  -m "stage5g candidate submission: fixed 50/50 raw + tweedie"
```

## Recommendation

Proceed to Kaggle submission only if the human reviewer accepts the fixed blend and its caveats.
