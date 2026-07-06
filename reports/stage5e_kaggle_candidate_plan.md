# Stage 5-E Kaggle Candidate Plan

## Purpose

Prepare the first Kaggle submission candidate from the stronger Stage 5-E feature set. This is an improvement step on top of the frozen Stage 5 S5-B official result, not a replay of the plain model stage.

## Relationship to S5-B

- Stage 5 S5-B is the current official Kaggle result.
- Stage 5-E adds stronger cutoff-safe demand, price, discount, group, and interaction features on the same global LightGBM backbone.
- The local validation evidence for Stage 5-E is materially better than S5-B, so it is the next candidate path.

## Local validation evidence

- Stage 5-E F1 WMAE: 19.6069995273
- Stage 5-E F2 WMAE: 21.3218385474
- Stage 5-E F3 WMAE: 18.7434564665
- Stage 5-E F4 WMAE: 18.4974751850
- Stage 5-E mean WMAE: 19.5424424315
- Stage 5-E mean WAPE: 0.2027897892
- Stage 5-E mean bias: -0.0472425016
- Improvement vs S5-B local mean WMAE: 0.9211434498

## Final training design

- One global LightGBM model.
- Raw sales target.
- Same LightGBM configuration as the committed plain model unless a mechanical feature-count constraint requires a non-material adjustment.
- Twelve cutoff-safe historical origins are used to build the training set up to the official cutoff.
- Historical target-derived features are generated only from data available on or before each origin.

## 76-feature contract

- Stage 5-B relative price/discount features are retained.
- Stage 5-E adds cutoff-safe lag, rolling, price/discount dynamics, group-demand, and interaction features.
- The model matrix must remain exactly the approved 76-feature allowlist.
- Forbidden fields remain excluded: future `sales`, future `total_orders`, future availability, weights, `id`, `sales_hat`, and any solution target leakage.

## Cutoff logic for the official test horizon

- Final training cutoff: 2024-06-02.
- Official forecast window: 2024-06-03 through 2024-06-16.
- Training labels are historical only and end on or before the final cutoff.
- Historical price and demand references for official test rows are built only from data through the final cutoff.

## Known-future price/discount policy

- `sell_price_main` and discount fields in the official test grid are treated as Kaggle-known-future benchmark covariates.
- This is benchmark-specific and must not be generalized as operational availability outside the competition.
- These fields may be used, but they must never be mixed with future sales or future `total_orders`.

## Forbidden fields

- Future sales
- Future `total_orders`
- Future availability
- Weights
- `id`
- `sales_hat`
- Solution target values
- Any train/validation leakage

## Clipping decision

- Zero clipping is applied to the final candidate.
- The S5-A diagnostic showed clipping was safe and only marginally beneficial.
- Clipping is a documented business-valid post-processing step, not a training trick.

## Candidate validation steps

1. Build the final training frame from the twelve approved origins.
2. Build the official forecast frame from `sales_test.csv` and the solution template grid.
3. Validate exact row count, IDs, ordering, and schema against `solution.csv`.
4. Confirm the candidate uses only the approved 76-feature contract.
5. Confirm no forbidden fields enter the model matrix.
6. Confirm predictions are numeric and non-null.
7. Confirm zero clipping removes any negative predictions.
8. Save the candidate CSV under `submissions/`.

## Submission policy

- No Kaggle submission is made in this step.
- The candidate is prepared for a later pre-submit audit only.
