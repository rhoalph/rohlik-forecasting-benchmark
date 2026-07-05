# Stage 5 Kaggle Candidate Plan

## Purpose

Prepare one Kaggle submission candidate for Stage 5 using the public-solution-informed feature batch that outperformed the Stage 3 plain model locally.

This is not a new benchmark freeze. It is a candidate-preparation step after the S5-B diagnostic showed a consistent local gain over the Stage 3 plain model.

## Relationship to Stage 4

- Stage 4 is the frozen audited first submission.
- Stage 5 keeps the same plain LightGBM training structure and official grid alignment.
- Stage 5 adds a controlled relative price/discount feature batch based on public Kaggle solution ideas and the Stage 5 local ablation results.
- Stage 5 does not overwrite Stage 4 artifacts.

## Local validation evidence

The approved S5-B diagnostic established the following local all-fold results:

- Stage 3 plain all-fold mean WMAE: 20.6963289733
- Stage 5 S5-B all-fold mean WMAE: 20.4635858813
- Mean WMAE delta vs Stage 3 plain: -0.2327430920
- Stage 5 S5-B all-fold mean WAPE: 0.2110805695
- Stage 5 S5-B all-fold mean bias: -0.0471988703
- Every fold improved over the Stage 3 plain model

This is the evidence base for preparing the candidate.

## Final training design

- Final cutoff: 2024-06-02
- Official forecast window: 2024-06-03 through 2024-06-16
- The candidate uses the same twelve cutoff-safe historical origins style as Stage 4, shifted to the final cutoff.
- Training labels remain historical only and end on or before the final cutoff.
- Historical feature statistics are computed only from rows available on or before each origin.

## Final feature list

The candidate keeps the approved Stage 3 feature contract and adds ten relative price/discount features.

Stage 3 approved features retained:

- `unique_id`
- `product_unique_id`
- `warehouse`
- `L1_category_name_en`
- `L2_category_name_en`
- `L3_category_name_en`
- `L4_category_name_en`
- `horizon_day`
- `day_of_week`
- `iso_week`
- `month`
- `weekend_flag`
- `holiday`
- `shops_closed`
- `winter_school_holidays`
- `school_holidays`
- `sell_price_main`
- `type_0_discount`
- `type_1_discount`
- `type_2_discount`
- `type_3_discount`
- `type_4_discount`
- `type_5_discount`
- `type_6_discount`
- `last_observed_sales`
- `last_observed_available`
- `trailing_7_mean`
- `trailing_7_available`
- `trailing_14_mean`
- `trailing_14_available`
- `same_weekday_sales`
- `same_weekday_direct_available`
- `historical_mean_sales`
- `historical_median_sales`
- `historical_stats_available`
- `observed_history_row_count`

Added Stage 5 relative price/discount features:

- `price_relative_to_item_history_median`
- `price_relative_to_item_history_mean`
- `price_change_vs_last_observed`
- `price_rank_within_warehouse_date`
- `price_zscore_vs_item_history`
- `any_discount_active`
- `max_discount`
- `total_discount`
- `active_discount_count`
- `discount_depth_relative_to_item_history`

## Cutoff logic for historical price references

- Historical price and discount reference values are computed only from data at or before each training origin.
- For the official candidate, historical reference values are computed only through 2024-06-02.
- No future sales are used in any reference calculation.
- No future `total_orders` are used.
- No future availability is used.

## Known-future policy for official test price/discount fields

- `sell_price_main` and the seven discount fields in the official test set are treated as Kaggle-known-future benchmark covariates.
- They are allowed because they are present in the official test request grid.
- This policy is benchmark-specific and does not imply operational availability outside Kaggle.
- They must not be mixed with future sales or future `total_orders`.

## Forbidden fields

The candidate excludes:

- `sales`
- `weight`
- `total_orders`
- `availability`
- `id`
- `sales_hat`
- any hidden future label values from `solution.csv`

## Clipping decision and rationale

Clipping at zero will be applied to the final candidate.

Rationale:

- The S5-A diagnostic showed clipping is safe.
- The full all-fold mean WMAE improves slightly with clipping.
- Clipping is a business-valid post-processing choice.
- The number of clipped rows will be reported in the candidate report.

If the candidate validation finds any mismatch, the unclipped diagnostic can be rebuilt separately, but the current plan is to submit the clipped candidate file only after audit approval.

## Candidate validation steps

Before any submission, validate:

- file columns are exactly `id` and `sales_hat`
- row count is exactly 47,021
- IDs match `solution.csv`
- row order matches `solution.csv`
- no duplicates or missing IDs
- `sales_hat` is numeric
- no null predictions
- negative prediction count is recorded before and after clipping
- approved feature set only is used
- no target leakage
- no future `total_orders`
- no future availability
- no weights in the feature matrix
- `eval/` and `dataguard/` remain unchanged

## Files to be created

- `scripts/run_stage5_kaggle_candidate.py`
- `tests/test_stage5_kaggle_candidate.py`
- `reports/stage5_kaggle_candidate_report.md`
- `submissions/stage5_s5b_price_discount_candidate.csv`

## What will not be changed

- `eval/`
- `dataguard/`
- Stage 4 official result rows
- Stage 4 submission file
- Kaggle submission history

## No Kaggle submission yet

This is candidate preparation only. The file will be generated and validated locally, but not submitted until a separate human decision.

