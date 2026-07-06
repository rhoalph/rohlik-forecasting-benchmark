# Stage 5-E Stronger Feature Engineering Plan

## Purpose

Test whether a stronger, cutoff-safe feature set built on top of the current Stage 5 S5-B candidate can improve local F1-F4 validation using one global LightGBM model.

This is public-solution-informed feature engineering. It is not independent discovery, and it is not a Kaggle submission candidate yet.

## Public-solution idea basis

Public winning solutions and strong notebooks for Rohlik consistently leaned on:

- richer lag and rolling demand features
- explicit price and discount dynamics
- warehouse/category and product-level historical context
- simple interactions that expose discount response by horizon or weekday

Stage 5-C/D showed that horizon routing and target transformation alone did not beat the current S5-B candidate, so the next controlled step is to strengthen the feature space while keeping one global model.

## Proposed feature list

This experiment keeps the approved Stage 5-B features and adds only cutoff-safe features from the following groups.

Additional lag features:

- `lag_7_sales`
- `lag_14_sales`
- `lag_21_sales`
- `lag_28_sales`
- `same_weekday_2w_sales`
- `same_weekday_3w_sales`
- `same_weekday_4w_sales`

Additional rolling features:

- `rolling_7_median_sales`
- `rolling_14_median_sales`
- `rolling_28_mean_sales`
- `rolling_28_median_sales`
- `rolling_7_to_28_mean_ratio`
- `recent_trend_7_vs_28`

Price and discount dynamics:

- `price_relative_to_28d_item_median`
- `price_relative_to_28d_item_mean`
- `price_change_vs_last_observed_price`
- `discount_active_count`
- `discount_change_vs_last_observed_total_discount`
- `price_x_any_discount`
- `price_x_max_discount`

Group-level historical demand features:

- `warehouse_category_l2_mean_sales`
- `warehouse_category_l2_median_sales`
- `product_mean_sales_across_warehouses`
- `product_median_sales_across_warehouses`
- `category_l2_mean_sales`
- `item_share_of_warehouse_category_l2_sales`

Horizon interaction features:

- `horizon_x_any_discount`
- `horizon_x_max_discount`
- `horizon_x_recent_trend_7_vs_28`
- `dayofweek_x_any_discount`

## Cutoff logic

For every local fold:

- training rows come only from data on or before the fold cutoff
- feature statistics are computed only from rows dated on or before the origin used for that training slice
- lag features are direct lookups only when the source date is on or before the origin; otherwise a cutoff-safe fallback is used
- rolling features use only the history window ending at the origin
- group features use only pre-origin history

For the official candidate path, if one is later approved:

- historical reference values will use data only through 2024-06-02
- official test price/discount values are allowed as Kaggle-known-future covariates
- no future sales, future `total_orders`, future availability, weights, or target leakage are allowed

## Leakage risks

Primary risks:

- lag features accidentally pulling from beyond the origin
- rolling windows accidentally including validation rows
- group features accidentally using future rows or request-period labels
- price and discount interactions accidentally mixing known-future covariates with target information

Mitigations:

- all feature builders will assert cutoff-safe history
- the model matrix will be checked against an explicit allowlist
- `solution.csv`, target columns, `total_orders`, `availability`, `weight`, `id`, and `sales_hat` remain forbidden

## Runtime and memory risks

The experiment is heavier than S5-B because it adds additional historical lookups and group aggregations.

Expected risks:

- longer feature generation per fold
- higher peak RSS due to larger matrices
- more LightGBM training time from the larger feature set

Guardrails:

- keep runtime under 2 hours
- keep peak RSS under 12 GiB
- do not write large caches or processed intermediates

## Success criteria

This experiment is successful only if all of the following are true:

- mean F1-F4 WMAE beats S5-B mean WMAE of 20.4635858813
- no fold worsens materially
- bias does not worsen materially
- runtime stays under 2 hours
- peak RSS stays under 12 GiB
- no leakage or cutoff concern appears
- tests and Stage 1 validation pass

## Stop criteria

Stop immediately and do not convert this into a Kaggle candidate if:

- feature construction shows a cutoff violation
- any forbidden field appears in a feature matrix
- local performance improves suspiciously without a defensible reason
- runtime or memory moves toward the guardrail limits
- tests fail
- Stage 1 validation fails

