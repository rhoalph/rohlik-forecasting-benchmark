# Stage 5-H Supply-Chain Category Pressure Hypothesis Plan

## Purpose

Test whether the current best benchmark recipe can be improved by adding
cutoff-safe warehouse/category pressure features on top of the Stage 5-G
fixed 50/50 raw L1 + Tweedie 1.1 blend recipe.

The executive hypothesis is:

> SKU demand should not be forecast in isolation. It should be informed by
> warehouse-category demand pressure, category mean reversion, and the SKU’s
> changing share within its warehouse/category.

This is a local diagnostic stage only. It is not a Kaggle submission stage.

## Base recipe

- Start from the Stage 5-E 76-feature contract.
- Train one global LightGBM raw L1 model.
- Train one global LightGBM Tweedie 1.1 model.
- Blend the two models with a fixed 50/50 rule.
- Keep the same no-target-transform, no-horizon-routing, no-Optuna policy.

## Added feature groups

Only the following Stage 5-H features are allowed:

### A. Warehouse-category demand pressure

- `wh_cat_l2_sales_7d_sum`
- `wh_cat_l2_sales_14d_sum`
- `wh_cat_l2_sales_28d_sum`
- `wh_cat_l2_sales_7d_mean`
- `wh_cat_l2_sales_28d_mean`
- `wh_cat_l2_7d_vs_28d_ratio`
- `wh_cat_l2_7d_minus_28d_mean`
- `wh_cat_l2_reversion_pressure`

`wh_cat_l2_reversion_pressure` is defined as:

`wh_cat_l2_sales_7d_mean / wh_cat_l2_sales_28d_mean - 1`

with a safe zero fallback when the denominator is missing or zero.

### B. Item share within category

- `item_share_of_wh_cat_l2_7d`
- `item_share_of_wh_cat_l2_28d`
- `item_share_7d_minus_28d`
- `item_share_7d_to_28d_ratio`

### C. Simple interactions

- `horizon_x_wh_cat_l2_reversion_pressure`
- `discount_x_wh_cat_l2_reversion_pressure`
- `relative_price_x_wh_cat_l2_reversion_pressure`

## Cutoff safety design

For each training origin:

- compute category totals only from sales history dated on or before the origin;
- exclude validation-period labels from all category totals;
- do not use future sales;
- do not use future `total_orders`;
- do not use future availability;
- keep known-future price and discount values only in the benchmark-approved way.

For the final test-style feature frame, historical statistics must come only from
data available through `2024-06-02`.

## Validation design

Run the four approved local folds:

- F1 cutoff: `2024-05-19`
- F2 cutoff: `2024-05-05`
- F3 cutoff: `2024-04-21`
- F4 cutoff: `2024-04-07`

For each fold, score:

- Stage 5-H raw L1 model
- Stage 5-H Tweedie 1.1 model
- Stage 5-H fixed 50/50 raw + Tweedie blend

Record WMAE, WAPE, bias, runtime, coverage, and clipped diagnostics.

## Success criteria

The hypothesis is considered supported if the fixed blend:

- beats the Stage 5-G fixed 50/50 local mean WMAE of `19.0089170845`,
- improves most folds, and
- does not materially worsen bias or stability.

## Stop criteria

Stop and document the result if any of the following occur:

- the new features do not beat Stage 5-G on mean WMAE;
- the fold pattern is unstable or suspicious;
- a leakage concern appears;
- the feature batch becomes slow or memory heavy;
- the model matrix violates the approved feature contract.

## Governance

- No Kaggle submission in this stage.
- No modification to frozen `eval/` or `dataguard/`.
- No new features outside the approved Stage 5-H allowlist.
- No fold-specific model or blend selection.

