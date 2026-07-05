# Stage 5 S5-B Relative Price and Discount Feature Results

## Public-solution idea basis

The feature family is motivated by public competition writeups that emphasized price/discount signal, relative pricing, and engineered demand context, plus Stage 3 ablations showing that price/discount materially contributes to score.

## Feature list

- price_relative_to_item_history_median
- price_relative_to_item_history_mean
- price_change_vs_last_observed
- price_rank_within_warehouse_date
- price_zscore_vs_item_history
- any_discount_active
- max_discount
- total_discount
- active_discount_count
- discount_depth_relative_to_item_history

## Feature availability and cutoff logic

| Feature | Availability | Cutoff logic |
|---|---|---|
| price_relative_to_item_history_median | historical only | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| price_relative_to_item_history_mean | historical only | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| price_change_vs_last_observed | historical only | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| price_rank_within_warehouse_date | known future | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| price_zscore_vs_item_history | historical only | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| any_discount_active | known future | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| max_discount | known future | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| total_discount | known future | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| active_discount_count | known future | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |
| discount_depth_relative_to_item_history | historical only | Uses only historical references through each origin or the final cutoff; request-side price/discount values remain benchmark-known-future covariates |

## Fold results

| Fold | WMAE | WAPE | Bias | Runtime (min) | Peak RSS (MiB) | Negative predictions | Beats Stage 3 plain? |
|---|---:|---:|---:|---:|---:|---:|---|
| F1 | 20.3036449915 | 0.2140886895 | -0.0119983091 | 4.132 | 1629.65 | 15 | Yes |
| F2 | 22.3078009161 | 0.2266340500 | -0.0927869585 | 5.424 | 1878.68 | 20 | Yes |
| F3 | 20.2015537523 | 0.2006601345 | -0.0128619734 | 4.204 | 1911.01 | 16 | Yes |
| F4 | 19.0413438652 | 0.2029394039 | -0.0711482402 | 4.081 | 1911.01 | 8 | Yes |

## Comparison to Stage 3 plain model

- Stage 3 plain all-fold mean WMAE: 20.6963289733
- Stage 5 price/discount mean WMAE: 20.4635858813
- Mean WMAE delta: -0.2327430920
- Stage 5 price/discount mean bias: -0.0471988703
- Stage 4 official private WMAE (context only): 21.91884
- Stage 2 all-fold mean WMAE (floor context): 30.5644166593

## Bias and stability assessment

The feature batch should be considered useful only if it improves WMAE without making bias materially worse across folds.

## Leakage risk assessment

Main risk is accidental leakage through price/discount reference statistics or price ranks. The implementation keeps all historical statistics cutoff-safe and excludes target, total_orders, availability, weights, ids, and solution labels.

## Runtime and memory

- Total runtime: 17.841 min
- Peak RSS: 1911.01 MiB

## Recommendation

Prepare a Kaggle candidate after review.
