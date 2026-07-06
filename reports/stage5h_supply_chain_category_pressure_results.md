# Stage 5-H Supply-Chain Category Pressure Results

## Purpose

Test whether a supply-chain hypothesis adds incremental signal beyond the current Stage 5-G benchmark recipe.

## Business hypothesis

SKU demand should not be forecast in isolation. It should be informed by warehouse-category demand pressure, category mean reversion, and the SKU’s changing share within its warehouse/category.

## Implemented features

- Base contract: Stage 5-E approved feature count = 76
- Added Stage 5-H feature count: 15
- Total feature count: 91
- Warehouse-category demand pressure features at 7, 14, and 28 days
- Item share within warehouse/category at 7 and 28 days
- Simple interactions with horizon, discount, and relative price

## Skipped features and why

- No fold-specific category selection was used; membership is fixed by warehouse and L2 category metadata.
- No broader tuning sweep was run beyond the approved feature list.
- No future sales, future `total_orders`, or future availability were used.
- No horizon-specific routing was introduced.

## Cutoff-safety design

- All category totals were computed only from history dated on or before each origin.
- Validation labels were not used in any category totals.
- Zero or missing denominators used safe fallbacks.
- The official-test-style diagnostics remain benchmark-specific and use only data available through the benchmark cutoff.

## F1-F4 fold results

| Fold | Variant | WMAE | WAPE | Bias | Clipped WMAE | Clipped WAPE | Clipped Bias | Negative before clip | Clipped rows | Beats Stage 5-G? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | raw_l1 | 19.8037339046 | 0.2101740218 | -0.0291480336 | 19.8021382058 | 0.2101675606 | -0.0291415724 | 34 | 34 | No |
| F1 | tweedie_1_1 | 19.5050326119 | 0.2066152679 | 0.0163496788 | 19.5050326119 | 0.2066152679 | 0.0163496788 | 0 | 0 | No |
| F1 | fixed_blend | 19.2746009764 | 0.2032521877 | -0.0063991774 | 19.2745970948 | 0.2032521198 | -0.0063991095 | 2 | 2 | No |
| F2 | raw_l1 | 21.2140702216 | 0.2143638700 | -0.0835808698 | 21.2086473806 | 0.2143324645 | -0.0835494643 | 92 | 92 | No |
| F2 | tweedie_1_1 | 20.4775655335 | 0.2023341515 | -0.0645530266 | 20.4775655335 | 0.2023341515 | -0.0645530266 | 0 | 0 | No |
| F2 | fixed_blend | 20.4772965914 | 0.2044148704 | -0.0740669482 | 20.4772267446 | 0.2044140903 | -0.0740661681 | 11 | 11 | No |
| F3 | raw_l1 | 18.6860801154 | 0.1912760829 | -0.0136332045 | 18.6854714863 | 0.1912721090 | -0.0136292306 | 29 | 29 | Yes |
| F3 | tweedie_1_1 | 18.6791705416 | 0.1855633148 | 0.0118371148 | 18.6791705416 | 0.1855633148 | 0.0118371148 | 0 | 0 | Yes |
| F3 | fixed_blend | 18.3469617805 | 0.1836080025 | -0.0008980449 | 18.3469617805 | 0.1836080025 | -0.0008980449 | 0 | 0 | Yes |
| F4 | raw_l1 | 18.5019409927 | 0.1977430984 | -0.0726252053 | 18.5005910515 | 0.1977311505 | -0.0726132574 | 52 | 52 | Yes |
| F4 | tweedie_1_1 | 17.9969267407 | 0.1867679438 | -0.0427853246 | 17.9969267407 | 0.1867679438 | -0.0427853246 | 0 | 0 | Yes |
| F4 | fixed_blend | 17.9742811587 | 0.1886476452 | -0.0577052649 | 17.9741270890 | 0.1886463563 | -0.0577039760 | 13 | 13 | Yes |

## Aggregate metrics

| Variant | Mean WMAE | Mean WAPE | Mean bias | Runtime (min) | Peak RSS (MiB) | Beats Stage 5-G mean? |
|---|---:|---:|---:|---:|---:|---|
| raw_l1 | 19.5514563086 | 0.2033892683 | -0.0497468283 | 5.191 | 2505.42 | No |
| tweedie_1_1 | 19.1646738569 | 0.1953201695 | -0.0197878894 | 5.414 | 2505.42 | No |
| fixed_blend | 19.0182851267 | 0.1949806765 | -0.0347673588 | 10.606 | 2505.42 | No |

- The fixed-blend runtime includes both component model fits plus the blend/scoring step.

## Comparison to prior benchmark references

- Stage 5-E raw L1 local mean WMAE: 19.5424424315
- Stage 5-G fixed 50/50 local mean WMAE: 19.0089170845
- Stage 5-G official private WMAE: 20.14904

## Decision: hypothesis_rejected_for_current_model

The fixed blend did not materially improve mean WMAE over Stage 5-G, so the hypothesis is rejected for the current model.

## Caveats

- This remains a benchmark setting.
- Price and discount still depend on benchmark-specific known-future covariates.
- Category logic may be dataset-specific.
- Local validation may not transfer exactly to Kaggle hidden scoring.
