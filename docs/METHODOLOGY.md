# Methodology

This project was run as a retrospective, audited benchmark on the Kaggle Rohlik Sales Forecasting Challenge V2.

## Stage 1: frozen evaluation and leakage controls

Stage 1 established the protected evaluation layer:

- official Kaggle-style scoring
- weight handling and key alignment
- time-based backtest splits
- cutoff-safe validation logic
- leakage guards in `eval/` and `dataguard/`

The rule was simple: no later stage could be promoted if it violated the frozen evaluation or leakage constraints.

## Stage 2: naive baselines

Stage 2 established a diagnostic floor with simple baselines:

- zero forecast
- global median
- last observed value
- same weekday last week
- trailing 7-day mean
- trailing 14-day mean
- item/day-of-week median

These were local diagnostics only. They were used to validate scoring and to prevent accidental leakage before model work began.

## Stage 3 / 4: plain LightGBM and first official submission

Stage 3 introduced one global LightGBM model with a small safe feature set.

Stage 4 created the first official Kaggle submission:

- raw target
- no target transform
- no recursion
- no ensembling
- no tuning sweep

Stage 4 is the first externally scored benchmark point, but not the final benchmark result.

## Stage 5-B: price/discount features

Stage 5-B added benchmark-approved price and discount features.

The key assumption was explicit:

- price and discount values provided in the Kaggle test set were treated as known-future benchmark covariates
- that policy is benchmark-specific and should not be generalized to operational forecasting without review

## Stage 5-E: stronger features

Stage 5-E expanded the feature contract to a larger cutoff-safe set:

- additional lags
- rolling statistics
- historical price and discount dynamics
- cross-warehouse and category-level aggregates
- simple interactions

Stage 5-E materially improved the official score.

## Stage 5-F / G: objective diversity and fixed blend

Stage 5-F compared raw L1, sqrt, log1p, Tweedie, and Poisson objectives.

Stage 5-G promoted a fixed blend:

- one global raw L1 model
- one global Tweedie 1.1 model
- fixed 50/50 blend

Why this was allowed:

- membership was globally fixed
- weights were fixed across folds and applicable to the official test set
- predictions were aligned by key before blending
- no validation labels were used to choose a different blend per fold

Why the fold-specific equal blend was rejected:

- membership changed by fold
- that made it non-generalizable to the official test set

## Stage 5-H: supply-chain hypothesis rejected

Stage 5-H tested a warehouse-category pressure hypothesis:

> SKU demand should not be forecast in isolation; it should be informed by warehouse-category demand pressure, category mean reversion, and changing item share.

The features were cutoff-safe and audited, but the experiment did not beat Stage 5-G. The hypothesis was documented and rejected for the current model.

## Forbidden fields and cutoff-safety rules

Across the project, the following were forbidden as model features:

- future sales
- future `total_orders`
- future availability
- weights
- `id`
- `sales_hat`
- solution targets from `solution.csv`

Historical features had to obey forecast-origin discipline:

- every target-derived feature had to use only data available on or before the cutoff/origin
- rolling windows could not include validation labels
- known-future values were allowed only where explicitly documented

## Why fixed 50/50 blending was allowed

The Stage 5-G blend was a fixed rule, not a fold-specific choice.

It was allowed because:

- the two models were trained with the same feature contract
- their predictions were aligned on the same request keys
- the blend rule was fixed in advance
- the rule could be applied unchanged to the Kaggle test set

