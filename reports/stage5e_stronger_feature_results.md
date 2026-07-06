# Stage 5-E Stronger Feature Results

## Purpose

Evaluate whether a stronger cutoff-safe feature set on one global LightGBM can beat the current Stage 5 S5-B local mean WMAE.

## Feature list

- Approved feature count: 76
- Stage 5-B features retained.
- New lag, rolling, price/discount dynamics, group-demand, and interaction features added.

## Skipped features

- No approved Stage 5-E feature group was skipped.
- Horizon-specific routing was intentionally not used because the prior S5-C/D diagnostic failed to beat S5-B overall.

## Fold-by-fold results

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Training rows | Validation rows | Feature count | Negative preds | Clipped rows | Runtime (min) | Peak RSS (MiB) | Beats S5-B? | Beats Stage 3 plain? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| F1 | 19.6069995273 | 0.2071636796 | -0.0217893794 | 19.6060885659 | 0.2071586518 | -0.0217843517 | 492826 | 44212 | 76 | 35 | 35 | 6.512 | 1959.53 | Yes | Yes |
| F2 | 21.3218385474 | 0.2158200345 | -0.0851968572 | 21.3163474886 | 0.2158022919 | -0.0851791147 | 490224 | 43433 | 76 | 65 | 65 | 6.393 | 2069.71 | Yes | Yes |
| F3 | 18.7434564665 | 0.1912725362 | -0.0131566914 | 18.7427525639 | 0.1912693329 | -0.0131534880 | 487826 | 42794 | 76 | 21 | 21 | 6.246 | 2095.95 | Yes | Yes |
| F4 | 18.4974751850 | 0.1969029064 | -0.0688270785 | 18.4964147713 | 0.1968938074 | -0.0688179796 | 486315 | 42035 | 76 | 44 | 44 | 6.367 | 2212.24 | Yes | Yes |

## Aggregate result table

| Variant | Mean WMAE | Mean WAPE | Mean bias | Mean runtime (min) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Raw primary | 19.5424424315 | 0.2027897892 | -0.0472425016 | 6.379 | 2212.24 |
| Clipped diagnostic | 19.5404008474 | 0.2027810210 | -0.0472337335 | 6.379 | 2212.24 |

## Comparison context

- Stage 3 plain all-fold mean WMAE: 20.6963289733
- Stage 5 S5-B all-fold mean WMAE: 20.4635858813
- Stage 5 official private WMAE: 21.61114
- Stage 5-E all-fold mean WMAE improvement vs S5-B: 0.9211434498

## Bias and stability analysis

The stronger feature set materially improved over S5-B on every fold. The all-fold mean WMAE dropped from 20.4635858813 to 19.5424424315, and the bias stayed in the same general range with a slight improvement in magnitude after clipping.

## Runtime and memory behavior

- Raw load seconds: 7.675
- Mean runtime: 6.379 min
- Peak RSS: 2212.24 MiB

## Leakage and cutoff safety assessment

- Feature construction remained cutoff-safe.
- No forbidden fields entered the model matrix.
- Historical references used only pre-origin data.
- The experiment stayed within the hack-box memory guardrail.

## Recommendation

Prepare a new Kaggle candidate after review.
