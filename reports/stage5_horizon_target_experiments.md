# Stage 5 Horizon-Specific Direct Model Experiments

## Purpose

Test whether direct horizon-specific models, plus a sqrt target transform, improve on the Stage 5 S5-B candidate enough to justify another Kaggle submission candidate.

This is public-solution-informed diagnostics, not a new freeze.

## Public-solution idea basis

Public competition writeups often used direct horizon models rather than a single global regressor. This experiment isolates that idea while keeping the rest of the Stage 5 policy fixed.

## Experiment design

- Four local folds were evaluated: F1 through F4.
- One LightGBM model was trained per horizon_day, from 1 to 14.
- Two target modes were run:
  - raw sales
  - sqrt(sales)
- The S5-B relative price/discount feature set was included.
- Clipping was evaluated as a final post-processing step.

## Feature contract

Approved feature set used for both variants:

- Stage 3 approved features
- plus the 10 Stage 5 relative price/discount features

No forbidden fields were allowed in the model matrix:

- no future `total_orders`
- no future availability
- no `sales` target column
- no weights
- no `id` or `sales_hat`
- no `solution.csv` target leakage

## Target transform details

- Raw variant: the model predicts sales directly.
- Sqrt variant: the model predicts sqrt(sales), and the inverse is applied by squaring.
- Negative sqrt-space predictions were clipped to zero before inverse squaring.
- Clipping was also applied to the raw variant as a separate final post-processing step.

## Fold-by-fold results

### S5-C raw target

| Fold | Clipped WMAE | Clipped WAPE | Clipped bias | Unclipped WMAE | Unclipped WAPE | Unclipped bias | Training rows | Validation rows | Horizon models | Negative raw preds | Clipped rows | Runtime (min) | Peak RSS (MiB) | Beats Stage 3 plain? | Beats S5-B? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| F1 | 21.6376930586 | 0.2779990123 | -0.0638248544 | 21.6389836142 | 0.2780142181 | -0.0638400602 | 492826 | 44212 | 14 | 54 | 54 | 2.291 | 1591.43 | No | No |
| F2 | 23.1572069431 | 0.2824418576 | -0.1481084727 | 23.1587519564 | 0.2824547069 | -0.1481213220 | 490224 | 43433 | 14 | 56 | 56 | 2.335 | 1739.21 | No | No |
| F3 | 20.6056331634 | 0.2613394672 | -0.0729308903 | 20.6078542948 | 0.2613577416 | -0.0729491647 | 487826 | 42794 | 14 | 59 | 59 | 2.352 | 1812.79 | No | No |
| F4 | 20.3572808638 | 0.2734932853 | -0.1570878538 | 20.3581457375 | 0.2735035307 | -0.1570980992 | 486315 | 42035 | 14 | 52 | 52 | 2.274 | 1812.79 | No | No |

### S5-D sqrt target

| Fold | Clipped WMAE | Clipped WAPE | Clipped bias | Unclipped inverse WMAE | Unclipped inverse WAPE | Unclipped inverse bias | Training rows | Validation rows | Horizon models | Negative sqrt preds | Clipped rows | Runtime (min) | Peak RSS (MiB) | Beats Stage 3 plain? | Beats S5-B? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| F1 | 21.3136476985 | 0.2665437637 | -0.0523073938 | 21.3138127321 | 0.2665455574 | -0.0523056001 | 492826 | 44212 | 14 | 42 | 42 | 2.330 | 1591.43 | No | No |
| F2 | 22.8123250157 | 0.2692157701 | -0.1395141950 | 22.8125682282 | 0.2692179654 | -0.1395119996 | 490224 | 43433 | 14 | 42 | 42 | 2.206 | 1739.21 | No | No |
| F3 | 20.1397196386 | 0.2466832742 | -0.0592865914 | 20.1400222094 | 0.2466862140 | -0.0592836515 | 487826 | 42794 | 14 | 45 | 45 | 2.229 | 1812.79 | Yes | Yes |
| F4 | 19.9843135015 | 0.2588194248 | -0.1430091234 | 19.9843834441 | 0.2588200524 | -0.1430084959 | 486315 | 42035 | 14 | 52 | 52 | 2.229 | 1812.79 | No | No |

## Aggregate results

### S5-C raw target

- Mean clipped WMAE: 21.4394535072
- Mean clipped WAPE: 0.2738184056
- Mean clipped bias: -0.1104880178
- Mean unclipped WMAE: 21.4409339007
- Mean unclipped bias: -0.1105021615

### S5-D sqrt target

- Mean clipped WMAE: 21.0625014636
- Mean clipped WAPE: 0.2603155582
- Mean clipped bias: -0.0985293259
- Mean unclipped inverse WMAE: 21.0626966534
- Negative sqrt-space predictions total: 181

## Comparison to existing references

- Stage 3 plain all-fold mean WMAE: 20.6963289733
- Stage 5 S5-B all-fold mean WMAE: 20.4635858813
- Stage 5 official private WMAE: 21.61114

## Runtime, memory, and CPU behavior

- Raw-load time: 8.726 s
- S5-C total runtime: 555.077 s
- S5-D total runtime: 539.642 s
- Combined runtime: about 18.2 min
- Peak RSS: 1812.79 MiB
- CPU utilization remained high throughout the run, which is consistent with a compute-bound LightGBM workload.

## Hack box assessment

The hack box handled the run comfortably.

- Peak RSS stayed well below the 12 GiB guardrail.
- No large caches or processed artifacts were written.
- The job completed without interruption.

## Leakage and cutoff safety assessment

- Training origins remained cutoff-safe.
- Validation labels did not leak into feature generation.
- The approved feature set remained intact.
- No future `total_orders`, no future availability, and no solution leakage were introduced.

## Recommendation

Reject horizon-specific direct routing as the next Kaggle candidate path.

The raw variant was worse than both Stage 3 plain and Stage 5 S5-B on every fold.
The sqrt variant improved F3, but not enough to improve the all-fold mean or to beat S5-B overall.

If further Stage 5 work is approved, it should target a different hypothesis rather than horizon routing alone.

