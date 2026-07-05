# Stage 3 F1 Plain LightGBM Results

Date: 2026-07-05
Status: F1 complete; pending adversarial leakage review
Scope: local F1 diagnostic only; not a Kaggle result

## Outcome

The approved F1-only plain LightGBM run completed on the frozen F1 mask. All declared metric, resource, test, and protected-directory gates passed. The WMAE improvement over the Stage 2 F1 reference is 34.03%, exceeding the plan's 20% suspicious-score threshold. The result is therefore recorded as `pending_adversarial_review` and must not be treated as trusted performance evidence yet.

No F2–F4 fold was run. No prediction was clipped or rounded. No model, processed dataset, or submission artifact was written. Nothing was submitted to Kaggle.

## Model configuration

| Setting | Value |
|---|---|
| Library | LightGBM 4.6.0 native API |
| Structure | One global model |
| Objective | Unweighted `regression_l1` |
| Target | Raw `sales` |
| Boosting rounds | 300 fixed rounds |
| Learning rate | 0.05 |
| Leaves | 31 |
| Minimum rows per leaf | 100 |
| Row/feature fractions | 1.0 / 1.0 |
| Seed | 42 |
| Threads | 6 |
| Early stopping | None |
| Target transform | None |
| Clipping/rounding | None |
| Recursive prediction | None |
| Training sample weights | None; official weights used only by frozen scoring |

## Feature list and availability

The final matrix contains exactly 36 approved features.

### Static metadata

- `unique_id`
- `product_unique_id`
- `warehouse`
- `L1_category_name_en`
- `L2_category_name_en`
- `L3_category_name_en`
- `L4_category_name_en`

### Known future

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

Price and discounts are Kaggle-known-future benchmark covariates. Their use is allowed because they are present in the official test set. This does not prove operational availability outside the competition. They were joined without future `sales`, `total_orders`, or `availability` in the feature frame.

### Historical only

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

All target-derived values were computed from non-null observed rows dated on or before the relevant origin. Absent item-date rows were not treated as zero.

### Excluded or forbidden

Future `sales`, `total_orders`, target-window `availability`, evaluation `weight`, `solution.csv` fields, test covariate values in local folds, train-plus-validation fitted transforms, recursive predictions, target encodings, ensembles, and transformed targets were excluded.

## Training-origin summary

| Origin | Label window | Examples | Maximum historical feature date | Origin runtime (s) |
|---|---|---:|---|---:|
| 2024-05-05 | 2024-05-06–2024-05-19 | 43,433 | 2024-05-05 | 11.318 |
| 2024-04-21 | 2024-04-22–2024-05-05 | 42,794 | 2024-04-21 | 11.386 |
| 2024-04-07 | 2024-04-08–2024-04-21 | 42,035 | 2024-04-07 | 11.181 |
| 2024-03-24 | 2024-03-25–2024-04-07 | 40,993 | 2024-03-24 | 11.945 |
| 2024-03-10 | 2024-03-11–2024-03-24 | 41,138 | 2024-03-10 | 11.616 |
| 2024-02-25 | 2024-02-26–2024-03-10 | 41,190 | 2024-02-25 | 11.753 |
| 2024-02-11 | 2024-02-12–2024-02-25 | 41,077 | 2024-02-11 | 10.951 |
| 2024-01-28 | 2024-01-29–2024-02-11 | 41,208 | 2024-01-28 | 11.086 |
| 2024-01-14 | 2024-01-15–2024-01-28 | 40,749 | 2024-01-14 | 11.941 |
| 2023-12-31 | 2024-01-01–2024-01-14 | 38,945 | 2023-12-31 | 10.643 |
| 2023-12-17 | 2023-12-18–2023-12-31 | 38,434 | 2023-12-17 | 10.843 |
| 2023-12-03 | 2023-12-04–2023-12-17 | 40,830 | 2023-12-03 | 11.859 |
| **Total** | Twelve non-overlapping windows | **492,826** | No source after its origin | — |

Each same-weekday source-date maximum was also on or before its corresponding origin. The latest training label was 2024-05-19. F1 labels begin on 2024-05-20 and were not available to fitting or feature generation.

## Leakage and correctness assertions

The run asserted:

- every historical feature batch ends on or before its origin;
- every same-weekday source is on or before its origin;
- every historical training label is on or before the F1 cutoff;
- the twelve origin row counts equal the approved contract;
- missing sales labels are excluded and never converted to zero;
- target, weight, `total_orders`, availability, `id`, and `sales_hat` are absent from the model matrix;
- final feature columns exactly equal the approved 36-column allowlist;
- every feature has a declared availability class;
- inventory joins are many-to-one and warehouse-consistent;
- calendar joins are many-to-one by warehouse/date;
- training and validation matrices are generated separately;
- no outer-validation row influences model fitting or categorical mappings;
- prediction keys contain no duplicates;
- frozen outer one-to-one scoring rejects any missing or extra prediction key;
- no recursive prediction, clipping, rounding, or target transformation occurs.

Focused Stage 3 tests cover the exact feature allowlist, cutoff-safe rolling and same-weekday behavior, unchanged negative discounts, post-cutoff history rejection, and forbidden request-field rejection.

## F1 metrics

| Metric | Stage 2 trailing 14-day baseline | Stage 3 plain model | Change |
|---|---:|---:|---:|
| WMAE | 31.1985803308 | 20.5827509351 | -10.6158293957 (34.03% lower) |
| WAPE | 0.3021110128 | 0.2186937932 | -0.0834172196 |
| Bias | +0.0303957661 | -0.0153873760 | -0.0457831420 |

Bias of -0.01539 means aggregate predictions under-forecast actual demand by approximately 1.54%. This is a local business diagnostic, not a Kaggle score.

| Coverage measure | Result |
|---|---:|
| Training rows | 492,826 |
| F1 validation rows scored | 44,212 |
| Official-mask rows requested | 47,021 |
| F1 coverage | 94.026073% |
| Negative raw predictions | 13 |
| Minimum prediction | -1.13617 |
| Maximum prediction | 14,553.91666 |

The 13 negative predictions were scored unchanged under the approved no-clipping policy.

## Runtime and memory

| Phase | Time |
|---|---:|
| Raw loading | 8.621 seconds |
| Twelve-origin training feature generation | 137.502 seconds |
| LightGBM fitting | 49.403 seconds |
| F1 feature generation | 12.151 seconds |
| Prediction and frozen scoring | 2.149 seconds |
| **Total raw-to-F1-score runtime** | **209.827 seconds / 3.497 minutes** |
| **Peak process RSS** | **1,517.70 MiB / 1.48 GiB** |

An initial execution completed model fitting and scoring but failed while serializing one NumPy boolean in the final audit JSON. No result or artifact was recorded from that attempt. The boolean was explicitly converted to a Python `bool`, and the unchanged feature/model configuration was rerun successfully. The runtime above is the successful clean rerun only.

## Success-gate assessment

| Gate | Required | Actual | Result |
|---|---:|---:|---|
| WMAE | Below 31.1985803308 | 20.5827509351 | Pass |
| WAPE | At most 0.3081532331 | 0.2186937932 | Pass |
| Absolute bias | At most 0.10 | 0.0153873760 | Pass |
| Runtime | Below 30 minutes | 3.497 minutes | Pass |
| Peak RSS | Below 12 GiB | 1.48 GiB | Pass |
| F1 rows and coverage | 44,212 / 94.026073% | Exact match | Pass |
| Unit tests | All pass | 37 passed | Pass |
| Full raw-data validation | Pass | Passed | Pass |
| `eval/` and `dataguard/` | Unchanged | Unchanged | Pass |

All declared success gates pass numerically. This does not override the suspicious-score rule.

## Concerns and recommendation

The relative WMAE improvement is 34.03%, above the predeclared 20% suspicious-score threshold. That is a mandatory leakage alarm, not permission to claim model performance. The result may be explained by a global model combining ID, target-date price/discount information, calendar inputs, and multiple cutoff-safe demand summaries, but this explanation must be tested rather than assumed.

Required next step: run a dedicated Stage 3 F1 adversarial leakage and correctness review before committing the implementation or running another fold. It should trace target-window joins, origin-specific aggregates, price/discount isolation, LightGBM matrix columns, category mappings, label alignment, and exact reproducibility of the metric.

Do not run F2–F4, tune parameters, add features, clip predictions, or submit to Kaggle until that review is accepted.
