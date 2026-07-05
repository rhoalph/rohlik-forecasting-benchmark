# Stage 3 F1 Diagnostic Ablation Results

Date: 2026-07-05
Status: First approved diagnostic batch complete
Scope: F1 only; diagnostic evidence, not tuning or Kaggle performance

## Purpose

This batch tests why the trusted-with-caveats Stage 3 F1 model reached WMAE 20.5827509351, a 34.03% improvement over the Stage 2 trailing 14-day reference. It addresses the independent review's medium-risk questions about price/discount contribution and multi-family feature effects without changing hyperparameters, adding features, running another fold, or submitting to Kaggle.

The approved batch contains only A9, A10, A1, A2, A3, and A4. Priority 2 and Priority 3 ablations were not run.

## Execution design summary

- F1 cutoff: 2024-05-19.
- F1 validation: 2024-05-20 through 2024-06-02.
- Scored rows: 44,212 of 47,021 requested rows.
- Coverage: 94.026073%.
- A9 and A10 reuse the committed Stage 2 baseline functions.
- A1–A4 use the same 492,826 training rows from the twelve approved origins.
- The full Stage 3 training and validation matrices were generated once; each model received only its explicit approved feature subset.
- LightGBM configuration remained raw-sales, unweighted L1, 300 rounds, learning rate 0.05, 31 leaves, minimum 100 rows per leaf, seed 42, and six threads.
- No clipping, rounding, target transformation, recursive prediction, tuning, ensemble, or new feature was used.
- Frozen F1 grid alignment, weights, WMAE, WAPE, and bias were used unchanged.

## A9 and A10 baseline reproduction

| ID | Reference | WMAE | WAPE | Bias | Difference from committed reference | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| A9 | Trailing 14-day mean | 31.1985803308 | 0.3021110128 | 0.0303957661 | 0 for all metrics | 0.185 |
| A10 | ID/day-of-week median | 30.7150203221 | 0.3411060097 | -0.1929146153 | 0 for all metrics | 0.983 |

Both references reproduced exactly on 44,212 rows and the same frozen F1 coverage. Their displayed runtimes exclude the shared 9.679-second split/context preparation.

## A1–A4 result table

Positive “WMAE change vs full” means the ablation is worse than the complete Stage 3 F1 model. “Gain retained” is the fraction of the full model's 10.615829-point improvement over the trailing 14-day reference that remains under the ablation.

| ID | Feature family | Features | Train rows | Validation rows | WMAE | WAPE | Bias | WMAE change vs full | Gain retained | Runtime (s) | Peak RSS (MiB) | Negative predictions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | All approved except price/discount | 28 | 492,826 | 44,212 | 25.1617710449 | 0.2571474873 | -0.0578253518 | +4.5790201098 | 56.87% | 42.614 | 1,494.87 | 11 |
| A2 | Historical demand only | 12 | 492,826 | 44,212 | 25.7205849285 | 0.3120152621 | -0.0740466250 | +5.1378339934 | 51.60% | 22.175 | 1,494.87 | 33 |
| A3 | Static metadata only | 7 | 492,826 | 44,212 | 28.2910454553 | 0.2879614449 | -0.1073275737 | +7.7082945202 | 27.39% | 39.582 | 1,494.87 | 17 |
| A4 | Known future plus static; no historical demand | 24 | 492,826 | 44,212 | 22.6285595972 | 0.2338271624 | -0.0575188849 | +2.0458086621 | 80.73% | 37.548 | 1,494.87 | 5 |

Model runtimes cover subset selection, fitting, raw prediction, and frozen scoring. They exclude the shared raw loading and 142.559-second feature-generation phase.

## Exact feature sets

### A1 — no price or discount

All approved Stage 3 features except `sell_price_main` and `type_0_discount` through `type_6_discount`. No commercial covariate reached the A1 LightGBM matrix.

### A2 — historical demand only

`last_observed_sales`, `last_observed_available`, `trailing_7_mean`, `trailing_7_available`, `trailing_14_mean`, `trailing_14_available`, `same_weekday_sales`, `same_weekday_direct_available`, `historical_mean_sales`, `historical_median_sales`, `historical_stats_available`, and `observed_history_row_count`.

### A3 — static metadata only

`unique_id`, `product_unique_id`, `warehouse`, and `L1_category_name_en` through `L4_category_name_en`.

### A4 — known future plus static

The seven A3 static features plus `horizon_day`, `day_of_week`, `iso_week`, `month`, `weekend_flag`, four calendar flags, `sell_price_main`, and seven discount fields. No historical-demand feature reached the A4 matrix.

## Runtime and memory

| Shared phase | Result |
|---|---:|
| Raw loading | 7.276 seconds |
| Baseline split/context preparation | 9.679 seconds |
| Twelve-origin plus F1 feature generation | 142.559 seconds |
| Total six-diagnostic wall time | 302.963 seconds / 5.049 minutes |
| Peak process RSS | 1,494.87 MiB / 1.46 GiB |

The run wrote no processed dataset or model artifact.

## Interpretation

### Price and discount contribution

Price and discounts are a major contributor, but not the only contributor. Removing all eight commercial covariates in A1 worsened WMAE by 4.579020 points versus the full model. This gap represents 43.13% of the full model's improvement over the trailing 14-day baseline and exceeds the committed plan's 25% diagnostic threshold.

A1 still scored 25.161771, retaining 56.87% of the full gain over Stage 2. Therefore the result does not depend exclusively on price/discount, but the benchmark-specific commercial covariate policy materially explains the 20.58 reference score.

### Historical-demand contribution

Historical-demand-only A2 scored 25.720585 and retained 51.60% of the full WMAE gain. Historical demand is useful and independently beats both Stage 2 WMAE references, but it is not sufficient to reproduce the full result. Its WAPE of 0.312015 is slightly worse than the trailing 14-day WAPE, and its -7.40% bias shows stronger under-forecasting.

Adding historical demand to the known-future/static A4 family improves WMAE by 2.045809 points to reach the full model. This is meaningful but smaller than the A1 price/discount removal gap. Because tree interactions are non-additive, these deltas are explanatory diagnostics rather than causal feature attributions.

### Static and known-future sufficiency

Static-only A3 scored 28.291045, retaining 27.39% of the full gain. Static identity and hierarchy carry real signal but are not enough to explain the full model. A3 also has -10.73% bias, making it the least balanced model ablation.

Known-future plus static A4 scored 22.628560 and retained 80.73% of the full gain. It is the closest ablation to the complete model. This indicates the combined static, calendar, date, price, and discount family is the dominant tested driver. It also reinforces the caveat that the result depends substantially on Kaggle-known-future covariates and does not prove equivalent operational performance outside the benchmark.

### Plausibility of the original result

The original 20.582751 WMAE remains plausible. Every removal ablation regressed relative to the full model, while different approved families retained distinct portions of the gain. A9 and A10 exactly reproduced the comparison references, ruling out baseline drift. A1 and A2 each remain substantially better than Stage 2 WMAE, and A4 explains most of the improvement without using historical demand.

The evidence supports complementary signal rather than an obvious metric or grid anomaly. It does not remove the benchmark-specific price/discount caveat or quantify the twelve-origin advantage, which remains the main unanswered Mistral finding.

## Leakage and correctness assessment

No new leakage concern was observed:

- all A1–A4 models used the same origin-safe shared feature matrices;
- every feature subset was asserted against an exact allowlist;
- forbidden fields remained absent;
- A1 explicitly excluded all price/discount columns;
- A2 and A3 excluded all known-future fields;
- A4 excluded every historical-demand field;
- prediction keys and coverage remained unchanged;
- frozen scoring rejected any missing, extra, or duplicate key;
- no F2–F4 data, future `total_orders`, future availability, clipping, or tuning was used.

The negative predictions were scored unchanged under the approved policy.

## Recommendation for the next gate

Proceed to human review for Priority 2 ablations, not to F2–F4.

The first batch explains the feature-family contribution sufficiently to support the full F1 result's plausibility. The remaining medium-risk question is the advantage from twelve origins. If approved, A5 should compare the single 2024-05-05 origin with the fixed full feature set. A6 may then assess native categorical handling. Do not run either without explicit approval.

F2–F4 and Kaggle submission remain blocked until Priority 2 results and a new adversarial review are accepted.
