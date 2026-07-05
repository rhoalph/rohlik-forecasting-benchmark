# Stage 3 F1 Diagnostic Ablation Plan

Date: 2026-07-05
Status: Plan only; execution requires explicit human approval
Reference verdict: Stage 3 F1 is trusted with caveats

## Purpose and restrictions

These ablations are diagnostic, not tuning. Their purpose is to explain the 34.03% local F1 WMAE improvement and address the three medium-risk findings in the independent Mistral review:

1. suspicious improvement magnitude;
2. unverified contribution from Kaggle-known-future price and discounts; and
3. unquantified advantage from twelve historical training origins.

The following controls apply to every ablation:

- Use F1 only: cutoff 2024-05-19 and validation 2024-05-20 through 2024-06-02.
- Use the same 44,212 scored rows, 47,021 requested rows, and 94.026073% coverage.
- Use frozen key alignment, weights, WMAE, WAPE, and bias from `eval/`.
- Use frozen cutoff and availability controls from `dataguard/`.
- Do not modify `eval/` or `dataguard/`.
- Do not run F2–F4 before the ablations and their review are complete.
- Do not submit to Kaggle.
- Do not perform a hyperparameter search.
- Keep the fixed LightGBM configuration, raw-sales target, unweighted L1 objective, and seeds unless an ablation explicitly concerns prediction post-processing.
- Do not add features. Each training ablation may only select a subset of the approved 36 features or approved origins.
- Do not choose a feature set based on which score is best. Report all approved ablations, including regressions.
- Append each completed diagnostic as a separate, clearly labeled ablation row in `results.csv`.

The trusted-with-caveats full-model reference is:

| Measure | Reference |
|---|---:|
| WMAE | 20.5827509351 |
| WAPE | 0.2186937932 |
| Bias | -0.0153873760 |
| Stage 2 trailing 14-day WMAE | 31.1985803308 |
| Stage 2 ID/day-of-week median WMAE | 30.7150203221 |
| Full-model improvement over trailing 14-day | 10.6158293957 WMAE points |

No individual ablation “passes” merely by improving the score. Success means producing a reproducible explanation of which approved information and design choices account for the reference result without creating leakage or changing the evaluation population.

## Shared execution and reporting contract

Every training ablation must report:

- exact feature names and availability classes;
- exact historical training origins and training-row count;
- WMAE, WAPE, bias, scored rows, and coverage;
- absolute and relative change from the full Stage 3 F1 model;
- comparison with both Stage 2 F1 reference baselines;
- feature generation, model fitting, prediction, and total runtime;
- peak process RSS;
- negative prediction count and range;
- all cutoff, forbidden-field, join, feature-allowlist, and exact-key assertions;
- confirmation that no hyperparameter or frozen scoring code changed.

The existing Stage 3 model must not be overwritten or retrospectively relabeled based on an ablation. Ablation rows remain diagnostics.

## Priority 1: Feature-family ablations

### A1. No price or discount features

| Item | Plan |
|---|---|
| Purpose | Quantify how much of the large improvement depends on Kaggle-known-future commercial covariates. |
| Feature set | Approved 36-feature matrix minus `sell_price_main` and `type_0_discount` through `type_6_discount`; 28 features remain. |
| Expected diagnostic value | Directly addresses Mistral finding M2 and separates benchmark-specific price/promotion value from the remaining model. |
| Changes model training | Yes. Retrain the same fixed LightGBM configuration on the same twelve origins using the reduced matrix. |
| Changes scoring | No. Score raw predictions on the identical frozen F1 grid. |
| Leakage risks | Lower than the full model, but historical target features must still be origin-safe. Assert the eight removed fields are absent rather than silently retained. |
| Runtime/memory risk | Similar feature-generation cost; slightly lower model memory and fitting time. |
| Interpretation | Contribution is `WMAE_without_price - 20.5827509351`. A large positive delta explains part of the full result. If the delta exceeds 2.65396 points, price/discount explains more than 25% of the full 10.61583-point improvement. Little change means price/discount is not the main driver. An improvement after removal suggests harmful or unstable commercial covariates and requires investigation. |

### A2. Historical-demand-only features

| Item | Plan |
|---|---|
| Purpose | Measure the value of cutoff-safe demand history without static, calendar, price, or discount inputs. |
| Feature set | `last_observed_sales`, `last_observed_available`, `trailing_7_mean`, `trailing_7_available`, `trailing_14_mean`, `trailing_14_available`, `same_weekday_sales`, `same_weekday_direct_available`, `historical_mean_sales`, `historical_median_sales`, `historical_stats_available`, and `observed_history_row_count`. |
| Expected diagnostic value | Shows whether nonlinear combinations of simple historical statistics explain most of the improvement over the Stage 2 baselines. |
| Changes model training | Yes. Retrain with the same twelve origins and fixed LightGBM settings using 12 historical-only features. |
| Changes scoring | No. |
| Leakage risks | Target lineage is the main risk. Every statistic and fallback must be recomputed independently through each origin. No target-window sales may enter features. |
| Runtime/memory risk | Origin aggregation remains expensive; fitting should be faster and smaller. |
| Interpretation | A result near 20.58275 indicates historical-demand interactions are the primary driver. A result near the 30.6–31.2 baseline range indicates most gains require static or known-future information. A worse-than-baseline result shows historical statistics alone do not justify the model gain. |

### A3. Static-metadata-only features

| Item | Plan |
|---|---|
| Purpose | Establish how much signal comes from stable item, product, warehouse, and category identity alone. |
| Feature set | `unique_id`, `product_unique_id`, `warehouse`, and `L1_category_name_en` through `L4_category_name_en`; 7 features. |
| Expected diagnostic value | Quantifies memorization and hierarchy effects without date, commercial, or historical-demand information. |
| Changes model training | Yes. Retrain with the same twelve origins and fixed configuration. Labels remain origin-specific even though features are static. |
| Changes scoring | No. |
| Leakage risks | Inventory snapshot is the only substantive assumption. Category encodings must still come solely from inventory and never labels. |
| Runtime/memory risk | Training-frame assembly remains required; model fitting should be materially cheaper. |
| Interpretation | Poor performance is expected and establishes a static-information floor. A result near the full model would indicate strong ID memorization or an unintended design advantage and trigger deeper review. |

### A4. Known-future plus static, without historical demand

| Item | Plan |
|---|---|
| Purpose | Determine whether static identity plus target-date calendar, price, and discount information can explain the result without historical demand. |
| Feature set | The 7 static features plus all 17 known-future features: horizon/date fields, four calendar flags, `sell_price_main`, and seven discounts; 24 features total. No historical-demand feature or fallback value. |
| Expected diagnostic value | Separates calendar and Kaggle-known-future commercial signal from target-derived history. |
| Changes model training | Yes. Retrain on the same twelve origins with fixed settings. |
| Changes scoring | No. |
| Leakage risks | Target-window price and discount joins require the same strict covariate-only selection. Future `sales`, `total_orders`, and `availability` must be absent. |
| Runtime/memory risk | Lower aggregate-feature cost may be possible, but use the reviewed data path rather than introducing an optimized alternative during diagnostics. Model cost is moderate. |
| Interpretation | A result near the full model means known-future/static information drives most of the gain and strengthens the benchmark-specific caveat. A large regression combined with a strong historical-only result means demand history is essential. Unexpectedly extreme performance triggers renewed join and covariate review. |

## Priority 2: Training-design and categorical diagnostics

### A5. Single most-recent training origin

| Item | Plan |
|---|---|
| Purpose | Quantify the advantage from twelve historical origins identified in Mistral finding M3. |
| Feature set | All approved 36 features. |
| Training data | Only origin 2024-05-05, label window 2024-05-06 through 2024-05-19, and exactly 43,433 available-label rows. |
| Expected diagnostic value | Provides the closest direct estimate of how much temporal coverage and multiple demand regimes help the global model. |
| Changes model training | Yes. Same configuration, one origin instead of twelve. No compensation through extra rounds or tuning. |
| Changes scoring | No. |
| Leakage risks | The single training target window still ends before F1. All historical features must end on 2024-05-05. |
| Runtime/memory risk | Substantially lower than the full run. |
| Interpretation | Similar WMAE indicates little multi-origin benefit. A large regression quantifies multi-origin coverage as a major contributor. Better performance suggests older origins introduce drift and must be documented, not optimized away during this diagnostic phase. |

### A6. No native categorical handling

| Item | Plan |
|---|---|
| Purpose | Test whether LightGBM's native categorical treatment materially contributes to the result. |
| Feature set | All 36 approved features, with the same deterministic integer codes, but pass an empty categorical-feature list so LightGBM treats the codes as numeric. |
| Expected diagnostic value | Isolates categorical handling while keeping matrix columns and training rows unchanged. |
| Changes model training | Yes. Only the categorical interpretation changes; all hyperparameters remain fixed. |
| Changes scoring | No. |
| Leakage risks | No new target leakage, but integer codes have arbitrary ordinal meaning. This run is diagnostic and cannot replace the reviewed model. Assert category mappings are identical to the reference. |
| Runtime/memory risk | Similar or slightly lower model cost. |
| Interpretation | Degradation attributes value to native categorical splits. Little change indicates categorical handling is not a major driver. Improvement indicates ordinal-code artifacts or instability and requires review. If reviewers consider numeric treatment misleading, replace this run with a separately approved removal of the nine categorical columns rather than running both variants. |

## Priority 3: Prediction and segment diagnostics

### A7. Clip negative predictions to zero

| Item | Plan |
|---|---|
| Purpose | Measure sensitivity to the 13 negative raw predictions. |
| Feature set | No feature change; use the full reference model's raw prediction vector. |
| Expected diagnostic value | Quantifies the practical effect of nonnegative demand enforcement. |
| Changes model training | No. Compute raw and clipped diagnostics from the same fitted model and prediction vector during an approved reproducibility run. |
| Changes scoring | The frozen scorer is unchanged, but it receives `max(0, sales_hat)` for a separately labeled diagnostic score. Raw score remains the official Stage 3 F1 reference. |
| Leakage risks | None from targets if clipping is a fixed zero bound. Do not choose a threshold using labels. |
| Runtime/memory risk | Negligible after predictions exist. The current project writes no model artifact, so obtaining the vector may require the approved reference consistency run. |
| Interpretation | A tiny delta confirms clipping is immaterial. A meaningful gain concentrated in high-weight rows requires documentation but is post-processing, not model improvement. A regression would be unexpected and should trigger alignment verification. |

### A8. Per-warehouse and per-category improvement analysis

| Item | Plan |
|---|---|
| Purpose | Determine whether gains are broad or concentrated in specific operational segments. |
| Feature set | No training change. Join reference and Stage 2 predictions to static warehouse and category metadata by validated keys. |
| Expected diagnostic value | Identifies warehouses or L1/L2 categories driving WMAE improvement, WAPE, or bias and exposes concentration in high-weight or high-volume segments. |
| Changes model training | No. |
| Changes scoring | Global frozen metrics remain unchanged. Segment metrics must call frozen metric functions on exact keyed subsets with official weights; no alternative formula. |
| Leakage risks | Labels are used only after prediction for diagnostics. Joins must be many-to-one and must not feed segment results back into model selection. |
| Runtime/memory risk | Low once reference and baseline prediction vectors are available. May require an approved consistency run because vectors were not persisted. |
| Interpretation | Broad improvement supports plausibility. Improvement isolated to a small number of segments or weights requires targeted explanation. Segment regressions remain visible and must not be hidden by the global average. |

## Priority 4: Reference stability checks

### A9. Exact Stage 2 F1 trailing 14-day reproduction

| Item | Plan |
|---|---|
| Purpose | Confirm the principal comparison baseline remains reproducible under the unchanged frozen grid. |
| Feature set | Existing Stage 2 trailing 14-day mean by ID with last-observed/global fallback; observed-row policy unchanged. |
| Expected diagnostic value | Rules out baseline drift or an accidental comparison against different rows, weights, or labels. |
| Changes model training | No LightGBM training. Run only the existing reviewed baseline path. |
| Changes scoring | No. Use frozen F1 scoring. |
| Leakage risks | Existing cutoff-safe baseline controls apply. Do not reimplement it in the ablation module. |
| Runtime/memory risk | Low; prior baseline runtime and memory were small. |
| Interpretation | WMAE must reproduce 31.1985803308, with WAPE 0.3021110128, bias 0.0303957661, 44,212 rows, and 94.026073% coverage within floating-point tolerance. Any mismatch stops all ablations until explained. |

### A10. ID/day-of-week median comparison

| Item | Plan |
|---|---|
| Purpose | Compare the Stage 3 result with the strongest alternative Stage 2 F1 reference. |
| Feature set | Existing Stage 2 median by `unique_id` and day of week with reviewed fallbacks. |
| Expected diagnostic value | Shows whether the conclusion depends on selecting only the trailing 14-day reference. |
| Changes model training | No. |
| Changes scoring | No. |
| Leakage risks | Existing cutoff-safe baseline controls apply. Reuse reviewed code. |
| Runtime/memory risk | Low. |
| Interpretation | Reproduce WMAE 30.7150203221, WAPE 0.3411060097, and bias -0.1929146153 on the exact F1 population. Stage 3 comparison should be reported against both baseline values without declaring either a Kaggle score. |

## Proposed execution order and human gates

1. Reproduce A9 and A10 first as reference-stability checks.
2. Run Priority 1 feature-family ablations A1–A4 with fixed settings and log every result.
3. Stop for human review of Priority 1 before running A5 or A6.
4. If approved, run Priority 2 diagnostics A5–A6 and stop for a second review.
5. Run A7–A8 only if the required reference prediction vectors can be obtained through an explicitly approved consistency run.
6. Perform a new adversarial review of all ablation code, matrices, and results.
7. Only a human decision after that review may authorize F2–F4.

The next approval should specify exactly which ablations are authorized. This plan alone does not authorize execution.
