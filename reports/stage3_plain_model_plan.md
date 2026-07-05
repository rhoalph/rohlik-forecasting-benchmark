# Stage 3 Plain Supervised Model Plan

Date: 2026-07-05
Status: Plan only; implementation requires human approval
Question: Can one simple, leakage-safe global model beat the Stage 2 trailing 14-day mean baseline?

## Scope and decision boundary

Stage 3 will test one global gradient-boosted tree model against the accepted Stage 2 reference baseline. It will not introduce ensembles, automated tuning, recursive prediction, public-solution code, or advanced feature engineering.

The first execution will stop after F1. F2–F4 will run sequentially only after the F1 correctness, leakage, resource, and score gates are reviewed.

This plan does not authorize a Kaggle submission. Local results remain diagnostics and must not be compared directly with Kaggle final/private leaderboard scores.

## Proposed model library

Use the native LightGBM training API.

Local compatibility check on 2026-07-05:

| Component | Result |
|---|---|
| Python | 3.14.4 |
| LightGBM | 4.6.0 installed and imports successfully |
| NumPy | 2.5.0 installed |
| pandas | 3.0.3 installed |
| XGBoost | Not installed |
| scikit-learn | Not installed |

LightGBM is therefore the lowest-risk local choice. The native `lightgbm.train` API avoids adding a scikit-learn dependency. No installation is needed for the first local run.

Proposed fixed first-run configuration:

- One global model across all IDs, warehouses, and horizons.
- Objective: `regression_l1`.
- Target: raw `sales`.
- Fixed 300 boosting rounds; no outer-fold early stopping.
- Learning rate: 0.05.
- `num_leaves`: 31.
- `min_data_in_leaf`: 100.
- Full row and feature fractions for the first run.
- Deterministic seeds: 42.
- Native categorical handling for declared categorical fields.
- Predictions remain continuous and are scored without clipping. Negative-prediction counts will be reported. Clipping may be tested only later as a separately approved and logged experiment.
- Official evaluation weights are not model features and will not be passed as training sample weights in the first run. They remain inside frozen scoring only.

No target transform will be used initially. `log1p(sales)` and `sqrt(sales)` may be considered only as separately documented later experiments after the raw-target result is reviewed.

## Direct multi-horizon training-origin design

### Meaning of twelve cutoff-safe origins

The first run evaluates outer fold F1, whose cutoff is 2024-05-19 and whose untouched validation window is 2024-05-20 through 2024-06-02. The proposed supervised training set uses twelve earlier forecast simulations. Each historical origin is 14 days before the preceding origin, and each supplies labels for the next 14 days.

For an origin `O`, one requested `(unique_id, target_date)` row becomes a training example only when:

1. it exists in the official ID/horizon mask shifted to `O`;
2. its target date is in `O+1` through `O+14`; and
3. `sales_train.csv` contains a non-null sales label for that exact key.

The sales value in the target window is used only as supervised `y`. It is never available to feature generation. Missing labels are excluded, not converted to zero.

The counts below were verified with the frozen grid and split logic. The 12 windows contain 492,826 available-label training examples from 564,252 requested rows.

| # | Historical origin/cutoff | Target prediction window | Rows becoming examples | Data allowed for feature creation | Data forbidden from features |
|---:|---|---|---:|---|---|
| 1 | 2024-05-05 | 2024-05-06–2024-05-19 | 43,433 | Sales history through 2024-05-05; static inventory; target-date calendar, price, and discounts | Sales after 2024-05-05; `total_orders`; target-date `availability`; weights; F1 labels |
| 2 | 2024-04-21 | 2024-04-22–2024-05-05 | 42,794 | Sales history through 2024-04-21; static inventory; target-date calendar, price, and discounts | Sales after 2024-04-21; `total_orders`; target-date `availability`; weights; F1 labels |
| 3 | 2024-04-07 | 2024-04-08–2024-04-21 | 42,035 | Sales history through 2024-04-07; static inventory; target-date calendar, price, and discounts | Sales after 2024-04-07; `total_orders`; target-date `availability`; weights; F1 labels |
| 4 | 2024-03-24 | 2024-03-25–2024-04-07 | 40,993 | Sales history through 2024-03-24; static inventory; target-date calendar, price, and discounts | Sales after 2024-03-24; `total_orders`; target-date `availability`; weights; F1 labels |
| 5 | 2024-03-10 | 2024-03-11–2024-03-24 | 41,138 | Sales history through 2024-03-10; static inventory; target-date calendar, price, and discounts | Sales after 2024-03-10; `total_orders`; target-date `availability`; weights; F1 labels |
| 6 | 2024-02-25 | 2024-02-26–2024-03-10 | 41,190 | Sales history through 2024-02-25; static inventory; target-date calendar, price, and discounts | Sales after 2024-02-25; `total_orders`; target-date `availability`; weights; F1 labels |
| 7 | 2024-02-11 | 2024-02-12–2024-02-25 | 41,077 | Sales history through 2024-02-11; static inventory; target-date calendar, price, and discounts | Sales after 2024-02-11; `total_orders`; target-date `availability`; weights; F1 labels |
| 8 | 2024-01-28 | 2024-01-29–2024-02-11 | 41,208 | Sales history through 2024-01-28; static inventory; target-date calendar, price, and discounts | Sales after 2024-01-28; `total_orders`; target-date `availability`; weights; F1 labels |
| 9 | 2024-01-14 | 2024-01-15–2024-01-28 | 40,749 | Sales history through 2024-01-14; static inventory; target-date calendar, price, and discounts | Sales after 2024-01-14; `total_orders`; target-date `availability`; weights; F1 labels |
| 10 | 2023-12-31 | 2024-01-01–2024-01-14 | 38,945 | Sales history through 2023-12-31; static inventory; target-date calendar, price, and discounts | Sales after 2023-12-31; `total_orders`; target-date `availability`; weights; F1 labels |
| 11 | 2023-12-17 | 2023-12-18–2023-12-31 | 38,434 | Sales history through 2023-12-17; static inventory; target-date calendar, price, and discounts | Sales after 2023-12-17; `total_orders`; target-date `availability`; weights; F1 labels |
| 12 | 2023-12-03 | 2023-12-04–2023-12-17 | 40,830 | Sales history through 2023-12-03; static inventory; target-date calendar, price, and discounts | Sales after 2023-12-03; `total_orders`; target-date `availability`; weights; F1 labels |

“Target-date price and discounts” means only the benchmark-known-future covariate columns, joined separately from the isolated `sales` label. It does not authorize any other target-window sales field.

### Difference from F1–F4 validation folds

The twelve rows above are inner historical training origins for the F1 model. F1 itself remains an outer, untouched evaluation window: 2024-05-20 through 2024-06-02. No F1 sales label is present in training, feature computation, category construction, model fitting, or parameter selection.

The first three F1 training origins have the same dates as the published F2, F3, and F4 folds. Their labels may train the F1 model because all three windows end on or before the F1 cutoff. They must not train their own outer-fold models. If F2 is later evaluated, its twelve training origins are regenerated relative to the F2 cutoff, beginning at 2024-04-21; the F2 validation window is then excluded. F3 and F4 follow the same rebasing rule. A fitted model or training matrix is never reused across outer folds.

Historical target windows are adjacent but do not overlap. For every training batch, target-derived feature history ends at its origin, while its labels occupy only the following 14 days. The latest F1 training label is dated 2024-05-19, one day before F1 validation begins. These boundaries prevent validation-period and future target information from entering training.

The official test mask may define which IDs and horizon days are simulated, as it already does in frozen evaluation. `sales_test.csv` covariate values are not used in local feature matrices; local known-future price and discount values come from the historical target rows in `sales_train.csv` under the approved benchmark policy.

## Feature contract

The following table is the complete proposed F1 model matrix. “Target-derived” means the feature depends on historical `sales` values or on whether a historical sales observation exists. All such features are recomputed independently for each origin from rows dated on or before that origin.

| Feature name | Source table | Availability class | Cutoff logic | Target-derived | Leakage risk | Included in F1 |
|---|---|---|---|---|---|---|
| `unique_id` | Shifted official grid; validated against `inventory.csv` | Static metadata | Requested ID only; no target aggregation | No | Low | Yes, categorical |
| `product_unique_id` | `inventory.csv` | Static metadata | Many-to-one join by `unique_id`; snapshot value | No | Low | Yes, categorical |
| `warehouse` | `inventory.csv` | Static metadata | Many-to-one join by `unique_id`; must agree with sales key | No | Low | Yes, categorical |
| `L1_category_name_en` | `inventory.csv` | Static metadata | Many-to-one join by `unique_id`; snapshot value | No | Low | Yes, categorical |
| `L2_category_name_en` | `inventory.csv` | Static metadata | Many-to-one join by `unique_id`; snapshot value | No | Low | Yes, categorical |
| `L3_category_name_en` | `inventory.csv` | Static metadata | Many-to-one join by `unique_id`; snapshot value | No | Low | Yes, categorical |
| `L4_category_name_en` | `inventory.csv` | Static metadata | Many-to-one join by `unique_id`; snapshot value | No | Low | Yes, categorical |
| `horizon_day` | Shifted official grid | Known future | Integer difference between target date and origin; 1–14 | No | Low | Yes |
| `day_of_week` | Target date | Known future | Derived only from requested target date | No | Low | Yes, categorical |
| `iso_week` | Target date | Known future | Derived only from requested target date | No | Low | Yes |
| `month` | Target date | Known future | Derived only from requested target date | No | Low | Yes, categorical |
| `weekend_flag` | Target date | Known future | Derived only from requested target date | No | Low | Yes |
| `holiday` | `calendar.csv` | Known future | Join target date and warehouse, validated many-to-one | No | Low | Yes |
| `shops_closed` | `calendar.csv` | Known future | Join target date and warehouse, validated many-to-one | No | Low | Yes |
| `winter_school_holidays` | `calendar.csv` | Known future | Join target date and warehouse, validated many-to-one | No | Low | Yes |
| `school_holidays` | `calendar.csv` | Known future | Join target date and warehouse, validated many-to-one | No | Low | Yes |
| `sell_price_main` | Historical target row in `sales_train.csv`; eventually official `sales_test.csv` | Known future | Exact ID/date value for target window under Kaggle policy; never used to derive sales | No | Medium | Yes, raw numeric |
| `type_0_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `type_1_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `type_2_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `type_3_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `type_4_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `type_5_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `type_6_discount` | Same as price | Known future | Exact ID/date target-window value under Kaggle policy; unchanged | No | Medium | Yes, raw numeric |
| `last_observed_sales` | `sales_train.csv` | Historical only | Last non-null observed ID row with date `<= origin`; global origin median fallback | Yes | Medium | Yes |
| `last_observed_available` | `sales_train.csv` | Historical only | One only when the ID has a non-null observation through origin | Yes | Medium | Yes |
| `trailing_7_mean` | `sales_train.csv` | Historical only | Mean of non-null observed ID rows from origin-6 through origin | Yes | Medium | Yes |
| `trailing_7_available` | `sales_train.csv` | Historical only | One only when that 7-day window contains an observed non-null ID row | Yes | Medium | Yes |
| `trailing_14_mean` | `sales_train.csv` | Historical only | Mean of non-null observed ID rows from origin-13 through origin | Yes | Medium | Yes |
| `trailing_14_available` | `sales_train.csv` | Historical only | One only when that 14-day window contains an observed non-null ID row | Yes | Medium | Yes |
| `same_weekday_sales` | `sales_train.csv` | Historical only | Literal target-date minus 7 only if source `<= origin`; otherwise last/global fallback | Yes | Medium | Yes |
| `same_weekday_direct_available` | `sales_train.csv` | Historical only | One only when the literal source date is cutoff-safe and observed | Yes | Medium | Yes |
| `historical_mean_sales` | `sales_train.csv` | Historical only | ID mean over all non-null observed rows through origin | Yes | Medium | Yes |
| `historical_median_sales` | `sales_train.csv` | Historical only | ID median over all non-null observed rows through origin | Yes | Medium | Yes |
| `historical_stats_available` | `sales_train.csv` | Historical only | One only when the ID has non-null history through origin | Yes | Medium | Yes |
| `observed_history_row_count` | `sales_train.csv` | Historical only | Count of non-null observed ID rows through origin | Yes | Medium | Yes |

For missing historical statistics, fallback order is last observed and then the global median calculated separately from non-null sales history through that origin. The corresponding availability flags remain zero. Absent item-date rows remain absent and are not converted to zero.

Category vocabularies come from static `inventory.csv` metadata or the historical training portion only. They are not learned from validation labels or concatenated train/validation data. No category, warehouse, or product target encoding is proposed.

### Explicit exclusions

| Field or method | Source | Availability class | Cutoff/use rule | Target-derived | Leakage risk | Included in F1 |
|---|---|---|---|---|---|---|
| Future `sales` | `sales_train.csv` | Forbidden / excluded | Post-origin sales may be isolated `y` only; never a feature | Yes | High | No |
| `total_orders` | Sales files | Forbidden / excluded | Excluded from all future feature frames, even though present in test | No | High | No |
| Target-date `availability` | `sales_train.csv` | Forbidden / excluded | Not known at origin and absent from official test | No | High | No |
| `weight` | `test_weights.csv` | Forbidden / excluded | Frozen scoring only; not a feature or first-run training weight | No | Medium | No |
| `name` | `inventory.csv` | Static metadata | Omitted as redundant high-cardinality identity | No | Low | No |
| `holiday_name` | `calendar.csv` | Known future | Deferred to keep the first categorical contract minimal | No | Low | No |
| `id`, `sales_hat` | `solution.csv` | Forbidden / excluded | Submission-only fields; no local model input | No | High | No |
| `sales_test.csv` covariate values | `sales_test.csv` | Forbidden / excluded for local folds | Official mask may define rows; its covariate values cannot enter local F1–F4 matrices | No | Medium | No |
| Train+validation fitted transforms | Any | Forbidden / excluded | No encoder, imputer, scaler, or stopping rule may fit on outer validation | Potentially | High | No |
| Recursive prediction | Derived | Forbidden / excluded | No predicted value may become another horizon's input | Yes | High | No |
| Ensembles, Optuna, public solution code | External/multiple | Forbidden / excluded | Outside Stage 3 plain-model scope | Varies | Medium | No |

## Price and discount policy

`sell_price_main` and `type_0_discount` through `type_6_discount` are Kaggle-known-future benchmark covariates. They are allowed because the official competition test set supplies them for every requested test row. Their inclusion is a competition-specific assumption; it does not prove that equivalent future prices or promotion decisions would be available at an operational forecast origin outside Kaggle.

For each local historical origin, price and discount values are taken from the exact target-window ID/date row in `sales_train.csv` and joined into a covariate-only frame. The `sales` label is selected into a separate frame. The join must retain only the approved price/discount columns and must assert that `total_orders`, `sales`, and `availability` are absent from the resulting feature matrix.

No price history aggregate, discount history aggregate, implicit sales ratio, or price-derived target encoding is included. Raw negative discount values remain unchanged in the first run. Price and discounts must never be combined with future sales or future `total_orders` to derive another feature.

## Target and objective contract

The first run predicts raw `sales`. This keeps the learned target on the same scale as WMAE, WAPE, bias, and the Stage 2 baselines and avoids introducing a back-transform that can alter error asymmetry or bias.

LightGBM's L1 regression objective is proposed because the official benchmark metric is weighted mean absolute error. The model will train with unweighted L1 loss in the first run because official weights remain evaluation-only under the frozen contract. Frozen WMAE, not LightGBM's internal metric, determines the result.

`log1p` and square-root targets are excluded from the first run. Either changes the optimization scale, requires an inverse transform, and can systematically under-forecast high-volume items. They may be tested later only as separate logged experiments after the raw-target result establishes a clean reference.

No clipping or rounding is permitted in the first run. Raw continuous predictions will be passed to frozen scoring, and the count and range of negative predictions will be reported. Zero clipping may be considered later only as an explicit, separately scored, logged, and approved post-processing experiment.

## Validation and comparison contract

Use the frozen F1–F4 folds and exact available-label mask:

| Fold | Cutoff | Validation window | Scored rows | Stage 2 reference WMAE |
|---|---|---|---:|---:|
| F1 | 2024-05-19 | 2024-05-20 through 2024-06-02 | 44,212 | 31.1985803308 |
| F2 | 2024-05-05 | 2024-05-06 through 2024-05-19 | 43,433 | 32.9412795963 |
| F3 | 2024-04-21 | 2024-04-22 through 2024-05-05 | 42,794 | 30.1713535275 |
| F4 | 2024-04-07 | 2024-04-08 through 2024-04-21 | 42,035 | 27.9464531827 |

For each fold report:

- WMAE using frozen official-weight alignment.
- WAPE.
- Bias with business-language direction.
- Scored rows and coverage.
- Feature-generation, fitting, prediction, and total runtime.
- Peak process memory.
- Negative raw-prediction count and range; no clipped score in the first run.
- Absolute WMAE improvement: `baseline WMAE - model WMAE`.
- Relative WMAE improvement: `(baseline WMAE - model WMAE) / baseline WMAE`.

The accepted Stage 2 macro reference is:

- Mean WMAE: 30.5644166593.
- Mean WAPE: 0.2974236747.
- Mean bias: -0.0077277837.

Macro results are equal-weight arithmetic means of fold metrics, not pooled-row scores.

## Exact success criteria

### F1 implementation gate

F1 is technically clean only if all of the following hold:

1. Unit tests and full raw-data validation pass.
2. `eval/` and `dataguard/` remain unchanged.
3. F1 scores exactly 44,212 rows at 94.026073% coverage.
4. Prediction keys exactly match the frozen F1 label keys.
5. Every target-derived feature records a maximum source date on or before its training origin or F1 cutoff.
6. No forbidden field enters the model matrix.
7. No non-finite feature, label, or prediction reaches scoring.
8. Peak RSS stays below 12 GiB and F1 completes within 30 minutes on the hack box.

The F1 result is successful only if all of these predeclared score gates pass:

| Gate | Stage 2 F1 reference | Required Stage 3 F1 result |
|---|---:|---:|
| WMAE | 31.1985803308 | Strictly below 31.1985803308 |
| WAPE | 0.3021110128 | At most 0.3081532331, limiting relative worsening to 2% |
| Bias | +0.0303957661 | Absolute bias at most 0.10; neither over- nor under-forecast by more than 10% in aggregate |
| Runtime | Stage 2 is not directly comparable because it has no model fit | At most 30 minutes from raw load through F1 scoring |
| Peak RSS | Hack-box constraint | Below 12 GiB with no sustained swap use |

A material F1 WMAE improvement is at least 1%, meaning WMAE at or below 30.8865945275. This materiality label is reported separately; the strict pass/fail requirement remains that WMAE beats 31.1985803308 while all other gates pass.

The implementation turn must stop after reporting F1. Even if every gate passes, F2–F4 require explicit human approval. A failed gate triggers revision rather than more folds.

### All-fold Stage 3 success

The plain model answers the Stage 3 question positively if:

1. Mean WMAE across F1–F4 is strictly below 30.5644166593.
2. Relative mean-WMAE improvement is at least 1% for a material result, corresponding to mean WMAE at or below 30.2587724927.
3. At least three of four folds do not regress on WMAE.
4. Fold coverage and scored rows exactly match Stage 2.
5. Leakage and correctness review finds no high- or medium-risk issue.

WAPE and bias remain diagnostics, not substitutes for WMAE, but the F1 guardrails prevent a nominal WMAE gain from hiding materially worse volume error or a large directional error.

## Leakage controls and required assertions

- Use frozen `dataguard` cutoff and availability checks.
- Keep outer validation labels physically separate from feature generation.
- Assert `max(history.date) <= origin` for every historical-origin feature batch.
- Assert same-weekday source dates never exceed the corresponding origin.
- Assert every historical training target date is at most the outer-fold cutoff.
- Assert training and outer-validation keys are disjoint.
- Assert target, weight, `total_orders`, and `availability` are absent from feature columns.
- Assert calendar joins are unique by warehouse/date.
- Assert inventory joins are unique by ID.
- Fit no learned encoder, imputer, normalizer, early-stopping rule, or other transformation on outer-validation rows.
- Build categorical domains from declared static metadata or training data only.
- Use frozen exact-key scoring so missing, extra, or duplicate predictions fail closed.

An adversarial leakage review is required after F1 and again after the all-fold run.

### Using frozen `dataguard` without protected changes

The frozen availability registry classifies raw fields, not every future derived feature name. Stage 3 must use it at the raw-data boundary:

1. Pass raw static and known-future source columns through frozen availability selection before any feature derivation.
2. Filter historical target data with frozen cutoff helpers before calculating target-derived statistics.
3. Create date-derived and target-derived columns only in a non-protected Stage 3 feature module.
4. Define a non-protected final-matrix allowlist that exactly matches the included rows in the feature contract above.
5. Assert final feature columns equal that allowlist and have no intersection with `sales`, `weight`, `total_orders`, `availability`, `id`, or `sales_hat`.
6. Test source-date lineage and origin isolation for every target-derived feature family.

This preserves the frozen raw-field and cutoff controls without adding derived feature names to `dataguard/`. If implementation reveals that a protected change is actually necessary, Stage 3 must stop and request a new human approval rather than using the override.

## Dependency note and reproducibility risks

1. LightGBM 4.6.0 imports successfully on local Python 3.14.4, but it is not currently declared in `requirements.txt`.
2. Kaggle notebook package and Python versions may differ from the hack box; final notebook compatibility must be tested separately.
3. LightGBM must be added to `requirements.txt` only after Stage 3 implementation is explicitly approved.
4. XGBoost and scikit-learn are not installed locally and must not be used in Stage 3 unless LightGBM fails and a replacement is separately approved.
5. LightGBM categorical behavior depends on stable category coding. Category mappings must be deterministic and shared without fitting on validation labels.
6. Fixed thread count and seeds improve reproducibility, but small floating-point differences may remain across LightGBM builds and hardware.

After approval, add a reviewed LightGBM dependency declaration appropriate for both local and Kaggle execution. Do not install fallback libraries preemptively.

## Runtime and memory risks

The hack box currently has approximately 14 GiB RAM, 10 GiB available, 4 GiB swap, and 269 GiB free disk. CPU remains the likely constraint.

Primary risks:

- Recomputing origin-safe aggregates 12 times per fold.
- Materializing unnecessary copies of the 4-million-row history.
- Expanding categorical/string columns before downcasting.
- Retaining multiple fold matrices simultaneously.
- LightGBM histogram memory growth from high-cardinality IDs.

Controls:

- Run one outer fold at a time, beginning with F1.
- Run historical origins sequentially and concatenate only compact typed training batches.
- Use categorical/integer codes and `float32` features while retaining scoring precision in `float64`.
- Avoid dense item-date expansion outside the shifted mask.
- Release raw intermediates before model fitting where practical.
- Record peak RSS and stop before sustained swap pressure.
- Do not write processed caches in the first run unless separately approved.

## Stop or revise conditions

Stop immediately and do not run F2–F4 if:

- any cutoff, source-date, key-alignment, or forbidden-column assertion fails;
- `eval/` or `dataguard/` changes;
- coverage differs from the frozen fold contract;
- outer-validation information affects a fitted transform or training decision;
- LightGBM import or native training is unstable on Python 3.14;
- F1 exceeds 12 GiB peak RSS, uses sustained swap, or exceeds 30 minutes;
- predictions or metrics are non-finite;
- F1 WMAE is greater than or equal to 31.1985803308;
- F1 WAPE exceeds 0.3081532331 or absolute bias exceeds 0.10;
- relative F1 WMAE improvement exceeds 20% without a documented, mechanically verified reason, triggering the suspicious-score rule;
- feature generation is too slow or memory-heavy to remain within the declared F1 limits;
- any test fails;
- proceeding would require a change to frozen `eval/` or `dataguard/`.

Revise the plan rather than adding complexity when a stop condition occurs. The first revision should inspect data assembly, target lineage, category encoding, and fixed model settings. It should not jump to ensembles, tuning, or public solution code.

## Fallback-library decision

If native LightGBM becomes unusable, evaluate fallbacks in this order:

1. XGBoost, only after checking Python 3.14 wheel availability and memory behavior.
2. scikit-learn `HistGradientBoostingRegressor`, only after installing a compatible scikit-learn build and defining deterministic categorical encoding.

Neither fallback is currently installed, so switching libraries requires human approval. No fallback should be installed during planning.

## Recommended F1-only implementation path

After explicit approval:

1. Add the approved LightGBM dependency declaration; do not install XGBoost or scikit-learn.
2. Add non-protected Stage 3 feature-contract and historical-origin modules with toy-data tests before any full-data fit.
3. Test each join cardinality, cutoff boundary, missing-target rule, fallback, and final feature allowlist.
4. Build the twelve F1 historical-origin batches sequentially, recording origin, label window, example count, maximum feature source date, runtime, and peak memory.
5. Concatenate only the compact typed training batches and fit the single fixed LightGBM model once.
6. Build the F1 matrix from history through 2024-05-19 and the isolated 2024-05-20 through 2024-06-02 request rows. Do not expose F1 labels until frozen scoring.
7. Score raw continuous predictions with frozen evaluation and compare them with the exact F1 Stage 2 reference gates.
8. Run the full tests and raw-data validation, confirm protected directories are unchanged, write the F1 result and leakage review, then stop.

No F2–F4 execution, model revision, clipping experiment, target transform, or Kaggle submission is part of this first implementation turn.

## Approval requested before implementation

Human approval is required for:

1. LightGBM 4.6.0 as the first model library.
2. The 12 biweekly historical-origin training design.
3. The exact first-run feature set, including benchmark-known-future price and discounts.
4. Raw sales with unweighted L1 training and unmodified continuous predictions.
5. The F1 performance/resource gates and the rule controlling whether F2–F4 run.
6. Adding LightGBM to project dependency declarations during implementation.

No Stage 3 model or feature implementation should begin until these decisions are approved.
