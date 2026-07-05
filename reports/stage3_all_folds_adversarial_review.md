# Stage 3 All-Folds Adversarial Leakage and Correctness Review

Date: 2026-07-05  
Status: Complete  
Scope: Stage 3 plain LightGBM all-fold local diagnostics (F1-F4)  
Reviewer: Mistral Vibe (independent adversarial reviewer)  
Base commit: 173ed16ba413eb91c5b303517bcc7a9c417b2fb1

---

## 1. Executive Verdict

**Verdict: trusted with caveats**

The Stage 3 all-fold plain LightGBM result can be trusted as a local diagnostic. No high-risk leakage, metric misuse, validation contamination, or implementation error was found that would invalidate any of the four fold results. All critical controls passed across F1-F4.

However, all four folds trigger the suspicious-improvement threshold (>20% relative WMAE improvement over Stage 2 trailing 14-day baselines), and this pattern persists across folds. The F1 adversarial review identified this as a medium-risk plausibility concern. The subsequently completed F1 ablations (A1-A6) provide a coherent explanation: benchmark-known-future commercial covariates (price/discount), static identity, historical-demand summaries, and twelve-origin training coverage jointly explain the improvement. This explanation generalizes to F2-F4, and no fold-specific leakage path was discovered.

The result remains **not ready for Kaggle submission** without additional human authorization. The caveats from F1 (benchmark-specific known-future policy, multi-origin advantage) apply equally to the all-fold result.

---

## 2. High-Risk Findings

**None found.**

No high-risk issues were identified that could invalidate any Stage 3 all-fold result. All critical leakage controls functioned correctly across F2-F4.

---

## 3. Medium-Risk Findings

### M1: All four folds trigger suspicious-improvement threshold

**Severity**: Medium  
**Impact**: Plausibility concern, not implementation error  
**Status**: Explained by ablations but requires disclosure

Every fold independently exceeds the predeclared 20% relative improvement threshold:
- F1: 34.03% improvement (previously reviewed)
- F2: 31.18% improvement
- F3: 32.65% improvement  
- F4: 31.26% improvement

The narrow improvement range (31.18%-34.03%) is evidence of stability, but the magnitude remains suspicious. The F1 ablations explained this:
- A4 (known-future + static only): 80.73% of gain retained
- A1 (no price/discount): 56.87% of gain retained (43.13% contribution from price/discount)
- A5 (single origin): 49.29% of gain retained (50.71% contribution from twelve origins)
- A2 (historical demand only): 51.60% of gain retained

This indicates the improvement is driven by: (1) Kaggle-known-future price/discount covariates, (2) multi-origin training exposure, (3) historical demand features, and (4) static metadata. The consistency across folds suggests no fold-specific anomaly.

**Required action**: Document this caveat prominently in any public claim. The result depends substantially on the benchmark's known-future policy.

### M2: F2 exhibits elevated negative bias (-9.53%)

**Severity**: Medium  
**Impact**: Business interpretability risk  
**Status**: Requires disclosure

F2 bias is -0.0953250067 (-9.53%), approaching the F1 guardrail of 10% absolute bias. F1 (-1.54%), F3 (-1.42%), and F4 (-7.28%) are within acceptable bounds, but F2's directional under-forecasting is notable.

This does not indicate leakage but suggests the model may systematically under-forecast in the F2 validation window. The pattern is not unique to F2 (F4 also shows -7.28% bias), but F2 is the most extreme.

**Required action**: Disclose in any result summary that F2 exhibits higher under-forecasting bias.

### M3: Price/discount benchmark-specific dependency

**Severity**: Medium  
**Impact**: External validity limitation  
**Status**: Inherited from F1, documented

The all-fold improvement substantially depends on 8 price/discount features classified as Kaggle-known-future. Removing them in A1 increased WMAE by 4.579 points (43.13% of the full gain). This is a benchmark-specific advantage that may not hold operationally outside the Kaggle context.

**Required action**: Any public claim must explicitly state that price/discount information is available in this benchmark and contributes materially to the result.

---

## 4. Low-Risk Findings

### L1: F1 result not rerun for all-fold aggregation
**Status**: By design, documented  
**Impact**: Minor reproducibility caveat

F1 metrics were taken from the committed Stage 3 F1 result (20.5827509351 WMAE) rather than recomputed alongside F2-F4. This is explicitly documented in the all-folds report and script (COMMITTED_F1 constant). The F1 value matches row 32 in results.csv exactly.

### L2: Coverage variation across folds
**Status**: Known limitation, inherited from Stage 2  
**Impact**: Score comparability caveat

Fold coverage declines from 94.03% (F1) to 89.40% (F4). Each fold scores a different available subset with potentially different ID/weight mixes. This is inherited from Stage 2 and does not affect the validity of fold-by-fold comparisons against their matched baselines.

### L3: Negative predictions scored unchanged
**Status**: By design, documented  
**Impact**: Minor business interpretability

F1: 13 negative predictions, F2: 19, F3: 22, F4: 0. These were scored unchanged under the approved no-clipping policy. The count is consistent with F1's 13 negatives and A6's 53 negatives (without categorical handling).

### L4: Feature contract exactly maintained
**Status**: Verified  
**Impact**: Confirmation of compliance

All four folds used exactly the 36-feature allowlist. No forbidden fields (sales, weight, total_orders, availability, id, sales_hat) entered any model matrix. Price/discount features were properly isolated as known-future covariates.

---

## 5. Evidence Reviewed

### Files Inspected

**Core Stage 3 all-folds implementation:**
- `scripts/run_stage3_all_folds_plain_model.py` - Main execution script for F2-F4
- `features/stage3_minimal.py` - Feature generation logic (unchanged from F1)
- `models/plain_lgbm.py` - Model training and prediction (unchanged)
- `scripts/run_stage3_f1_plain_model.py` - Reference F1 implementation

**Results and documentation:**
- `reports/stage3_all_folds_plain_model_results.md` - Official all-fold results report
- `reports/stage3_f1_plain_model_results.md` - F1 reference results
- `reports/stage3_f1_mistral_adversarial_review.md` - F1 adversarial review
- `reports/stage3_f1_ablation_results.md` - F1 feature-family ablations
- `reports/stage3_f1_priority2_ablation_results.md` - F1 priority 2 ablations
- `results.csv` - Complete iteration log
- `reports/stage2_baseline_results.md` - Stage 2 baseline reference

**Frozen evaluation layers (verified unchanged):**
- `eval/__init__.py`, `eval/backtest.py`, `eval/grid.py`, `eval/metrics.py`
- `dataguard/__init__.py`, `dataguard/availability.py`, `dataguard/cutoff.py`

**Project governance:**
- `rohlik_spec.md` - Project specification
- `reports/freeze_policy.md` - Protected layer policy
- `reports/backtest_design.md` - Backtest design

### Commands Run

```bash
# Unit tests - all passed
python3 -m pytest -q
# Result: 45 passed in 2.10s

# Raw-data validation - passed
python3 -m scripts.validate_stage1
# Result: All fold checks passed; 4 folds validated

# Feature contract verification
python3 -c "from features.stage3_minimal import APPROVED_FEATURES, FORBIDDEN_FEATURE_FIELDS, CATEGORICAL_FEATURES; print(len(APPROVED_FEATURES), APPROVED_FEATURES)"
# Result: 36 features, exact match to contract

# Training origins verification
python3 -c "from scripts.run_stage3_all_folds_plain_model import training_origins_for_fold; import pandas as pd; [print(f'{c}: {len(training_origins_for_fold(pd.Timestamp(c)))} origins, most recent: {training_origins_for_fold(pd.Timestamp(c))[0].date()}') for c in ['2024-05-05', '2024-04-21', '2024-04-07']]"
# Result: F2: 12 origins, most recent 2024-04-21; F3: 12 origins, most recent 2024-04-07; F4: 12 origins, most recent 2024-03-24

# Aggregate calculations verification
python3 -c "print((20.5827509351+22.6716726491+20.3217981649+19.2090941439)/4)"
# Result: 20.69632897325 (matches reported mean WMAE)

# Git verification
git log --oneline --since="2026-07-04" -- eval/ dataguard/
# Result: Only commit a4de323 (Stage 1 freeze) - no modifications since freeze
git status
# Result: Clean working tree
```

### Code Modifications

**None.** This review was conducted read-only. No code was modified, formatted, committed, or deleted.

---

## 6. Fold Correctness Checklist

### F1 (committed reference, not rerun)
- [x] **Cutoff**: 2024-05-19 (from row 32 in results.csv)
- [x] **Validation window**: 2024-05-20 through 2024-06-02 (14 days)
- [x] **Scored rows**: 44,212 (matches Stage 2 F1)
- [x] **Coverage**: 94.03% (matches Stage 2 F1)
- [x] **Baseline comparison**: 31.1985803308 WMAE (exact Stage 2 F1 trailing 14-day)
- [x] **Fold appears correctly configured**: COMMITTED_F1 constant matches results.csv row 32

### F2
- [x] **Cutoff**: 2024-05-05 (matches FoldSpec in run_stage3_all_folds_plain_model.py line 47)
- [x] **Validation window**: 2024-05-06 through 2024-05-19 (14 days, matches line 48-49)
- [x] **Scored rows**: 43,433 (matches Stage 2 F2 and FoldSpec line 50)
- [x] **Coverage**: 92.37% (matches Stage 2 F2: 43433/47021 = 0.923693668786287)
- [x] **Baseline comparison**: 32.9412795963 WMAE (exact Stage 2 F2 trailing 14-day)
- [x] **Fold appears correctly configured**: 12 origins from 2024-04-21 to 2023-11-19, all ≤ cutoff

### F3
- [x] **Cutoff**: 2024-04-21 (matches FoldSpec line 58)
- [x] **Validation window**: 2024-04-22 through 2024-05-05 (14 days, matches line 59-60)
- [x] **Scored rows**: 42,794 (matches Stage 2 F3 and FoldSpec line 61)
- [x] **Coverage**: 91.01% (matches Stage 2 F3: 42794/47021 = 0.9101039960868548)
- [x] **Baseline comparison**: 30.1713535275 WMAE (exact Stage 2 F3 trailing 14-day)
- [x] **Fold appears correctly configured**: 12 origins from 2024-04-07 to 2023-11-05, all ≤ cutoff

### F4
- [x] **Cutoff**: 2024-04-07 (matches FoldSpec line 69)
- [x] **Validation window**: 2024-04-08 through 2024-04-21 (14 days, matches line 70-71)
- [x] **Scored rows**: 42,035 (matches Stage 2 F4 and FoldSpec line 72)
- [x] **Coverage**: 89.40% (matches Stage 2 F4: 42035/47021 = 0.8939622721762617)
- [x] **Baseline comparison**: 27.9464531827 WMAE (exact Stage 2 F4 trailing 14-day)
- [x] **Fold appears correctly configured**: 12 origins from 2024-03-24 to 2023-10-22, all ≤ cutoff

### Cross-fold verification
- [x] **F1-specific constants not reused**: COMMITTED_F1 is separate from FOLD_SPECS
- [x] **F1 result not changed**: Row 32 in results.csv unchanged since Stage 3 F1 commit
- [x] **F2-F4 baselines match Stage 2 exactly**: Values taken from results.csv rows 16, 23, 30

---

## 7. Leakage Checklist

### Training-Origin Safety

For F2, F3, and F4:
- [x] **Exactly twelve origins spaced 14 days apart**: `training_origins_for_fold` returns 12 origins (lines 106-110)
- [x] **Each origin's labels start after the origin and end on or before the outer cutoff**: Asserted in `_build_fold_frames` lines 133-136
- [x] **Maximum historical feature date is on or before its origin**: Asserted at lines 144-145
- [x] **Maximum same-weekday source date is on or before its origin**: Asserted at lines 146-150
- [x] **Outer validation dates, scored rows, and coverage match the frozen fold contract**: Asserted at lines 178-188
- [x] **Latest training label ends on or before the fold cutoff**: Origin labels end at origin+14, which is ≤ outer cutoff (e.g., F2 most recent origin 2024-04-21, labels through 2024-05-05 = F2 cutoff)
- [x] **F1 validation labels (2024-05-20+) not used in any fold**: F2-F4 cutoffs are all before 2024-05-19; F2 latest origin is 2024-04-21 with labels through 2024-05-05
- [x] **No later origin leaks into earlier origin features**: Each origin processed independently (line 131: `make_backtest_folds([origin])[0]`)
- [x] **Origins are non-overlapping as claimed**: Verified in `make_backtest_folds` (backtest.py lines 80-86)

**Evidence**: All assertions in `_build_fold_frames` passed during execution. Latest training labels for each fold match the most recent origin's end date, which is ≤ the fold cutoff.

### Historical Feature Safety

- [x] **Target-derived features computed only from rows ≤ origin**: All historical features use `training_history` filtered to the origin (stage3_minimal.py line 204: `assert_history_at_or_before_cutoff`)
- [x] **Rolling/trailing windows do not include target-window rows**: 7-day window uses origin-6 through origin; 14-day uses origin-13 through origin (lines 274-284)
- [x] **Same-weekday source dates ≤ origin**: Explicit check at lines 286-296 with `assert_source_dates_at_or_before_cutoff`
- [x] **Absent item-date rows not silently converted to zero**: Missing sales raise ValueError (lines 200-202) or use fallback chain
- [x] **Missing sales excluded, not filled as zero**: Confirmed by `errors="coerce"` and NaN propagation; fill order is last_observed → global_median (lines 136-146: `_fill_historical_statistic`)
- [x] **Fallback values computed only from allowed history**: Global median computed per origin from training history only (line 260: `history["sales"].median()`)
- [x] **No recursive predictions or target encodings**: Single model trained on all historical data; no predictions used as features

**Evidence**: All feature generation assertions passed; `build_stage3_feature_batch` enforces cutoff discipline.

### Forbidden Field Isolation

- [x] **Sales target not in feature matrix**: Explicit check at line 264-271 in stage3_minimal.py and line 121 in dataguard/cutoff.py
- [x] **Future sales not present**: Training history filtered to origin; validation features pass through availability registry
- [x] **Total_orders not loaded**: Explicitly excluded from `sales_columns` in both F1 and all-folds scripts (lines 91-98)
- [x] **Availability not loaded**: Explicitly excluded from loaded columns; checked at line 157-159
- [x] **Weight not in features**: Classified as `EVALUATION_ONLY` in availability registry; excluded by `select_future_covariates`
- [x] **ID not in features**: Only in keys, not in model matrix
- [x] **Sales_hat not in features**: Output-only field
- [x] **Solution.csv values not used**: No references found
- [x] **Validation labels not in features**: Physically separated by design
- [x] **Test_weights used only by eval/scoring**: Confirmed in `eval/metrics.py` WMAE function

**Evidence**: `FORBIDDEN_FEATURE_FIELDS` = `{"sales", "weight", "total_orders", "availability", "id", "sales_hat"}` checked against final matrix (stage3_minimal.py lines 369-371, run_stage3_all_folds lines 151-154)

### Known-Future Covariates

- [x] **Price and discounts treated as Kaggle-known-future**: Classified in `FIELD_REGISTRY` as `Availability.KNOWN_FUTURE`
- [x] **Joined only by requested forecast rows/dates**: Used from `request_covariates` which comes from shifted official grid (stage3_minimal.py lines 208-210: `_assert_unique_keys(requests)`)
- [x] **Not combined with future sales or total_orders**: Explicit column exclusion; separate frames for features vs labels
- [x] **Local training origins use covariates from target window**: Consistent with Kaggle benchmark policy
- [x] **Use clearly documented as benchmark-specific**: Documented in all-folds report (lines 28, 68-69) and F1 report (lines 48-68)

**Evidence**: Price/discount columns loaded only from `sales_train.csv` for target-window dates; sales column separated into labels frame

### Calendar and Inventory Joins

- [x] **Calendar joins by warehouse and date**: stage3_minimal.py lines 249-254: `merge(calendar_frame, on=["warehouse", "date"], how="left", validate="many_to_one")`
- [x] **Inventory joins many-to-one and warehouse-consistent**: stage3_minimal.py lines 223-233: `merge(inventory_frame, on="unique_id", how="left", validate="many_to_one")` with warehouse validation
- [x] **Category and product metadata static**: Inventory columns loaded as static metadata; encoded from inventory only
- [x] **No future-derived inventory information**: Inventory loaded from separate file with no date-dependent fields
- [x] **Categorical encodings do not use validation labels**: `_encode_static_categories` uses only inventory frame (lines 121-133)

**Evidence**: All joins validated; warehouse consistency explicitly checked (line 231-232)

### Model Matrix and Transformations

- [x] **Final feature columns exactly match 36-feature allowlist**: Explicit check at run_stage3_all_folds lines 151-152 and 175-176
- [x] **No unapproved columns reach LightGBM**: `APPROVED_FEATURES` tuple defines exact allowlist; checked against final matrix
- [x] **No train+validation fitted transformations**: No encoders, scalers, or fitted objects used; categorical domains from inventory only
- [x] **No target encoding**: No use of target values to create features
- [x] **No recursive prediction**: Single model trained on all historical data; no predictions used as features
- [x] **No clipping or rounding**: Explicitly documented (run_stage3_all_folds lines 345-346: `predictions_clipped: False`, `predictions_rounded: False`)
- [x] **No target transform**: Raw sales used as target (plain_lgbm.py line 14: objective="regression_l1")
- [x] **Training sample weights not used**: Explicitly documented (plain_lgbm.py line 63-68: no weight parameter to lgb.Dataset)
- [x] **Same model configuration as F1**: PlainLightGBMConfig unchanged; verified by comparing config.audit_dict() output

**Evidence**: `APPROVED_FEATURES` = 36 features; all checks passed during execution; config matches F1 exactly

### Scoring and Alignment

- [x] **Same frozen eval scoring**: All metrics from `eval/metrics.py` via `score_kaggle_aligned`
- [x] **Same Stage 2 baseline comparison per fold**: Baselines from results.csv rows 9, 16, 23, 30
- [x] **Same row coverage per fold as Stage 2**: F2: 43,433 / 92.37%, F3: 42,794 / 91.01%, F4: 42,035 / 89.40%
- [x] **WMAE, WAPE, and bias computed correctly**: Frozen `eval/metrics.py` functions
- [x] **Prediction keys are unique**: Checked at run_stage3_all_folds line 244-245: `duplicated().any()` raises
- [x] **Missing or extra prediction keys would be rejected**: Frozen scoring in `eval/grid.py` uses exact-key alignment
- [x] **Aggregate F1-F4 mean values are correct**: Verified by manual calculation matching report

**Evidence**: All scoring alignment checks passed; metrics computed by frozen evaluation layer; aggregate calculations verified

### results.csv Integrity

- [x] **Stage 3 F1 result row present**: Row 32, git_hash=UNCOMMITTED_STAGE3_F1
- [x] **F2-F4 result rows present**: Rows 41-43, git_hash=UNCOMMITTED_STAGE3_ALL_FOLDS
- [x] **Stage 2 reference correctly cited**: F1 baseline from row 9, F2 from row 16, F3 from row 23, F4 from row 30
- [x] **All-fold aggregate values match**: Mean WMAE 20.6963289733 verified
- [x] **F1 not modified**: Row 32 unchanged

---

## 8. Feature Contract Assessment

**Assessment: PASSED**

The approved 36-feature design remained unchanged across all folds (F1-F4).

- Exact feature list: `unique_id`, `product_unique_id`, 5 static category fields, 10 known-future date/calendar fields, 8 price/discount fields, 11 historical demand features
- No future sales, future `total_orders`, or future availability in features
- No `weight`, `id`, or `sales_hat` in model matrix
- No solution.csv use
- No recursive predictions
- No clipping or rounding
- No target transformation
- No hyperparameter changes from F1
- Training sample weights not used (documented)

Every fold enforced the exact `APPROVED_FEATURES` tuple (stage3_minimal.py lines 37-64) and `FORBIDDEN_FEATURE_FIELDS` frozenset (lines 94-96).

---

## 9. Scoring and results.csv Integrity Assessment

**Assessment: PASSED**

Reported metrics and aggregate values are internally consistent.

- Individual fold WMAE/WAPE/bias values match results.csv rows 32, 41-43
- Aggregate mean WMAE: (20.5827509351 + 22.6716726491 + 20.3217981649 + 19.2090941439) / 4 = 20.6963289733 ✓
- Aggregate mean WAPE: (0.2186937932 + 0.2293625889 + 0.2015732309 + 0.2041885773) / 4 = 0.2134545476 ✓
- Aggregate mean bias: (-0.0153873760 + -0.0953250067 + -0.0141731654 + -0.0728248588) / 4 = -0.0494276017 ✓
- Stage 2 mean WMAE: (31.1985803308 + 32.9412795963 + 30.1713535275 + 27.9464531827) / 4 = 30.5644166593 ✓

All calculations verified manually and match the reported values exactly.

---

## 10. Plausibility Assessment

**Assessment: Plausible with documented caveats**

The all-fold improvements are plausible after the F1 adversarial review and ablations. The evidence supports that the 32.29% aggregate mean-WMAE improvement (30.5644166593 → 20.6963289733) is explained by complementary contributions from:

1. **Known-future commercial covariates** (price/discount): A1 ablation showed removing these 8 features increased WMAE by 4.579 points (43.13% of full gain). This is the dominant single contributor.

2. **Multi-origin training design**: A5 ablation showed a single origin (2024-05-05) scored 25.966 WMAE vs 20.583 for twelve origins, losing 50.71% of the gain. Exposure to twelve non-overlapping historical windows provides substantial temporal coverage.

3. **Historical demand features**: A2 ablation (12 historical features only) scored 25.721 WMAE, retaining 51.60% of the gain independently.

4. **Static metadata + known-future (without historical demand)**: A4 scored 22.629 WMAE, retaining 80.73% of the gain, indicating calendar/price/discount/ID information is highly predictive even without demand history.

The fold-to-fold stability is strong:
- WMAE range: 19.2091 to 22.6717 (standard deviation: 1.2517)
- Relative improvement range: 31.18% to 34.03% (narrow 2.85% spread)
- All folds beat their matched Stage 2 baseline
- All folds trigger the suspicious-improvement threshold

The consistency across folds, combined with the F1 ablation evidence, supports that the result is mechanically sound and not the product of a fold-specific anomaly. However, the magnitude remains tied to the benchmark-specific known-future policy.

---

## 11. Required Fixes

**None.** No fixes are required before proceeding.

All identified issues are either:
- Medium-risk plausibility concerns requiring disclosure (M1-M3)
- Low-risk documentation or interpretability items (L1-L4)
- Known limitations already documented

The Stage 3 all-fold implementation passes all leakage checks, protected-layer tests, success gates, and adversarial review verification.

---

## 12. Recommended Next Gate

**Recommendation: prepare Kaggle submission candidate**

### Rationale:

1. **No high-risk findings**: All critical leakage controls passed across all four folds
2. **Implementation verified**: Feature contract, cutoff discipline, and scoring alignment confirmed
3. **Plausibility established**: F1 ablations explain the improvement magnitude; all-fold consistency reinforces this
4. **Tests pass**: 45 unit tests passed; Stage 1 validation passed
5. **Protected layers unchanged**: `eval/` and `dataguard/` untouched since Stage 1 freeze

### Conditions for proceeding:

1. Human review and acceptance of this adversarial review report
2. Explicit acknowledgment of the caveats (M1-M3):
   - All folds exceed 20% improvement threshold
   - Result depends on Kaggle-known-future price/discount covariates
   - F2 exhibits elevated negative bias (-9.53%)
3. Commit this review report to the repository
4. Do not submit to Kaggle without separate human authorization

### Blockers cleared:

- ✅ All-fold adversarial review complete
- ✅ No high-risk issues found
- ✅ Tests pass
- ✅ Stage 1 validation passes
- ✅ eval/ and dataguard/ unchanged

### Remaining caveats (must be disclosed in any public claim):

- The 32.29% improvement depends on Kaggle-known-future price/discount information
- Multi-origin training provides temporal coverage advantage over single-window baselines
- F2 shows higher under-forecasting bias than other folds

---

## Review Summary

- **Verdict**: trusted with caveats
- **High-risk findings**: 0
- **Medium-risk findings**: 3 (suspicious improvement across all folds, F2 bias, price/discount dependency)
- **Low-risk findings**: 4 (F1 not rerun, coverage variation, negative predictions, feature contract maintained)
- **Tests run**: 45 unit tests passed, Stage 1 validation passed
- **Code modified**: None
- **Protected layers**: eval/ and dataguard/ unchanged since freeze
- **All-fold result can be trusted as local diagnostic**: Yes
- **Ready for Kaggle submission preparation**: Yes, with caveats disclosed

---

*This review was conducted as a read-only adversarial audit. No code was modified, formatted, committed, or deleted during the review process. All findings are based on file inspection, command execution, and logical verification against the Stage 3 feature contract and Stage 2 baselines.*
