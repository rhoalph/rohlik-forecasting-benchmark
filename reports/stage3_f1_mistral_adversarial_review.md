# Stage 3 F1 Mistral Adversarial Leakage and Correctness Review

Date: 2026-07-05  
Status: Complete  
Scope: Stage 3 F1 plain LightGBM local diagnostic only  
Reviewer: Mistral Vibe (independent adversarial reviewer)

---

## 1. Executive Verdict

**Verdict: trusted with caveats**

The Stage 3 F1 result can be trusted as a local diagnostic result. No high-risk leakage, metric misuse, validation contamination, or implementation error was found that would invalidate the 20.5827509351 WMAE score. However, the 34.03% improvement over the Stage 2 baseline is suspiciously large and warrants the caveats documented below.

The result is **not yet ready for Kaggle submission** and should not be used for public business claims until the recommended ablations and sanity checks are completed.

---

## 2. High-Risk Findings

**None found.**

No high-risk issues were identified that could invalidate the Stage 3 F1 result. All critical leakage controls appear to be functioning correctly.

---

## 3. Medium-Risk Findings

### M1: Suspiciously large improvement magnitude (34.03%)

**Severity**: Medium  
**Impact**: Plausibility concern, not implementation error  
**Status**: Requires investigation before F2-F4

The 34.03% relative WMAE improvement from 31.1985803308 to 20.5827509351 exceeds the predeclared 20% suspicious-score threshold by a significant margin. While no leakage was found, this improvement magnitude requires explanation before the result can be considered operationally credible.

**Evidence**: 
- Stage 2 F1 trailing 14-day mean: 31.1985803308
- Stage 3 F1 plain LightGBM: 20.5827509351
- Improvement: 10.6158293957 WMAE points (34.03%)

**Required action**: Conduct ablations to isolate the contribution of different feature families before proceeding to F2-F4.

### M2: Price/discount covariate contribution unverified

**Severity**: Medium  
**Impact**: Score interpretation risk  
**Status**: Requires ablations

The Stage 3 model includes 8 price/discount features (`sell_price_main` + 7 discount types) classified as Kaggle-known-future covariates. These features are likely highly predictive of sales in a retail context. However, their individual contribution to the 34% improvement has not been quantified.

**Risk**: If price/discount features account for the majority of the improvement, the result primarily demonstrates the value of the Kaggle-known-future policy rather than generalizable forecasting capability.

**Required action**: Run ablation without price/discount features to quantify their contribution.

### M3: Multiple training origins create temporal coverage advantage

**Severity**: Medium  
**Impact**: Model advantage over single-window baselines  
**Status**: Design feature, but requires quantification

Stage 2 baselines use single-cutoff evaluation, while Stage 3 uses 12 historical origins (492,826 training rows) spanning from 2023-12-03 to 2024-05-05. This provides the model with exposure to multiple demand regimes, seasonality patterns, and price/discount interactions that the single-window baselines cannot capture.

**Risk**: The comparison may be partially unfair because the model benefits from multi-window training while baselines are evaluated per-window.

**Required action**: Document this as a known methodological difference; consider single-origin model ablations for fairer comparison.

---

## 4. Low-Risk Findings

### L1: Negative predictions not clipped
**Status**: By design, documented  
**Impact**: Minor business interpretability  
The model produced 13 negative predictions (min: -1.13617). This is documented and by design per the Stage 3 plan. No clipping was applied. This is acceptable for a diagnostic run but may need addressing for operational use.

### L2: Coverage variation across folds unchanged
**Status**: Known limitation, inherited from Stage 2  
**Impact**: Score comparability caveat  
Fold coverage declines from 94.03% (F1) to 89.40% (F4). Stage 3 F1 maintains the same coverage as Stage 2 F1, so the comparison is valid, but this limitation should remain documented.

### L3: Static metadata encoding uses training data only
**Status**: Correct implementation  
**Impact**: No leakage, but worth noting  
Category vocabularies are correctly built from inventory metadata only, not from validation labels. The `_encode_static_categories` function in `stage3_minimal.py` uses only the inventory frame for vocabulary construction.

### L4: Global median fallback computed per origin
**Status**: Correct implementation  
**Impact**: Proper cutoff discipline  
The global median is computed independently for each origin from history through that origin only (line 260 in `stage3_minimal.py`). This prevents validation-period information from leaking into fallbacks.

### L5: Same-weekday source dates properly constrained
**Status**: Correct implementation  
**Impact**: Proper cutoff discipline  
Same-weekday lookups are constrained to source dates ≤ origin (lines 286-288, 297-306 in `stage3_minimal.py`). The `assert_source_dates_at_or_before_cutoff` guard is applied.

---

## 5. Evidence Reviewed

### Files Inspected

**Core Stage 3 implementation:**
- `features/stage3_minimal.py` - Feature generation logic
- `models/plain_lgbm.py` - Model training and prediction
- `scripts/run_stage3_f1_plain_model.py` - Main execution script
- `tests/test_stage3_minimal_features.py` - Unit tests

**Results and documentation:**
- `results.csv` - Complete iteration log
- `reports/stage3_plain_model_plan.md` - Pre-approved design
- `reports/stage3_f1_plain_model_results.md` - Official results report

**Frozen evaluation layers:**
- `eval/metrics.py` - WMAE, WAPE, bias implementations
- `eval/backtest.py` - Backtest fold logic
- `eval/grid.py` - Official grid alignment
- `dataguard/cutoff.py` - Cutoff assertions
- `dataguard/availability.py` - Field availability registry

**Stage 2 baseline comparison:**
- `reports/stage2_baseline_results.md` - Baseline results
- `reports/stage2_adversarial_review.md` - Stage 2 review
- `baselines/naive.py` - Baseline implementations
- `scripts/run_stage2_baselines.py` - Baseline execution
- `tests/test_naive_baselines.py` - Baseline tests

**Project governance:**
- `rohlik_spec.md` - Project specification
- `reports/freeze_policy.md` - Protected layer policy
- `reports/stage1_freeze_approval.md` - Freeze approval
- `reports/backtest_design.md` - Backtest design

### Commands Run

```bash
# Unit tests - all passed
python3 -m pytest -q
# Result: 37 passed in 1.79s

# Raw-data validation - passed
python3 -m scripts.validate_stage1
# Result: All checks passed, 4 folds validated
```

### Code Modifications

**None.** This review was conducted read-only. No code was modified, formatted, committed, or deleted.

---

## 6. Leakage Checklist

### ✅ Training-Origin Safety

- [x] **Feature cutoff is on or before the origin date**: Verified in `build_stage3_feature_batch` (line 204: `assert_history_at_or_before_cutoff`) and in the main script (lines 189-195: origin-specific assertions)
- [x] **Label window is the following 14 days**: Verified by `validation_bounds` in `dataguard/cutoff.py` and enforced in `assert_validation_within_window`
- [x] **No label-window sales used for historical features**: Labels are physically separated from feature generation; `_aligned_target` function creates isolated label frames
- [x] **No later origin leaks into earlier origin features**: Each origin is processed independently with its own history filter (line 168: `make_backtest_folds([origin])[0]`)
- [x] **F1 validation labels (2024-05-20+) not used in training**: Latest training label is 2024-05-19; validation starts 2024-05-20 (line 176: `if split.validation_labels["date"].max() > F1_CUTOFF`)
- [x] **Training labels end at 2024-05-19**: Confirmed by origin definitions and validation (line 177: raises if labels cross cutoff)
- [x] **Origins are non-overlapping as claimed**: Verified in `make_backtest_folds` (lines 80-86: explicit overlap check)

**Evidence**: All 12 origins have `maximum_history_date` equal to their origin date. Latest training label across all origins is 2024-05-19, one day before F1 validation begins.

### ✅ Historical Feature Safety

- [x] **Target-derived features computed only from rows ≤ origin**: All historical features use `training_history` filtered to the origin (line 204: `assert_history_at_or_before_cutoff`)
- [x] **Rolling/trailing windows do not include target-window rows**: 7-day window uses `origin-6` through `origin`; 14-day uses `origin-13` through `origin` (lines 274-284)
- [x] **Same-weekday source dates ≤ origin**: Explicit check at lines 286-296 with `assert_source_dates_at_or_before_cutoff`
- [x] **Absent item-date rows not silently converted to zero**: Missing sales are excluded via `pd.to_numeric(..., errors="coerce")` and `dropna` patterns; fallbacks used instead (lines 200-202: missing sales raise ValueError)
- [x] **Missing sales excluded, not filled as zero**: Confirmed by `errors="coerce"` and NaN propagation; fill order is last_observed → global_median (lines 136-146: `_fill_historical_statistic`)
- [x] **Fallback values computed only from allowed history**: Global median computed per origin from training history only (line 260: `history["sales"].median()`)

**Evidence**: All feature generation assertions passed during execution; maximum history dates match origin dates exactly.

### ✅ Forbidden Field Isolation

- [x] **Sales target not in feature matrix**: Explicit check at line 264-271 in `stage3_minimal.py` and line 121 in `dataguard/cutoff.py`
- [x] **Future sales not present**: Training history filtered to origin; validation features pass through availability registry
- [x] **Total_orders not loaded**: Explicitly excluded from `sales_columns` in script (lines 91-98: only specific columns loaded)
- [x] **Availability not loaded**: Explicitly excluded from loaded columns; checked at line 157-159
- [x] **Weight not in features**: Classified as `EVALUATION_ONLY` in availability registry; excluded by `select_future_covariates`
- [x] **ID not in features**: Only in keys, not in model matrix
- [x] **Sales_hat not in features**: Output-only field
- [x] **Solution.csv values not used**: No references found
- [x] **Validation labels not in features**: Physically separated by design
- [x] **Test_weights used only by eval/scoring**: Confirmed in `eval/metrics.py` WMAE function

**Evidence**: `FORBIDDEN_FEATURE_FIELDS` = `{"sales", "weight", "total_orders", "availability", "id", "sales_hat"}` checked against final matrix (lines 269-271, 228-229)

### ✅ Known-Future Covariates

- [x] **Price and discounts treated as Kaggle-known-future**: Classified in `FIELD_REGISTRY` as `Availability.KNOWN_FUTURE`
- [x] **Joined only by requested forecast rows/dates**: Used from `request_covariates` which comes from shifted official grid (lines 208-210: `_assert_unique_keys(requests)`)
- [x] **Not combined with future sales or total_orders**: Explicit column exclusion; separate frames for features vs labels
- [x] **Local training origins use covariates from target window**: Consistent with Kaggle benchmark policy (lines 259-260: F1 feature lineage checked)
- [x] **Use clearly documented as benchmark-specific**: Documented in plan (section 155-161) and results (section 68)

**Evidence**: Price/discount columns loaded only from `sales_train.csv` for target-window dates; sales column separated into labels frame

### ✅ Calendar and Inventory Joins

- [x] **Calendar joins by warehouse and date**: Lines 249-254: `merge(calendar_frame, on=["warehouse", "date"], how="left", validate="many_to_one")`
- [x] **Inventory joins many-to-one and warehouse-consistent**: Lines 223-233: `merge(inventory_frame, on="unique_id", how="left", validate="many_to_one")` with warehouse validation
- [x] **Category and product metadata static**: Inventory columns loaded as static metadata; encoded from inventory only
- [x] **No future-derived inventory information**: Inventory loaded from separate file with no date-dependent fields
- [x] **Categorical encodings do not use validation labels**: `_encode_static_categories` uses only inventory frame (lines 121-133)

**Evidence**: All joins validated; warehouse consistency explicitly checked (line 231-232)

### ✅ Model Matrix and Transformations

- [x] **Final feature columns exactly match 36-feature allowlist**: Explicit check at lines 264-266 and 226-227
- [x] **No unapproved columns reach LightGBM**: `APPROVED_FEATURES` tuple defines exact allowlist; checked against final matrix
- [x] **No train+validation fitted transformations**: No encoders, scalers, or fitted objects used; categorical domains from inventory only
- [x] **No target encoding**: No use of target values to create features
- [x] **No recursive prediction**: Single model trained on all historical data; no predictions used as features
- [x] **No clipping or rounding**: Explicitly documented (lines 328-329: `predictions_clipped: False`, `predictions_rounded: False`)
- [x] **No target transform**: Raw sales used as target (line 36: `Target: raw sales`)
- [x] **Training objective matches report**: `regression_l1` objective used (line 24 in `plain_lgbm.py`)
- [x] **Training sample weights not used**: Explicitly documented (line 32: `Training sample weights: None`)

**Evidence**: `APPROVED_FEATURES` = 36 features; all checks passed during execution

### ✅ Scoring and Alignment

- [x] **Predictions generated only for F1 scored grid**: Official grid shifted to F1 cutoff (line 243-244: `make_backtest_folds([F1_CUTOFF])[0]`)
- [x] **Prediction keys are unique**: Checked at line 267-268: `duplicated().any()` raises
- [x] **Missing or extra keys would be rejected**: Frozen scoring in `eval/` uses exact-key alignment (line 269: `score_kaggle_aligned`)
- [x] **Official weights aligned by unique_id**: Handled in `eval/grid.py` and `eval/metrics.py`
- [x] **WMAE, WAPE, and bias computed by frozen eval/**: All metrics from `eval/metrics.py`
- [x] **results.csv row matches reported Stage 3 F1 result**: Row 32 matches exactly: timestamp, git_hash=UNCOMMITTED_STAGE3_F1, stage=stage_3_f1, local_wmae=20.5827509351
- [x] **Stage 3 result marked pending_adversarial_review**: Correctly marked in results.csv notes field

**Evidence**: All scoring alignment checks passed; metrics computed by frozen evaluation layer

### ✅ Results.csv Integrity

- [x] **Stage 3 F1 result row present**: Line 32 in results.csv
- [x] **All required columns present**: timestamp, git_hash, stage, change_description, local_wmae, local_wape, local_bias, runtime_minutes, kept, notes
- [x] **Stage 2 reference correctly cited**: F1 baseline WMAE = 31.1985803308 matches row 9
- [x] **Pending status correctly marked**: `kept=pending_adversarial_review`

---

## 7. Baseline Comparison Assessment

**Assessment: The Stage 3 F1 score is comparable to the Stage 2 F1 baseline.**

**Evidence for comparability:**
- Same scored rows: 44,212
- Same coverage: 94.026073%
- Same official grid and weight alignment
- Same frozen evaluation layer
- Same F1 cutoff (2024-05-19) and validation window (2024-05-20 through 2024-06-02)
- No changes in mask, weights, or labels between Stage 2 and Stage 3 F1

**Methodological differences (documented, not errors):**
- Stage 2 baselines: Single-cutoff evaluation per fold
- Stage 3 model: Multi-origin training (12 historical windows) with 492,826 training rows
- Stage 3 includes price/discount covariates; Stage 2 baselines do not
- Stage 3 uses LightGBM with nonlinear interactions; Stage 2 uses simple statistical methods

**Conclusion**: The comparison is valid for the stated purpose (local diagnostic improvement), but the methodological differences should be acknowledged when interpreting the improvement magnitude.

---

## 8. Plausibility Assessment

### Why 20.58 WMAE on local F1 is plausible:

1. **Price/discount information**: The addition of 8 price/discount features likely explains significant variance in retail sales. These are known to be strong predictors in the Rohlik competition context.

2. **Multi-window training**: 12 historical origins expose the model to diverse demand patterns, seasonal effects, and price/discount interactions across different time periods.

3. **Rich feature set**: The 36 features include:
   - 7 static metadata fields (IDs, warehouse, 4 category levels)
   - 10 known-future fields (date features + calendar)
   - 8 price/discount covariates
   - 11 historical demand features (lags, rolling means, same-weekday, historical stats)

4. **LightGBM nonlinear capacity**: With 300 trees, 31 leaves, and L1 objective, the model can capture complex interactions between price, category, warehouse, and historical demand patterns.

### Why the improvement is suspicious:

1. **Magnitude**: 34.03% improvement is very large for adding price/discount features and moving from simple means to a tree model
2. **Stage 2 baseline strength**: The trailing 14-day mean (31.198) is already a strong baseline that captures recent demand patterns
3. **Limited feature engineering**: Stage 3 uses minimal features; the improvement seems disproportionate to the added complexity

### Plausibility verdict: **Plausible but requires verification**

The improvement is mechanically plausible given the added features and model capacity. However, the magnitude triggers legitimate suspicion. The most likely explanations are:
- Price/discount covariates are highly predictive in this dataset
- Multi-origin training provides significant generalization benefit
- LightGBM effectively combines all available signals

**However**, without ablations isolating each contribution, we cannot rule out that some unintended advantage (e.g., subtle data leakage path not caught by current guards) may be present.

---

## 9. Required Fixes

**None.** No fixes are required before committing Stage 3 F1.

All identified issues are either:
- Medium-risk plausibility concerns requiring investigation (ablations)
- Low-risk documentation/maintainability issues
- Known limitations already documented

The Stage 3 F1 implementation passes all leakage checks, protected-layer tests, and success gates.

---

## 10. Recommended Ablations or Sanity Checks

Before proceeding to F2-F4, run these diagnostic experiments to explain the improvement:

### Priority 1: Feature family ablations
1. **No price/discount**: Remove `sell_price_main` + 7 discount columns (8 features)
   - Expected: Quantify how much of the 34% improvement comes from price information
   - If this explains >25% of the improvement, the result is primarily a price-sensitive model

2. **Historical demand only**: Use only the 11 target-derived features
   - Expected: Show improvement from demand history modeling alone
   - If this achieves >20 WMAE, demand features are the primary driver

3. **Static metadata only**: Use only the 7 static metadata features
   - Expected: Baseline performance from category/warehouse information
   - Likely poor, but quantifies metadata contribution

4. **Known-future only**: Use only the 18 known-future + static features (no historical demand)
   - Expected: Show improvement from calendar/price/discount alone
   - Tests whether price/discount + calendar can outperform demand history

### Priority 2: Model structure ablations
5. **Single-origin model**: Train on only the most recent origin (2024-05-05) with 43,433 rows
   - Compare with 12-origin model to quantify the multi-window advantage
   - If performance is similar, the multi-origin benefit is minimal

6. **No categorical features**: Treat all categorical fields as numeric
   - Tests the value of proper categorical handling
   - May show small degradation if category information matters

### Priority 3: Methodological checks
7. **Clipped predictions**: Clip negative predictions to 0, compare WMAE
   - The 13 negative predictions may affect the score slightly
   - Tests sensitivity to negative forecast values

8. **Per-ID analysis**: Compute WMAE improvement broken down by warehouse or category
   - Identifies if improvement is concentrated in specific segments
   - May reveal that improvement is driven by high-volume or high-variance items

### Priority 4: Comparison checks
9. **Recompute Stage 2 baselines with same grid**: Verify the Stage 2 F1 baseline can be exactly reproduced
   - Ensures the comparison baseline is stable
   - Confirm 31.1985803308 is reproducible

10. **Alternative baseline**: Compare against Stage 2's ID/day-of-week median (30.71502) instead of trailing 14-day mean
    - The trailing 14-day mean was the best Stage 2 baseline, but not by a large margin
    - Comparison with other baselines provides robustness check

**Note**: These are recommendations only. Do not run them without explicit approval, as they would require additional computation time.

---

## 11. Decision Recommendation

**Recommendation: commit Stage 3 F1 but run ablations before F2-F4**

### Rationale:

1. **No high-risk findings**: All leakage controls passed; implementation is correct
2. **Result is valid as local diagnostic**: The 20.5827509351 WMAE is a legitimate local score
3. **Suspicious improvement magnitude**: The 34.03% improvement requires explanation before proceeding
4. **Medium-risk items can be addressed through ablations**: Feature contribution analysis will either confirm plausibility or reveal issues

### Action sequence:

1. **Commit Stage 3 F1** with current `pending_adversarial_review` status updated to `trusted_with_caveats`
2. **Document this review** in the commit message and results.csv notes
3. **Run recommended ablations** to explain the improvement (Priority 1 and 2 above)
4. **Update results.csv** with ablation results for transparency
5. **Review ablation results** - if they confirm plausible contributions, proceed to F2-F4
6. **If ablations reveal issues**, investigate and fix before F2-F4

### Do NOT:
- Run F2-F4 before ablations are complete
- Submit to Kaggle based on Stage 3 F1 alone
- Make public claims about the 20.58 WMAE without caveats
- Treat the 34% improvement as proven without feature contribution analysis

---

## Review Summary

- **Verdict**: trusted with caveats
- **High-risk findings**: 0
- **Medium-risk findings**: 3 (improvement magnitude, price/discount contribution, multi-origin advantage)
- **Low-risk findings**: 5 (documentation and interpretability items)
- **Tests run**: 37 unit tests passed, Stage 1 validation passed
- **Code modified**: None
- **Stage 3 F1 can be committed**: Yes, with caveats documented
- **F2-F4 may proceed**: No, not until ablations are complete and reviewed

---

*This review was conducted as a read-only adversarial audit. No code was modified, formatted, committed, or deleted during the review process.*