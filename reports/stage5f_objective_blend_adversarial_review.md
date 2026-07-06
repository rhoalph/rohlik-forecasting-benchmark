# Stage 5-F/G Adversarial Review

Verdict: trusted_with_caveats

## Evidence reviewed

- `reports/stage5f_objective_blend_results.md`
- `scripts/run_stage5f_objective_blend_experiments.py`
- `results.csv`
- `features/stage5e_stronger_features.py`
- `models/plain_lgbm.py`
- `eval/`
- `dataguard/`

## Commands run

- `python3 - <<'PY' ...` to recompute fold means from `results.csv`
- `python3 -m pytest -q`
- `python3 -m scripts.validate_stage1`
- `git diff --quiet HEAD -- eval dataguard`

## Recomputed means from saved Stage 5-F/G rows

### Single variants

| Variant | Mean WMAE | Mean WAPE | Mean bias |
|---|---:|---:|---:|
| raw_l1 | 19.5424424315 | 0.2027897892 | -0.0472425016 |
| sqrt_l1 | 19.3595468048 | 0.2010934456 | -0.0494941989 |
| log1p_l1 | 19.3080560058 | 0.1990826432 | -0.0445688066 |
| tweedie_1_1 | 19.1888923539 | 0.1958707021 | -0.0137296116 |
| tweedie_1_3 | 19.2458478281 | 0.1967545508 | -0.0155743556 |
| poisson | 19.4601334855 | 0.1984511040 | -0.0165099164 |

### Blend families as logged in `results.csv`

| Blend family / bucket | Mean WMAE | Mean WAPE | Mean bias | Notes |
|---|---:|---:|---:|---|
| 70% raw + 30% best non-raw | 19.2586753710 | 0.1991718991 | -0.0453981718 | Fold-specific best non-raw selection |
| 50% raw + 50% best non-raw | 19.1352184341 | 0.1975483016 | -0.0441686186 | Fold-specific best non-raw selection |
| Equal-blend family | 18.9336363115 | 0.1929785522 | -0.0270310181 | Fold-specific membership; not promotable |

The previously reported `17.7209781384` number was not a valid overall blend mean. It was the F4 bucket for one fold-specific equal-blend configuration.

## High-risk findings

- None.

## Medium-risk findings

1. The blend experiment is fold-specific in two ways:
   - the “best non-raw” member in the 70/30 and 50/50 blends was chosen separately per fold;
   - the equal-blend family membership also changed by fold.

   That means the blend logic is not directly transferable to the official test set without seeing labels. It is diagnostic only, not a promotable candidate.

2. The report aggregate previously treated the best fold-specific equal-blend bucket as if it were an overall blend mean. That is incorrect and has been corrected in the committed report.

## Low-risk findings

1. The raw L1 control reproduces Stage 5-E exactly at the mean level, which is a clean consistency check.
2. Key alignment for scoring/blending is consistent by construction: all predictions are produced on the same fold validation key order and are combined only after key-order checks.
3. The Stage 5-E feature contract remained unchanged and no forbidden fields were found in the model matrix.

## Target/objective handling

### raw_l1

- Valid control.
- Matches Stage 5-E local mean WMAE exactly.

### sqrt_l1

- Target transform and inverse are correct.
- Final clipped predictions are nonnegative.
- No future target leakage observed.

### log1p_l1

- Target transform and inverse are correct.
- Final clipped predictions are nonnegative.
- No future target leakage observed.

### tweedie_1_1

- Target handling is raw nonnegative sales.
- Objective is cleanly supported and behaved stably.
- Best single model in this stage.
- Safe to promote.

### tweedie_1_3

- Same objective family as above.
- Behaved stably but was slightly worse than Tweedie 1.1.
- Safe as a diagnostic, but not the best promotion target.

### poisson

- Mechanically safe and stable.
- Not the best single model.
- Safe as a diagnostic only.

## Prediction/key alignment

- Predictions were aligned by fold validation key order before scoring.
- No fold mixed rows from different validation sets.
- Blend predictions were formed only after the alignment check in the runner.
- Individual variant scoring and blend scoring use the same labels and keys for each fold.

## Results.csv integrity assessment

- The per-fold Stage 5-F/G rows in `results.csv` are internally consistent as local diagnostics.
- No individual row was found to have incorrect metric values.
- The issue was in the report-level aggregate interpretation of the blend family, not in the row-level metrics.

## Explicit decisions

### Tweedie 1.1

Decision: safe to promote.

Reason:
- It beats the Stage 5-E mean WMAE on every fold.
- It is stable and does not rely on fold-specific blend selection.
- It is a fixed, official-test-applicable model variant.

### Blend

Decision: reject due to fold-specific selection / non-generalizable design.

Reason:
- The blend member set changed by fold.
- The equal-blend family’s reported aggregate was initially misread as a single candidate.
- This cannot be used as an official test-set candidate without a new fixed-blend design.

## Recommendation

Prepare a Tweedie 1.1 candidate only.
Do not promote the blend family as currently defined.
If a blend is still desired later, run one fixed-blend follow-up with a globally predeclared membership and weights.
