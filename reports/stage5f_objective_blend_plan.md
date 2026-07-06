# Stage 5-F/G Objective and Blend Experiment Plan

## Purpose

Test whether target and objective diversity on top of the approved Stage 5-E 76-feature contract can improve local F1-F4 validation beyond the current Stage 5-E local mean WMAE of `19.5424424315`.

This stage is still public-solution-informed work, but it is narrower than the Stage 5-E feature expansion:

- no new features
- no horizon-specific routing
- no Optuna
- no broad hyperparameter search
- no Kaggle submission

## Approved variants

### S5-F raw L1 control

- One global LightGBM model
- Stage 5-E 76-feature contract
- Raw `sales` target
- L1 objective
- Control variant expected to match Stage 5-E closely

### S5-F sqrt L1

- One global LightGBM model
- Stage 5-E 76-feature contract
- Train on `sqrt(sales)`
- Inverse prediction with squaring
- Clip final predictions at zero as a diagnostic

### S5-F log1p L1

- One global LightGBM model
- Stage 5-E 76-feature contract
- Train on `log1p(sales)`
- Inverse prediction with `expm1`
- Clip final predictions at zero as a diagnostic

### S5-F Tweedie

- One global LightGBM model
- Stage 5-E 76-feature contract
- Raw nonnegative `sales` target
- Tweedie objective only
- Variance power values predeclared at `1.1` and `1.3`

### S5-F Poisson

- Optional diagnostic only
- Run only if LightGBM accepts the configuration cleanly and the fit remains stable
- Skip and document the reason if not

## Blend diagnostics

After scoring the single variants, test only the following predeclared blends:

1. `70%` Stage 5-E raw L1 + `30%` best non-raw variant
2. `50%` Stage 5-E raw L1 + `50%` best non-raw variant
3. Equal average of all variants whose mean WMAE is within `0.50` of the best local single variant

No other blend weights will be searched.

## Scoring and comparison policy

- Run F1-F4 local validation only.
- Record fold WMAE, WAPE, and bias.
- Record unclipped and clipped diagnostics where applicable.
- Use the same frozen scoring and fold definitions as Stage 5-E.
- Compare each variant to the Stage 5-E local mean WMAE of `19.5424424315`.

## Promotion criteria

A new candidate is justified only if:

- a single variant or a predeclared blend beats `19.5424424315`
- preferably it beats Stage 5-E on at least 3 of 4 folds
- bias does not materially deteriorate
- leakage policy remains clean
- tests pass
- Stage 1 validation passes

## Stop criteria

Stop and do not promote if:

- no variant or blend beats Stage 5-E
- bias materially worsens
- clipping becomes the only apparent source of improvement
- any leakage concern appears
- runtime or memory becomes unreasonable
- validation or tests fail

## Logging

- Append diagnostic rows to `results.csv` for each completed variant and blend.
- Do not overwrite official Kaggle submission rows.
- Do not confuse local diagnostic rows with Kaggle scores.

