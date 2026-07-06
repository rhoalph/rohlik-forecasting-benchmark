# Stage 5-G Fixed Blend Kaggle Candidate Plan

## Purpose

Prepare a Kaggle submission candidate using the fixed 50/50 blend of two already-approved Stage 5-E model families:

- raw L1 LightGBM
- Tweedie LightGBM with variance power 1.1

This is a promotion of the fixed blend only. It is not a new feature stage.

## Why the fixed 50/50 blend is promoted

The Stage 5-G adversarial review found that the fixed 50/50 blend:

- uses globally fixed membership and fixed weights,
- aligns predictions by identical validation keys before blending,
- applies the same rule to every fold and to the official Kaggle test set,
- improves the mean local WMAE versus both Stage 5-E raw L1 and Tweedie 1.1,
- and beats Stage 5-E on all folds while beating Tweedie 1.1 on three of four folds.

## Why the fold-specific equal blend is rejected

The fold-specific equal-blend family is not promotable because its member set changes by fold. That makes the rule non-generalizable to the official test set.

## Local validation evidence

- Stage 5-E raw L1 mean WMAE: 19.5424424315
- Tweedie 1.1 mean WMAE: 19.1888923539
- Fixed 50/50 raw + Tweedie mean WMAE: 19.0089170845
- Improvement vs Stage 5-E raw L1: 0.5335253470
- Improvement vs Tweedie 1.1: 0.1799752694

## Final training design

- Model A: one global LightGBM trained on the Stage 5-E 76-feature contract with raw sales and an unweighted L1 objective.
- Model B: one global LightGBM trained on the same Stage 5-E 76-feature contract with raw nonnegative sales and a Tweedie objective.
- Both models use the same twelve cutoff-safe historical origins and the same final cutoff logic as Stage 5-E.

## 76-feature contract

The Stage 5-E feature contract remains unchanged at 76 features.

No additional features are introduced.

## Model A parameters

- Objective: `regression_l1`
- Raw sales target
- Same LightGBM configuration as the Stage 5-E official candidate

## Model B Tweedie parameters

- Objective: `tweedie`
- Tweedie variance power: `1.1`
- Raw nonnegative sales target
- Same LightGBM configuration as Model A aside from the objective parameters

## Fixed blend formula

For every official test row:

`sales_hat = 0.5 * raw_l1_prediction + 0.5 * tweedie_1_1_prediction`

The blend is computed only after the two prediction vectors are aligned by identical `id` / row order.

## Prediction alignment policy

- Generate both model predictions on the same official request frame.
- Validate that the `id` order is identical before blending.
- Reject any mismatch.
- Use the same fixed blend rule for every fold during local validation and for the official Kaggle test set.

## Known-future price / discount policy

- Price and discount fields remain Kaggle-known-future benchmark covariates.
- They are allowed because they are present in the official test set.
- They are not mixed with future sales, future `total_orders`, future availability, weights, or solution targets.

## Forbidden fields

The candidate must exclude:

- future sales
- future `total_orders`
- future availability
- weights
- `id`
- `sales_hat`
- solution `sales` targets
- any other forbidden fields already frozen by Stage 5-E and Stage 3

## Clipping policy

- Apply zero clipping after blending if needed.
- Record the number of affected rows.
- Do not round predictions.
- Do not alter the individual model predictions before blending.

## No Kaggle submission yet

This plan prepares the candidate only. Submission is deferred until the pre-submit audit is complete.
