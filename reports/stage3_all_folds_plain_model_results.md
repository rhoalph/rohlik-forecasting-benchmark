# Stage 3 Plain LightGBM All-Fold Results

Date: 2026-07-05
Status: F1–F4 local diagnostics complete; pending all-fold adversarial review
Scope: frozen local backtests only; not a Kaggle score or submission

## Purpose

This run tests whether the approved Stage 3 plain LightGBM design generalizes across the remaining frozen local folds. F1 was not rerun; its committed metrics are included only in aggregate reporting. F2, F3, and F4 were built, trained, predicted, and scored sequentially with their own safely adapted twelve-origin training windows.

No hyperparameter, feature, target, categorical, clipping, rounding, or scoring change was made. No ablation, model artifact, processed dataset, Kaggle submission, or public claim was produced.

## Model configuration

| Setting | Value |
|---|---|
| Model | One global LightGBM per outer fold |
| Features | Exact approved 36-feature contract |
| Target | Raw sales |
| Objective | Unweighted `regression_l1` |
| Boosting rounds | 300 fixed rounds |
| Learning rate | 0.05 |
| Leaves | 31 |
| Minimum rows per leaf | 100 |
| Seed / threads | 42 / 6 |
| Categorical handling | Same nine approved categorical fields |
| Prediction policy | Continuous, unrounded, unclipped, non-recursive |
| Commercial covariates | Price/discount treated as Kaggle-known-future only |

Future `total_orders` and future availability remained excluded.

## Fold-by-fold results

| Fold | Cutoff | Training rows | Scored rows | Coverage | WMAE | WAPE | Bias | Runtime (min) | Peak RSS (MiB) | Negative predictions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 2024-05-19 | 492,826 | 44,212 | 94.03% | 20.5827509351 | 0.2186937932 | -0.0153873760 | 3.497 | 1,517.70 | 13 |
| F2 | 2024-05-05 | 490,224 | 43,433 | 92.37% | 22.6716726491 | 0.2293625889 | -0.0953250067 | 3.165 | 1,510.11 | 19 |
| F3 | 2024-04-21 | 487,826 | 42,794 | 91.01% | 20.3217981649 | 0.2015732309 | -0.0141731654 | 3.208 | 1,531.65 | 22 |
| F4 | 2024-04-07 | 486,315 | 42,035 | 89.40% | 19.2090941439 | 0.2041885773 | -0.0728248588 | 3.126 | 1,536.51 | 0 |

F1 values come from the committed Stage 3 F1 result and were not recomputed. F2–F4 runtimes cover fold-specific feature generation, fitting, prediction, and scoring but exclude the shared 6.933-second raw load.

## Training-origin coverage

| Fold | Number of origins | Most recent origin | Oldest origin | Latest training label | Validation begins |
|---|---:|---|---|---|---|
| F1 | 12 | 2024-05-05 | 2023-12-03 | 2024-05-19 | 2024-05-20 |
| F2 | 12 | 2024-04-21 | 2023-11-19 | 2024-05-05 | 2024-05-06 |
| F3 | 12 | 2024-04-07 | 2023-11-05 | 2024-04-21 | 2024-04-22 |
| F4 | 12 | 2024-03-24 | 2023-10-22 | 2024-04-07 | 2024-04-08 |

Each origin predicts its immediately following non-overlapping 14-day window. Every target-derived feature uses history ending at that origin, and every outer-fold training label ends on or before the outer cutoff.

## Stage 2 trailing 14-day comparison

| Fold | Stage 2 WMAE | Stage 3 WMAE | Absolute improvement | Relative improvement | Beats baseline | Suspicious >20% flag |
|---|---:|---:|---:|---:|---|---|
| F1 | 31.1985803308 | 20.5827509351 | 10.6158293957 | 34.03% | Yes | Yes |
| F2 | 32.9412795963 | 22.6716726491 | 10.2696069471 | 31.18% | Yes | Yes |
| F3 | 30.1713535275 | 20.3217981649 | 9.8495553626 | 32.65% | Yes | Yes |
| F4 | 27.9464531827 | 19.2090941439 | 8.7373590388 | 31.26% | Yes | Yes |

All four folds beat their matched Stage 2 reference on exactly the corresponding frozen scored population. Every fold also crosses the predeclared suspicious-improvement threshold, so the consistency is evidence for stability but does not waive adversarial review.

## F1–F4 aggregate metrics

The aggregates are equal-weight arithmetic means of the four fold metrics, not pooled-row scores.

| Metric | Stage 2 trailing 14-day mean | Stage 3 plain-model mean | Change |
|---|---:|---:|---:|
| WMAE | 30.5644166593 | 20.6963289733 | -9.8680876861 |
| WAPE | 0.2974236747 | 0.2134545476 | -0.0839691272 |
| Bias | -0.0077277837 | -0.0494276017 | -0.0416998180 |

The ratio of aggregate mean-WMAE improvement is 32.29%. The mean of the four fold-specific relative improvements is 32.28%.

## Runtime and memory

| Measurement | Result |
|---|---:|
| F2–F4 wall time including one raw load | 577.103 seconds / 9.618 minutes |
| F1 committed runtime | 209.827 seconds / 3.497 minutes |
| Combined F1 plus F2–F4 diagnostic runtime | 786.930 seconds / 13.115 minutes |
| Maximum observed RSS | 1,536.51 MiB / 1.50 GiB |

CPU and repeated origin-specific aggregation remain the main cost. Memory stayed far below the 12 GiB Stage 3 gate.

## Leakage and cutoff assertions

For every F2–F4 fold, the runner asserted:

- exactly twelve origins spaced 14 days apart;
- each origin's labels start after the origin and end on or before the outer cutoff;
- maximum historical feature date is on or before its origin;
- maximum same-weekday source date is on or before its origin;
- combined training and validation matrices match the exact 36-feature allowlist;
- target, weight, future `total_orders`, future availability, output fields, and other forbidden columns are absent from model features;
- outer validation dates, scored rows, and coverage match the frozen fold contract;
- prediction keys are unique and frozen scoring enforces exact key equality;
- known-future price and discounts remain isolated from sales labels;
- no model setting or feature transform is fitted using outer validation.

`eval/` and `dataguard/` were not modified.

## Stability and business diagnostics

WMAE ranges from 19.2091 to 22.6717, with a population standard deviation of 1.2517. More importantly, relative improvement versus each fold's baseline stays in a narrow 31.18%–34.03% range. This is materially more stable than a one-fold result.

WAPE is also lower than the matched baseline in every fold. Bias is consistently negative, indicating under-forecasting:

- F1 and F3 are near neutral at -1.54% and -1.42%.
- F4 under-forecasts by 7.28%.
- F2 under-forecasts by 9.53%, close to the predeclared 10% F1 guardrail.

The mean bias is -4.94%. F2's directional error requires attention in the all-fold review even though its absolute-error metrics improve.

Negative raw predictions remain uncommon: 13, 19, 22, and 0 for F1–F4 respectively. They were scored unchanged under the approved no-clipping policy.

## Suspicious-score assessment

Every fold independently triggers the greater-than-20% suspicious-improvement rule. The narrow improvement range and successful feature-family/training-design ablations provide a plausible explanation: benchmark-known-future commercial covariates, broader multi-origin training, static identity, and historical demand all contribute. No fold-specific discontinuity or new leakage signal appeared.

Nevertheless, these results remain local diagnostics. The repeated trigger requires an all-fold adversarial review before the result can be trusted as a multi-fold reference or used to prepare a Kaggle submission candidate.

## Recommendation for the next gate

Proceed to an all-fold adversarial leakage and correctness review.

The review should verify the generalized origin schedule, per-fold label cutoffs, price/discount isolation, exact feature matrices, committed F1 inclusion, aggregate calculations, F2 bias, and reproduction of fold metrics. Do not prepare or submit a Kaggle candidate until that review is accepted and a separate human authorization is given.
