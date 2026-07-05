# Stage 3 F1 Priority 2 Ablation Results

Date: 2026-07-05
Status: A5 and A6 complete; pending human review
Scope: F1 diagnostic ablations only

## Purpose

Priority 2 tests the two remaining design questions from the trusted-with-caveats review:

- A5 measures how much the twelve-origin training design contributes relative to using only the most recent approved origin.
- A6 measures whether LightGBM's explicit native categorical treatment materially explains the full Stage 3 F1 result.

No F2–F4 fold, Priority 3/4 diagnostic, tuning, new feature, clipping, ensemble, model artifact, or Kaggle submission was run.

## Shared execution contract

Both ablations used:

- F1 cutoff 2024-05-19 and validation 2024-05-20 through 2024-06-02;
- 44,212 scored rows and 94.026073% coverage;
- the same frozen key alignment, official weights, WMAE, WAPE, and bias;
- all 36 approved Stage 3 features;
- raw sales, unweighted L1, 300 rounds, learning rate 0.05, 31 leaves, minimum 100 rows per leaf, seed 42, and six threads;
- unrounded and unclipped predictions;
- no recursive prediction or target transformation.

The shared twelve-origin and F1 matrices were built through the existing reviewed feature path. A5 selected the first approved training-origin block. A6 used the full matrix and changed only the categorical-feature argument passed to LightGBM.

## Results

| ID | Diagnostic | Features | Training rows | Validation rows | WMAE | WAPE | Bias | Runtime (s) | Peak RSS (MiB) | Negative predictions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 | Single origin: 2024-05-05 | 36 | 43,433 | 44,212 | 25.9660495815 | 0.3016912121 | -0.0104714406 | 13.048 | 1,518.09 | 27 |
| A6 | Full twelve origins; categorical handling disabled | 36 | 492,826 | 44,212 | 20.7608561587 | 0.2287776437 | -0.0130177486 | 41.619 | 1,518.09 | 53 |

Displayed model runtimes cover fitting, prediction, and frozen scoring. They exclude shared raw loading and feature generation.

| Shared phase | Result |
|---|---:|
| Raw loading | 7.936 seconds |
| Twelve-origin plus F1 feature generation | 146.960 seconds |
| Total Priority 2 wall time | 209.651 seconds / 3.494 minutes |
| Peak process RSS | 1,518.09 MiB / 1.48 GiB |

## Comparison with reference and A1–A4

| Run | WMAE | Change vs full Stage 3 | Improvement vs Stage 2 trailing 14-day | Full-model gain retained |
|---|---:|---:|---:|---:|
| Full Stage 3 F1 | 20.5827509351 | — | 10.6158293957 | 100.00% |
| A6 no categorical handling | 20.7608561587 | +0.1781052236 | 10.4377241721 | 98.32% |
| A4 known-future + static | 22.6285595972 | +2.0458086621 | 8.5700207336 | 80.73% |
| A1 no price/discount | 25.1617710449 | +4.5790201098 | 6.0368092859 | 56.87% |
| A2 historical only | 25.7205849285 | +5.1378339934 | 5.4779954023 | 51.60% |
| A5 single origin | 25.9660495815 | +5.3832986464 | 5.2325307493 | 49.29% |
| A3 static only | 28.2910454553 | +7.7082945202 | 2.9075348756 | 27.39% |
| Stage 2 trailing 14-day | 31.1985803308 | +10.6158293957 | 0 | 0% |

Positive change versus the full model means the diagnostic is worse. These are non-additive model comparisons, not causal attribution.

## A5 interpretation: contribution from twelve origins

The twelve-origin design materially contributes to the full model result.

A5 uses only the most recent origin, 2024-05-05, with 43,433 training rows. Its WMAE is 25.966050, which is 5.383299 points worse than the twelve-origin reference. The single-origin model retains 49.29% of the full model's improvement over Stage 2; conversely, removing the older eleven origins loses 50.71% of that improvement.

A5 still beats the trailing 14-day WMAE by 5.232531 points, so the model and approved features retain substantial signal without multiple origins. Its WAPE of 0.301691 is almost equal to the Stage 2 trailing 14-day WAPE of 0.302111, while bias remains close to neutral at -1.05%.

This result explains a large part of the original improvement: exposure to twelve non-overlapping historical forecast windows is a major design advantage, not a hidden scoring change. It also confirms the Mistral caveat that the full model and simple single-cutoff baselines have materially different learning capacity.

## A6 feasibility and interpretation

A6 was meaningful and safe to run. The reviewed feature builder already represents `unique_id`, product, warehouse, categories, day of week, and month as stable deterministic integer codes. The reference model explicitly passes those nine columns as categorical to LightGBM. A6 keeps identical rows, columns, codes, target, objective, and parameters but passes no categorical fields.

Treating those codes as numeric imposes arbitrary ordinal relationships, so A6 is diagnostic only and cannot replace the reviewed reference model.

A6 WMAE is 20.760856, only 0.178105 points or 0.87% worse than the full model. It retains 98.32% of the full WMAE gain. Native categorical handling therefore does not materially explain the 20.58 WMAE result.

A6 WAPE worsens by 0.010084 and produces 53 negative predictions instead of 13, so categorical handling remains useful for volume-error behavior and prediction shape even though its WMAE contribution is small.

## Updated explanation of the full result

The combined ablations provide a coherent explanation:

1. Known-future plus static information is the strongest tested feature-family driver, retaining 80.73% of the gain.
2. Price and discounts materially contribute: removing them loses 43.13% of the gain.
3. Historical-demand features independently retain 51.60% of the gain and improve the full model beyond A4.
4. Twelve historical origins are material: a single origin retains only 49.29% of the gain.
5. Static metadata alone carries some signal but is insufficient.
6. Native categorical handling changes WMAE only marginally and is not the primary explanation.

The full 20.582751 local F1 result remains plausible as the combination of benchmark-known-future commercial covariates, static identity, historical-demand summaries, and much broader temporal training coverage than the naive baselines.

## Leakage and correctness assessment

No new leakage concern appeared.

- A5 used exactly the 2024-05-05 origin and its 2024-05-06 through 2024-05-19 labels; all historical features ended on or before 2024-05-05.
- A5 retained the exact 36-feature contract and native categorical list.
- A6 used the same full training and F1 matrices as the reference design and changed only categorical treatment.
- Neither ablation changed labels, grid, weights, feature values, target, hyperparameters, or scoring.
- Forbidden columns remained absent and exact key alignment passed.
- Predictions remained raw, continuous, unrounded, and unclipped.
- `eval/` and `dataguard/` were not modified.

## Recommendation for the next gate

Recommend proceeding to F2–F4 after human acceptance of this report and the required protected/test checks. No implementation fix is indicated.

Priority 2 explains the remaining medium-risk design question: multi-origin training is a major contributor, while categorical handling is not. Priority 3 clipping and segment diagnostics may still be useful for business interpretation, but they are not necessary to establish the F1 result's mechanical plausibility and should not block multi-fold validation.

Do not run F2–F4 or submit to Kaggle without a new explicit human authorization.
