# Stage 5-G Fixed Blend Adversarial Review

## Verdict

fixed_blend_safe_to_promote

## Scope reviewed

This follow-up audits only globally fixed blends built from:

- Stage 5-E raw L1 predictions
- Tweedie 1.1 predictions

The reviewed fixed blends are:

- `fixed_blend_raw_70_tweedie_30`
- `fixed_blend_raw_50_tweedie_50`

The earlier fold-specific equal-blend family remains rejected because the member set changed by fold and is not directly transferable to the official Kaggle test set.

## Findings summary

- High-risk findings: 0
- Medium-risk findings: 0
- Low-risk findings: 2

Low-risk findings:

1. The fixed 70/30 blend is valid and generalizable, but it is weaker than the 50/50 blend.
2. The fold-specific equal-blend family was previously misread as an aggregate; its corrected fold-specific family mean is 18.9336363115, but it is still non-promotable because membership varies by fold.

## Blend validity checks

The fixed blends were computed with globally fixed membership and weights:

- `fixed_blend_raw_70_tweedie_30`: always raw L1 + Tweedie 1.1, weights `[0.7, 0.3]`
- `fixed_blend_raw_50_tweedie_50`: always raw L1 + Tweedie 1.1, weights `[0.5, 0.5]`

Checks passed:

- Raw L1 predictions and Tweedie 1.1 predictions aligned by identical validation keys before blending.
- The same weights were used for all folds.
- The same blend rule is applicable to the official Kaggle test set.
- No fold-specific member selection was used.
- No leakage or forbidden fields were introduced by blending.
- `eval/` and `dataguard/` were unchanged.

## Fold-by-fold results

| Blend | Fold | WMAE | WAPE | Bias | Clipped WMAE | Clipped WAPE | Clipped Bias | Negative before clip | Clipped rows | Beats Stage 5-E fold? | Beats Tweedie 1.1 fold? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| fixed_blend_raw_70_tweedie_30 | F1 | 19.2821124571 | 0.2036832135 | -0.0084678710 | 19.2820350915 | 0.2036826515 | -0.0084673089 | 7 | 7 | Yes | Yes |
| fixed_blend_raw_70_tweedie_30 | F2 | 20.7604447502 | 0.2083798160 | -0.0788251242 | 20.7595164321 | 0.2083761769 | -0.0788214851 | 36 | 36 | Yes | No |
| fixed_blend_raw_70_tweedie_30 | F3 | 18.4429250148 | 0.1856174417 | -0.0037030222 | 18.4428125171 | 0.1856167961 | -0.0037023766 | 7 | 7 | Yes | Yes |
| fixed_blend_raw_70_tweedie_30 | F4 | 18.0800372590 | 0.1902745040 | -0.0577585212 | 18.0797120134 | 0.1902720018 | -0.0577560189 | 16 | 16 | Yes | No |
| fixed_blend_raw_50_tweedie_50 | F1 | 19.2012438455 | 0.2028848391 | 0.0004131347 | 19.2012416947 | 0.2028848195 | 0.0004131544 | 1 | 1 | Yes | Yes |
| fixed_blend_raw_50_tweedie_50 | F2 | 20.5095173420 | 0.2047465276 | -0.0745773022 | 20.5094816811 | 0.2047461413 | -0.0745769159 | 5 | 5 | Yes | No |
| fixed_blend_raw_50_tweedie_50 | F3 | 18.3929644687 | 0.1838232055 | 0.0025994240 | 18.3929640459 | 0.1838231990 | 0.0025994305 | 1 | 1 | Yes | Yes |
| fixed_blend_raw_50_tweedie_50 | F4 | 17.9319426820 | 0.1875257024 | -0.0503794829 | 17.9318776934 | 0.1875252013 | -0.0503789819 | 6 | 6 | Yes | Yes |

## Aggregate results

| Blend | Mean WMAE | Mean WAPE | Mean bias | Clipped mean WMAE | Clipped mean WAPE | Clipped mean bias |
|---|---:|---:|---:|---:|---:|---:|
| fixed_blend_raw_70_tweedie_30 | 19.1413798703 | 0.1969887438 | -0.0371886346 | 19.1410190135 | 0.1969869066 | -0.0371867974 |
| fixed_blend_raw_50_tweedie_50 | 19.0089170845 | 0.1947450686 | -0.0304860566 | 19.0088912788 | 0.1947448403 | -0.0304858282 |

Reference means:

- Stage 5-E raw L1 mean WMAE: 19.5424424315
- Tweedie 1.1 mean WMAE: 19.1888923539

Improvement vs Stage 5-E raw L1:

- 70/30 blend: 0.4010625613
- 50/50 blend: 0.5335253470

Improvement vs Tweedie 1.1:

- 70/30 blend: 0.0475124837
- 50/50 blend: 0.1799752694

## Correction to the earlier fold-specific equal-blend interpretation

The previously reported `17.7209781384` was not a valid overall equal-blend mean. It was the F4 bucket from the fold-specific equal-blend family.

Corrected fold-specific equal-blend family mean WMAE:

- 18.9336363115

Improvement vs Stage 5-E raw L1:

- 0.6088061200

This family remains rejected because the membership changes by fold, so the rule is not directly usable on the official test set.

## Decision

### Fixed 70/30 blend

Safe to promote, but not the strongest option.

### Fixed 50/50 blend

Safe to promote. It:

- beats Stage 5-E raw L1 on all four folds,
- beats Tweedie 1.1 on three of four folds,
- and improves the mean WMAE versus both reference models.

## Recommendation

Prepare the fixed 50/50 raw + Tweedie 1.1 blend as the next candidate if a blend candidate is desired.

Keep Tweedie 1.1 as the single-model fallback.

Do not promote the fold-specific equal-blend family.
