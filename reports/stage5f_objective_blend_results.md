# Stage 5-F/G Objective and Blend Results

## Purpose

Test whether target transforms, objective diversity, and a small number of predeclared blends can improve local F1-F4 validation on top of the approved Stage 5-E 76-feature contract.

## Completed variants

- Raw L1 control
- Sqrt target + L1
- Log1p target + L1
- Tweedie variance power 1.1
- Tweedie variance power 1.3
- Poisson diagnostic (if stable)

## Skipped variants

- None if all variants complete successfully.
- Any skipped diagnostic will be documented explicitly in the row notes and below.

## Single-variant summary

| Variant | Mean WMAE | Mean WAPE | Mean bias | Runtime (min) | Peak RSS (MiB) | Beats Stage 5-E? |
|---|---:|---:|---:|---:|---:|---|
| raw_l1 | 19.5424424315 | 0.2027897892 | -0.0472425016 | 1.088 | 2376.49 | No |
| sqrt_l1 | 19.3595468048 | 0.2010934456 | -0.0494941989 | 1.185 | 2376.49 | Yes |
| log1p_l1 | 19.3080560058 | 0.1990826432 | -0.0445688066 | 1.120 | 2376.49 | Yes |
| tweedie_1_1 | 19.1888923539 | 0.1958707021 | -0.0137296116 | 1.143 | 2376.49 | Yes |
| tweedie_1_3 | 19.2458478281 | 0.1967545508 | -0.0155743556 | 1.152 | 2376.49 | Yes |
| poisson | 19.4601334855 | 0.1984511040 | -0.0165099164 | 1.264 | 2376.49 | Yes |

## Fold-by-fold table

### raw_l1

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 19.6069995273 | 0.2071636796 | -0.0217893794 | 19.6060885659 | 0.2071586518 | -0.0217843517 | 35 | 35 | 35 | No |
| F2 | 21.3218385474 | 0.2158200345 | -0.0851968572 | 21.3163474886 | 0.2158022919 | -0.0851791147 | 65 | 65 | 65 | No |
| F3 | 18.7434564665 | 0.1912725362 | -0.0131566914 | 18.7427525639 | 0.1912693329 | -0.0131534880 | 21 | 21 | 21 | No |
| F4 | 18.4974751850 | 0.1969029064 | -0.0688270785 | 18.4964147713 | 0.1968938074 | -0.0688179796 | 44 | 44 | 44 | No |

### sqrt_l1

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 19.5165757395 | 0.2084753150 | -0.0239931983 | 19.5165757395 | 0.2084753150 | -0.0239931983 | 26 | 0 | 0 | Yes |
| F2 | 21.1423934756 | 0.2135973899 | -0.0882551505 | 21.1423934756 | 0.2135973899 | -0.0882551505 | 55 | 0 | 0 | Yes |
| F3 | 18.5591120592 | 0.1890052964 | -0.0164698575 | 18.5591120592 | 0.1890052964 | -0.0164698575 | 32 | 0 | 0 | Yes |
| F4 | 18.2201059448 | 0.1932957811 | -0.0692585894 | 18.2201059448 | 0.1932957811 | -0.0692585894 | 44 | 0 | 0 | Yes |

### log1p_l1

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 19.7003628698 | 0.2083901319 | -0.0157072882 | 19.7001304436 | 0.2083894051 | -0.0157065615 | 32 | 32 | 32 | No |
| F2 | 20.9840388995 | 0.2113583445 | -0.0885612668 | 20.9837027228 | 0.2113573427 | -0.0885602650 | 45 | 45 | 45 | Yes |
| F3 | 18.6561303492 | 0.1894245943 | -0.0140485325 | 18.6560370819 | 0.1894241163 | -0.0140480545 | 33 | 33 | 33 | Yes |
| F4 | 17.8916919046 | 0.1871575021 | -0.0599581391 | 17.8916536122 | 0.1871571702 | -0.0599578072 | 33 | 33 | 33 | Yes |

### tweedie_1_1

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 19.5378993082 | 0.2080415302 | 0.0226156488 | 19.5378993082 | 0.2080415302 | 0.0226156488 | 0 | 0 | 0 | Yes |
| F2 | 20.4371715588 | 0.2013694463 | -0.0639577471 | 20.4371715588 | 0.2013694463 | -0.0639577471 | 0 | 0 | 0 | Yes |
| F3 | 18.7420127849 | 0.1866363130 | 0.0183555393 | 18.7420127849 | 0.1866363130 | 0.0183555393 | 0 | 0 | 0 | Yes |
| F4 | 18.0384857637 | 0.1874355189 | -0.0319318874 | 18.0384857637 | 0.1874355189 | -0.0319318874 | 0 | 0 | 0 | Yes |

### tweedie_1_3

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 19.6855129603 | 0.2104534005 | 0.0196119236 | 19.6855129603 | 0.2104534005 | 0.0196119236 | 0 | 0 | 0 | No |
| F2 | 20.5965128193 | 0.2037731350 | -0.0654611613 | 20.5965128193 | 0.2037731350 | -0.0654611613 | 0 | 0 | 0 | Yes |
| F3 | 18.7258054795 | 0.1859266947 | 0.0132077108 | 18.7258054795 | 0.1859266947 | 0.0132077108 | 0 | 0 | 0 | Yes |
| F4 | 17.9755600535 | 0.1868649731 | -0.0296558954 | 17.9755600535 | 0.1868649731 | -0.0296558954 | 0 | 0 | 0 | Yes |

### poisson

| Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped WMAE | Clipped WAPE | Clipped bias | Negative transformed | Negative final before clip | Clipped rows | Beats Stage 5-E fold? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 19.6969473971 | 0.2089054236 | 0.0161666519 | 19.6969473971 | 0.2089054236 | 0.0161666519 | 0 | 0 | 0 | No |
| F2 | 20.9322674318 | 0.2069603235 | -0.0550934068 | 20.9322674318 | 0.2069603235 | -0.0550934068 | 0 | 0 | 0 | Yes |
| F3 | 18.9525712414 | 0.1875963574 | 0.0150571207 | 18.9525712414 | 0.1875963574 | 0.0150571207 | 0 | 0 | 0 | No |
| F4 | 18.2587478717 | 0.1903423113 | -0.0421700315 | 18.2587478717 | 0.1903423113 | -0.0421700315 | 0 | 0 | 0 | Yes |

## Blend diagnostics

| Blend | Fold | Raw WMAE | Raw WAPE | Raw bias | Clipped rows | Beats Stage 5-E fold? |
|---|---|---:|---:|---:|---:|---|
| S5-G blend 70 percent raw control plus 30 percent best non-raw variant | F1 | 19.5047799164 | 0.2067252789 | -0.0224505251 | 19.5042933221 | Yes |
| S5-G blend 50 percent raw control plus 50 percent best non-raw variant | F1 | 19.4726920254 | 0.2068485564 | -0.0228912889 | 19.4723972391 | Yes |
| S5-G equal blend of variants within 0.50 WMAE of best single: raw_l1, sqrt_l1, log1p_l1, tweedie_1_1, tweedie_1_3, poisson | F1 | 19.2175974561 | 0.2034901182 | -0.0005159403 | 19.2175974561 | Yes |
| S5-G blend 70 percent raw control plus 30 percent best non-raw variant | F2 | 20.7604447502 | 0.2083798160 | -0.0788251242 | 20.7595164321 | Yes |
| S5-G blend 50 percent raw control plus 50 percent best non-raw variant | F2 | 20.5095173420 | 0.2047465276 | -0.0745773022 | 20.5094816811 | Yes |
| S5-G equal blend of variants within 0.50 WMAE of best single: tweedie_1_1, tweedie_1_3, poisson | F2 | 20.4729505116 | 0.2021365987 | -0.0615041051 | 20.4729505116 | Yes |
| S5-G blend 70 percent raw control plus 30 percent best non-raw variant | F3 | 18.6022260769 | 0.1896370836 | -0.0141506412 | 18.6017418185 | Yes |
| S5-G blend 50 percent raw control plus 50 percent best non-raw variant | F3 | 18.5475232079 | 0.1889853598 | -0.0148132744 | 18.5471853790 | Yes |
| S5-G equal blend of variants within 0.50 WMAE of best single: raw_l1, sqrt_l1, log1p_l1, tweedie_1_1, tweedie_1_3, poisson | F3 | 18.3230191397 | 0.1823866069 | 0.0004908816 | 18.3230191397 | Yes |
| S5-G blend 70 percent raw control plus 30 percent best non-raw variant | F4 | 18.1672507404 | 0.1919454179 | -0.0661663967 | 18.1665425153 | Yes |
| S5-G blend 50 percent raw control plus 50 percent best non-raw variant | F4 | 18.0111411610 | 0.1896127625 | -0.0643926088 | 18.0106640786 | Yes |
| S5-G equal blend of variants within 0.50 WMAE of best single: sqrt_l1, log1p_l1, tweedie_1_1, tweedie_1_3, poisson | F4 | 17.7209781384 | 0.1839008849 | -0.0465949085 | 17.7209781384 | Yes |

## Aggregate comparison

| Item | Mean WMAE | Mean WAPE | Mean bias |
|---|---:|---:|---:|
| raw_l1 | 19.5424424315 | 0.2027897892 | -0.0472425016 |
| sqrt_l1 | 19.3595468048 | 0.2010934456 | -0.0494941989 |
| log1p_l1 | 19.3080560058 | 0.1990826432 | -0.0445688066 |
| tweedie_1_1 | 19.1888923539 | 0.1958707021 | -0.0137296116 |
| tweedie_1_3 | 19.2458478281 | 0.1967545508 | -0.0155743556 |
| poisson | 19.4601334855 | 0.1984511040 | -0.0165099164 |
| S5-G blend 70 percent raw control plus 30 percent best non-raw variant | 19.2586753710 | 0.1991718991 | -0.0453981718 |
| S5-G blend 50 percent raw control plus 50 percent best non-raw variant | 19.1352184341 | 0.1975483016 | -0.0441686186 |
| S5-G equal blend of variants within 0.50 WMAE of best single: raw_l1, sqrt_l1, log1p_l1, tweedie_1_1, tweedie_1_3, poisson | 18.7703082979 | 0.1929383626 | -0.0000125293 |
| S5-G equal blend of variants within 0.50 WMAE of best single: tweedie_1_1, tweedie_1_3, poisson | 20.4729505116 | 0.2021365987 | -0.0615041051 |
| S5-G equal blend of variants within 0.50 WMAE of best single: sqrt_l1, log1p_l1, tweedie_1_1, tweedie_1_3, poisson | 17.7209781384 | 0.1839008849 | -0.0465949085 |

## Comparison to Stage 5-E

- Stage 5-E reference mean WMAE: 19.5424424315
- Raw L1 control mean WMAE: 19.5424424315
- Best single mean WMAE: 19.1888923539 (tweedie_1_1)
- Best non-raw single mean WMAE: 19.1888923539 (tweedie_1_1)
- Best blend mean WMAE: 17.7209781384 (S5-G equal blend of variants within 0.50 WMAE of best single: sqrt_l1, log1p_l1, tweedie_1_1, tweedie_1_3, poisson)

## Runtime and memory

- Raw load seconds: 14.081
- Peak RSS across folds and variants: 2376.49 MiB

## Leakage and cutoff safety assessment

- Stage 5-E feature contract remained unchanged.
- No future sales, total_orders, availability, or solution leakage entered the model matrix.
- Blends were predeclared and aligned exactly by key/order before scoring.

## Recommendation

A follow-up candidate is justified for human review.

