# Stage 5 S5-A Clipping Diagnostic

## Purpose

Assess whether clipping negative predictions to zero changes local validation or the prepared Stage 4 candidate.

## Local fold comparison

| Fold | Unclipped WMAE | Clipped WMAE | Unclipped WAPE | Clipped WAPE | Unclipped bias | Clipped bias | Negatives before | Negatives after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 20.5827509351 | 20.5823932950 | 0.2186937932 | 0.2186925287 | -0.0153873760 | -0.0153861115 | 13 | 0 |
| F2 | 22.6716726491 | 22.6604366054 | 0.2293625889 | 0.2293387945 | -0.0953250067 | -0.0953012124 | 19 | 0 |
| F3 | 20.3217981649 | 20.3215123710 | 0.2015732309 | 0.2015723598 | -0.0141731654 | -0.0141722943 | 22 | 0 |
| F4 | 19.2090941439 | 19.2090941439 | 0.2041885773 | 0.2041885773 | -0.0728248588 | -0.0728248588 | 0 | 0 |

## Stage 4 candidate inspection

- rows: 47021
- negative predictions: 20
- min negative value: -26.20026228259746
- rows changed by clipping: 20
- min after clipping: 0.0
- max after clipping: 14486.535310110636
- mean after clipping: 111.69270325863495
- median after clipping: 42.32233805392728

## Runtime and memory

- Clipped-pass runtime (F1-F4): 15.226 min
- Peak RSS: 1878.68 MiB

## Assessment

Clipping is low-risk and only marginally beneficial: the full F1-F4 mean WMAE improves from 20.6963289733 to 20.6933591038, and the bias change is negligible. The candidate file has 20 negative predictions, all of which would move to zero under clipping. The unmodified path remains acceptable if explanation simplicity is preferred, but clipping is a safe diagnostic improvement.

## Submission status

No Kaggle submission was made for the clipped diagnostic.
