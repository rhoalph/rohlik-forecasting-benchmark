# Stage 5-E Kaggle Submission Result

Submission command context:
```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage5e_stronger_features_candidate.csv \
  -m "stage5e candidate submission: approved stronger feature candidate"
```

Submission ref: `54395405`
File name: `stage5e_stronger_features_candidate.csv`
Timestamp: `2026-07-06 12:57:53.943000Z`

## Scores

- Public WMAE: `21.09367`
- Private WMAE: `20.61497`

## Comparisons

- Versus Stage 5 S5-B official result (`21.99264` public, `21.61114` private):
  - Public improvement: `0.89897`
  - Private improvement: `0.99617`
- Versus Stage 4 official result (`22.37834` public, `21.91884` private):
  - Public improvement: `1.28467`
  - Private improvement: `1.30387`
- Versus Stage 5-E local validation mean WMAE `19.5424424315`:
  - Kaggle public score was higher than local validation, but the direction of improvement transferred strongly.

## Interpretation

- Stage 5-E materially improved the official score.
- Local improvement transferred strongly to Kaggle.
- This is the current best official result.
- It is still not top-solution territory.
- The improvement came from richer cutoff-safe feature engineering, not tuning, ensembling, target transform, or horizon routing.

## Caveats

- This is public-solution-informed work, not independent discovery.
- Price/discount dependence remains benchmark-specific.
- Kaggle validates the prediction file and hidden-score result, not the full methodology.
- Method credibility here comes from frozen `eval/` and `dataguard/`, tests, audits, logs, and documented leakage controls.
