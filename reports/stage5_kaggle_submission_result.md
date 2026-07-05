# Stage 5 Kaggle Submission Result

## Submission details

- Command:

```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage5_s5b_price_discount_candidate.csv \
  -m "stage5 candidate submission: approved s5b price discount candidate"
```

- Submission ref: `54374863`
- File name: `stage5_s5b_price_discount_candidate.csv`
- Timestamp: `2026-07-05 22:14:02.783000`
- Status: `SubmissionStatus.COMPLETE`

## Official scores

- Public WMAE: `21.99264`
- Private WMAE: `21.61114`

## Comparison to Stage 4

- Stage 4 public WMAE: `22.37834`
- Stage 4 private WMAE: `21.91884`
- Public improvement: `0.38570`
- Private improvement: `0.30770`

## Comparison to local validation

- Stage 5 S5-B local all-fold mean WMAE: `20.4635858813`

The local validation remained optimistic relative to the official Kaggle scores, but the improvement transferred in the correct direction and produced a materially better submission than Stage 4.

## Interpretation

- Stage 5 improved over the frozen Stage 4 submission on both public and private Kaggle scores.
- The local gain from the public-solution-informed relative price/discount features transferred directionally to Kaggle.
- The private score is now below the Stage 4 value by a meaningful margin.
- This is still not top-solution territory, but it is a credible improvement and a better official result than Stage 4.
- The improvement came from the S5-B relative price/discount feature batch plus the documented clipping post-processing.

## Caveats

- This is a public-solution-informed stage.
- Price/discount covariates are benchmark-specific Kaggle-known-future inputs.
- Clipping was applied as documented post-processing, affecting 29 predictions.
- No horizon-specific models or target transforms were used yet.
- Official Kaggle scores are the external benchmark; local scores are still optimistic.

## Recommendation

Freeze Stage 5 as the current best official result. Decide whether to package the public write-up now or approve heavier Stage 5-C / Stage 5-D experiments next.

