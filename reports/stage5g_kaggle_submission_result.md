# Stage 5-G Kaggle Submission Result

## Submission details

- Submission ref: `54401053`
- File: `stage5g_fixed_50_50_raw_tweedie_candidate.csv`
- Timestamp: `2026-07-06 16:17:42.693000`
- Submission command:

```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage5g_fixed_50_50_raw_tweedie_candidate.csv \
  -m "stage5g candidate submission: fixed 50/50 raw + tweedie"
```

## Scores

- Public WMAE: `20.62022`
- Private WMAE: `20.14904`

## Comparison to prior official results

### Versus Stage 5-E

- Stage 5-E public WMAE: `21.09367`
- Stage 5-E private WMAE: `20.61497`
- Public improvement: `0.47345`
- Private improvement: `0.46593`

### Versus Stage 4

- Stage 4 public WMAE: `22.37834`
- Stage 4 private WMAE: `21.91884`
- Public improvement: `1.75812`
- Private improvement: `1.76980`

## Comparison to local validation

- Stage 5-G fixed 50/50 local mean WMAE: `19.0089170845`
- The local result was directionally optimistic, but the improvement transferred clearly to Kaggle.

## Interpretation

- Stage 5-G materially improved the official score.
- The fixed objective-diversity blend transferred to Kaggle.
- This is the current best official result.
- Improvement came from a globally fixed 50/50 raw L1 + Tweedie 1.1 blend.
- It did not depend on fold-specific selection, tuning, Optuna, horizon routing, or new features.

## Caveats

- Public-solution-informed stage.
- Price / discount benchmark dependency remains.
- Local improvement may not always transfer exactly to Kaggle.
- Kaggle validates the prediction file and hidden score, not the full methodology.

## Recommendation

- Freeze Stage 5-G as the current final benchmark result.
- Rank it against the downloaded leaderboard.
- Move to executive insights / packaging.
