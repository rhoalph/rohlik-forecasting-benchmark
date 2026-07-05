# Stage 4 Kaggle Submission Result

Date: 2026-07-05
Status: complete

## Submission details

- Submission command:

```bash
kaggle competitions submit \
  -c rohlik-sales-forecasting-challenge-v2 \
  -f submissions/stage4_plain_lgbm_candidate.csv \
  -m "stage4 candidate submission: approved plain lightgbm"
```

- Submission ref: `54371523`
- File name: `stage4_plain_lgbm_candidate.csv`
- Timestamp: `2026-07-05 18:53:37.297000`
- Status: `SubmissionStatus.COMPLETE`
- Public score: `22.37834`
- Private score: `21.91884`

## Comparison to smoke test

The Stage 1.5 infrastructure smoke submission used the untouched official zero template and returned:

- ref: `54338855`
- public WMAE: `80.38078`
- private WMAE: `81.29170`

Relative to that smoke test, the official candidate improved by:

- public WMAE: `-58.00244`
- private WMAE: `-59.37286`

This confirms the final submission was valid and materially better than the infrastructure placeholder.

## Comparison to local validation

The local Stage 3 all-fold mean WMAE was `20.6963289733`.

Compared with that local mean:

- Kaggle public score was higher by `1.6820110267`
- Kaggle private score was higher by `1.2225110267`

Interpretation: local validation was optimistic relative to Kaggle, but still directionally useful and substantially better than the Stage 2 baselines.

## Interpretation

This is the first external proof point for the audited Stage 4 model submission.

- The submission is official and valid.
- The local validation pipeline was useful but optimistic versus Kaggle.
- The result is a credible first audited model submission.
- This is not a leaderboard-chasing claim and not a top-solution claim.

## Caveats

- Price/discount covariates are benchmark-specific Kaggle-known-future inputs.
- No clipping was applied despite 20 negative predictions in the candidate.
- All local folds triggered the suspicious-improvement threshold, but that was reviewed and documented.
- F2 showed elevated negative bias in the diagnostic review.

## Recommendation

- Freeze the Stage 4 result.
- Optionally prepare a public artifact or write-up.
- Only pursue Stage 5 improvements if explicitly approved.
