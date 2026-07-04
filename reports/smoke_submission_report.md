# Stage 1.5 Infrastructure Smoke Submission

Date: 2026-07-04  
Competition: `rohlik-sales-forecasting-challenge-v2`  
Submission reference: `54338855`  
Status: `COMPLETE`

## Purpose

This was an explicitly authorized infrastructure smoke test using the untouched official zero template. It is not a model, baseline, forecasting result, or performance experiment.

The test was limited to confirming:

- Late-submission availability.
- Account submission permission.
- Submission schema acceptance.
- Kaggle scoring response.

## File submitted

```text
/home/tucan/rohlik-forecasting-benchmark/data/raw/solution.csv
```

File properties immediately before submission:

| Property | Verified value |
|---|---|
| Columns | `id`, `sales_hat` |
| Rows | 47,021 |
| Unique IDs | 47,021 |
| Missing values | 0 |
| Predictions | All `sales_hat` values exactly zero |
| Official grid/order | Exact match |
| File size | 837,252 bytes |
| SHA256 | `73d6ed98a980ed009f4357b2f6a552fd840ffc968a2c29adc41395bb433e3a93` |

The raw file was submitted in place. No copy, rewrite, reserialization, clipping, or prediction change occurred before upload.

## Command

```bash
kaggle competitions submit -c rohlik-sales-forecasting-challenge-v2 -f /home/tucan/rohlik-forecasting-benchmark/data/raw/solution.csv -m "stage1.5 infrastructure smoke test: untouched official zero template"
```

## Initial Kaggle response

The command exited with status 0 and returned:

```text
Successfully submitted to Rohlik Sales Forecasting Challenge
```

The first status lookup returned:

```json
{
  "ref": 54338855,
  "fileName": "solution.csv",
  "date": "2026-07-04T18:17:33.237000",
  "description": "stage1.5 infrastructure smoke test: untouched official zero template",
  "status": "SubmissionStatus.PENDING",
  "publicScore": "",
  "privateScore": ""
}
```

No second submission was attempted. The same submission record was polled again.

## Completed Kaggle response

```json
{
  "ref": 54338855,
  "fileName": "solution.csv",
  "date": "2026-07-04T18:17:33.237000",
  "description": "stage1.5 infrastructure smoke test: untouched official zero template",
  "status": "SubmissionStatus.COMPLETE",
  "publicScore": "80.38078",
  "privateScore": "81.29170"
}
```

## Result interpretation

| Result | Value |
|---|---:|
| Public Kaggle WMAE | 80.38078 |
| Final/private Kaggle WMAE | 81.29170 |

These values are recorded only to prove scoring integration. They must not be compared as a baseline or used as evidence of forecasting performance.

## What this proves

1. Late submissions are enabled for this competition as of the submission timestamp.
2. The configured Kaggle account is permitted to submit.
3. The accepted schema is `id,sales_hat` with the official 47,021-row ID grid.
4. The untouched official template is accepted as a valid submission file.
5. Kaggle returns both public and private scores for a late submission.
6. The final/private score is the relevant external benchmark for the accepted project score bands.

## What this does not prove

1. It does not establish any forecasting skill.
2. It does not validate a model, feature, baseline, or data-processing pipeline.
3. It does not provide local WAPE or bias because hidden test labels remain unavailable.
4. It does not prove that local backtest WMAE will equal Kaggle WMAE.
5. It does not reveal the hidden public/private row membership.
6. It does not justify any public performance claim.

## Audit handling

The submission is logged in `results.csv` as `stage1.5_smoke_submission` with `kept=audit_only`. Local metric columns remain `NA`; public/private Kaggle scores and completion status are stored in the notes field to avoid mislabeling an external score as local validation.
