# Stage 1 Freeze Candidate

Date: 2026-07-04  
Status: ready for human-approved freeze; not yet frozen  
Dedicated repository root: `/home/tucan/rohlik-forecasting-benchmark`

## Git baseline

| Field | Value |
|---|---|
| Baseline commit | `a4de32354f43361607fa48cb8e1e00663979f0a4` |
| Baseline commit message | `Establish Stage 1 evaluation baseline` |
| Branch | `main` |
| Raw data tracked | No |
| Processed data tracked | No; directory placeholder only |
| Generated submissions tracked | No; directory placeholder only |

The dedicated repository was initialized because the project had previously been an untracked directory inside the parent `/home/tucan` repository. `.gitignore` excludes `data/raw/`, generated processed data, generated submissions, archives, caches, local environments, and common secret-key files.

The committed baseline contains the specification, README, requirements, results log, reports, `eval/`, `dataguard/`, scripts, and tests. No raw competition CSV or downloaded archive is committed.

## Human decisions incorporated

1. Final/private Kaggle WMAE is the external benchmark metric.
2. WAPE and bias remain business diagnostics only.
3. Public solutions may inform independently implemented and reviewed ideas; their code may not be copied.
4. Future `total_orders` remains entirely excluded from forecast-window features.
5. Test-supplied price and discount fields are Kaggle-known-future covariates, with a benchmark-specific disclosure requirement.
6. One untouched-zero-template infrastructure smoke submission was authorized.

## Smoke submission status

| Field | Value |
|---|---|
| Submission reference | `54338855` |
| Timestamp | `2026-07-04T18:17:33.237000Z` |
| Description | `stage1.5 infrastructure smoke test: untouched official zero template` |
| File | `data/raw/solution.csv` |
| SHA256 | `73d6ed98a980ed009f4357b2f6a552fd840ffc968a2c29adc41395bb433e3a93` |
| Status | `COMPLETE` |
| Public WMAE | `80.38078` |
| Final/private WMAE | `81.29170` |

The file had exactly 47,021 rows, unique ordered IDs, columns `id,sales_hat`, no missing values, and all-zero predictions. Kaggle accepted and scored it. No second submission was attempted.

The returned scores are audit-only infrastructure outputs. They are not a baseline, model result, local metric, or performance claim.

The smoke submission proves that late submissions are enabled, this account may submit, the official template schema is accepted, and Kaggle returns public/private scores. It does not validate forecasting quality or reveal hidden labels/split membership.

Full details: `reports/smoke_submission_report.md`.

## Metric compatibility

Status: **compatible with the published Kaggle definition**.

The implemented official metric is:

$$
\mathrm{WMAE} =
\frac{\sum_i w_{u_i}\lvert y_i-\hat{y}_i\rvert}
     {\sum_i w_{u_i}}.
$$

Compatibility checks:

- Weights are joined by `unique_id` and repeated over requested item-date rows.
- Predictions and labels are aligned by (`unique_id`, `date`).
- Duplicate, missing, and extra prediction keys are rejected.
- Missing, NaN, and infinite numeric inputs are rejected.
- No unrequested clipping is applied; Kaggle documents no mandatory zero clipping.
- `solution.csv` ordering is required to match the official template. This is stricter than documented key-based acceptance but safe.
- WAPE and bias are computed separately and do not affect WMAE.

The zero-template score cannot be locally reconstructed because hidden test targets and public/private row membership are unavailable. This is an external-data limitation, not a confirmed metric mismatch.

## Availability-policy verification

| Field group | Frozen-candidate policy | Verification |
|---|---|---|
| `sales` | Target; isolated from validation features | Passed |
| `total_orders` | Historical only; forbidden as future feature | Passed |
| `availability` | Historical only | Passed |
| `sell_price_main` | Kaggle-known future | Passed |
| `type_0_discount` … `type_6_discount` | Kaggle-known future | Passed |
| Inventory fields | Static metadata | Passed |
| Calendar fields | Known future | Passed |
| `weight` | Evaluation only | Passed |
| Unclassified fields | Fail closed | Passed unit tests |

The required disclosure remains: price and discount availability is benchmark-specific and does not prove operational availability outside Kaggle.

## Test and validation results

### Unit suite

Command:

```bash
python3 -m pytest -q
```

Result:

```text
28 passed in 1.58s
```

Coverage includes:

- Hand-calculated WMAE, WAPE, and bias.
- Invalid metric inputs and denominator handling.
- Official grid and weight joins.
- Exact solution-template IDs/order.
- Missing prediction rejection.
- Field classification and fail-closed behavior.
- Future-`total_orders` exclusion.
- Cutoff filtering and future-source-date rejection.
- Validation-window boundaries.
- Train/validation key disjointness.
- Physical target isolation.
- Four approved fold definitions.

### Official grid checks

| Check | Result |
|---|---|
| Grid rows | 47,021 |
| Grid IDs | 3,625 |
| Horizon | Days 1–14 |
| Weight coverage | Complete |
| Solution order | Exact match |
| Future `total_orders` excluded | Yes |
| Price included | Yes |
| All seven discounts included | Yes |
| `eval/` and `dataguard/` diff from baseline | None |

### Full-data split validation

Command:

```bash
python3 -m scripts.validate_stage1
```

Result: completed with exit status 0.

| Fold | Cutoff | Validation period | Training rows | Training IDs | Requested rows | Scored rows | Missing labels | Coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|
| F1 | 2024-05-19 | 2024-05-20–2024-06-02 | 3,960,048 | 5,390 | 47,021 | 44,212 | 2,809 | 94.03% |
| F2 | 2024-05-05 | 2024-05-06–2024-05-19 | 3,912,496 | 5,380 | 47,021 | 43,433 | 3,588 | 92.37% |
| F3 | 2024-04-21 | 2024-04-22–2024-05-05 | 3,864,436 | 5,371 | 47,021 | 42,794 | 4,227 | 91.01% |
| F4 | 2024-04-07 | 2024-04-08–2024-04-21 | 3,816,573 | 5,356 | 47,021 | 42,035 | 4,986 | 89.40% |

All folds explicitly exclude the same 52 historical rows with missing targets from training.

## Accepted final/private score bands

| Band | Final/private Kaggle WMAE | Interpretation |
|---|---:|---|
| Minimum publishable | ≤ 20.25 | Must also beat our own naive baselines and pass honesty controls |
| Strong | ≤ 19.00 | Approximately top 10% of the historical final leaderboard |
| Excellent | ≤ 18.10 | Approximately top 1.5% / top 12 historical teams |
| Elite/podium context | ≤ 17.50 | Reference only; not a project target |

These bands apply only to a returned final/private Kaggle score. They must not be applied directly to local WMAE.

## Remaining risks

1. **Incomplete local grid coverage:** Fold coverage is 89.40%–94.03%, so fold-specific weight and item mixes differ from Kaggle's full test grid.
2. **Hidden leaderboard subsets:** Public/private row membership is unavailable.
3. **Local calibration gap:** Public podium evidence reports local validation materially below Kaggle WMAE in some solutions.
4. **No empirical hidden-label metric reconstruction:** The smoke submission confirms scoring, not formula equality on hidden labels.
5. **Deliberate leaderboard trade-off:** Excluding future `total_orders` improves defensibility but makes our feature policy stricter than some high-ranking solutions.
6. **Benchmark-only future covariates:** Price/discount availability cannot be generalized to private operational systems.
7. **Recent-window emphasis:** The four folds do not include a separate seasonal/holiday stress fold.
8. **Freeze enforcement not installed:** `eval/` and `dataguard/` are still mutable until the approved pre-commit guard and freeze commit are created.

## Freeze recommendation

**Recommendation: `eval/` and `dataguard/` are ready for a human-approved freeze.**

Rationale:

- Human metric and availability-policy decisions are explicit.
- The exact published WMAE formula is implemented and tested.
- Official grid, weights, schema, and ordering are validated.
- The submission endpoint and scoring response are confirmed.
- Cutoff and field-availability guards fail closed.
- All tests and full-data validations pass.
- Protected directories have no changes relative to the dedicated baseline commit.
- Known limitations are documented and do not require changing the evaluation formula or leakage policy.

This report is a freeze candidate, not the freeze itself. The actual freeze should occur only after human approval and should:

1. Add the specified pre-commit protection for `eval/` and `dataguard/` with an explicit override mechanism.
2. Commit that guard and identify the freeze commit.
3. Require explicit human permission for later protected-directory changes.

No baseline, model, forecasting feature, or performance claim was created during this work.
