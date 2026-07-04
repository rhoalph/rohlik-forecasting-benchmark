# Stage 1 Freeze Approval

Approval date: 2026-07-04  
Decision: approved for freeze with Git hook enforcement  
Protected paths: `eval/` and `dataguard/`

## Human approval

The human reviewer approved the Stage 1 evaluation and data-guard logic for freeze after reviewing the Stage 1 freeze candidate and Stage 1.5 benchmark intelligence.

The approved protected layer contains official scoring, grid alignment, backtest split logic, field-availability policy, cutoff enforcement, and leakage guards. Any later protected-layer change requires a new explicit approval and the evidence specified in `reports/freeze_policy.md`.

## Git references

| Reference | Commit |
|---|---|
| Stage 1 protected implementation baseline | `a4de32354f43361607fa48cb8e1e00663979f0a4` |
| Current HEAD immediately before freeze work | `315284315adda669d1bb760dec1a97e4dd01119f` |

## Infrastructure smoke submission

| Field | Value |
|---|---|
| Submission reference | `54338855` |
| Public WMAE | `80.38078` |
| Final/private WMAE | `81.29170` |
| Status | `COMPLETE` |

The smoke submission used the untouched official all-zero template. It is infrastructure-only and is not a baseline, model result, forecasting result, or performance claim.

It confirmed late-submission availability, account permission, accepted schema/grid, and Kaggle scoring response.

## Approved validation evidence

| Check | Approved result |
|---|---|
| Unit tests | 28 passed |
| Official grid validation | Passed: 47,021 rows, 3,625 IDs, horizon days 1–14 |
| Solution ordering validation | Passed: exact official order |
| Future `total_orders` exclusion | Passed |
| Price/discount Kaggle-known-future policy | Passed |
| Four-fold raw-data validation | Passed |
| Protected-directory diff from baseline | None before freeze documentation |

The price and discount policy is benchmark-specific. It does not establish operational availability outside the Kaggle competition.

## Four-fold coverage

| Fold | Cutoff | Validation period | Scored rows | Requested rows | Coverage |
|---|---|---|---:|---:|---:|
| F1 | 2024-05-19 | 2024-05-20–2024-06-02 | 44,212 | 47,021 | 94.03% |
| F2 | 2024-05-05 | 2024-05-06–2024-05-19 | 43,433 | 47,021 | 92.37% |
| F3 | 2024-04-21 | 2024-04-22–2024-05-05 | 42,794 | 47,021 | 91.01% |
| F4 | 2024-04-07 | 2024-04-08–2024-04-21 | 42,035 | 47,021 | 89.40% |

## Accepted final/private score bands

| Band | Final/private Kaggle WMAE | Use |
|---|---:|---|
| Minimum publishable | ≤ 20.25 | Project threshold, subject to baseline and honesty controls |
| Strong | ≤ 19.00 | Historical comparison band |
| Excellent | ≤ 18.10 | Historical comparison band |
| Elite/podium context | ≤ 17.50 | Reference only; not a project target |

The official Kaggle WMAE is the external benchmark metric. WAPE and bias remain business diagnostics only.

## Remaining accepted risks

1. Local fold coverage is incomplete.
2. Hidden public/private split membership is unavailable.
3. Local scores may be optimistic relative to Kaggle scores.
4. No seasonal stress fold exists yet.
5. Price and discount known-future status is benchmark-specific.
6. Excluding future `total_orders` is stricter than some public high-ranking solutions.

These limitations are documented and do not invalidate the approved scoring or leakage-control implementation. They must remain visible in future reports and score interpretation.

## Freeze enforcement decision

The freeze is enforced by a local `.git/hooks/pre-commit` hook installed from `scripts/install_git_hooks.sh`.

The hook blocks staged changes under `eval/` and `dataguard/` unless:

```text
ALLOW_PROTECTED_CHANGES=1
```

is set. This override is permitted only with explicit human approval and does not remove the requirement to rerun the full unit suite and raw-data validation.

### Enforcement test

The installed hook was tested with a temporary staged comment in `eval/metrics.py`:

- A normal `git commit` attempt was blocked with exit status 1 and identified `eval/metrics.py` as protected.
- Running the same hook state with `ALLOW_PROTECTED_CHANGES=1` returned exit status 0 and printed the approval/testing warning.
- No protected test commit was created.
- The temporary marker was removed and the index refreshed.
- `eval/` and `dataguard/` matched `HEAD` after the test.
- `eval/metrics.py` returned to approved SHA256 `7df60530a59bf3cfe459d835ec411e9248b179b38ec3a34270c18c53094e2ce8`.

### Post-install revalidation

After the hook test was reverted:

- `python3 -m pytest -q` passed all 28 tests in 1.28 seconds.
- `python3 -m scripts.validate_stage1` completed with exit status 0.
- The official grid remained 47,021 rows across horizon days 1–14.
- The solution template remained an exact ordered match.
- Four-fold coverage remained 94.03%, 92.37%, 91.01%, and 89.40%.
- All folds continued to exclude 52 missing-target training rows.
- `eval/` and `dataguard/` remained unchanged.

### Audit-bundle correction

Future Stage 1 freeze audit bundles must include `scripts/validate_stage1.py`. They should also include the installed `.git/hooks/pre-commit` when packaging local repository metadata and the tracked `scripts/install_git_hooks.sh` so enforcement can be reproduced after cloning.

No Stage 2 baseline, model, or feature work is authorized by this approval.
