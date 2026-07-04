# Stage 1 Protected-Layer Freeze Policy

Effective date: 2026-07-04  
Approval status: human-approved  
Protected paths: `eval/` and `dataguard/`

## Freeze declaration

The `eval/` and `dataguard/` directories are frozen following human review and approval of the Stage 1 freeze candidate.

These directories contain the benchmark's trusted control layer:

- Official Kaggle WMAE scoring and business diagnostic metrics.
- Official test-grid and weight alignment.
- Time-based backtest fold definitions and split materialization.
- Field-availability classification.
- Forecast-cutoff enforcement and future-target leakage guards.
- Assertions for target isolation, validation windows, source dates, and train/validation key separation.

Baseline protected implementation commit:

```text
a4de32354f43361607fa48cb8e1e00663979f0a4
```

Freeze approval is documented in `reports/stage1_freeze_approval.md`.

## Change-control requirement

Any future change to a file under `eval/` or `dataguard/` requires explicit human approval before the change is committed.

An approved protected-layer change must document all of the following:

1. Reason for the change.
2. Exact protected files changed.
3. Diff summary describing behavioral impact.
4. Full unit-test result from:

   ```bash
   python3 -m pytest -q
   ```

5. Full raw-data validation result from:

   ```bash
   python3 -m scripts.validate_stage1
   ```

6. Commit hash containing the approved change.

The approval record must be retained in `reports/` and must state who authorized the exception in the project workflow.

## Enforcement

The local Git pre-commit hook blocks staged changes under:

```text
eval/
dataguard/
```

The hook is installed at `.git/hooks/pre-commit`. Git hooks inside `.git/` are local repository metadata and are not tracked by Git. The tracked installer is:

```text
scripts/install_git_hooks.sh
```

Install or refresh the hook with:

```bash
scripts/install_git_hooks.sh --install
```

The only permitted override is:

```bash
ALLOW_PROTECTED_CHANGES=1 git commit ...
```

`ALLOW_PROTECTED_CHANGES=1` may be used only after explicit human approval. The override does not waive testing or documentation requirements.

## Pipeline gate

No baseline, model, or feature stage may proceed if any `eval/` or `dataguard/` test fails.

After any approved protected-layer change:

- The full unit suite must pass.
- The full raw-data Stage 1 validation must pass.
- Official grid and solution-order checks must pass.
- Future `total_orders` exclusion must pass.
- The protected change must be committed and documented before forecasting work resumes.

Bypassing the hook without approval, omitting validation, or continuing with failed protected-layer tests violates the project contract.
