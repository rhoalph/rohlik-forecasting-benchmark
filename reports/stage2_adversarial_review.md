# Stage 2 Adversarial Leakage and Correctness Review

Date: 2026-07-04  
Status: Complete; pending human review  
Scope: Stage 2 local naive-baseline diagnostics only

## Conclusion

The Stage 2 scores can be trusted for their stated purpose: local diagnostics on the available-label portion of the shifted official Kaggle grid. No future-sales leakage, target contamination, predictive use of evaluation weights, or use of future covariates was found.

A fresh read-only replay reproduced all 28 WMAE, WAPE, and bias rows in `results.csv` within the recorded precision. It also reproduced every scored-row count and coverage value. The 28 rounded metric rows in `reports/stage2_baseline_results.md` match `results.csv`.

No leakage or scoring fix is required before committing Stage 2. Stage 3 may proceed after the Stage 2 commit and human acceptance of this review. The low-risk audit observations below should remain visible; none invalidates the scores.

These scores are not Kaggle final/private scores and must not be interpreted using Kaggle leaderboard score bands.

## Files reviewed

- `baselines/naive.py`
- `scripts/run_stage2_baselines.py`
- `reports/stage2_baseline_results.md`
- `results.csv`
- `tests/test_naive_baselines.py`
- `eval/__init__.py`
- `eval/backtest.py`
- `eval/grid.py`
- `eval/metrics.py`
- `dataguard/__init__.py`
- `dataguard/availability.py`
- `dataguard/cutoff.py`
- Supporting frozen-layer tests under `tests/`

## Review methods and evidence

The review traced the data path from raw-column loading through fold materialization, baseline context construction, prediction, key alignment, and metric calculation. It also searched the Stage 2 implementation for file writes, submission calls, solution-template use, and future-covariate references.

Verification results:

- Fresh replay: 28 baseline/fold outputs.
- `results.csv` metric mismatches: 0.
- `results.csv` scored-row or coverage mismatches: 0.
- Report metric rows checked: 28.
- Report rounding mismatches: 0.
- Unit tests: 31 passed.
- Protected-directory diff against `HEAD`: clean.
- Stage 2 file-write calls: none.
- Kaggle submission calls: none.
- `solution.csv` use: none.
- `total_orders`, price, discount, and calendar-field use: none.

Runtime changed on replay, as expected. Runtime variability does not affect deterministic predictions or scores.

## Risk findings

### High risk

None found.

### Medium risk

None found.

### Low risk

#### L1: Fold coverage changes the scored population

Coverage declines from 94.03% in F1 to 89.40% in F4 because only shifted-grid rows with available labels are scored. Each fold therefore has a different row population and effective weight mix. This is correctly reported and does not create leakage, but fold scores and their macro average should not be interpreted as if every fold scored the same complete grid.

Action: no pre-commit fix required. Continue reporting coverage and scored rows with every local result.

#### L2: Same-weekday fallback reasons are aggregated

The `primary_fallback_rows` count combines unsafe `t-7` source dates and missing historical lookups. For horizons 8–14, `t-7` lies in the validation period and is deliberately rejected in favor of the last-observed/global fallback. The behavior is safe, but the count does not distinguish the two fallback causes.

Action: no pre-commit fix required. Preserve the documented interpretation; optionally split fallback-reason counters in a later non-protected diagnostics change.

#### L3: Direct baseline unit coverage is incomplete

`tests/test_naive_baselines.py` directly tests cutoff rejection, safe same-weekday behavior, the 7-day mean, and the ID/day-of-week path. It does not contain dedicated tests for zero, global median, last observed, 14-day mean, or an explicit missing-date observed-row example. Those implementations are short and were covered by the successful full-data replay, frozen alignment tests, finite-output checks, and static review.

Action: no correctness fix is required before commit. Adding focused tests later would improve regression protection without changing implementation behavior.

#### L4: Results-log Git lineage uses placeholders

The F1 rows retain `UNCOMMITTED_F1` even though the reviewed F1 work now has commit `05b269f5b84e6d384540b6bddbe341cc95f2c4d4`. F2–F4 correctly remain `UNCOMMITTED_ALL_FOLDS` before their commit. This is an audit-traceability issue only; it does not affect scores.

Action: resolve lineage explicitly as part of the Stage 2 commit/finalization workflow without rewriting metric history ambiguously. The final Stage 2 commit should be recorded in the audit trail.

## Fold-discipline assessment

The frozen default cutoffs are exactly:

| Fold | Cutoff | Inclusive validation window |
|---|---|---|
| F1 | 2024-05-19 | 2024-05-20 through 2024-06-02 |
| F2 | 2024-05-05 | 2024-05-06 through 2024-05-19 |
| F3 | 2024-04-21 | 2024-04-22 through 2024-05-05 |
| F4 | 2024-04-07 | 2024-04-08 through 2024-04-21 |

`materialize_backtest_split` filters training history to dates on or before each cutoff, drops missing training targets, and asserts the cutoff. It shifts the official test grid by horizon day, joins labels separately, removes rows without labels from the scored population, and asserts that training and validation keys are disjoint.

`prepare_context` receives only the filtered training history and label-free `unique_id`/`date` validation keys. It independently reasserts the training cutoff, validation window, duplicate-key absence, missing-target absence, and target-column absence. Global and last-observed fallbacks are computed from this filtered history.

Validation labels exist only in the separate `labels` frame passed to frozen scoring. Baseline functions receive no reference to that frame.

## Baseline-by-baseline trust assessment

| Baseline | Assessment | Leakage/correctness rationale |
|---|---|---|
| Zero forecast | Trusted | Creates a constant zero vector from validation-key length; reads no sales or covariates. |
| Global median | Trusted | Uses the median of non-null training sales after cutoff filtering. Validation labels are unavailable to the function. |
| Last observed by ID | Trusted | Sorts filtered training rows by ID/date and selects the last row on or before cutoff. Missing IDs fall back to the cutoff-safe global median. |
| Same weekday last week | Trusted with semantic caveat | Uses literal `t-7` only when that source date is on or before cutoff and present in training. Unsafe or unavailable lookups fall back to last observed, then global median. Horizons 8–14 are therefore mostly fallback forecasts, not literal last-week forecasts. |
| Trailing 7-day mean | Trusted | Filters training to cutoff-6 through cutoff and averages non-null observed rows by ID. Validation rows cannot enter the window. |
| Trailing 14-day mean | Trusted | Filters training to cutoff-13 through cutoff and averages non-null observed rows by ID. Validation rows cannot enter the window. |
| ID/day-of-week median | Trusted | Computes ID/weekday medians using all non-null training history through cutoff only, then uses cutoff-safe fallbacks. Its negative bias is a model-property diagnostic, not evidence of leakage. |

For both trailing means, absent item-date rows are absent observations and are not filled with zero. Missing targets were removed before context construction and are never converted to zero.

## Feature and covariate isolation

The runner reads these columns only:

- `sales_train.csv`: `unique_id`, `date`, `sales`
- `sales_test.csv`: `unique_id`, `date`
- `test_weights.csv`: `unique_id`, `weight`

`sales_test.csv` supplies only the official request mask and horizon-day structure. Its dates are not used as sales evidence, and no test covariates are loaded. `solution.csv` is not loaded.

Future `total_orders`, price, discount, calendar, inventory, and availability fields are not loaded or referenced. Test weights are attached to the shifted official grid and passed only with isolated labels into scoring. Baseline contexts contain no weight column.

## Scoring and alignment

The official grid is built from the test ID/date mask, preserves grid order, validates the 14 horizon days, and joins one official weight per `unique_id` with a many-to-one constraint. The grid is shifted to each local cutoff by `horizon_day`.

Only shifted-grid rows with non-null historical labels are retained for scoring. Coverage is calculated as scored rows divided by the 47,021 requested rows:

| Fold | Scored rows | Coverage |
|---|---:|---:|
| F1 | 44,212 | 94.03% |
| F2 | 43,433 | 92.37% |
| F3 | 42,794 | 91.01% |
| F4 | 42,035 | 89.40% |

Frozen scoring normalizes keys, rejects duplicate label or prediction keys, performs an outer one-to-one merge, and rejects any missing or extra prediction key. Official per-ID weights are repeated over that ID's scored rows and used only in WMAE. WAPE and bias intentionally remain unweighted business diagnostics.

The fresh replay confirmed that `results.csv` metrics, scored rows, and coverage match the executable output. The Markdown report's displayed metrics match the logged values at five-decimal rounding.

## Code hygiene and test assessment

Stage 2 logic is confined to `baselines/` and `scripts/`, outside protected directories. `git diff --quiet HEAD -- eval dataguard` returned success, confirming no protected changes relative to the reviewed F1 `HEAD`.

The tests are meaningful rather than purely superficial:

- cutoff tests reject post-origin historical rows and target-bearing feature frames;
- split tests verify approved dates, isolated labels, missing-label exclusion, unsafe-covariate exclusion, and disjoint keys;
- grid tests verify horizon shifting, weight repetition, key-based alignment, and missing-prediction rejection;
- metric tests verify formulas and reject invalid inputs;
- baseline tests verify the critical same-weekday horizon boundary and reject post-cutoff training rows.

The runner has no processed-data writes, model-artifact writes, hidden cache writes, Kaggle CLI calls, or submission calls. Its only output is JSON to standard output.

## Decision recommendation

- Trust Stage 2 scores: **yes**, as local available-label diagnostics.
- Leakage or scoring fixes required before commit: **no**.
- Audit cleanup recommended: **yes**, preserve final commit lineage and consider the low-risk test/counter improvements separately.
- Proceed to Stage 3 after Stage 2 commit and human acceptance: **yes**.
- Modify frozen `eval/` or `dataguard/`: **no**.

