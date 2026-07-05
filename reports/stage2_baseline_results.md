# Stage 2 All-Fold Naive Baseline Results

Date: 2026-07-04  
Status: F1–F4 complete; pending human review  
Scope: simple non-model local diagnostics

## Evaluation contract

All folds use the frozen `eval/` and `dataguard/` layers, the official shifted Kaggle test-grid mask, and official weights joined by `unique_id`. Only shifted-grid rows with available non-null historical labels are scored.

The baseline runner loads only:

- `unique_id`
- `date`
- float64 `sales`

It does not load `total_orders`, price, discounts, availability, inventory, calendar fields, or other future covariates. No processed data is written and no Kaggle submission is made.

WAPE and bias are ratios. Runtimes in the per-fold table cover prediction generation plus frozen metric alignment/scoring for that baseline; shared loading and fold preparation are reported separately.

## Coverage by fold

| Fold | Cutoff | Validation window | Training rows | Training IDs | Scored rows | Requested rows | Missing labels | Coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|
| F1 | 2024-05-19 | 2024-05-20–2024-06-02 | 3,960,048 | 5,390 | 44,212 | 47,021 | 2,809 | 94.03% |
| F2 | 2024-05-05 | 2024-05-06–2024-05-19 | 3,912,496 | 5,380 | 43,433 | 47,021 | 3,588 | 92.37% |
| F3 | 2024-04-21 | 2024-04-22–2024-05-05 | 3,864,436 | 5,371 | 42,794 | 47,021 | 4,227 | 91.01% |
| F4 | 2024-04-07 | 2024-04-08–2024-04-21 | 3,816,573 | 5,356 | 42,035 | 47,021 | 4,986 | 89.40% |

Every fold excludes the same 52 missing-target rows from training.

## Per-fold results

| Fold | Baseline | WMAE | WAPE | Bias | Runtime (s) | Scored rows | Coverage | Primary fallbacks | Global fallbacks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | Zero | 77.27816 | 1.00000 | -1.00000 | 0.184 | 44,212 | 94.03% | 0 | 0 |
| F1 | Global median | 58.46590 | 0.81943 | -0.65855 | 0.207 | 44,212 | 94.03% | 0 | 0 |
| F1 | Last observed | 41.05448 | 0.40151 | -0.13466 | 0.179 | 44,212 | 94.03% | 0 | 0 |
| F1 | Same weekday last week | 37.75263 | 0.36385 | -0.05253 | 2.182 | 44,212 | 94.03% | 23,102 | 0 |
| F1 | Trailing 7-day mean | 32.14859 | 0.31139 | 0.02649 | 0.219 | 44,212 | 94.03% | 594 | 0 |
| F1 | Trailing 14-day mean | 31.19858 | 0.30211 | 0.03040 | 0.252 | 44,212 | 94.03% | 537 | 0 |
| F1 | ID/day-of-week median | 30.71502 | 0.34111 | -0.19291 | 1.159 | 44,212 | 94.03% | 36 | 0 |
| F2 | Zero | 81.34003 | 1.00000 | -1.00000 | 0.197 | 43,433 | 92.37% | 0 | 0 |
| F2 | Global median | 61.25356 | 0.81663 | -0.67148 | 0.170 | 43,433 | 92.37% | 0 | 0 |
| F2 | Last observed | 40.08274 | 0.34733 | -0.14049 | 0.171 | 43,433 | 92.37% | 43 | 43 |
| F2 | Same weekday last week | 39.33035 | 0.33269 | -0.08647 | 1.223 | 43,433 | 92.37% | 22,396 | 43 |
| F2 | Trailing 7-day mean | 36.59245 | 0.30956 | -0.03258 | 0.206 | 43,433 | 92.37% | 605 | 43 |
| F2 | Trailing 14-day mean | 32.94128 | 0.29535 | -0.04836 | 0.206 | 43,433 | 92.37% | 429 | 43 |
| F2 | ID/day-of-week median | 33.95448 | 0.35181 | -0.22341 | 1.412 | 43,433 | 92.37% | 64 | 43 |
| F3 | Zero | 77.10689 | 1.00000 | -1.00000 | 0.129 | 42,794 | 91.01% | 0 | 0 |
| F3 | Global median | 58.04184 | 0.81508 | -0.65745 | 0.122 | 42,794 | 91.01% | 0 | 0 |
| F3 | Last observed | 33.79834 | 0.32259 | -0.05513 | 0.122 | 42,794 | 91.01% | 47 | 47 |
| F3 | Same weekday last week | 32.30534 | 0.30036 | -0.00835 | 1.132 | 42,794 | 91.01% | 21,939 | 47 |
| F3 | Trailing 7-day mean | 30.44921 | 0.28715 | 0.03600 | 0.187 | 42,794 | 91.01% | 385 | 47 |
| F3 | Trailing 14-day mean | 30.17135 | 0.29699 | 0.04891 | 0.183 | 42,794 | 91.01% | 288 | 47 |
| F3 | ID/day-of-week median | 31.70350 | 0.33395 | -0.19227 | 0.940 | 42,794 | 91.01% | 95 | 47 |
| F4 | Zero | 78.52162 | 1.00000 | -1.00000 | 0.221 | 42,035 | 89.40% | 0 | 0 |
| F4 | Global median | 58.32213 | 0.81905 | -0.67486 | 0.235 | 42,035 | 89.40% | 0 | 0 |
| F4 | Last observed | 31.39747 | 0.32578 | -0.10908 | 0.236 | 42,035 | 89.40% | 96 | 96 |
| F4 | Same weekday last week | 30.59714 | 0.31469 | -0.08443 | 1.417 | 42,035 | 89.40% | 22,171 | 96 |
| F4 | Trailing 7-day mean | 27.81799 | 0.28711 | -0.06038 | 0.206 | 42,035 | 89.40% | 556 | 96 |
| F4 | Trailing 14-day mean | 27.94645 | 0.29524 | -0.06186 | 0.202 | 42,035 | 89.40% | 512 | 96 |
| F4 | ID/day-of-week median | 31.46102 | 0.35167 | -0.24043 | 1.338 | 42,035 | 89.40% | 114 | 96 |

F1 metrics reproduced the reviewed F1 run exactly. Runtime variation between runs is expected and does not affect scores.

## Macro averages and ranking

These are unweighted arithmetic means of the four fold-level metrics. They are not pooled-row scores.

| Rank | Baseline | Mean WMAE | Mean WAPE | Mean bias | Mean runtime (s) |
|---:|---|---:|---:|---:|---:|
| 1 | Trailing 14-day mean | 30.56442 | 0.29742 | -0.00773 | 0.211 |
| 2 | Trailing 7-day mean | 31.75206 | 0.29880 | -0.00762 | 0.204 |
| 3 | ID/day-of-week median | 31.95851 | 0.34463 | -0.21225 | 1.212 |
| 4 | Same weekday last week | 34.99636 | 0.32790 | -0.05794 | 1.488 |
| 5 | Last observed | 36.58326 | 0.34930 | -0.10984 | 0.177 |
| 6 | Global median | 59.02086 | 0.81755 | -0.66558 | 0.183 |
| 7 | Zero | 78.56167 | 1.00000 | -1.00000 | 0.183 |

## Business-language bias interpretation

Positive bias means systematic over-forecasting; negative bias means systematic under-forecasting.

- Trailing 14-day and 7-day means are nearly balanced on average, with approximately -0.77% and -0.76% bias respectively. Their fold-level bias still moves between over- and under-forecasting.
- Same-weekday last-week averages approximately -5.79% bias, indicating moderate under-forecasting.
- Last observed averages approximately -10.98% bias.
- ID/day-of-week median averages approximately -21.23% bias despite competitive WMAE on F1; it systematically forecasts too little demand across folds.
- Global median and zero forecasts strongly under-forecast and serve only as diagnostic floors.

Bias does not replace WMAE. It explains the direction of errors and must be reviewed alongside absolute-error metrics.

## Fallback behavior

| Baseline | F1 primary/global | F2 primary/global | F3 primary/global | F4 primary/global |
|---|---:|---:|---:|---:|
| Last observed | 0 / 0 | 43 / 43 | 47 / 47 | 96 / 96 |
| Same weekday last week | 23,102 / 0 | 22,396 / 43 | 21,939 / 47 | 22,171 / 96 |
| Trailing 7-day mean | 594 / 0 | 605 / 43 | 385 / 47 | 556 / 96 |
| Trailing 14-day mean | 537 / 0 | 429 / 43 | 288 / 47 | 512 / 96 |
| ID/day-of-week median | 36 / 0 | 64 / 43 | 95 / 47 | 114 / 96 |

For same-weekday last-week, literal `t-7` is used only when the source date is on or before the fold cutoff and exists in history. Horizons 8–14 generally cross into the validation period, so they fall back to last observed sales rather than reading future actuals.

Rolling means average observed rows inside the final 7 or 14 calendar days. Absent item-date rows are not treated as zero. Missing window/group estimates fall back to last observed sales and then the global median.

Global fallback counts increase in older folds because some requested IDs have no history at those earlier cutoffs.

## Runtime and memory

| Measurement | Result |
|---|---:|
| Shared raw loading and official-grid construction | 6.553 seconds |
| F1 preparation / total | 11.557 / 16.305 seconds |
| F2 preparation / total | 9.741 / 13.741 seconds |
| F3 preparation / total | 10.950 / 14.087 seconds |
| F4 preparation / total | 9.075 / 13.340 seconds |
| Complete all-fold wall time | 64.264 seconds |
| Peak process RSS | 888.58 MiB |

The sequential full-data run is well within hack-box memory. CPU and repeated fold preparation dominate wall time; baseline calculations themselves are small.

## Stage 2 reference baseline recommendation

Use **trailing 14-day mean by unique_id with last-observed/global fallback** as the Stage 2 reference baseline.

Reasons:

- Lowest macro-average WMAE: 30.56442.
- Lowest macro-average WAPE: 0.29742 among the evaluated baselines.
- Near-neutral macro bias: -0.00773.
- Simple cutoff-safe definition.
- Low mean per-fold runtime: approximately 0.211 seconds after preparation.
- More consistent all-fold behavior than the ID/day-of-week median, which has substantial negative bias.

This recommendation identifies a local diagnostic reference only. It is not a Kaggle score or a performance claim.

## Correctness and leakage notes

Checks and controls:

- Frozen fold cutoffs and validation windows are used.
- Training rows are restricted to each cutoff.
- Missing targets are excluded.
- Official weights are aligned by ID.
- Prediction keys are validated by frozen `eval/` for every baseline/fold.
- Validation targets remain separate from baseline inputs.
- No future covariates are loaded.
- Future `total_orders` is not loaded or used.
- Same-weekday source dates are cutoff-asserted.
- No validation-period actual is used recursively.
- No processed data is written and no Kaggle submission is made.

Remaining interpretation concerns:

1. Fold coverage declines from 94.03% to 89.40%, so each fold scores a different available subset and effective weight mix.
2. Same-weekday last-week is mostly a last-observed fallback in horizons 8–14; using literal `t-7` actuals there would be leakage.
3. Rolling means use observed rows rather than a dense zero-filled demand calendar.
4. ID/day-of-week medians may reflect stale historical regimes.
5. Macro averages weight folds equally, not individual scored rows.
6. These local metrics cannot be interpreted using final/private Kaggle score bands.

No correctness or leakage issue was found that invalidates the baseline results. The work stops here pending human review.
