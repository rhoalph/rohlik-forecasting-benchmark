# Initial Risk Register

This register is intentionally conservative. Controls are proposals for Stage 1 and later; they are not yet implemented.

| ID | Risk | Likelihood | Impact | Initial control / decision |
|---|---|---|---|---|
| R01 | Future `sales` leaks into lags, rolling windows, aggregates, encodings, or fallback statistics. | High | Critical | All target-derived computation must receive an explicit forecast cutoff and assert `source_date <= cutoff`. |
| R02 | Train/test contamination occurs when files are concatenated before fitting transforms or aggregations. | Medium | Critical | Permit concatenation only for target-independent schema/calendar operations; fit all learned or target-derived state on cutoff-safe training rows. |
| R03 | Rolling windows include the current row or validation labels because shifting is omitted or sorting/grouping is wrong. | High | Critical | Sort by item/date, shift before rolling, test boundary rows, and audit per-origin lineage. |
| R04 | Product, warehouse, or category target aggregates use records after the forecast cutoff. | High | Critical | Build aggregates separately for each origin from cutoff-filtered history; never precompute them globally. |
| R05 | Future `total_orders` is used as though it were operationally known. It is present in test but may summarize realized future-day demand. | High | High | Maintain a strict track excluding future `total_orders` and a clearly labeled Kaggle-provided-covariate track only after human approval. |
| R06 | Test-supplied price or discounts are treated as scheduled without evidence, or validation uses realized values unavailable in practice. | Medium | High | Classify them as benchmark-known future; require a documented operational assumption before broader claims. |
| R07 | Same-day `availability` is used in validation although it is absent from test. | Medium | High | Treat availability as historical only; allow only cutoff-safe lagged values, with a missing/fallback policy at prediction time. |
| R08 | Calendar rows are joined on date alone, mixing countries/warehouses, or duplicate joins multiply rows. | Medium | High | Join on (`warehouse`, `date`), assert many-to-one cardinality and unchanged row count. |
| R09 | Validation is random, uses a horizon unlike the 14-day test, or lets later-origin data train earlier-origin forecasts. | High | Critical | Use multiple fixed 14-day forward windows with an explicit cutoff immediately before each window. |
| R10 | Sparse test-grid structure is ignored; absent item-date rows are incorrectly converted to zero demand. | Medium | High | Score requested/observed keys only and preserve a horizon-day mask; do not infer labels for absent rows. |
| R11 | Missing target rows are imputed silently and contaminate scoring/training. | Medium | Medium | Exclude 52 missing-label rows from label-based operations and log the policy. Do not interpret missing as zero. |
| R12 | Cached features built with one cutoff are reused for another origin. | Medium | Critical | Include source-data fingerprint, cutoff, feature version, and availability policy in cache keys; reject mismatches. |
| R13 | `test_weights.csv` is joined positionally or used as a predictive feature. | Low | High | Join weights by `unique_id`, assert complete scoring coverage, and keep weights inside the evaluation layer. |
| R14 | Submission IDs are reordered, duplicated, omitted, or parsed incorrectly. | Low | High | Assert exact set, uniqueness, row count, and required ordering against `solution.csv`. |
| R15 | Negative discount values or extreme prices are silently clipped as data errors. | Medium | Medium | Profile and document anomalies; change values only with evidence and a logged experiment. |
| R16 | Different warehouse start dates produce unfair history or cold-start behavior. | High | Medium | Report performance and coverage by warehouse; require cutoff-safe fallback behavior for short histories. |
| R17 | Public solution logic is copied without understanding its availability assumptions. | Medium | High | Require independent rationale, source attribution where applicable, one feature family per commit, leakage review, and logged validation impact. |
| R18 | Local WMAE diverges from Kaggle because metric normalization or weight joins are wrong. | Medium | Critical | Freeze a reviewed metric implementation, test hand-calculated cases, and investigate any score divergence. |
| R19 | A surprisingly strong score is accepted without leakage investigation. | Medium | Critical | Apply the suspicious-score rule and perform adversarial review before retaining the change. |
| R20 | Public statements over-claim autonomy, production readiness, planner replacement, or live competition. | Medium | High | Tie every claim to logged evidence and enforce the framing in `rohlik_spec.md`. Human approval is required. |

## Immediate review gates

Before Stage 1 is frozen, a human should decide:

1. Whether the primary backtest is the strict operational track or the Kaggle-provided-covariate track.
2. Whether future `total_orders` is admissible at all.
3. Whether price and discounts can be described as scheduled known-future inputs.
4. The exact official WMAE formula and normalization, checked against competition documentation.
5. The treatment of the 52 rows with missing `sales` and `total_orders`.
