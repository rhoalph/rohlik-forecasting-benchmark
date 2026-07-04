# Backtest and Evaluation Design

Status: implemented and tested in Stage 1; **not frozen** pending human review.

## Adopted policy

This implementation follows the Kaggle-provided covariate policy selected for the benchmark:

- `sell_price_main` and `type_0_discount` through `type_6_discount` are known-future inputs because they are supplied in `sales_test.csv`.
- Future `total_orders` is explicitly excluded, even though Kaggle supplies it as a challenge covariate.
- `availability` is historical only because Kaggle states that it is unknown when the prediction is made and omits it from test.
- `sales` is the target. It may appear only in training history through a fold cutoff or in a physically separate validation-label frame.
- Calendar fields are known future.
- Inventory fields are static competition metadata.
- `weight` is evaluation only and cannot enter feature frames.

The central registry is `dataguard/availability.py`. Unknown fields fail closed: a new field must be classified before it can enter a forecast-window feature frame.

## Official metric definition

The authenticated Kaggle competition Evaluation page says submissions use weighted mean absolute error and links to scikit-learn's `mean_absolute_error`. `test_weights.csv` provides the evaluation weight by `unique_id`.

For requested row `i`, with inventory ID `u_i`, the implemented formula is:

$$
\mathrm{WMAE} =
\frac{\sum_i w_{u_i}\lvert y_i - \hat{y}_i\rvert}
     {\sum_i w_{u_i}}.
$$

The inventory weight is repeated over every requested test-grid row for that ID. Predictions are aligned by (`unique_id`, `date`), never by row position. Missing, extra, or duplicate prediction keys are rejected.

Sources:

- [Kaggle competition evaluation](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview/evaluation)
- [scikit-learn weighted mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Business metrics

WAPE is returned as a ratio:

$$
\mathrm{WAPE} =
\frac{\sum_i \lvert y_i - \hat{y}_i\rvert}
     {\sum_i \lvert y_i\rvert}.
$$

Multiply by 100 for percentage display. Competition sample weights are not applied to WAPE; actual demand volume supplies its weighting.

Bias is returned as a ratio:

$$
\mathrm{Bias} =
\frac{\sum_i (\hat{y}_i - y_i)}
     {\sum_i y_i}.
$$

Positive bias means over-forecasting; negative bias means under-forecasting.

All metrics reject empty, non-finite, or misaligned inputs. WMAE rejects negative weights and non-positive total weight. WAPE and bias reject zero denominators.

## Official grid contract

`eval/grid.py` builds the local evaluation grid from the official `sales_test.csv` and `test_weights.csv`:

1. Require unique (`unique_id`, `date`) test keys.
2. Verify that dates span horizon days 1 through 14 after 2024-06-02.
3. Preserve original test row order as `grid_order`.
4. Join weights many-to-one by `unique_id` and require complete coverage.
5. Verify `solution.csv` has exactly the same ordered IDs.
6. Shift the public (`unique_id`, horizon-day) mask to each local cutoff.

Raw-data validation confirmed:

- 47,021 official grid rows.
- 3,625 test IDs.
- All horizon days 1–14 present.
- Every test ID has a weight.
- `solution.csv` IDs and ordering exactly match the official test grid.

## Backtest folds

| Fold | Training cutoff | Validation window | Horizon |
|---|---|---|---:|
| F1 | 2024-05-19 | 2024-05-20 through 2024-06-02 | 14 days |
| F2 | 2024-05-05 | 2024-05-06 through 2024-05-19 | 14 days |
| F3 | 2024-04-21 | 2024-04-22 through 2024-05-05 | 14 days |
| F4 | 2024-04-07 | 2024-04-08 through 2024-04-21 | 14 days |

The windows are non-overlapping. For each fold:

- Training contains only rows with `date <= cutoff` and a non-null target.
- The official test (`unique_id`, horizon-day) mask is shifted to the local origin.
- Available historical `sales` values on those shifted keys become isolated labels.
- Validation features pass through the field-availability registry.
- Target, `total_orders`, `availability`, and weight are absent from validation features.
- Weight stays with validation labels for WMAE only.
- Training and validation keys must be disjoint.

The 52 historical rows with missing `sales` are explicitly excluded from training in all four folds. They are never interpreted as zero.

## Full-data split validation

| Fold | Training rows | Training IDs | Requested grid rows | Scored rows | Missing labels | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 3,960,048 | 5,390 | 47,021 | 44,212 | 2,809 | 94.03% |
| F2 | 3,912,496 | 5,380 | 47,021 | 43,433 | 3,588 | 92.37% |
| F3 | 3,864,436 | 5,371 | 47,021 | 42,794 | 4,227 | 91.01% |
| F4 | 3,816,573 | 5,356 | 47,021 | 42,035 | 4,986 | 89.40% |

The declining ID count reflects inventories that did not yet have history at older cutoffs. The missing-label counts include shifted grid keys absent from historical data and any null labels. They are dropped from local scoring rather than imputed.

## Leakage controls

`dataguard/cutoff.py` provides assertions and filters for:

- Historical rows at or before the cutoff.
- Validation dates strictly in cutoff+1 through cutoff+14.
- Target-derived lineage dates at or before cutoff.
- Physical target absence from feature frames.
- Disjoint train/validation keys.
- Valid, non-missing date values.

`eval/backtest.py` materializes labels and features separately and yields folds one at a time to limit memory use. Any unclassified future column stops materialization.

## Tests and reproducibility

Run unit tests:

```bash
python3 -m pytest -q
```

Current result: 28 tests passed.

Run the full raw-data contract check:

```bash
python3 -m scripts.validate_stage1
```

This validates the official grid, solution ordering, weights, fold boundaries, training cutoffs, missing targets, and per-fold coverage. It does not train or score a model.

## Issues and uncertainties before freeze

1. **Historical grid coverage is below 100%.** The shifted public test mask does not have an observed non-null label for every historical item-date combination. Local WMAE uses the exact Kaggle weighting formula and official mask structure on available rows, but each fold has fewer rows than an actual Kaggle submission. This can change the effective ID/weight mix between folds.
2. **Future `total_orders` policy intentionally differs from Kaggle availability.** Kaggle describes it as known for test as part of the challenge. Excluding it is the approved choice, but local scores may be weaker than solutions that use it.
3. **Known-future price and discounts are benchmark assumptions.** Their presence in test supports Kaggle use but does not prove equivalent operational availability outside this retrospective benchmark.
4. **Negative discounts remain untransformed.** Kaggle says negative discount values mean no discount. Stage 1 performs no feature transformation; Stage 3 or 4 must decide whether to map negatives to zero and log the change.
5. **No seasonal validation fold is included.** The four approved windows measure recent stability, not holiday or annual seasonality.
6. **The project directory is currently untracked inside a parent Git repository.** A dedicated repository/commit baseline is needed before `git_hash` can identify Stage 1 code unambiguously and before later one-feature-family-per-commit discipline.
7. **Evaluation equivalence cannot be empirically checked against hidden labels yet.** The formula and alignment match Kaggle's published definition, but final confirmation requires comparing a reproducible submission's local reconstruction, where possible, with its Kaggle score.

## Freeze recommendation

Do not freeze `eval/` or `dataguard/` yet. Human review should explicitly approve:

1. Dropping absent historical grid labels and accepting fold-specific coverage.
2. The metric formulas and denominator behavior.
3. Excluding future `total_orders` while retaining future prices and discounts.
4. The four fold dates without a seasonal fold.
5. Initializing a dedicated Git repository before the freeze commit.

After approval, add the protected-directory pre-commit guard, establish a reviewed freeze commit, and only then proceed to Stage 2 baselines.
