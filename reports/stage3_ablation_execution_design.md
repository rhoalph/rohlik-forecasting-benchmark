# Stage 3 F1 Ablation Execution Design

Date: 2026-07-05
Status: Implementation approach only; no ablations authorized or run

## First execution scope

Run exactly six F1 diagnostics, in this order:

1. A9 — reproduce the Stage 2 trailing 14-day baseline.
2. A10 — reproduce the Stage 2 ID/day-of-week median baseline.
3. A1 — LightGBM without price or discount features.
4. A2 — LightGBM with historical-demand-only features.
5. A3 — LightGBM with static-metadata-only features.
6. A4 — LightGBM with known-future plus static features and no historical demand.

Do not implement or run A5–A8 in this execution. Do not run F2–F4 or submit to Kaggle.

## Execution matrix

| ID | Training action | Exact feature set | Expected result row |
|---|---|---|---|
| A9 | No model training. Reuse the reviewed Stage 2 trailing 14-day baseline function and F1 split. | Existing observed-row trailing 14-day mean with last/global fallback. | `stage=stage_3_f1_ablation`, description `A9 reproduce Stage 2 F1 trailing 14-day baseline`, `kept=diagnostic_only` |
| A10 | No model training. Reuse the reviewed Stage 2 ID/day-of-week median function and F1 split. | Existing ID/day-of-week median with last/global fallback. | `stage=stage_3_f1_ablation`, description `A10 reproduce Stage 2 F1 ID-day-of-week median baseline`, `kept=diagnostic_only` |
| A1 | Train one fixed LightGBM model on all twelve approved origins. | Approved 36 features minus `sell_price_main` and `type_0_discount` through `type_6_discount`; 28 features. | `stage=stage_3_f1_ablation`, description `A1 LightGBM without price-discount features`, `kept=diagnostic_only` |
| A2 | Train one fixed LightGBM model on all twelve approved origins. | The 12 historical-only features listed in the committed ablation plan. | `stage=stage_3_f1_ablation`, description `A2 LightGBM historical-demand-only`, `kept=diagnostic_only` |
| A3 | Train one fixed LightGBM model on all twelve approved origins. | `unique_id`, `product_unique_id`, `warehouse`, and `L1_category_name_en` through `L4_category_name_en`; 7 static features. | `stage=stage_3_f1_ablation`, description `A3 LightGBM static-metadata-only`, `kept=diagnostic_only` |
| A4 | Train one fixed LightGBM model on all twelve approved origins. | The 7 static features plus all 17 known-future features; 24 features. No historical-demand column. | `stage=stage_3_f1_ablation`, description `A4 LightGBM known-future-plus-static`, `kept=diagnostic_only` |

Exactly six new `results.csv` rows are expected. Existing Stage 2 and Stage 3 reference rows must not be rewritten.

## Proposed implementation structure

- Add one non-protected ablation runner under `scripts/` and focused tests under `tests/` only after approval.
- Reuse `baselines.naive` directly for A9 and A10; do not reimplement either baseline.
- Reuse the committed Stage 3 feature builder to generate the exact twelve-origin full matrices once.
- Select explicit feature-name allowlists for A1–A4 from that same matrix. Assert each subset equals its committed contract and contains no forbidden field.
- Train A1–A4 sequentially with the unchanged raw-sales target, unweighted L1 objective, 300 rounds, learning rate, leaves, minimum leaf size, seeds, threads, and categorical mappings.
- Release each LightGBM model before fitting the next one. Do not select or blend models based on results.
- Score every prediction vector on the same frozen F1 label grid: 44,212 scored rows and 94.026073% coverage.

## Runtime and memory expectation

- A9 and A10 together should require one F1 split preparation plus seconds of baseline prediction and scoring.
- Shared twelve-origin Stage 3 feature generation previously required about 138 seconds.
- Each A1–A4 fit should take less than the 49-second full-model fit because it uses fewer columns; run them sequentially.
- Expected combined wall time is approximately 5–8 minutes, allowing for shared loading, feature generation, four fits, scoring, and cleanup.
- Peak RSS should remain near or below the 1.52 GiB Stage 3 reference if only one model is retained at a time. The existing 12 GiB stop limit remains active.

These estimates are planning bounds, not performance gates or optimization targets.

## Results and audit labeling

Every row and report section must state:

- F1 diagnostic ablation only;
- no hyperparameter tuning;
- no feature added;
- same frozen scoring and grid;
- not a Kaggle score or submission;
- not eligible to replace the trusted-with-caveats reference automatically.

Report WMAE, WAPE, bias, coverage, training rows, feature names, runtime, peak RSS, and delta from both the full Stage 3 F1 result and the relevant Stage 2 reference. Preserve regressions and unexpected outcomes.

## Protected-layer controls

- `eval/` and `dataguard/` remain read-only and unchanged.
- Use frozen backtest materialization, key alignment, official weights, WMAE, WAPE, and bias.
- Use frozen cutoff checks before creating historical features.
- Add all ablation selection and reporting logic outside protected directories.
- Run the full unit suite and `python3 -m scripts.validate_stage1` before and after the diagnostics.
- Stop if a feature subset, key set, coverage value, baseline reproduction, or protected-directory check differs from the committed contract.

The next human approval should authorize A9, A10, and A1–A4 explicitly before implementation or execution begins.
