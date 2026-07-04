# Rohlik Forecasting Benchmark Run

This repository is a retrospective public benchmark using the Kaggle Rohlik Sales Forecasting Challenge V2. Its purpose is to produce a reproducible, measured, and auditable forecasting run. It is not a forecasting product build.

The work will report two metrics separately:

- Kaggle's official WMAE, used for the competition score.
- WAPE, used as a business-facing description of total absolute forecast error relative to total actual demand.

The workflow is agentic but supervised. A human reviews the evaluation logic, leakage controls, results, and any public claims. Leakage checks and time-based backtest discipline will be established before any model is built.

This benchmark does not claim to replace planners or data scientists, and it is not presented as a live competition against the original teams. It is a retrospective test of reproducibility, measurement, and workflow discipline on public data.

## Current status

Stage 1 implements the metric, official-grid alignment, rolling backtest splits, and leakage guards. The design is documented in `reports/backtest_design.md`. The `eval/` and `dataguard/` layers are pending human review and have not been frozen. No features or models have been implemented.

Run the unit tests with `python3 -m pytest -q` and the full raw-data split validation with `python3 -m scripts.validate_stage1`.

The permanent project contract is in [rohlik_spec.md](rohlik_spec.md).
