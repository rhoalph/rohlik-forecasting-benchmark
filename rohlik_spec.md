# Rohlik Forecasting Benchmark Run

Project Specification v1.0  
Scope: one clean benchmark run, not a product build

## 1. Purpose and Positioning

Produce a credible, reproducible demonstration that a real European retail forecasting problem can now be turned into a scored, auditable forecasting pipeline through an agentic coding workflow under human supervision.

The benchmark is the Kaggle **Rohlik Sales Forecasting Challenge V2**. The public story is not “We beat Kaggle” but “We revisited a real retail forecasting problem and measured how much of the work can now be automated, checked, and reproduced through an agentic workflow with human supervision.”

This is not a forecasting product, not a vendor demo, and not an unsupervised agent experiment. It is a public proof object: official Kaggle score, business‑readable WAPE, runtime, public notebook, leakage controls, results log, one‑page summary, and LinkedIn post.

## 2. Benchmark Selection

**Competition**: Rohlik Sales Forecasting Challenge V2  
**Platform**: Kaggle  
**Task**: predict future sales for selected warehouse–inventory–date combinations.  
**Business context**: European e‑grocery demand forecasting.  
**Forecast horizon**: short‑term retail demand forecasting (approximately two weeks).  
**Official metric**: **WMAE** (weighted mean absolute error).  
**Business metric added by us**: **WAPE** (weighted absolute percentage error).  
**Validation**: Kaggle late submission against the hidden test set.

Rohlik is chosen because it is a European retail dataset with public solutions and reflects realistic operational forecasting with promotions, holidays, perishables, and sparse demand.

## 3. Definition of Done

The run is complete only when:

1. The Kaggle late‑submission score is recorded.
2. The final submission is generated from a single reproducible Kaggle notebook.
3. The official WMAE score is captured.
4. The business‑facing WAPE is calculated locally.
5. The runtime is recorded from raw Kaggle data to final submission file.
6. A results log is maintained across iterations.
7. A public notebook is prepared with pipeline, metric definitions, validation method, leakage notes, hardware/runtime notes, and reproducibility instructions.
8. A one‑page business summary is prepared.
9. A LinkedIn post is prepared in plain English for a second‑language audience.

## 4. Public Claim Discipline

### Allowed claims

- We revisited a real European retail forecasting problem.
- The workflow produced a reproducible scored forecast.
- The official Kaggle WMAE was **X**.
- The business‑readable WAPE was **Y**.
- The run took **Z** minutes/hours on Kaggle hardware.
- The workflow was agentic but supervised.
- This exercise shows how the economics of forecasting work are changing.

### Forbidden claims

- “AI beat data scientists.”
- “AI solved forecasting.”
- “This replaces planners.”
- “This proves the same result on private data.”
- “This is a production‑ready product.”
- “This was a fair live competition against the original teams.”
- “Fully autonomous end‑to‑end forecasting.”

### Correct framing

This was a retrospective public benchmark. The data, leaderboard, and public solutions exist. That is what makes it useful: it lets us test whether a workflow is reproducible, measurable, and honest.

## 5. Metrics

### 5.1 Official Metric: WMAE

Implement WMAE according to the competition definition. WMAE is the external proof point. Compare the local WMAE to the Kaggle score and investigate any divergence.

### 5.2 Business Metric: WAPE

Weighted absolute percentage error is defined as:

$$
\mathrm{WAPE} = \frac{\sum |\text{actual} - \text{forecast}|}{\sum |\text{actual}|}.
$$

Report WAPE as a percentage and explain it in business terms: a WAPE of 12 % means the total absolute forecast error is equal to 12 % of total actual demand.

### 5.3 Bias

Bias is defined as:

$$
\mathrm{Bias} = \frac{\sum (\text{forecast} - \text{actual})}{\sum \text{actual}}.
$$

Positive bias indicates over‑forecasting; negative bias indicates under‑forecasting.

### 5.4 Runtime

Measure the runtime from data loading through preprocessing, feature generation, model training, prediction, and submission file creation. Do not include manual notebook editing time in the runtime. Report the final runtime as “raw data to submission file: X minutes on Kaggle notebook hardware.”

## 6. Honesty Rules

### 6.1 Forecast‑Origin Discipline

Every feature used for a prediction must be available at the time the forecast would have been made. No feature may use future sales or target‑derived information from the prediction window.

### 6.2 Known‑Future Inputs

Classify every feature as one of the following:

- **Historical only** – Available only up to the forecast origin.
- **Known future** – Legitimately known in advance (e.g. calendar dates, scheduled promotions if provided in test set).
- **Static metadata** – Product/warehouse attributes that do not vary with time.
- **Forbidden / leakage risk** – Contains future target information or cannot be known at forecast time.

All price or promotion features must be classified as historical or known future. Future sales and other target‑derived features are forbidden.

### 6.3 Frozen Evaluation Layer

Create and freeze the following directories:

- **/eval** – Contains WMAE, WAPE, and bias implementations; backtest split generator; score reporting.
- **/dataguard** – Contains safe data access functions; cutoff‑date filtering; feature availability checks; assertions preventing future target leakage.

After human review and approval, do not modify **/eval** or **/dataguard** without explicit permission. Use a pre‑commit hook to reject commits touching these directories unless an override flag is set.

### 6.4 Results Log

Maintain a **results.csv** file with a row for every meaningful change. Required columns: `timestamp`, `git_hash`, `stage`, `change_description`, `local_wmae`, `local_wape`, `local_bias`, `runtime_minutes`, `kept`, `notes`.

### 6.5 Adversarial Review

After each major feature phase, run a separate review prompt to examine the codebase for any place where future information could leak into training, validation, or test predictions. Save findings in `/reviews/leakage_review_YYYYMMDD.md` and address any issues before proceeding.

### 6.6 Suspicious Score Rule

If a score improves dramatically without a clear reason, treat it as a leakage alarm, not a success. Investigate date joins, rolling windows, target encodings, aggregation cutoffs, validation split contamination, price or promotion features, and any known future fields.

## 7. Environment and Division of Labor

**Implementation tool:** Codex (the agent) writes code, tests, scripts, and notebooks.  
**Human role:** Review diffs; approve evaluation logic, leakage rules, and public claims; decide which results are publishable.  
Other models may be used only as reviewers or critics.

**Development environment:** Primary development may happen locally or in a Codespace. The final run must happen in Kaggle Notebooks using the official competition data.

## 8. Pipeline Stages

### Stage 0: Data Understanding and Contract

Goal: Load all Kaggle files; document tables, columns, row counts, date ranges, missing values, ID columns, target column, forecast horizon, static metadata fields, historical fields, known‑future fields, and leakage‑risk fields. Create `/reports/data_contract.md` and `/reports/initial_risk_register.md`. Propose a backtest design in `/reports/proposed_backtest_design.md`. Do not train models or create features.

### Stage 1: Evaluation and Backtest Foundation

Goal: Implement WMAE, WAPE, and bias. Create time‑based backtest splits using multiple forecast origins that mimic the test period. Confirm no future target leakage. Freeze the `/eval` and `/dataguard` directories. Document the backtest design in `/reports/backtest_design.md`.

### Stage 2: Naive Baselines

Goal: Implement simple baselines to establish a floor and verify the evaluation harness:

- Last observed value.
- Same weekday last week.
- Trailing 7‑day mean.
- Trailing 14‑day mean.
- Product‑warehouse median.
- Sparse fallback (e.g. zero forecast).

Score each baseline on all backtest windows. Append results to `results.csv`.

### Stage 3: Plain Gradient Boosting Model

Goal: Train a simple LightGBM (or XGBoost if necessary) model using minimal safe features (IDs, date features, lags, rolling means, etc.). Score on backtests and compare to baselines. Append results.

### Stage 4: Feature Engineering Phase 1

Goal: Implement known‑good retail forecasting features:

1. **Lags** – t‑1, t‑2, t‑3, t‑7, t‑14, t‑21, t‑28, same weekday lags.
2. **Rolling statistics** – trailing 7‑day, 14‑day, 28‑day, 56‑day means; medians; mins; maxes; standard deviations.
3. **Demand pattern features** – days since first sale, days since last non‑zero sale, share of zero‑sales days, trailing non‑zero mean, acceleration/deceleration metrics, short vs. long window ratios.
4. **Calendar features** – day of week, weekend flag, month, week of year, holiday flag, days before and after holidays, interactions with warehouse/country.
5. **Price and promotion features** – if provided and safe, include price level, discount flags, price change relative to history, promotion indicators, recent promotion share, known future promotion if legitimately available.
6. **Hierarchy features** – product‑level, warehouse‑level, category‑level, warehouse‑category means; item velocity bands; all calculated only from data before the cutoff.

Each feature family must be added in its own commit and logged in `results.csv`.

### Stage 5: Feature Engineering Phase 2 — Business Hypotheses

Goal: Test business‑relevant hypotheses:

1. **Fresh or fast‑moving items need shorter memory.** Use shorter lag windows for perishable or high‑turnover items.
2. **Holidays behave differently by country and warehouse.** Interact holiday features with warehouse/country.
3. **Demand volatility matters.** Add volatility bands and recent instability flags.
4. **High‑volume items deserve separate handling.** Segment high‑volume items or weight them differently.
5. **Sparse items should not be treated like regular items.** Provide special fallback logic for low‑history items.
6. **Over‑ and under‑forecasting have different business meaning.** Track bias by warehouse and category and adjust accordingly.

Log each hypothesis as a proposed change with business rationale, score impact, and decision to keep or reject.

### Stage 6: Model Structure

Start with one global LightGBM model. Compare against alternative structures (per horizon, per warehouse group, per volume segment) and simple blends. Retain complexity only if it improves backtests without compromising runtime or explainability. Prefer models that can be explained in a few lines.

### Stage 7: Pruning and Speed

Remove low‑value features, downcast data types, cache intermediates, reduce duplicated joins, and lower memory usage. Do not sacrifice score integrity for speed.

### Stage 8: Kaggle Finale

Port the final pipeline into a Kaggle notebook. Attach the competition data. Run from a clean session, recording start time, end time, total runtime, package versions, and random seeds. Generate the submission.csv, submit via Kaggle, and record public and private scores. Export the final notebook. Prepare a one‑page business summary and LinkedIn post.

## 9. Deliverables

1. **Public notebook** – Reproduces the final run.  
2. **results.csv** – Complete iteration log.  
3. **one_pager.md** – Business‑readable summary (what was forecast, scores, runtime, method outline, honesty caveats, implications).  
4. **leakage_review.md** – Adversarial review of leakage risks.  
5. **linkedin_post.md** – Plain‑English LinkedIn post.

## 10. Success Bands

- **Minimum publishable result:** Clean, reproducible run; official WMAE recorded; WAPE, bias, runtime reported; result competitive with simple baselines; no leakage concerns.
- **Strong result:** Score close to top public solution range; clear improvement over baselines; final notebook runs cleanly on Kaggle; one‑pager understandable to a business reader.
- **Excellent result:** Score approaches top public write‑ups; runtime short enough to support the “months to minutes” story; pipeline simple enough to explain without hiding behind complexity.

## 11. Risks and Mitigations

**Risk:** The workflow becomes a Kaggle trick project.  
**Mitigation:** Keep business metrics visible (WAPE, bias, runtime) and maintain honesty rules.

**Risk:** Copying public solutions without understanding them.  
**Mitigation:** Require one feature family per commit and log results; human reviews logic.

**Risk:** WAPE conflicts with Kaggle WMAE.  
**Mitigation:** Be clear: WMAE is the official score; WAPE is the business translation.

**Risk:** Leakage creates a fake improvement.  
**Mitigation:** Frozen /eval and /dataguard, cutoff‑based feature creation, adversarial review, suspicious score rule.

**Risk:** The post sounds like AI hype.  
**Mitigation:** Avoid hype language; do not claim autonomy; say “agentic workflow under human supervision.”

**Risk:** The result is not top‑tier.  
**Mitigation:** The post is valid if the pipeline is reproducible, measured, and honest; the point is the compression of work, not only the final score.

**Risk:** The problem sounds too technical.  
**Mitigation:** Explain through business language: forecast error, bias, over‑forecasting, under‑forecasting, warehouse planning, waste, inventory, service level, repeatable workflow.

## 12. Codex Working Rules

1. Do not optimize for leaderboard score before the evaluation harness is frozen.  
2. Do not use future target information.  
3. Do not modify `/eval` or `/dataguard` after freeze without explicit approval.  
4. Do not add complex models before baselines and LightGBM are complete.  
5. Do not keep a feature unless it improves validation or improves business explanation without hurting score.  
6. Do not hide failed experiments.  
7. Do not delete `results.csv` history.  
8. Do not make public claims not supported by logged results.  
9. Do not use “AI beat humans” framing.  
10. Always write code as if a sceptical data scientist will inspect it.

## 13. First Prompt to Codex

Start with Stage 0 only. Do not build models yet. Tasks: inspect data; create the initial repo structure; write `/reports/data_contract.md`; write `/reports/initial_risk_register.md`; propose safe backtest design in `/reports/proposed_backtest_design.md`; initialize `results.csv` with headers; write a README explaining the project.

## 14. Second Prompt to Codex

Proceed to Stage 1. Implement WMAE, WAPE, and bias; create time‑based backtest splits; build `/dataguard` functions; add tests for date leakage; create `/reports/backtest_design.md`; freeze `/eval` and `/dataguard` after human review.

## 15. Third Prompt to Codex

Proceed to Stage 2. Build naive baselines (last value, same weekday last week, trailing means, median, sparse fallback); score each baseline on all backtest windows; append results to `results.csv`; write `/reports/baseline_results.md`.

## 16. Fourth Prompt to Codex

Proceed to Stage 3. Build the first plain LightGBM model using only safe minimal features; score it on all backtest windows; compare against naive baselines; append results; write a report; identify the highest‑value next feature families.

## 17. Leakage Review Prompt

Review the current codebase as an adversarial leakage auditor. Find any place where future information could leak into training, validation, or test prediction, focusing on date joins, rolling windows, lag creation, target encodings, aggregates, price and promotion fields, calendar fields, train/test concatenation, validation design, cached features, and feature availability at forecast time. Return high‑risk, medium‑risk, and low‑risk findings, with recommended fixes and an assessment of whether the current score should be trusted.
