# Stage 1.5 — Benchmark Intelligence and Score Calibration

Date: 2026-07-04  
Competition: [Rohlik Sales Forecasting Challenge V2](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2)  
Status: research complete; no submission made; `eval/` and `dataguard/` remain unfrozen and unchanged

## Executive conclusion

The competition is directly usable as a retrospective score benchmark. Authenticated Kaggle metadata exposes a complete 777-team final leaderboard. The winning final score is 17.08084 WMAE, the podium spans 17.08084–17.23470, the top 10 ends at 18.05386, the top 5% ends near 18.60099, and the top 10% ends near 19.00455.

Public winning-solution traces exist for first, second, and third place. First place published a detailed methodology discussion but no linked final code. Second and third place published detailed discussions and public Kaggle notebooks. Public baseline notebooks also exist with claimed leaderboard scores around 20.25–20.76.

The current `eval/` implementation is compatible with Kaggle's published metric definition. No confirmed metric mismatch was found, so this stage did not modify `eval/` or `dataguard/`. The remaining uncertainties are late-submission availability, Kaggle's undisclosed hidden row split, and incomplete historical-label coverage when the official test mask is shifted to local backtest periods.

## Research method

Authenticated metadata was collected with the Kaggle CLI and installed Kaggle API client. The main commands were:

```bash
kaggle competitions list --search rohlik-sales-forecasting-challenge-v2 --format json
kaggle competitions leaderboard rohlik-sales-forecasting-challenge-v2 --show --page-size 200 --format json
kaggle competitions submissions rohlik-sales-forecasting-challenge-v2 --page-size 200 --format json
kaggle competitions files rohlik-sales-forecasting-challenge-v2 --format json
kaggle competitions topics list rohlik-sales-forecasting-challenge-v2 --page 1 --format json
kaggle competitions topics show rohlik-sales-forecasting-challenge-v2 563215
kaggle competitions topics show rohlik-sales-forecasting-challenge-v2 563117
kaggle competitions topics show rohlik-sales-forecasting-challenge-v2 563064
kaggle kernels list --competition rohlik-sales-forecasting-challenge-v2 --sort-by voteCount --page-size 100 --format json
kaggle kernels list --competition rohlik-sales-forecasting-challenge-v2 --sort-by scoreAscending --page-size 100 --format json
```

The Kaggle API was paginated in memory to inspect all 777 final leaderboard rows. No leaderboard or notebook code was downloaded. Public web searches used the requested exact search phrases and variants across Kaggle, GitHub, LinkedIn, and the broader web.

## 1. Leaderboard intelligence

### Competition metadata

| Field | Finding | Confidence/source |
|---|---|---|
| Competition ID | 88742 | Authenticated Kaggle API |
| Deadline | 2025-02-14 23:00 UTC | Authenticated Kaggle API |
| Category | Community | Authenticated Kaggle API |
| Prize | USD 10,000 | Authenticated Kaggle API |
| Teams | 777 | Authenticated Kaggle API |
| Account entered | Yes (`userHasEntered: true`) | Authenticated Kaggle API |
| Maximum daily submissions | 5 | Authenticated Kaggle API |
| Account submissions | None | `kaggle competitions submissions` returned `No submissions found` |
| Competition files | Six expected CSVs listed successfully | Authenticated Kaggle CLI |
| Total submissions | 12,256 claimed | Participant LinkedIn posts; not exposed by the current competition API response |
| Late submissions enabled | **Unconfirmed** | Neither CLI/API metadata nor public page source exposes a late-submission flag |
| Account can submit now | **Unconfirmed without submitting** | Authentication, entry, rules acceptance, and daily-limit metadata pass; post-deadline acceptance is not testable without consuming a submission |

The project specification assumes a late submission is possible. Current metadata does not confirm or contradict that assumption. A deliberately approved smoke submission is the only conclusive test of the submission endpoint after the deadline.

### Final/private leaderboard

The CLI returns one completed-competition leaderboard score column. The values match the private scores stated by podium authors, so this report treats it as the final/private leaderboard.

| Rank | Team | Final WMAE |
|---:|---|---:|
| 1 | Golden | 17.08084 |
| 2 | Mr Croissant | 17.08768 |
| 3 | Hardy Xu | 17.23470 |
| 5 | Rafał Pawłowski | 17.80010 |
| 10 | GOOO | 18.05386 |
| 25 | Chaitanyagarg2 | 18.45180 |
| 50 | ensemble 3 solution | 18.66892 |
| 100 | Toma Shirai | 19.05662 |
| 200 | Audac.IA | 19.83877 |
| 300 | haro3003 | 20.61420 |
| 400 | TG | 22.31101 |

The tail contains failed/extreme submissions, including a maximum score around `1.048e25`; it should not be used to characterize normal performance.

### Distribution landmarks

| Landmark | Maximum WMAE at landmark |
|---|---:|
| Podium | 17.23470 |
| Top 10 | 18.05386 |
| Top ~1% | 17.94553 |
| Top ~5% | 18.60099 |
| Top ~10% | 19.00455 |
| Top ~25% | 19.82415 |
| Median | approximately 21.94617 |

Threshold counts from the final leaderboard:

| WMAE threshold | Teams at or below | Approximate percentile |
|---:|---:|---:|
| 18.10 | 12 | Top 1.54% |
| 18.60 | 38 | Top 4.89% |
| 19.00 | 75 | Top 9.65% |
| 19.50 | 139 | Top 17.89% |
| 20.00 | 246 | Top 31.66% |
| 20.25 | 277 | Top 35.65% |
| 20.75 | 309 | Top 39.77% |

### Public leaderboard

The authenticated CLI does not expose a separate historical public leaderboard after competition close. Public-score evidence is therefore source-specific rather than a complete table:

- First-place author Laurence Greig reports 17.81 public / 17.08 private for the selected final solution, and a best public score of 17.59 that would have scored about 17.05 private.
- Second-place author Farukcan Saglam reports public/private pairs for individual models and ensembles, including a main notebook around 17.58 public / 17.35 private.
- A 36th-place participant reports finishing 25th public and 36th private, consistent with a leaderboard shake-up.

These author statements establish that public and private scores differ materially. They do not reconstruct the complete public leaderboard.

## 2. Winning-solution research

Winning-solution traces **do exist** for all three podium places.

### First place

Source: [1st Place Solution](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/discussion/563215)  
Author: Laurence Greig (`LGreig`), winning team `Golden`  
Relevance: V2, final winner  
Code: no linked final solution code found  
Disposition: high-value reference; do not treat as a reproducible baseline

Claims summarized:

- Final score 17.08 private and 17.81 public.
- Direct models separated by each of the 14 forecast horizons improved about 0.4 over one model for the full horizon.
- Each horizon used appropriately offset/truncated lag and rolling features.
- LightGBM and XGBoost were averaged for each day; blending improved about 0.2.
- Tweedie objectives were used.
- Important feature ideas included sales/order mean encodings, relative price/discount features, and competing-product availability.
- The primary validation period exactly matches our F1 window; a second period closely matches F2.
- Full feature generation and 28 models reportedly took roughly two days; a simpler one-shot pipeline took about four hours and scored around 17.4–17.5 private.

The time/complexity trade-off is directly relevant to this project's preference for explainability and runtime discipline.

### Second place

Sources:

- [2nd Place Solution — Recursive NN + GBDT Ensemble Model](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/discussion/563117)
- [Rohlik — Recursive Prediction notebook](https://www.kaggle.com/code/greysky/rohlik-recursive-prediction)

Author: Farukcan Saglam (`greysky`), team `Mr Croissant`  
Relevance: V2, final second place  
Code: public Kaggle notebook available  
Disposition: high-value advanced reference, not a baseline

Claims summarized:

- Recursive prediction made previous horizon predictions available as later lags.
- The target was square-root transformed and modeled with L2-style objectives.
- Price- and order-ratio estimators were used as base estimates; note that this solution used future `total_orders`, which our approved policy excludes.
- Roughly 117 features included lags at 1–7, 14, 21, and 28 days.
- Models included RealMLP, LightGBM, XGBoost, CatBoost, and TabM.
- Reported private scores for individual models were approximately 17.79–18.81; an ensemble was around 17.39 in the comparison table.
- The main notebook reportedly took about 1 hour 50 minutes and scored around 17.58 public / 17.35 private.
- Validation used the last two training weeks.

The solution is informative but not directly comparable to our future pipeline while `total_orders` remains excluded.

### Third place

Sources:

- [3rd Place Solution — a little XGBoost, a lot of feature engineering](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/discussion/563064)
- [Simplified 3rd Place Solution — Rohlik Sales](https://www.kaggle.com/code/hardyxu52/simplified-3rd-place-solution-rohlik-sales)

Author: Hardy Xu  
Relevance: V2, final third place  
Code: public simplified Kaggle notebook available  
Disposition: high-value advanced reference; selected feature ideas may later become logged hypotheses

Claims summarized:

- Final leaderboard score 17.23470.
- Square-root target transformation with XGBoost performed better than a direct MAE objective.
- Two recent two-week holdouts were used for feature decisions, including an April no-holiday window close to our F4 period.
- Seventeen Optuna-derived hyperparameter/model variants were averaged, improving WMAE by about 0.55.
- A neural-network stack added about 0.04.
- Local holdout WMAE remained approximately 1.5–2.5 better than leaderboard WMAE even after leakage precautions.
- LightGBM underperformed XGBoost in this solution and did not improve the blend.

The reported local-to-leaderboard gap is important calibration evidence for our own incomplete-grid folds.

### Other public traces

| Source | Author | Claim | Code | V1/V2 | Disposition |
|---|---|---|---|---|---|
| [Laurence Greig profile entry](https://au.linkedin.com/in/laurence-greig-8ab4b5180) | Laurence Greig | Confirms first-place result and links the winning discussion | No separate code | V2 | Corroborating reference |
| [36th-place LinkedIn summary](https://www.linkedin.com/posts/zulqarnainalipk_datascience-machinelearning-salesforecasting-activity-7297996098080493569-uTLq) | Zulqarnain Ali | Claims 25th public, 36th private; LightGBM/CatBoost/XGBoost blend with lags and hierarchy trends | Notebook promised but not identified in the post | V2 | Supplementary reference |
| [WaveNet + Transformer write-up](https://www.linkedin.com/pulse/kaggle-competition-rohlik-sales-forecasting-challenge-karnjanavivin-mpbhe) | Apiwit Karnjanavivin | Reports a private score around 88 and a hybrid neural approach | [GitHub code](https://github.com/MisterFOURXXX/kaggle-competitions/tree/main/rohlik-sales-forecasting-challenge) | V2 | Ignore for score targeting; retain only as a low-ranking implementation trace |
| [Kushal Devanabanda result](https://www.linkedin.com/posts/kushal-devanabanda_rohlik-sales-forecasting-challenge-activity-7297325549201301504-1EaN) | Kushal Devanabanda | Reports rank 271, 20.86121 public and 20.24298 private | Links external iterations | V2 | Useful mid-board calibration reference |
| [RealMLP/pytabkit trace](https://x.com/DHolzmueller/status/1899128207695077695) | David Holzmüller | Notes RealMLP use in the second-place solution | Generic model library exists, not Rohlik final pipeline | V2 reference | Model-family context only |

No dedicated first-place GitHub repository, first-place public notebook, or independent podium blog post was found. The Kaggle discussions remain the strongest primary sources.

## 3. Public notebook intelligence

The following notebooks were returned by Kaggle's competition-filtered kernel search. Vote counts are metadata observed on 2026-07-04. Notebook title scores are author claims; no code was copied or rerun in this stage.

| Notebook | Author | Votes | What metadata/title indicates | Category |
|---|---|---:|---|---|
| [Rohlik Sales 2024 — Get Started](https://www.kaggle.com/code/thiagomantuani/rohlik-sales-2024-get-started) | Thiago Mantuani | 163 | Highest-vote getting-started notebook | Reference / likely submission orientation |
| [Rohlik Sales — LightGBM LB 20.75](https://www.kaggle.com/code/meryentr/rohlik-sales-lightgbm-lb-20-75) | Yentür | 162 | LightGBM; title claims public LB 20.75 | Baseline reference |
| [RSFC Yunbase](https://www.kaggle.com/code/yunsuxiaozi/rsfc-yunbase) | yunsuxiaozi | 135 | High-vote competition pipeline; score not in metadata | Reference |
| [Not a Winner, but Maybe some Inspiration](https://www.kaggle.com/code/macarrony00/not-a-winner-but-maybe-some-inspiration) | João Varela | 103 | High-vote post-competition notebook | Reference only |
| [Starter notebook with lagged features — LB 20.25](https://www.kaggle.com/code/hiarsl/starter-notebook-with-lagged-features-lb-20-25) | Matthias | 85 | Lagged starter; title claims LB 20.25 | Strong baseline reference |
| [Simplified 3rd Place Solution](https://www.kaggle.com/code/hardyxu52/simplified-3rd-place-solution-rohlik-sales) | Hardy Xu | 78 | Public simplified podium pipeline | Advanced reference |
| [LightGBM FLAML — LB 20.76](https://www.kaggle.com/code/christph/lightgbm-flaml-lb20-76) | christph | 62 | Auto-tuned LightGBM; title claims LB 20.76 | Baseline reference |
| [Rohlik2 Lama v6 weighted](https://www.kaggle.com/code/samvelkoch/rohlik2-lama-v6-weighted) | Samvel Kocharyan | 53 | Explicitly weight-aware by title | Metric/model reference; metadata alone is insufficient to endorse its metric logic |
| [Rohlik Sales Forecasting — XGBRegressor](https://www.kaggle.com/code/yzokulu/rohlik-sales-forecasting-xgbregressor) | mldynamics | 39 | XGBoost pipeline | Model-family reference |
| [Rohlik — Recursive Prediction](https://www.kaggle.com/code/greysky/rohlik-recursive-prediction) | Farukcan Saglam | 38 | Public second-place recursive pipeline | Advanced reference |
| [Simple CatBoost Baseline](https://www.kaggle.com/code/crustacean/simple-catboost-baseline) | Fabian Henning | 34 | CatBoost baseline | Baseline reference |
| [AutoGluon with lagged features](https://www.kaggle.com/code/liangshutong/autogluon-with-lagged-features) | LiNuS | 25 | Automated ensemble with lags | Reference; unnecessary complexity for early stages |
| [Rohlik Sales — XGBoost lagged features 21.48](https://www.kaggle.com/code/sagarmamodia/rohlik-sales-xgboost-lagged-features-21-48) | Sagar Mamodia | 19 | XGBoost lag pipeline; title claims 21.48 | Baseline reference |
| [Naive baseline — median per ID and weekday](https://www.kaggle.com/code/tom3141592/naive-baseline-median-per-id-and-day-of-week) | TomVdB | 10 | Explicit naive statistical baseline | Stage 2 baseline reference |

Notebook conclusions:

- Public V2 notebooks cover naive medians, LightGBM, XGBoost, CatBoost, AutoGluon, neural approaches, recursive forecasting, and valid leaderboard submissions.
- The most useful early comparison points are 20.75 for a popular LightGBM notebook and 20.25 for a popular lagged starter notebook.
- Public scores in notebook titles prove those notebooks generated accepted submissions, but they do not guarantee leakage-free validation or reproducibility.
- No public notebook was accepted as evaluation authority; competition pages and the official weights file remain authoritative.

## 4. Metric and submission verification

### Official contract

The [Kaggle Evaluation page](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview/evaluation) states that submissions are evaluated by Weighted Mean Absolute Error and links scikit-learn's weighted `mean_absolute_error` definition.

For row `i` and inventory `u_i`:

$$
\mathrm{WMAE} =
\frac{\sum_i w_{u_i}\lvert y_i-\hat{y}_i\rvert}
     {\sum_i w_{u_i}}.
$$

Metric/submission findings:

| Question | Finding |
|---|---|
| Exact official name | Weighted Mean Absolute Error (WMAE) |
| Exact normalization | Weighted absolute-error sum divided by sample-weight sum |
| Weight granularity | One weight per `unique_id` in `test_weights.csv`; repeat it over that ID's requested rows |
| Required test coverage | Submission instructions require a prediction for each ID in the test set; `solution.csv` contains all 47,021 test keys |
| Rows used in one leaderboard score | Hidden subset. Final/private and public scores differ; a participant reports 70% private, but this split was not found in official API metadata |
| Clip predictions at zero | No clipping rule found in official evaluation/data pages |
| Missing predictions | Not documented as allowed; a complete keyed submission is required. Treat missing predictions as invalid |
| Submission ordering | Kaggle specifies keyed `id,sales_hat` rows rather than an ordering rule. Exact template order is the safest reproducible policy |
| Submission columns | Exactly `id` and `sales_hat` |

### Comparison with current `eval/`

| Current behavior | Compatibility assessment |
|---|---|
| Computes `sum(weight * abs(error)) / sum(weight)` | Exact match to published formula |
| Joins weights by `unique_id`, then repeats per grid row | Match |
| Aligns labels/predictions by (`unique_id`, `date`) | Match to ID semantics; safer than positional scoring |
| Rejects duplicate, missing, or extra prediction keys | Compatible with complete-submission requirement |
| Verifies exact `solution.csv` ID order | Stricter than documented Kaggle behavior, but safe and reproducible |
| Does not clip predictions | Match; no official clipping requirement found |
| Rejects NaN/infinite predictions | Appropriate for an accepted numeric submission |
| WAPE and bias are separate from WMAE | Correct; they do not affect Kaggle score |

**Result: no confirmed mismatch.** No changes to `eval/` or `dataguard/` are justified at Stage 1.5.

Compatibility limitations:

1. Hidden test labels are unavailable, so local code cannot reproduce a Kaggle test score before submission.
2. Kaggle's public/private row membership is undisclosed; local evaluation cannot reproduce those exact subsets.
3. Shifted local folds score only 89.40%–94.03% of the official grid mask because some historical item-date labels do not exist.
4. Local WMAE may be systematically optimistic. Third place reported local holdouts 1.5–2.5 WMAE below leaderboard scores.

## 5. Submission smoke-test proposal

No submission was made.

| Option | Safe? | What it proves | What it does not prove | Slot/audit impact |
|---|---|---|---|---|
| Official `solution.csv` unchanged / all zeros | Structurally safest | Confirms post-deadline submission acceptance, authentication, file schema, complete IDs, and Kaggle scoring response | Does not validate forecasting quality or local-vs-Kaggle metric equivalence | Consumes one submission if accepted; creates a permanent poor score but is easy to label as infrastructure smoke test |
| Global training median | Safe if generated deterministically and clipped policy is documented | Confirms a generated prediction vector and scoring integration; gives a weak scalar baseline | Does not validate item-level history or time-series logic | Consumes one slot and begins the experimental score trail before Stage 2 |
| Last-known sales by ID with fallback | Safe only after Stage 2 implementation/tests | Confirms per-ID historical lookup and a meaningful baseline submission | Does not validate learned models or full feature pipeline | Consumes one slot; should be logged as a real baseline, not smoke-only |

### Recommendation

After explicit human approval, submit the unchanged official zero template once with a description such as:

```text
stage1.5 infrastructure smoke test: untouched official zero template
```

Why this option:

- It cleanly separates submission plumbing from baseline/model quality.
- It is already verified locally to contain exactly the official 47,021 ordered IDs.
- It conclusively answers whether late submissions are enabled and whether this account may submit.
- Its poor score is transparent and can be excluded from model comparisons while retained in the audit trail.

Do not submit a global median or last-value forecast until Stage 2, when it can be reproduced, backtested, and logged as a baseline. Before the zero submission, archive the exact file hash and intended description in the results/audit log.

## 6. Updated score bands

These proposed bands apply to the **final/private Kaggle WMAE** returned by a late submission, not local backtest WMAE. They do not override the requirement to beat naive baselines, report WAPE/bias/runtime, and pass leakage review.

### Minimum publishable: WMAE ≤ 20.25

- Approximately top 35.7% of the final leaderboard distribution.
- Matches or beats the claimed score of the popular lagged starter notebook.
- Better than the popular LightGBM/FLAML references around 20.75–20.76.
- Still requires demonstrated improvement over our own Stage 2 naive baselines.

### Strong: WMAE ≤ 19.00

- Approximately top 9.7% of the final leaderboard.
- Clearly beyond common public starter notebooks.
- Leaves substantial distance from podium solutions, so it supports a credible strong result without implying winner equivalence.

### Excellent: WMAE ≤ 18.10

- Approximately top 1.5% and within the top 12 final teams.
- Close to the top 10 cutoff of 18.05386.
- Podium context remains 17.08084–17.23470; an 18.10 score is excellent but not podium-equivalent.

### Reference-only elite band

- WMAE ≤ 17.50 corresponds to the three podium teams in the final leaderboard.
- This is context, not a project target or claim.

Because third place reported local scores 1.5–2.5 below leaderboard scores, these bands must not be applied directly to local WMAE. Local score bands should be calibrated only after Stage 2 baselines and at least one approved Kaggle submission.

## 7. Open risks and remaining questions

1. **Late submission status:** Is the competition accepting late files? Metadata is inconclusive; only an approved submission can answer.
2. **Submission permission:** The account is authenticated, entered, and has a five-per-day limit, but the post-deadline endpoint is untested.
3. **Hidden score subset:** Public/private membership is unknown. A full local test-score reconstruction is impossible without hidden labels.
4. **Local coverage:** Shifted historical test masks omit 6%–11% of labels, changing the fold-specific effective weight mix.
5. **Local-to-Kaggle gap:** Public evidence suggests local validation can be materially optimistic.
6. **`total_orders` comparability:** Strong public solutions used future `total_orders`; our approved exclusion improves operational honesty but weakens direct leaderboard comparability.
7. **Known-future claim scope:** Price and discount fields are benchmark-known future, not proven operationally scheduled outside the competition.
8. **No clipping rule:** Negative forecasts are not forbidden by documentation, but nonnegative clipping should be tested later as a logged modeling decision, not inserted into evaluation.
9. **Notebook reproducibility:** Notebook title scores and author claims were not rerun; treat them as intelligence, not verified benchmarks.
10. **Git baseline:** The project still needs its own tracked repository/freeze commit before protected-layer enforcement is meaningful.

## Recommendation before freeze

Keep `eval/` and `dataguard/` unchanged and unfrozen. Human review should decide:

1. Whether to authorize the one-slot zero-template smoke submission.
2. Whether the proposed final-score bands are accepted.
3. Whether current incomplete-grid folds are sufficient for Stage 2 comparisons or need a secondary all-observed grid.
4. Whether to add an explicit local-vs-Kaggle calibration field after the first meaningful submission.
5. Whether to initialize a dedicated Git repository now and then freeze Stage 1 in a reviewed commit.

Only after these decisions should Stage 1 be frozen or Stage 2 baselines begin.
