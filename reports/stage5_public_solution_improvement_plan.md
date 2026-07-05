# Stage 5 Public-Solution-Informed Improvement Plan

Date: 2026-07-05
Status: planning only
Scope: improve the official Kaggle score using public competition knowledge

## A. Stage 5 objective

Stage 4 is the frozen audited first official submission. Stage 5 is a separate improvement stage whose purpose is to improve the Kaggle score by learning from public Kaggle solutions, public notebooks, and writeups.

This stage is intentionally not a “plain model” exercise. Public-solution-informed work is allowed here, and that fact must be stated plainly in reports, commits, and any public write-up. Borrowed ideas must be attributed. The goal is score improvement, not preservation of the Stage 4 framing.

Stage 4 remains frozen and unchanged. Stage 5 must not overwrite Stage 4 artifacts or blur Stage 4 and Stage 5 results.

## B. Evidence base

The public benchmark intelligence gathered earlier points to a clear pattern in strong solutions:

- Winning writeups used horizon-specific or direct forecasting models rather than one monolithic undifferentiated fit.
- LightGBM and XGBoost both appear in strong solutions.
- Tweedie or other alternative objectives show up in top writeups, alongside raw-target models.
- Public winners used sales/order mean encodings, relative price/discount features, and competing-product or availability-style signals where the benchmark allowed them.
- Strong public workflows used validation folds aligned to F1/F2-style periods and relied on heavy feature generation.
- A public second-place style notebook used recursive prediction, sqrt targets, and many engineered features.
- A third-place style solution used XGBoost, heavy feature engineering, Optuna variants, and ensembling.
- Public starter notebooks around roughly 20.25 to 20.76 WMAE provide a useful lower bound and reference band.

Stage 4 private score was 21.91884, so the first improvement target should be to move into the starter/public-notebook territory before trying to chase the stronger public bands.

## C. Allowed and forbidden rules for Stage 5

### Allowed

- Public solution ideas with attribution
- New features
- Target transforms
- Alternative objectives
- Horizon-specific models
- Post-processing such as clipping if tested and disclosed
- Kaggle notebooks or free/cheap public compute if needed
- Multiple submissions, provided every submission is logged

### Still forbidden unless explicitly approved

- Hidden labels
- Leakage from `solution.csv`
- Future sales
- Future target information
- Future `total_orders` unless a separate policy decision is made and disclosed
- Modifying frozen `eval/` or `dataguard/`
- Hiding failed experiments
- Overwriting Stage 4 results or mixing them into Stage 5 claims

## D. Candidate improvement ideas ranked by expected value, risk, and runtime

### 1. Clip negative predictions to zero

- Expected value: low to medium
- Risk: low
- Runtime: trivial
- Rationale: Stage 4 produced 20 negative predictions. Clipping may reduce a small amount of absolute error and improves business readability.

### 2. Relative price and discount features

Examples:

- item price relative to recent item median
- warehouse/category price rank
- discount intensity summary
- active-discount flag
- maximum discount
- number of active discounts

- Expected value: high
- Risk: medium
- Runtime: low to medium
- Rationale: the Stage 3 ablations showed that price/discount covariates are a major driver of the improvement.

### 3. Horizon-specific direct models

- Expected value: high
- Risk: medium
- Runtime: medium to high
- Rationale: strong public solutions often split by horizon or use direct horizon-aware training.

### 4. Target transform or alternative objective

Candidates:

- sqrt target
- log1p target
- Tweedie objective
- direct comparison to the raw L1 objective

- Expected value: medium to high
- Risk: medium
- Runtime: low to medium
- Rationale: public solutions used transformed targets and alternative objectives frequently enough that this is worth testing.

### 5. More lag and rolling features

Examples:

- 28-day mean
- 7/14/28-day median
- trend between recent windows
- same weekday 2/3/4 weeks back

- Expected value: medium
- Risk: medium
- Runtime: medium

### 6. Warehouse/category interaction features

Examples:

- warehouse-category historical means
- product-category demand profiles

- Expected value: medium
- Risk: medium to high
- Runtime: medium
- Rationale: useful in public solutions, but leakage control must stay strict.

### 7. Availability / competing-product logic

- Expected value: potentially high
- Risk: high
- Runtime: medium to high
- Rationale: may help if benchmark policy and safe joins are respected, but this is the most leakage-sensitive family.

### 8. Ensembling

- Expected value: potentially high after single-model gains
- Risk: medium
- Runtime: high
- Rationale: not the first step; only consider after individual model improvements are proven.

## E. Proposed first experiment batch

The first Stage 5 batch should be small and ordered. Do not try everything at once.

### S5-A: clipped Stage 4 candidate diagnostic

- Purpose: test whether negative predictions are causing avoidable error
- Change from Stage 4: same model, same features, same training, clip predictions at zero before scoring
- Local validation plan: evaluate on the same frozen local folds and compare against Stage 4
- Kaggle submission policy: only submit if the local result is better and the change is documented
- Expected runtime: trivial
- Leakage risk: low
- Success criteria: lower WMAE without a meaningful bias regression
- Stop criteria: no measurable gain or worse WAPE/bias

### S5-B: relative price/discount features added to the Stage 4 model

- Purpose: capture the price/discount signal that the ablations suggest is material
- Change from Stage 4: add safe relative-price and discount-summary features, while keeping the approved leak controls
- Local validation plan: same frozen folds, compare against Stage 4, report fold-by-fold deltas
- Kaggle submission policy: only after the local result is clearly better and the feature contract is reviewed
- Expected runtime: moderate increase
- Leakage risk: medium
- Success criteria: clear improvement over Stage 4 and stable bias
- Stop criteria: suspicious score jump without explanation, or any leakage concern

### S5-C: horizon-specific direct LightGBM models

- Purpose: test whether direct per-horizon training beats the single global model
- Change from Stage 4: split the model by horizon or add explicit horizon-specific training logic
- Local validation plan: frozen folds, compare to Stage 4 and to Stage 2 baselines
- Kaggle submission policy: only if the per-horizon design is materially better and still auditable
- Expected runtime: medium to high
- Leakage risk: medium
- Success criteria: meaningful gain that justifies the added complexity
- Stop criteria: no gain, unstable fold behavior, or excessive complexity

### S5-D: sqrt target transform test

- Purpose: check whether a variance-stabilizing target helps
- Change from Stage 4: transform target with sqrt during training, invert at prediction time, compare against raw target
- Local validation plan: same folds and same scoring
- Kaggle submission policy: only if it beats Stage 4 cleanly and does not worsen business diagnostics
- Expected runtime: low to medium
- Leakage risk: low to medium
- Success criteria: lower WMAE and acceptable bias
- Stop criteria: no gain or worse calibration

## F. Submission policy

- Every Kaggle submission must use a unique message.
- Every submission must be logged in `results.csv`.
- Do not submit silently or “for a quick check” without logging it.
- Compare each submission against Stage 4 official result, not just local folds.
- Limit the first wave to a small approved batch before re-review.
- Record public and private scores explicitly.
- Do not cherry-pick results without documenting the failed attempts that led there.

## G. Hardware policy

- Start locally if runtime remains reasonable.
- Use Kaggle or other free/public compute only if experiments become too heavy for the local machine.
- If Kaggle notebooks are used, document the notebook environment and the notebook path or link.
- Cheap/free hardware is a useful constraint, but it should not block reasonable improvement.

## H. LinkedIn / artifact framing

Core claim for later public write-up:

- Stage 4 showed an audited official submission from a supervised agentic workflow.
- Stage 5 tests how quickly that workflow can absorb public expert knowledge and improve.
- This is comparable to how real data science teams build on prior work.
- The claim is workflow compression, not independent invention.

## Recommended first experiment order

1. S5-A clipped Stage 4 diagnostic
2. S5-B relative price/discount features
3. S5-D sqrt target transform
4. S5-C horizon-specific direct models

This order is intentionally conservative: start with the cheapest, safest check, then move into the likely high-value feature family, then the transform test, then the more structural horizon split.
