# Executive Insights: Rohlik Forecasting Benchmark

## 1. Executive summary

The final audited submission landed around the top 28% of the private leaderboard, better than roughly 72% of teams.

This is not winner territory. It is strong enough to demonstrate workflow capability, not model superiority.

The project shows how a domain expert can supervise an agentic workflow to produce a valid, audited, externally scored forecasting benchmark.

Final benchmark evidence:

- Stage 5-G fixed 50/50 raw L1 + Tweedie blend
- submission ref: `54401053`
- public WMAE: `20.62022`
- private WMAE: `20.14904`
- estimated rank slot: `215 / 777`
- top percentile: `27.67%`
- better than approximately `72.33%` of teams

## 2. What the benchmark proves

This benchmark shows that agentic workflows can compress the technical scaffolding around forecasting.

The human role does not disappear. It shifts:

- from manual execution to hypothesis framing,
- from ad hoc iteration to audit gates,
- from “try things until something works” to controlled decision points,
- and from local metric chasing to interpretation of externally scored evidence.

Domain expertise remains necessary because the workflow still depends on:

- what information is available at forecast time,
- which features are legitimate,
- which validation rules are frozen,
- which candidate improvements are acceptable,
- and how to interpret a score change in business terms.

External scoring matters because it disciplines local validation optimism. Local folds are useful, but the official Kaggle score was the truth gate.

## 3. What the benchmark does not prove

This benchmark does not prove:

- that AI can replace a data science team,
- that the result is production-ready,
- that the method is independently novel,
- that the same approach transfers unchanged to every enterprise dataset,
- or that governance, data quality, monitoring, and business ownership can be skipped.

The result is evidence about workflow quality under supervision, not proof of autonomous forecasting.

## 4. The operating loop

The project ran as a controlled loop:

Business hypothesis
→ field availability check
→ cutoff-safe feature design
→ frozen validation
→ adversarial review
→ candidate generation
→ external scoring
→ executive decision

Concrete examples from the project:

- Stage 1: frozen `eval/` and `dataguard/`
- Stage 2: naive baselines
- Stage 3 / 4: plain model development and the first official submission
- Stage 5-B / 5-E: feature improvements from public-solution-informed ideas
- Stage 5-F / G: objective diversity and fixed blend experiments
- rejected fold-specific blend: an example of governance working, because it was not promotable

This structure matters. The project did not just generate models; it enforced decision gates that prevented weak or non-generalizable ideas from becoming claims.

## 5. Key executive insights

1. AI changes the economics of experimentation.

   The bottleneck moves from raw implementation effort to deciding what is worth testing.

2. The hard part is no longer only modeling; it is controlled iteration.

   A model is easy to train compared with maintaining cutoff discipline, auditability, and reproducible evaluation.

3. Domain experts become more powerful when they can test hypotheses directly.

   The workflow turned forecasting ideas into measurable experiments instead of long speculative discussions.

4. Governance must move into the workflow, not sit at the end.

   Frozen evaluation, protected leakage layers, and adversarial review were part of the model-making process, not an afterthought.

5. Local validation is useful, but external scoring is the truth gate.

   Several local improvements transferred well; others did not. The official score is the only result that matters externally.

6. The best agentic work includes rejection, not just generation.

   The fold-specific blend was rejected because it was not generalizable. That rejection is a strength, not a failure.

7. Benchmark success should be interpreted as workflow evidence, not production proof.

   The output is a disciplined benchmark result, not a claim about enterprise deployment.

## 6. Forecasting-specific lessons

- Item demand should not be treated only as isolated SKU history.
  Group structure, calendar effects, and cross-item context matter.

- Price and discount features mattered.
  They were useful here, but they are benchmark-specific known-future covariates, not a universal promise of operational availability.

- Objective diversity helped.
  Raw L1 and Tweedie captured complementary error patterns, and the fixed blend improved the official score.

- Fixed blends are explainable and promotable.
  Fold-specific selection is not, because it does not transfer cleanly to the official test set.

- Better forecasting came from combining business logic, statistical validation, and audit discipline.
  The score improvement was not just “better modeling”; it was better control over the whole workflow.

## 6.1 Example: a supply-chain hypothesis that did not earn promotion

Business hypothesis:

> SKU demand should not be forecast in isolation; it should be informed by warehouse-category demand pressure, category mean reversion, and changing item share.

Result:

- Stage 5-H fixed blend local mean WMAE: `19.0182851267`
- Stage 5-G fixed blend local mean WMAE: `19.0089170845`
- Stage 5-H was worse by `0.0093680422`

Interpretation:

- The hypothesis was plausible.
- The features were cutoff-safe and audited.
- The experiment did not beat the current best local benchmark.
- No Kaggle candidate was promoted.
- This is evidence of governance working, not a failure.

The workflow made the hypothesis cheap to operationalize, but the validation framework prevented us from promoting extra complexity without evidence.

This rejects the current feature design for the current model, not the entire business idea forever.

## 7. Executive hypothesis backlog

These are future experiments, not proven findings.

| Hypothesis | Business question | Possible cutoff-safe feature direction | Risk / caveat | Promotion gate |
|---|---|---|---|---|
| Category-level mean reversion | Do category demand levels revert toward category means after shocks? | Rolling category means and deviations from category baseline | Can wash out true local trends | Must improve external score and pass leakage review |
| Promotion decay | Do promotions have short-lived lift that fades predictably? | Lagged promotion indicators, post-promo decay flags | Easy to leak if promo timing is mishandled | Must be cutoff-safe and explainable |
| Substitution within category | Do items in the same category substitute for each other? | Category share features, relative rank, peer demand context | High leakage risk if peer demand uses future rows | Requires explicit audit and stable benefit |
| Warehouse-specific demand regimes | Do warehouses behave differently under the same calendar conditions? | Warehouse interactions, warehouse seasonality | Can overfit small warehouses | Needs fold stability and business rationale |
| Price elasticity by category | Does demand change differently by category when price moves? | Relative price, price change, category interactions | Benchmark-specific known-future assumption | Must not rely on unavailable operational data |
| Holiday recovery patterns | Does demand rebound after holidays in predictable ways? | Pre/post holiday windows, recovery lags | Calendar leakage if holiday windows are misused | Must beat current benchmark and remain robust |
| Cold-start / sparse-item behavior | Do sparse items need special logic separate from regular items? | History length bands, sparse-item fallback features | Sparse data can create noisy gains | Must show stable improvement on sparse folds |

## 8. Recommended next steps

- Freeze Stage 5-G as the benchmark result.
- Keep Stage 5-H as a documented rejected hypothesis.
- Package the public repository and final artifacts.
- Write a long-form article from the benchmark story.
- Create the executive slide narrative later.
- Future executive-hypothesis tests should be narrow, separately staged, and promoted only if they beat Stage 5-G locally and survive audit.

## 9. Public wording

Approved phrasing:

> A domain-supervised agentic workflow produced an audited, officially scored Kaggle result that landed around the top 28% of the private leaderboard, better than roughly 72% of teams.

Phrases to avoid:

- AI beat Kaggle
- replaced data scientists
- production ready
- top tier
- near winner
- autonomous forecasting system

## 10. Closing note

The value of the benchmark is not that it magically solved forecasting. The value is that it showed how far a supervised agentic workflow can get when it is constrained by frozen evaluation, leakage control, adversarial review, and external scoring.
