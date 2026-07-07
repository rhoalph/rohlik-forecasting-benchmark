# Rohlik Forecasting Benchmark Run

Retrospective public benchmark of the Kaggle Rohlik Sales Forecasting Challenge V2.

This repository demonstrates a domain-supervised, agentic forecasting workflow that was measured against the official Kaggle score. It is not a product build, and it is not presented as autonomous forecasting.

Final official result:

- public WMAE: `20.62022`
- private WMAE: `20.14904`
- estimated rank slot: `215 / 777`
- around top `28%` of the private leaderboard
- better than roughly `72%` of teams

Approved public wording:

> A domain-supervised agentic workflow produced an audited, officially scored Kaggle result that landed around the top 28% of the private leaderboard, better than roughly 72% of teams.

What this repository demonstrates:

- frozen evaluation and leakage controls
- local baselines and backtests
- official Kaggle submissions
- adversarial reviews and rejected hypotheses
- public-solution-informed improvement, clearly labeled as such
- human supervision over model selection, promotion, and claims

It does not claim:

- AI beat Kaggle
- replaced data scientists
- production ready
- top tier
- near winner
- autonomous forecasting system

The work is public-solution-informed and audited. It is a workflow and governance demonstration, not independent algorithmic novelty.

## Key references

- [Methodology](docs/METHODOLOGY.md)
- [Results](docs/RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Executive insights](docs/EXECUTIVE_INSIGHTS.md)
- [Permanent project contract](rohlik_spec.md)

## Repository contents

- `eval/` and `dataguard/` contain frozen scoring and leakage controls.
- `reports/` contains experiment reports, reviews, and executive summaries.
- `scripts/` contains reproducible experiment and candidate-generation scripts.
- `features/` and `models/` contain the approved feature and model code.
- `submissions/` contains committed candidate CSVs and official Kaggle submission artifacts.

## Reproducibility

Raw Kaggle data is intentionally excluded from version control. To reproduce the benchmark, download the Rohlik Sales Forecasting Challenge V2 data into `data/raw/` and follow [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Run the test and validation checks with:

```bash
python3 -m pytest -q
python3 -m scripts.validate_stage1
```

## License

MIT License. See [LICENSE](LICENSE).
