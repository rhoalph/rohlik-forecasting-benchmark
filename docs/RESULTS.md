# Results

## Official Kaggle submissions

| Stage | Submission ref | Public WMAE | Private WMAE | Notes |
|---|---:|---:|---:|---|
| Smoke template | 54338855 | 80.38078 | 81.29170 | Infrastructure-only smoke test |
| Stage 4 | 54371523 | 22.37834 | 21.91884 | First audited plain-model submission |
| Stage 5-B | 54374863 | 21.99264 | 21.61114 | Relative price/discount features |
| Stage 5-E | 54395405 | 21.09367 | 20.61497 | Stronger 76-feature set |
| Stage 5-G | 54401053 | 20.62022 | 20.14904 | Fixed 50/50 raw L1 + Tweedie 1.1 |

## Local benchmark history

| Stage | Local mean WMAE | Local mean WAPE | Local mean bias | Decision |
|---|---:|---:|---:|---|
| Stage 3 plain model | 20.6963289733 | 0.2134545476 | -0.0494276017 | Baseline model |
| Stage 5-E stronger features | 19.5424424315 | 0.2027897892 | -0.0472425016 | Promoted |
| Stage 5-G fixed 50/50 blend | 19.0089170845 | 0.1947450686 | -0.0304860566 | Promoted |
| Stage 5-H supply-chain hypothesis | 19.0182851267 | 0.1949806765 | -0.0347673588 | Rejected |

## Leaderboard placement

Stage 5-G private WMAE `20.14904` corresponded to an estimated private leaderboard slot of `215 / 777`, or about the top `27.67%`, better than roughly `72.33%` of teams.

## Stage 5-H rejected hypothesis

The supply-chain hypothesis was plausible and cutoff-safe, but it did not beat the current best local benchmark.

- Stage 5-H fixed blend local mean WMAE: `19.0182851267`
- Stage 5-G fixed blend local mean WMAE: `19.0089170845`
- Stage 5-H was worse by `0.0093680422`

No Kaggle candidate was promoted from Stage 5-H.

## Final public wording

> A domain-supervised agentic workflow produced an audited, officially scored Kaggle result that landed around the top 28% of the private leaderboard, better than roughly 72% of teams.

## Phrases to avoid

- AI beat Kaggle
- replaced data scientists
- production ready
- top tier
- near winner
- autonomous forecasting system

## Caveats

- The benchmark is public-solution-informed.
- Price/discount handling was benchmark-specific.
- Local validation is useful but not the same as the hidden Kaggle score.
- Stage 5-H rejects the current feature design for the current model, not the broader business idea forever.

