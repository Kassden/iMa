# Research Evaluation Protocol

The agentic optimizer uses a protected research evaluator before it is allowed
to compare recipes. This is separate from the legacy `ExperimentSpec` runner and
does not change race-day inference.

## Protocol

- Races are sorted by `date`, `race_no`, and `race_id`; all runners in a race
  stay in the same partition.
- Each fold is expanding and ordered `train < calibration < score`.
- Baselines are evaluated on the same score population as model recipes:
  `uniform`, `raw_market`, and `calibrated_market`.
- Scores are stored per race first, then averaged with equal race weights.
- Fold summaries publish mean, population standard deviation, and worst fold for
  every numeric metric. Metric directions are stored in `result.json`.
- The protocol manifest hashes the ordered race IDs, target kind, and fold
  boundaries. Changing the data or folds creates a new `protocol_id`.

## Targets

Primary target:
- `win_probability`: probability that a runner wins a race.

Exploratory targets:
- `ranking_strength`: within-race ordering signal.
- `placing_top_k`: probability of finishing in the declared top-k.
- `adjusted_finish_time_or_speed`: speed normalized within race conditions.
- `market_odds_forecast`: market probability as a research target only.

Target contracts materialize labels and reject malformed or leaking inputs. These
targets are diagnostic until the optimizer has a predeclared multi-objective
selection rule.

## Metric Contract V2

All objectives are minimized by Optuna. Metrics ending in arrows below retain
their natural interpretation in MLflow.

| Target | Optimizer objective | Main diagnostics |
| --- | --- | --- |
| `win_probability` | race log loss (lower) | race Brier and ECE (lower), top-pick and winner Top-3 rates, mean winner rank, winner MRR, per-race loss dispersion |
| `placing_top_k` | equal-race-weighted Brier (lower) | binary log loss and ECE (lower), exact precision/recall/F1@K and top-pick place rate (higher) |
| `ranking_strength` | negative equal-race-weighted NDCG@3 (lower objective; NDCG higher) | full-field NDCG, pairwise accuracy, race-wise Spearman/Kendall, mean winner rank, winner MRR |
| `adjusted_finish_time_or_speed` | equal-race-weighted MAE (lower) | RMSE (lower), race-wise Spearman/Kendall (higher) |
| `market_odds_forecast` | equal-race-weighted MAE (lower) | RMSE (lower) |

Accuracy and F1 are not probability objectives. The place F1 is calculated from
the exactly K highest-probability runners in each race, avoiding an arbitrary
global threshold. Ranking metrics never compare runners from different races.

The legacy short keys `brier`, `mae`, and `spearman` remain as compatibility
aliases. New consumers should use the explicit `race_*` keys.
