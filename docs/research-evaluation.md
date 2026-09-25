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
