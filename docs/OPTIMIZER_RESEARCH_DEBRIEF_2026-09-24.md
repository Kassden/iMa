# Optimizer Research Debrief - 2026-09-24

## Goal

Give the next model a compact, evidence-backed research brief for improving race-level log loss in the iMa research system without touching race-day deterministic behavior.

## Acceptance Criteria

- Explain the current live optimizer state and what it is actually exploring.
- Identify why the current deterministic sweep is low-yield.
- List low-hanging research changes most likely to improve log loss.
- Define guarded next experiments another model can implement and verify.

## Out Of Scope

- Do not restart or mutate the currently running cortex-server optimizer campaign unless explicitly requested.
- Do not change race-day inference or betting execution.
- Do not promote any model automatically.

## Live Campaign Snapshot

Source: `cortex-server:/home/imaopt/iMa/artifacts/agentic-learning/cortex-full-pipeline/trials.jsonl`

- Campaign: `cortex-full-pipeline`
- Process: `tmux` session `ima-full-pipeline`
- Profile: `adaptive`
- Completed when sampled: 1,984 runs
- Long profile size before adaptive generation: 2,616 fixed specs
- Current run surface: mostly `baseline-v1` schema, `logit` and `boosted` hyperparameters
- MLflow model registry: `ima-racing-candidates`

Current best blended log loss:

| Metric | Value | Trial | Run |
|---|---:|---|---|
| Best blended race log loss | 2.011806155 | `trial-1938` | `logit-long-c80-balanced-lbfgs-intercept-tol0p0003` |
| Best fundamental-only race log loss in live campaign | 2.233797442 | `trial-0007` | `boost-lr003-leaf15` |
| Best blended top-pick win rate | 0.310355208 | `trial-0005` | `logit-c200-balanced` |

Earlier reference values:

- Market baseline race log loss was about 2.013455.
- First 16-run best blended log loss was about 2.011819.
- The long sweep improved blended log loss by only about 0.000013 over the earlier best.

## What The Current Optimizer Is Doing

The current deterministic long catalogue is mostly hyperparameter search:

- Logistic regression: `C`, `class_weight`, solver, intercept on/off, tolerance.
- HistGradientBoosting: learning rate, max iterations, leaf nodes, L2.
- Feature schema remains effectively `baseline-v1`.
- Dataset construction remains fixed.
- Blend method remains fitted multiplicative market/fundamental blend.

This means the campaign is not yet testing the most important research degrees of freedom:

- Rich schemas.
- Feature-family drops.
- Rolling split alternatives.
- Ranking/listwise objectives.
- Different market/fundamental combination methods.
- Residual modeling against market odds.

## Key Diagnosis

### 1. The sweep is improving market-weight calibration, not horse signal

The top blended runs are all balanced logistic-regression variants with tiny positive fundamental weights around 0.03 and market weights around 1.06.

Top blended examples:

| Trial | Blended Log Loss | Fundamental Log Loss | Fundamental Weight | Market Weight | Run |
|---|---:|---:|---:|---:|---|
| `trial-1938` | 2.011806155 | 2.297610 | 0.030123 | 1.0599 | `logit-long-c80-balanced-lbfgs-intercept-tol0p0003` |
| `trial-1858` | 2.011806725 | 2.299745 | 0.031188 | 1.0597 | `logit-long-c50-balanced-lbfgs-intercept-tol0p0003` |
| `trial-1423` | 2.011809085 | 2.297946 | 0.031029 | 1.0596 | `logit-long-c5-balanced-lbfgs-nointercept-tol0p0003` |

Interpretation: blended log loss is mostly a market model with a small correction. Searching more logistic `C` values is likely near exhaustion.

### 2. Better fundamental models are being ignored by the blend

Boosted models have better fundamental-only log loss than logit, but the blend often assigns `fundamental_weight=0`.

Top fundamental examples:

| Trial | Fundamental Log Loss | Blended Log Loss | Fundamental Weight | Run |
|---|---:|---:|---:|---|
| `trial-0007` | 2.233797442 | 2.011932845 | 0 | `boost-lr003-leaf15` |
| `trial-1983` | 2.236793143 | 2.011932845 | 0 | `boost-long-lr0p015-iter80-leaf15-l20p3` |

Interpretation: fundamental models can rank horses better than logit, but the current blend objective on validation finds no additive value over market probabilities. This may be because:

- The model duplicates public odds information rather than adding orthogonal signal.
- The multiplicative blend is too restrictive.
- Validation split favors final-odds calibration so strongly that useful ranking signal is suppressed.
- The model output is poorly calibrated relative to market probabilities.

### 3. Existing rich-feature evidence is underused

Local artifacts show richer feature systems have already produced stronger fundamental models:

- `artifacts/experiments/notebook-rich-v2/results.csv`
  - `notebook-boost-lr006-leaf15`
  - fundamental log loss about 2.200113
  - blended log loss about 2.011933
  - fundamental weight 0

- `artifacts/feature-study/report.json`
  - selected rich model: `benter-rich-v1-boosted`
  - rich boosted fundamental log loss about 2.229455
  - top feature families by permutation importance:
    - `current_condition`
    - `performance_adjustments`
    - `past_performance`
    - `post_position`
  - strongest individual features included `jockey_top3_rate`, `avg_result_3`, `avg_result_2`, `last_result`, `last_speed_ratio`.

Interpretation: feature richness helps the standalone model, but the optimizer profile is not exercising it. The next research loop should explicitly test rich/notebook schemas and feature families.

## Low-Hanging Research Moves

### Priority 1: Add schema-aware optimizer specs

Current `ExperimentSpec` does not carry a schema choice; `run_experiments()` receives one global `feature_schema`.

Add:

- `feature_schema_name` to `ExperimentSpec`.
- Optional `drop_feature_families`.
- Dataset loader selection:
  - baseline dataset for `baseline-v1`
  - `load_full_rich_history()` for `benter-rich-v1` and `notebook-rich-v2`

First experiments:

- `notebook-rich-v2` boosted leaf 15/31 with learning rates 0.015, 0.03, 0.045.
- `benter-rich-v1` boosted leaf 15/31 with same learning rates.
- Drop weak families:
  - drop `source_quality`
  - drop `preferences`
  - keep only top families: `current_condition`, `performance_adjustments`, `past_performance`, `post_position`, `current_race`

Success criteria:

- Fundamental log loss beats 2.200113.
- Blended log loss beats 2.011806155 or produces stable positive fundamental blend weight.

### Priority 2: Try market residual modeling

Current blend combines calibrated fundamental probability with market probability after the model is trained to predict `target_win` directly.

Low-hanging alternative:

- Train model on residual signal relative to market:
  - target: winner indicator
  - features: horse/race features plus `logit(market_probability)` offset or baseline
  - output: correction factor over market probability

Candidate formulas:

- `score = market_probability ** market_weight * exp(alpha * model_margin)`
- `score = market_probability * exp(model_residual_score)`
- `logit(p) = logit(market_probability) + model_score`, then race-normalize

Why this matters:

- It forces the model to learn information not already in odds.
- It directly optimizes the only thing that can improve blended log loss.

Success criteria:

- Validation blend chooses positive residual weight.
- Test blended log loss beats 2.011806155.

### Priority 3: Optimize blend family, not just model hyperparameters

Current blend:

- multiplicative powers
- non-negative weights
- bounds 0 to 4

Add deterministic blend specs:

- market-only recalibration baseline
- fundamental-only
- multiplicative blend with wider/finer bounds
- additive logit blend
- residual-logit blend
- per-segment blend by odds bucket or field size

Important guardrail:

- Fit blend only on validation.
- Do not use test to pick blend family.

Success criteria:

- Beat market recalibration on validation and test.
- Avoid improvements that come only from test-set peeking.

### Priority 4: Change split evaluation before trusting small gains

Current results use one chronological split. The observed blended improvement is tiny enough that it may be split noise.

Add rolling time splits:

- Example folds:
  - train <= 2017, validation 2018-2020, test 2021
  - train <= 2019, validation 2020-2021, test 2022
  - train <= 2021, validation 2022-2023, test 2024-2025

Score by:

- mean race log loss
- worst-fold race log loss
- positive fundamental weight stability
- calibration stability

Success criteria:

- Improvement beats baseline across folds, not only on the current test period.

### Priority 5: Ranking objective as secondary, not first

Ranking models may improve top-pick rate but not log loss unless calibrated carefully.

Try after residual/blend work:

- pairwise ranker trained within race
- LambdaMART-style model if adding a mature library is acceptable
- convert ranking scores to probabilities via validation calibration and race normalization

Success criteria:

- Log loss improves, not only top-pick rate.
- Winner rank improves without calibration degradation.

## Why The Current Deterministic Generator Is Not Good Enough

It is deterministic and operationally safe, but it is too narrow:

- It spends most runs in logistic regularization space.
- It does not use evidence from rich-feature artifacts.
- It does not generate pipeline-level variations.
- It does not target the observed bottleneck: market-orthogonal residual signal.
- It will likely generate adaptive hyperparameters around the same shallow blended optimum, because its input space only contains existing model hyperparameters.

The generator should be upgraded from:

`results -> nearby hyperparameters`

to:

`results -> choose research axis -> generate schema/model/blend/split experiments -> verify across folds`

## Recommended Next Agent Plan

### Phase 1: Evidence Package

Objective:
Create a compact local summary artifact from `trials.jsonl` and MLflow showing top blended, top fundamental, blend weights, and per-kind performance.

Touch files:

- Add `scripts/summarize_optimizer_campaign.py`
- Add tests for parsing JSONL and ranking metrics.

Verification:

- Run against a tiny fixture.
- Run against current remote campaign after pulling or streaming JSONL.

### Phase 2: Schema-Aware Specs

Objective:
Let optimizer specs choose `baseline-v1`, `benter-rich-v1`, or `notebook-rich-v2`.

Touch files:

- `ima/experiments.py`
- `ima/optimizer.py`
- `ima/feature_sets.py`
- tests under `tests/test_optimizer.py` and/or `tests/test_experiments.py`

Verification:

- A dry-run shows schema in proposal payload.
- A smoke run executes one baseline and one rich spec.
- MLflow run params include `feature_schema`.

### Phase 3: Residual Market Model

Objective:
Add a model kind or wrapper that learns market-orthogonal correction rather than raw win probability.

Touch files:

- `ima/modeling.py`
- `ima/experiments.py`
- tests for probability normalization and leakage boundaries.

Verification:

- Unit test: probabilities normalize per race.
- Smoke experiment: residual model logs fundamental and blended metrics.

### Phase 4: Rolling Split Evaluation

Objective:
Prevent tiny one-split gains from driving the self-optimizer.

Touch files:

- `ima/data.py` or new split helper.
- `ima/experiments.py`
- optimizer ranking/voting.

Verification:

- Fixture tests for chronological fold boundaries.
- Report shows mean and worst-fold log loss.

## Concrete First Experiments To Queue

1. `notebook-rich-v2` boosted, leaf 15, lr 0.015/0.03, max_iter 80/120, L2 0/0.3/1.
2. `benter-rich-v1` boosted, same grid.
3. `notebook-rich-v2` logit with only top feature families, balanced, C 1/5/20.
4. Residual-logit market correction using `baseline-v1`.
5. Residual-boosted market correction using `notebook-rich-v2`.
6. Additive-logit blend for current top logit and top boosted artifacts.

## Guardrails

- Keep race-day deterministic path unchanged.
- Do not train on final odds for live deployment; historical final odds are only a research benchmark.
- Do not pick blend/model family from test split.
- Treat improvements below 0.00005 log loss as noise until confirmed across rolling folds.
- Register candidate models in MLflow, but do not promote automatically.

## Verdict

The current optimizer is healthy operationally, but its search space is the wrong bottleneck. The fastest plausible log-loss improvement is not another thousand logistic hyperparameters; it is schema-aware rich features plus a market-residual/blend redesign evaluated on rolling time splits.
