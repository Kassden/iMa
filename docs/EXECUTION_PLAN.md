# Live Racing Decision Pipeline

## GOAL
- Build and verify an end-to-end HKJC live odds, horse data, race-aware modeling, calibration, market blending, risk-constrained wagering, and Hong Kong deployment pipeline.

## Acceptance Criteria
- [ ] Browser-backed HKJC provider captures a real meeting, runners, WIN/PLACE odds, and timestamps on the Hong Kong egress.
- [ ] Existing horse profile/form/trackwork/veterinary/movement enrichment feeds the same normalized snapshot and model schema.
- [x] Historical source research identifies and verifies the best obtainable 2005-2025 race, runner, result, dividend, and pre-race odds data, including an HKJC archive-scraping proof of concept where permitted.
- [x] Every historical source has a coverage, provenance, licensing/access, timestamp semantics, missingness, and reconciliation report before use in training.
- [ ] Historical data is transformed without future leakage and split chronologically with complete races kept together.
- [ ] Market baseline, conditional-logit-style race model, and boosted tabular benchmark report race log loss, Brier score, calibration, and simulated WIN return.
- [ ] Calibration and fundamental/market blending are fit on validation data only and evaluated once on held-out races.
- [ ] Models produce a coherent joint finish-order distribution from which valid WIN, PLACE, and exotic-pool combinations are ranked by calibrated probability.
- [ ] Every candidate combination reports model probability, model-implied fair odds, available market odds or projected dividend, uncertainty, and market discrepancy without conflating fair odds with pari-mutuel payout forecasts.
- [ ] Wager sizing is constrained by bankroll, race exposure, edge, uncertainty, and configured drawdown budget; all pools fit a shared contract while WIN ships first.
- [ ] Every race stores immutable pre-race features, market odds, model probabilities, recommendations, and eventual official outcomes so predicted versus actual performance is auditable.
- [ ] Completed Hong Kong meetings enrich the canonical dataset and trigger a reproducible champion/challenger retraining workflow approximately twice weekly without automatically promoting a worse model.
- [ ] Authenticated HKJC web execution is isolated behind paper mode, explicit live enablement, idempotency, limits, and audit records.
- [ ] A Hong Kong host can install, scrape, train, recommend, and run a smoke check from documented commands.

## Research
- Built-in options: pandas/scikit-learn/statsmodels; preserve the existing Python 3.11 environment.
- Off-the-shelf options: HistGradientBoosting/CatBoost-style tabular ranking before neural race-set models; browser automation only for access paths rejected for direct HTTP.
- Papers: Benter for fundamental plus public-odds combination; Kelly and chance-constrained work for bankroll growth under estimation risk.
- Historical data research: inspect official HKJC race results, horse records, race cards, sectional timing, dividends, and archived odds surfaces; compare public datasets and licensed vendors; verify whether archived odds are true pre-race snapshots or only final/starting prices.
- Pool modeling research: compare Plackett-Luce/Harville-style order models, conditional logit, permutation simulation, calibrated learning-to-rank, and race-set neural models for coherent exact-order and unordered-combination probabilities; study pool takeout, breakage, liquidity, and dividend forecasting separately from outcome probability.

## Regression Guardrails
- Planned edit surface: `scrapper/**`, `ima/**`, `tests/**`, `scripts/**`, `deploy/**`, `pyproject.toml`, `README.md`, `docs/**`.
- Protected behaviors: existing fixture normalization, legacy model columns, horse page extraction, user dataset/notebook reorganization, raw source data.
- Likely consumers: CLI operators, training jobs, inference jobs, strategy engine, future authenticated transaction worker.
- Damage radius: systemic.
- Proof plan: unit tests, temporal leakage assertions, probability-sum assertions, deterministic strategy tests, CLI smoke tests, real browser snapshot, and deployment config validation.
- Atomic units: browser provider; dataset/splits; model evaluation; calibration/blend; strategy/execution boundary; deployment/runbook.
- Commit plan: no commits unless requested; preserve user worktree changes.

## Phase 1: Discovery and contracts

### Subphase 1.1: Lock domain and pool contracts
- Commit: define race, odds, prediction, recommendation, and execution contracts.
- Tests: contract serialization and probability validation tests.
- Success Criteria: WIN works now and other pools can be represented without changing ingestion boundaries.
- Planned Touch Files: `ima/domain.py`, `tests/test_domain.py`, `docs/EXECUTION_PLAN.md`.
- Checklist:
  - [ ] Define timestamped pool snapshot and decision contracts.
  - [ ] Define fail-closed validation and audit fields.

## Phase 2: Browser live ingestion

### Subphase 2.1: Parse rendered HKJC race pages
- Commit: add Chrome-backed odds provider and provider selection.
- Tests: recorded rendered-HTML parser tests plus existing scraper tests.
- Success Criteria: stable DOM IDs produce normalized runners and real odds; CLI writes a real snapshot.
- Planned Touch Files: `scrapper/browser_odds.py`, `scrapper/pipeline.py`, `scrapper/cli.py`, `scrapper/tests/**`, `pyproject.toml`.
- Checklist:
  - [ ] Implement Chrome invocation and rendered DOM parsing.
  - [ ] Map DOM data into the existing provider payload contract.
  - [ ] Capture all displayed pool combinations, odds/dividend estimates, sell status, pool totals, update timestamps, and scratches rather than only WIN/PLACE runner odds.
  - [ ] Run and preserve ignored live snapshot evidence.

## Phase 3: Historical data research and acquisition

### Subphase 3.1: Discover and score 2005-2025 sources
- Commit: add a historical-source registry and evidence-backed acquisition recommendation.
- Tests: sampled source URLs/files are reachable and parsed fields match documented semantics.
- Success Criteria: source matrix records years, meetings, races, runners, results, dividends, odds timestamp quality, horse history, cost, licensing, access controls, and known gaps.
- Planned Touch Files: `docs/data/HISTORICAL_SOURCE_RESEARCH.md`, `docs/data/source-registry.json`, `scripts/audit_historical_sources.py`.
- Checklist:
  - [x] Research official HKJC historical pages and downloadable/archive interfaces.
  - [x] Research public academic/Kaggle/GitHub datasets and commercial or licensed feeds.
  - [x] Separate closing/final odds from timestamped pre-post odds and reject ambiguous fields.
  - [x] Sample multiple seasons and venue/race types to verify coverage claims.
  - [x] Document access, robots/terms, rate limits, geography, authentication, and retention constraints.

### Subphase 3.2: Prototype historical HKJC collection
- Commit: add checkpointed, rate-limited historical race and horse archive collectors.
- Tests: fixture parser tests, resumability, duplicate reconciliation, and sampled live archive verification.
- Success Criteria: a bounded season/date-range scrape writes immutable raw pages plus normalized races/runners/results/dividends with provenance.
- Planned Touch Files: `scrapper/historical/**`, `scrapper/tests/historical/**`, `scripts/scrape_history.py`.
- Checklist:
  - [x] Discover archive navigation and stable race/horse identifiers through the browser.
  - [x] Implement date/meeting/race enumeration with retries, checkpoints, and conservative rate limits.
  - [x] Scrape race cards, results, dividends, sectional data, and point-in-time horse records where available.
  - [ ] Detect revisions, scratches, dead heats, abandoned races, coupled entries, and identifier changes.
  - [x] Produce a raw-to-canonical reconciliation report for a pilot season.

### Subphase 3.3: Build the historical coverage manifest
- Commit: validate and merge approved sources into a versioned bronze/silver dataset.
- Tests: row-count reconciliation, uniqueness, winner invariants, date bounds, field-level missingness, and source conflict reports.
- Success Criteria: approved 2005-2025 data has explicit coverage by year and field; unresolved gaps remain visible rather than silently imputed.
- Planned Touch Files: `ima/ingestion/historical.py`, `ima/validation.py`, `docs/data/HISTORICAL_COVERAGE.md`, `data/manifests/**`.
- Checklist:
  - [x] Preserve immutable raw source artifacts and checksums.
  - [x] Normalize identifiers and schema without overwriting source values.
  - [x] Reconcile overlapping sources with deterministic precedence and conflict logs.
  - [x] Version each dataset build and retain reproducible acquisition parameters.

## Phase 4: Offline feature pipeline and evaluation

### Subphase 4.1: Build point-in-time race dataset
- Commit: canonical historical feature builder and chronological race splits.
- Tests: no future rows, one winner per race, disjoint ordered race splits.
- Success Criteria: every model consumes one canonical runner table with target and timestamps.
- Planned Touch Files: `ima/data.py`, `tests/test_data.py`, `scripts/train.py`.
- Checklist:
  - [ ] Build pre-race and historical-only features.
  - [ ] Split by ordered meetings/races, never runner rows.

## Phase 5: Models calibration and market blend

### Subphase 5.1: Benchmark race probabilities
- Commit: market baseline, race-normalized logit, boosted model, metrics, calibration, and blend.
- Tests: race probabilities sum to one; calibration/blend fit excludes test data.
- Success Criteria: reproducible JSON report compares all candidates on held-out races.
- Planned Touch Files: `ima/modeling.py`, `ima/evaluation.py`, `tests/test_modeling.py`, `scripts/train.py`.
- Checklist:
  - [ ] Train and evaluate baseline models.
  - [ ] Calibrate and tune market blend on validation only.
  - [ ] Record model recommendation and deep-learning gate.

### Subphase 5.2: Model finish orders and rank pool combinations
- Commit: add joint finish-order inference and pool-specific combination ranking.
- Tests: valid-combination generation, permutation invariance, probability conservation, dead-heat handling, scratches, and deterministic simulation seeds.
- Success Criteria: each supported pool returns ranked valid combinations with calibrated probabilities derived from one coherent race distribution.
- Planned Touch Files: `ima/modeling/order.py`, `ima/pools.py`, `ima/prediction.py`, `tests/test_order_model.py`, `tests/test_pools.py`.
- Checklist:
  - [ ] Produce horse latent strengths and a joint finish-order distribution using Plackett-Luce as the first benchmark.
  - [ ] Generate exact and unordered combination probabilities analytically where practical and by convergence-tested Monte Carlo otherwise.
  - [ ] Support WIN and PLACE first, then QUINELLA, QUINELLA PLACE, TRIO, TIERCE, FIRST 4, QUARTET, and configured multi-race combinations.
  - [ ] Remove scratched runners and invalidate affected combinations before scoring or wagering.
  - [ ] Calibrate each pool separately because marginal horse calibration does not guarantee calibrated exotic-combination probabilities.
  - [ ] Rank combinations by probability, expected value, uncertainty-adjusted edge, and portfolio contribution rather than probability alone.

### Subphase 5.3: Estimate fair odds, market odds, and pari-mutuel dividends
- Commit: add fair-price conversion, market/dividend forecasting, and discrepancy analysis.
- Tests: fair-odds identities, overround/takeout normalization, no zero-probability division, timestamp alignment, and payout interval coverage.
- Success Criteria: reports clearly distinguish outcome probability, fair odds, currently displayed market information, and projected final dividend with uncertainty intervals.
- Planned Touch Files: `ima/market.py`, `ima/modeling/odds.py`, `ima/reporting.py`, `tests/test_market.py`, `tests/test_odds_model.py`.
- Checklist:
  - [ ] Convert calibrated combination probability `p` to decimal fair odds `1 / p` before risk margins.
  - [ ] Normalize displayed market prices into market-implied probabilities after accounting for pool takeout and available pool totals.
  - [ ] Train a separate timestamp-aware model for final odds/dividends using current odds trajectory, pool liquidity, race state, scratches, and historical betting patterns.
  - [ ] Predict a distribution or interval for final dividend, not only a point estimate.
  - [ ] Report fundamental-versus-market probability residual, fair-odds-versus-current-price gap, and projected-final-versus-current-price movement.
  - [ ] Prevent closing odds, final pool totals, or post-race dividends from entering pre-race outcome features.
  - [ ] Evaluate odds forecasts with proper timestamped error metrics and interval coverage; evaluate outcome probabilities separately with log loss, Brier score, and calibration.

## Phase 6: Wager strategy, outcome tracking, and continual training

### Subphase 6.1: Risk-constrained recommendations and execution port
- Commit: constrained Kelly optimizer, paper ledger, and disabled live HKJC adapter interface.
- Tests: no negative stakes, exposure caps, no bet below edge threshold, duplicate prevention, live fail-closed.
- Success Criteria: WIN recommendations are deterministic and all-pool contracts exist; no live wager can occur without explicit enablement.
- Planned Touch Files: `ima/strategy.py`, `ima/execution.py`, `tests/test_strategy.py`, `tests/test_execution.py`.
- Checklist:
  - [ ] Implement risk-budget sizing and recommendation audit record using ranked pool combinations, payout uncertainty, correlated exposures, and pool-specific limits.
  - [ ] Implement paper execution and secure live adapter boundary.

### Subphase 6.2: Reconcile predictions and wagers with official results
- Commit: add official-result ingestion, immutable prediction ledger, and race-level performance reports.
- Tests: deterministic result matching, dead-heat/dividend handling, no mutation of pre-race records, and bankroll ledger reconciliation.
- Success Criteria: each completed race links the exact pre-race snapshot and model version to finish order, official dividends, realized return, and calibration/ranking errors.
- Planned Touch Files: `ima/outcomes.py`, `ima/ledger.py`, `ima/reporting.py`, `scrapper/results.py`, `tests/test_outcomes.py`, `tests/test_ledger.py`.
- Checklist:
  - [ ] Persist model version, dataset version, feature timestamp, odds timestamp, probabilities, edge, stake, execution status, and idempotency key before post time.
  - [ ] Fetch and reconcile official finish order, scratches, dead heats, pool dividends, and abandoned races after declaration.
  - [ ] Calculate race log loss, Brier score, calibration residual, ranking metrics, closing-line comparison, expected value, realized return, and bankroll change.
  - [ ] Compare predicted combination ranking, fair odds, projected final dividend, displayed pre-race odds, official dividend, and actual winning combination for every supported pool.
  - [ ] Produce meeting, rolling 30/90-day, venue, distance, class, odds-band, and model-version reports.
  - [ ] Keep prediction quality separate from execution quality and betting return.

### Subphase 6.3: Enrich data and retrain after completed meetings
- Commit: add meeting-finalization, feature refresh, model registry, and champion/challenger training orchestration.
- Tests: point-in-time cutoff enforcement, reproducible dataset build, failed-job recovery, and promotion-gate tests.
- Success Criteria: each finalized meeting appends verified outcomes to a new dataset version, trains challengers, evaluates them on untouched rolling holdouts, and promotes only models satisfying configured gates.
- Planned Touch Files: `ima/training/pipeline.py`, `ima/training/registry.py`, `ima/training/promotion.py`, `scripts/finalize_meeting.py`, `scripts/retrain.py`, `tests/training/**`.
- Checklist:
  - [ ] Treat a completed meeting, not an individual race, as the retraining transaction boundary.
  - [ ] Run approximately twice weekly after official results and dividends stabilize.
  - [ ] Append newly observed horse form, ratings, weights, jockey/trainer relationships, veterinary, movement, trackwork, odds, and outcomes with source timestamps.
  - [ ] Recompute only point-in-time-safe historical features and retain dataset/model lineage.
  - [ ] Train conditional-logit/race-normalized and boosted-tree challengers; run deeper models only when their data and compute gates are met.
  - [ ] Compare challenger against the production champion, public-market baseline, and no-bet strategy on recent and long-window holdouts.
  - [ ] Require non-inferior calibration and log loss, minimum sample size, bounded turnover, and risk-adjusted return evidence before promotion.
  - [ ] Keep automatic rollback available and require manual approval before a promoted model can submit live wagers.

### Subphase 6.4: Detect drift and control retraining risk
- Commit: add data-quality, feature-drift, calibration-drift, and strategy-drift monitoring.
- Tests: synthetic drift alerts, stale-odds rejection, missing-source quarantine, and model rollback smoke tests.
- Success Criteria: bad or incomplete meeting data cannot contaminate training or silently change production wagering behavior.
- Planned Touch Files: `ima/monitoring.py`, `ima/training/quality.py`, `tests/test_monitoring.py`, `docs/MODEL_GOVERNANCE.md`.
- Checklist:
  - [ ] Alert on schema changes, missing runners, stale odds, identifier mismatches, abnormal field distributions, and result reconciliation failures.
  - [ ] Monitor probability calibration, market-relative performance, abstention rate, exposure, drawdown, and realized-versus-expected value.
  - [ ] Quarantine incomplete meetings and rerun finalization when corrected official data arrives.
  - [ ] Version and retain every production model, training dataset, configuration, metric report, and promotion decision.

## Phase 7: Hong Kong deployment and end-to-end verification

### Subphase 7.1: Package operator workflow
- Commit: deployment service definitions, environment contract, and runbook.
- Tests: clean install, CLI smoke, config validation, live scrape, training report, paper recommendation.
- Success Criteria: documented commands run on a Hong Kong network host and produce auditable artifacts.
- Planned Touch Files: `deploy/**`, `.env.example`, `README.md`, `docs/OPERATIONS.md`, `scripts/smoke.py`.
- Checklist:
  - [ ] Add secrets/risk configuration and service templates.
  - [ ] Run end-to-end live scrape to paper-decision verification.
  - [ ] Schedule pre-race collection, post-meeting reconciliation, twice-weekly retraining, evaluation, and guarded promotion as separate jobs.
