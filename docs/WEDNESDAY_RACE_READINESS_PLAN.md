# Wednesday Race Readiness

## GOAL
- Prepare iMa for the next Wednesday Hong Kong race meeting, expected Wednesday 2026-09-23 Hong Kong time unless the official HKJC racecard says otherwise, so it can collect live data, produce human-operator betting recommendations, record pre-race decisions, settle official results, and iterate models in a controlled self-learning loop.

## Out of Scope
- Automated live wager submission.
- A complicated dashboard or rich race-day UI.
- Any race-day interface that requires browsing multiple screens, interpreting charts, or manually joining artifacts before seeing the estimate.
- Storing HKJC credentials, browser profiles, screenshots with account data, or payment/session tokens in the repository.
- Promoting a challenger model to champion without held-out evaluation against the current champion and market baseline.
- Claiming a real betting edge from backtests alone.

## Acceptance Criteria
- [ ] Environment preflight passes with Python 3.11, project dependencies, Chrome or Chromium, and the test suite subset needed for live collection, modeling, pools, strategy, ledger, and operations.
- [ ] The official next meeting is read back from HKJC through the scraper/browser path; race date, venue, race count, first post, and runner counts are written to an operations artifact.
- [ ] A pinned champion model artifact is selected and its feature schema is compatible with freshly scraped Wednesday runners.
- [ ] Pre-race snapshots preserve raw HKJC data, normalized runners, model-ready CSV, and all available pool prices for every race that can be collected.
- [ ] Human-facing recommendations are generated with probability, fair odds, market odds, expected value, proposed stake, reason, risk cap, and "do not bet" rows when edge or data quality is insufficient.
- [ ] Race-day operation is deterministic and terminal-first: the system identifies the active/next race or accepts an explicit race override, the operator runs one estimate command, and the output is a simple ranked table plus JSON readback.
- [ ] Any graphical race-day surface, if added later, is only a one-button wrapper around the same deterministic terminal command: pull race, estimate, show ranked probabilities; no complex custom dashboard.
- [ ] Every recommendation accepted or rejected by the human operator is recorded before post time in a ledger with timestamp, race, pool, combination, stake, odds snapshot, model version, and operator action.
- [ ] No code path submits a live HKJC wager; `HKJCWebExecutor` remains fail-closed unless a separate authenticated execution project is approved.
- [ ] After official declaration, results and dividends are collected, the ledger is settled, and profit/loss plus calibration diagnostics are generated.
- [ ] An agentic self-learning loop proposes one bounded experiment at a time, changes hyperparameters, feature sets, transformations, dataset construction, model family, calibration, or market-blend settings, runs the experiment, reads back metrics, and records the next decision.
- [ ] Every agent-generated experiment has a hypothesis, exact data window, leakage guard, changed parameters/features, expected metric movement, cost budget, result summary, and follow-up decision.
- [ ] The loop trains challengers from finalized data, compares them on untouched recent races and market baseline, logs results to MLflow or the registry, and requires operator approval before promotion.
- [ ] A final race-day report can be opened locally and read back with meeting summary, predictions, operator decisions, settlement status, and next model action.

## Initial Repo Audit Findings
- Existing live paths: `scrapper/cli.py`, `scrapper/pipeline.py`, `scrapper/browser_odds.py`, `scripts/smoke.py`, `scripts/simulate_tomorrow.py`, `scripts/predict_pools.py`, and `scripts/paper_trade.py`.
- Existing learning paths: `scripts/finalize_meeting.py`, `ima/operations.py`, `ima/training.py`, `ima/registry.py`, `ima/mlflow_tracking.py`, and MLflow commands in `docs/OPERATIONS.md`.
- Existing experiment paths: `scripts/run_experiments.py`, `scripts/run_benter_grid.py`, `scripts/run_feature_study.py`, `scripts/run_schema_feature_study.py`, `scripts/run_experiments.py`, `scripts/enrich_pool_metrics.py`, `scripts/export_mlflow_dashboard.py`, and `ima/experiments.py`.
- Existing safety boundary: `ima/execution.py` has `PaperExecutor` and a fail-closed `HKJCWebExecutor`.
- Existing domain logic: `ima/simulator.py`, `ima/strategy.py`, `ima/pools.py`, `ima/market.py`, `ima/inference.py`, `ima/modeling.py`, and `ima/feature_sets.py`.
- Existing tests: operation, inference, data, browser pools, ledger registry, pools, modeling, strategy, simulator, MLflow tracking, auxiliary models, bulk history, and historical-source tests.
- Current benchmark warning from `README.md`: tested fundamental models underperform final WIN market in recent held-out races; the system should treat no-edge/abstention as a valid outcome.

## Research
- Built-in options:
  - Reuse repo-native scrape, simulate, paper-trade, finalize, train, registry, and MLflow commands before adding new workflow code.
  - Reuse the fail-closed execution boundary and paper ledger for human-operator bets.
- Off-the-shelf options:
  - Use scikit-learn/joblib pipelines already declared in `pyproject.toml` for champion/challenger artifacts.
  - Use MLflow already documented for experiment tracking and model/run comparison.
  - Use Playwright/Chromium already declared under dev dependencies for browser-backed HKJC collection checks.
- Official / standards sources:
  - Official HKJC racecard, odds, results, and dividend pages are the race-day source of truth.
  - Local `docs/OPERATIONS.md` is the current operational standard for live collection, credentials, and meeting cycle.
  - Local `README.md` is the current standard for benchmark interpretation and no-edge caveats.

## SOTA, Standards, And Best Practices
- Current SOTA / prior art:
  - Parimutuel horse-race betting systems should separate a fundamental probability model from market-implied probabilities, then wager only when calibrated probability beats price after takeout and risk limits.
  - Benter-style practice favors rigorous feature history, market blending, calibration, bet sizing discipline, and continuous post-race feedback rather than blind model confidence.
  - Modern model iteration should use time-aware validation, champion/challenger comparisons, and immutable pre-race snapshots to avoid leakage.
  - Agentic model optimization should behave like a disciplined lab assistant, not an unconstrained optimizer: it proposes a bounded hypothesis, changes one coherent unit, runs a reproducible experiment, reads metrics, records the result, and chooses the next move from evidence.
  - AutoML-style search is useful for hyperparameters and model families, but horse-racing leakage risk means the agent must also own dataset/version provenance, feature timing, and market-baseline comparisons.
- Official docs, standards, specs, or platform guidance:
  - HKJC displayed racecard, odds, official results, and dividends must be treated as current truth and rechecked on race day.
  - macOS Keychain or an external vault is required for any future credentials; source, logs, screenshots, and artifacts must remain credential-free.
- Mature libraries, frameworks, or built-in mechanisms:
  - Existing Python package commands, Playwright/Chromium, pandas/scikit-learn/joblib, MLflow, `PaperExecutor`, and `ModelRegistry`.
- Best-practice constraints to integrate:
  - Prediction-time features only; no final odds/results/dividends in pre-race decisions.
  - Timestamp all odds and decisions.
  - Record abstentions.
  - Cap per-race, per-combination, and total bankroll exposure.
  - Use expected value and calibration diagnostics, not just top-pick accuracy.
  - Keep live submission separate from collection/modeling.
  - Prefer a boring deterministic CLI over a custom UI for race day; terminal output is easier to audit, copy, rerun, and debug under time pressure.
  - If a UI is ever introduced, it must be a thin shell over the CLI and must not own racing logic, model logic, state transitions, or wager decisions.
  - Make each agentic experiment atomic: one hypothesis, one changed surface, one reproducible command, one result readback, one next-action decision.
  - Keep a hard validation firewall: the agent may propose features and transforms, but it may not use post-race outcomes, dividends, final odds, or future race information in pre-race features.
  - Bound compute and search: maximum trials, timeout, metric gates, and early stopping must be explicit before the loop runs.
- Rejected approaches:
  - Reject fully automated live betting for Wednesday because authenticated execution, MFA, receipt parsing, exposure limits, and kill switch are not complete.
  - Reject a complex race-day dashboard because the operator needs speed, determinism, and auditability more than visualization.
  - Reject any UI-specific prediction path because it risks diverging from the terminal and artifact workflow.
  - Reject retraining on Wednesday results before settlement and validation because it leaks post-race information into decision-making.
  - Reject using legacy notebook joblibs as trusted production artifacts without provenance and compatibility checks.
  - Reject "always bet the top pick"; no-edge and data-quality abstentions are required.
  - Reject an unconstrained agent that blindly mutates many parameters/features per run, because the result cannot be attributed or trusted.
  - Reject selecting the best backtest by ROI alone; high-variance betting returns need calibration, log loss, market comparison, drawdown, and stability gates.
- Implementation decision:
  - Operate Wednesday in supervised paper/human mode through a terminal-first deterministic command: pull the current/selected race, estimate ranked outcomes, let the human place or skip bets, record the actual decision ledger, settle from official results, then run a bounded agentic champion/challenger learning loop after the meeting.

## Dependency and Tooling Preflight
- Required verification tools:
  - `.venv/bin/python`, Python 3.11, project editable install with dev dependencies, Chrome or Chromium, Playwright browser support, unittest, and optional MLflow.
- Existing tooling found:
  - `pyproject.toml` declares `ima-racing`, `ima-scrape`, and dev Playwright.
  - `README.md` and `docs/OPERATIONS.md` document setup, live scrape, simulation, prediction, finalize, and training commands.
- Install or repair commands:
  - `python3.11 -m venv .venv`
  - `.venv/bin/python -m pip install --upgrade pip`
  - `.venv/bin/python -m pip install -e '.[dev]'`
  - `.venv/bin/python -m playwright install chromium`
- Browser/runtime binaries:
  - Chrome or Chromium is required because direct HKJC GraphQL calls are documented as rejected with `WHITELIST_ERROR`.
- Real blockers that would prevent setup:
  - HKJC public site unavailable, egress not accepted by HKJC, browser binary unavailable after install attempt, missing local Python 3.11, or incompatible model artifacts.

## Deterministic Real-User Test
- Entry point:
  - A single race-day terminal command from the repository root, for example `ima-raceday estimate`.
- User workflow:
  - Set up environment, run fixture-backed scrape, run unit tests, run live race smoke, run one estimate command, read the ranked terminal table, inspect the generated JSON artifact only when needed, and record one paper/operator decision through the ledger path.
- Stable inputs or fixtures:
  - `scrapper/tests/fixtures/race_snapshot.json` for fixture-backed scrape.
  - Existing checked-in historical samples and test fixtures for strategy/model/ledger tests.
- User-observable assertions:
  - Commands exit zero.
  - Terminal output contains race number/date/venue, model version, submission disabled, and a ranked horse table with win probability, fair odds, market odds, edge, and ratable/data-quality status.
  - Generated artifacts contain race identifiers, runners, probabilities, recommendations or abstentions, and submission disabled.
  - Ledger records are idempotent for duplicate decision IDs.
- Command:
  - `.venv/bin/python -m unittest tests.test_operations_inference tests.test_strategy tests.test_simulator tests.test_ledger_registry tests.test_browser_pools -v`
  - `.venv/bin/ima-scrape --race 1 --fixture scrapper/tests/fixtures/race_snapshot.json --output artifacts/readiness/fixture-scrape`
  - `.venv/bin/ima-raceday estimate --race auto --model <champion> --output artifacts/race-day/2026-09-23/terminal`
- Evidence to record:
  - Megaskill evidence records for setup, tests, fixture scrape, live smoke, simulation, and artifact readback.

## Fulfillment and Readback Proof
- Requested final user-visible result:
  - Terminal output plus `artifacts/race-day/2026-09-23/` containing official meeting readback, per-race snapshots, prediction JSON, human decision ledger, settlement report, and learning report.
- Required write/mutation operation:
  - Write pre-race snapshots before each race.
  - Write prediction recommendations before post time.
  - Append operator decisions to the paper/human ledger before post time.
  - Write official results and settlement after declaration.
  - Register challenger model artifacts only after validation.
- Readback surface:
  - JSON/CSV files under `artifacts/race-day/2026-09-23/`, `artifacts/predictions/`, `artifacts/simulations/`, ledger JSONL, model registry records, and MLflow/exported dashboard if enabled.
- Expected content, rows, fields, counts, or behavior:
  - Meeting artifact has official date, venue, and race count.
  - Each collected race has runner count greater than zero and normalized model rows.
  - Terminal prediction rows include rank, horse number/name when available, win probability, optional fastest/auxiliary estimate when available, fair odds, market odds, probability edge, model version, and data-quality flags.
  - JSON prediction rows include probability, market probability or odds, fair odds, EV/discrepancy, stake if recommending, pool, combination, model version, and data-quality flags.
  - Ledger rows include operator action and remain append-only.
  - Settlement joins recommendations to official finishing position/dividends and reports P/L.
  - Learning report records champion, challengers, experiment hypotheses, agent decisions, changed hyperparameters/features/transforms, evaluation metrics, promotion status, and approval requirement.
- Command or tool call evidence:
  - `scripts.smoke --live`, `ima-scrape`, `scripts.predict_pools`, `scripts.simulate_tomorrow`, `scripts.paper_trade`, `scripts.finalize_meeting`, and test commands.
- Not-done conditions:
  - Official race date is not read back.
  - Predictions are generated from stale snapshots.
  - Recommendations lack odds or stake arithmetic.
  - Race-day estimates require a complex UI or manual artifact joining before the operator can see ranked probabilities.
  - Terminal and JSON outputs disagree.
  - Operator decisions are not recorded before post time.
  - Results are collected but not settled against the ledger.
  - Challenger model is trained but not compared on untouched recent races.
  - Agentic loop changes multiple dimensions without an experiment record.
  - Agentic loop chooses a winner without reading back metrics and comparing to market/champion baselines.
  - Any code path submits live wagers.

## Armageddon Mode
- Damage-scaled attack scope:
  - Moderate for Wednesday readiness because it touches live scraping, prediction artifacts, operator decisions, and post-race learning, while keeping live wagering disabled.
- Edge cases to try:
  - No displayed Wednesday meeting.
  - Displayed meeting date differs from requested date.
  - Scrape fails on race 1.
  - One race has inactive/scratched runners.
  - Missing horse pages or overseas simulcast without HK horse IDs.
  - Missing pool odds for exotic markets.
  - Bankroll too small, zero EV, all abstentions.
  - Duplicate operator decision ID.
  - Official results unavailable or partial.
  - Terminal width is narrow; output must remain readable or degrade to simple rows.
  - Operator explicitly overrides race number when auto-detect selects the wrong race.
- Failure modes to simulate:
  - Wrong model path.
  - Incompatible joblib.
  - Missing feature columns.
  - Agent proposes feature that is not available before post time.
  - Agent proposes transform that leaks future outcome or same-race final market information.
  - Agent overfits by selecting the top historical ROI run with weak log loss/calibration.
  - Agent repeats failed experiment because prior run state was not read.
  - Browser launch failure.
  - HKJC egress rejection.
  - Network timeout.
  - Corrupt snapshot JSON.
  - Ledger file already exists with prior decisions.
  - Auto race detection cannot identify a race; command must fail closed with the exact attempted races.
- Optimization opportunities to inspect:
  - One terminal command should orchestrate the race-day estimate if current scripts require too many manual joins.
  - A second command should orchestrate agentic experiment loops with a resumable run ledger.
  - Prediction reports should separate "estimate", "bet candidate", "watch", and "abstain" with reasons, but the terminal view should stay compact.
  - Settlement should produce calibration and EV diagnostics automatically.
  - Live snapshots should be timestamped densely enough to support future pool-level market blending.
- Must-fix threshold:
  - Any live submission path, stale-date prediction, missing ledger write, no artifact readback, complex mandatory UI, leakage from final results into pre-race predictions, unbounded agent loop, or promotion without validation blocks Wednesday readiness.
- Evidence to record:
  - Command logs, artifact readbacks, error snapshots, and a concise race-day report.

## Generality Guardrail
- Existing mechanism to reuse:
  - `scrapper.pipeline.scrape_race`, `scripts.simulate_tomorrow`, `scripts.predict_pools`, `PaperExecutor`, `ModelRegistry`, MLflow export, `finalize_meeting`, `scripts.run_experiments`, `scripts.run_benter_grid`, feature-study scripts, and `ima.experiments`.
- Similar existing use cases:
  - Live scrape checks, pool prediction, tomorrow simulation, paper trading, official-history finalization, and retraining.
- Recurrence likelihood:
  - High; Hong Kong meetings recur weekly, and the operator needs the same cycle every race day.
- General mechanism to create if none exists:
  - A single terminal-first race-day command should own active-race detection, scrape, estimate, ranked terminal output, JSON readback, and optional ledger recording instead of adding one-off Wednesday scripts.
  - A single agentic experiment controller should own propose -> run -> evaluate -> decide -> record; individual experiments should not become ad hoc notebooks.
- Specific use case implementation through the mechanism:
  - Wednesday 2026-09-23 becomes one configuration of the recurring race-day workflow: date, bankroll, model version, max race, pools, and output directory.
  - Post-Wednesday learning becomes one bounded campaign configuration: data cutoff, validation window, allowed model families, allowed feature families, max trials, metric gates, and promotion policy.
- One-off justification, if any:
  - Only the plan artifact is one-off; code should be reusable for future Wednesday/Sunday meetings.

## Ordered State and Dashboard
- State ledger path:
  - `.mega/state.jsonl`
- Dashboard output path:
  - `.mega/dashboards/WEDNESDAY_RACE_READINESS_PLAN.html`
- Open behavior:
  - Regenerate the dashboard after plan edits and open it in Chromium for operator review.
- State events to record:
  - Plan created, audit completed, guardrails accepted, tooling preflight, fixture tests, live smoke, prediction artifact, operator ledger, settlement, learning report, blockers.
- Evidence links:
  - `.mega/evidence.jsonl`
- Dashboard readback assertions:
  - Dashboard contains the goal, acceptance criteria, Wednesday date, fail-closed execution boundary, and all five phases.

## Regression Guardrails
- Planned edit surface:
  - Planning turn: `docs/WEDNESDAY_RACE_READINESS_PLAN.md` and generated `.mega/dashboards/WEDNESDAY_RACE_READINESS_PLAN.html`.
  - Execution turns: prefer a recurring orchestrator plus focused tests; avoid broad rewrites of model, scraper, or strategy internals unless a failing gate proves they are needed.
- Protected behaviors:
  - Paper mode stays default.
  - Live execution remains disabled.
  - Existing tests and CLI entry points keep working.
  - Historical benchmarks and source-registry boundaries remain intact.
  - No secrets enter source, logs, screenshots, or artifacts.
- Likely consumers:
  - Human race-day operator, future scheduled jobs, model researcher, settlement process, registry/MLflow dashboard, and any future authenticated executor.
- Damage radius:
  - Planning: small.
  - Full readiness implementation: moderate.
  - Authenticated live wagering, if ever approved: systemic and must be a separate branch/project phase.
- Branch strategy:
  - Current branch is acceptable for this planning artifact.
  - A dedicated work branch is recommended before implementing the full recurring race-day orchestrator.
- Proof plan:
  - Plan check, dashboard readback, targeted tests, fixture scrape, live smoke, race-day artifact readback, ledger readback, settlement readback, and learning report readback.

## Phase 1: Audit

### Subphase 1.1: Map Current Capabilities and Gaps
- Commit: `docs: add Wednesday race readiness plan`
- Tests: `python3.11 /Users/milkingthesun/.codex/skills/megaskill/scripts/mega_plan_check.py docs/WEDNESDAY_RACE_READINESS_PLAN.md`
- Success Criteria: The plan names the current repo capabilities, missing readiness gaps, protected safety boundaries, and exact next execution commands.
- Planned Touch Files:
  - `docs/WEDNESDAY_RACE_READINESS_PLAN.md`
  - `.mega/dashboards/WEDNESDAY_RACE_READINESS_PLAN.html`
- Checklist:
  - [ ] Inspect README, operations docs, race-day scripts, execution boundary, and tests.
  - [ ] State acceptance criteria for Wednesday readiness.
  - [ ] Regenerate and open the plan dashboard in Chromium.

### Subphase 1.2: Prove Local Tooling Baseline
- Commit: `test: record race-day readiness baseline`
- Tests: `.venv/bin/python -m unittest tests.test_operations_inference tests.test_strategy tests.test_simulator tests.test_ledger_registry tests.test_browser_pools -v`
- Success Criteria: A focused readiness suite passes, or each blocker is classified as setup, browser, live HKJC, model artifact, or code defect.
- Planned Touch Files:
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
  - `artifacts/readiness/`
- Checklist:
  - [ ] Confirm `.venv` exists or create it using README commands.
  - [ ] Install editable dev dependencies if missing.
  - [ ] Install Chromium browser support if needed.
  - [ ] Run the focused readiness test suite and record evidence.

## Phase 2: Readiness

### Subphase 2.1: Verify Official Meeting and Live Scrape Path
- Commit: `ops: capture Wednesday meeting readiness snapshot`
- Tests: `.venv/bin/python -m scripts.smoke --live --race 1`
- Success Criteria: The official next displayed HKJC meeting is read back, race 1 can be scraped, and a meeting summary artifact records date, venue, races, runner counts, and source URLs/paths.
- Planned Touch Files:
  - `artifacts/race-day/2026-09-23/meeting.json`
  - `artifacts/race-day/2026-09-23/snapshots/`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Run live smoke from accepted Hong Kong egress.
  - [ ] If displayed date differs, stop predictions unless `--allow-next-available` is an intentional operator decision.
  - [ ] Scrape race 1 with all pools and no credentials.
  - [ ] Read back raw snapshot, runner CSV, model CSV, and report JSON.

### Subphase 2.2: Select Champion Model and Validate Feature Compatibility
- Commit: `ops: pin Wednesday champion model`
- Tests: `.venv/bin/python -m scripts.predict_pools --model <champion> --runners <fresh model.csv> --pools WIN PLACE QIN QPL TRI TIERCE FIRST4 QUARTET --top 20 --output artifacts/race-day/2026-09-23/predictions/race-1-pools.json`
- Success Criteria: The selected model loads, predicts fresh runners, emits pool candidates, and fails clearly if required features are missing.
- Planned Touch Files:
  - `artifacts/race-day/2026-09-23/model-selection.json`
  - `artifacts/race-day/2026-09-23/predictions/`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Prefer the documented Benter-rich champion if present and compatible.
  - [ ] Fall back only to a freshly trained artifact if the champion is missing or incompatible.
  - [ ] Record model path, checksum, training/evaluation summary, and known caveats.
  - [ ] Validate prediction output against runner count and pool availability.

## Phase 3: Terminal-First Estimation

### Subphase 3.1: Define the Minimal Race-Day Terminal Command
- Commit: `feat: add deterministic race-day estimate command`
- Tests: `.venv/bin/ima-raceday estimate --fixture scrapper/tests/fixtures/race_snapshot.json --race 1 --model <test-model> --output artifacts/readiness/terminal`
- Success Criteria: One command prints a simple ranked table and writes matching JSON without requiring any custom dashboard.
- Planned Touch Files:
  - `pyproject.toml`
  - `ima/raceday.py`
  - `scripts/raceday.py`
  - `tests/test_raceday.py`
  - `artifacts/readiness/terminal/`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Command name and behavior: `ima-raceday estimate`.
  - [ ] Inputs: `--race auto|N`, `--model`, `--output`, optional `--date`, `--venue`, `--top`, `--fixture`, `--without-horse-pages`.
  - [ ] Auto mode deterministically probes races in order and chooses the first currently available race with active runners and WIN odds; explicit `--race N` overrides auto.
  - [ ] Terminal output shows only the essentials: race/date/venue, model version, submission disabled, then ranked horses.
  - [ ] Ranked rows show horse number/name when available, win probability, fair odds, market odds, probability edge, ratable/data-quality status, and optional auxiliary fastest/finish-time estimate when an auxiliary model is supplied.
  - [ ] JSON output exactly matches the terminal ranking and includes snapshot paths for audit.
  - [ ] No browser UI, HTML dashboard, or chart is required for race-day estimation.

### Subphase 3.2: Generate Human-Operator Bet Candidates Without Hiding Abstentions
- Commit: `feat: add compact terminal bet-candidate summary`
- Tests: `.venv/bin/ima-raceday estimate --fixture scrapper/tests/fixtures/race_snapshot.json --race 1 --model <test-model> --bankroll 1000 --output artifacts/readiness/terminal`
- Success Criteria: The terminal output optionally shows compact bet candidates or "no bet" reasons, while the ranking remains visible and deterministic.
- Planned Touch Files:
  - `ima/raceday.py`
  - `scripts/raceday.py`
  - `tests/test_raceday.py`
  - `artifacts/readiness/terminal/`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Generate estimates before post time from fresh odds snapshots.
  - [ ] Make stake math explicit and capped by bankroll rules when bet candidates are enabled.
  - [ ] Include "skip" reasons for low edge, missing odds, scratches, unratable rows, or data quality.
  - [ ] Keep live submission disabled and visible in terminal output.
  - [ ] Read back JSON and confirm it matches the printed table.

### Subphase 3.3: Record Human Operator Decisions Before Post Time
- Commit: `feat: record supervised race-day decisions`
- Tests: `.venv/bin/python -m scripts.paper_trade --race <race_no> --model <champion> --bankroll <bankroll> --output artifacts/race-day/2026-09-23/paper --without-horse-pages`
- Success Criteria: Every placed or skipped bet has an append-only ledger row with timestamp, source odds, model version, decision ID, operator action, stake, and receipt status.
- Planned Touch Files:
  - `artifacts/race-day/2026-09-23/ledger/orders.jsonl`
  - `artifacts/race-day/2026-09-23/operator-actions.jsonl`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Confirm the human operator, not the system, places any real bets.
  - [ ] Record actual operator accept/reject/modify action before race post.
  - [ ] Preserve odds timestamp used for the decision.
  - [ ] Re-run duplicate decision check and confirm idempotent paper receipts.

## Phase 4: Settlement

### Subphase 4.1: Collect Official Results and Settle Ledger
- Commit: `feat: settle race-day operator ledger`
- Tests: `.venv/bin/python -m scripts.finalize_meeting --date 2026/09/23 --venue <ST|HV> --output data/meetings --artifacts artifacts/models/meetings --registry artifacts/registry`
- Success Criteria: Official results/dividends are read back, ledger rows are matched to outcomes, P/L is calculated, and incomplete official data is quarantined rather than learned from.
- Planned Touch Files:
  - `data/meetings/2026-09-23/`
  - `artifacts/race-day/2026-09-23/settlement.json`
  - `artifacts/race-day/2026-09-23/settlement.csv`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Collect official race results after declaration.
  - [ ] Validate no duplicate races, every race has runners, and every race has an official winner.
  - [ ] Join decisions to official outcomes and dividends.
  - [ ] Mark unsettled races explicitly if results are unavailable.

### Subphase 4.2: Publish Race-Day Report for Review
- Commit: `docs: publish Wednesday race-day report`
- Tests: `test -s artifacts/race-day/2026-09-23/report.html || test -s artifacts/race-day/2026-09-23/report.json`
- Success Criteria: The operator can open one local report and see meeting, predictions, operator actions, settlement, P/L, data quality, and next model action.
- Planned Touch Files:
  - `artifacts/race-day/2026-09-23/report.html`
  - `artifacts/race-day/2026-09-23/report.json`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Summarize recommendations and actual operator actions.
  - [ ] Summarize profit/loss and calibration against official outcomes.
  - [ ] Include data-quality warnings and races excluded from learning.
  - [ ] Open/read back the report.

## Phase 5: Agentic Learning

### Subphase 5.1: Build the Agentic Experiment Contract
- Commit: `feat: define agentic experiment contract`
- Tests: `.venv/bin/python -m unittest tests.test_experiments tests.test_feature_analysis tests.test_mlflow_tracking -v`
- Success Criteria: The learning loop has a machine-readable contract for hypotheses, allowed edit/search surfaces, data windows, leakage checks, commands, metrics, result readback, and next-action decisions.
- Planned Touch Files:
  - `ima/experiments.py`
  - `ima/feature_analysis.py`
  - `ima/mlflow_tracking.py`
  - `scripts/run_experiments.py`
  - `scripts/run_benter_grid.py`
  - `scripts/run_feature_study.py`
  - `scripts/run_schema_feature_study.py`
  - `artifacts/agentic-learning/2026-09-23/experiment-contract.json`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Define the experiment record schema: `experiment_id`, parent run, hypothesis, changed surface, data cutoff, validation window, feature set, transforms, model family, hyperparameters, calibration, market blend, budget, metrics, conclusion, and next action.
  - [ ] Define allowed agent actions: adjust hyperparameters, select existing feature families, propose new pre-race features, add leakage-safe transforms, alter dataset windows, choose model families already supported by the repo, and adjust calibration/blend settings.
  - [ ] Define forbidden actions: use post-race results as features, use final odds/dividends for pre-race decisions, mutate live wager execution, promote a model, delete source data, or run unbounded searches.
  - [ ] Define metric gates: race log loss, top-1/top-3, calibration, Brier score if available, market-baseline delta, expected value, drawdown, stability across venues/distances/classes, and abstention quality.
  - [ ] Add result readback requirements for MLflow/registry/dashboard artifacts.

### Subphase 5.2: Run the First Bounded Agentic Search Campaign
- Commit: `feat: run bounded post-meeting experiment campaign`
- Tests: `.venv/bin/python -m scripts.run_experiments --output artifacts/agentic-learning/2026-09-23/campaign-001`
- Success Criteria: The agent runs a small campaign of sequential experiments where each next trial is selected from the previous result, and every trial is reproducible from its recorded command/config.
- Planned Touch Files:
  - `artifacts/agentic-learning/2026-09-23/campaign-001/`
  - `artifacts/models/wednesday-challengers/`
  - `artifacts/registry/`
  - `artifacts/mlflow/`
  - `public/results.json`
  - `public/results.csv`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Include Wednesday data only after official settlement and validation; otherwise run on the latest finalized historical dataset.
  - [ ] Start from the champion/baseline and record its metrics as trial zero.
  - [ ] Let the agent choose the next bounded move from evidence: hyperparameter search, feature-family toggle, transformation, dataset window, calibration, model family, or market blend.
  - [ ] Limit the first campaign to a small budget, for example 5-20 trials or a fixed runtime, so we learn safely before spending compute.
  - [ ] Require each trial to change one coherent surface so attribution is clear.
  - [ ] Preserve timestamped odds for future market-blend learning.
  - [ ] Log every trial, metric, artifact, and decision to MLflow/registry plus local JSONL.
  - [ ] Stop early if all challengers underperform champion and market baseline, or if validation metrics degrade materially.

### Subphase 5.3: Agent Evaluates and Chooses the Next Experiment
- Commit: `feat: add experiment decision readback`
- Tests: `.venv/bin/python -m scripts.export_mlflow_dashboard --tracking-uri "$IMA_MLFLOW_URI" --experiment ima-racing --base-results public/results.json`
- Success Criteria: The agent produces a decision memo after the campaign: promote candidate, continue search with a named next hypothesis, collect more data, or stop because no edge is proven.
- Planned Touch Files:
  - `artifacts/agentic-learning/2026-09-23/campaign-001/decision.json`
  - `artifacts/agentic-learning/2026-09-23/campaign-001/decision.md`
  - `artifacts/race-day/2026-09-23/learning-report.json`
  - `artifacts/race-day/2026-09-23/learning-report.html`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Read back all trial metrics before deciding.
  - [ ] Explain which metric moved, why the agent believes the change helped or hurt, and what to try next.
  - [ ] Separate statistical/model improvement from tradable betting edge.
  - [ ] If no candidate passes gates, record "no promotion" and the next data/feature need.
  - [ ] If a candidate passes gates, mark it as "candidate awaiting operator approval", not champion.

### Subphase 5.4: Champion/Challenger Decision and Next Meeting Loop
- Commit: `ops: record model promotion decision`
- Tests: `.venv/bin/python -m scripts.export_mlflow_dashboard --tracking-uri "$IMA_MLFLOW_URI" --experiment ima-racing --base-results public/results.json`
- Success Criteria: The learning report compares challenger, champion, and market baseline on untouched recent races; promotion status is either approved by the operator or left as awaiting approval.
- Planned Touch Files:
  - `artifacts/race-day/2026-09-23/learning-report.json`
  - `artifacts/race-day/2026-09-23/learning-report.html`
  - `public/results.json`
  - `public/results.csv`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Compare top-1 accuracy, race log loss, calibration, EV, drawdown, and abstention quality.
  - [ ] Keep champion unchanged if challenger does not beat champion and market baseline gates.
  - [ ] Record operator approval for any promotion.
  - [ ] Convert the agent's next-hypothesis decision into the next campaign config.
  - [ ] Schedule the same supervised race-day workflow and bounded learning loop for the next meeting.
