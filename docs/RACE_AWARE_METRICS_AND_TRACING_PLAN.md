# Race-Aware Metrics and Experiment Tracing

## GOAL
- Implement target-appropriate race-aware metrics and MLflow orchestration traces with token and cost accounting.

## Acceptance Criteria
- [x] Winner-probability evaluation reports race log loss, race Brier, calibration, top-pick/winner-top-k, winner rank, MRR, and race-level dispersion without changing the protected temporal split.
- [x] Placing evaluation reports equal-race-weighted binary log loss and Brier, calibration, and exact top-k selection precision/recall/F1.
- [x] Ranking evaluation reports equal-race-weighted NDCG@3 and full-field NDCG, pairwise accuracy, winner rank/MRR, and race-wise Spearman/Kendall; NDCG@3 replaces global Spearman as the ranking objective.
- [x] Finish-speed evaluation reports equal-race-weighted MAE and RMSE plus race-wise Spearman/Kendall; race-mean MAE remains the objective.
- [x] Every research MLflow run exposes summary, fold, baseline, and shuffled-control metrics using stable searchable names.
- [x] Every completed optimizer cycle can produce one idempotent MLflow trace containing evidence identity, decision source, hypotheses, recipes, trial outcomes, aggregate/best result, planner token usage, and exact USD cost when supplied by OpenRouter.
- [x] Missing provider cost is represented explicitly as unavailable, never silently estimated as zero.
- [x] Focused tests, the complete unit suite, a deterministic fixture campaign, and MLflow run/trace readback pass.

## Research
- Built-in options: extend the existing `research_evaluation`, `secondary_target_diagnostics`, executor fold protocol, controller decision ledger, and MLflow tracking module.
- Off-the-shelf options: use scikit-learn metric implementations where their contracts fit; use MLflow 3 manual tracing APIs and standard token/cost attributes rather than building a trace store.
- Official / standards sources: MLflow manual tracing and token/cost documentation, scikit-learn metrics/calibration documentation, and XGBoost learning-to-rank guidance for query/race grouping and NDCG.

## Root-Cause Baseline
- Trigger scope: not an incident or outage; this is a bounded observability and evaluation enhancement.
- Minimum evidence inventory: current metric implementations, executor objective mapping, MLflow run logger, OpenRouter response persistence, controller lifecycle, tests, and installed MLflow API signatures were inspected.
- Hypothesis model: proven gaps are global ranking Spearman, row-weighted secondary losses, sparse target diagnostics, and no MLflow cycle trace; provider usage already exists in raw planner responses but is only summed for a spend cap.
- Missing-evidence policy: provider-specific live response variants are unavailable locally; accept documented OpenRouter usage shapes and test absent/malformed fields without making live calls.
- Mutation boundary: research evaluation/tracking only; no race-day inference, betting, source data, server service, or credential changes.
- Remediation mapping: race-aware metrics address evaluation mismatch; standard MLflow spans address missing trace visibility; normalized provider usage addresses cost ambiguity.
- Not-done conditions: metrics compare runners across unrelated races, costs default to zero when unknown, traces omit failed trials, or tracing failures stop model training.

## SOTA, Standards, And Best Practices
- Current SOTA / prior art: proper scoring rules for probabilities, NDCG and pairwise concordance for grouped ranking, MAE/RMSE plus within-group rank diagnostics for continuous speed, and multi-metric temporal evaluation against market/control baselines.
- Official docs, standards, specs, or platform guidance: MLflow 3 recommends a connected root trace with nested spans and `mlflow.chat.tokenUsage` / `mlflow.llm.cost`; XGBoost learning-to-rank treats each query as a group, mapping directly to one race.
- Mature libraries, frameworks, or built-in mechanisms: NumPy/pandas/scipy/scikit-learn for metrics; MLflow native tracing and search/readback APIs for observability.
- Best-practice constraints to integrate: equal race weights, finite-value validation, deterministic tie handling, explicit metric direction, temporal folds, baseline parity, idempotent tracking, bounded trace payloads, and no secrets/raw authorization headers.
- Rejected approaches: F1 or accuracy as the sole probability objective; global Spearman across races; direct ROI optimization; custom trace database; inferred OpenRouter price tables; one MLflow model version per trace.
- Implementation decision: retain race log loss for winner probability, use race-mean Brier for top-k placing, use negative race-mean NDCG@3 for ranking, and use race-mean MAE for speed; expose a broader diagnostic suite and trace each orchestration cycle separately from single-model MLflow runs.

## Dependency and Tooling Preflight
- Required verification tools: Python 3.11-3.13, unittest, scikit-learn/scipy, MLflow >=3.10 for cost-aware traces, Optuna for campaign smoke, and Playwright/Chromium for a read-only MLflow UI browser smoke.
- Existing tooling found: `.venv` has MLflow 3.15.0 with `start_span`; project dependencies and research extras are declared; fixture campaign, local SQLite MLflow tests, and Playwright dev dependency already exist.
- Install or repair commands: use `.venv/bin/pip install -e '.[research]'` only if a declared dependency is missing.
- Browser/runtime binaries: Chromium is required only to verify the existing MLflow Runs and Traces views render the logged records; no MLflow UI code is changed.
- Real blockers that would prevent setup: incompatible remote MLflow version or unavailable server would block live readback only, not local implementation.

## Deterministic Real-User Test
- Entry point: `.venv/bin/python -m scripts.optimize run --campaign <tmp>/campaign --config <tmp>/config.json`.
- User workflow: run a seeded fixture campaign with local MLflow tracking, then inspect campaign completion, model runs, metrics, and cycle traces.
- Stable inputs or fixtures: `tests/fixtures/research_races.csv`, one-fold protocol, fixture planner, fixed seed, local SQLite MLflow store.
- User-observable assertions: command exits zero; trials complete; run metrics include the richer suite; traces are searchable; trace inputs/outputs link decisions to outcomes; no fake planner cost appears.
- Command: automated by the end-to-end and MLflow integration tests plus a direct temporary-directory smoke and Playwright browser smoke against the local MLflow UI.
- Evidence to record: exact unittest/smoke commands and MLflow readback counts in `.mega/evidence.jsonl`.

## Fulfillment and Readback Proof
- Requested final user-visible result: richer MLflow run metrics and optimizer-cycle traces with usage/cost.
- Required write/mutation operation: log run metrics and trace spans to an MLflow tracking store, then persist trace linkage for the cycle.
- Readback surface: `mlflow.get_run`, `mlflow.search_traces`, and `mlflow.get_trace` against the same tracking URI.
- Expected content, rows, fields, counts, or behavior: one model run per completed trial; one trace per traced cycle; target metrics on runs; decision/trial spans on traces; exact usage/cost attributes when provided; explicit unavailable status otherwise.
- Command or tool call evidence: local SQLite integration test, deterministic campaign smoke, and browser readback of the local MLflow experiment Runs and Traces views.
- Not-done conditions: only JSONL contains decisions, only MLflow runs exist, duplicate traces are emitted on retry, cost is silently zero, or trace logging can crash training.

## Armageddon Mode
- Damage-scaled attack scope: moderate, covering metric helpers, executor consumers, provider usage parsing, trace idempotency, and disabled/unreachable tracking behavior.
- Edge cases to try: unequal race sizes, fewer than three runners, tied predictions, constant vectors, malformed probabilities, missing usage, alternate token field names, malformed cost, failed trials, repeated trace logging, and tracing disabled.
- Failure modes to simulate: MLflow unavailable/disabled, trace write failure, provider response without cost, duplicate cycle reconciliation, and invalid metric inputs.
- Optimization opportunities to inspect: avoid repeated label materialization/grouping where practical and cap trace previews while preserving artifacts as source of truth.
- Must-fix threshold: non-finite metrics, cross-race ranking leakage, changed winner objective, false zero cost, duplicate cycle trace, leaked secrets, or optimizer interruption from tracking failure.
- Evidence to record: focused adversarial tests, full suite, scope check, and local MLflow readback.

## Generality Guardrail
- Existing mechanism to reuse: shared research evaluator, target diagnostics, executor folds, MLflow config/logger, and controller tracking reconciliation.
- Similar existing use cases: winner-probability metrics, per-fold baseline comparisons, model package runs, and durable decision/trial ledgers.
- Recurrence likelihood: high because every new target and every future planner cycle needs the same metric/trace contracts.
- General mechanism to create if none exists: reusable race aggregation helpers and one generic cycle-trace logger with normalized provider usage.
- Specific use case implementation through the mechanism: all current targets dispatch through shared metric helpers; controller sends normalized decision/results to the generic tracer.
- One-off justification, if any: none.

## Ordered State and Dashboard
- State ledger path: `.mega/state.jsonl`.
- Dashboard output path: `.mega/dashboards/RACE_AWARE_METRICS_AND_TRACING_PLAN.html`.
- Open behavior: generated without opening; this task has no UI requirement.
- State events to record: plan validation, guardrail, each implementation phase, focused/full tests, smoke/readback, Armageddon, commits, and final gate.
- Evidence links: `.mega/evidence.jsonl` entries link verification commands to phase state.
- Dashboard readback assertions: generated HTML contains this plan title and phase checklist state.

## Regression Guardrails
- Planned edit surface: `ima/research_evaluation.py`, `ima/research_models.py`, `ima/research_executor.py`, `ima/mlflow_tracking.py`, `ima/openrouter_orchestrator.py`, `ima/research_controller.py`, `tests/test_research_evaluation.py`, `tests/test_research_executor.py`, `tests/test_research_controller.py`, `tests/test_mlflow_tracking.py`, `tests/test_openrouter_orchestrator.py`, `tests/test_agentic_optimizer_e2e.py`, `docs/research-evaluation.md`, `docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md`, and this plan.
- Protected behaviors: temporal folds, label leakage rejection, deterministic recipes, Optuna tell semantics, durable campaign resume, model package loadability/registration, degraded local planner fallback, and race-day paths.
- Likely consumers: optimizer controller/workers, MLflow UI/API, model registry, evidence summaries, and future model-promotion policy.
- Damage radius: moderate.
- Branch strategy: dedicated `feat/race-aware-metrics-mlflow-traces` branch keeps the shared evaluation/tracking change isolated from `main`.
- Proof plan: targeted metric/tracing tests, controller/E2E tests, full suite, deterministic CLI smoke, MLflow API and Playwright UI readback, scope check, and atomic commits.

## Phase 1: Metric Contracts

### Subphase 1.1: Add reusable race-aware metric suites
- Commit: `feat(metrics): add race-aware target evaluation suites`
- Tests: `tests.test_research_evaluation`, target diagnostics tests in `tests.test_research_executor`, and existing modeling tests.
- Success Criteria: all target metrics are finite, equal-race-weighted, deterministic, and verified on hand-computable fixtures.
- Planned Touch Files:
  - `ima/research_evaluation.py`
  - `ima/research_models.py`
  - `tests/test_research_evaluation.py`
- Checklist:
  - [x] Implement safe race aggregation, ranking, classification, and regression diagnostics.
  - [x] Extend winner-probability diagnostics with MRR and per-race dispersion.
  - [x] Test unequal fields, ties, constants, and top-k truncation.

## Phase 2: Executor and MLflow Runs

### Subphase 2.1: Wire metrics into objectives, artifacts, and searchable runs
- Commit: `feat(optimizer): expose target metrics in research runs`
- Tests: `tests.test_research_executor`, `tests.test_mlflow_tracking`, package reload tests, and objective-name assertions.
- Success Criteria: each fold stores model/baseline/control metrics; aggregate metrics and metric direction are explicit; MLflow run readback contains the new values.
- Planned Touch Files:
  - `ima/research_executor.py`
  - `ima/mlflow_tracking.py`
  - `tests/test_research_executor.py`
  - `tests/test_mlflow_tracking.py`
- Checklist:
  - [x] Replace ranking objective with negative race-mean NDCG@3 and preserve minimize semantics.
  - [x] Aggregate target diagnostics across folds with mean/std/worst summaries.
  - [x] Flatten stable summary and fold metrics into MLflow without logging labels or raw rows.

## Phase 3: Orchestration Traces

### Subphase 3.1: Trace planner decisions, usage/cost, and trial outcomes
- Commit: `feat(mlflow): trace optimizer decisions and experiment cost`
- Tests: `tests.test_openrouter_orchestrator`, `tests.test_mlflow_tracking`, controller tests, and `tests.test_agentic_optimizer_e2e`.
- Success Criteria: one idempotent cycle trace connects evidence -> planner decision -> proposals -> terminal results, reports exact provider usage/cost when present, marks missing cost unavailable, and never blocks training on tracking failure.
- Planned Touch Files:
  - `ima/openrouter_orchestrator.py`
  - `ima/mlflow_tracking.py`
  - `ima/research_controller.py`
  - `tests/test_research_controller.py`
  - `tests/test_openrouter_orchestrator.py`
  - `tests/test_mlflow_tracking.py`
  - `tests/test_agentic_optimizer_e2e.py`
- Checklist:
  - [x] Normalize OpenRouter token and cost fields without estimation.
  - [x] Build one root agent span with planner, proposal, and trial child spans.
  - [x] Persist trace linkage and make repeat calls idempotent.
  - [x] Surface trace errors through existing tracking errors while allowing trials to continue.

## Phase 4: Verification and Documentation

### Subphase 4.1: Prove the complete local workflow and document interpretation
- Commit: `docs(optimizer): document metrics and experiment traces`
- Tests: complete unittest suite, deterministic fixture CLI campaign, MLflow run/trace API readback, local MLflow Playwright browser smoke, scope check, plan check, and Armageddon cases.
- Success Criteria: all checks pass; docs identify primary versus diagnostic metrics and cost-availability semantics; final Megaskill gate passes.
- Planned Touch Files:
  - `docs/research-evaluation.md`
  - `docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md`
  - `docs/RACE_AWARE_METRICS_AND_TRACING_PLAN.md`
- Checklist:
  - [x] Document metric direction, grouping, objectives, and baseline interpretation.
  - [x] Document Runs versus Models versus Traces and token/cost visibility.
  - [x] Run deterministic user workflow and read back the actual MLflow records.
  - [x] Run full regression and adversarial checks, then complete state/evidence ledgers.
