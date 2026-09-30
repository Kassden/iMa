# Agentic Feature Discovery for Race Log Loss

## GOAL
- Add a reproducible feature-work loop in which the orchestrator proposes data repairs, point-in-time feature definitions, feature subsets, and transforms; a deterministic executor tests them on matched race folds; and only verified candidates can enter model search.
- Keep the current v3 cycle running during development; after local tests and a v4 server canary pass, stop v3 at a cycle boundary and start a separate v4 campaign. Preserve race-day inference until a separately approved model promotion.

## Acceptance Criteria
- Every candidate has a source snapshot hash, row/column lineage, feature definition, availability timestamp rule, transform state, model recipe, and fold metrics in MLflow and local artifacts.
- The orchestrator can choose among registered source/feature/transform operations and suggest new definitions for review, but cannot execute arbitrary generated code in the live campaign.
- The same source snapshot and recipe reproduce an identical feature matrix and trial result within declared numerical tolerance.
- A candidate is scored against the frozen Benter baseline on the same race folds, with standalone race log loss primary, calibrated-market and blended scores diagnostic, and an untouched later test period used once for promotion.
- A terminal command prints coverage drift, feature-family ablations, score differences with race-bootstrap intervals, and a keep/reject decision for each proposed feature program.
- V3 is allowed to finish its then-current cycle; the transition records its last completed decision and trial counts, confirms zero running v3 workers, starts v4 under `imaopt`, and reads back fresh v4 planner and completed-trial evidence in MLflow.

## Out Of Scope
- Live betting, race-day UI, replacing the running v3 dataset in place, or giving feature importance scores the meaning of trainable model weights.
- Automatic ingestion of unverified external data or arbitrary agent-written Python into a running trainer.

## Research
- Repository evidence: the v3 win contract, current dataset hash/coverage, `research_transforms.py`, `rich_features.py`, `feature_analysis.py`, and historical source normalizer were inspected before this plan.
- Official/tool sources: scikit-learn Pipeline, Featuretools cutoff time, Optuna search/pruning, and CatBoost importance docs cited below. The selected implementation reuses existing tools first; Featuretools is conditional on a time-safe benchmark.

## Root-Cause Baseline
- Evidence inventory: read-only live SQLite ledger summary (1,034 completed win trials), best-run fold metrics and recipe, score protocol dates, historical data-column coverage by era, and local source/config readback. Keep this inventory as baseline evidence before implementation.
- Proven: The live v3 dataset is fixed at `rich-history.csv.gz`; the win lane uses conditional logit only; current transforms are clipping and within-race rank; 14 of 172 winning-recipe feature columns have zero training observations; official archive normalization sets runner rating to `None`. Recent `horse_rating` is therefore absent in the scored era.
- Likely: missing recent rating/age and unrepresented interactions cap fundamental-model accuracy; repeated tuning on the same development folds overstates tiny gains.
- Possible: other source coverage, identity, same-day ordering, or pre-race availability defects. Audit before changing the dataset.
- Disproven: adding more L2 trials alone has produced a material new optimum within the current search space.
- Missing evidence: legal/timestamped source for recent rating/age; exact benefit of each repaired feature; performance of boosted win models on matched folds; genuinely untouched test score.
- Mutation boundary: immutable raw sources and current campaign; create a new dataset version and new campaign identity. No data overwrite or server restart during planning.
- Remediation mapping: source coverage audit -> missing-data class; point-in-time feature builder -> leakage class; paired challenger study -> model-capacity class; untouched final test -> repeated-selection class.

## SOTA, Standards, And Best Practices
- Use existing `ima/rich_features.py` for race-domain, time-aware history and `sklearn` Pipeline/ColumnTransformer for train-fitted preprocessing. Official scikit-learn guidance explains why fitting transforms inside training folds prevents leakage: https://sklearn.org/stable/modules/compose.html
- Evaluate Featuretools as an optional generator for horse/jockey/trainer relational aggregates. Its cutoff-time and training-window APIs support time-aware feature matrices, but do not substitute for our availability contract: https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html
- Keep Optuna for bounded feature switches, transform parameters, regularization and model parameters inside one orchestrator-approved program: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- Treat permutation importance and SHAP as diagnostic explanations, not causal effects or direct coefficient weights. Use paired refit ablations for keep/drop decisions. CatBoost provides native importance/SHAP tooling for a boosted challenger: https://catboost.ai/docs/en/concepts/fstr
- Reject a broad AutoML replacement, unbounded polynomial explosion, feature selection on the final test set, and hand-assigned feature weights. These obscure lineage or leak selection into evaluation without solving the missing-source problem.

## Dependency and Tooling Preflight
- Existing: pandas, scikit-learn, CatBoost, Optuna, MLflow, pytest; `ima/feature_analysis.py` already computes association, redundancy and permutation diagnostics.
- Optional: Featuretools only after a small benchmark demonstrates safe cutoff behavior and acceptable memory/runtime on the 271k-runner archive. Pin a compatible version if adopted.
- Before implementation: capture exact git revision, dataset hash, source-manifest hashes, Python/package versions, campaign identity, and matched baseline predictions. Check disk/RAM and server user boundary under `imaopt`.
- No browser is required: this is terminal-first. Use MLflow readback for recorded experiment lineage.
- No install is needed for the first three phases: existing core dependencies and pytest suffice. If Featuretools is selected after the benchmark, use `python -m pip install 'featuretools==<validated-version>'` in the project research environment and run its cutoff-time fixture tests before use. No Playwright/browser smoke or screenshot proof is needed because no UI is being changed; verify MLflow through its API.

## Deterministic Real-User Test
- Entry point: one terminal `feature-program run --spec <versioned-spec> --baseline <attempt-id> --output <directory>` command (exact CLI name finalized against existing project CLI conventions).
- Fixture: two races with two horses each, repeated horse/jockey histories, a later result row, and a source event captured after the earlier race.
- Assertions: future result/event mutations cannot change earlier features; race-relative transforms use only that race's pre-race rows; train-fitted state is not refit on calibration/score; feature matrix hash, result and report are reproducible.
- Readback: inspect the emitted manifest, coverage report, predictions, race losses and MLflow run/traces, not just process exit status.

## Fulfillment and Readback Proof
- Deliver an immutable candidate dataset version, a declarative feature-program spec, a baseline/challenger comparison report, and MLflow artifacts/tags linking source -> feature matrix -> model trial -> decision.
- Verify a recovered feature actually has nonzero coverage in the 2022-2025 score era; zero-observation fields must be flagged rather than silently imputed.
- Verify one intentionally leaky feature and one future-dated source event are rejected by tests and executor.
- Do not call this done if the agent can only suggest hyperparameters or if an apparent gain disappears on an untouched later test set.

## Armageddon Mode
- Attack scope: source ingestion, as-of feature construction, fold fitting, candidate comparison, and artifact replay. Treat any leakage or source-timestamp ambiguity as a must-fix blocker.
- Attack scratched/non-single-winner races, duplicate horse IDs, same-day race order, historical source conflicts, missing timestamps, later profile snapshots, odds captured after post time, all-null columns, unknown categories, zero-variance columns, and tiny/large fields.
- Check that a full-historical rebuild cannot read calibration/test labels to compute as-of aggregates; compare earliest feature rows before/after adding future source records.
- Check source coverage and score separately by era, venue, race class, field size, debut status and official/third-party source; investigate concentrated regression before promotion.
- Check campaign identity guards reject a feature-matrix hash mismatch and concurrent workers never read a partly written matrix.
- Optimization opportunity: cache identical content-addressed source-to-matrix builds, prune all-null features before model fitting, and compare runtime/memory against the frozen baseline without changing scientific folds.

## Generality Guardrail
- Recurrence: source repair, feature proposal, and ablation will recur every research cycle; one reusable program contract is justified, while a one-off notebook patch is not.
- Reuse the existing `PipelineRecipe`, research executor, temporal protocol, research ledger and MLflow tracking rather than create a second optimizer.
- Add one versioned `FeatureProgram` contract owning source selection, as-of joins, feature definitions, subset masks and transform choices. Store generated matrices content-addressed outside the running campaign.
- Keep feature *weights* in the model: conditional-logit coefficients and regularization; use ablation/permutation/SHAP only to guide selection. Optional per-family penalties or feature gates require an explicit matched trial, never arbitrary manual multipliers.

## Ordered State and Dashboard
- Use the existing `.mega/state.jsonl` and `.mega/evidence.jsonl` for implementation progress. A generated plan dashboard may summarize these; it is not another source of truth.
- MLflow is the scientific readback: parent feature-program run, child model trials, source/matrix hashes, coverage artifacts, paired metric deltas and orchestrator decision/cost trace.

## Regression Guardrails
- Planned edit surface: `ima/historical_sources.py`, `ima/rich_features.py`, `ima/feature_sets.py`, `ima/research_transforms.py`, `ima/research_specs.py`, `ima/research_executor.py`, `ima/research_controller.py`, `ima/feature_analysis.py`, `ima/mlflow_tracking.py`, dedicated feature-program module/CLI, tests, and new versioned config/docs. Narrow each subphase to its own files.
- Protected: frozen v3 campaign, original source files, chronological folds, race-normalized probabilities, market baseline, audit lineage, MLflow history, model-package inference, and fail-closed live wagering.
- Consumers: optimizer, model packages, race-day CLI, MLflow. Damage radius: systemic. Implement on a dedicated branch; create a separate candidate campaign and dataset identity before server execution.
- Branch strategy: dedicated feature-discovery branch before any implementation or data mutation; this document alone does not change the running branch or campaign.
- Proof: focused leakage/contract tests, matched 3-fold baseline comparison, race-bootstrap intervals, untouched future test, package replay, and a read-only MLflow verification. Commit one independently valid subphase at a time; do not deploy or promote implicitly.

## Phase 1: Data and score baseline

### Subphase 1.1: Freeze matched comparison
- Commit: add a reproducible baseline manifest/report for the current best fundamental attempt, its 3,000 score races, market comparator, and a reserved later test period.
- Tests: dataset/protocol hash checks; unique race membership and chronological order; paired race-loss computation.
- Success Criteria: terminal readback reproduces baseline loss and identifies which races are unavailable for optimizer feedback.
- Planned Touch Files:
  - `ima/research_evaluation.py`
  - `ima/feature_analysis.py`
  - `config/agentic_v4_features_protocol.json`
  - `scripts/report_v4_programs.py`
  - `tests/test_report_v4_programs.py`
  - `tests/test_research_evaluation.py`
  - `docs/AGENTIC_FEATURE_DISCOVERY_PLAN.md`
- Checklist:
  - [x] Freeze baseline predictions and score-race IDs.
  - [x] Reserve and lock a sufficiently sized later test period; if the ~98 races after the existing folds are too few, move development folds earlier under a new protocol ID rather than pretending the small tail is decisive. Never show test labels to the orchestrator.

### Subphase 1.2: Audit and repair source coverage
- Commit: produce per-era/per-source coverage and timestamp audits; add only verified pre-race rating/age/event recovery to a new canonical dataset version.
- Tests: raw-to-canonical provenance, late-source rejection, identity collision, source precedence, coverage and null-rate regression fixtures.
- Success Criteria: report names supported sources and shows measured recent-era coverage changes without modifying the live dataset.
- Planned Touch Files:
  - `ima/historical_sources.py`
  - `ima/rich_features.py`
  - `scripts/build_historical.py`
  - `scripts/build_v4_dataset.py`
  - `tests/test_historical_sources.py`
  - `tests/test_rich_features.py`
- Checklist:
  - [x] Audit rating/age availability dates before attempting recovery.
  - [x] Build a new source snapshot; reject unavailable or post-race values.

## Phase 2: Feature-program contract

### Subphase 2.1: Declarative generation and materialization
- Commit: add validated feature-program specs with lineage, source allowlist, as-of availability, registered primitives and atomic content-addressed matrix materialization.
- Tests: reproducibility, source hashes, future-data invariance, all-null/constant flags, invalid spec rejection, concurrent writer/read safety.
- Success Criteria: one versioned spec builds a matrix and coverage report; future source changes cannot alter historical pre-race rows.
- Planned Touch Files:
  - `ima/feature_sets.py`
  - `ima/rich_features.py`
  - `ima/feature_program.py`
  - `ima/research_specs.py`
  - `ima/research_executor.py`
  - `ima/research_model_package.py`
  - `tests/test_research_specs.py`
  - `tests/test_research_transforms.py`
  - `tests/test_feature_program.py`
- Checklist:
  - [x] Start with registered horse/jockey/trainer rolling aggregates, race-relative deltas and interactions; evaluate Featuretools separately.
  - [ ] Record observation time and source path for every new feature.

### Subphase 2.2: Fold-fitted transforms and selection
- Commit: extend registered transforms and feature gates, fitting state only on each train fold and storing it in the model package.
- Tests: train/calibration/score isolation, replay parity, unseen category, missing-value and transformed-schema assertions.
- Success Criteria: Optuna can choose feature-family switches and numeric transform parameters without touching raw source rows or final-test labels.
- Planned Touch Files:
  - `ima/research_transforms.py`
  - `ima/research_specs.py`
  - `ima/research_executor.py`
  - `tests/test_research_executor.py`
- Checklist:
  - [x] Separate deterministic as-of feature creation from train-fitted scaling/imputation/clipping/selection.
  - [ ] Interpret coefficients, ablation deltas and importance separately; no free-form weight knob.

## Phase 3: Agentic scientific loop

### Subphase 3.1: Program proposal, matched experiments and readback
- Commit: let the orchestrator propose bounded feature programs from coverage/gap/ablation evidence; Optuna tunes only within the approved program; compare a matched conditional-logit baseline and boosted win challenger.
- Tests: valid/invalid proposal contracts, fixed concurrency, dataset identity, equal-fold comparison, paired bootstrap, decision lineage and MLflow artifact readback.
- Success Criteria: a complete terminal cycle proposes -> builds -> trains -> reports -> accepts/rejects, with explicit reasons and planner cost.
- Planned Touch Files:
  - `ima/research_controller.py`
  - `ima/openrouter_orchestrator.py`
  - `ima/research_executor.py`
  - `ima/mlflow_tracking.py`
  - `ima/optimizer.py`
  - `scripts/optimize.py`
  - `tests/test_research_controller.py`
  - `tests/test_agentic_planner.py`
  - `tests/test_mlflow_tracking.py`
- Checklist:
  - [x] Allocate trials by information gain, not a mandatory full batch; preserve the global concurrency ceiling.
  - [x] Record all attempted programs, including failed validation and rejected hypotheses.

### Subphase 3.2: Promotion gate and package replay
- Commit: add a no-auto-promotion decision gate and test candidate packages against the untouched later period and terminal inference replay.
- Tests: fresh-period score, race probability sums, missing-source fallback, package feature-order parity, and campaign rollback/readback.
- Success Criteria: a candidate is eligible only after predeclared race-loss improvement with uncertainty, stable era slices, leakage audit and package replay; otherwise current champion remains unchanged.
- Planned Touch Files:
  - `ima/research_controller.py`
  - `ima/research_executor.py`
  - `ima/mlflow_tracking.py`
  - `tests/test_research_controller.py`
  - `tests/test_research_executor.py`
- Checklist:
  - [ ] Run final test once for the selected challenger, not every feature proposal.
  - [x] Keep deployment and live wagering behind separate human approval.

## Phase 4: Guarded v4 server handoff

### Subphase 4.1: Staged canary while v3 runs
- Commit: add v4 config and isolated supervisor launcher with distinct campaign directory, release path, MLflow experiment and identity; deploy only under `imaopt`.
- Tests: server import/dependency check, protocol/dataset hash check, fixed-seed v4 proposal -> feature matrix -> training -> MLflow readback; no v3 file mutation.
- Success Criteria: at least one completed v4 canary trial has valid race probabilities, feature lineage and a readable decision/cost trace while v3 service remains active.
- Planned Touch Files:
  - `config/agentic_v4_features.json`
  - `config/agentic_v4_canary.json`
  - `deploy/systemd/ima-feature-v4-supervisor`
  - `deploy/systemd/ima-feature-v4-supervisor.service`
  - `deploy/systemd/ima-v4-revision`
  - `docs/AGENTIC_V4_RUNBOOK.md`
- Checklist:
  - [x] Stage a separate release and campaign as `imaopt`; keep v3 active.
  - [x] Read back v4 MLflow and artifact evidence before switchover.

### Subphase 4.2: Cycle-boundary switchover and watch
- Commit: document the explicit v3 STOP-at-boundary and v4 activation procedure; do not stop v3 before the canary gate.
- Tests: read v3 active decision/trial state, request graceful stop, poll to zero v3 workers, start v4, verify systemd active and two successive v4 decisions/trial completions, check CPU/RAM and other services.
- Success Criteria: v3 retains its complete current cycle and is inactive; v4 is active and autonomously completes repeated decisions without a local computer connection.
- Planned Touch Files:
  - `docs/AGENTIC_V4_RUNBOOK.md`
  - `docs/AGENTIC_FEATURE_DISCOVERY_PLAN.md`
- Checklist:
  - [ ] Stop v3 only after the active cycle finishes and the v4 canary passes.
  - [ ] Watch v4 for two cycles and read back MLflow; retain a rollback command but do not run it unless needed.
