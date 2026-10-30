# Continuous Feature Exploration

## GOAL
Continuously generate, build and test substantially more independent feature hypotheses, with paired model feedback visible to the running research planner and MLflow, without cancelling existing fits or weakening scientific checks.

## Acceptance Criteria
- [ ] Durable queue refills beyond the original eight example definitions, deduplicates work, survives restart and accepts planner suggestions.
- [ ] Cached baseline and single-feature/family candidates use identical protected temporal populations, target, model settings and training-only preprocessing.
- [ ] Multiple new feature builds and completed paired tests are visible in MLflow and the shared registry; the main planner can read their feedback.
- [ ] Independent own-user service continues without this computer, preserves current fits and keeps combined model/feature RAM at or below 100 GB decimal.

## Runtime Contract
- Mode: execute; task feature-exploration-20261015; session 01a0c46f-3ae6-79f3-b206-199aa3eff2f6.
- The preceding throughput-20261014 task is complete; do not reopen or replace its outcome.
- Proof artifacts: implementation, tests, sealed release manifest, live queue/MLflow/registry readbacks.
- Delegation: at most three owned children, disjoint edits, no recursive delegation or child commits/branch changes; native close receipts required.

## Decision Record Policy
Record queue ownership, comparison semantics and resource partition before implementation. Challenge historical qualification versus safety of new work, baseline-specific merit versus universal improvement, and fixed worker capacity versus unlimited candidate supply. Promote durable accepted rationale to docs/adr/20261015-feature-exploration.md.

## Research
Reuse FeatureLifecycle/FeatureRegistry, typed NumericExpression validation, existing Featuretools historical discovery, execute_recipe, paired_feature_report, BoundedFitExecutor and MlflowClient. A new queue is coordination, not a replacement formula engine or model trainer.

## Root-Cause Baseline
Live metrics, logs and config/state readbacks establish the following baseline.
Evidence inventory: live ledger, latest planner evidence, shared feature registry, controller source and MLflow build receipts inspected on 2026-10-14 UTC / 2026-10-15 JST. Proven: only two ready feature datasets, no builds pending; last-24-hour updated rows contain one definition fit and one discovery fit. Feature creation only runs when a model recipe requests definitions. Deterministic examples stop at 3-8; no replenished independent queue. The preserved discovery fit took over three hours, making expensive composed models a poor first screening vehicle. Hypothesis: independent cached cheap controls materially increase feature throughput. Disproven: a currently saturated feature-build queue. Missing evidence: candidate usefulness and speed require real model executions; never infer gains from publication or coverage. Mutation boundary: own imaopt user/services/registry only, no active main-fit cancellation, no unrelated dirty files. Remediation mapping: independent durable queue fixes missing supply; cheap fixed controls fix expensive screening; paired publication fixes missing baseline feedback; MLflow stage runs fix observability. Resource partition is generic risk reduction, not a discovered OOM cause.

## SOTA, Standards, And Best Practices
Featuretools DFS primitive restrictions and cutoff-aware history: https://docs.featuretools.com/en/stable/guides/specifying_primitive_options.html . Scikit-learn feature selection and held-out permutation caveats: https://scikit-learn.org/stable/modules/feature_selection.html and https://scikit-learn.org/stable/modules/permutation_importance . Implementation decision: deterministic, unit-compatible pre-race typed formulas plus accepted planner suggestions; existing historical Featuretools pipeline remains available. Validity/coverage/variation checks precede model testing; no labels guide formula generation. Full existing development folds avoid incompatible screening scores. Rejected approaches: random unrestricted arithmetic, independent unrestricted 100 GB services, duplicate publication, forcing every candidate into an expensive pool, automatic champion promotion, or a second competing LLM planner.

## Dependency and Tooling Preflight
Existing tools: .venv pytest/unittest, pandas/NumPy/scikit-learn/MLflow, SSH and own-user systemd. No install is needed; install or repair project-local dependencies if inspection proves otherwise. Real blockers: unavailable credentials, external outage or denied destructive action. No permission is implied to touch Cortex/Solar or other users.

## Deterministic Real-User Test
Entry point: scripts/feature_exploration.py once/run/status. Stable inputs: registered synthetic HKJC dataset and typed feature definitions. User workflow: start queue, observe generated/built/tested counts, restart, inspect baseline-linked delta and MLflow records. Observable assertions: more than eight distinct queued hypotheses, one stable cached baseline, no duplicate execution, paired scored-runner populations, failures isolated, and tracking retries not scientific failures. Commands: .venv/bin/python -m pytest tests/test_feature_candidates.py tests/test_feature_exploration.py tests/test_feature_exploration_integration.py -q.

## Fulfillment and Readback Proof
Required deployment mutation: own sealed feature-exploration service, queue, derived datasets and MLflow records. Resource allocation: reserve 8 GB decimal and up to two CPU threads for one feature worker; own model service retains 92 GB decimal, so combined hard memory limits remain 100 GB. Apply live only after fresh current/reserved model memory proves safe; do not restart model fits. Record partition explicitly; restore the model's original limit if feature service stops. Main full-trial ceiling stays 26; feature screening is a separate bounded preparation workflow. Existing main capability function reads the shared registry dynamically, so new paired outcomes reach its next evidence bundle without a code hotpatch or fit interruption. Readback: several new builds and at least three successful paired tests, MLflow FINISHED runs with baseline linkage, current planner evidence containing new datasets/results, ongoing main fits, no OOM and total limits at most 100 GB. Staged-only, a process without completed tests, or disconnected feature results are not done.

## Armageddon Mode
Large scheduling/data damage radius. Adversarial edge cases: tainted or unknown units, constant/empty formula, duplicate AST, replay/restart, locked loop, failed model, corrupt baseline identity, different scored runners, tracking outage, disk/headroom exhaustion and unsafe live budget reduction. Must-fix: leakage, false matched comparison, cancellation of existing fits, double 100 GB reservation, hidden errors or repeated identical hypotheses. Inspect CPU/RAM/runtime before increasing workers.

## Generality Guardrail
Reuse the existing general mechanism for immutable feature builds and training. New owner is only a durable deterministic queue plus result bridge. Candidate supply is independent of worker capacity and model-program budgets. No fixed lifetime experiment count; idle exhaustion is reported honestly. Feature merits remain model/target/protocol-specific and provisional after adaptive selection.

## Regression Guardrails
- Planned edit surface: ima/feature_candidates.py, ima/feature_exploration.py, scripts/feature_exploration.py, tests/test_feature_candidates.py, tests/test_feature_exploration.py, tests/test_feature_exploration_integration.py, deploy/systemd/ima-feature-exploration-budget.py, deploy/systemd/ima-feature-exploration.service, docs/CONTINUOUS_FEATURE_EXPLORATION_PLAN_20261015.md, docs/CONTINUOUS_FEATURE_EXPLORATION_VALIDATION_20261015.md, docs/adr/20261015-feature-exploration.md and operational .tmp release scripts/receipts.
- Protected behaviors: existing fits and probabilities, immutable parent data, quarantined confirmation labels, current targets, main planner/Optuna authority, 26 full-trial ceiling, total 100 GB memory, existing cold-workload guard, other users/repos/services and preexisting dirty edits.
- Likely consumers: shared DatasetRegistry, main planner capabilities, own services and MLflow.
- Damage radius: large.
- Branch strategy: dedicated branch feat/continuous-feature-exploration.
- Proof plan: unit/adversarial tests, real dataset-consumer integration, sealed server canary, multiple live completed feature tests and main-planner readback.

## Phase 1: Candidate Supply
### Subphase 1.1: Guarded Candidate Stream
- Commit: feat(features): generate deduplicated typed exploration candidates.
- Tests: deterministic generation, more than eight candidates, unsafe-input and unit tests; plan validation.
- Success Criteria: queue supply is not the eight-example cap and uses the existing validated expression vocabulary.
- Planned Touch Files:
  - `ima/feature_candidates.py`
  - `tests/test_feature_candidates.py`
  - `docs/CONTINUOUS_FEATURE_EXPLORATION_PLAN_20261015.md`
  - `docs/adr/20261015-feature-exploration.md`
- Checklist:
  - [ ] Record root cause, alternatives and runtime contract.
  - [ ] Generate deterministic candidates plus accepted planner definitions without changing shared formula semantics.

## Phase 2: Durable Paired Exploration
### Subphase 2.1: Queue, Evaluation And Feedback
- Commit: feat(features): run restart-safe paired feature exploration with MLflow feedback.
- Tests: tests/test_feature_exploration.py and tests/test_feature_exploration_integration.py plus existing lifecycle/registry/worker regression tests.
- Success Criteria: cached fixed baseline, independent candidate testing, isolated failures, retryable tracking and paired registry feedback are proven through real consumers.
- Planned Touch Files:
  - `ima/feature_exploration.py`
  - `scripts/feature_exploration.py`
  - `tests/test_feature_exploration.py`
  - `tests/test_feature_exploration_integration.py`
- Checklist:
  - [ ] Persist idempotent queue and baseline identity; ingest planner suggestions.
  - [ ] Run bounded cheap models on matched existing folds; publish comparisons and separate feature-stage runs.
  - [ ] Exercise restart, failed fit, invalid candidate, mismatched population and tracking outage.

## Phase 3: Live Deployment
### Subphase 3.1: Safe Own-User Resource Partition And Fulfillment
- Commit: fix(deploy): reserve shared feature capacity and verify continuous exploration.
- Tests: own-user budget preflight/canary and live queue, MLflow, registry and planner-evidence assertions.
- Success Criteria: own unattended service completes multiple new feature tests, feeds planner evidence and preserves active main fits under combined 100 GB limit.
- Planned Touch Files:
  - `deploy/systemd/ima-feature-exploration-budget.py`
  - `deploy/systemd/ima-feature-exploration.service`
  - `docs/CONTINUOUS_FEATURE_EXPLORATION_VALIDATION_20261015.md`
- Checklist:
  - [ ] Seal/push release, verify fresh safe memory partition, start own service without restarting model controller.
  - [ ] Verify at least three new paired tests and main-planner receipt; record resource use and close owned children.
