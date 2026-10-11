# Probit Runtime and Feature Lifecycle Repair

## GOAL
Make probit progress-aware with safe fold-local warm starts and connect audited feature creation, immutable datasets, MLflow lineage and model selection.

## Acceptance Criteria
- Probit initially gets at least 14400 seconds, adjusted using exact-workload successful runtimes and censored lower bounds.
- Iteration/objective/fold/stage progress is observable; productive fits get bounded durable extensions, not infinite lifetimes.
- Related fits reuse coefficients only for identical training rows, features, transformations and preprocessing; no evaluation labels or cross-fold state.
- A real feature build produces non-null validated definitions, a reusable content-addressed dataset and its own MLflow run; downstream models and planner see lineage and matched outcomes.
- Tests, canaries, immutable own-user rollout and fresh server/MLflow readback prove use, not just installation. Improvement is a scientific hypothesis, not a promised score.

## Research
- Existing owners: SciPy L-BFGS-B/analytical gradients, bounded NumPy batches, Featuretools, FeatureRegistry, DatasetRegistry, Pebble, JobEstimator and MLflow outbox.
- Sources: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html ; https://featuretools.alteryx.com/en/stable/getting_started/handling_time.html ; https://mlflow.org/docs/latest/tracking/tracking-api/ .

## Root-Cause Baseline
- Minimum evidence inventory: live ledger/status readbacks, recipe defaults, worker deadline code, estimator samples and preserved user-observed timeout counts.
- Proven: live V7 has 13 heteroscedastic attempts: 12 deadline expirations, one completion in 1382.57s, loss 2.1726159. Proposal default deadline is 1200s. Execution uses proposal budget rather than measured estimate; deadlines persist across restart.
- Proven: 683 saved recipes have four discoveries and zero custom definitions. Champion has null discovery/definitions. Accepted proposals are not executed features.
- Proven: typed formula AST, strict availability validation, content-addressed feature registry and asynchronous immutable dataset builds already exist.
- Likely: generic deadline defaults, absent optimizer progress, conservative expensive admission and opaque feature reuse discourage exploration.
- Possible: conditioning and integration refinement dominate some fits; profile before assigning cores.
- Disproven: all timeout failures are numerical or OOM; all paid calls are rejected; feature construction is absent.
- Missing evidence: historical iteration trajectories cannot be recovered; collect prospectively. Gains cannot be guaranteed.
- Mutation boundary: dedicated branch; own imaopt release/service only. Preserve active fits, historical records, dirty user edits and other services. Safe drain before activation.
- Remediation mapping: family deadline fixes budget mismatch; progress/extensions resolve slow/stuck ambiguity; fold-keyed cache reduces duplicate fit work; feature stage/MLflow/catalog resolves sparse execution and visibility. Profiling is risk reduction.
- Not-done conditions: dangling modules, unsafe warm starts, queued-only features, unsupported improvement claims or unverified activation.

## SOTA, Standards, And Best Practices
- Implementation decision: reuse mature libraries and existing owners with additive policies.
- Use SciPy callback intermediate_result, retain analytical gradients and numerical verification. Use existing strict cutoff synthesis and training-fold selection.
- One controller with feature-building and model-training actions sharing a candidate catalog, not two competing paid loops.
- Separate MLflow feature-build runs with role, dataset/definition hashes, diagnostics and downstream links. Compare only identical target/population/protocol outcomes.
- Reject unlimited deadlines, treating CPU heartbeat as objective progress, unconverged success, arbitrary Python formulas, forced adoption and historical identity mutation.

## Dependency and Tooling Preflight
- Install or repair commands: no install required when .venv/bin/python imports scipy, joblib, mlflow, featuretools and pebble; otherwise .venv/bin/python -m pip install -e . using existing locked project requirements.
- Required: .venv pytest/scipy/joblib/MLflow/Featuretools/Pebble, SSH own-user server environment.
- Existing tooling found: local environment and remote dependency venv. Repair project-local imports if needed; do not change system packages.
- Browser/runtime binaries: no UI change; MLflow API is readback surface.
- Real blockers: server outage, unavailable credentials or unsafe active-fit handover.

## Deterministic Real-User Test
- Entry: feature lifecycle CLI and BoundedFitExecutor.
- Stable inputs: seeded complete races, pre-race catalog, validated formula and strict folds.
- Assertions: feature definition IDs, immutable successor manifest, separate build run, consumer linkage; real toy probit callback and warm-start receipt.
- Command: .venv/bin/python -m pytest -q tests/test_probit_runtime_policy.py tests/test_probit_progress_warmstart.py tests/test_feature_lifecycle.py tests/test_feature_lifecycle_integration.py
- Evidence: artifacts and server/MLflow fields read back, not route-only proof.

## Fulfillment and Readback Proof
- Readback surface: server files and MLflow API; expected content includes real feature definition IDs, successful build run, dataset digest, downstream lineage and optimizer progress.
- Read progress.json, runtime extension receipts, feature manifest, successor dataset manifest, MLflow build run and planner candidate feedback.
- Deploy sealed additive own-user adapter only after tests/canaries and safe drain.
- Not done if modules are unused, successor features cannot enter training or features only exist as proposals.

## Armageddon Mode
- Damage: large. Attack stale/NaN progress, stalled optimizer, hard cap, restart, cancellation race, fold mismatch, changed preprocessor, future/label AST refs, wrong units, corrupt hashes, duplicate builds and tracking outage.
- Must-fix: leakage, false completion, unbounded extension, wrong-user process action, lost cost/native token identity.
- Optimization: stage timings and sampled memory before assigning threads.

## Generality Guardrail
- Existing mechanism to reuse: shared registry, estimator, worker executor and tracking infrastructure; recurrence is high across research programs.
- Reuse JobEstimator/BoundedFitExecutor and FeatureRegistry/DatasetRegistry; add general family policy and feature lifecycle owner. Same feature candidate can feed different models; matched objective contracts remain separate.

## Ordered State and Dashboard
- .mega/state.jsonl and .mega/evidence.jsonl; generated read view .mega/dashboards/PROBIT_FEATURE_LIFECYCLE_REPAIR_PLAN_20261011.html.

## Regression Guardrails
- Branch strategy: dedicated fix/probit-runtime-feature-lifecycle for large shared-worker change.
- Damage radius: large. Branch: fix/probit-runtime-feature-lifecycle, created before edits.
- Protected: convergence/quadrature, strict cutoff, metric contracts, free allocation, memory100GB, native cost/tokens, current fits, immutable science/history, other users/services.
- Preexisting dirty files preserved; stage owned files/hunks only. Scope expansions recorded before editing.
- Planned edit surface: explicit paths below. Atomic runtime, feature, qualification and deployment commits.

## Phase 1: Evidence
### Subphase 1.1: Baseline and contract
- Commit: docs(research): lock runtime and feature repair contracts
- Tests: plan checker, live ledger readback, dependency imports.
- Success Criteria: proven causes separated from hypotheses and safe mutation boundary named.
- Planned Touch Files:
  - `docs/PROBIT_FEATURE_LIFECYCLE_REPAIR_PLAN_20261011.md`
- Checklist:
  - [x] Verify timeout/discovery counts, inspect owners and research primary sources.
  - [x] Lock acceptance criteria, branch and preservation boundaries.

## Phase 2: Runtime
### Subphase 2.1: Measured deadlines and fold-safe progress/reuse
- Commit: feat(probit): add measured productive budgets and fold-safe reuse
- Tests: runtime policy, progress/warmstart, existing worker/probit/gradient/refinement suites.
- Success Criteria: 4h initial, exact-workload lower bounds, durable productive extensions, training-only cache and real iteration profiling.
- Planned Touch Files:
  - `ima/probit_runtime_policy.py`
  - `ima/performance_probit.py`
  - `ima/research_worker_runtime.py`
  - `ima/research_specs.py`
  - `ima/research_executor.py`
  - `tests/test_probit_runtime_policy.py`
  - `tests/test_probit_progress_warmstart.py`
- Checklist:
  - [ ] Implement policy, callbacks, profile and warm-start cache.
  - [ ] Integrate actual executor and test restart/ownership/numerical safety.

## Phase 3: Features
### Subphase 3.1: Reusable tracked feature builds
- Commit: feat(features): connect definition creation, datasets and model feedback
- Tests: lifecycle, integration, existing formulas/registry/discovery/MLflow/context suites.
- Success Criteria: actual definition build, immutable dataset, separate MLflow run and concise reusable candidate feedback.
- Planned Touch Files:
  - `ima/feature_lifecycle.py`
  - `scripts/feature_lifecycle.py`
  - `ima/research_expansion.py`
  - `ima/research_planner_context.py`
  - `ima/mlflow_tracking.py`
  - `tests/test_feature_lifecycle.py`
  - `tests/test_feature_lifecycle_integration.py`
- Checklist:
  - [ ] Build safe deterministic and planner AST candidates with validators and lineage.
  - [ ] Expose to controller/models and test leakage/idempotency/tracking failures.

## Phase 4: Qualification
### Subphase 4.1: End-to-end and adversarial proof
- Commit: test(research): qualify real feature and runtime consumers
- Tests: full targeted suites and isolated server canaries; workload timings/threads/memory.
- Success Criteria: no dangling paths, preserved numerical and billing contracts, artifact readback.
- Planned Touch Files:
  - `tests/test_feature_lifecycle_integration.py`
  - `docs/PROBIT_FEATURE_LIFECYCLE_REPAIR_VALIDATION_20261011.md`
- Checklist:
  - [ ] Complete tests and adversarial attacks.
  - [ ] Read back real server feature/dataset/probit/MLflow records.

## Phase 5: Deployment
### Subphase 5.1: Immutable own-user rollout
- Commit: feat(deploy): activate qualified research lifecycle repair
- Tests: sealed self-test, safe drain, fresh active service/progress/candidate readback and final gate.
- Success Criteria: repaired behavior used on server with predecessor audit preserved.
- Planned Touch Files:
  - `deploy/systemd/ima-v7-probit-feature-lifecycle.py`
  - `docs/PROBIT_FEATURE_LIFECYCLE_REPAIR_VALIDATION_20261011.md`
- Checklist:
  - [ ] Push scoped commits, stage/qualify and preserve running fits.
  - [ ] Activate at safe boundary and verify fresh outputs, documenting uncertainty.
