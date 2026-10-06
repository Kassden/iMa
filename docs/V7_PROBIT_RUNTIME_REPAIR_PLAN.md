# V7 Probit Runtime Repair

## GOAL
Preserve Gaussian race-winner likelihoods while making real heteroscedastic trials practical, then deploy and verify a pinned V7 successor with corrected feature admission.

## Acceptance Criteria
- [ ] Profile identifies objective and preparation costs.
- [ ] Analytic objective/gradients match the independent scalar reference, including heterogeneous scales, tiny fields, shuffled runners and rare winners.
- [ ] Same-shape measured objective throughput improves at least 3x with bounded scratch allocation.
- [ ] Real three-fold heteroscedastic trial completes within the original 2,400-second limit, with diagnostics and MLflow linkage.
- [ ] Source is pushed, CI is green, and pinned V7 successor accepts an actual OpenRouter decision and dispatches fits.
- [ ] Old V6 fits untouched; prior V7 drains without hotpatching; 80/20 dispatch and frozen evaluation unchanged.

## Research
Use existing NumPy/SciPy compiled array kernels and SciPy L-BFGS-B; Joblib threading only if measurement justifies it. Retain the scalar mathematical oracle and adaptive QUADPACK prediction checks.

## Root-Cause Baseline
- Proven: attempt-c3b02be1602226a18efa043e1e23149bffd7c1f2d7420f4facc262c198b68738 ended at 2,400 seconds with execution_timeout; private peak 3,531,145,216 bytes. Heteroscedastic, order 32, max_iter 500, three whole-meeting folds with at least 9,000 training races.
- Proven: repeated full-runner scans per race in matrix/scale centering; Python loops over races and runners for quadrature.
- Proven: baseline-only admission rejects properly declared registered speed extras. Prior-turn repair exists locally and must be verified/committed separately.
- Likely: grouping scans and tiny quadrature calls dominate; establish their relative shares with cProfile before choosing optimizations.
- Possible: post-fit convergence verification is expensive; expose stage timings rather than assuming optimizer nonconvergence.
- Disproven as recorded failure: this attempt is not marked OOM. Its memory peak does not describe all host processes.
- Missing evidence: interrupted trial has no completed optimizer diagnostics; do not claim exact CPU saturation or mathematical nonconvergence.
- Mutation boundary: isolated benchmarks and dedicated branch first; no hotpatch, cancellation, database rewriting or other-user service changes.
- Remediation mapping: linear grouped reductions and bounded batched quadrature address measured loops; diagnostics expose runtime; bounded planner-failure feedback addresses repeated invalid decisions.
- Generic hardening: stage diagnostics reduce future uncertainty but do not alone fix runtime.
- Not-done conditions: synthetic-only speedup, restarted-service-only evidence, or missing real trial/tracking readback.

## SOTA, Standards, And Best Practices
- [SciPy log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html): existing stable compiled log-CDF array kernel.
- [NumPy bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html): weighted grouped reductions without per-race full-frame scans.
- Mature mechanisms: NumPy broadcasting/bounded chunks, SciPy analytic-gradient optimizer, existing worker/deadline allocation, measured Joblib native-kernel threading if beneficial.
- Rejected: finite differences, weakened integration checks, altered winner labels, GPU/JAX migration, unbounded full-corpus tensors, nested all-core pools and infinite timeouts.
- Implementation decision: retain normalized identified likelihood and exact variance-link derivatives; optimize its existing computation, not the statistical objective.

## Dependency and Tooling Preflight
- Existing: local Python/pytest, Tailscale SSH, own-user Python 3.12, git/gh, MLflow SDK, megaskill scripts.
- Install or repair commands: no install is needed; NumPy/SciPy, pytest and SSH are already usable. Inspect own deployment/unit before mutation.
- Browser/runtime binaries: no new UI; SDK readback is the tracking proof.
- Real blockers: SSH/provider/CI outage, missing credentials or scientific mismatch, never an assumed missing-tool excuse.

## Deterministic Real-User Test
- Entry: execute_recipe on the saved failed recipe and identical dataset/protocol, seed 42.
- Stable dataset: dataset-19aaa959e948d6e145ce62c8160413087b46ba3c2578e0de15c452054d9a83db, three whole-meeting folds.
- Command: scripts/benchmark_probit.py for profiling; isolated owned-server scientific canary receipt wrapper for full execution.
- Assertions: completed, finite fundamental log loss, unchanged cutoff/splits, stage diagnostics, bounded resources and real MLflow linkage.
- Evidence: profile/benchmark JSON, canary result, pinned revision and tracking readback, not promoted synthetic metrics.

## Fulfillment and Readback Proof
- Requested final user-visible result: completed probit trial and executing repaired research campaign in MLflow.
- Required write/mutation operation: publish source through GitHub and immutable own-user release; start clearly named agentic_v7 successor.
- Readback surface: unit, config/revision identity, accepted planner decision, dispatched fits, scientific completion and MLflow cost/token/model linkage.
- Not done: unknown cost, repeated extra rejection, cancelled preserved fits, changed code under old identity, or service-active flag alone.

## Armageddon Mode
- Damage scope: moderate numerical changes with guarded deployment boundaries.
- Edge cases: single/two/mixed fields, shuffled rows, unequal scales, extreme tails, coarse quadrature, missing extras and future availability.
- Failures: invalid parent, interrupted worker and bounded-memory/deadline behavior; preserve failure receipts.
- Must-fix: gradient/reference mismatch, unbounded allocation, weakened leakage/convergence guard, regressions or failed live canary.
- Performance thresholds belong in benchmark receipts, not flaky CI wall-clock assertions.

## Generality Guardrail
- Reuse GaussianRaceProbit, DatasetFeatureProfile, executor, telemetry and immutable release mechanism.
- Group/chunk helpers serve both homogeneous and heterogeneous objectives; use worker allocations rather than a new scheduler.
- Recurrence: high; future probit programs benefit from the same numerical owner.
- No alternate optimizer, bespoke model registry or special-case hidden experiment path.

## Regression Guardrails
- Branch strategy: dedicated fix/v7-probit-runtime; preserve all unrelated dirty files.
- Planned edit surface: ima/performance_probit.py, ima/research_controller.py, ima/research_expansion.py, tests/test_probit_gradient.py, tests/test_research_controller.py, tests/test_research_expansion.py, scripts/benchmark_probit.py, docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md; own-user temporary ops/packaging receipts.
- Protected: scalar probability API, normalization, analytic derivatives, convergence checks, leakage gates, dataset identity, old fits, 80/20 allocation, costs and shared services.
- Consumers: direct probit, composed models, executor/replay, planner admission and MLflow.
- Damage radius: moderate numerical source plus tightly isolated deployment.
- Proof plan: numerical tests, measured profile, real saved-recipe canary, full regression, CI, rollout and tracking readback.
- Atomic units: admission/feedback; probit acceleration/tests/benchmark; deployment evidence/documentation.

## Phase 1: Baseline
### Subphase 1.1: Profile And Bound
- Commit: docs(research): record probit runtime boundaries.
- Tests: fixed-seed cProfile benchmark and own-unit/campaign inspection.
- Success Criteria: measured before timing/function shares and known resource boundaries.
- Planned Touch Files:
  - `docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md`
  - `scripts/benchmark_probit.py`
- Checklist:
  - [ ] Collect profile and saved attempt identity.
  - [ ] Validate plan and preserve old fits.

## Phase 2: Implementation
### Subphase 2.1: Admission And Feedback
- Commit: fix(research): admit verified extra predictors and expose failed decisions.
- Tests: registered extras; missing/unknown/future input rejection; bounded failed-decision feedback; saved five-proposal replay.
- Success Criteria: real proposals pass without paid calls and feature gates remain intact.
- Planned Touch Files:
  - `ima/research_controller.py`
  - `ima/research_expansion.py`
  - `tests/test_research_controller.py`
  - `tests/test_research_expansion.py`
- Checklist:
  - [ ] Verify catalog/availability protection.
  - [ ] Commit scoped admission repair.

### Subphase 2.2: Numerical Acceleration
- Commit: perf(research): batch Gaussian likelihood and linearize race reductions.
- Tests: scalar/finite-difference parity, mixed/shuffled fields, tails, prediction parity and benchmark.
- Success Criteria: equivalent objective and >=3x throughput; bounded scratch memory; diagnostic stage/evaluation timings.
- Planned Touch Files:
  - `ima/performance_probit.py`
  - `tests/test_probit_gradient.py`
  - `scripts/benchmark_probit.py`
- Checklist:
  - [ ] Replace repeated scans and batch integration.
  - [ ] Preserve stable tails and convergence safeguards.

## Phase 3: Verification
### Subphase 3.1: Scientific And Regression Proof
- Commit: test(research): record scientific runtime verification.
- Tests: full pytest, CI and real saved three-fold trial with 2,400-second deadline.
- Success Criteria: finite completed model/metrics and exact lineage, no numerical regressions.
- Planned Touch Files:
  - `docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md`
- Checklist:
  - [ ] Complete regression and CI.
  - [ ] Execute/read back real scientific canary.

## Phase 4: Rollout
### Subphase 4.1: Pinned V7 Successor
- Commit: docs(research): record pinned rollout proof.
- Tests: own-unit/config/revision, planner acceptance, dispatch, MLflow cost/tokens and completed model linkage.
- Success Criteria: new release truly executing; previous V7 drains; old V6 fits untouched; no shared-service restart.
- Planned Touch Files:
  - `docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md`
- Checklist:
  - [ ] Push/merge and stage immutable release.
  - [ ] Start successor and verify real research/tracking.
