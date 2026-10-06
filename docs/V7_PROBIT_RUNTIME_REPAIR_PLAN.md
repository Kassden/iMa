# V7 Probit Runtime Repair

## GOAL
Preserve Gaussian race-winner likelihoods while making real heteroscedastic trials practical, then deploy and verify a pinned V7 successor with corrected feature admission.

## Acceptance Criteria
- [x] Profile identifies objective and preparation costs.
- [x] Analytic objective/gradients match the independent scalar reference, including heterogeneous scales, tiny fields, shuffled runners and rare winners.
- [x] Same-shape measured objective throughput improves at least 3x with bounded scratch allocation.
- [x] Real three-fold heteroscedastic trial completes within the original 2,400-second limit, with diagnostics and MLflow linkage.
- [x] Source is pushed, CI is green, and pinned V7 successor accepts an actual OpenRouter decision and dispatches fits.
- [x] Old V6 fits untouched; prior V7 drains without hotpatching; 80/20 dispatch and frozen evaluation unchanged.

## Research
Use existing NumPy/SciPy compiled array kernels and SciPy L-BFGS-B; Joblib threading only if measurement justifies it. Retain the scalar mathematical oracle and adaptive QUADPACK prediction checks.

## Root-Cause Baseline
- Proven: attempt-c3b02be1602226a18efa043e1e23149bffd7c1f2d7420f4facc262c198b68738 ended at 2,400 seconds with execution_timeout; private peak 3,531,145,216 bytes. Heteroscedastic, order 32, max_iter 500, three whole-meeting folds with at least 9,000 training races.
- Proven: repeated full-runner scans per race in matrix/scale centering; Python loops over races and runners for quadrature.
- Proven: baseline-only admission rejects properly declared registered speed extras. Prior-turn repair exists locally and must be verified/committed separately.
- Likely: grouping scans and tiny quadrature calls dominate; establish their relative shares with cProfile before choosing optimizations.
- Possible: post-fit convergence verification is expensive; expose stage timings rather than assuming optimizer nonconvergence.
- Follow-on proven by saved-recipe replay: first accelerated candidate reaches its numerical guard after 529 seconds but order 32 is underresolved. Preserve the rejection receipt. Batch strict fine-probability verification and warm-start increasing integration order up to 512, within the original total iteration and worker time budgets. Record configured/effective orders and every refinement round; do not pretend this was a successful fit or lower tolerances.
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
- Planned edit surface: ima/performance_probit.py, ima/research_controller.py, ima/research_expansion.py, tests/test_probit_gradient.py, tests/test_research_controller.py, tests/test_research_expansion.py, scripts/benchmark_probit.py, docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md; own-user temporary ops/packaging receipts. Scope expansion: deploy/systemd/ima-research-expansion-supervisor and tests/test_research_expansion_deploy.py, only to admit correctly named agentic_v7_* campaign directories while preserving existing identity/own-root checks.
- Numerical follow-on scope: tests/test_performance_probit.py and tests/test_probit_refinement.py, to verify bounded refinement and batched prediction/oracle parity after the real trial exposed underresolution.
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
  - [x] Collect profile and saved attempt identity.
  - [x] Validate plan and preserve old fits.

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
  - [x] Verify catalog/availability protection.
  - [x] Commit scoped admission repair.

### Subphase 2.2: Numerical Acceleration
- Commit: perf(research): batch Gaussian likelihood and linearize race reductions.
- Tests: scalar/finite-difference parity, mixed/shuffled fields, tails, prediction parity and benchmark.
- Success Criteria: equivalent objective and >=3x throughput; bounded scratch memory; diagnostic stage/evaluation timings.
- Planned Touch Files:
  - `ima/performance_probit.py`
  - `tests/test_probit_gradient.py`
  - `scripts/benchmark_probit.py`
- Checklist:
  - [x] Replace repeated scans and batch integration.
  - [x] Preserve stable tails and convergence safeguards.

## Phase 3: Verification
### Subphase 3.1: Scientific And Regression Proof
- Commit: test(research): record scientific runtime verification.
- Tests: full pytest, CI and real saved three-fold trial with 2,400-second deadline.
- Success Criteria: finite completed model/metrics and exact lineage, no numerical regressions.
- Planned Touch Files:
  - `docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md`
- Checklist:
  - [x] Complete regression and CI.
  - [x] Execute/read back real scientific canary.

## Phase 4: Rollout
### Subphase 4.1: Pinned V7 Successor
- Commit: docs(research): record pinned rollout proof.
- Tests: own-unit/config/revision, planner acceptance, dispatch, MLflow cost/tokens and completed model linkage.
- Success Criteria: new release truly executing; previous V7 drains; old V6 fits untouched; no shared-service restart.
- Planned Touch Files:
  - `docs/V7_PROBIT_RUNTIME_REPAIR_PLAN.md`
- Checklist:
  - [x] Push/merge and stage immutable release.
  - [x] Start successor and verify real research/tracking.

## Verification Evidence
- Source `efe1f6bcca752c6b36047f3b6daa3cb93df558be`; PRs 19 and 20 merged. PR20 merge `17918ea7c3af6c1718a581ea89fee284cd3b5dac`; research CI `37417236494` passed.
- Final local regression: 977 tests, 493 subtests passed. Independent tester added 11 oracle, tail, bounded-allocation and failure tests.
- Unprofiled server objective benchmark: seed 42, 9,000 races, 14 runners, 150 predictors, one native thread, two measured repetitions after warm-up. Median 41.440810 seconds before versus 6.861340 seconds after: 6.039755x throughput. Loss difference below 1e-12; maximum gradient difference 1.665335e-16. Receipt: `/home/imaopt/research-v2/ops-parent/v7-probit-runtime-20261006/raw-benchmark.json`.
- Full scientific replay is separate from the benchmark: immutable 171,782-runner dataset and saved failed recipe, three whole-meeting folds, unchanged 2,400-second deadline. Completed in 1,859.065627 seconds (31 minutes), fundamental development race log loss 2.1789541533645167. Fold fits: 565.662/606.105/618.330 seconds, 11,580/12,085/12,585 training races, 134 predictors, 46/47/46 total iterations. All three passed at effective integration order 128, maximum refinement errors 4.166479e-05/2.821682e-05/3.379968e-05. Private-memory peak 3,670,790,144 bytes; no recorded OOM. This proves runtime feasibility, not superiority to existing champions or market.
- MLflow readback: FINISHED run `e9c33c376fbd4ae99a9f2764192e7e05` in experiment 7; READY version 285 of `ima-agentic-v6-research-candidates-win-probability`. Per-fold diagnostics logged as an artifact. Historical experiment/model names are retained; run and campaign identifiers explicitly identify V7.
- Activated successor: `agentic_v7_probit_runtime_efe1f6b`, own-user unit `ima-v7-probit-runtime-efe1f6b.service`, enabled/active at 05:43 UTC. New fit ceiling 23 initially reserved three older fit slots; the old V7 graph finished naturally before activation, leaving two preserved legacy V6 fits and a combined ceiling of 25. 24 CPU-thread allocation and 100 GB decimal memory budget unchanged. Old V7 drained through its STOP marker, not cancellation; both legacy V6 fits are preserved.
- Live decision D000001 accepted from `deepseek/deepseek-v4.1-flash`: five programs, chosen/allocated 12/12 trials (10 classical Benter, two experimental budgets), full research memo SHA256 acknowledged. Programs cover fixed Benter/boosted controls, Benter transforms, horse/jockey discovery and composed Benter+boosted pooling. Trial ceiling 260 is not a mandated spend, fit capacity remains independent. Deterministic cumulative 80/20 dispatch logic is unchanged.
- Native decision trace `tr-c98c4ef14878ff00210491dae11c1b71`: 27,838 input + 9,975 output = 37,813 tokens; reported USD 0.0203214. SDK verified exact native metadata and written-summary derivation. Fresh Chromium readback verified populated Tokens/Cost columns and planner-only filtering in [V7 Planner View](http://100.95.24.121:5000/#/experiments/7/traces?traceViewShareKey=1791265534106manrjv3b).
- Production readback: first completed attempt `attempt-21210ff5b4bc7ead9cd8a79cc127fa1965114a37835f28ebf7fab4c26a194017` has FINISHED run `ba538581432c4bccab8542ec557daf01`, READY model version 286, exact pinned-source lineage and objective 2.174290501294411. Four more fits running at observation; adaptive fit cap advanced 2 to 4 after measured headroom. No failed trial, planner/tracking error or pending tracking/trace delivery observed. The source revision also passed post-merge main CI `37417862873`.
