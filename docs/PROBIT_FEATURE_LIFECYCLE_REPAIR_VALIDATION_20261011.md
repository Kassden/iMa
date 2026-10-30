# Probit and Feature Lifecycle Repair: Validation

Verified on 2026-10-12. This distinguishes code, live use, canaries and scientific gains.

## Outcome

The core repair is active in the existing own-user V7 campaign. It is not a new independent research loop. One planner controls feature recipes, dataset choices and model programs; deterministic builders, validators, scheduling and fold-local optimization execute those choices.

Active release: `10d6f094b0752cc5fdfa8d6e3a1706d25be02c83`.
Release manifest SHA256: `b709302ca24e66b995131c860068eee36b233e5fe59afb8a4c972bebe21fc48f`.
Service: `ima-v7-openrouter-recovery-20261007.service`, user `imaopt`.
Activated: 2026-10-12 13:49:34 UTC; zero fits cancelled.
MemoryMax remains 100,000,000,000 bytes; CPU quota remains 24 CPU-seconds per second. Other users/services were not changed.

Additional defensive namespace binding/assertions at `9057606` are committed, pushed and separately qualified, not hotpatched into the running fit. A read-only canary against the active release proved `_capabilities`, `_planner_spend`, `_worker`, `_prepare_program` and `BoundedFitExecutor` all reach the actual loop; its globals are the module namespace. Thus that extra hardening does not block the requested live behavior. No automatic handover timer has been installed.

## Runtime Contracts

- `initial_seconds`: max(14,400, explicitly requested allowance, comparable successful p95 * 1.25, comparable failed/censored maximum * 1.25).
- Comparability uses the workload fingerprint. Cold cost extrapolations are not deadlines. Timeout observations never become successful samples.
- `hard_seconds`: initial allowance + 7,200 seconds. Up to four 1,800-second productive extensions, durably recorded.
- Progress must be fresh within 300 seconds and belong to the same process/task/fold. Iteration/objective improvement, numerical verification and successful fold progress count; CPU heartbeat alone does not.
- Fresh nonproductive work and stale/stalled progress have different health labels. Unconverged or numerically unverified models are not promoted as successes.
- Warm-start keys cover actual transformed training rows/labels/matrix, features, fitted preprocessing, fold/data/context and implementation dependencies. No cross-fold/evaluation-label transfer.
- Profiling records preprocessing, objective, optimizer, verification and total fit time. No unmeasured core increase was made.

## Feature Stage

Separate MLflow experiment: [ima-feature-builds](http://100.95.24.121:5000/#/experiments/14).
Build run: [f0aadd9664ea4ace975c3eee24b372e3](http://100.95.24.121:5000/#/experiments/14/runs/f0aadd9664ea4ace975c3eee24b372e3).
Build: `feature-build-6146674d55533569911ca8f64ae9897c79ccaa303a44b076b20850ea32df0ff1`.
Successor: `dataset-38574eb5c9913b61e1972cbaf1fbde2dac7455d552cdc0c6f622480d22947116`.
Population retained: 171,782 rows, 13,924 races; parent row keys, protocol and confirmation identities preserved.

Validated deterministic formulas:

| Feature | Formula | Unit |
| --- | --- | --- |
| body_to_carried_weight_ratio | declared_weight / actual_weight | 1 |
| distance_relative_to_previous | distance / last_distance | 1 |
| distance_shift_from_previous | distance - last_distance | m |

Content IDs: `1c60317c660db7e2994c886a`, `662ee09f0791898be9f061d6`, `c1feefd23facd4da62e5726e`.
Feature IDs are `dfs_` plus each content ID. Persisted registry definitions and predictor catalog bind the formulas to these IDs. Optional `definition_id` in a recipe input may be null before registration; that is not missing materialized lineage.

Typed AST suggestions from the planner use the same validator/builder. Label/future refs, wrong units, nonfinite/constant/all-missing outputs and inadequate coverage are rejected. Arbitrary executable Python is not accepted.

Catalogs include bounded candidates, typed definitions, diagnostics, downstream outcomes and omission counts. They use a cheap index and bounded dataset verification. Scores remain separated by target, actual evaluation population, protocol, metric and probability basis; unpaired observations are not causal feature merit.

Feature builds have their own durable tracking outbox and artifact-hash readback. Model feedback failure preserves successful training and records a retryable pending-feedback artifact. Model runs expose `feature_build_id`, `feature_build_run_id` and `feature_definition_ids`.

## Actual Consumers

An isolated real one-fold Benter consumer trained using all three new features. Its package, model feedback and MLflow parameters were read back:

- Model run: [9e116e80e926461db1bfa36c4c86fffd](http://100.95.24.121:5000/#/experiments/15/runs/9e116e80e926461db1bfa36c4c86fffd).
- Score: 2.1883918912087057; functional canary only, not champion-comparable.

The live planner then selected the successor dataset itself:

- D237 accepted; chosen budget 3, including one new Benter trial and extensions. Cost USD 0.023030788.
- D238 accepted; chosen budget 3. Cost USD 0.022092594.
- Both decisions see the feature catalog and the 14,400-second probit capability.
- Fresh production completion: `attempt-fe607550bbf3cc34f0bcf30705b1c49c795c728fb174232af5266b1fc714062c`, objective value 2.1562666448551417, non-null feature build ID and lifecycle manifest hash.
- Latest readback at 14:20 UTC: 745 completed, 38 failed, one active Benter fit, zero pending tracking/tells. Known and total planner spending USD 11.032910359; no unresolved decision IDs.

Reusing a materialized successor deliberately needs no duplicate recipe.feature_definitions/feature_discovery. The model's build/definition lineage, dataset and actual selected features provide the proof. Old champion recipes/history were not rewritten.

An isolated heteroscedastic probit fit on 80 complete historical races / 995 runners converged and verified quadrature, emitted optimizer/fold telemetry and safely reused coefficients on an identical training fold. Profiling attributed approximately 3.8 seconds to objective evaluation and 1.2 seconds to numerical verification; this small canary is not representative of full-fold wall time or evidence that more cores would help.

## Billing and Context Incident

D236 failed before a provider request because local context measured 300,011 > 300,000 bytes. Empty transport/call receipts alone are insufficient to waive unknown billing.

The repair uses an independently inspected, release-sealed qualification for exactly D236's decision and evidence hashes, plus revalidation that there is no dispatch/response evidence. Altered/missing/forged inputs remain unresolved. It records `verified_not_dispatched`, not a fabricated provider cost or receipt. Optional historical narratives are bounded without dropping parent IDs, safety metadata or pending recipes. Fresh D237/D238 demonstrate planning recovery.

## Verification and Limits

Final targeted command covered performance/probit gradients/refinement, runtime policy, progress/warm starts, worker lifecycle, feature lifecycle, real consumer integration, expression/dataset contracts and paid request guard: **182 tests and 89 subtests passed**. Six expected warnings concern all-missing prize values in a synthetic fixture. No test was treated as passed after a failing exit code.

Adversarial checks cover stale/NaN/mismatched progress, hard caps, restart ownership, fold/matrix/preprocessing mismatches, target/future leakage, corrupt datasets, duplicate builds, tracking failure and forged billing exemptions. Deployment canaries checked original request-guard parity and actual consumers. Failed staging attempts were retained as evidence, never activated.

Commits are pushed on `fix/probit-runtime-feature-lifecycle`; this branch was not merged into main. GitHub returned no CI runs for the branch; local and server qualification are the verified checks. Preexisting dirty files and unrelated branch history were preserved; the global scope checker cannot claim a clean whole-worktree pass.

Not demonstrated yet: planner-invented novel formula adoption, a completed full-campaign heteroscedastic trial under its extended deadline, or a statistically confirmed improvement from these features. The 2.1494228320541287 pooled champion remains the historical benchmark. A new feature build or a better score on a different protocol is not proof of gain.
