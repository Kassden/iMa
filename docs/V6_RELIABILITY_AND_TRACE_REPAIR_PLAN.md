# V6 Reliability And Trace Repair

## GOAL
Implement and verify the evidence-backed repairs, including written MLflow trace summaries, then launch a pinned successor after canaries pass. Execution authorized by the user after the initial proposal-only delivery.

## Acceptance Criteria
- [x] Inspect live ledger, queue, journal, source and trace evidence.
- [x] Separate proven mechanisms, hypotheses and missing historical evidence.
- [x] Correct earlier statements about absent probit execution and historical OOM.
- [x] Specify atomic repairs, tests, rollout and factual trace summaries.
- [x] Validate this document and generate its read-only dashboard.

## Root-Cause Baseline
Audit date: 2026-10-05. Artifacts have different snapshot times. The host journal displays Shanghai (+08:00), not UTC.
- Hypothesis model: proven C1-C7, likely H1-H3, possible H4; evidence and counter-evidence below.
- Missing evidence: historical lock owner and OOM victim unavailable; no unverified allocation blame or fee assertion.
- Mutation boundary: read-only server investigation, documentation-only local edits.
- Remediation mapping: A to C5/H4; B to C3/H1; C to C1/C2/H2/H3; D to C4/H3; E to C6/C7.

### Evidence Inventory
- Campaign: `/home/imaopt/research-v2/campaigns/agentic_v6_openrouter_continuous_4864156`; pinned source `48641567656ae693ca5664f5f49bc01bea572b18`.
- Own unit: `ima-v6-openrouter-continuous-4864156.service`.
- Durable ledger: 87 completed, 15 failed, 5 reserved, 2 running. Audited tracking/tell/outbox queues drained.
- Probit program `999404172b3c42b6`, authorized by D7: queue records two dispatches Oct04 17:19:49UTC, resumed Oct04 19:14:32UTC after OOM.
- Running attempts: `attempt-240c092717471ee415c4c2663b8d31cc1d3e5eb2f5a6921038fd9d8ee1739f95` and `attempt-9d38747dedc38ed8c67ccff50da11a0bdadee10a32e4b3f9482695c53fded912`.
- Workers 1094560/1094564: roughly8.8 hours elapsed after restart, over11 CPU-hours each, first-fold training. They are computing, not cache-lock waiting.
- Journal: Oct05 03:12:21Shanghai = Oct04 19:12:21UTC, kernel OOM kill, unit failed and automatically restarted; reported peak93.1G. New-invocation zero OOM counters do not erase this event.
- Fifteen terminal failures are cache lock timeouts; thirteen belong to discovery program `6593be53cf2ede7f`, Oct04 17:26-17:38UTC.
- D17: DNS ConnectError after10.07 seconds, before TCP completion/TLS/HTTP transmission; no generation ID, response or physical-call receipt. Original usage empty; subsequent paid planning frozen.
- Known reported planner subtotal USD0.477406735; total unresolved. Do not substitute zero for missing fees.
- D17 trace delivered with OK state; preview omits failed planning, DNS phase and billing uncertainty.
- Current-campaign best fundamental loss2.1678962151517527, boosted, still worse than compatible prior V6 champion2.166148682757164.

### Proven Causes
| ID | Mechanism | Consequence |
| --- | --- | --- |
| C1 | Restart ramp resets to2; advancement requires a new successful measurement. Both slots hold slow probit fits. | Ready jobs report `max_fits` despite free CPU/RAM. |
| C2 | `max_wall_seconds` exists only in the program schema, not execution enforcement. | A costly trial can run indefinitely relative to its declared budget. |
| C3 | Preparation defaults to120-second lock wait, holding exclusive flock around `builder()`. | Infrastructure contention becomes failed scientific trials. |
| C4 | Current invocation counters omit previous OOM/restart. | Status-based health checks miss historical pressure. |
| C5 | Any missing usage is treated as unknown charge, even D17's pre-dispatch failure. | Planner cannot recover from an unsent network failure. |
| C6 | Trace preview presents counts/budgets/champions, omitting runtime failures; serialization hardcodes OK. | Delivered trace looks successful when planning failed. |
| C7 | `trials.jsonl` represents terminal results, unlike durable ledger running/reserved state. | Reading only it falsely suggested no probit had dispatched. |

### Hypotheses And Missing Evidence
- H1 likely: expensive same-key fold construction exceeded120 seconds while sibling trials waited. Historical owner/key not recovered; do not attribute a specific builder as proven.
- H2 likely: Gaussian probit numerical differentiation is costly. Source proves L-BFGS-B receives no gradient and evaluates race/horse quadrature per parameter evaluation; heteroscedasticity doubles coefficient dimension. Runtime share needs profiling.
- H3 likely: probit cold estimate750 seconds is inadequate; cost fingerprints omit quadrature and heteroscedasticity. Underestimated RAM/private copies may contribute to OOM, but exact victim/allocation is unknown because kernel journal access was unavailable.
- H4 possible: resolver/Tailscale/upstream DNS trouble. Exact network origin unproven; no route changes justified.
- Disproven: probit never dispatched; campaign never OOMed; low resources necessarily mean insufficient server capacity. No evidence credits caused DNS failure.
- Missing-evidence policy: retain uncertainty, add future stage/key/PID telemetry, never reconstruct absent receipts or pretend DNS exception alone proves free billing.
- Original audit mutation boundary: read-only server inspection. The subsequently authorized execution allows additive own-user dependencies, qualified immutable successor releases and bounded planner canaries. No source hotpatch, cache deletion or other-user/service changes. User explicitly chose to keep waiting for both old probit fits: do not cancel them or switch the old campaign while they remain active.
- Remediation mapping: packages A-E below map C1-C7/H1-H4; numerical acceleration/copy reduction are measured improvements, not proven historical root causes.

## Research
- Built-in options: existing ledger, flock cache, SciPy optimizer, transport telemetry and MLflow outbox.
- Off-the-shelf options: Joblib sharing and HTTPX transport phases; no new distributed framework needed.
- Official sources and implementation decisions follow.

## SOTA, Standards, And Best Practices
- Implementation decision: extend mature libraries and existing owners using official guidance; profile before optimizing.
- Reuse Joblib read-only, uncompressed memmaps and existing content-addressed cache. Memmaps do not prove downstream pandas/estimator conversions avoid copies. [Memmap example](https://joblib.readthedocs.io/en/stable/auto_examples/parallel_memmap.html), [parallelism](https://joblib.readthedocs.io/en/latest/user_guide/parallel.html).
- Reuse SciPy optimization with a verified `jac` if profiling supports it; preserve Gaussian winner likelihood, identification and quadrature convergence gates.
- Classify HTTPX physical attempts by actual transport phases, not exception class alone. [HTTPX exceptions](https://www.python-httpx.org/exceptions/).
- Reuse MLflow native preview, tags, span attributes and durable outbox. [Manual tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing), [trace table](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui), [tags](https://www.mlflow.org/docs/latest/genai/tracing/attach-tags/).
- Reject blanket concurrency halving, cap above100 decimal GB, longer lock timeout as sole remedy, assumed zero fees, killing a shared pool to cancel one trial, or paid narration of deterministic counts.

## Regression Guardrails
- Damage radius: small for this documentation delivery; proposed runtime implementation is large.
- Branch strategy: documentation remains on `feat/agentic-research-expansion`; separate branch `fix/v6-runtime-and-trace-reliability` required before proposed source implementation.
- Current edit surface: this Markdown and generated `.mega` state/evidence/dashboard only. Existing dirty files protected.
- Planning damage radius small: current `feat/agentic-research-expansion` branch. Implementation damage radius large: dedicated `fix/v6-runtime-and-trace-reliability` branch from reviewed base before source edits.
- Protected: datasets/splits/confirmation labels, source identity, 80/20 dispatched-trial accounting, single ledger owner, exactly-once Optuna tells, model/run lineage, original receipts and reported costs.
- Host boundary: only own research-v2 paths as `imaopt`; Cortex/Solar, MLflow service, acquisition services, routes, credentials and other users untouched. No live betting.
- Consumers: planner/controller/builders/model workers, terminal operator and MLflow. Proof: fixture fault drills, focused/full tests, Linux canaries, backend readback and bounded live observation.
- Atomic commit units: billing, preparation, runtime deadlines, numerical optimization separately, memory accounting, summaries, rollout evidence.

## Dependency and Tooling Preflight
No install needed for this documentation delivery. Install/repair commands for future verification: `.venv/bin/python -m pip check`; project-local `npx playwright install chromium` after verifying the project's Playwright dependency.
Existing local `.venv` and server private Python3.12, unittest, SciPy, Joblib, Optuna, HTTPX and MLflow. Inventory/pip-check before implementation; no upgrades merely for planning. Browser proof eventually requires project-local Playwright/Chromium; install if absent. Verify installed MLflow capabilities, not latest documentation alone. Paid live tests require audited cost recovery first.

## Deterministic Real-User Test
- Entry: this plan and generated dashboard; eventually CLI summary and MLflow preview/detail.
- Fixture: two slow probit trials, cold same-key prep with26 followers, ready cheap Benter/boosted work, pre-dispatch DNS failure and ambiguous post-send failure.
- Assertions: correct active/terminal counts, elapsed/deadline, blockers, freeze, unknown-total-vs-known-subtotal, matched champion delta, historical OOM and source identity. No paid HTTP in fixtures.
- Current command: `python3 /Users/milkingthesun/.codex/skills/megaskill/scripts/mega_plan_check.py docs/V6_RELIABILITY_AND_TRACE_REPAIR_PLAN.md`.
- Record test commands, snapshots and backend readbacks in `.mega/evidence.jsonl`.

## Fulfillment and Readback Proof
- Final user-visible result: canonical diagnosis/proposal and generated dashboard, explicitly not deployed code.
- Required write/mutation operation: write this Markdown and generated dashboard only.
- Readback surface: Markdown and dashboard on the shared filesystem.
- Expected content, rows, fields, counts, or behavior: seven evidenced failure classes, six repair packages, corrected timeline and written summary.
Current fulfillment is a validated diagnosis and proposal, not deployed repairs. Read back Markdown and dashboard. Future proof requires small probit completion, cheap work completing alongside a slow trial, one accepted planner decision using completed results, native cost/status/model links in backend traces, and healthy recovery. A running service or delivered trace alone is insufficient.

## Armageddon Mode
Attack scope: large future runtime surface; failure modes and edge cases below are must-fix release gates.
Test producer >120s/death, corrupt/pinned cache, concurrent followers, high quadrature, optimizer stall, runaway descendant, OOM/restart, pre-send DNS and post-send timeout, duplicate delivery/cost, missing/mismatched champions, unavailable MLflow, misleading OK state, and no post-restart completion. Accounting ambiguity, label leakage, source drift, whole-pool termination or false success are release blockers.

## Generality Guardrail
Existing mechanism: durable controller/store/cache/telemetry owners. Recurrence: high across model families. General mechanism decision: shared snapshot and supervision, not per-model controller forks.
Extend existing store/scheduler/cache/telemetry/transport owners. One snapshot contract serves CLI, artifacts and MLflow; no competing controller or hand-maintained summary. These issues recur across models, not just probit.

## Implementation Packages
Each package names edit ownership, tests, commit and exit criteria. Re-read baseline before each; create repair branch first.

### A: Dispatch-Aware Billing (C5/H4)
- Files: `ima/openrouter_transport.py`, `ima/openrouter_orchestrator.py`, `ima/research_expansion.py` and corresponding tests.
- Persist each physical attempt: invocation, resolution, connect, TLS, HTTP headers/body start, response, generation ID and usage. Distinguish `not_dispatched`, `dispatched_cost_unknown`, `reported_cost` from operation success.
- Bounded retry only when ALL attempts are proven unsent. Prior sent retry followed by DNS still stays frozen. Ambiguous transmission remains blocked until provider receipt or explicit audited resolution.
- Add immutable reconciliation receipt for D17, not overwrite it. Keep known subtotal, unresolved IDs and null total separate. Do not manufacture usage/cost zero from an exception.
- Tests: pre-send DNS; TLS; earlier-sent retry; body timeout; receipt reconciliation/restart; duplicate cost. Saved D17 phase replay requires no paid call.
- Commit: `fix(planner): distinguish unsent failures from uncertain billing`.
- Success: only proven pre-dispatch failure permits bounded recovery; ambiguous billing still freezes.

### B: Single-Producer Fold Preparation (C3/H1)
- Files: `ima/research_preparation.py`, `ima/research_executor.py`, `ima/research_expansion.py`, `ima/research_store.py`; cache/executor/controller tests.
- Compute complete fold keys before fit admission; schedule one producer per key. Followers await artifact dependencies without occupying fit workers/loading full datasets while flock waits.
- Preserve process-safe locks, atomic manifests/checksums, reader pins and safe eviction. Replace dead producer only with owner/lease evidence; never delete a live lock.
- Expose key/owner/stage/heartbeat/deadline/follower count. Treat transient infrastructure waits separately from scientific failure. Reuse immutable attempt identity for bounded recovery, no simultaneous duplicate fit. Preserve original15 failures as historical evidence.
- Tests: one build/26 followers; >120s producer; crash before publish; corrupt artifact; restart/replay; pinned eviction; different labels/splits/transforms yield different keys.
- Commit: `fix(preparation): schedule shared fold builds as dependencies`.
- Success: one build per key, identical validated artifact for followers, unrelated cheap work continues.

### C: Deadlines, Fair Recovery And Probit Cost (C1/C2/H2/H3)
- Files: `ima/research_expansion.py`, `ima/research_resources.py`, `ima/research_specs.py`, `ima/performance_probit.py`; optional new `ima/research_worker_runtime.py`; focused runtime/probit/resource tests.
- Define `max_wall_seconds` as per-attempt execution deadline; report dependency wait separately. Persist start/deadline/active time; restart cannot reset allowance indefinitely.
- Supervise separately terminable attempt process groups. Running `Future.cancel()` is insufficient. Kill only expired attempt/descendants, preserve diagnostic artifacts, release resources and record one terminal result/tell.
- Heartbeat fold/stage/iteration, wall/CPU duration and convergence. Active heartbeat never excuses exceeded deadline.
- At recovery cap2, allow at most one expensive trial and leave a slot for cheap measured work. Respect deterministic80/20 reservations; report legitimate lane restriction rather than silently bypass it. Admission must consider expensive-family occupancy before choosing both slots.
- Fingerprint mean/variance dimensions, race/runner/feature count, quadrature order, heteroscedasticity, iterations, folds and threads. Retain censored timeout/OOM bounds.
- Profile small fixture first. Benchmark analytic gradient/batched race evaluation against reference before numerical edits. Keep convergence and identification gates.
- Tests: isolated timeout, child cleanup, one tell, replay, cheap backfill, finite-difference gradient agreement, equal/unequal scales, complete races, finite normalized probabilities, quadrature parity.
- Runtime commit: `fix(runtime): enforce trial deadlines and recovery fairness`.
- Numerical commit separately, only with measured gain: `perf(probit): add verified likelihood gradients`.
- Success: small heteroscedastic canary finishes or explicitly reports convergence failure; no unexplained nine-hour fold; slow trial does not monopolize recovery.

### D: Restart-Aware Memory (C4/H3)
- Files: `ima/research_resources.py`, `ima/research_scheduler.py`, `ima/research_expansion.py`; own supervisor only if needed; resource/scheduler/deploy tests.
- Preserve100 decimal GB max,24CPU budget,26fit ceiling and host reserve. Ceilings are not targets.
- Persist invocation/OOM/interruption/high-watermark history; feed censored peaks into estimates. Current zero counters cannot clear prior incident.
- Charge controller and idle worker residency, preparation/selection growth, fit private copies, backend/uploader work and cgroup charged memory. Deduplicate true shared identities only.
- Profile pandas reconstruction/estimator conversions; drop unused columns early, share immutable matrices and cache identical training-only selection. Keep learned processing fold-local.
- Ramp using sustained30-second headroom and representative evidence; let a cheap isolated canary supply post-restart progress. Costly families need their own measurements before expanded occupancy.
- Tests: history across restart, cold family, prep underestimation, cap2/4/8/12/16/26, failed/censored samples, private-vs-shared accounting, headroom drop and protected services.
- Commit: `fix(resources): retain restart pressure and measured admission`.
- Success: replay rejects unsafe combinations; live canaries stay under hard envelope without OOM.

### E: Written Trace Summary (C6/C7)
- Files: `ima/research_telemetry.py`, `ima/research_expansion.py`, `ima/research_store.py`, `scripts/optimize.py`; new shared `ima/research_summary.py` if existing owner cannot contain formatter cleanly; summary/telemetry tests.
- Versioned summary from consistent ledger+decision+resource snapshot. Include UTC, invocation/revision and input/output watermarks. Render the same object to CLI, JSON, Markdown and MLflow outputs.
- Names: `<campaign> | decision D000017`, `<campaign> | execution S001099`, `<campaign> | dataset B000001`. Separate sequences, not ambiguous cycles.
- No paid LLM calls to explain counters. Preserve planner's actual hypothesis/reason separately; do not invent rationale.
- Decision fields: outcome, model/target/programs, chosen/allocated/held/continuing budgets, rejected fields, next action, reported USD/unknown receipts/transport phase, generation link, exact memo ACK.
- Execution fields: accepted/queued/preparing/running/completed/failed/timeout/interrupted; each active model/fold/progress/elapsed/deadline; wait reasons; cap/ceiling; CPU%, RAM current/peak/limit; historical OOM/restarts; upload/tell backlog.
- Champion fields: target/objective/comparison key and dataset/protocol/population identity, model family, prior/current value, absolute/relative delta, attempt/run/model links. Compare matched contracts only; absent comparable score is not `best None`.
- Preview priority: health, operation outcome, blocker, champion delta, cost. Full written paragraph/table in root outputs and `summary.json`/`summary.md`.
- Tags: `ima.operation_status`, `ima.blocker`, `ima.model_families`, `ima.active_trial_count`, `ima.completed_count`, `ima.summary_schema_version`. Native USD only for reported receipts. Verify installed UI custom-column support rather than promise arbitrary columns.
- A planning failure is a failed domain operation even when telemetry delivery succeeds. Set operation status/error and MLflow span/trace failure where supported; preserve original old traces. Separate delivery status.
- Emit on meaningful changes plus bounded heartbeat; avoid thousands of identical full traces. Locally persist heartbeat history/current summary; existing trace outbox remains exactly-once.
- Historical annotations must be labeled audits with original time/source; never rewrite original evidence or reaggregate cost.
- Tests: D17, unknown cost, no champions, mixed targets, long errors, input/output snapshot separation, running absent from terminal file, restart OOM, duplicate upload, MLflow outage. Backend readback plus Playwright preview/detail proof.
- Commit: `feat(observability): publish factual campaign trace summaries`.
- Success: operator can see stalled probit, planner freeze, historical OOM and matched progress without an SSH investigation.

### F: Canary And Successor Rollout
- Only own research-v2 release/config/unit. No scientific hotpatch or unrelated merge.
- Focused command: `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -W ignore -m unittest tests.test_research_preparation tests.test_research_scheduler tests.test_research_resources tests.test_research_expansion tests.test_research_telemetry tests.test_performance_probit` plus new tests; then full discover, Linux canaries and CI.
- Fault drills before live paid tests. Verify source/config/dependency/dataset/memo hashes and protected-host baseline.
- User maintenance decision: keep both overdue fits running until they finish. Qualify the successor separately; do not cancel the old fits or switch the old campaign before their completion. Archive final progress and preserve originals.
- Start pinned successor with compatible reference champions and explicit interrupted ancestry. Small scientific canaries first, progressive measured capacity, not immediate26 expensive jobs.
- Live proof: completed Benter, boosted and small heteroscedastic probit; cheap completion alongside slow fit; accepted planner decision uses those results; native cost/status/model links read from MLflow; at least30-minute steady run plus isolated recovery drill.
- Commit: `docs(operations): record verified reliability successor`.
- Blockers: ambiguous sent-call fee, failed numerical canary, nonisolated cancellation, wrong lineage, duplicated prep, absent backend readback, memory overshoot. No healthy verdict while unresolved.

## Example Written Summary (Audit, Not A New Published Trace)
**V6 degraded:87 completed /15 failed /2 running /5 reserved.** Two heteroscedastic probit fits remain on fold1 after approximately8.8 hours since restart. Recovery cap2 is occupied; ready jobs wait on `max_fits`. Fifteen earlier discovery trials failed cache waits. OOM Oct04 19:12:21UTC caused restart. Planner frozen at D17 after pre-dispatch DNS failure; known subtotal USD0.477406735, total unresolved. Best current fundamental loss2.167896(boosted), worse than compatible prior V6 best2.166149. Active computation is not proof of healthy autonomous progress.

## Ordered State and Dashboard
State/evidence: `.mega/state.jsonl`, `.mega/evidence.jsonl`. Dashboard: `.mega/dashboards/v6-reliability-and-trace-repair.html`, generated from this document. Record investigation/proposal/check/readback; no implementation/deployment event in this delivery.

## Phase 1: Investigation
### Subphase 1.1: Read-Only Root-Cause Audit
- Commit: documentation-only audit; no server mutation.
- Tests: ledger/queue/journal/source/trace inspection and independent executor/telemetry audits.
- Success Criteria: proven mechanisms separated from historical attribution gaps.
- Planned Touch Files:
  - `docs/V6_RELIABILITY_AND_TRACE_REPAIR_PLAN.md`
- Checklist:
  - [x] Audit live evidence and correct probit/OOM statements.
  - [x] Record root causes, hypotheses and missing evidence.

## Phase 2: Proposal
### Subphase 2.1: Repair And Trace Summary Contract
- Commit: `docs(research): propose V6 reliability and trace repairs`; exclude unrelated dirty files.
- Tests: plan checker, Markdown/dashboard readback, scope/state final gate.
- Success Criteria: another model can execute A-F without guessing authority, files, tests or acceptance.
- Planned Touch Files:
  - `docs/V6_RELIABILITY_AND_TRACE_REPAIR_PLAN.md`
- Checklist:
  - [x] Specify repair packages, adversarial tests and safe rollout.
  - [x] Specify factual written summary and billing/status semantics.

## Phase 3: Implementation
### Subphase 3.1: Atomic Runtime Repairs
- Commit: separate billing, cache, runtime, numerical, resource and telemetry commits.
- Tests: package A-E focused suites, isolated timeout fault drill and full suite.
- Success Criteria: no ambiguous paid retries, shared prewarm dependencies, bounded isolated fits, analytic gradient parity, factual persisted summaries.
- Planned Touch Files:
  - `ima/openrouter_transport.py`
  - `ima/research_expansion.py`
  - `ima/research_preparation.py`
  - `ima/research_executor.py`
  - `ima/performance_probit.py`
  - `ima/research_resources.py`
  - `ima/research_scheduler.py`
  - `ima/research_worker_runtime.py`
  - `ima/research_runtime_history.py`
  - `ima/research_summary.py`
  - `ima/research_telemetry.py`
  - `scripts/optimize.py`
  - `scripts/run_v6_reliability_canary.py`
  - `pyproject.toml`
  - `requirements-research.lock`
  - `ima/research_specs.py`
  - `tests/test_openrouter_transport_classification.py`
  - `tests/test_probit_gradient.py`
  - `tests/test_performance_probit.py`
  - `tests/test_research_preparation.py`
  - `tests/test_research_executor_v6.py`
  - `tests/test_research_runtime_history.py`
  - `tests/test_research_resources.py`
  - `tests/test_research_summary.py`
  - `tests/test_research_telemetry.py`
  - `tests/test_research_worker_runtime.py`
  - `tests/test_research_expansion.py`
  - `tests/test_research_expansion_betting.py`
  - `tests/test_research_expansion_resources.py`
  - `tests/test_research_scheduler.py`
  - `tests/test_research_billing.py`
  - `tests/test_research_fold_dependencies.py`
  - `tests/test_research_column_projection.py`
  - `tests/test_v6_reliability_canary.py`
- Checklist:
  - [x] Implement and independently review packages A-E.
  - [x] Verify focused fault drills and full suite.

## Phase 4: Rollout
### Subphase 4.1: Verified Pinned Successor
- Commit: rollout evidence documentation after exact source qualification.
- Tests: Linux canaries, backend MLflow readback, planner receipt and bounded live observation.
- Success Criteria: successor completes scientific canaries, planner chooses from feedback, no unexplained stalls or OOM.
- Planned Touch Files:
  - `docs/V6_RELIABILITY_AND_TRACE_REPAIR_PLAN.md`
  - `deploy/systemd/ima-research-expansion-supervisor` (immutable package coverage)
  - `.github/workflows/research-expansion.yml` (matching package coverage)
- Operational sequencing: user chose to keep old fits running. Qualify a separate bounded canary under the imaopt account, leave the old unit running, and record the production handover as waiting until both fits complete. Do not claim that the successor is the active campaign before that handover.
- Checklist:
  - [x] Qualify immutable release and verify trace summaries.
  - [ ] Start successor and record healthy observed behavior or exact external blocker.

## Execution Evidence
- Final local source-freeze suite: 944 tests passed in 248.963 seconds; `.tmp/v6-reliability-final-full-suite.log`. Earlier failing integration runs are retained, not reclassified as successes.
- Real terminal canary assertions verify five completed fits, final champion metrics, zero active fits, and a cleared final invocation record. The bounded process/preparation startup allowance is 120 seconds, not the old 30-second process-pool assumption.
- Edge suite: 59 billing/scheduler/worker checks passed; state suite: 51 checks passed; offline three-model scientific canary: eight tests passed. Numerical gradient tests compare finite differences and normalized race likelihoods.
- Independent review led to additional fixes for incomplete billing coverage, recycled PID safety, surviving worker descendants, absolute queued deadlines, durable fold preparation recovery, pinned readiness, replay trace numbers and final summaries.
- Deadline cleanup deliberately refuses to signal unproven process identities. Short-lived children not recorded before an abrupt root death remain a residual limitation; the own-unit cgroup is the final containment boundary. No other-user or other-service mutation is permitted.
- Early column projection is limited to flat untransformed win recipes. Discovery, transformations, graphs and performance-distribution recipes retain their established input path; no claim of universal memory optimization.
- Own-server Pebble 5.2.2 was installed additively from a locally downloaded wheel over Tailscale. Dependency verification passes with the existing private overlay on `PYTHONPATH`; without that overlay, the base Jupyter environment reports missing HTTPX.
- Production handover is waiting on the user's explicit keep-waiting instruction. Old two probit fits and old unit remain untouched. Qualification is not a claim of live autonomous progress.
- Linux qualification of `e45cb21` passed all 16 bounded-worker identity, deadline, replacement and descendant-cleanup tests. All 260 extracted release files matched the committed archive; the own-user launcher preflight passed.
- The real accepted-HKJC scientific canary retained a failed receipt: Benter and boosted completed with one shared read-only fold artifact; heteroscedastic probit rejected insufficient numerical integration accuracy (order 512; normalization error 2.917e-6, refinement error 3.412e-5). This is a proven numerical qualification failure, not an OOM or worker deadline failure. Preserve this receipt and qualify a new source revision; do not hotpatch a pinned release or relax probability accuracy gates.
- Additional numerical repair uses [SciPy QUADPACK integration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html) as a deterministic fallback for sharp unequal-scale winner integrals, with explicit transition points, infinite tails, reported error estimates and a second tighter integration. Retain the fitted likelihood's independent coarse/fine rejection. The scientific canary increases its declared training quadrature from 32 to 128; this does not change production model defaults or reclassify failed trials.
- Written-summary qualification was delivered and read back from MLflow experiment 8, `ima-v6-reliability-qualification`, trace `tr-2ef07c68d8d4a6d779fdb27e50924193`. Chromium verified the table preview and summary output. This is explicitly a diagnostic replay of observed predecessor evidence: 87 completed, 15 failed, two running and five reserved; no paid call or successor fit. Its native cost remains absent rather than fabricated zero.
- Repair pull request: https://github.com/Kassden/iMa/pull/18, targeting the existing V6 expansion branch rather than silently merging unrelated expansion work into main. CI source qualification is separate from production handover.
- Final runtime source qualification is `0a50bf4aa61874b199074499f3e2235b893aac2e`; archive SHA256 `3f4e4c9c21af3ff7e5f640310ea14e64679fa0b54693441c6e9af511f346c9bb`. All 260 committed files matched after transfer. The prior failed `e45cb21` and `f6ec04f` canary receipts remain preserved.
- Adaptive-integration source `f6ec04f` passed the full local suite: 949 tests in 247.855 seconds. Final canary-only scale regularization change separately passed all eight canary tests in 6.850 seconds. The smaller 200-race scientific control uses quadrature 128 and scale regularization 10; production defaults, optimizer freedom and accuracy gates are unchanged.
- Final real accepted-HKJC canary `agentic_v6_reliability_science_0a50bf4` completed all three models: Benter 7.619 seconds, boosted 7.797 seconds, heteroscedastic Gaussian probit 63.697 seconds. Each fit hit the same validated read-only fold artifact; measured private peaks were approximately 2.20-2.32 GB. This is small-protocol reliability evidence, not new production champion or profit evidence.
- Prepared successor `agentic_v6_reliability_0a50bf4` preserves the accepted dataset, full research memo and three compatible historical champion references. Its separate own-user `ima-v6-reliability-0a50bf4.service` passes launcher preflight and unit verification and remains **inactive**: 26-fit ceiling, 24-thread CPU budget, 100,000,000,000-byte kernel RAM limit, no swap. No production planner call or training loop was started during qualification.
- Remaining acceptance: predecessor fits finish without cancellation; then production handover, actual new OpenRouter receipt/native USD readback, completed successor work and next planner feedback. Do not describe the inactive unit as continuously running or promise automatic handover: no handover timer has been installed.
