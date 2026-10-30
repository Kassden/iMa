# Workload-Aware Throughput Repair

## GOAL
Restore useful concurrent experiments alongside slow fits without cancelling active work or weakening scientific admission.

## Acceptance Criteria
- [ ] A qualified resumed campaign admits independent small fits alongside a large fit under existing memory/CPU/cold guards.
- [ ] Duplicate pending proposals reach bounded paid-call correction before downstream rejection.
- [ ] Sealed deployment activates safely and live ledger shows concurrent work and subsequent completions.

## Runtime Contract
- Mode: execute.
- Task ID and session binding: throughput-20261014 / 01a0c46f-3ae6-79f3-b206-199aa3eff2f6.
- Proof artifacts: policy/tests, adapter manifest, canary and live readback receipts.
- Child cleanup: no delegation; critical-path work remains local.

## Decision Record Policy
Record qualified-resume capacity rationale, alternatives, assumptions, self-critique, disconfirming test and reversibility in .mega/decisions.jsonl. Promote to docs/adr/20261014-workload-throughput.md.

## Research
Inspect existing ResourceAdmission, ProgressiveCapacity, JobEstimator, bounded executor and planner correction. Preserve mature built-in mechanisms rather than changing the execution engine during recovery.

## Root-Cause Baseline
Evidence inventory: live status metrics, logs, config/state, ledger readbacks and job-estimates.
At 2026-10-14 14:18 UTC own campaign had 771 completions, 42 failures, one running fit, 8.1% CPU busy, 100 GB decimal cgroup cap and no OOM. Two funded programs were blocked by max_expensive_fits. Proven causes: the global largest measured private-memory peak requires 120 GiB to ramp 2->4; this holds cap two and expensive cap one. All graphs count as expensive. Recent Benter/boosted measurements require about 5 GiB and 165-270 seconds, while a probit fit took 26,519 seconds. Pending duplicate rejection occurs after paid planning without matching semantic preflight. Possible aging starvation is not proven live; do not change it in this repair. Missing evidence: future planner supply cannot be guaranteed and must be checked live. Mutation boundary: own imaopt unit and sealed additive policy only, preserve active fits and other users. Remediation mapping: qualified-resume slots address worst-fit global gate; per-job reservations remain exact. Duplicate preflight addresses wasted rejection calls. Generic estimate shrinkage is explicitly excluded. Not-done: static cap change or staged-only deployment.

## SOTA, Standards, And Best Practices
Official Dask task resource constraints and scheduler-state documentation support per-task admission: https://distributed.dask.org/en/latest/resources.html and https://distributed.dask.org/en/stable/scheduling-state.html. Implementation decision: reuse existing executor/admission, with an additive qualified-resume policy; no new dependencies. Rejected approaches: forcing 26 jobs, lowering measured reservations, cancelling fits, relaxing fixed-control checks or migrating engines. Restore configured slot ceiling only after five successful uncensored historical measurements; fresh campaigns keep progressive qualification.

## Dependency and Tooling Preflight
Existing tools: .venv Python unittest, git, SSH, own-user systemd and lifecycle self-test. No install is needed. Real blockers: unavailable credentials or external outage. A live fit awaiting graceful handover is a dependency, never authorization to cancel.

## Deterministic Real-User Test
Entry point: own controller ResourceAdmission and semantic_preflight. Stable fixtures: active 52 GiB fit plus independent 6 GiB work under 93.13 GiB budget, 8 GiB emergency reserve and 24 CPU threads. User-observable assertions: safe fit admitted; oversized/cold/CPU/disk probes rejected; duplicate pending decision corrected. Command: .venv/bin/python -m unittest tests.test_campaign_throughput_policy tests.test_planner_request_guard. Record output and assertions.

## Fulfillment and Readback Proof
Required deployment mutation: sealed own-user successor activates only after existing fits finish. Readback surface: status.json, ledger.sqlite, queue.jsonl and unit manifest. Assert policy version, more than one active fit, new completions, no OOM or untracked interruption. User-visible result is actual useful concurrent work, not a higher configured cap. Staged-only, pending activation, no new completions or stale receipts are not done.

## Armageddon Mode
Large scheduling damage radius. Adversarial edge cases: cold campaign, failed/censored measurements, source drift, duplicate within decision, memory exhaustion, CPU/disk constraints and cold-family limit. Optimization opportunities: global largest-fit multiplication and delayed duplicate detection. Must-fix: bypassed safety/scientific checks, manifest drift or hidden cancellation.

## Generality Guardrail
Reuse the existing mechanism: exact resource admission, typed two-call correction and sealed graceful handover. Recurrence is high in mixed expensive/cheap workloads. New general mechanism is a qualified-resume slot policy, not campaign-specific model tuning. No arbitrary trial budgets or forced catalogue.

## Regression Guardrails
- Planned edit surface: ima/campaign_throughput_policy.py, tests/test_campaign_throughput_policy.py, ima/planner_request_guard.py, tests/test_planner_request_guard.py, deploy/systemd/ima-v7-probit-feature-lifecycle.py, docs/CAMPAIGN_THROUGHPUT_REPAIR_20261014.md, docs/adr/20261014-workload-throughput.md; operational .tmp helper allowlist only.
- Protected behaviors: active fits, fit mathematics, immutable identities, 100 GB decimal memory cap, 26 ceiling, emergency and cold guards, bounded paid correction, cost accounting and all other users/services.
- Likely consumers: own controller, workers and paid planner.
- Damage radius: large.
- Branch strategy: dedicated branch fix/workload-aware-throughput; leave preexisting dirty work unstaged.
- Proof plan: deterministic replay, regression suite, sealed canary, preserved-fit handover, concurrent live fits and completions.

## Phase 1: Diagnosis
### Subphase 1.1: Seal Root Cause And Decision
- Commit: docs(research): record measured throughput root cause and guarded remedy.
- Tests: plan validation and live baseline assertions.
- Success Criteria: cause and uncertainty separated; active fit preservation explicit.
- Planned Touch Files:
  - `docs/CAMPAIGN_THROUGHPUT_REPAIR_20261014.md`
  - `docs/adr/20261014-workload-throughput.md`
- Checklist:
  - [ ] Preserve baseline and record alternatives.
  - [ ] Validate plan and register runtime contract.

## Phase 2: Repair
### Subphase 2.1: Qualified Capacity And Duplicate Correction
- Commit: fix(research): admit mixed workloads and repair duplicate planner decisions.
- Tests: policy, request guard and lifecycle regression suite; exact source cardinality fails closed.
- Success Criteria: independent small fits admit without memory/CPU bypass and duplicates reach bounded correction.
- Planned Touch Files:
  - `ima/campaign_throughput_policy.py`
  - `tests/test_campaign_throughput_policy.py`
  - `ima/planner_request_guard.py`
  - `tests/test_planner_request_guard.py`
  - `deploy/systemd/ima-v7-probit-feature-lifecycle.py`
- Checklist:
  - [ ] Implement additive policy and actionable preflight.
  - [ ] Prove adverse cases and exact runtime binding.

## Phase 3: Rollout
### Subphase 3.1: Safe Activation And Live Fulfillment
- Commit: chore(research): record sealed throughput rollout evidence.
- Tests: server lifecycle self-test and live own-user ledger/status assertions.
- Success Criteria: policy active, concurrent useful fits and fresh completed outcomes, original fit preserved to terminal state.
- Planned Touch Files:
  - `docs/CAMPAIGN_THROUGHPUT_REPAIR_20261014.md`
- Checklist:
  - [ ] Push scoped commit and stage hash-checked release.
  - [ ] Activate at safe boundary and verify actual concurrent completions.
