# Planner Preflight Recovery

## GOAL
Prevent growing context from stranding V7, distinguish locally unsent failures from uncertain paid requests, reject unsupported residual graphs and resume own-user research.

## Acceptance Criteria
- D239 context replays within a bounded budget with explicit omissions and all safety/parent identities retained.
- A local failure before the first HTTP call emits durable pre-dispatch evidence; a failure after any call remains potentially billed.
- Adjusted-speed residual graphs are rejected inside the existing bounded planner correction loop, before training.
- Original records stay intact, genuine unknown spending stays frozen, and a fresh accepted live decision plus execution proves recovery.

## Research
- Existing owners: JSON structured projection, request guard, HTTPX transport receipts, controller billing reconciliation, typed recipe validation and systemd own-user releases.
- Primary references: HTTPX tracing documentation https://www.python-httpx.org/advanced/extensions/ ; existing transport state machine and qualification fixtures are the source of local contracts.

## Root-Cause Baseline
- Minimum evidence inventory: live status metrics/readbacks, D239 decision/evidence files, transport/call history, config/state and local dispatch/validator code.
- Evidence: D239 failed at 304084>300000 bytes before dispatch, empty calls/transport; 747 completed/39 failed, no active jobs. D238 residual arm failed because a graph needs a nested residual-fit contract.
- Proven: the previous projection removed a fixed number of narratives, not enough to handle subsequent growth; local preflight failures never entered transport classification; semantic preflight omitted a known unsupported residual graph.
- Likely: duplicative evidence and capability diagnostics account for recurring growth. Quantify fields before reducing them.
- Possible: provider failures also occur but are not the cause of D239.
- Disproven: D239 incurred a demonstrated provider charge or training is still running.
- Missing evidence: no new preflight receipt exists for D239; qualify that historical local failure against inspected sealed hashes, never infer from absent receipts alone.
- Mutation boundary: branch fix/planner-context-predispatch; clean guard/transport/lifecycle/deploy modules and owned tests only; preserve dirty user files, active fits, raw provider history and other users/services.
- Remediation mapping: adaptive optional projection fixes growth; call-boundary receipts fix misclassified local failures; residual preflight fixes unsupported execution; sealed D239 qualification is a one-off historical recovery only.
- Not done: larger arbitrary cap only, unknown-charge blanket waiver, lost safety data, or active unit without fresh accepted work.

## SOTA, Standards, And Best Practices
- Reuse mature JSON/HTTPX tracing and existing classifiers, not inferred zero-cost events from exception names.
- Keep soft projection target below hard request boundary; truncate only explicitly optional history, preserve full durable evidence and label omissions.
- Reject silent removal of parent IDs, availability/taint/contracts, or changing scientific recipes during projection.
- Implementation decision: instrument actual choose/post boundaries, isolate only validated pre-dispatch event pairs, and retain provider cost uncertainty after dispatch.

## Dependency and Tooling Preflight
- Required tools: .venv pytest, SSH own-user server environment, git and megaskill scripts. Existing tools available; no install required.
- Browser/runtime binaries: no UI change; MLflow API and stored trace readback are the verification surface.
- Real blockers: Tailscale approval/outage, unavailable credentials, unsafe running-fit boundary.

## Deterministic Real-User Test
- Entry: guarded planner call with saved D239 evidence and mocked network.
- Assert bounded self-contained context, zero HTTP calls on local failure, true unsent receipt, and no unsent classification following a paid correction attempt.
- Command: .venv/bin/python -m pytest tests/test_planner_request_guard.py tests/test_openrouter_transport.py tests/test_research_lifecycle.py -q
- Evidence: saved context bytes, dispatch receipts, real replay/canary and fresh live decision.

## Fulfillment and Readback Proof
- Requested result: own V7 resumes planning and executing valid experiments.
- Mutation: deploy sealed own-user adapter at an idle/safe boundary, preserve original failures and predecessor release.
- Readback surface: server status, new decision, transport/usage, MLflow trace and trial progress.
- Not done if only code/tests or service active; require actual new accepted work or explicit external blocker.

## Armageddon Mode
- Damage radius: large. Attack cap+1/cap+4096, huge mandatory safety, repeated projection, post-first-call validation failure, forged/truncated events, sent sibling, stale historical hashes, unsupported residual graph and corrected valid recipe.
- Must-fix: scientific leakage, lost identities, misclassified paid request, unbounded retries, or other-user mutation.

## Generality Guardrail
- High recurrence: general bounded projection and pre-dispatch transport contract replace per-decision workarounds for future failures.
- Existing mechanisms: request guard, transport classifier, lifecycle qualification and sealed deploy adapter. D239 recovery alone is explicitly historical and hash bound.

## Regression Guardrails
- Branch strategy: dedicated fix/planner-context-predispatch for large shared-contract risk, created before edits.
- Dedicated branch fix/planner-context-predispatch. Protect 4-hour probit policy, warm starts, feature lineage, free planner budgets, 100GB ceiling, native cost/tokens and original scientific data.
- Atomic units: plan, guarded projection/receipts, classifier, historical recovery, deploy and validation.
- Scope checker may see preexisting dirty/untracked work; stage owned paths only and disclose that baseline.
- Deployment scope expansion: an own-user, single-use systemd handover service makes the graceful wait survive SSH/client disconnection. It must not terminate the productive fit or mutate scientific records.

## Phase 1: Evidence
### Subphase 1.1: Lock incident contracts
- Commit: docs(planner): lock preflight recovery plan
- Tests: live D239 and context-size readback, plan checker.
- Success Criteria: root causes and boundaries explicit.
- Planned Touch Files:
  - `docs/PLANNER_PREFLIGHT_RECOVERY_PLAN_20261013.md`
- Checklist:
  - [x] Inspect failure, counts and existing owners.
  - [x] Create dedicated branch and acceptance criteria.

## Phase 2: Repair
### Subphase 2.1: Projection, preflight and historical recovery
- Commit: fix(planner): bound optional context and record genuine local preflight
- Tests: guard/transport/lifecycle regression and adversarial tests.
- Success Criteria: safe projection and billing classification; invalid graph rejected before execution.
- Planned Touch Files:
  - `ima/planner_request_guard.py`
  - `ima/openrouter_transport.py`
  - `ima/research_lifecycle.py`
  - `tests/test_planner_request_guard.py`
  - `tests/test_openrouter_transport.py`
  - `tests/test_research_lifecycle.py`
- Checklist:
  - [x] Implement adaptive projection and actual dispatch-boundary receipts.
  - [x] Validate residual graphs and D239 recovery with preservation tests.

## Phase 3: Qualification
### Subphase 3.1: Real context and failure replay
- Commit: test(planner): qualify recurrent preflight recovery
- Tests: targeted suites, exact server evidence and no-paid-call canaries.
- Success Criteria: cap compliance and billing safety verified through actual consumers.
- Planned Touch Files:
  - `docs/PLANNER_PREFLIGHT_RECOVERY_VALIDATION_20261013.md`
- Checklist:
  - [x] Complete adversarial and runtime tests.
  - [x] Read back saved context, unchanged identities and billing evidence.

## Phase 4: Deployment
### Subphase 4.1: Resume own V7
- Commit: feat(deploy): activate qualified planner preflight recovery
- Tests: sealed adapter, graceful idle handover, fresh decision/trial/MLflow readback.
- Success Criteria: campaign running valid work without unknown local charges.
- Planned Touch Files:
  - `deploy/systemd/ima-v7-probit-feature-lifecycle.py`
  - `deploy/systemd/ima-preflight-handover-20261013.service`
  - `docs/PLANNER_PREFLIGHT_RECOVERY_VALIDATION_20261013.md`
- Checklist:
  - [x] Push/stage/qualify owned release.
  - [x] Activate and verify fresh end-to-end behavior.
