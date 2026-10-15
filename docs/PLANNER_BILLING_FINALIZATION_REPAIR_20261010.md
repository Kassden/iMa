# Planner Billing Finalization Repair

## GOAL
Recover the actual charge for aborted OpenRouter streams and resume the existing
V7 campaign without restarting its controller or changing scientific code.

## Acceptance Criteria
- [x] Matching terminal-error generation cost is recoverable without finish_reason.
- [x] Missing cost, ambiguous provider identity and in-flight generations stay unknown.
- [x] D000198 has one billing-only receipt; old decisions/transport remain unchanged.
- [x] Existing controller records a new decision and dispatches training.

## Root-Cause Baseline
Evidence inventory: fresh status readbacks, failed recovery-unit journal, transport
receipt and two matching authoritative GET responses. Observed Oct 10: own controller awaits decision at 676 completed/34 failed/no active
fits. GET generation metadata returned HTTP 200, total_cost USD 0.00932526,
native input/output tokens 76866/13374, generation_time 298295, upstream identity
matching final provider response HTTP 499; finish_reason and native_finish_reason
null, cancelled false. Recovery rejects this terminal provider-error record because
it requires cancelled=true or a nonempty completion finish reason. Proven cause:
too-narrow billing finalization validation. This is not evidence of exhausted credits.
Missing evidence: no completed planner output for D198; metadata recovery cannot
create an accepted proposal. Further planner success must be verified live.
Mutation boundary: local scoped billing validation/tests, isolated own billing release
and service configuration; append one authentic receipt. No old records overwritten.
Remediation mapping: recognize matched terminal provider error with timing and native
usage; keep ambiguous/in-flight records rejected; preserve known/unknown accounting.

## Research
Use existing GET-only reconciliation, atomic receipt publication, transport identity,
native usage and provider metadata. Do not replay the old paid inference request.

## SOTA, Standards, And Best Practices
Official [OpenRouter generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
is authoritative for total_cost and native usage. Mature httpx/systemd/pytest tooling
already exists. Implementation decision: require matching final upstream response
HTTP 400-599 plus finite duration and native counters for the no-finish-reason path.
Rejected approaches: accept all nonnull costs, mark unknown zero, reset spend, change
budgets, overwrite the failed decision, or disable billing guards.

## Dependency and Tooling Preflight
No install is needed: existing local venv and server dependency site-packages contain
httpx/pytest dependencies. Set server PYTHONPATH explicitly; bare server venv cannot
import httpx. Real blockers: SSH auth, credentials, provider outage, denied mutation.

## Regression Guardrails
- Damage radius: moderate, isolated recovery helper plus own-user deployment.
- Branch strategy: dedicated branch `fix/planner-billing-finalization`; all existing dirty changes preserved.
- Planned touch files: `ima/openrouter_billing.py`, `tests/test_openrouter_billing.py`, this plan and validation memo; isolated billing service environment only.
- Protected behaviors: all model math/data/protocols, controller policy, immutable decisions, generation identity, zero/unknown distinction, other users and services.
- Consumers: reconciliation CLI, existing timer, controller spend reader and MLflow billing evidence.
- Proof plan: targeted/adjacent tests, GET-only idempotent recovery, hashes and fresh live decision/training.
- Commit plan: scoped billing implementation/tests, followed by evidence documentation.

## Phase 1: Correct And Verify Billing Contract
### Subphase 1.1: Terminal Provider Error Recovery
- Objective: implement narrowly validated billing finalization, never inference success.
- Commit: `fix(billing): recover charges for terminated provider streams`.
- Tests: `.venv/bin/python -m pytest tests/test_openrouter_billing.py tests/test_research_billing.py tests/test_openrouter_transport.py tests/test_openrouter_transport_classification.py -q`.
- Success Criteria: native counts/cost preserved, wrong IDs/in-flight/missing charges rejected, duplicate reconciliation causes zero additional requests, old decisions unchanged.
- Checklist:
  - [x] Tests and adversarial cases pass.
  - [x] Only scoped code/test files staged.

## Phase 2: Isolated Deployment And Recovery
### Subphase 2.1: Deploy Billing-Only Release
- Objective: pin new helper in a fresh own billing release and reconcile once.
- Commit: `docs(billing): record pinned recovery and receipts`.
- Tests: server imports, own paths, deployed hashes, original decision/transport hashes, authoritative receipt value and spend-known readback.
- Success Criteria: own billing unit points to new immutable release; D198 recovered once; existing training/controller remains unchanged.
- Checklist:
  - [x] Previous billing configuration preserved for rollback.
  - [x] Provider receipt and accounting verified without completion replay.

### Subphase 2.2: Verify Automatic Campaign Progress
- Objective: prove new planning and actual dispatch, not merely active PID.
- Commit: `docs(billing): record resumed campaign verification`.
- Tests: fresh status/decision timestamps, accepted decision receipt, progress/running attempt, planner spend and tracking/tell counts.
- Success Criteria: decision number advances beyond198 and training starts under same controller; final report separates resumed operation from predictive gain.
- Checklist:
  - [x] New planner decision and actual fit verified.
  - [x] No controller restart, unrelated services changed or record deleted.

## Deterministic Real-User Test
Entry point: existing reconcile CLI. Fixture D198 has failed transport, matching
generation metadata with HTTP499, known charge, null finish reason. GET-only recovery
publishes receipt, spend reader becomes known, second invocation requests nothing.
Live workflow additionally requires automatic new planner decision and training.

## Armageddon Mode
Edge cases: ambiguous/mismatched last upstream ID, successful or provisional provider
status without finish reason, fallback still active, malformed types, nonfinite cost,
missing tokens/duration, duplicate receipt, symlink, stale transport, provider outage.
Must-fix: invented charge, false model success, overwritten history or bill counted twice.

## Generality Guardrail
Use existing mechanism for all terminated provider errors meeting the same evidence
contract; do not hard-code D198, provider names or generation IDs in production code.

## Fulfillment and Readback Proof
- Required write/mutation operation: update helper/tests; publish isolated own billing release and authentic receipt.
- Readback surface: assert hashes, receipt metadata, known spend, fresh status and new attempts.
- Expected content, rows, fields, counts, or behavior: one D198 billing-only record USD0.00932526 and 90240 native tokens; subsequent planning/dispatch.
- Not-done conditions: tests alone, restored credits alone, controller active but idle, unresolved charge or no new training.
