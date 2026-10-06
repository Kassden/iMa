# V7 Planner Billing Recovery

## GOAL
Recover authoritative OpenRouter charges after interrupted requests automatically, resume the existing campaign without a source hotpatch, and distinguish active V7 research from legacy draining telemetry.

## Acceptance Criteria
- [ ] D9 charge retrieved by its recorded generation ID, never guessed or zeroed.
- [ ] Missing billing receipts are reconciled idempotently by a bounded periodic job.
- [ ] Wrong identities, missing IDs, nonfinite charges, existing receipts and transient provider errors fail closed.
- [ ] V7 accepts a new planner decision and completes additional model work after recovery.
- [ ] Active-campaign MLflow view excludes legacy V6 heartbeat traces without deleting history or stopping preserved fits.
- [ ] Code/tests pushed and merged, pinned own-user timer deployed and read back.

## Root-Cause Baseline
- Proven: V7 D9 received HTTP 200 and generation ID `gen-1791269393-76ySmlR9x6UIxpUXhgri`; its response body was cancelled at the 300-second absolute deadline.
- Proven: no full response/cost receipt persisted, so `_planner_spend` correctly froze further paid calls. No automatic generation lookup existed to resolve that state.
- Proven: OpenRouter generation GET reports cancelled=true and total_cost USD0.00851584. An additive billing-only physical receipt resolves campaign cost to USD0.155923212 without modifying the failed decision.
- Proven: repeated 87 completed/15 failed/5 reserved/2 running snapshots belong to legacy `agentic_v6_openrouter_continuous_4864156`, which is draining while preserving two fits. V7 reached 67 completed/two failed before the freeze.
- Likely: generation latency/output length exceeded the configured deadline; this is not proof of credit exhaustion or geographic outage.
- Missing evidence: full lost proposal output and provider-side cancellation reason beyond reported metadata.
- Disproven: no V7 work ever happened; a healthy service alone meant continued planner progress; the repeated legacy counts describe V7.
- Generic hardening: separate immutable billing recovery receipts and campaign-filtered views make recovery and history reviewable.
- Mutation boundary: additive authoritative receipts only; no guessed charges, original-response replacement, inference retries inside billing recovery, source hotpatch, old-fit cancellation or Cortex/Solar changes.

## SOTA, Standards, And Best Practices
- OpenRouter generation metadata API supplies exact cost and request identity: https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation.
- Reuse existing transport generation capture and physical-receipt accounting; systemd user oneshot/timer supplies bounded, observable periodic execution.
- Existing HTTPX and standard-library atomic file publication; no new dependencies, bespoke task scheduler or billing estimator.
- Rejected: treating unknown costs as zero, replaying inference to recover billing, overwriting response history, disabling the spend guard, or restarting a training campaign merely to deploy recovery.

## Research
Use provider generation metadata, existing HTTPX and systemd timers. Recovery queries billing metadata, not a model or estimated pricing table.

## Generality Guardrail
Existing mechanism: physical receipt accounting for every campaign. Recurrence: interrupted provider requests can recur. General mechanism decision: add one missing-receipt lookup/publication owner, with explicit refusal to replace already observed responses.

## Dependency and Tooling Preflight
- Existing SSH/Tailscale, own-user credentials, HTTPX, Python, pytest, git/gh and MLflow.
- Install or repair commands: none needed; use the existing own-user environment and route.
- Browser/runtime binaries: existing Chromium/Playwright for saved-view readback.

## Regression Guardrails
- Damage radius: moderate billing/admission effect; dedicated `fix/v7-planner-billing-recovery` branch from merged main.
- Branch strategy: dedicated repair branch, preserving unrelated dirty files.
- Protected: honest unknown-cost freezes when lookup is unavailable; exact identity/transport coverage; historical failed decisions; model/data identities; preserved legacy fits and other services.
- Consumers: existing `_planner_spend`, planner retry loop, physical receipt history and MLflow.
- Planned touch files: `ima/openrouter_billing.py`, `scripts/reconcile_planner_billing.py`, `tests/test_openrouter_billing.py`, `deploy/systemd/ima-planner-billing-reconcile.service`, `deploy/systemd/ima-planner-billing-reconcile.timer`, `deploy/systemd/ima-planner-billing-reconcile`, this document.
- Atomic units: recovery implementation/tests; deploy templates; verification documentation.
- Generality decision: one reusable missing-receipt reconciler for any campaign, using existing receipt format; no experiment-specific exception in accounting.

## Deterministic Real-User Test
- Entry point: `scripts/reconcile_planner_billing.py --campaign PATH`; stable fixture is the saved D9 transport and provider generation identity.
- Replay recorded D9 transport and authoritative generation metadata in a temp campaign; existing `_planner_spend` must change unknown to exact known cost without changing its failed decision.
- Live: same generation GET through existing server route, additive receipt, next accepted decision, new completed model and fresh current-campaign trace.

## Fulfillment and Readback Proof
- Required write/mutation: immutable billing-only receipt, pushed source, pinned own-user recovery timer, campaign-filtered saved view.
- Readback: provider ID/cost, native campaign accounting, timer invocation/result, accepted new planner decision, FINISHED run/READY model and browser view.
- Not done: service-active-only, receipt from another generation, fabricated cost, or merely hiding frozen campaign traces.

## Armageddon Mode
- Attack scope and edge cases: billing evidence publication and path ownership, not model/data mutation.
- Must-fix threshold: any identity mismatch, original-file replacement, falsely known charge or inference request.
- Attack wrong ID, multiple IDs, path traversal/symlinks, NaN/negative/bool charge, concurrent publication, existing successful/unpriced receipt, 404/429/timeout, duplicate runs and interrupted publication.
- Must fail closed and preserve original files; no extra inference or other-user writes.

## Phase 1: Recover And Automate
### Subphase 1.1: Authoritative Receipt Recovery
- Commit: fix(billing): reconcile missing OpenRouter generation receipts.
- Tests: independent deterministic reconciliation, identity/accounting/idempotency/error cases and full regression.
- Success Criteria: existing campaign accounting accepts exact provider receipts; no fabricated response/proposal.
- Checklist:
  - [ ] Confirm incident identity and exact provider charge.
  - [ ] Implement and test bounded reconciliation.

## Phase 2: Deployment
### Subphase 2.1: Pinned Own-User Timer
- Commit: feat(ops): run bounded planner billing reconciliation periodically.
- Tests: wrapper/unit syntax, own-root/revision checks, no-secret preflight and timer/service readback.
- Success Criteria: minute-scale recovery runs independently of live training source; no existing campaign restart.
- Checklist:
  - [ ] Push/merge green source and stage immutable recovery release.
  - [ ] Enable own-user timer and verify its receipt/result.

## Phase 3: Scientific Continuation And Visibility
### Subphase 3.1: Real Recovery Proof
- Commit: docs(ops): record V7 planner recovery proof.
- Tests: accepted post-freeze decision, additional completed work, MLflow accounting and active-campaign browser view.
- Success Criteria: new model work actually resumes and old heartbeat traces are recognizably excluded.
- Checklist:
  - [ ] Verify automatic continuation and new completed models.
  - [ ] Verify active view and final runtime readback.
