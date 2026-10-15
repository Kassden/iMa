# Billing Repair Validation

## Result
Billing-only repair deployed and existing V7 planning/training resumed. This does
not implement the other phases of the broader campaign reliability plan.

Code commit: `242d049660a5db2054e9c64112858af6cf4647bf`, branch
`fix/planner-billing-finalization`. Only the billing helper and its tests entered
that commit; all unrelated working-tree changes remain preserved.

## Root Cause And Corrective Contract
Request `gen-1791533012-HQAazBCAXL8eg3izQuHG` reached response headers, then exceeded
the existing 300-second planner deadline. Authoritative metadata identifies the
matching final upstream provider response HTTP499, finite generation duration,
native input/output tokens 76866/13374 and total_cost USD0.00932526. Completion
finish reasons were null and cancelled=false. Old recovery rejected this terminal
error despite its billed usage. New recovery recognizes a strictly matching
terminal upstream error with duration/native counters; billing_only=true prevents
this from becoming an accepted inference result or successful planner decision.

[Official generation metadata API](https://openrouter.ai/docs/api/api-reference/generations/get-generation).
Unknown costs, identity mismatch, unfinished provider status and conflicting fallback
responses remain fail-closed. No charge guessed, waived, or reset; no old request replayed.

## Tests
Command: `.venv/bin/python -m pytest tests/test_openrouter_billing.py
tests/test_research_billing.py tests/test_openrouter_transport.py
tests/test_openrouter_transport_classification.py -q`.

Result: **75 passed, 112 subtests passed in 11.45 seconds**. Tests include malformed
types/extreme duration, ambiguous final provider identity, HTTP200/102 without a
finish reason, missing charge/counters, duplicate GET recovery, original-record
preservation, tokens, spend-known transition, symlinks and in-flight generation exclusion.
Plan checker and scoped commit checker passed; `git diff --cached --check` passed.

## Deployment
Own user only: imaopt. New immutable billing recovery subset:
`/home/imaopt/research-v2/live-releases/ima-billing-242d049660a5db2054e9c64112858af6cf4647bf`.
Contains committed billing helper, package initializer, reconciliation CLI and existing
launcher, plus revision/manifest/checkpoint. Not a scientific model release.

Existing `ima-planner-billing-reconcile.service` now uses this release; result success,
ExecMainStatus0. No controller restart, scientific hotpatch or other-user/service change.
Previous configuration preserved at
`/home/imaopt/.config/imaopt/planner-billing.env.before-242d0496`.
Rollback restores that configuration if necessary; never delete recovered authentic receipts.

Recovered one receipt `planner-calls/D000198-01.json`: billing_only=true,
billing_source=openrouter_generation_api, billing_finalization_basis=provider_terminal_error,
cost0.00932526, input76866/output13374/total90240 tokens, exact generation identity.
Original SHA256 readback still matches:

- Decision D198: `b730c9a7b72b30fdaa4c47abc000b3c57ec1077c37fc89441c0e6415e5fc3a5b`.
- Transport D198-01: `35f995d509b2ed4bcf79661bcf213b6f318bbcc89a41e8ff6f19accd91b9d9e2`.
- Deployed helper: `8eb7835ba23be833cd0410fad18ba9ef29202c5d99acc7d5111831d4e7747a9a`.

## Live Verification
Readback **2026-10-10 02:56:15 UTC / 11:56:15 JST / 10:56:15 HKT**:

- Controller active, unchanged PID1705906; status training; planner error null.
- Decision D000199 accepted; native tokens input115538/output21299/total136837;
  cost USD0.0602202, reported. Response completed in about119 seconds.
- Total reported planner spend USD9.858684383; spend_unknown=false; unresolved IDs empty.
- Ledger:676 completed,34 failed,1 running; pending tells/tracking0.
- Running attempt `attempt-b71d11782d13e978c7fdee37fd74870f0a6032deedc273858df7d430d0729f94`,
  program0886c780c034a3aa, Benter conditional logit, fold003 of3, elapsed184 seconds.
- Own unit memory9,679,306,752 bytes (9.68 GB). No memory/cap settings changed.

Verified outcome is new accepted planning and actual training, not just active service.
No new predictive gain is claimed. Future independent planner timeout/validation/model
failures remain possible; this repair specifically removes the observed billing deadlock.

Final refresh at **02:59:40 UTC / 11:59:40 JST**: completed advanced to677, failed34,
one further Benter attempt running on fold3/3. Spend USD9.912816503 remains fully
accounted with zero unresolved IDs. D000200 was rejected with `Proposal has stale
evidence` (cost USD0.05413212); this is a separate proposal-validation problem already
covered by the broader reliability plan, not recurrence of the billing freeze or a
training failure. Do not claim the entire campaign is error-free. Repair branch pushed
to origin; not merged. Controller/model code and the original D198 history still unchanged.
