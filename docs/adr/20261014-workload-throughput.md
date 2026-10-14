---
schema_version: 1
decision_id: "qualified-workload-slots"
event_id: "devt-0d01ee093b7f4fd58ab70cbdde436467"
task_id: "throughput-20261014"
plan: "/Users/milkingthesun/Projects/iMa/docs/CAMPAIGN_THROUGHPUT_REPAIR_20261014.md"
plan_sha256: "e6c8f60f71dd6c1b601af9592b3d9ab943da1268bcebd6cb27662d1067e5d8ca"
phase: "1"
subphase: "1.1"
kind: "policy"
status: "accepted"
timestamp: "2026-10-14T14:30:09.655376+00:00"
durable: true
supersedes: null
evidence_refs: []
state_event_ids: []
---

# qualified-workload-slots: How should a qualified mixed-workload campaign resume capacity?

## Context and question

How should a qualified mixed-workload campaign resume capacity?

## Decision

Restore configured slots after five successful uncensored measurements while preserving all per-job resource guards

## Rationale

Global largest observed fit demands 120 GiB for a two-slot increase despite 5 GiB independent workloads

## Alternatives considered

- Keep global maximum ramp
- Force 26 fits or lower memory reservations
- Replace executor with Dask during incident

## Assumptions

- Original ResourceAdmission checks every submitted workload

## Self-critique

Historical successes qualify admission machinery not new recipe memory safety or scientific value

## Disconfirming evidence

No live OOM observed; independent estimates remain conservative and job supply may still be absent

## Cheapest disconfirming test

52 GiB active plus 6 GiB work admits; oversized CPU and cold probes must fail

## Consequences and tradeoffs

More simultaneous independent work; no estimator shrinkage; cold probes still bounded

## Reversibility

Restore prior sealed own-user dropin at safe boundary without cancelling fits

## Follow-up triggers

Verify actual concurrency and fresh completions after graceful handover
