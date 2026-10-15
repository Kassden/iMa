---
schema_version: 1
decision_id: "feature-queue-controls-budget"
event_id: "devt-d1371d648f5a4ba6ac39201ab8a43aa0"
task_id: "feature-exploration-20261015"
plan: "/Users/milkingthesun/Projects/iMa/docs/CONTINUOUS_FEATURE_EXPLORATION_PLAN_20261015.md"
plan_sha256: "7f03910e12b3da52f127f02e8dfdd8918172fc15bb6282a604df79e5413c61db"
phase: "1"
subphase: "1.1"
kind: "architecture"
status: "accepted"
timestamp: "2026-10-14T23:44:17.691162+00:00"
durable: true
supersedes: null
evidence_refs: []
state_event_ids: []
---

# feature-queue-controls-budget: How can feature search progress without waiting for long main-model programs?

## Context and question

How can feature search progress without waiting for long main-model programs?

## Decision

Independent durable typed-feature queue using cached fixed-model controls and shared registry feedback, with an 8 GB feature and 92 GB model hard-memory partition

## Rationale

No queued feature supply exists and only two builds are published; dynamic main capabilities already read the shared registry

## Alternatives considered

- Wait for main planner requests only: proven sparse supply
- Run unrestricted standalone100GB feature worker: violates shared budget
- Restart main controller to add feature work: would delay behind preserved multi-hour fits

## Assumptions

- The shared immutable registry is dynamically read by the current main planner capabilities

## Self-critique

A baseline-v1 gain may not transfer to a rich pool; model settings and scored populations must match and exploration is not confirmation

## Disconfirming evidence

No evidence yet that generated formulas improve predictive accuracy; existing full-history feature discovery can be slow

## Cheapest disconfirming test

Real fixture plus at least three live paired feature tests and a new main planner evidence bundle must show consumption; combined cgroup maxima must not exceed100GB

## Consequences and tradeoffs

One feature worker receives2CPU and8GB; main fits retain26-job ceiling and92GB instead of100GB while the worker is active; no second LLM planner

## Reversibility

Stop own feature service and restore only its model-memory dropin to100GB; immutable feature artifacts and interruption records remain

## Follow-up triggers

Audit observed throughput and cross-model transfer, then revisit resource split using measurements
