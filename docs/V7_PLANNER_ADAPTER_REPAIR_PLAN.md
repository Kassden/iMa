# V7 Planner Adapter Repair

## GOAL
Repair adapter guidance and retries, deploy a pinned V7 successor, and quantify matched market-blend evidence.

## Acceptance Criteria
- [ ] Advertised adapter recipes validate and execute chronologically.
- [ ] Retries retain rejected JSON and precise errors; budgets, identities and two-call limits stay enforced.
- [ ] Tests and CI pass, source is pushed/merged, and successor accepts a live decision.
- [ ] Market outlook separates predictive lift, uncertainty and executable betting edge.

## Research
Reuse existing typed graphs, sklearn residual models and CatBoost uncertainty. No new dependencies.

## Root-Cause Baseline
Evidence inventory: D14 response and errors, campaign ledger, pinned config/state and prompt source. Remediation mapping: examples address missing compatibility guidance; retained assistant JSON addresses retry regeneration.
Proven: prompt has only a probability-pooling example; D14 failed on an extra root key, then regenerated incompatible bare regressors targeting win. Retry omits previous assistant JSON. Audited V6/V7 ledgers contain zero explicit distribution-adapter graph recipes.
Likely: missing examples and lost candidate context cause avoidable regeneration. Possible: token pressure or planner preferences discourage adapters. Not proven: adapters outperform champions. Disproven: native probit is absent; a homoscedastic run completed. Missing evidence: live effect on selections. Mutation boundary: own imaopt campaign/release/unit only, preserve predecessor code and legacy fits. Map examples to discoverability, retained JSON/error paths to retry drift. Not done with source-only repair.

## SOTA, Standards, And Best Practices
Use Pydantic structured error locations/types/messages without input dumps: [official documentation](https://docs.pydantic.dev/latest/errors/errors/). Reuse chronological calibration and registered graphs. Reject silent partial admission, additional paid retries, in-sample blending and hotpatching pinned code. Financial conclusions need actionable timestamped quotes and unseen evaluation.

## Dependency and Tooling Preflight
Install or repair commands: no install is needed because dependencies already exist. Browser smoke: not applicable; no UI changes.
Existing .venv, unittest, GitHub CLI, SSH/Tailscale and server dependency environment suffice. No installs or browser changes. External inference outage blocks live-decision proof, not validation.

## Deterministic Real-User Test
Entry: choose_research_decision. Replay extra-field rejection, repair same programs, execute each example on seeded historical speed fixtures and predict later complete races. Assert finite normalized probabilities, serialization, paid usage and budgets. Command: `.venv/bin/python -m unittest tests.test_research_expansion.PlannerAPITests tests.test_pipeline_graph`.

## Fulfillment and Readback Proof
Requested final result: repaired live planner and honest market report. Required write/mutation: committed source and new own-user release/unit. Readback surface: decision JSON, native MLflow receipt and report. Expected behavior: accepted decisions retain constraints. Evidence: tests and live receipts. Not-done conditions: staged-only or stale results.
Require merged commit, immutable archive/revision, own-unit config, accepted planner decision and MLflow receipt. Canaries do not prove planner selection. Market report identifies matching populations, run IDs, calibrated-market controls and coefficients. Active service alone is insufficient.

## Armageddon Mode
Edge cases: malformed JSON and incompatible models. Failure modes: validation failure and remote timeout. Optimization opportunities: avoid regenerating valid programs. Must-fix threshold: any lost constraints or unaccounted paid call.
Moderate damage radius. Test extra fields, malformed JSON, absent choices, unsupported model/target, budget mismatch, limits and chronology. Preserve two-call cap, cost accumulation, leakage checks and whole-decision admission. Full CI; never kill preserved fits for deployment speed.

## Generality Guardrail
Existing mechanism: choose_research_decision and typed graphs. Recurrence likelihood: high. General mechanism decision: reusable example owner, not a new framework.
Extend shared research decision boundary; one lightweight owner supplies reusable tested examples. Operational staging scripts produce immutable-release receipts, not a new framework.

## Regression Guardrails
- Branch strategy: dedicated fix/v7-planner-adapter-guidance branch because live planning is affected.
Branch: fix/v7-planner-adapter-guidance. Surface: examples, planner prompt/retry, focused tests and docs. Protected: 80/20 dispatch, independent budget/concurrency, two paid attempts, dataset/protocol identities, old V6 fits, Cortex/Solar. Consumers: expansion controller, OpenRouter planner and MLflow. Proof: tests, CI, remote canary, live accepted decision.

## Phase 1: Repair
### Subphase 1.1: Guidance And Minimal Retry
- Commit: fix(planner): explain adapters and preserve rejected candidates
- Tests: planner API, graph suites and real-fit examples.
- Success Criteria: executable examples; original JSON and actionable feedback retained without bypass.
- Planned Touch Files:
  - `ima/research_planner_examples.py`
  - `ima/research_specs.py`
  - `ima/openrouter_orchestrator.py`
  - `tests/test_research_expansion.py`
  - `tests/test_pipeline_graph.py`
- Checklist:
  - [ ] Implement and verify prompt/examples/retry.
  - [ ] Review scoped diff and commit.

## Phase 2: Qualification
### Subphase 2.1: Market Audit And CI
- Commit: docs(research): record repair and matched market outlook
- Tests: full CI, read-only matched prediction audit, paired meeting bootstrap.
- Success Criteria: no scoring-set fitted blends or profit overclaims.
- Planned Touch Files:
  - `docs/V7_PLANNER_ADAPTER_REPAIR_PLAN.md`
  - `docs/V7_MARKET_BLEND_OUTLOOK_2026-10-06.md`
- Checklist:
  - [ ] Finish independent market analysis and review.
  - [ ] Push, pass CI and merge.

## Phase 3: Deployment
### Subphase 3.1: Immutable V7 Successor
- Commit: operational evidence, no source hotpatch
- Tests: archive, supervisor preflight, remote adapter canary, resource counts, live trace/decision.
- Success Criteria: predecessor drains, legacy fits untouched, successor accepts a decision with guidance, billing timer follows.
- Planned Touch Files:
  - `docs/V7_PLANNER_ADAPTER_REPAIR_PLAN.md`
- Checklist:
  - [ ] Qualify and stage before draining.
  - [ ] Start, monitor and record live proof.
