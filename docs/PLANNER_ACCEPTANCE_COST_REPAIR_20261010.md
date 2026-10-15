# Planner Acceptance And Cost Repair

## GOAL
Stop repeated paid proposal rejections, bind request-owned identity deterministically,
bound growing planner context and restore accepted planning and actual training.

## Acceptance Criteria
- [x] Empty per-proposal evidence IDs are bound only when the decision exactly matches the current request.
- [x] Explicit stale identities/unknown parents remain rejected; raw provider response and charge preserved.
- [x] Large historical paper payloads cannot grow requests without bound; critical schemas, memo, champions, feature safety and valid parents remain.
- [x] Repeated same structural rejection at the same result watermark opens a persistent paid-call circuit; new outcomes permit reconsideration.
- [x] Native MLflow tokens/USD and actual accepted decisions, fits and results verified live.
- [x] At least two live decisions accepted after repair; no claim that unrelated scientific failures are eliminated.

## Root-Cause Baseline
Evidence inventory: Oct10 live status/journal, D199-D203 decisions, planner responses,
exact evidence IDs and native MLflow trace metadata. 678 completed,34 failed, no
active fits; D200-D203 rejected. Top-level decision evidence IDs are correct but
programs[].evidence_ids empty. Dispatcher validates exact per-program identity only
after a paid response. Four calls cost USD0.22552578. Billing is fully accounted.
Proven: optional/default envelope IDs are being delegated to the model, and no
deterministic binder fills omission. The latest raw evidence contains444013 bytes
paper history; compression doesn't adequately bound nested paper metrics/pools.
Likely: repeated historical data increases input/native cost and obscures instructions.
Possible: other semantic proposal errors emerge after linkage is repaired; validate
them, do not blindly authorize recipes. Disproven: every ERROR trace is an API transport
failure or billing recovery failed again. Missing evidence: no guarantee about later
model output quality; local replay and live successive decisions are required.
Mutation boundary: own STOP drain flag set with fresh assertion of zero active fits
and no planner in flight. Preserve all records; only scoped new planner-policy module,
tests and additive immutable own-user launcher allowed. No model/data/scoring edits.
Remediation mapping: request binder repairs empty ID defect; payload budget addresses
history growth; circuit addresses repeated paid post-validation failures.

Oct11 live qualification found a second proven defect in D205: matched-control
comparison treats recipe.dataset_ref=null (current configured dataset) differently
from that exact registered dataset ID. Remedy: canonicalize dataset defaults only
in copies used by the proposal audit, after verifying configured dataset bytes and
protocol parameters against the registry. Do not alter actual execution recipes or
raw responses; explicit different dataset IDs must still trigger an axis violation.

D207 revealed a legitimate undeclared transform change rejected after the planner
returned. Move provable fixed-control axis validation inside the existing two-call
schema-correction boundary, so errors repair the existing candidate rather than
start another whole research decision. Keep authority unchanged: do not silently add
axes or restore transforms. Also preflight the proven unsupported fold-local graph
CatBoost classifier path (one failed fit) and explain that capability to the planner.

## Research
Inspect existing choose_research_decision, curated context, authoritative apply_decision,
discipline admission and deployed immutable policy. Test exactly the failing outputs
without purchasing new completions. Native transport/cost observers remain unchanged.

## SOTA, Standards, And Best Practices
Official [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
are endpoint-dependent; do not require an unverified provider capability as an emergency
fix. Existing [Pydantic model validation](https://docs.pydantic.dev/latest/concepts/models/)
enforces semantic output. Implementation decision: bind controller-owned omitted IDs,
validate explicit values, bound request materialization and reuse existing orchestrator.
Mature JSON/Pydantic/httpx/systemd/pytest mechanisms, not a new agent loop.
Reject silently overwriting explicit stale IDs, relaxing parent/leakage validation,
counting unknown charges as zero, hiding historical errors or adding finite spend limits.

## Dependency and Tooling Preflight
No install is needed: existing local/server venv and dependency directories suffice.
Install or repair only scoped missing verification tooling if necessary; real blockers
are SSH auth, provider outage, missing credentials or a denied destructive operation.
Generated dashboard content readback and browser smoke when opening are required;
no new application interface. Official version support checked before strict-schema API use.

## Regression Guardrails
- Damage radius: large, paid planner boundary and own launcher.
- Branch strategy: dedicated branch `fix/planner-evidence-and-context`; preserve all prior dirty work.
- Planned Touch Files: `ima/planner_request_guard.py`, `tests/test_planner_request_guard.py`, `deploy/systemd/ima-v7-planner-request-guard.py`, focused policy test and this plan/validation memo.
- Protected: model science, immutable scientific release, chronological evaluation, feature availability, all valid parent identities, planner trial budgets, resource caps, existing memo/champions, native costs/tokens, other users/services.
- Consumers: OpenRouter decision planner, authoritative admission, existing controller, MLflow, deployment wrapper.
- Proof: pure fixtures, adversarial identity/circuit/budget tests, actual frozen evidence replay, isolated server self-test, consecutive live accepted decisions and training results.
- Atomic units: envelope/circuit, context budget, adapter/deployment and live readback.
- Commit plan: independent code/tests and immutable additive adapter; separate verified documentation.

## Phase 1: Contracts And Cost Controls
### Subphase 1.1: Request Identity And Retry Circuit
- Objective: normalize omitted envelope fields without inventing evidence or hypotheses.
- Allowed surface: new guard module and focused tests; no edits to dirty shared modules.
- Implementation: original planner runs with same scientific validation; before returning
  validate decision ID and evidence ID, bind absent/empty proposal IDs to current evidence,
  reject any explicitly conflicting value and unknown parents. Append binding receipt;
  preserve raw_response and paid usage. Track request-policy receipts per decision.
  Canonicalize verified-equivalent current dataset references in audit copies only;
  record the equivalence proof and preserve all execution recipes.
  Two matching post-policy semantic rejections with unchanged terminal watermark stop
  further paid calls; no circuit for live in-flight responses or independent training failures.
- Commit: `fix(planner): bind request identity and guard repeated rejected calls`.
- Tests: missing/empty IDs, explicit wrong IDs, correct IDs, top-level mismatch, unknown
  parents, malformed proposal, immutable raw outputs, persistent circuit/restart, new results.
- Success Criteria: historical empty-ID defects repaired locally; explicit stale proposals
  never authorized; circuit blocks before an API call.
- Checklist:
  - [x] Identity binding is deterministic and audited.
  - [x] Repeated failure protection survives restart.

### Subphase 1.2: Bounded Curated Context
- Objective: prevent historical growth from inflating each decision.
- Allowed surface: guard module and tests; use existing projection as base.
- Implementation: summarize nested paper history/metrics with labeled omissions and
  durable artifact references; drop zero remaining capacities and execution artifact paths;
  preserve all current champions/recipes, memo, registered feature metadata and valid parents.
  Keep paper taint/coverage/assumptions and current request limits explicit. Hard byte cap
  before sending, not an unverifiable tokenizer claim; mandatory context over budget fails
  locally with a precise reason, never truncated into unsafe data. Stable prefix remains.
- Commit: `perf(planner): bound historical context and outbound request size`.
- Tests: hundreds of growing histories, deep metrics arrays, protected content parity,
  no input mutation, idempotent projection, oversized essential safety material, static prefix.
- Success Criteria: request bounded regardless of historical pool search size; actual
  target-host fixture yields measured smaller request; native live input/cost read back.
- Checklist:
  - [x] All protected decision information retained.
  - [x] Byte budget and omitted data explicit in projection receipts.

## Phase 2: Safe Integration And Qualification
### Subphase 2.1: Immutable Additive Policy And Replay
- Objective: wrap existing verified policy, not overwrite its manifest or scientific code.
- Allowed surface: own new deployment adapter, self-test, config drop-in, validation memo.
- Implementation: verify original launcher hash and original module identity; install
  guard after original policy installation. Wrap existing context and planner, retain
  original transport observers. Atomic own unit drop-in points to new pinned adapter.
  Existing STOP preserved until self-tests pass. Archive owned STOP rather than deleting
  original evidence; no current fits cancelled. Capture old unit/drop-in for rollback.
- Commit: `feat(deploy): install pinned planner request guard`.
- Tests: current sealed policy canaries, frozen D200-D203 identity replay, bounded actual
  request projection, startup metadata and source hashes, import parity.
- Success Criteria: current scientific release unchanged and archived failed records
  unmodified; startup advertises exact new planner-policy hash and budget.
- Checklist:
  - [x] Own-user scope and manifests verified.
  - [x] Additive adapter self-test passes before paid planning.

### Subphase 2.2: Consecutive Live Decisions And MLflow
- Objective: prove ongoing acceptance, not one successful call followed by silence.
- Allowed surface: own read-only collectors, evidence/validation memo; refine policy
  only when a newly proven blocker requires it and rerun relevant tests.
- Commit: `docs(planner): verify bounded accepted planning and resumed training`.
- Tests: two accepted post-policy decisions, valid proposal IDs and action receipts,
  real worker progress/terminal result/next dispatch, native MLflow cost/tokens,
  billing reconciliation timer still healthy and truthful prior ERROR traces retained.
- Success Criteria: no repetition of empty-ID rejection, live context reduced, all
  charges accounted, multiple accepted decisions and actual training/results visible.
- Checklist:
  - [x] Live acceptance and feedback verified across decisions.
  - [x] Cost/token evidence and historical preservation read back.

## Deterministic Real-User Test
Entry point: guarded existing research decision planner with recorded valid top-level
identity, empty proposal IDs, mock paid response and actual current evidence. Expected
observable assertions: binding receipt, matching normalized IDs, one paid call, unchanged
raw_response, normal semantic admission and Optuna authorization. Wrong explicit IDs
fail; repeated wrong output opens circuit before another call. Deployment uses same path.

## Armageddon Mode
Edge cases: forged explicit IDs, no completed parents, unknown parent, null/type errors,
duplicate proposals, zero-trial review, live schema repair, delayed response, restarts,
oversized mandatory metadata, long paper studies, secret-bearing exception, malformed
state receipts, stale status, active fits during handover, MLflow delivery outage.
Must-fix: unsafe admission, hidden bill, stripped feature safety, deleted history,
unbounded paid repeated semantic failures, or success claimed before real training.

## Generality Guardrail
Reuse existing mechanism; generic request-owned identity binding and context limits,
not hard-coded decision IDs or special-case model names. No new orchestrator service.

## Fulfillment and Readback Proof
- Requested final user-visible result: restored accepted planning/training with reduced context and protected spending.
- Required write/mutation operation: scoped code/tests, pinned own adapter/drop-in, durable binding/circuit receipts; preserve predecessor history.
- Readback surface: verify actual unit, accepted decisions, normalized/raw IDs, hashes, native MLflow traces and worker results.
- Expected content, rows, fields, counts, or behavior: at least two accepted decisions, current evidence binding, bounded request, accounted USD/tokens and fit progress.
- Not-done conditions: unit active but rejected proposals, one accepted call with no training, smaller fixture only, unsafely bypassed validation or unknown spend hidden.
