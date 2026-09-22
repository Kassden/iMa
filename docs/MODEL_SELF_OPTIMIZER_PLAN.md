# Model Self Optimizer Loop

## GOAL
- Build a terminal-run, agent-assisted model optimization loop that proposes bounded experiments, executes local training/evaluation, reads metrics, and decides the next model, feature, transformation, calibration, or hyperparameter trial without automatic promotion.

## Out of Scope
- Race-day estimation UI or terminal workflow; keep `docs/WEDNESDAY_RACE_READINESS_PLAN.md` for later.
- Automated live betting or any wager execution.
- Automatic champion promotion.
- Unbounded AutoML searches.
- Letting an LLM directly run shell commands, edit files, delete data, or bypass validation gates.
- Using final odds, dividends, results, or future race data as pre-race features.

## Acceptance Criteria
- [ ] A dedicated terminal command exists conceptually as `ima-optimize run`, with dry-run and resume modes.
- [ ] The optimizer stores every campaign, trial, metric readback, and decision in append-only JSON/JSONL artifacts.
- [ ] Each trial has a single bounded hypothesis and one coherent changed surface: hyperparameters, feature family, transform, dataset window, model family, calibration, or market blend.
- [ ] A local deterministic controller, not the LLM, validates allowed actions, leakage rules, budgets, and promotion gates.
- [ ] An optional OpenRouter orchestrator can propose the next `ExperimentSpec` as structured JSON, but the local controller owns execution.
- [ ] The loop can run without OpenRouter in deterministic local-policy mode for testing and offline work.
- [ ] The loop uses OpenRouter `service_tier: "flex"` for lower-cost synchronous planner calls where latency can be tolerated, with explicit fallback behavior when flex capacity is unavailable.
- [ ] The optimizer can use OpenRouter Batch API for async multi-proposal calls that do not need immediate responses, then execute multiple independent local trials concurrently while preserving deterministic campaign ordering, isolated output directories, and append-only result readback.
- [ ] The first campaign can run with a tiny max-trial fixture or selected existing experiment specs and produce a campaign report.
- [ ] Heavy optimizer campaigns can run on the `cortex-server` host under a newly created isolated OS user and separate checkout, without touching existing users, repos, services, or deployments.
- [ ] Result evaluation compares challenger runs against champion, market baseline, and previous trial history using race log loss, top-1/top-3, calibration/Brier where available, incremental pseudo-R2, pool metrics, drawdown/EV simulation when available, and stability checks.
- [ ] No candidate becomes champion unless it passes gates and is marked `awaiting_operator_approval`.

## Research
- Built-in options:
  - Reuse `ima.experiments.ExperimentSpec`, `run_experiments`, default grids, `results_frame`, MLflow tracking, `ModelRegistry`, `scripts.run_experiments`, `scripts.run_benter_grid`, and feature study scripts.
  - Reuse existing chronological race splits and race-aware evaluation instead of inventing validation.
- Off-the-shelf options:
  - Use OpenRouter-compatible chat completions only as a structured proposal source.
  - Use OpenRouter `service_tier: "flex"` for discounted synchronous planner calls when available.
  - Use OpenRouter Batch API for asynchronous batches of independent proposal/review requests when a 24-hour completion window is acceptable.
  - Prefer JSON schema / tool-style outputs where supported; fall back to local deterministic policy if unavailable.
  - Keep scikit-learn, pandas, joblib, MLflow, and existing repo experiment primitives as execution substrate.
- Official / standards sources:
  - OpenRouter API reference documents `response_format` with JSON object / JSON schema outputs and tool-calling request fields.
  - OpenRouter tool-calling docs state the model suggests tool calls while client code executes the tool; this matches the safety boundary.
  - OpenRouter service-tier docs define `flex` as a discounted tier that trades latency and availability for lower price; flex routing is restricted to flex endpoints when present and does not fall back to default-tier endpoints.
  - OpenRouter Batch API docs describe asynchronous multi-request submission and retrieval with a 24-hour completion window across chat completions, Responses, Anthropic Messages, and embeddings shapes.
  - Existing repo docs and tests are the local standard for leakage, market baseline, and experiment reporting.

## SOTA, Standards, And Best Practices
- Current SOTA / prior art:
  - Model search should be champion/challenger, time-aware, and reproducible.
  - LLM orchestration is useful for proposing next experiments from metric summaries, but unsafe as an executor.
  - AutoML-style search is helpful for hyperparameters; horse-racing models additionally need feature timing, market baseline, and leakage controls.
- Official docs, standards, specs, or platform guidance:
  - OpenRouter structured output and tool-calling APIs support schema-oriented orchestration, but compatibility is model/provider-specific.
  - OpenRouter flex can be requested with `service_tier: "flex"`; for optimizer v1 prefer explicit service tier so cost behavior is visible in logs.
  - OpenRouter Batch API is for work that does not need immediate response; do not use it for interactive terminal decisions unless the command is explicitly running in async batch mode.
  - OpenRouter `openrouter/free` can route randomly among free models, so it is acceptable for plumbing tests but not for reproducible optimizer decisions.
- Mature libraries, frameworks, or built-in mechanisms:
  - Existing `ima.experiments`, `ima.modeling`, `ima.feature_sets`, `ima.mlflow_tracking`, `scripts.run_benter_grid`, and `scripts.run_experiments`.
- Best-practice constraints to integrate:
  - Local deterministic controller owns all mutation, execution, validation, and promotion decisions.
  - LLM outputs are schema-validated suggestions only.
  - Every trial is reproducible from saved config and command.
  - Every campaign has budget limits: max trials, timeout, max concurrent trials, max proposal batch size, max OpenRouter batch requests, max consecutive failures, and early-stop rules.
  - One trial changes one coherent surface so attribution is possible.
  - Parallel trials must be independent, write to isolated directories, and append results through a deterministic merge/readback step.
  - Batched model proposals are proposals only; each item must pass the same local schema, leakage, and duplicate-trial validation as a single proposal.
  - OpenRouter Batch API responses must be joined back to campaign request IDs before any local experiment executes.
  - Keep validation windows untouched during proposal generation.
- Rejected approaches:
  - Reject a pure LLM agent that can run commands or edit code directly.
  - Reject random free-model routing for reproducible optimizer decisions.
  - Reject selecting a winner by ROI alone.
  - Reject silent auto-promotion.
  - Reject broad feature construction without leakage classification.
- Implementation decision:
  - Build a terminal-first optimizer with local policy mode plus OpenRouter planner modes: synchronous flex planner calls for cheap immediate decisions, and async OpenRouter Batch API calls for multi-proposal work. Local experiment execution can run in bounded batches for independent trials.

## Dependency and Tooling Preflight
- Required verification tools:
  - Python 3.11, editable project install, unittest, existing experiment scripts, optional MLflow, optional OpenRouter API key.
  - For server campaigns: current Tailscale reachability, key-only SSH, `sudo` approval for user creation, Python build tooling, Git, disk space, and resource readback from `free -h`, `nproc`, `lscpu`, and `df -h`.
- Existing tooling found:
  - `ima/experiments.py` has `ExperimentSpec`, default experiment specs, `run_experiments`, result flattening, dashboard rendering, and MLflow logging.
  - `scripts/run_benter_grid.py` can run selected default specs against rich schemas.
  - `tests/test_experiments.py` covers experiment grids, dashboards, run history, feature studies, and publisher contracts.
- Install or repair commands:
  - `python3.11 -m venv .venv`
  - `.venv/bin/python -m pip install -e '.[dev]'`
- Browser/runtime binaries:
  - None required for the optimizer loop. No Playwright/browser smoke, screenshot, visual, pixel, canvas, or accessibility proof is required because the product surface is a terminal CLI, not a browser UI.
  - The generated Megaskill plan dashboard is only a read view; verify it by static HTML readback, not product browser QA.
- Real blockers that would prevent setup:
  - Missing Python environment, missing processed historical data for rich campaigns, missing OpenRouter API key for remote planner mode, OpenRouter model has no flex-capable endpoint when flex is required, OpenRouter Batch API unavailable, or too-expensive full-history runtime for large campaigns.
  - `cortex-server` Tailscale peer offline, SSH unavailable, no approval to create a new user, insufficient disk quota, or evidence that a planned command would touch existing users/repos/services.

## Server Execution Target
- Target host:
  - `cortex-server` over Tailscale, previously known at `100.95.24.121`; re-verify current peer address before use.
- Current observed state on 2026-09-22:
  - Local route to `100.95.24.121` goes through Tailscale interface `utun8`.
  - `tailscale status` reports `cortex-server` offline, last seen about 6 hours before the probe.
  - TCP/22 and SSH to `cortex@100.95.24.121` timed out.
  - Therefore no live server resource claim is current yet; the user-provided `128GB RAM and dual CPU` spec must be verified after the peer comes online.
- Isolation model:
  - Create a new OS user, proposed name `imaopt`, only after explicit approval.
  - Home/work root: `/home/imaopt/iMa`.
  - Artifact root: `/home/imaopt/iMa/artifacts/agentic-learning`.
  - Virtualenv owned by `imaopt`, not shared with existing service users.
  - No writes under `/srv/apps/cortex`, existing `cortex` user homes, production repos, systemd units, nginx configs, databases, or deployment directories.
- Access model:
  - Use key-only SSH.
  - Avoid password auth.
  - Prefer `sudo install`, `sudo useradd`, or equivalent only for the isolated user setup after approval.
  - After setup, run optimizer commands as `imaopt`.
- Resource policy if 128GB RAM and dual CPU are verified:
  - Default `--max-concurrent-trials 4` for full historical campaigns.
  - Allow `--max-concurrent-trials 6-8` only after one measured smoke campaign confirms CPU/RAM headroom.
  - Keep proposal batching independent from local training concurrency: OpenRouter can batch proposal calls; local training still obeys resource limits.
- Server acceptance:
  - Read back `whoami`, `pwd`, `hostname`, `free -h`, `nproc`, `df -h`, `git remote -v`, and `python --version` as `imaopt`.
  - Run a dry-run campaign as `imaopt`.
  - Run a one-trial smoke campaign as `imaopt`.
  - Confirm no files changed outside `/home/imaopt`.

## Deterministic Real-User Test
- Entry point:
  - `ima-optimize run` from the repository root.
- User workflow:
  - Run `ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 1 --policy local --dry-run`.
  - Run `ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 4 --proposal-batch-size 2 --max-concurrent-trials 2 --policy local --dry-run`.
  - In remote mode, use OpenRouter flex planner calls by default and use `--openrouter-batch` only when the operator accepts async proposal turnaround.
  - Run the same command without `--dry-run` on a tiny fixture or selected specs.
  - Read terminal summary and campaign artifacts.
  - Resume the same campaign and confirm it does not repeat completed trials.
  - For server mode, SSH to `cortex-server`, switch to the isolated optimizer user, and run the same terminal workflow from `/home/imaopt/iMa`.
- Stable inputs or fixtures:
  - Existing small unit-test frames where possible.
  - A selected known spec such as `boost-lr006-leaf15` for smoke mode.
  - Existing historical sample data for integration mode.
- User-observable assertions:
  - Terminal output names campaign, trial number, hypothesis, changed surface, command/config, metrics, decision, and next action.
  - Batch mode output names proposal batch size, OpenRouter batch ID when used, concurrent worker limit, queued trials, running trials, completed trials, and deterministic merge order.
  - `campaign.json`, `trials.jsonl`, `decisions.jsonl`, and `report.md` are written.
  - Dry-run writes or prints a valid proposed trial without training.
  - Resume reads existing state and advances deterministically.
- Command:
  - `.venv/bin/ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 1 --policy local --dry-run`
  - `.venv/bin/ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 4 --proposal-batch-size 2 --max-concurrent-trials 2 --policy local --dry-run`
  - `.venv/bin/python -m unittest tests.test_experiments tests.test_feature_analysis tests.test_mlflow_tracking -v`
  - `ssh -o BatchMode=yes cortex@100.95.24.121 'hostname; whoami; free -h; nproc; df -h'`
  - `ssh -o BatchMode=yes imaopt@100.95.24.121 'cd ~/iMa && .venv/bin/ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 1 --policy local --dry-run'`
- Evidence to record:
  - Megaskill evidence for command help, dry-run, one-trial smoke, resume, invalid config, and targeted tests.

## Fulfillment and Readback Proof
- Requested final user-visible result:
  - A terminal optimizer loop plus persisted campaign artifacts under `artifacts/agentic-learning/<campaign>/`.
- Required write/mutation operation:
  - Write campaign metadata, proposed specs, trial results, decisions, reports, optional models, optional MLflow runs.
- Readback surface:
  - Terminal output, `campaign.json`, `trials.jsonl`, `decisions.jsonl`, `best_candidate.json`, `report.md`, generated model artifact paths, and MLflow/exported results when enabled.
- Expected content, rows, fields, counts, or behavior:
  - Campaign records data cutoff, validation window, budget, policy, OpenRouter model, `service_tier` preference, async batch preference, allowed actions, metric gates, max proposal batch size, max concurrent trials, and champion baseline.
  - Trial records include `trial_id`, parent, OpenRouter request id or batch item id when applicable, local batch id, worker id if applicable, hypothesis, changed surface, config, command, start/end time, exit status, metrics, artifacts, and conclusion.
  - Decision records include continue/stop/promote-candidate/no-promotion, rationale, confidence, and next action.
  - `best_candidate.json` is absent or marked awaiting approval unless gates pass.
- Command or tool call evidence:
  - `ima-optimize run --dry-run`, one-trial local smoke, resume smoke, invalid-proposal rejection, targeted unit tests.
- Not-done conditions:
  - LLM can execute commands directly.
  - Trial config is not reproducible.
  - Metrics are not read back from artifacts.
  - Resume repeats completed work.
  - Parallel trials write into shared output paths.
  - Batch result merge is nondeterministic.
  - Promotion happens automatically.
  - Leakage gates are absent or advisory only.
  - Server campaign runs as an existing production user instead of the isolated optimizer user.
  - Server setup writes into existing `cortex-server` repos/services or shared production directories.

## Armageddon Mode
- Damage-scaled attack scope:
  - Moderate: this touches model training/evaluation and generated artifacts, but not live wagering or production deployment.
- Edge cases to try:
  - Missing OpenRouter key.
  - OpenRouter returns invalid JSON.
  - OpenRouter proposes forbidden action.
  - Local policy has no valid next move.
  - OpenRouter flex endpoint has no capacity and the optimizer must fall back or fail clearly depending on configuration.
  - OpenRouter model has no flex-capable endpoint.
  - OpenRouter Batch API returns partial success.
  - OpenRouter batch is still pending when the local timeout expires.
  - Max trials is zero or negative.
  - Proposal batch size exceeds max trials.
  - Concurrent worker count exceeds proposal count.
  - Campaign already complete.
  - Existing campaign has partial/corrupt JSONL.
  - Trial command fails.
  - Candidate improves top-1 but worsens log loss/calibration.
  - Candidate beats champion but not market.
  - `cortex-server` offline.
  - SSH works for existing user but not isolated optimizer user.
  - Resource readback does not match expected 128GB RAM / dual CPU class.
  - Optimizer checkout attempts to reuse an existing repo.
- Failure modes to simulate:
  - Wrong model family.
  - Unsupported feature schema.
  - Leakage-prone feature proposal.
  - Missing data file.
  - Timeout.
  - Repeated failing trial.
  - Duplicate trial ID.
  - Two parallel trials attempt the same run id.
  - One trial in a batch fails while others succeed.
  - One OpenRouter batch proposal is invalid while others are valid.
  - Remote disk fills during artifacts write.
  - Remote process is interrupted and campaign must resume from JSONL state.
- Optimization opportunities to inspect:
  - Whether selected existing grid specs are enough for v1 before adding arbitrary hyperparameter synthesis.
  - Whether local policy can cover 80% of useful iterations before spending OpenRouter calls.
  - Whether OpenRouter flex reduces planner cost enough without unacceptable latency.
  - Whether OpenRouter Batch API reduces proposal/review cost or rate-limit pressure without making the terminal loop too slow.
  - Whether local parallelism speeds campaigns without saturating CPU/RAM or making results harder to compare.
  - Whether `cortex-server` resources allow 4+ concurrent full-history trials safely after measured smoke results.
  - Whether trial summaries are concise enough for cheap orchestrator context.
- Must-fix threshold:
  - Any bypass of deterministic validation, reproducibility, leakage gates, or promotion approval blocks done.
- Evidence to record:
  - Invalid proposal rejection output, resume output, one-trial smoke artifacts, and targeted tests.

## Generality Guardrail
- Existing mechanism to reuse:
  - `ima.experiments.ExperimentSpec`, `run_experiments`, `results_frame`, `merge_run_history`, MLflow logging, `scripts.run_benter_grid`, feature schemas, and model registry.
- Similar existing use cases:
  - Static grids, rich-feature grids, feature studies, MLflow imports/exports, dashboard result histories.
- Recurrence likelihood:
  - High; every post-race or research cycle should use the same optimizer loop.
- General mechanism to create if none exists:
  - `ima.optimizer` for campaign/trial/decision contracts and local policy.
  - `scripts/optimize.py` for terminal command entry.
  - Optional `ima.openrouter_orchestrator` for schema-validated remote planner calls.
  - OpenRouter planner abstraction with synchronous flex calls, async Batch API calls, and strict local validation.
  - Remote runner profile for isolated `cortex-server` execution under the dedicated optimizer user.
- Specific use case implementation through the mechanism:
  - Initial campaign optimizes existing iMa model grids and feature schemas, then later expands to new transforms/features.
- One-off justification, if any:
  - None; this must be a reusable recurring optimizer.

## Ordered State and Dashboard
- State ledger path:
  - `.mega/state.jsonl`
- Dashboard output path:
  - `.mega/dashboards/MODEL_SELF_OPTIMIZER_PLAN.html`
- Open behavior:
  - Generate dashboard for plan review; race-day dashboard remains separate.
- State events to record:
  - Plan, contract, dry-run, batched dry-run, smoke execution, OpenRouter flex request, OpenRouter batch request, `cortex-server` reachability, isolated user setup approval, parallel trial execution, resume, invalid-proposal rejection, final gate.
- Evidence links:
  - `.mega/evidence.jsonl`
- Dashboard readback assertions:
  - Dashboard contains terminal command, OpenRouter flex/Batch API planner boundary, local controller safety boundary, phases, and acceptance criteria.
  - Product verification remains CLI/artifact based; no Playwright browser smoke is needed unless a future browser UI is explicitly added.

## Regression Guardrails
- Planned edit surface:
  - Plan phase: `docs/MODEL_SELF_OPTIMIZER_PLAN.md` and generated `.mega/dashboards/MODEL_SELF_OPTIMIZER_PLAN.html`.
  - Implementation phase: `ima/optimizer.py`, optional `ima/openrouter_orchestrator.py`, `scripts/optimize.py`, `pyproject.toml`, and focused tests.
- Protected behaviors:
  - Existing experiment scripts and dashboard exports keep working.
  - Existing chronological validation and leakage boundaries remain intact.
  - Race-day readiness plan stays unchanged except by explicit request.
  - No live wager execution is introduced.
  - No model promotion without operator approval.
- Likely consumers:
  - Research operator, post-race learning process, MLflow dashboard, future scheduled optimizer jobs.
- Damage radius:
  - Planning: small.
  - Optimizer v1: moderate because it adds a new orchestration layer around model training.
  - Automated promotion/live trading remains out of scope and systemic.
- Branch strategy:
  - Current branch is acceptable for plan-only work.
  - Use a dedicated work branch before implementation if the repo policy or user requests commits/pushes.
- Proof plan:
  - Plan check, generated-dashboard HTML readback, targeted tests, CLI help, dry-run, batched dry-run, mocked OpenRouter flex request, mocked OpenRouter batch request, `cortex-server` read-only reachability/resource probe, isolated-user readback, one-trial smoke, bounded parallel smoke, resume smoke, invalid-proposal rejection, artifact readback.

## Phase 1: Discovery

### Subphase 1.1: Map Existing Experiment Substrate
- Commit: `docs: add model self optimizer plan`
- Tests: `python3.11 /Users/milkingthesun/.codex/skills/megaskill/scripts/mega_plan_check.py docs/MODEL_SELF_OPTIMIZER_PLAN.md`
- Success Criteria: The plan names existing experiment primitives, scripts, tests, and the terminal optimizer target.
- Planned Touch Files:
  - `docs/MODEL_SELF_OPTIMIZER_PLAN.md`
  - `.mega/dashboards/MODEL_SELF_OPTIMIZER_PLAN.html`
- Checklist:
  - [ ] Inspect `ima/experiments.py`, `scripts/run_experiments.py`, `scripts/run_benter_grid.py`, and `tests/test_experiments.py`.
  - [ ] Keep `docs/WEDNESDAY_RACE_READINESS_PLAN.md` untouched for later.
  - [ ] Validate and regenerate optimizer plan dashboard.

### Subphase 1.2: Define Campaign Inputs and Metric Gates
- Commit: `feat: define optimizer campaign gates`
- Tests: `.venv/bin/python -m unittest tests.test_experiments tests.test_feature_analysis -v`
- Success Criteria: Campaign config names dataset, cutoff, validation window, champion baseline, market baseline, allowed actions, max trials, timeout, and metric gates.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
- Checklist:
  - [ ] Define campaign config dataclass/schema.
  - [ ] Define metric gate defaults.
  - [ ] Add OpenRouter `service_tier` preference fields and fallback policy.
  - [ ] Add OpenRouter async batch preference fields.
  - [ ] Add `proposal_batch_size` and `max_concurrent_trials`.
  - [ ] Add validation for invalid budgets, invalid concurrency, and missing baselines.

## Phase 2: Contract

### Subphase 2.1: Add Trial and Decision Schemas
- Commit: `feat: add optimizer trial contracts`
- Tests: `.venv/bin/python -m unittest tests.test_optimizer -v`
- Success Criteria: `ExperimentProposal`, `TrialResult`, and `AgentDecision` validate allowed actions, changed surfaces, reproducibility fields, and promotion states.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
- Checklist:
  - [ ] Define schemas as dataclasses or typed dictionaries with JSON serialization.
  - [ ] Validate one changed surface per trial.
  - [ ] Reject forbidden fields/actions.
  - [ ] Persist append-only `trials.jsonl` and `decisions.jsonl`.

### Subphase 2.2: Add Local Deterministic Policy
- Commit: `feat: add local optimizer policy`
- Tests: `.venv/bin/python -m unittest tests.test_optimizer -v`
- Success Criteria: The optimizer can propose a next trial without any network model and does not repeat completed trials.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
- Checklist:
  - [ ] Seed policy from existing default experiment specs.
  - [ ] Use prior metrics to choose continue/stop/try-next-grid action.
  - [ ] Support proposing a deterministic batch of independent next specs.
  - [ ] Support dry-run proposal output.
  - [ ] Support resume state.

## Phase 3: Orchestrator

### Subphase 3.1: Add Optional OpenRouter Planner Boundary
- Commit: `feat: add openrouter optimizer planner`
- Tests: `.venv/bin/python -m unittest tests.test_optimizer -v`
- Success Criteria: OpenRouter planner mode can be configured but is skipped/fails clearly without `OPENROUTER_API_KEY`; mocked responses validate strict schema handling, `service_tier: flex`, and async Batch API proposal handling.
- Planned Touch Files:
  - `ima/openrouter_orchestrator.py`
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
  - `.env.example`
- Checklist:
  - [ ] Read model from `IMA_OPTIMIZER_MODEL`, defaulting to a pinned OpenRouter model after live verification.
  - [ ] Read `OPENROUTER_API_KEY` from environment only.
  - [ ] Read flex preference from `IMA_OPTIMIZER_SERVICE_TIER=flex`.
  - [ ] Send `service_tier: "flex"` for sync planner calls when enabled.
  - [ ] Support `--openrouter-batch` for async multi-proposal requests.
  - [ ] Ask for structured JSON, not free-form prose.
  - [ ] Support requesting a batch of proposals.
  - [ ] Validate response against local schema.
  - [ ] Fall back to local policy when disabled.

### Subphase 3.2: Add Invalid Proposal Rejection
- Commit: `test: reject unsafe optimizer proposals`
- Tests: `.venv/bin/python -m unittest tests.test_optimizer -v`
- Success Criteria: Forbidden LLM proposals are rejected before any experiment command is generated.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
- Checklist:
  - [ ] Reject final odds/dividends/results as features.
  - [ ] Reject live execution or promotion actions.
  - [ ] Reject multiple changed surfaces.
  - [ ] Reject duplicate specs inside a proposal batch.
  - [ ] Reject OpenRouter batch responses that cannot be matched to campaign request IDs.
  - [ ] Reject unknown model families or unsupported schemas.

## Phase 4: Execution

### Subphase 4.1: Add Terminal CLI
- Commit: `feat: add terminal optimizer command`
- Tests: `.venv/bin/ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 1 --policy local --dry-run`
- Success Criteria: CLI prints current campaign state, next proposed trial or proposal batch, and dry-run decision; writes no model artifacts in dry-run.
- Planned Touch Files:
  - `scripts/optimize.py`
  - `pyproject.toml`
  - `tests/test_optimizer.py`
- Checklist:
  - [ ] Add `ima-optimize` console script.
  - [ ] Add `run` subcommand.
  - [ ] Add `--campaign`, `--policy`, `--max-trials`, `--timeout-minutes`, `--proposal-batch-size`, `--max-concurrent-trials`, `--service-tier flex`, `--openrouter-batch`, `--dry-run`, and resume defaults.
  - [ ] Print compact terminal output.

### Subphase 4.2: Prepare Isolated Cortex-Server Runner
- Commit: `ops: prepare isolated cortex-server optimizer runner`
- Tests: `ssh -o BatchMode=yes imaopt@100.95.24.121 'cd ~/iMa && pwd && whoami && python3 --version && df -h . && free -h && nproc'`
- Success Criteria: `cortex-server` has an isolated optimizer user, separate checkout, verified resources, and no writes outside `/home/imaopt`.
- Planned Touch Files:
  - `docs/MODEL_SELF_OPTIMIZER_PLAN.md`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Planned Remote Touch Surface:
  - `/home/imaopt/`
  - `/home/imaopt/iMa/`
  - `/home/imaopt/iMa/.venv/`
  - `/home/imaopt/iMa/artifacts/agentic-learning/`
- Explicitly Protected Remote Surfaces:
  - Existing users' home directories.
  - Existing repos on `cortex-server`.
  - `/srv/apps/cortex` and any production app checkout.
  - Existing systemd units, nginx configs, databases, queues, deployment secrets, and running services.
- Checklist:
  - [ ] Re-verify Tailscale peer state, route, TCP/22, and key-only SSH.
  - [ ] Request explicit approval before creating the `imaopt` OS user.
  - [ ] Create or verify `/home/imaopt` and install only optimizer-local dependencies there.
  - [ ] Clone/copy the iMa repo into `/home/imaopt/iMa` without reusing existing repos.
  - [ ] Read back `whoami`, `pwd`, `free -h`, `nproc`, `df -h`, `git status`, and `python --version`.
  - [ ] Run optimizer dry-run as `imaopt`.
  - [ ] Confirm no writes outside `/home/imaopt`.

### Subphase 4.3: Execute Bounded Trial Batch
- Commit: `feat: execute optimizer trial batch smoke`
- Tests: `.venv/bin/ima-optimize run --campaign artifacts/agentic-learning/smoke --max-trials 2 --proposal-batch-size 2 --max-concurrent-trials 2 --policy local`
- Success Criteria: A bounded batch of independent trials runs through existing experiment machinery or selected spec wrappers, writes isolated trial artifacts, and reads metrics back in deterministic order. On `cortex-server`, this runs only as `imaopt` under `/home/imaopt/iMa`.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `scripts/optimize.py`
  - `tests/test_optimizer.py`
  - `artifacts/agentic-learning/smoke/`
- Checklist:
  - [ ] Generate reproducible command/config per trial.
  - [ ] Allocate isolated output directory per trial.
  - [ ] Run selected trials with bounded concurrency.
  - [ ] Read resulting metrics.
  - [ ] Append trial results and decisions in deterministic order.
  - [ ] Print terminal summary.

## Phase 5: Evaluation

### Subphase 5.1: Campaign Report and Best Candidate Readback
- Commit: `feat: summarize optimizer campaign`
- Tests: `.venv/bin/python -m unittest tests.test_optimizer tests.test_experiments -v`
- Success Criteria: Campaign report identifies best candidate, no-promotion reason, or awaiting-approval status with metric deltas.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
  - `artifacts/agentic-learning/smoke/report.md`
- Checklist:
  - [ ] Compare champion, market, prior trials, and challenger.
  - [ ] Penalize unstable or mixed-metric improvements.
  - [ ] Write `report.md`.
  - [ ] Write `best_candidate.json` only when gates pass.

### Subphase 5.2: Armageddon and Final Gate
- Commit: `test: harden optimizer loop`
- Tests: `.venv/bin/python -m unittest tests.test_optimizer tests.test_experiments tests.test_mlflow_tracking -v`
- Success Criteria: Invalid config, invalid LLM proposal, resume, failed trial, and no-promotion paths are covered; final plan gate passes.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `tests/test_optimizer.py`
  - `.mega/state.jsonl`
  - `.mega/evidence.jsonl`
- Checklist:
  - [ ] Run invalid proposal tests.
  - [ ] Run invalid OpenRouter flex/fallback tests.
  - [ ] Run mocked OpenRouter Batch API partial-result tests.
  - [ ] Run batch duplicate/isolation tests.
  - [ ] Run resume tests.
  - [ ] Run targeted experiment tests.
  - [ ] Record evidence.
  - [ ] Run Megaskill final gate when implementation is in scope.
