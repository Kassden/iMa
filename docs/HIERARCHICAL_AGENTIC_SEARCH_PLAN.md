# Hierarchical Agentic Search Repair

## GOAL
- A planner selects bounded research directions and Optuna tunes parameters within each direction, using comparable feedback and rejecting unusable features before training.

## Acceptance Criteria
- [x] Each accepted proposal becomes a durable program with recipe template, numeric space, target/objective identity, hypothesis, parent IDs, and trial budget.
- [x] Every trained program trial has real Optuna parameters; distinct target/program objectives never share a study.
- [x] The planner sees recipes, best comparable results, program outcomes, failures, and feature coverage; cycle two responds to cycle one.
- [x] Unavailable transform inputs are rejected before worker execution and are not repeatedly sampled.
- [x] A seeded CLI campaign proves two feedback cycles, resume, MLflow linkage, and no change to live v12.

## Research
- Built-in options: reuse `ResearchLedger`, `PipelineRecipe`, evaluator, campaign identity, MLflow outbox/traces, and CLI.
- Off-the-shelf options: Optuna `Study.ask/tell`, `Trial.suggest_*`, `JournalStorage`, `TPESampler`; Pydantic strict contracts; pandas profile.
- Official sources: [Optuna ask/tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html), [Study API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html), [JournalStorage](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/011_journal_storage.html), [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking).

## Root-Cause Baseline
- Trigger scope: live v12 research quality and the planner/Optuna feedback contract, with no server outage. The isolated controller was online with 26 workers on 2026-09-26.
- Evidence inventory: current controller/search/executor/provider code and remote v12 decisions, trials, Optuna journal. Snapshot: 455 complete, 148 failed; 253 completed planner recipes and 202 completed local Optuna recipes; all 279 planner study trials had zero `trial.params`; one minimizing study held winner, placing, and ranking objectives; 144 failures were `TransformError` on unavailable `horse_rating`.
- Proven causes: `reserve_recipe` stores planner recipes as user attrs only; global `ask` samples its own fixed axes; `_build_evidence` exports only last 32 IDs/hashes/headline scores; `tell` mixes target objectives; transform availability is checked after worker start.
- Likely: thin evidence and unrestricted refill repeat low-value directions. Possible: provider quality and feature timing issues; neither established by this audit. Disproven: a shared journal and `tell()` do not mean TPE learned planner parameters.
- Missing: fresh untouched holdout and full point-in-time feature availability. They limit predictive claims, not structural repair.
- Mutation boundary: dedicated local branch, fixture campaigns, then new isolated server release/campaign. Preserve v12, old journal, and shared services.
- Remediation mapping: parameterless trials -> program-scoped `suggest_*`; mixed objectives -> separate studies; weak evidence -> comparable bundle; transform failures -> admission profile; interruption risk -> resume test.
- Not-done: claim log-loss gains from wiring, relabel old trial history, change active campaign identity, or lose attempts on resume.

## SOTA, Standards, And Best Practices
- Prior art: hierarchical AutoML separates search-space design from numeric tuning; Optuna TPE requires actual parameters and comparable objective values.
- Official APIs: use `Study.ask()` plus `Trial.suggest_*` and `Study.tell()`, with JournalStorage under one controller; MLflow runs/traces retain lineage.
- Mature mechanisms: Pydantic contracts, pandas availability profiling, Optuna sampling, MLflow tracking.
- Constraints: freeze data/protocol/code identity; no raw labels, holdout feedback, freeform code, arbitrary paths, or evaluator overrides in planner output; preserve cost cap.
- Rejected: one study across targets, zero-param planner trials, freeform code generation, and restarting v12.
- Decision: proposals become durable programs with fixed structural recipe fields and bounded numeric model parameters. Optuna samples every numeric variant and programs expire at budget; controller summarizes outcomes before replanning.

## Dependency and Tooling Preflight
- Required: project `.venv/bin/python`, Optuna, Pydantic, pandas, unittest, fixture dataset, local SQLite MLflow, Tailscale SSH for isolated server canary.
- Found: Python 3.11 venv, declared research dependencies in `pyproject.toml`, existing end-to-end tests.
- Install or repair: use `uv sync --extra research` if the local venv lacks declared packages. The terminal workflow has no product frontend; browser smoke for the unchanged generated plan dashboard is limited to opening and checking its HTML when generated.
- Blockers: remote credentials/Tailscale outage, incompatible wheels, or inability to create an isolated server release.

## Deterministic Real-User Test
- Entry: `.venv/bin/python -m scripts.optimize run --campaign <temp> --config <fixture>`.
- Workflow: bootstrap; complete first batch; fixture planner creates a bounded program; run multiple Optuna samples; inspect comparative evidence; accept second program; stop/resume; inspect status/MLflow.
- Stable input: `tests/fixtures/research_races.csv`, fixed seeds, one-fold protocol.
- Assertions: program trials have nonempty `trial.params`, proper objective identity and parents; two feedback cycles, no duplicates, visible trace/run/model links; invalid transform rejected before execution.
- Command: `.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e tests.test_research_search tests.test_research_controller tests.test_agentic_planner`.
- Evidence: exit/status, decisions/program index/journal and MLflow readback.

## Fulfillment and Readback Proof
- Final user-visible result: executable hierarchical optimizer on this branch and a fresh isolated canary; v12 unaffected.
- Required write/mutation operation: commit code/tests/docs, run new local and server campaigns, and write MLflow runs, models, and traces; no shared-service restart.
- Readback surface: verify CLI status, program studies, decisions/evidence JSON, MLflow API records, and server sessions.
- Expected content, rows, fields, counts, or behavior: program budgets consumed by parameter-bearing Optuna samples, per-target comparisons, no repeated invalid transform failures, and second cycle cites first-cycle results.
- Command or tool call evidence: focused suite, CLI campaign, MLflow client queries, server session check, and git commit/push readback.
- Not-done conditions: fixture-only claim of live rollout, copied code without canary, parameterless planner trials, or missing readback.

## Armageddon Mode
- Large scope: invalid bounds, all-null transform, duplicate/stale proposal, exhausted budget, empty provider response, mixed targets, partial batch, repeated run.
- Failure modes: provider timeout/cap, worker failure, interrupted tell, MLflow outage, malformed journal, stale evidence.
- Optimization: avoid repeated data scans/fits, retain bounded concurrency and dedupe.
- Must fix: mixed study, zero-param planner trial, repeated invalid transform, lost resume linkage, or shared-service impact.
- Evidence: unit/e2e tests, campaign readback, server session check.

## Generality Guardrail
- Reuse: `ResearchLedger`, `PipelineRecipe`, `RecipeSearchController`, `ResearchProposal`, evaluator, and MLflow.
- Similar uses: winner, ranking, placing, speed, and odds targets all require scoped tuning. Recurrence: high.
- New general mechanism: durable research program plus program-scoped Optuna study and shared admissibility profile.
- Application: OpenRouter, fixture, and local bootstrap use the same validator/controller path. One-off: no custom live UI or target-specific optimizer fork.

## Ordered State and Dashboard
- State `.mega/hierarchical-state.jsonl`; evidence `.mega/hierarchical-evidence.jsonl`; generated read view `.mega/dashboards/HIERARCHICAL_AGENTIC_SEARCH_PLAN.html`. These plan-specific ledger paths keep earlier plans' phase numbers out of this dashboard.
- Record phase starts, tests, commits, local/server readbacks, and any blocker. Regenerate dashboard with `mega_plan_dashboard.py --no-open` and inspect output.

## Regression Guardrails
- Planned edit surface: `ima/research_specs.py`, `ima/research_search.py`, `ima/research_controller.py`, `ima/openrouter_orchestrator.py`, focused tests, `docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md`, this plan. MLflow linkage requires `ima/mlflow_tracking.py` because a program proposal ID spans multiple trial IDs; leave the executor unchanged.
- Protected: legacy optimizer policy, frozen evaluation folds/labels, MLflow loadability, crash recovery, v12 release/session, Cortex/solar services.
- Consumers: terminal optimizer, server worker, research tests, MLflow operator. Damage radius: large.
- Branch strategy: `feat/hierarchical-agentic-search`, created before implementation. Proof: contracts, seeded CLI, resume, relevant suite, isolated canary/readback.
- Atomic units/commits: program contract/search; controller/evidence/admission; end-to-end/docs/rollout. Push after proof.

## Phase 1: Baseline

### Subphase 1.1: Freeze evidence and design
- Commit: `docs(optimizer): plan hierarchical research search`.
- Tests: `mega_plan_check.py docs/HIERARCHICAL_AGENTIC_SEARCH_PLAN.md`; dashboard HTML readback.
- Success Criteria: observed causes, protected behaviors, architecture, and execution boundaries are explicit.
- Planned Touch Files:
  - `docs/HIERARCHICAL_AGENTIC_SEARCH_PLAN.md`
- Checklist:
  - [x] Record audited counts and official sources.
  - [x] Validate plan and dashboard.

## Phase 2: Search Contract

### Subphase 2.1: Program-scoped Optuna
- Commit: `feat(optimizer): tune within durable research programs`.
- Tests: `.venv/bin/python -m unittest tests.test_research_search tests.test_research_specs`.
- Success Criteria: a bounded planner program yields multiple parameter-bearing Optuna trials in one comparable study; budget, dedupe and resume work.
- Planned Touch Files:
  - `ima/research_specs.py`
  - `ima/research_search.py`
  - `tests/test_research_search.py`
  - `tests/test_research_specs.py`
- Checklist:
  - [x] Add strict numeric search-space and program contract.
  - [x] Replace mixed global `ask` with program studies and typed sampling.
  - [x] Verify study separation, budgets, and resume.

## Phase 3: Controller Feedback

### Subphase 3.1: Connect planner, evidence, and admission
- Commit: `feat(optimizer): direct Optuna with planner evidence`.
- Tests: `.venv/bin/python -m unittest tests.test_research_controller tests.test_agentic_planner`.
- Success Criteria: planner sees comparable recipes/results; proposals drive trials; invalid transforms are rejected before training; fallback uses approved programs.
- Planned Touch Files:
  - `ima/research_controller.py`
  - `ima/openrouter_orchestrator.py`
  - `ima/mlflow_tracking.py`
  - `tests/test_research_controller.py`
  - `tests/test_agentic_planner.py`
- Checklist:
  - [x] Persist decisions and program outcomes.
  - [x] Enrich evidence and validate feature coverage/lineage.
  - [x] Route every request through an active program study.

## Phase 4: Verification And Rollout

### Subphase 4.1: Real-user proof and isolated canary
- Commit: `test(optimizer): prove hierarchical feedback and resume`.
- Tests: `.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e`; relevant full suite; CLI run/stop/status; fresh server campaign if reachable.
- Success Criteria: two dependent cycles and target-specific parameter-bearing trials read back from CLI and MLflow; v12 online.
- Planned Touch Files:
  - `tests/test_agentic_optimizer_e2e.py`
  - `docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md`
  - `docs/HIERARCHICAL_AGENTIC_SEARCH_PLAN.md`
- Checklist:
  - [x] Run adversarial and resume cases; record outputs (225-test suite passed).
  - [x] Launch isolated canary; verify studies, MLflow, v12.
  - [x] Check scope/commit gate, push branch, final whole-plan gate.

## Execution Readback
- Local suite: 226 tests passed after the scheduler correction, including focused search/e2e coverage that requires newly approved programs to train.
- The first server canary exposed scheduler starvation: newly approved programs were registered but not sampled. This was corrected in `00e4c6a`; its before/after campaigns remain separate.
- `hierarchical-canary-v2` completed six trials in two cycles, with zero pending tells/tracking. Cycle one approved programs `14343bfec6b77563` and `f253273ce8592a41`; both were trained in that cycle. All six completed Optuna trials had nonempty parameter maps.
- MLflow stored six run/model links and two cycle traces; an API query found the new program's `trial_id` and proposal ID in run parameters. The fixture dataset lacks `horse_rating`, and the evidence marked it unavailable. The fixture scores are a workflow proof, not a predictive-accuracy estimate.
- The existing `ima-feedback-v2` tmux session and v12 processes remained online. No shared service was restarted; the new code was exercised only in `imaopt`-owned `canary-releases/hierarchical-v2` and a fresh campaign.
- Branch `feat/hierarchical-agentic-search` was pushed to `origin`; live v12 was deliberately not migrated in place.
