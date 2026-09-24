# Agentic Optimizer Feedback Repair

## GOAL
Deliver one terminal controller that proposes executable recipes, trains them, evaluates them, versions models and uses completed results to choose subsequent experiments autonomously on Cortex.

Status: PLANNED, NOT IMPLEMENTED. Date: 2026-09-25. This is the corrective execution sequence for AGENTIC_OPTIMIZER_V2_PLAN.md; its acceptance criteria remain binding. This request authorizes planning, not implementation or server changes. Race-day readiness stays deferred.

## Acceptance Criteria
- [ ] Non-dry agentic mode executes real recipe training, never silently returns preview or substitutes the legacy catalogue.
- [ ] Two consecutive provider planning cycles consume actual completed evidence; cycle two cites cycle-one trials and changes an executed recipe/search space. Counterfactual fixture evidence changes the next proposal.
- [ ] Six successful win trials cover two schemas, two families, an ablation, a transform and a training-window change. Multi-axis proposals include matched controls.
- [ ] Ranking, placing and adjusted finish-time/speed each have a real trained probe, target-specific baseline, shuffled-label control, leakage test and package readback. Odds forecasting is enabled only with valid timestamped snapshots; otherwise record unavailable with evidence.
- [ ] Selection uses protected rolling development scores; train/calibration/score differ. Holdout and legacy test scores never influence v2 selection.
- [ ] Resume preserves completed results and reconciles Optuna/MLflow without duplicate terminal records or model versions.
- [ ] Every successful candidate has a loadable fitted package, recipe, hypothesis, parents, data/protocol/code/environment hashes and MLflow run/version linkage.
- [ ] One detached imaopt controller completes the live canary, survives disconnection and controlled restart, then continues unlimited trials under explicit spend/resource caps.
- [ ] Status distinguishes provider planning, degraded local search, training, paused admission, pending tracking and stopped execution. Dry run does not consume real trials or paid API calls.
- [ ] Completion does not require improved log loss. Historical development gains are not claims of unseen predictive performance.

## Root-Cause Baseline
- Trigger scope: missing integration and overstated completion, not a server outage.
- Evidence inventory: optimizer.py:380 and :715-720; openrouter_orchestrator.py:248; research_search.py:107 and :221; experiments.py:334; test_agentic_optimizer_e2e.py:21; original plan acceptance/architecture. Read-only tmux/process/log/JSONL checks around 00:10 local time on 2026-09-25.
- Proven R1: agentic execution always returns dry_run. Recipes and actual training use disconnected campaigns.
- Proven R2: choose_research_proposals has no production caller; completed worker results never call RecipeSearchController.tell. No provider orchestrator runs in that proposal loop.
- Proven R3: actual training uses legacy splits and test-metric voting, not the new protected evaluator.
- Proven R4: fallback generated all six fixed recipes. Further requests raise exhaustion while its shell wrapper retries. The separate six-trial trainer had already finished and exited at audit time.
- Proven R5: E2E test performs two dry runs and reloads a separately constructed dummy package. Six targeted tests passed without proving feedback.
- Proven R6: ResearchLedger is not connected to execution. Resource admission is partial; package supports only predict_proba and lacks complete dataset/environment lineage.
- Likely hypothesis: component-level test success was mistaken for end-to-end fulfillment. This explains reporting, not predictive performance.
- Possible risks: rich-schema temporal leakage, target parse coverage, provider Flex support, MLflow artifact access and availability of odds snapshots. Inspect before claiming support.
- Disproven interpretation: two live processes establish feedback. Process count and liveness are insufficient.
- Missing evidence: live provider execution, clean holdout and registered v2 trained-package proof were unavailable because the audited path never executed them. Recheck server state at deployment; do not assume sessions, dependencies, model availability or MLflow linkage from this snapshot. This uncertainty blocks completion claims, not local integration work.
- Mutation boundary: documentation only now. Future changes use an isolated release, dedicated venv and new campaign under imaopt. Preserve source data and old campaigns; never restart Cortex, solar, nginx or postgres.
- Remediation mapping: R1 -> phases 1/2/3; R2 -> phase 3; R3 -> phase 2; R4 -> phases 1/3/5; R5 -> phase 5; R6 -> phases 3/4. Leases and resource monitoring are operational risk reduction, not evidence of predictive lift.
- Not-done conditions: disconnected loops, dry-run-only proof, fixed proposals called learning, ignored recipe fields, fabricated target support or completion records without joined artifacts.

## Research
Official sources rechecked 2026-09-25. Use repository-pinned versions; do not upgrade merely because latest documentation differs.
- [Optuna ask/tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html): use external workers and report objective values by trial number through the existing search API.
- [MLflow registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/): log fitted artifacts, register versions linked to runs and verify registered-URI loading.
- [OpenRouter service tiers](https://openrouter.ai/docs/guides/features/service-tiers): verify model/provider support and returned tier. Flex is an inference request setting, independent of training concurrency.
- Original plan covers autoresearch, RD-Agent and AIDE. Retain the evidence/experiment discipline; no additional agent framework is needed for this repair.

## SOTA, Standards, And Best Practices
Implementation decision: integrate existing mature libraries under one controller, with the official sources above as references and repository pins as the runtime baseline.

Reuse sklearn estimators/Pipeline, existing feature builders, Pydantic, Optuna JournalStorage, SQLite and MLflow. Use an established ranking library if no tested ranking adapter exists; never hand-write a ranking engine. One controller owns bookkeeping, ask/tell and planning; supervised workers execute validated recipes. Reject shell bridges between campaigns, custom numeric optimizers, silent legacy conversion and an LLM call per parameter value. Default to one synchronous Flex request per planning cycle; optional Batch inference must not break sequential feedback dependencies.

## Architecture And Contracts
One CLI starts one controller with supervised worker subprocesses. One tmux session is the deployment wrapper.

```text
frozen dataset/protocol -> evidence -> provider proposal -> validator
                             ^                              |
                             |                    approved space / Optuna ask
                             |                              |
                         durable result <- evaluation <- recipe worker
                             |
                         Optuna tell + model package + MLflow outbox
```

- Owner: new ima/research_controller.py dispatched by optimizer.run_campaign for agentic policy only. New ima/research_executor.py provides request/result adapters. Preserve legacy local/openrouter behavior.
- Request: attempt/proposal/study/trial IDs, full recipe, immutable dataset/protocol manifests, code/environment hashes, seed, timeout and owned artifact directory.
- Result: terminal status, target/fold metrics, paired baseline deltas, prediction/package paths and hashes, runtime, peak process-tree RSS and structured error. Verify identity/artifact completeness before accepting success.
- Campaign files: campaign.json, dataset/protocol manifests, ledger.sqlite, search journals, evidence/cycle-N.json, planner/cycle-N request/response/validation, trials/attempt-ID artifacts, exported trials.jsonl/decisions.jsonl/report.md/status.json.
- SQLite owns execution truth, Optuna owns sampling, MLflow owns tracking, JSONL is export. Completed legacy trials remain readable but do not enter new-protocol selection.

## Execution Decisions
1. Freeze dataset fingerprint and whole-race scoring population before planning. Cache immutable source/features by data/schema/history version. Training-window selection changes estimator rows, never evaluation rows; reject empty windows.
2. For each expanding fold use train < calibration < score. Fit preprocessing/model on train, temperature/market calibration/blend on calibration, then score. Persist per-race predictions and paired uncertainty against uniform, raw market, recalibrated market and frozen incumbent on identical rows.
3. Target adapters own labels, baseline, compatible estimator and metrics. Place probabilities do not sum to one across runners. Ranking outputs are scores until calibrated; finish-time outputs retain documented units/normalization. Fit normalization on training data; realized pace is not a pre-race feature. Package non-probability targets with their proper predict contract.
4. Add a strict versioned bounded search-space contract alongside ResearchProposal. Agent chooses executable target/schema/families/transforms/window/model and approved numeric bounds; Optuna samples numbers. Registry exposes only implemented/tested capabilities. Never silently drop transforms or ablations through legacy conversion.
5. Separate studies/objectives by target, data, protocol and search-space version. Primary incumbent minimizes development log-loss delta versus recalibrated market, tie-breaking worst-fold delta then runtime. Voting is diagnostic; never compare time MAE with win log loss.
6. Seed diverse controls, then plan every 32 terminal trials or queue exhaustion. Canary replans after three successes, so six successes prove two cycles. Queue <= twice active workers. Reserve 25% of later trials for underexplored valid axes. Evidence includes recent 32 results, best 10, representative failures, coverage, running work and budgets. Allowlist development fields before serialization. Validate parent/evidence IDs; persist keep/revise/reject/inconclusive decisions.
7. Require an explicit verified provider model, positive total spend cap and per-call token cap. Record requested/resolved model, provider, tier, usage and spend. Reserve conservative maximum call cost before dispatch; unreconciled usage remains reserved. Use bounded retries/backoff. Never assume the CLI's example model is available.
8. Provider outage permits queued work and Optuna within already approved spaces, labeled degraded-local. Missing Optuna or psutil blocks research runtime startup; six-seed fallback is preview-only. Exhaustion causes bounded replanning or explicit pause, not a retry loop.
9. Max-trials counts unique successful executions. Failed/pruned attempts are separate; account for in-flight jobs to avoid finite-budget overshoot. Unlimited removes only the success ceiling. Limit transient retries to three per trial; ten consecutive execution failures pause admissions with a visible cause.
10. Dry-run validates and previews using a temporary study, with no paid call, real reservation or worker. Real agentic mode cannot return dry-run success.

## Durability, Resources And Tracking
- Acquire one OS campaign lock. Ledger states queued -> running -> completed/failed/pruned; persist owner, heartbeat, process identity and lease expiry. Distinct retry attempts share recipe lineage. Successful deduplication signature includes recipe/data/protocol/code/environment/fidelity/seed.
- Worker writes atomic artifacts; controller commits terminal result, then reconciles Optuna tell and MLflow independently. Check existing terminal states before replay. Recover orphan asks and expired leases without rerunning completed work. Remove research_store's optimizer.utc_now import if controller integration creates a cycle; use a neutral/local time helper.
- Start two workers, measure peak process-tree RSS/CPU for 60 seconds; add two slots every 30 seconds with headroom. Limit by CPUs minus two, configured ceiling and available memory after reserve divided by observed peak RSS plus 25%. Reserve max(16GiB,20% RAM); account for existing workloads; pin BLAS/estimator threads to one.
- Honor zero slots. Sustained CPU >95% or reserve breach for 30 seconds reduces new admissions; disk below 10GiB pauses admission. Display utilization and normalized load separately as percentages. Do not kill healthy trials for startup spikes. Timeouts terminate only owned process groups.
- Outbox records all terminal outcomes; successes additionally get registered packages. One serialized uploader reconciles campaign/attempt/signature tags before creating runs/versions, including crash after remote success before local acknowledgement. Persist run/model/version/artifact URI. Tracking outage does not discard results; disk reserve bounds backlog.
- Package fitted estimator, feature order, transforms, calibration/blend, target prediction contract and full lineage. Preserve fold packages or explicitly identify the final-fold package used for readback; no silent refitting on scoring data. Registered-URI MLflow load must reproduce direct saved predictions, including non-default indices and secondary targets.

## Dependency and Tooling Preflight
- Inspect pyproject.toml and requirements-research.lock first. Supported Python >=3.11,<3.14. Audit found Python 3.12 in old imaopt venv; do not mutate/reuse it as the deployment environment.
- Local: `.venv/bin/python -m pip install -e '.[research,dev]'`; verify pinned imports. Remote: create dedicated supported venv; obtain Linux-compatible wheels on a reachable builder and transfer over Tailscale, then install offline with --no-index --find-links. No macOS binary wheels or assumed direct China downloads.
- Required: unittest, fixture planner, Optuna, psutil, sklearn, Pydantic, local SQLite-backed MLflow server, subprocess interruption harness, SSH/tmux. Check mature ranking dependency when needed.
- Browser/runtime binaries: none for terminal product; generated plan dashboard gets HTML readback using existing renderer. Browser smoke/Playwright is not required because no product UI or renderer code changes; do not report browser behavior as tested.
- Real blockers: server outage, missing credentials/model access, incompatible wheels after transfer attempts, missing target source data. Mark specific acceptance criteria blocked; do not claim the whole system complete.

## Deterministic Real-User Test
- Entry point: python -m scripts.optimize run --policy agentic --config fixture-config.json --campaign TEMP --max-trials 6 --proposal-batch-size 3 --max-concurrent-trials 2. The --config option is to be implemented, not currently claimed available.
- Config specifies tiny immutable dataset/protocol and fixture planner; credentials remain environment-only. Fixture planner reads actual completed metrics and conditionally chooses a valid next recipe. Mock provider transport only, not fitting, transformation, evaluation, packaging or persistence.
- Workflow: run -> status -> interrupt between cycles -> resume same campaign -> inspect six successful records and registered fitted artifacts. Separate counterfactual replay changes evidence and must change the decision.
- Command: `.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e tests.test_research_controller tests.test_research_executor` after creating missing tests.
- Evidence: CLI output, evidence hashes, decision lineage, ledger/export/Optuna counts, predictions, MLflow IDs and exit codes.

## Fulfillment and Readback Proof
Planning fulfillment is this document, checker output and dashboard readback. Leave future implementation checkboxes unchecked; planning completion is not implementation completion.

Implementation fulfillment joins proposal -> attempt -> recipe -> evaluated result -> next evidence -> next provider proposal on the real server. Require six successful win trials plus secondary probes, two real provider cycles, registered-model prediction readback, disconnect/restart proof and a further completed unlimited batch. Match ledger/Optuna/MLflow IDs. Fixture success cannot substitute for paid provider proof. Record deployed revision/interpreter/config, session/PID, resources and shared-service before/after readbacks.

## Armageddon Mode
Attack scope: controller, worker, provider, tracking and recovery boundaries. Must-fix threshold: any silent recipe loss, incorrect metric selection, duplicate completion/version, leaked owned worker or unrelated-service mutation blocks rollout.

Large integration risk. Test duplicate controllers/proposals, invalid parents/parameters, empty training windows, missing odds, non-finish times, incomplete races, NaNs, non-default indices, future features and nested holdout fields. Assert identical scoring populations across schemas.

Kill controller after ask, worker write, ledger commit, Optuna tell and MLflow registration before acknowledgement. Inject hung/nonzero workers, 429/timeouts/malformed provider JSON, exhausted budget, MLflow outage, disk pressure and zero slots. Must yield bounded recovery or explicit pause/failure without duplicate versions, leaked workers, false success or unrelated service changes. Inspect redundant data builds and memory peaks; performance improvements cannot bypass acceptance.

## Generality Guardrail
Existing mechanism: typed recipes, Optuna search, research ledger and provider adapter. Recurrence is high: every trial/cycle uses the same execution and feedback path, so integration belongs in a general controller/executor, not one-off scripts.

Reuse existing recipe/search/ledger/provider/package APIs. Add the missing general controller and target-aware executor; route all schemas/targets through them. No per-target daemon, new UI, competing scheduler or shell bridge. Preserve legacy consumers with a versioned new result schema.

## Regression Guardrails
- Current edit surface: this plan, original plan status/cross-reference and generated .mega artifacts only.
- Implementation surface: files in subphases below; new campaign/release paths only. Preserve old data and experiment history.
- Protected behaviors: local/openrouter legacy execution, historical MLflow records, deterministic race-day inference, human-only betting and shared services.
- Consumers: optimize CLI, remote wrapper, experiment readers, MLflow and future inference.
- Damage radius: planning small; implementation large.
- Branch strategy: use a dedicated branch named fix/agentic-feedback-controller for implementation from reviewed HEAD, preserving unrelated edits. Documentation-only planning remains on feat/agentic-optimizer-v2.
- Proof/commit plan: targeted checks per atomic subphase, integrated full suite, then isolated server canary. Push verified units with accurate status; no merge on component-only success.

## Ordered State and Dashboard
- Use dedicated .mega/feedback-repair/state.jsonl and .mega/feedback-repair/evidence.jsonl with explicit --state/--evidence arguments. The current renderer otherwise reuses unrelated phase numbers from the default ledger; prior completion records are not acceptance evidence.
- Generate .mega/dashboards/agentic-feedback-repair.html with mega_plan_dashboard.py and those dedicated ledgers. Read back title, five phases and all subphases planned; read root-cause/acceptance details in the linked canonical Markdown. No automatic opening required by this request.
- Record planning validation separately. Whole-plan final-gate applies to subsequent implementation completion; do not mark future phases done merely to pass it during this planning task.

## Detailed Implementation Handoff

The following names/interfaces are requirements to implement, not descriptions of features that already work. Read the source before applying changes; do not paste pseudocode as production code. If an existing helper already satisfies a requirement, extend it and test the execution path instead of duplicating it.

### Module boundaries and interface requirements
| Owner | Required public behavior | Must not own |
| --- | --- | --- |
| optimizer.run_campaign | Dispatch legacy modes unchanged; dispatch real agentic mode to controller | New evaluation logic |
| research_controller.run_research_campaign(config) | Startup reconciliation, scheduling, planning cadence, graceful shutdown, persisted status | Model-specific fitting |
| research_executor.execute_recipe(request) | Load frozen inputs, train/evaluate all required folds, write atomic artifacts, return typed result | Paid planner calls, campaign DB writes, selecting its own evaluation population |
| research_store.ResearchLedger | Lock/lease records, attempts, decisions, terminal results, effect acknowledgements | Sampling or model training |
| research_search.RecipeSearchController | Ask from approved versioned spaces; tell successful objective/failure/prune once; reconcile persisted trial IDs | Reading holdout/legacy test metrics |
| research_evidence | Build bounded allowlisted evidence from accepted results and capability report | Reading arbitrary secrets or raw feature rows into prompts |
| openrouter_orchestrator | Validated request/response transport, usage/tier metadata, bounded provider errors | Executing shell or editing model code |
| research_model_package / mlflow_tracking | Target-aware inference bundle and idempotent tracking reconciliation | Re-training during artifact loading |

Dependency direction: CLI -> optimizer dispatcher -> controller -> store/search/provider/executor. Store/search/model modules must not import the controller or optimizer. Move the existing store time helper dependency before connecting imports. Add an import smoke test for every new public module.

### Configuration to implement
Add --config PATH to the existing run command. Strict JSON schema, reject unknown keys. Explicit CLI values override config; omitted CLI values do not overwrite config with argparse defaults. Persist the resolved non-secret config, and reject incompatible changes on resume.

Example shape, with placeholders that MUST be resolved before a real run:
```json
{
  "schema_version": 1,
  "dataset_manifest": "campaign-inputs/dataset.json",
  "protocol_manifest": "campaign-inputs/protocol.json",
  "planner": {
    "mode": "openrouter",
    "model": "REPLACE_WITH_VERIFIED_MODEL_ID",
    "service_tier": "flex",
    "max_output_tokens": 4000,
    "max_total_cost_usd": 5.0,
    "max_retries": 3,
    "replan_every_terminal_trials": 32
  },
  "resources": {
    "initial_workers": 2,
    "max_workers": 26,
    "trial_timeout_seconds": 1200,
    "reserve_memory_fraction": 0.2,
    "reserve_memory_gib_min": 16,
    "disk_pause_gib": 10
  }
}
```
The USD 5 cap is a conservative example, not user-authorized unlimited spend or an estimate of actual cost. Reuse any explicitly configured campaign cap when available. Resolve model availability/pricing before enabling paid execution and reserve input plus maximum output cost. On the 28-thread audit host, 26 is an example ceiling, not a requested launch count; recompute from observed host capacity.

Planner mode fixture is permitted only with explicit fixture configuration. It must be plainly labeled fixture in status and outputs. Fixture mode has no API access. An absent API key in openrouter mode is an explicit startup error, never an implicit fixture/local substitution.

Resume invariants: data hash, protocol hash, code/environment identity, target definition and metric version cannot silently change. A change requires a new campaign or explicit versioned migration with tests. Operational caps may be decreased/increased with an audit record; never reset accumulated spend by restarting.

Add terminal commands status --campaign PATH and stop --campaign PATH. Status reads persisted state even when controller is offline and reports heartbeat staleness. Stop requests graceful draining through a campaign stop marker; controller stops admissions and exits once owned active trials are recorded. Existing run against the same campaign resumes it. Do not require a separate service to handle these commands.

### Controller algorithm
1. Validate dependencies/config/credentials and acquire campaign lock. Load or create immutable manifests, capabilities and execution identity. Reject legacy or incompatible campaign directories.
2. Reconcile completed ledger results with Optuna and MLflow. Resolve orphan asks and leases before scheduling. A stored completed result takes precedence over a stale running study entry.
3. Reap finished workers individually, validate outputs, commit one result immediately, and enqueue tracking/tell effects. Never wait for the whole worker batch before persisting.
4. Update resource observations and leases, retry bounded pending effects, refresh status. If stopping, drain without new asks/provider calls.
5. When planning is due, build evidence ONLY from committed results. Persist evidence hash and provider request before dispatch. Reserve cost, call provider, record response/usage, validate references and all recipes, then persist accepted/rejected decisions.
6. For each accepted proposal create/reuse its versioned approved search space. Enqueue its fixed anchor and matched control; ask additional numeric trials only within that space. A deterministic seed bootstrap is labeled bootstrap, not provider reflection.
7. Admit no more than the minimum of free resource slots, queue capacity and remaining success budget minus in-flight reservations. Register attempt/study identity before starting worker. Enforce one thread per estimator/BLAS worker.
8. Refresh global per-target incumbents from comparable development results. Replan at configured cadence or exhausted queue. Never select only the most recent batch winner.
9. Exit only at finite successful budget, graceful stop or explicit blocked/fatal condition. Unlimited mode continues as long as valid work and caps allow. Paused admission remains observable and does not hot-spin.

Use a short bounded event-loop wait or futures polling; provider HTTP timeout must be bounded so worker results are not stranded. If planning can block for long Flex latency, run the HTTP request in one supervised background future while the controller continues recording workers. Only the controller thread mutates campaign DB and study state.

### State and recovery details
Extend the existing SQLite schema with migration/version handling; do not silently delete existing tables. Suggested records: campaign metadata; proposals/evidence references; recipe signatures; attempts/leases; accepted results; effects with type, stable key, payload, acknowledgement and retry state. Stable full hashes, not only 12-character prefixes, enforce uniqueness; use non-colliding attempt IDs.

| Crash boundary | Recovery requirement |
| --- | --- |
| Optuna ask before ledger association | Locate trial by persisted campaign/proposal/signature attributes; attach or mark abandoned with explicit state; never leave unlimited RUNNING trials |
| Ledger running before worker spawn | Confirm no live matching process; return attempt to retryable state |
| Worker artifact written before controller commit | Validate manifest/hash and accept once; partial temporary files are not completion |
| Result committed before Optuna tell | Replay pending tell; compare objective/state if already terminal |
| Tell succeeded before effect acknowledgement | Mark reconciled without another terminal transition |
| MLflow run/version created before acknowledgement | Find matching stable tags and verify artifact identity; reuse existing run/version |
| Controller restart with surviving workers | Check PID plus process start identity and ownership, then recover output or stop only verified owned orphans; PID alone is insufficient |
| Conflicting second result for same attempt | Reject conflict and report corruption; never silently overwrite |

Successful model versions must not be duplicated across crash retries. Do not claim a distributed transaction across SQLite/Optuna/MLflow: use stable IDs, one uploader and reconciliation, and prove the uncertain-success cases in tests.

### Dataset and target implementation details
- Manifest includes source paths, content hashes, row/race/date counts, feature availability policy, fixed eligibility exclusions, protocol ID and supported targets. Hash actual immutable content, not file modification time alone. Store exclusion counts/reasons before search.
- Win target: exactly one eligible winner per race; development objective is race-mean negative log predicted winner probability. Log standalone and blended losses separately. Market recalibration is fitted only on calibration rows. Paired baseline deltas use the same races and bootstrap seed as the original plan.
- Placing: explicitly distinguish fixed top-k from official betting place eligibility. Fixed top-k uses k / field_size as uniform baseline; train binary classifier without race-sum normalization. Report race-averaged Brier/log loss plus hit-rate diagnostics. Dead heats and non-finishers follow predeclared rules.
- Ranking: use the tested mature ranker adapter with whole-race groups; never feed random runner splits. Report NDCG, winner reciprocal rank and within-race pairwise accuracy; specify tie policy. Use a simple pre-race rating/market ordering baseline. Score-to-probability calibration is a separate experiment with calibration-only fitting.
- Finish time/speed: existing apply_target_contract normalizes speed by the realized race median and accepts insufficiently constrained time values. Preserve compatibility only under an explicitly named legacy target version. For the new contract require finite positive seconds/distance and fixed exclusions. Choose raw speed distance/seconds plus a train-fitted expected-speed model using available course/distance/surface/going/class as the initial normalized target; residual is actual speed minus expected speed. Persist the normalizer and its training identity. Never require realized race median/pace at inference. Report speed MAE/RMSE and rank diagnostics; claim finish-time seconds only if reconstructing time has been implemented and tested.
- Odds: current helper merely copies market_probability as a target. That is not sufficient evidence of forecasting. Require snapshot timestamp < forecast timestamp, with forecast-horizon labels from later snapshots; exclude later odds from features. If historical snapshots are absent, record blocked capability and continue other targets.
- Negative controls: shuffle labels within appropriate race/day groups while preserving target validity, train the same adapter and report expected loss of signal on a synthetic learnable fixture. Historical negative controls are diagnostics, not a brittle promise that every metric always worsens.
- ResearchRegressor currently drops categorical features. Either use the existing schema-aware encoder or declare unsupported families explicitly; a recipe requesting categorical families cannot silently ignore them. Tests inspect effective feature columns.

### Required named acceptance tests
Create these tests or equivalent clearly named tests that assert the same behavior; do not replace them with mocks of training.
- test_agentic_run_without_dry_flag_trains_recipe: output mode executed, fitted artifact exists, estimator fit observed on fixture rows.
- test_dry_run_does_not_advance_study_or_call_provider: compare study/ledger/cost before and after preview.
- test_cycle_two_consumes_cycle_one_results: second prompt references committed IDs/metrics, resulting recipe runs and carries lineage.
- test_counterfactual_evidence_changes_next_recipe: change a controlled metric in a separate replay, assert different selected axis/recipe.
- test_transform_ablation_and_window_change_effective_training: compare effective columns, transform fit bounds and training row identities.
- test_score_and_holdout_never_fit_or_plan: sentinel future fields/dates and nested metric keys cannot reach fitting, sampler or provider evidence.
- test_resume_after_each_effect_boundary: parametrized crash windows from the table, no duplicated result/tell/version.
- test_secondary_targets_train_and_reload: ranking/place/speed predictions preserve correct output semantics after registered-package load.
- test_resource_zero_slots_and_startup_spike: no admission at zero headroom and no unnecessary healthy worker kill.
- test_unlimited_progresses_beyond_seed_catalogue: fixture campaign completes more than six distinct valid recipes, then graceful stop.
- test_provider_outage_and_spend_cap: bounded requests, spend survives resume, local continuation stays within approved spaces.
- test_real_registered_model_matches_saved_predictions: load via MLflow URI, compare target-specific predictions with stated numerical tolerance, reject missing lineage.

### Evidence bundle required before completion
Create a campaign acceptance report with deployed/local commit, command/config hash, environment identity, six win trial IDs, secondary trial IDs, at least two planner request/response hashes, parent links, completed Optuna counts, model registry URIs and prediction comparison errors. Include pause/restart timestamps and reconciliation outcomes. Every claimed item links to an actual artifact or API readback.

The server report must separately list fixture evidence and real provider evidence. Preserve the original canary's six legacy logit trials as legacy; never count them toward new recipe/protected-evaluator acceptance. An active tmux session, a passing dummy model test, or generated recipe JSON satisfies none of the actual-training criteria by itself.

## Phase 1: Execution Contract

### Subphase 1.1: Honest modes and configuration
- Objective: inspect CampaignConfig, CLI, wrapper and dependency pins; implement explicit config, provider caps and preflight before launching work.
- Planned Touch Files: ima/optimizer.py, scripts/optimize.py, scripts/cortex_optimizer_worker.py, tests/test_optimizer.py, tests/test_cortex_optimizer_worker.py, docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md, pyproject.toml, requirements-research.lock.
- Commit: fix(optimizer): distinguish executable agentic mode from preview.
- Tests: `.venv/bin/python -m unittest tests.test_optimizer tests.test_cortex_optimizer_worker`.
- Success Criteria: preview has no paid/persistent search effects; real agentic mode fails explicitly until controller integration lands; missing required dependencies are visible.
- Checklist:
  - [ ] Add validated --config, provider model/spend/token caps and wrapper forwarding.
  - [ ] Remove silent non-dry fallback; preserve legacy modes and test CLI output.
  - [ ] Document staged implementation status honestly.
- Handoff: tested CLI contract, dependency report and secret-free example config.

## Phase 2: Recipe Execution

### Subphase 2.1: Protected win worker
- Objective: inspect feature builders/models/transforms/evaluator; execute full recipes without lossy legacy conversion.
- Planned Touch Files: ima/research_executor.py, ima/research_evaluation.py, ima/research_specs.py, ima/research_transforms.py, ima/research_models.py, tests/test_research_executor.py, tests/test_research_evaluation.py, tests/fixtures/research_races.csv.
- Commit: feat(research): execute recipes with temporal evaluation.
- Tests: `.venv/bin/python -m unittest tests.test_research_executor tests.test_research_evaluation tests.test_research_transforms tests.test_research_specs`.
- Success Criteria: six anchors actually exercise schemas, families, ablation, window and transform with distinct train/calibration/score populations.
- Checklist:
  - [ ] Implement immutable request/result contracts, fold fitting and fitted-transform reuse.
  - [ ] Save predictions, baselines, hashes and paired comparisons; reject invalid artifacts.
  - [ ] Inspect actual selected columns and training dates, not only recipe JSON.
- Handoff: worker API and real anchor outputs.

### Subphase 2.2: Secondary target execution
- Objective: connect contracts to compatible estimators/evaluation after auditing source coverage.
- Planned Touch Files: ima/research_executor.py, ima/research_targets.py, ima/research_models.py, ima/research_specs.py, tests/test_research_executor.py, tests/test_research_targets.py, tests/test_research_models.py, docs/research-cohort-v2.md.
- Commit: feat(research): execute ranking placing and time probes.
- Tests: `.venv/bin/python -m unittest tests.test_research_executor tests.test_research_targets tests.test_research_models`.
- Success Criteria: each required target trains and scores against baseline/negative control; unsupported odds snapshots are identified rather than fabricated.
- Checklist:
  - [ ] Record parse coverage, dead-heat/non-finish rules and feature availability constraints.
  - [ ] Implement target metric direction/prediction contract and separate study identities.
  - [ ] Advertise only executable tested adapters to planner.
- Handoff: target capability matrix and probe artifacts.

## Phase 3: Feedback Controller

### Subphase 3.1: Durable ask execute tell
- Objective: connect search, ledger and workers under one owner; resolve ledger/controller import cycle before integration.
- Planned Touch Files: ima/research_controller.py, ima/research_store.py, ima/research_search.py, ima/optimizer.py, tests/test_research_controller.py, tests/test_research_store.py, tests/test_research_search.py.
- Commit: feat(optimizer): connect durable recipe execution and search feedback.
- Tests: `.venv/bin/python -m unittest tests.test_research_controller tests.test_research_store tests.test_research_search tests.test_optimizer`.
- Success Criteria: real CLI trains; results update Optuna once; finite count does not overshoot; search continues beyond six seeds.
- Checklist:
  - [ ] Add lock, leases, attempt lineage, atomic completion and orphan recovery.
  - [ ] Reconcile ask/result/tell crash windows and preserve completed work.
  - [ ] Export status and graceful stop/resume through same controller.
- Handoff: six-success campaign with matching ledger/study/export identities.

### Subphase 3.2: Evidence-driven provider planning
- Objective: call choose_research_proposals from controller with completed evidence and validated bounded search-space control.
- Planned Touch Files: ima/research_controller.py, ima/research_evidence.py, ima/research_specs.py, ima/research_search.py, ima/openrouter_orchestrator.py, tests/test_research_controller.py, tests/test_agentic_planner.py, tests/test_research_evidence.py, tests/fixtures/agentic_planner.json.
- Commit: feat(optimizer): replan from completed research evidence.
- Tests: `.venv/bin/python -m unittest tests.test_research_controller tests.test_agentic_planner tests.test_research_evidence tests.test_research_specs`.
- Success Criteria: cycle two references real results and changes executed recipes; counterfactual evidence changes the decision; provider/budget failure is explicit.
- Checklist:
  - [ ] Implement strict bounded search spaces and matched-control validation.
  - [ ] Save redacted prompts/responses, parent/evidence validation, reflection, usage and tier.
  - [ ] Enforce caps/retries and approved-space-only degraded search.
- Handoff: causal two-cycle fixture evidence plus executable provider path.

## Phase 4: Recovery And Tracking

### Subphase 4.1: Resource admission
- Objective: inspect current partial hook and implement sustained observation with owned-process supervision.
- Planned Touch Files: ima/research_controller.py, ima/research_resources.py, tests/test_research_controller.py, tests/test_research_resources.py.
- Commit: feat(optimizer): admit workers from measured resource headroom.
- Tests: `.venv/bin/python -m unittest tests.test_research_resources tests.test_research_controller`.
- Success Criteria: zero slots launches nothing; startup spikes preserve healthy jobs; measured RSS constrains scaling.
- Checklist:
  - [ ] Implement observation, ramp, reserve, thread caps and sustained pressure handling.
  - [ ] Verify timeouts, graceful stop, disk pressure and other-workload accounting.
- Handoff: reproducible resource traces and owned-process cleanup proof.

### Subphase 4.2: Trained model registry
- Objective: package actual trained outputs and reconcile outbox after tracking/controller interruption.
- Planned Touch Files: ima/research_model_package.py, ima/mlflow_tracking.py, ima/research_store.py, ima/research_controller.py, tests/test_research_model_package.py, tests/test_mlflow_tracking.py, tests/test_research_store.py.
- Commit: feat(mlflow): reconcile target-aware trained model versions.
- Tests: `.venv/bin/python -m unittest tests.test_research_model_package tests.test_mlflow_tracking tests.test_research_store`.
- Success Criteria: registered URI reload reproduces direct predictions for all executed targets; crash after registration creates no duplicate.
- Checklist:
  - [ ] Include fitted components, prediction contract, lineage and artifact hashes.
  - [ ] Test interrupted upload, reconciliation, non-default indices and complete-race validation.
  - [ ] Log failures and expose pending uploads distinctly from missing training.
- Handoff: API readbacks, version IDs and numerical prediction comparisons.

## Phase 5: Acceptance And Server Rollout

### Subphase 5.1: Real terminal acceptance
- Objective: replace dry-run-only proof with deterministic trained feedback and adversarial interruption tests.
- Planned Touch Files: tests/test_agentic_optimizer_e2e.py, tests/test_research_controller.py, tests/fixtures/agentic_planner.json, scripts/optimize.py, docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md, docs/AGENTIC_OPTIMIZER_V2_PLAN.md.
- Commit: test(optimizer): prove trained feedback and restart recovery.
- Tests: `.venv/bin/python -m unittest tests.test_agentic_optimizer_e2e`; then `.venv/bin/python -m unittest discover -s tests`.
- Success Criteria: all local acceptance artifacts join correctly; completion status matches proven scope.
- Checklist:
  - [ ] Run six-success CLI, secondary probes, counterfactual reflection and interrupted resume.
  - [ ] Execute Armageddon cases and archive exact commands/evidence.
  - [ ] Publish actual run/status/stop/resume/unlimited commands using implemented flags.
- Handoff: local acceptance bundle; unexplained failures block rollout.

### Subphase 5.2: One isolated Cortex controller
- Objective: refresh read-only host state, transfer dependencies via Tailscale and deploy verified code as imaopt.
- Planned Touch Files: scripts/cortex_optimizer_worker.py, tests/test_cortex_optimizer_worker.py, docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md; remote imaopt-owned new release/venv/campaign paths only.
- Commit: feat(worker): deploy verified feedback controller canary.
- Tests: `.venv/bin/python -m unittest tests.test_cortex_optimizer_worker`; record actual SSH, provider, ledger, Optuna and MLflow readbacks in runbook.
- Success Criteria: two real provider cycles, six successful win trials, secondary probes, registry readback, disconnect/restart proof and subsequent unlimited progress.
- Checklist:
  - [ ] Inspect sessions/process ownership, services/start times, resources, revision and dependency imports; preserve old evidence.
  - [ ] Stage immutable release under /home/imaopt/research-v2/releases and dedicated venv. Do not edit active worker files or old iMa venv.
  - [ ] Archive logs, then stop only the positively identified obsolete ima-agentic-v2 preview wrapper. Recheck any trainer before touching it; never broad pkill.
  - [ ] Launch one ima-feedback-v2 tmux controller with two initial workers, verified provider model/Flex, spend cap and new campaign. Verify API access from server itself.
  - [ ] Disconnect/reconnect while trials advance; restart only this controller to prove recovery; read back cycle-two provider evidence and registered predictions.
  - [ ] Resume --max-trials unlimited with auto admission after canary passes; observe at least one further completed batch. Keep spend/resource caps.
  - [ ] Confirm shared-service start times/configuration unchanged. Rollback stops only owned controller/worker process groups and preserves evidence.
- Handoff: deployed revision, PID/session, active model/tier, campaign path, spend/resources, success/failure counts, decisions, registered versions and observed continued progress. Missing live provider evidence means agentic acceptance remains blocked, even if local search works.
