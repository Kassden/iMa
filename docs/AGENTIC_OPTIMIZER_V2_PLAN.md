# Agentic Optimizer V2: Execution Plan

## GOAL
Build a terminal-operated research loop that uses completed evidence to choose features, transforms, training windows, model families and hyperparameters; runs experiments autonomously as imaopt; and records reproducible decisions, metrics and model versions in MLflow.

Status: PARTIALLY IMPLEMENTED; end-to-end acceptance NOT MET. The 2026-09-25 audit found disconnected proposal/training execution and no automatic feedback. Follow [Agentic Optimizer Feedback Repair](AGENTIC_OPTIMIZER_FEEDBACK_REPAIR_PLAN.md) for the corrective execution sequence; the acceptance criteria below remain binding. Original research date: 2026-09-24. Original reviewed code: f6b560f138b780b9506f9924c65beed31fcb845b.

## Acceptance Criteria
- [ ] Two consecutive agent cycles consume completed results and generate valid new recipes outside the legacy catalogue; recorded replay proves cycle two responds to cycle one.
- [ ] Trials cover two schemas, a feature-family ablation, a training-window change, a transform and two model families.
- [ ] Exploratory secondary targets run behind explicit target contracts and tests: ranking, placing/top-k and adjusted finish-time or speed. Each target has leakage checks, baselines and separate metrics.
- [ ] Selection uses rolling development scores only; calibration and scoring windows differ; holdout metrics never enter the planner or sampler.
- [ ] Interrupt/resume preserves completed work and produces no duplicate terminal results or registered versions.
- [ ] Every completed trained candidate has a loadable versioned bundle, recipe, dataset/split/code hashes, hypothesis, parent and MLflow linkage.
- [ ] Server canary survives laptop disconnection, uses measured resource headroom and preserves Cortex/solar workloads.
- [ ] Engineering success does not require lower log loss. Any improvement claim needs paired evaluation against market recalibration and a frozen incumbent.

## Scope And Conversation
Interpret "auto email, ML" as AutoML in this context; no email integration. Keep Wednesday race-day readiness for later, including deterministic inference and human betting. Keep terminal operation and MLflow; no new application. Unlimited trials do not imply unlimited spend, retries, memory or disk. First deliver agent-selected composable recipes. Arbitrary generated code is a separately gated later milestone, not falsely claimed as implemented by recipe optimization. This planning turn permits local documentation and read-only remote inspection, not replacing the live campaign.

## Research
Primary sources checked 2026-09-24; prior art is not evidence of predictive lift on this dataset.
- [Karpathy autoresearch README](https://github.com/karpathy/autoresearch/blob/master/README.md) and [program.md](https://github.com/karpathy/autoresearch/blob/master/program.md): constrained edit/train/evaluate loop, fixed evaluator, retain/discard history. Adapt its experimental discipline, not its GPU workload or reset-based handling.
- [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent): hypotheses, feature/data/model development and feedback-driven refinement. Borrow the research/evaluation separation instead of replacing the existing executor.
- [AIDE](https://github.com/WecoAI/aideml): branching draft/debug/improve search. Retain parent lineage and failed ideas rather than following only one incumbent.
- [Optuna ask/tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html), [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [JournalStorage](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/011_journal_storage.html): adaptive parameter search, external execution and persistence.
- [scikit-learn nested evaluation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): separate model selection from performance estimation.
- [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html): useful later independent benchmark; its standard evaluation does not establish race-grouped temporal validity.
- [MLflow registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/): versioned packages need load/readback proof, not merely a registry entry.
- [OpenRouter service tiers](https://openrouter.ai/docs/guides/features/service-tiers) and [Batch API](https://openrouter.ai/docs/batch-quickstart): distinguish inference tier, inference batching and local training concurrency; check actual provider/tier/cost.

## Root-Cause Baseline
- Trigger scope: low-yield research and incomplete feedback, not an observed server outage.
- Minimum evidence inventory: earlier report, current code, remote trials/decisions JSONL, bounded process logs and three MLflow API readbacks; see docs/AGENTIC_OPTIMIZER_V2_EVIDENCE.md.
- Proven: 2,048 completed sampled trials; 1,966 logit, 82 boosted, all baseline-v1; 1,006 zero fundamental weights; best blended loss 2.0118061545903827.
- Proven: optimizer _trial_sort_key/voting_rank use test metrics. The former test period is already part of development. Repartitioning previously inspected outcomes does not restore an untouched holdout.
- Proven: planner_messages sends available specs without results; _proposals_from_remote_payload accepts only existing IDs. Live policy is local, not an active OpenRouter agent.
- Proven: adaptive generation drains 2,616 specs before varying parameters around top 24. It cannot change features/datasets/transforms.
- Proven: execute_proposals persists after the entire batch; workers rebuild data per trial; auto concurrency uses CPU count without memory feedback.
- Likely hypothesis: redundant logistic settings and odds calibration explain low yield. Matched predictions are needed to quantify this.
- Possible: richer features or market-conditional training add useful information; current evidence does not prove this.
- Disproven interpretation: zero weight does not demonstrate blend failure. Lower standalone log loss does not by itself demonstrate better ranking.
- Missing evidence: prediction-level uncertainty, trustworthy unseen dates, feature availability timestamps, environment lock, paid provider behavior. Collect in Phase 1; do not invent calendar folds or a generalization claim.
- Mutation boundary: isolated new checkout/campaign; no running-worker edits, restarts or shared-service changes during research.
- Remediation mapping: contaminated evaluation -> 1.2; narrow search -> 2/3/5; absent feedback -> 4; persistence/resource limitations -> 3.2; artifact loadability -> 5.2.
- Generic hardening: leases/outbox/backpressure reduce operational risk; they are not proven causes of weak predictions.

## Corrections To Earlier Report
Repair evaluation FIRST. Treat 2.011806 as a historical development result, not a fresh-protocol acceptance target. Replace the arbitrary 0.00005 noise threshold with paired uncertainty and a predeclared practical threshold; intervals after adaptive selection still do not establish unseen performance. Freeze equal evaluation populations across schemas.

The existing multiplicative blend equals softmax(a*log(p_model)+b*log(p_market)). Renaming it an additive-log blend changes nothing. Residual training must change the fitted objective/features. Raw y-minus-market regression is only a hypothesis, not guaranteed orthogonal signal. Existing joblib registry versions should remain available, but new versions need standard MLflow loadability.

## SOTA, Standards, And Best Practices
- Implementation decision: retain optimizer/experiments, sklearn, MLflow, feature registries and remote wrapper. Add Optuna for parameter search, Pydantic for strict recipes, psutil for observations.
- One controller owns ask/tell and bookkeeping. Use local JournalStorage for Optuna; no distributed SQLite writers.
- Agent chooses research axes and bounded search spaces; Optuna chooses numeric settings. No LLM call per C value.
- Fix evaluation/split manifests; let training windows vary, never evaluation populations.
- Rejected approaches: endless tolerance grids, test-driven voting, random runner CV, blanket framework replacement, automatic promotion and arbitrary planner shell execution.
- Mature LightGBM ranking is optional after dependency/evaluation gates; no handwritten LambdaMART.
- "Learning" here means feedback, persistent experiment memory and adaptive search. Fine-tuning the orchestrator is separate.

## Dependency and Tooling Preflight
Inspect pyproject.toml and tests/test_optimizer.py, tests/test_experiments.py, tests/test_mlflow_tracking.py, tests/test_cortex_optimizer_worker.py. Supported Python: >=3.11,<3.14; verify actual interpreters. Add optional research dependencies and pin compatible exact versions in requirements-research.lock. Set sampler options explicitly rather than inheriting changing defaults.
- Install or repair commands: project venv pip install -e '.[research,dev]' after adding the research group; verify wheels before optional LightGBM.
- China host: prepare Linux-compatible wheels on a reachable builder and transfer over verified Tailscale route, or use verified tailnet tunnel. macOS wheels are not Linux wheels.
- No install is needed for this documentation deliverable.
- Required verification tools: unittest, fixture planner, local MLflow store, process interruption harness.
- Browser/runtime binaries: no product frontend; generated dashboard receives HTML content readback. Playwright browser smoke is not required for this documentation-only dashboard generated by the unchanged Megaskill renderer.
- Real blockers: incompatible runtime/wheels after setup attempts, missing server/API credentials, external outage. No clean holdout blocks generalization claims, not development.

## Architecture And Data Contracts
Frozen dataset/protocol -> evidence bundle -> agent proposal -> validator -> Optuna ask -> leased worker -> protected evaluator -> atomic result -> Optuna tell + MLflow outbox -> refreshed evidence -> next proposal.

PipelineRecipe v2 in ima/research_specs.py:
- schema_version=2, target.kind, target.parameters, feature_schema, sorted drop_feature_families, transforms, train_window, model.kind/parameters, calibration, blend, seed.
- Dataset fingerprint and protocol ID are campaign-owned. Recipe cannot choose labels, metric implementation, score population or protected paths.
- Target choices are registered capabilities, not free text. Initial allowed target kinds: win_probability, ranking_strength, placing_top_k, adjusted_finish_time_or_speed and market_odds_forecast.
- Registered transform operations: imputation/scaling, train-fitted clipping, race-relative ranks and lagged history. No eval/import strings.
- Initial train windows: all_history or trailing_3_years. Older point-in-time history may build features; estimator training rows obey the window.
- Signature hashes canonical recipe, data, protocol, code revision, environment and fidelity; seed included. Deduplicate pending/running/completed attempts.
- Preserve old ExperimentSpec constructors and legacy campaign reading. Summaries label old test metrics legacy_dev without rewriting historical records.

ResearchProposal v1:
proposal_id, parent_trial_ids, evidence_ids, hypothesis, changed_axes, recipe/search_space, expected_observation, falsification_rule, max_trials, max_wall_seconds. Strict registered capabilities/bounds; reject unknown keys and invalid model/target combinations. No executable code, arbitrary paths, secrets or evaluator overrides. Multi-axis proposals require a matched control and an ablation.

EvidenceBundle v1:
snapshot/hash IDs; schema capabilities; missingness/source coverage; paired development baseline loss and uncertainty; top 10 candidates; 10 representative failures; last 32 completed trials; family coverage; running recipes; budget; accepted/rejected hypotheses. Allowlist development fields before serialization. No raw runner rows, credentials or holdout metrics. Persist exact redacted input/response. Reflection must cite actual trial IDs and record keep/revise/reject/inconclusive.

Maintain a GLOBAL development incumbent separately from batch winner. Voting remains diagnostic. Primary selection: race-weighted mean development log-loss delta versus calibrated market; tie-break worst-fold delta then cost.

## Prediction Targets And Metrics
This is an exploratory research system, so secondary targets are allowed early, but every target needs its own label contract, baseline and tests. Do not collapse all targets into one leaderboard. A model can improve ranking while hurting probability calibration; that is a useful finding, not an automatic champion.

Primary target:
- win_probability: probability each runner wins the race. Primary score is race-weighted log loss versus calibrated market and frozen incumbent. Secondary diagnostics: top-pick accuracy, winner rank, calibration and per-race loss.

Secondary exploratory targets:
- ranking_strength: within-race ordering or latent strength. Initial metrics: winner rank, mean reciprocal rank, pairwise accuracy within a race, top-k hit rate and NDCG where appropriate. Ranking scores must be calibrated separately before they become win probabilities.
- placing/top_k: probability of finishing in top 2, top 3 or official place positions. Requires race-size/place-rule awareness and explicit dead-heat/non-finish handling. Metrics: top-k log loss or Brier score, place hit rate and calibration.
- adjusted_finish_time_or_speed: runner finish-time signal normalized by distance, course, surface, going, class and race pace where possible. Raw absolute finish time is not a universal target across race conditions. Metrics: MAE/RMSE on score folds, within-race rank correlation, winner-rank lift and residual diagnostics by distance/going/course.

Target guardrails:
- Labels are post-race facts and may never enter pre-race features except as lagged, point-in-time horse history.
- Finish-time targets require parse coverage, missingness report, non-finish quarantine rules and condition-normalization tests.
- Every target contract must include uniform/simple-history baselines and a shuffled-label negative control.
- The orchestrator may choose target.kind as a research axis, but the validator owns compatibility: classifiers cannot silently train regression targets, odds forecasts cannot be promoted as outcome champions, and ranking scores need calibration before probability/EV use.
- Cross-target selection is diagnostic unless a predeclared multi-objective rule exists. Production/race-day promotion still requires the win-probability path and human approval.

## Evaluation Protocol
Win probability is the primary target; ranking, placing and adjusted finish-time/speed are secondary exploratory targets with separate target contracts.
1. Derive immutable whole-race-day partitions from actual date coverage, never guessed years.
2. At least three expanding development folds where data allows, each train < calibration < score. Fixtures use explicit smaller partitions. All runners in a race stay together.
3. Fit preprocessing/estimator on train. Fit temperature, market recalibration and blend weights on calibration. Early stopping cannot inspect score.
4. Historical features only use outcomes available before the prediction timestamp; no current-race or unavailable same-day results.
5. Initially require one winner per eligible race. Quarantine dead heats/invalid labels using fixed rules and report counts. Recipes cannot filter hard evaluation cases.
6. Evaluate uniform, raw market, calibrated market and frozen incumbent on identical races/runners. Missing odds/population changes must be explicit.
7. Save per-race losses; aggregate equal race weights. Report worst fold, top pick, calibration, runtime and RSS as secondary metrics.
8. Use 2,000 seeded race-day-block bootstrap resamples for paired differences. Exploratory intervals after selection are descriptive.
9. Fresh holdout must have documented non-use by previous searches. If none exists, all current data is development; collect future outcomes before final claims. Do not relabel used data as clean.
10. Freeze finalist and protocol before holdout evaluation; do not feed holdout back into the same loop. Evaluation process owns labels; planner exports exclude them.
11. New data creates a new campaign fingerprint and reruns baselines. Never silently append data to an active study.

## Search, Cost And Resource Policy
- Study identity includes dataset/protocol/fidelity/search-space version. TPE seed=42 and constant_liar=True; persist sampler configuration/state.
- Initial ranges: logit C [0.003,80] log scale and balanced/None; boosted learning_rate [0.015,0.12], iterations [80,260], leaves {7,15,31,63}, L2 [0,10]. Fix solver/tolerance.
- Seed schema/model diversity. Reserve 25% of each cycle for underexplored valid recipes; 75% exploits evidence-backed spaces.
- Plan after initial diverse cohort, then every 32 terminal trials or queue exhaustion. Queue <=2*active_workers.
- Start two workers, sample peak RSS/CPU for 60 seconds, add two slots each 30 seconds with headroom. These are configurable initial policy constants, not measured optimal values.
- Slots <= min(CPUs minus two, floor((MemAvailable-reserve)/estimated_peak_trial_RSS), configured ceiling). Reserve max(16GiB,20% RAM); use observed high-water RSS plus 25%.
- Pin BLAS/OpenMP and estimator threads to one per trial. Show CPU utilization and normalized load separately as percentages; load includes runnable/I/O waiting tasks.
- After 30 seconds sustained CPU >95% or reserve violation, reduce new admissions. Do not kill healthy work due to a startup spike. Disk <10GiB pauses admission.
- Trial timeout defaults to 20 minutes, overridable by family. Kill only owned process groups for timeout/OOM.
- Single-controller SQLite ledger (stdlib) records unique attempts/leases; Optuna owns sampling, MLflow owns experiment tracking, JSONL is export.
- Persist each completed result before upload. Idempotent outbox reconciles stable IDs/tags after crash between remote success and local acknowledgement.
- max-trials counts completed successes; failures/prunes separate. Ten consecutive failures disable a recipe pending reflection.
- API outage: continue queued work and Optuna within approved spaces; bounded backoff and explicit planner-unavailable status.
- Paid planning requires a configured USD/token cap; unlimited trials never bypass it. At cap, local search may continue.
- Default one synchronous Flex planning call per cycle; optional inference Batch for independent critiques, not dependent sequential decisions.
- Current adapter targets /api/beta/batches; current official docs specify /api/v1/batches. Verify and implement create/poll/results/reconciliation. Legacy route failure is not proven without a call.
- Flex docs allow standard routing if no Flex endpoints exist. Check capability and actual tier/cost, reject unexpected tier under strict-cost policy. Do not assume Flex and Batch discounts stack.
- Model comes from IMA_OPTIMIZER_MODEL. Select cheapest provider/model passing a 20-case recorded decision benchmark: >=95% valid schemas, zero leakage requests, correct responses to plateau/failure/cost cases. Compare current costs during implementation; no reputation-based fixed winner.

## MLflow And Model Versions
New experiment ima-research-v2; leave ima-racing and old registry records intact. Campaign/proposal/trial/fold linkage via parent runs/tags. Log recipe/data/protocol/code/environment hashes, Optuna IDs, parent trials and evidence IDs.
Artifacts: recipe.json, evidence.json, decision.json, dataset_manifest.json, split_manifest.json, fold_metrics.json, compressed per-race loss CSV, full model package/signature.
Wrap full model/calibrator/blend as MLflow pyfunc. Input requires complete races plus schema/market columns. Reject incomplete races when completeness cannot be established from declared field_size. Keep legacy joblib consumers. Register completed trained candidates, not failed/pruned models. Download/load one remote version and match direct probabilities.
No champion aliases change. Best-so-far development charts use trial index and compute hours and mark protocol changes; never compare incompatible folds as one curve.

## First Research Cohort
- Controls: uniform/raw-market/calibrated-market.
- Six anchors: baseline-v1/benter-rich-v1/notebook-rich-v2 crossed with logit/boosted at fixed defaults.
- Best two development recipes: separate source_quality/preferences family drops where defined, train-fitted clipping, all-history versus trailing-three-year ablations.
- Secondary target probes: ranking_strength, placing/top-k and adjusted_finish_time_or_speed using the same frozen folds and explicit target contracts.
- Optuna expands promising spaces.
- Market-conditional boosted classifier includes research-available log market probability as a feature, trains on train, calibrates later. This is a mature-estimator baseline, not an offset-loss claim.
- Then bounded linear race-softmax offset model z_i=log(q_i)+X_i*w; mean race NLL plus L2, scipy.optimize and verified analytic gradient. Race-center features, omit unidentified common intercept; w=0 exactly recovers q. This is the narrow domain-specific component.
- Optional LightGBM winner ranker with race groups and separate temperature calibration; log loss is still primary.
- Optional nonnegative sum-to-one ensemble trained on aligned temporal out-of-fold predictions. No test-trained stacking or averaging uncalibrated scores.
Final odds are research_only. Live eligibility requires timestamped pre-race snapshots. None of these experiments guarantees improvement. Secondary-target wins do not imply tradable win-probability edge without calibrated conversion and paired evaluation.

## Deterministic Real-User Test
- Entry point to implement: ima-optimize run --policy agentic --campaign <new-dir> --protocol <manifest> --max-trials 6 --max-concurrent-trials 2 --planner-fixture <fixture>.
- User workflow: launch, stop after two completions, resume, inspect MLflow and load candidate.
- Stable inputs: synthetic chronological races with train-only signal, fixed seeds and two recorded planner cycles.
- User-observable assertions: six unique completions, two feedback cycles, recipe beyond old grid, no holdout fields, one recovered failure, loaded probabilities sum to one.
- Command: .venv/bin/python -m unittest tests.test_agentic_optimizer_e2e.
- Evidence: transcript, ledger, planner input, run/version IDs and prediction equality. Fixture proof is distinct from real-data research.

## Fulfillment and Readback Proof
- Required write/mutation operation: implementation creates a NEW research campaign, lineage and model packages.
- Readback surface: terminal, durable ledger, MlflowClient run/artifact/model APIs.
- Expected content: consistent IDs/counts/hashes, real second-cycle evidence, matching loaded predictions.
- Not-done conditions: catalog-only LLM selection, mock-only live claim, unloadable registered file, lost completion, holdout leakage.
- Planning-turn fulfillment: evidence-backed documents, plan checker and generated dashboard readback; future implementation remains unchecked.

## Armageddon Mode
- Attack scope: large future implementation, fixtures before remote canary.
- Edge cases: empty/one-race data, shuffled/duplicate runners, dead heats, missing odds, NaNs, future timestamps, all-dropped schema, unknown transform, excessive parameters.
- Failure modes: truncated JSON, unknown fields, duplicate/stale proposals, 429/timeouts, wrong tier, MLflow outage, disk full, OOM, worker/controller kill, upload success before acknowledgement crash.
- Leakage attacks: future-result feature, split/label/holdout modification, unavailable same-day outcomes; reject before training.
- Optimization opportunities: data rebuilds, duplicated trials/predictions, excessive queues, nested threads and reconciliation.
- Must-fix: silent leakage/loss, false version success, unintended service impact, path escape or spend bypass.
- Safe to defer: arbitrary code generation and extra frameworks, never recipe feedback.
- Evidence: tests, failure artifacts, resource measurements and remediation recorded in .mega/evidence.jsonl.

## Generality Guardrail
Existing mechanism: optimizer, feature registries, sklearn and MLflow. Recurrence is high across campaigns. Shared owners research_specs/evaluation/store serve all campaigns; summary script is a thin caller. No per-experiment evaluator copies, one-off UI or duplicate orchestration framework.

## Regression Guardrails
- Planned edit surface: per-subphase files below; this turn edits documentation only.
- Protected behaviors: legacy replay/model consumers, race-day determinism, human promotion and all other server workloads.
- Likely consumers: optimizer CLI, remote wrapper, MLflow and later inference.
- Damage radius: small planning; large implementation.
- Branch strategy: docs on current audit branch; implementation creates dedicated feat/agentic-optimizer-v2 checkout before edits, keeping live /home/imaopt/iMa unchanged.
- Proof plan: contracts, adjacent legacy tests, deterministic terminal workflow, failure injection and imaopt-only canary.
- Atomic commits: one per subphase. No unrelated cleanup or inferred production deployment.

## Ordered State and Dashboard
State .mega/state.jsonl; evidence .mega/evidence.jsonl; dashboard .mega/dashboards/agentic-optimizer-v2.html. Generate with mega_plan_dashboard.py --no-open; readback must show pending phases. Record starts, commits/tests and handoffs. Research completion does not complete implementation checkboxes.

## Phase 1: Evidence And Evaluation

### Subphase 1.1: Reproducible audit
- Objective: summarize JSONL/MLflow and inspect source/date manifests.
- Planned Touch Files: scripts/summarize_optimizer_campaign.py, ima/research_evidence.py, tests/test_research_evidence.py.
- Commit: feat(research): summarize campaign provenance.
- Tests: .venv/bin/python -m unittest tests.test_research_evidence.
- Success Criteria: counts/IDs agree; corrupt/truncated rows explicit; old test metrics labeled development.
- Checklist:
  - [ ] Snapshot trials/decisions without secrets; verify first/best/latest MLflow runs.
  - [ ] Export timestamped hashed summary and mismatches.
- Handoff: audit JSON, coverage/availability manifest and contamination note.

### Subphase 1.2: Protected evaluator
- Objective: inspect data.py, rich_features.py, experiments._run_one and implement Evaluation Protocol.
- Planned Touch Files: ima/research_evaluation.py, ima/experiments.py, tests/test_research_evaluation.py, tests/fixtures/research_races.csv, docs/research-evaluation.md.
- Commit: feat(research): isolate calibration and temporal scoring.
- Tests: .venv/bin/python -m unittest tests.test_research_evaluation tests.test_experiments tests.test_rich_features tests.test_research_targets.
- Success Criteria: no overlap/future leakage; equal populations; baselines; honest holdout status; target contracts reject leakage and malformed labels.
- Checklist:
  - [ ] Freeze actual date manifests and synthetic fixture.
  - [ ] Separate fit/calibration/score and save paired per-race losses.
  - [ ] Add target contracts for win probability, ranking/top-k and adjusted finish-time/speed with fixtures.
  - [ ] Reject evaluator overrides; preserve legacy defaults.
- Handoff: protocol hash/date counts and adversarial evidence.

## Phase 2: Pipeline Recipes

### Subphase 2.1: Typed schemas and transformations
- Objective: inspect feature_sets and load_full_rich_history; implement PipelineRecipe and routing.
- Planned Touch Files: ima/research_specs.py, ima/research_targets.py, ima/research_transforms.py, ima/experiments.py, ima/feature_sets.py, tests/test_research_specs.py, tests/test_research_targets.py, tests/test_research_transforms.py, pyproject.toml, requirements-research.lock.
- Commit: feat(research): add typed pipeline recipes.
- Tests: .venv/bin/python -m unittest tests.test_research_specs tests.test_research_targets tests.test_research_transforms tests.test_experiments tests.test_modeling.
- Success Criteria: old constructors pass; schemas/ablation/window/transform run on same score races; each exploratory target has deterministic labels and metrics.
- Checklist:
  - [ ] Pin dependencies; validate capabilities and recipe hashes.
  - [ ] Register target kinds and metric families; reject incompatible model/target combinations.
  - [ ] Implement train-only transforms and availability checks.
  - [ ] Cache immutable features by source/code/schema hash, never reuse fitted preprocessing across folds.
- Handoff: valid/invalid recipes and deterministic round trips.

## Phase 3: Search And Execution

### Subphase 3.1: Optuna inner loop
- Objective: inspect local_proposals/adaptive_experiment_specs; add persistent ask/tell for new campaigns.
- Planned Touch Files: ima/research_search.py, ima/optimizer.py, tests/test_research_search.py.
- Commit: feat(optimizer): add persistent recipe search.
- Tests: .venv/bin/python -m unittest tests.test_research_search tests.test_optimizer.
- Success Criteria: 100 fixture suggestions cover families, avoid duplicates and resume without test objectives.
- Checklist:
  - [ ] Preserve legacy profiles; version studies.
  - [ ] Seed diversity and persist sampler state/configuration.
  - [ ] Log completion/proposal order; asynchronous reproducibility is replay, not guaranteed identical timing.
- Handoff: study snapshot and fixed-event-order replay.

### Subphase 3.2: Durability and admission control
- Objective: inspect execute_proposals/wrapper and implement Search, Cost And Resource Policy.
- Planned Touch Files: ima/research_store.py, ima/research_resources.py, ima/optimizer.py, scripts/optimize.py, tests/test_research_store.py, tests/test_research_resources.py.
- Commit: feat(optimizer): checkpoint trials and resource admission.
- Tests: .venv/bin/python -m unittest tests.test_research_store tests.test_research_resources tests.test_optimizer tests.test_cortex_optimizer_worker.
- Success Criteria: kill/resume preserves completions; attempt IDs unique; measurements control admission.
- Checklist:
  - [ ] Add controller lock, leases, per-completion durability and outbox.
  - [ ] Own trial process groups; enforce timeouts/thread caps.
  - [ ] Simulate startup spike, memory pressure, stale workers and tracking outage.
- Handoff: interruption transcript and resource tests.

## Phase 4: Agent Feedback

### Subphase 4.1: Evidence-driven proposals
- Objective: extend OpenRouter with EvidenceBundle/ResearchProposal instead of available-ID-only selection.
- Planned Touch Files: ima/openrouter_orchestrator.py, ima/research_evidence.py, ima/research_specs.py, ima/optimizer.py, scripts/optimize.py, tests/test_agentic_planner.py.
- Commit: feat(optimizer): close agent feedback loop.
- Tests: .venv/bin/python -m unittest tests.test_agentic_planner tests.test_optimizer.
- Success Criteria: cycle two cites results and changes pipeline axis; no holdout export.
- Checklist:
  - [ ] Add agentic policy, keeping local/openrouter compatibility.
  - [ ] Persist hypotheses/parents/critique and exact evidence.
  - [ ] Two validation-repair attempts maximum, then explicit local fallback.
  - [ ] Implement model benchmark and cost/outage policy.
- Handoff: two-cycle replay and 20-case decision benchmark.

### Subphase 4.2: Provider contracts
- Objective: inspect current official docs; validate actual Flex tier and optional Batch lifecycle.
- Planned Touch Files: ima/openrouter_orchestrator.py, tests/test_openrouter_orchestrator.py, docs/research-provider.md.
- Commit: fix(orchestrator): reconcile tier and batch results.
- Tests: .venv/bin/python -m unittest tests.test_openrouter_orchestrator tests.test_agentic_planner.
- Success Criteria: create/poll/result IDs survive restart; mixed failures handled; paid smoke records actual cost/tier or blocker.
- Checklist:
  - [ ] Verify /v1/batches contract and provider capabilities.
  - [ ] Reconcile stable custom IDs; keep Batch off dependent training path.
  - [ ] No unsupported pricing/discount assumptions.
- Handoff: capability record, failure fixtures and live request ID if permitted/configured.

## Phase 5: Candidates And Tracking

### Subphase 5.1: Matched research cohort
- Objective: execute First Research Cohort using shared evaluator.
- Planned Touch Files: ima/modeling.py, ima/research_models.py, ima/research_specs.py, ima/research_targets.py, tests/test_research_models.py, tests/test_research_targets.py, docs/research-cohort-v2.md.
- Commit: feat(research): compare rich and market-conditional models.
- Tests: .venv/bin/python -m unittest tests.test_research_models tests.test_research_targets tests.test_modeling tests.test_research_evaluation.
- Success Criteria: zero offset equals baseline; analytic/finite-difference gradients agree; paired comparisons valid; secondary-target probes report target-specific baselines and diagnostics.
- Checklist:
  - [ ] Run six anchors and matched ablations.
  - [ ] Evaluate window/transform/market-conditional/offset variants.
  - [ ] Evaluate secondary targets without mixing them into the primary win-probability champion rule.
  - [ ] Record null results; optional ranker/ensemble admission requires dependency and evidence gates.
- Handoff: fold scores, paired losses, diagnostics and uncertainty.

### Subphase 5.2: Loadable MLflow lineage
- Objective: inspect log_experiment_run/consumers; package full candidate model.
- Planned Touch Files: ima/mlflow_tracking.py, ima/research_model_package.py, tests/test_mlflow_tracking.py, tests/test_research_model_package.py.
- Commit: feat(mlflow): package reproducible research models.
- Tests: .venv/bin/python -m unittest tests.test_mlflow_tracking tests.test_research_model_package.
- Success Criteria: loaded package matches direct probabilities; outbox retry creates no duplicate version.
- Checklist:
  - [ ] Preserve legacy records/consumers.
  - [ ] Log recipe/data/protocol/parent lineage and signature.
  - [ ] Reject incomplete races and verify remote package round trip.
- Handoff: run/version IDs, package hash and prediction readback.

## Phase 6: Acceptance And Server Canary

### Subphase 6.1: Terminal and adversarial tests
- Objective: run deterministic workflow and Armageddon Mode with new paths.
- Planned Touch Files: tests/test_agentic_optimizer_e2e.py, tests/fixtures/agentic_planner.json, scripts/optimize.py, docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md.
- Commit: test(optimizer): verify feedback and resume.
- Tests: .venv/bin/python -m unittest tests.test_agentic_optimizer_e2e; then .venv/bin/python -m unittest discover -s tests.
- Success Criteria: acceptance checks/legacy tests pass without unexplained failures.
- Checklist:
  - [ ] Demonstrate two cycles and resumed six-trial campaign.
  - [ ] Inject leakage/invalid recipe/tracking/worker failures.
  - [ ] Document exact run/status/stop/resume and budget configuration.
- Handoff: reproducible evidence bundle.

### Subphase 6.2: imaopt-only canary
- Objective: inspect wrapper guards and current headroom; prove detached execution.
- Planned Touch Files: scripts/cortex_optimizer_worker.py, tests/test_cortex_optimizer_worker.py, docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md.
- Commit: feat(worker): isolate agentic campaigns.
- Tests: .venv/bin/python -m unittest tests.test_cortex_optimizer_worker; record remote canary commands in runbook.
- Success Criteria: separate checkout/venv/new campaign; two real planner cycles or explicit API blocker; six real completed trials; package readback; laptop disconnect independence.
- Checklist:
  - [ ] Use /home/imaopt/research-v2 and distinct tmux session, pinned dependency transfer.
  - [ ] Baseline shared services read-only; no Cortex/solar restarts/config edits.
  - [ ] Start two workers, account for existing campaign load, ramp on observed headroom.
  - [ ] Restart only new canary controller to prove recovery.
  - [ ] Provide unlimited-run command with explicit spend/resource limits.
- Handoff: PID/session/revision, CPU/load/memory percentages, run/version IDs and unchanged-service readbacks. Rollback stops only new process group, preserving evidence.

## Optional Later Milestone: Generated Code
After recipe feedback proves useful, adopt AIDE-style draft/debug/improve plugin candidates. One worktree per candidate; pinned packages, read-only data/evaluator, no credentials/network and resource limits. A worktree is NOT a sandbox. Require container/equivalent isolation with filesystem/network/secret tests before executing generated code. Agent may implement fit/predict transforms/models, never targets/splits/scoring/controller. Preserve patches/parents regardless of outcome; no automatic merge. Write a separate implementation plan for this milestone.

## Execution Handoff
Read evidence companion, create dedicated branch, run preflight, then 1.1 and 1.2 before tuning. Execute atomic subphases with exact tests and evidence. A defensible null result is useful. Beating the already-searched 2.011806 does not demonstrate a new predictive edge.
