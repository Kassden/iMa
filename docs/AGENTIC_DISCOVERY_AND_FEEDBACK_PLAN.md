# Agentic Discovery, Experimental Memory, and Continuous Execution

## GOAL
- Extend research from tuning a fixed feature matrix to agent-directed feature creation, versioned dataset construction, cross-pipeline experiments, searchable hypothesis memory, and continuous resource-aware execution.
- Give the planner and MLflow the same truthful, comparable evidence. Preserve v4 unchanged until a separately authorized successor passes its canary.
- Status: execution authorized on 2026-10-01 by "execute the plan and start v5". Keep v4 unchanged through v5 validation, then perform the authorized safe successor handoff. Portfolio changes remain out of scope.
- Revision 2: deterministic discovery is a required first-class subsystem, not an optional LLM suggestion or an unspecified Featuretools benchmark. Sections below define its algorithms, contracts, defaults, and implementation work units.

## Acceptance Criteria
- A planner can define a previously absent historical feature, materialize it without future information, test it in two compatible model pipelines, and receive their outcomes in its next decision.
- With the LLM disabled, the same discovery specification enumerates the same feature IDs, builds the same past-only values, fits the same train-only selection masks, and emits the same ordered screening shortlist within declared numerical tolerances. This must work before agent integration.
- Every result links source snapshots, feature definition, dataset, protocol, hypothesis, program, attempt, model artifact, and decision IDs.
- Each training attempt is one MLflow run. Planner traces record hypotheses, evidence, decisions, actual USD cost, cycle bests, campaign bests, and new records per comparable objective.
- No successful mixed-objective cycle renders `best None=None`. Empty, failed, incomplete, and noncomparable results have explicit states.
- Planner budgets are chosen integers under an operator ceiling; neither the ceiling nor concurrency determines the requested budget. A ceiling of 260 and concurrency of 16 is valid.
- Finished workers accept eligible work from any approved program without waiting for a slower program to finish. CPU threads, RAM, fairness, and shared-host reservations constrain admission.
- Recovery produces one terminal outcome, one Optuna tell, and one linked MLflow run per attempt despite retries and tracking outages.
- Protected holdout is unavailable to planning, feature selection, budget allocation, and repeated confirmation.

## Scope and Relationship to Existing Plans
- This is the canonical plan for the newly discussed discovery/feedback/scheduling expansion. Read `docs/AGENTIC_FEATURE_DISCOVERY_PLAN.md` and `docs/AGENTIC_V4_RUNBOOK.md` for prior intent and deployment boundaries.
- Do not interpret older checked boxes as proof that a complete FeatureProgram builder exists. Reconcile each claim against code in Phase 1; preserve old documents and record the reconciliation.
- In scope: structured new feature definitions, source selection from validated local snapshots, fold-fitted transforms, diagnostic jobs, cross-pipeline reuse, hypothesis lifecycle, comparable metrics, observability, asynchronous admission, and successor canary.
- Out of scope: live betting, new race-day UI, automatic promotion, arbitrary generated Python in live workers, unapproved external sources, changing v4 in place, or restarting Cortex/Solar services.
- Keep the existing B/E1/E2/E3/E4 target/model mapping. Preserve the operator's 80% Benter / 20% experimental allocation until explicitly changed. Agent freedom is bounded by that policy, not permission to silently override it.

## Research
- Repository inspected: `ima/research_controller.py` uses `ProcessPoolExecutor` and `as_completed`; it fills submitted work within a batch but the controller awaits batch completion before planning again. `ima/research_models.py` implements native LightGBM/CatBoost. `ima/modeling.py` separates conditional race likelihood, calibration, and market blending.
- Existing registered transforms and schemas are useful building blocks; they do not constitute arbitrary new historical feature materialization.
- Benter paper: `research/1994-benter.pdf`. Fundamental feature development precedes race conditional logit, then out-of-sample market combination. Incremental information beyond market is distinct from standalone fundamental accuracy.
- Official sources checked on 2026-10-01:
  - Featuretools cutoff-time and training-window handling: https://docs.featuretools.com/en/stable/getting_started/handling_time.html
  - Ray Tune resource declarations and concurrency: https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html
  - Ray Tune lifecycle and resource admission: https://docs.ray.io/en/latest/tune/tutorials/tune-lifecycle.html
  - Ray Tune Optuna integration: https://docs.ray.io/en/latest/tune/index.html
  - MLflow run/dataset tracking: https://mlflow.org/docs/latest/ml/tracking/
  - MLflow trace inspection: https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui/
- These sources establish tooling capabilities, not evidence of predictive gains on our data. Benchmark before selecting optional dependencies.

## Root-Cause Baseline
- Proven from inspected code: generation is bounded by registered recipe operations; executor work completes through a batch-level controller boundary; fundamental-first selection and blended diagnostic metrics are different; target metrics have incompatible meanings.
- Historical verified observation from this conversation: mixed-objective previews displayed `best None=None`; its merged fix was intentionally not deployed into pinned v4. Reverify revision and API state before any future rollout.
- Likely hypotheses: narrow feature expressiveness and repeated development selection constrain useful discoveries; batch barriers can create idle tail time; limited evidence summaries obscure transferable findings.
- Possible, unproven: memory contention, slow tracking calls, straggler jobs, source gaps, or planning latency dominate throughput. Measure each before replacing execution infrastructure.
- Disproven as a general requirement: trial count must divide evenly by worker count; sixteen workers imply sixteen CPU threads; a stronger standalone fundamental score necessarily beats the market after blending.
- Missing evidence: current live utilization, queue timelines, exact source availability coverage, measured planner context completeness, and comparative Featuretools/Ray overhead. Do not infer them from this plan.
- Mutation boundary: planning edits only now. Later collect read-only baseline under `imaopt`; create successor datasets/releases outside pinned v4. No shared service changes.
- Mapping: lineage/build contract addresses feature expressiveness and leakage; shared evidence addresses siloing; typed summaries address observability; queue profiling/admission addresses barrier and resource hypotheses.

## SOTA, Standards, And Best Practices
- Implementation decision: extend existing mechanisms first; optional dependency adoption is gated by the benchmarks below.
- Reuse pandas/domain as-of logic, sklearn fold-fitted preprocessing, Optuna ask/tell, existing SQLite ledger/outbox, MLflow datasets/runs/traces, native model libraries, and model packages.
- Featuretools Deep Feature Synthesis (DFS) is the selected relational feature-generation engine, with a point-in-time adapter around it. Benchmarking is a correctness/resource gate, not permission to omit deterministic discovery. If its safe entity model fails the gate, document the exact incompatibility and review a replacement before proceeding; do not silently replace this phase with prompt-generated column names. Domain residual features still need reviewed implementations.
- Default scheduler: improve the existing single-host process executor with a long-lived resource-aware admission queue. Benchmark Ray Tune as an alternative; adopt only if its tested resource/recovery behavior justifies migration. One scheduler and one Optuna owner, never competing optimizer loops.
- Avoid a new workflow platform, separate graph database, or vector database initially. Structured ledger queries are sufficient for hypothesis retrieval; semantic retrieval is optional only after demonstrated need.
- Reject one universal scalar across targets, unbounded feature combinatorics, per-trial holdout access, arbitrary agent code execution, and ASHA pruning based on incomparable early chronological folds.
- Resource reservations are admission estimates, not enforced RAM guarantees. Use supported thread controls and approved user-service/cgroup limits where available; measure resident memory and fail safely.

## Deterministic Discovery Toolchain
There is no single universal industry-standard feature-discovery product. Use established tools for their specific responsibilities, with the domain-specific availability and race-evaluation adapter kept small.

| Responsibility | Selected Tool | Concrete Use | Boundary |
| --- | --- | --- | --- |
| Relational feature creation | Featuretools + Woodwork | DFS over horse/jockey/trainer history, explicit types and relationships, serialized feature definitions | Library cutoff semantics do not independently prove real publication availability |
| Constant/duplicate/redundancy filtering | Feature-engine | `DropConstantFeatures`, `DropDuplicateFeatures`, `SmartCorrelatedSelection` using missingness-based representative choice | Fit inside training folds; correlation groups are evidence, not automatic proof that a feature is useless |
| Supervised shortlist | sklearn | `mutual_info_classif` / `mutual_info_regression`, `SelectFromModel`, bounded `SequentialFeatureSelector` | Custom race scorer and explicit temporal splits; default random CV is prohibited |
| Importance and explanatory diagnostics | sklearn + native CatBoost | Group permutation/refit ablations; optional native SHAP summaries | Importance is not causality or a manual feature weight |
| Event-sequence features | tsfresh, optional extension | Small explicit calculator set over prior-start sequences | No wholesale calculator expansion; irregular sparse histories are not evenly sampled signals |
| Within-program search | Optuna | Selection strategy/window/transform and model tuning inside the approved search space | Same-budget baseline/challenger comparisons; no independent global feature planner |
| Execution and measurement | existing pool + psutil + threadpoolctl | Build/training reservations, thread limits, elapsed time and peak RSS | No hidden parallelism inside feature tools |

- Official sources: [DFS API](https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html), [primitive restrictions](https://docs.featuretools.com/en/stable/guides/specifying_primitive_options.html), [Feature-engine selectors](https://feature-engine.trainindata.com/en/latest/user_guide/selection/SmartCorrelatedSelection.html), [duplicate removal](https://www.feature-engine.trainindata.com/en/1.6.x/user_guide/selection/DropDuplicateFeatures.html), [model selection](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html), [sequential selection](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html), [correlated-feature importance caveat](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html), [tsfresh settings](https://tsfresh.readthedocs.io/en/stable/text/feature_extraction_settings.html).
- Required new feature extra: Featuretools and Feature-engine; resolve versions compatible with the existing pandas/numpy/Python environment in an isolated environment, save the exact resolved lock and `pip check` output. Optional separate extra: tsfresh. Do not adopt latest documentation signatures without checking the installed version.
- Do not add Feast merely for discovery: serving/online feature stores solve another problem. Do not add a broad AutoML framework that supplants our target contracts, folds, and ledger. Boruta-style shadow selection and symbolic regression are later experiments, not mandatory infrastructure.

## Deterministic Discovery Protocol
### D0: Freeze the Discovery Specification
- Inputs: source manifest, availability registry, fixed race protocol, candidate-generator version, generator configuration, baseline recipe IDs, target contracts, seed, and resource limits.
- Output: `DiscoverySpec` and canonical `discovery_id`. Canonicalize structured definitions, sort keys and feature IDs, hash content, record dependency versions and code revision. Names and timestamps alone cannot identify a feature.
- Enumerate without model labels using Featuretools `dfs(features_only=True)` with explicit aggregation/transform allowlists, allowed paths, ignored fields, and maximum depth. Disable default primitives, approximate cutoff bucketing, unrestricted categorical expansion, and nested Dask clusters.
- Proposed versioned starting limits: depth 2, 500 new canonical definitions per discovery, 64 definitions per build shard, 32 shortlisted new features per target, 12 candidate subsets per target, 8 temporal-selection fits concurrently only if resources allow. These are operator ceilings, not mandatory work counts or statistical truths.
- No random feature enumeration. If the ordered catalog exceeds the ceiling, emit deterministic shards with continuation IDs and deferred counts. Do not silently truncate the scientific record.

### D1: Build Safe Entity Tables
- Inspect actual canonical data before declaring a source usable. Construct stable tables: `horses`, `jockeys`, `trainers` (identity-only parent tables), `historical_starts` (one row per runner/start with result-derived measurements and availability), and `prediction_context` (runner IDs, prediction cutoff, race IDs, permitted pre-race attributes).
- Optional `workouts` and `veterinary_events` require observed-time/source audits; absence is reported, not fabricated. Mutable profile attributes belong in timestamped records, not unversioned parent tables.
- History columns such as prior result, speed, lengths beaten and prior odds become usable only after that start's results are available. Today's result, finishing time and final odds remain forbidden. Target-specific deny lists must permit legitimate prior results while excluding the current labels.
- Availability rule per history record: both event ordering and observed/publication availability must precede the current prediction cutoff. A post-event scrape time is not automatically the historical public availability time; use a documented publication rule only when defensible.
- Run separate keyed Featuretools passes for horse, jockey and trainer aggregates with per-entity/per-cutoff rows, then map those outputs back to unique current runner keys. Deduplicate identical entity/cutoff requests. Avoid paths from current runners through their current race to completed outcomes.
- If required by the supported Featuretools schema, construct a time-indexed visibility table using the vetted `available_at`; separately validate event-before-race conditions. Use `include_cutoff_time=False`, disable approximation, and prove semantics on same-day/late-observed fixtures.
- Current pre-race context is joined directly after availability validation; do not accidentally exclude it through a historical strict-cutoff rule. If only race dates exist, default to strictly earlier dates for historical outcomes; same-day earlier-race use needs validated race ordering and publication rules.

### D2: Enumerate and Materialize Candidate Families
- Relational grid: entities horse/jockey/trainer; history windows all-past, 90 days, 365 days; aggregations count/mean/std/min/max over allowed measurements. Exclude meaningless ID/string/date arithmetic. Finite grids, stable ordering, source capabilities and limits determine actual enumeration.
- Recent-start grid: previous 3/5/10 starts, using existing race-safe history helpers behind reviewed adapters. Emit explicit support counts and missing indicators; unsupported sequence length yields missing values, not fabricated zero performance.
- Domain grid: distance bands and matching surface/venue condition aggregates, race-peer rank/center differences, carried-weight change, and prior opponent-strength summaries. Distance-band definitions and direction conventions are versioned. Race-peer operations use current available attributes, never current outcomes.
- Transform grid: identity, signed-log1p for appropriate signed values, and train-fitted robust clipping/scaling where compatible. Inverse/square-root/log transforms require declared domains and fail-safe invalid-value handling. Shared cached matrices contain raw derived values; fitted transform state belongs to each fold/package.
- Benter-style adjusted-performance/residual family is a separately flagged learned family: auxiliary models fit on training history and produce chronological out-of-fold residuals for training observations; use only frozen past-fitted state for later rows. Compare against simpler unadjusted features and uncertainty/support-weighted shrinkage. Do not call existing simple means equivalent to Benter's residual construction.
- Initial tsfresh extension is disabled by default, but fully specified: past-start windows 5/10, sequence length/count, mean/std and linear trend on start index; separate elapsed-day feature. Require at least 3 observations for trend, leave missing otherwise. No FFT/autocorrelation claims from sparse irregular race dates. Enable only after an explicit matched feature-family trial and pinned calculator settings.
- Materialize shards with atomic temporary directories, file checksums, schema/type manifests, output runner-key hashes, feature definition IDs, coverage/support artifacts, and availability audit. A cache hit verifies content and lineage before reuse.

### D3: Deterministic Quality and Redundancy Gate
- Fit gates separately on every outer training fold. Never choose masks from calibration, development score, confirmation, or final holdout labels. Unsupervised coverage on later eras is descriptive evidence only.
- Gate order: availability/type validity; all-null and constant removal; exact duplicate removal; near-constant flags; train coverage; numeric correlation clusters. Reuse Feature-engine implementations instead of reimplementing their generic algorithms.
- Initial settings: exact constants dropped, missingness >95% quarantined, modal share >99.5% flagged, abs Spearman >=0.95 forms redundancy groups. Missing indicators can remain useful. Preserve the baseline columns for the control arm; produce an additional filtered-baseline arm if filtering is investigated.
- Within each correlation cluster choose an initial representative by highest training coverage, then simpler DAG, then canonical feature ID. Record every member so later target/model trials can reintroduce alternatives. Do not globally discard an interaction because one univariate association is weak.
- Output per-fold `quality-mask.json`, `redundancy-groups.json`, feature rejection codes, selector fit-race hash, and sparse-category diagnostics. Training-only gates can create different masks across folds; that is expected and must be visible.

### D4: Target-Aware Screening and Subset Search
- Derive inner chronological folds from each outer training partition: three ordered validation blocks, each preceded by its training history; keep races intact. Validate sufficient sizes; if impossible mark screening unsupported instead of reverting to random CV. Record the exact inner protocol hash.
- Use training-only mutual information as an inexpensive priority signal, not statistical proof. Horse-row p-values assume independence inadequately; do not use existing association/FDR flags as race-level acceptance gates. Odds uses regression screening; win/place classification; ranking prioritizes race-group-aware model evidence.
- Candidate masks per target: baseline; baseline plus each family bundle; baseline plus top 8/16/32 screened columns; embedded-model mask; and at most one forward-selection mask on the 12 highest-priority candidates. Deduplicate identical masks and respect the subset ceiling.
- `SelectFromModel` is a train-fitted proxy selector. Use actual baseline Benter plus boosted evaluation for win, LightGBM ranking for ranking, CatBoost classifier for placing, and CatBoost regressor for recorded odds when scoring final subsets. A proxy may prioritize work but cannot crown a target champion.
- `SequentialFeatureSelector` receives explicit inner split indices and a scorer/estimator adapter that preserves race IDs and race-weighted metrics. Never use sklearn default five-fold classification CV. Where metadata cannot be routed safely, use a thin tested adapter around library fitting/scoring, not a replacement feature search engine.
- Compare baseline/challengers with identical seeds, inner folds and fixed model configuration first; then a separate equal-budget hyperparameter comparison. This isolates feature gains from additional tuning effort.
- Screened masks are fold-specific. Shared feature definitions transfer across targets; target-selected masks do not transfer as globally proven features. Each receiving target reruns its selection on its own training folds.

### D5: Controlled Evaluation and Scientific Decision
- Evaluate shortlisted recipes on the unchanged outer development races. Report paired per-race losses/diagnostics and baseline-relative improvement with 2,000 fixed-seed bootstrap samples; add meeting/date-block sensitivity because races may be correlated. Intervals are descriptive after adaptive selection, not a cure for multiple testing.
- Report raw versus calibrated fundamental endpoints and blended market incremental value separately. Add family ablations and feature-group permutation on inner validation for selection; use development diagnostics only as explicitly recorded exploratory evidence for future decisions, never as a fitted selector for the same scored trial.
- Existing global row permutation can destroy coherent race-relative feature bundles. Use synchronized family perturbations with an explicit policy (within-race versus matched race/era blocks) and report what information the perturbation removes. Refit ablations remain the primary keep/drop evidence.
- Additive and drop-family trials need comparable same-budget controls. For nonlinear interactions, add the family jointly and record individual effects as secondary; no automatic rejection from univariate MI alone.
- Conclusion fields: estimated benefit, uncertainty, era stability, computation cost, source support, experimental selection exposure, and verdict keep/reject/inconclusive. Search-budget exhaustion or generator rejection is not a finding that the feature cannot work.
- Outer development findings guide future exploratory programs; repeated access is counted. Confirmation and final holdout stay behind their separate gates.

### D6: Publish Discovery Evidence and Let the Agent Direct the Next Search
- Deterministic discovery output exists without any LLM: catalog, materialized features, masks, screening rankings, matched evaluations, exclusions, deferred shards, and reproducibility manifest.
- Agent receives a bounded structured view: top target-specific candidates, family/model transfer evidence, coverage gaps, redundancy clusters, negative findings, costs, uncertainty, and links to details. It may request particular artifacts through audited retrieval rather than guessing from feature names.
- Agent can change source/entity/window/primitive/depth subsets within the operator envelope, propose composed domain definitions, request residual/sequence families, choose compatible pipelines, request ablations or diagnostic jobs, and set actual per-program budgets. All choices become versioned hypotheses.
- Deterministic code expands those choices into candidates, builds values, fits selectors, admits jobs, scores trials and records outcomes. LLM does not provide coefficients, alter fold membership, waive availability checks, or invent measured improvements.
- Fully depleted catalog triggers a planner decision to extend a permissible grid, investigate a source, transfer a feature, or pause with a reason. Never regenerate identical hashes indefinitely and call that learning.

## Discovery Artifacts and Data Layout
### Worked Research Program
This example is an intended contract fixture, not a report of existing features or measured improvements.

```json
{
  "schema_version": "discovery-v1",
  "source_manifest": "fixture-source-manifest.json",
  "protocol_id": "fixture-chronological-v1",
  "seed": 17,
  "generator": {
    "engine": "featuretools",
    "entities": ["horse", "jockey", "trainer"],
    "history_columns": ["prior_speed_mps", "prior_lengths_behind"],
    "day_windows": [90, 365],
    "aggregation_primitives": ["count", "mean", "std", "min", "max"],
    "transform_primitives": [],
    "max_depth": 2,
    "max_definitions": 500,
    "include_cutoff_time": false
  },
  "selection": {
    "missingness_limit": 0.95,
    "correlation_threshold": 0.95,
    "shortlist_sizes": [8, 16, 32],
    "inner_temporal_folds": 3,
    "max_subsets_per_target": 12
  },
  "targets": ["win_probability", "ranking_strength"],
  "resource_request": {"cpu_threads": 2, "ram_gib": 8}
}
```
- `prior_speed_mps` must be derived from validated historical distance and finish-time units, then attached to historical availability; it is not today's speed. If timing is absent/ambiguous, reject that measurement family and report support. `prior_lengths_behind` likewise refers only to past races.
- Deterministic phase emits candidates such as horse mean past speed over 90 days, its support count, and trainer spread of past lengths beaten. Validate actual values against the fixture before fitting selectors.
- Initial study: unchanged baseline versus baseline plus speed/support family at identical fixed parameters. For win use Benter first and boosted transfer; for ranking use a separate LightGBM comparison. These are three model programs, not one aggregate score.
- Subsequent planner may request distance-conditioned speed residuals if evidence suggests benefit, or reject/defer. Dispatcher preserves 80/20 allocation across training attempts. Shared feature-building/diagnostic costs are reported separately from portfolio trial counts.
- Predeclare target-specific success criteria: lower paired fundamental loss for win, higher paired NDCG@3 for ranking, stable era evidence and acceptable cost. Preserve separate conclusions if targets disagree.
- A request for 37 attempts persists budgets summing to 37 and dispatches under CPU/RAM/job caps. A ceiling of 500 feature definitions does not request 500 training attempts; definitions, build shards, screening fits, and training attempts have separate counts.

### Artifact Tree
```text
campaign/discovery/<discovery_id>/
  spec.json                 # frozen generator, source and resource choices
  source-manifest.json       # source hashes and audited availability registry
  entity-schema.json         # relationships, Woodwork types, forbidden paths
  catalog.jsonl              # canonical feature DAGs, engine version, shard IDs
  build/<matrix_id>/         # immutable column shards, checksums, key manifest
  folds/<fold_id>/
    inner-protocol.json
    quality-mask.json
    redundancy-groups.json
    screening.parquet
    candidate-masks.json
    fitted-selector.joblib
  comparisons/<comparison_id>/
    per-race.parquet
    paired-report.json
    ablation-report.json
  conclusion.json
  replay-manifest.json
```
- Featuretools definitions use the library's serialization plus canonical domain metadata; pickle alone is not an inspectable feature contract. Pin serialization versions and require replay/migration tests.
- Add `discovery_id`, `feature_program_id`, `selector_id`, `matrix_id`, `inner_protocol_id`, `comparison_id`, and `hypothesis_id` to corresponding MLflow runs, dataset inputs, trace metadata and packages. Large catalogs stay artifacts; table columns show counts, family, versions, source gaps and top evidence.
- Dataset composition is explicit: source/era/entity choices and allowable training windows form immutable candidate datasets. Different feature/source columns may be compared on matched labels/races; changed label availability or scored race population creates a different ComparisonKey and requires matched-intersection reporting.
- A dry-run CLI prints proposed feature counts, historical sources, expected column shards, denied paths, estimated resources and lineage before builds. A replay CLI verifies actual stored values and selector states, not merely matching names.

## Architecture and Ownership
```text
source manifests + availability rules
               -> FeatureProgram builder -> immutable matrix cache
EvidenceSnapshot -> planner -> HypothesisRecord -> ExperimentPrograms
ExperimentPrograms -> resource queue -> existing recipe executor
executor results -> ledger -> Optuna tell + MLflow outbox
ledger -> paired evaluations + diagnostics -> hypothesis conclusions
conclusions + champions + coverage -> next EvidenceSnapshot
```
- Ledger is the durable scientific truth; MLflow is its observable projection. Every planner input has a frozen evidence ID, sequence watermark, and schema version.
- Planner owns hypotheses, declarative features/transforms, compatible pipeline selection, search spaces, budgets, diagnostic requests, and keep/extend/reject recommendations.
- Optuna owns parameter suggestions within an approved program; it cannot change features, targets, sources, or protocol outside that program contract.
- Deterministic executor owns availability checks, data building, folds, numerical scoring, resource admission, retry classification, audit identity, and policy enforcement.
- Operator owns allowed sources, portfolio constraints, spending/resource ceilings, final evaluation, model promotion, and betting.

## Contracts
- `FeatureProgram`: schema/version, canonical ID, parent IDs, source hashes, entity keys, event-time and observed-time columns, prediction cutoff, join rules, feature DAG, windows/aggregations, missing-value policy, output schema, and builder revision.
- Creation primitives initially include lag, historical count/mean/std, elapsed time, recency-weighted aggregate, distance-conditioned history, pre-race peer difference/rank, safe arithmetic interaction, and residual feature from a fold-fitted auxiliary model. Bound DAG depth, output columns, build RAM, and runtime.
- Example novel feature: recency-weighted historical speed adjusted for carried weight and competition, followed by distance-specific residual preference. Auxiliary fitting must be nested in each training fold; no full-dataset fitted residuals.
- Distinguish deterministic past-only aggregations from learned transforms: learned residual encoders, imputers, scalers, selectors, and target encodings have per-fold fit state. Their cache keys include training race IDs, parameters, seed, and code revision.
- Missing observed timestamps require an audited source-specific availability rule; unsupported ambiguous inputs are rejected, not silently assigned event time.
- `HypothesisRecord`: ID, parent hypothesis, rationale/research references, evidence snapshot, expected mechanism, selected targets/models, feature program IDs, comparator, success threshold, uncertainty method, requested budget, costs, and append-only outcome events.
- Lifecycle: proposed -> validated -> queued -> running -> evaluated -> keep/reject/extend/inconclusive. Store failures separately from negative scientific evidence. Extensions create linked revisions.
- `ExperimentProgram`: hypothesis + feature program + pipeline + target/parameters + fixed protocol + search space + approved budget + resource profile. Cross-pipeline transfer creates another program linked to the same feature definition; no claim of benefit until retested.
- `ComparisonKey`: target kind/parameters, label version, metric version, score-race membership hash, fold protocol, weighting, and endpoint (fundamental/calibrated/blended). Dataset feature versions may differ if labels/races remain matched.
- `ResourceRequest`: reserved CPU threads, estimated peak RAM, time limit, model thread controls, priority, and build/training class. Resource profiles come from measurements; planner requests are clamped to operator policy.

## Comparable Results and Champion Rules
- Same ComparisonKey: preserve raw metrics, paired per-race differences, fixed-seed race bootstrap intervals, era/slice checks, and baseline IDs. Flag training-window differences as experimental treatments.
- Different targets: show relative baseline improvement, never treat equal percentages as equivalent achievements. For lower-is-better use `(baseline-candidate)/baseline`; for higher-is-better use `(candidate-baseline)/abs(baseline)`. Null/zero baselines produce unavailable, not invented scores.
- Show global target champion and family champion separately. Negative optimization forms such as negative NDCG must retain their raw direction and expose positive NDCG for interpretation.
- Win scoreboard has separate fundamental, calibrated-market, and blended endpoints. Place initially has uniform baseline; recorded odds has training-median baseline. Explicitly label absent market baselines. Add a placing market comparator only after a validated probability construction.
- Every new champion is provisional development evidence. Tiny improvements with intervals spanning zero are inconclusive. Do not promise log-loss improvement as a software acceptance test.
- Confirmation subset is separate from ordinary development and accessed only through a predeclared limited gate. Final holdout is operator-controlled and excluded from feedback. A consumed confirmation set is not subsequently called untouched.

## Configuration Semantics
- Retain `proposal_batch_size` as a legacy alias for proposal trial ceiling, introduce explicit `max_trials_per_decision`; reject conflicting values. Neither parameter sets a mandatory budget.
- `max_concurrent_trials` is a simultaneous job cap, initially preserve 16 pending measurements. Also enforce `cpu_thread_budget`, `ram_budget_gib`, `host_reserve_cpu_threads`, and `host_reserve_ram_gib` from audited host resources, not assumed 128 GB capacity alone.
- Planner returns per-program `trial_budget` whose sum is bounded by `max_trials_per_decision`. It may request 1, 19, 64, or another valid value, including nonmultiples of concurrency.
- `planning_checkpoint_completions`, `planning_checkpoint_seconds`, `queue_low_watermark`, and `max_inflight_programs` trigger bounded planning while work continues. Initial canary defaults: 16 completions, 300 seconds, queue watermark 8, and 8 active programs; tune from measured overhead.
- One planner call at a time, context sequence recorded, no duplicate decision emission. New trials use the newest available Optuna results at dispatch; do not pre-ask the entire budget.
- Preserve planner spend cap, failure budget, durable pause states, and explicit degraded-planning labels. Never silently substitute bootstrap proposals and call them agentic.
- Resource changes during a trial do not magically parallelize its algorithm. Request supported threads before dispatch; rescheduling/checkpointing is model-specific and out of initial scope.

## Dependency and Tooling Preflight
- No install is needed for planning. Real blockers for later setup include denied server authorization, unavailable SSH authentication, or incompatible pinned dependencies; record the exact failing command before claiming a blocker.
- Capture git revision/worktree, Python/package versions, installed MLflow client/server compatibility, ledger schema, dataset/protocol hashes, source manifests, service owner, server CPU topology, memory/disk, thread environment, and active shared workloads.
- Existing verification: `.venv/bin/python -m unittest discover -s tests`. Add focused unittest modules named below. Use project-local package management and pin compatible optional versions after benchmark, never install globally.
- Optional verification commands: `.venv/bin/python -m pip check`; inspect imports/versions for Featuretools, Ray, Optuna, sklearn, MLflow, pandas and native training libraries. Missing SSH authentication blocks server canary, not local implementation.
- Later implementation preflight commands: create isolated `.tmp/discovery-deps-venv`, resolve `featuretools`, `feature-engine`, and existing project research extras there, run `python -m pip check`, then save resolved exact versions and add a compatible `features` extra to `pyproject.toml`. Pin/lock optional tsfresh separately; do not install optional Ray or tsfresh just to pass an unrelated test.
- MLflow table verification needs a browser: reuse installed Playwright and Chromium; repair project-local dependencies/browser if absent. API tests alone do not prove visible columns render.
- No remote installs, service changes, credentials changes, or downloads are authorized by writing this plan. When later approved, use the established Tailscale network path for China-host dependencies.

## Deterministic Real-User Test
- Planned entry point: `.venv/bin/python scripts/run_discovery_canary.py --fixture tests/fixtures/discovery --output .tmp/discovery-canary --planner fixture --max-concurrent-trials 2`.
- Fixture planner creates a novel historical feature, proposes two compatible pipelines and a budget of five attempts, then retrieves a completed hypothesis and extends one program.
- Include a straggler and a short job. Prove the freed worker starts eligible work from another program before the straggler ends; verify reservations never exceed budgets.
- Before the agentic fixture test run deterministic discovery twice with the same spec and LLM disabled. Compare catalog IDs, numerical feature values, inner-fold IDs, mask IDs, shortlist order and reasons. Append a future race and repeat; all earlier as-of feature values remain unchanged.
- Append future outcomes and late-observed events; earlier feature rows must remain identical. Replay with identical seeds yields matching predictions within declared tolerances.
- Output must include feature definitions/manifests, evidence snapshots, hypotheses, queue timeline, predictions, paired results, and MLflow IDs. Fixture planner is explicitly not evidence of live LLM reasoning.

## Fulfillment and Readback Proof
- Requested final user-visible result: a replayable feature-discovery feedback cycle, searchable hypothesis records, comparable MLflow results, and verified cross-program worker replenishment. Required write/mutation operation for later execution: persist a new feature definition/matrix and hypothesis outcome, then read them back through the ledger and tracking APIs.
- Read newly built feature values, not only schema metadata. Verify their availability provenance and cross-pipeline program links.
- Read MLflow runs/dataset inputs and trace preview/cost via API, then inspect the actual traces table in Chromium with a screenshot. Do not hardcode a new cost column without testing installed-version UI support; if unsupported, document the limitation and supported visible USD preview.
- Read next planner input to prove it includes latest comparable champions, outcomes, evidence freshness, and transferable feature findings. A prompt template alone is insufficient.
- Recovery audit compares ledger attempts against Optuna states and MLflow run IDs; missing or duplicated side effects block completion.
- Successor canary requires actual planner output, a new feature build, two compatible pipelines, subsequent evidence-aware decision, and more than one completion checkpoint without local-computer dependence.

## Armageddon Mode
- Test future timestamps, unknown availability, same-day ordering, scratches, duplicate identities, non-single-winner races, missing finish positions, all-null/constant features, cyclic feature DAGs, explosive joins, unknown categories, and target leakage.
- Test stale evidence, incorrect metric direction, mixed objectives, new family versus global records, incomplete cycles, zero baselines, incomparable races, and selected artifacts differing from recorded hashes.
- Kill a worker/controller at each persist/tell/upload boundary; restart and prove idempotency. Simulate MLflow/OpenRouter outage, malformed planner JSON, exceeded spend, disk full, cache corruption, competing cache builders, and insufficient RAM.
- Scheduling tests cover slow jobs, failed builds, starvation, thread oversubscription, shared-host reservations, queue exhaustion, and stop/drain behavior. Report load as CPU-equivalent percentage with denominator plus actual measured utilization; never present load average as exact CPU usage.
- Must-fix: leakage, duplicate terminal/tell/run effects, misleading champions, unbounded resource admission, silent nonagentic fallback, or mutation of shared/pinned services.

## Generality Guardrail
- Existing mechanism: research controller, recipe executor, ledger/outbox, and Optuna search. Recurrence likelihood: every research decision; extend these shared owners rather than add one-off campaign scripts.
- Reuse the controller, ledger/outbox, recipe executor, search controller, evaluation, packages, and tracking. Do not introduce a separate planner/trainer loop that fails to exchange durable events.
- New owners only for recurring boundaries: feature program materialization, hypothesis/evidence projection, and resource admission. Keep scientific logic independent of scheduler backend.
- Avoid executing generated source code. New formulas are composed from validated declarative primitives; unsupported primitive requests are persisted for reviewed code extensions.

## Ordered State and Dashboard
- Canonical plan is this Markdown. Execution events: `.mega/state.jsonl`; verification evidence: `.mega/evidence.jsonl`.
- Generated planning dashboard: `.mega/dashboards/agentic-discovery-feedback.html`; it is a read view, not scientific experiment storage. For this planning revision use `.venv/bin/python .tmp/render_discovery_dashboard.py`: the wrapper includes the entire canonical Markdown because the stock Megaskill summary omits custom design sections. Use isolated `.mega/discovery-feedback/` state/evidence inputs until plan-scoped ledger filtering is proven; never inherit unrelated completion statuses from global phase numbers.
- Record baseline, subphase tests, benchmark decision, commits, canary readback, authorization, and rollout separately. Planned implementation checkboxes remain unchecked until proven.

## Regression Guardrails
- Branch strategy: dedicated implementation branch required before code changes; plan-only work remains on the existing branch and does not stage unrelated changes.
- Damage radius: systemic for implementation; current planning changes only this document and generated dashboard.
- Dedicated implementation branch: `feat/agentic-discovery-feedback` from verified current main. Preserve all unrelated dirty files. No branch switch, commit, push, or deployment required for this plan-only request.
- Execution surface clarification: `ima/research_v5.py` is the new controller integration module, `ima/research_scheduler.py` the admission owner, `ima/feature_discovery_specs.py` the spec owner, and `tests/test_research_discovery_state.py` their shared focused tests. These are authorized phase implementations; old `ima/ledger.py` references map to the actual existing `ima/research_store.py` unless explicitly discussing legacy code.
- Protected: pinned v4, raw historical files, shared host users/services, chronological folds, race-normalized endpoints, existing logs/model packages, operator portfolio, and manual betting/promotion.
- Consumers: planner, Optuna, ledger/recovery, native trainers, MLflow, terminal research commands, future inference packages.
- Atomic units and verification follow the phases below. No broad refactor; report scope expansion before edits.

## Phase 1: Baseline and Shared Evidence
### Subphase 1.1: Reconcile current capabilities
- Commit: `docs(research): record verified discovery and scheduling baseline`.
- Inspect: controller batch loop, `_execute_requests`, `_build_evidence`, ledger, search ask/tell, trace fix revision, live config and source availability under read-only access.
- Planned Touch Files: `docs/AGENTIC_DISCOVERY_BASELINE.md`, this plan.
- Tests: read-only capability inventory with file/line references; queue timing sample when server accessible.
- Success Criteria: proven versus hypothesized bottlenecks and actual versus historical configuration are separated; every older completion claim has an evidence verdict.
- Checklist:
  - [x] Capture baseline and protected identities; preserve pinned v4 code and shared services.

### Subphase 1.2: Typed comparison and evidence projection
- Commit: `feat(research): add comparable champion and evidence snapshots`.
- Planned Touch Files: `ima/research_evaluation.py`, `ima/research_evidence.py`, `ima/research_controller.py`, `tests/test_research_evidence.py`.
- Objective: implement ComparisonKey and one evidence projection from terminal ledger results, replacing duplicate champion logic only where necessary.
- Tests: `.venv/bin/python -m unittest tests.test_research_evidence tests.test_research_controller`; cover all five lanes, family/global champions, incomplete/failing records, metric directions and stale watermark.
- Success Criteria: deterministic snapshot identifies bests without mixing targets/endpoints/protocols and records freshness.
- Checklist:
  - [x] Produce coverage, hypothesis outcomes, baseline and champion views with artifact pointers rather than an unbounded prompt dump.

## Phase 2: Hypothesis Memory and MLflow
### Subphase 2.1: Durable experimental decision records
- Commit: `feat(research): persist searchable hypothesis lifecycle`.
- Planned Touch Files: `ima/ledger.py`, `ima/research_hypotheses.py`, `ima/research_specs.py`, `ima/openrouter_orchestrator.py`, `tests/test_research_hypotheses.py`, `tests/test_agentic_planner.py`.
- Objective: add versioned records/events and retrieval by feature mechanism, source, target, model, comparator and outcome. Migrate additively; legacy records stay readable.
- Tests: `.venv/bin/python -m unittest tests.test_research_hypotheses tests.test_agentic_planner`; duplicate proposals, extensions, failed versus rejected results, restart, malformed JSON, bounded retrieval.
- Success Criteria: second decision cites first hypothesis outcome and explicitly justifies repeat/extend/reject; unsupported features are recorded rather than silently ignored.
- Checklist:
  - [x] Validate lifecycle and hypotheses against frozen evidence; no secrets in prompts/artifacts.

### Subphase 2.2: Accurate runs, datasets, traces and cost
- Commit: `fix(mlflow): expose comparable records and decision lineage`.
- Planned Touch Files: `ima/mlflow_tracking.py`, `ima/research_controller.py`, `tests/test_mlflow_tracking.py`, `scripts/audit_discovery_tracking.py`.
- Objective: reuse merged mixed-objective fix; add typed best summaries, evidence/hypothesis links, dataset inputs, timing/queue/resource parameters, and planner cost fields supported by installed MLflow.
- Tests: `.venv/bin/python -m unittest tests.test_mlflow_tracking`; isolated local tracking server readback and Playwright trace-table screenshot.
- Success Criteria: one attempt equals one training run; parent program records are tagged administrative; new-record preview is explicit; unknown USD cost is unavailable rather than zero.
- Checklist:
  - [x] Separate cost of planner calls from CPU/runtime estimates; preserve compatible champion projection and correct checkpoint summaries.
  - [x] Do not deploy this into pinned v4 or rewrite historical metrics.

## Phase 3: Feature Creation and Dataset Materialization
Phase 3 is now mandatory deterministic discovery work, not one generic builder task. Execute 3.0 through 3.5 in order before integrating an agent-created feature into a campaign.

### Subphase 3.0: Dependency Lock and Discovery Contract
- Commit: `feat(features): define deterministic discovery specs and tooling extra`.
- Inspect: `pyproject.toml`, `ima/research_specs.py`, `ima/feature_analysis.py`, existing historical schemas and installed Python/pandas/numpy versions. Do not change runtime dependencies globally.
- Planned Touch Files: `pyproject.toml`, `ima/feature_discovery_specs.py`, `config/feature_discovery_defaults.json`, `tests/test_feature_discovery_specs.py`, project-compatible resolved dependency lock.
- Objective: model DiscoverySpec, candidate DAGs, supported primitive/type/domain contracts, selector recipe, ceilings, artifact schema, canonical IDs and invalid-operation reasons. Require an explicit versioned configuration matching D0-D6.
- Tests: `.venv/bin/python -m unittest tests.test_feature_discovery_specs`; isolated dependency imports/`pip check`, stable canonical hashes, unknown primitive/window/source rejection, cyclic DAGs, contradictory cutoffs and oversized grids.
- Success Criteria: a frozen valid spec can enumerate expected operation counts without labels or an LLM. Dependencies resolve within supported Python range; code contracts are machine-readable.
- Checklist:
  - [x] Record exact compatible dependency versions, benchmark gates and expected fixture catalog IDs.
  - [x] Include source/target deny lists and fully explicit resource/selection configuration.

### Subphase 3.1a: EntitySet and Availability Adapter
- Commit: `feat(features): build point-in-time Featuretools entity sets`.
- Planned Touch Files: `ima/feature_sources.py`, `ima/historical_sources.py`, `ima/feature_program.py`, `tests/test_feature_sources.py`, `tests/fixtures/discovery/`.
- Objective: implement D1 using supported Woodwork types and Featuretools APIs; enforce event and observed availability before constructing historical feature inputs.
- Fixture: two horses across four dates, repeated jockey/trainer, two races on one date, a late-published earlier result, a mutable profile update, and a debut runner. Keep expected values explicit.
- Tests: `.venv/bin/python -m unittest tests.test_feature_sources`; current-result exclusion, previous-result inclusion, same-day conservative behavior, late publication, identity collisions, entity cutoff mapping and missing availability.
- Success Criteria: manual expected counts/means equal Featuretools outputs; adding late/future rows leaves prior predictions unchanged; direct current pre-race attributes remain available.
- Checklist:
  - [x] Persist audited entity schema and source availability rules; block unsupported sources with structured reasons.

### Subphase 3.1b: Deterministic DFS and Domain Catalog
- Commit: `feat(features): enumerate reproducible DFS and domain candidates`.
- Planned Touch Files: `ima/feature_generators.py`, `ima/feature_program.py`, `ima/rich_features.py`, `scripts/build_feature_program.py`, `tests/test_feature_generators.py`.
- Objective: implement D0/D2 and bounded stable enumeration. Reuse Featuretools DFS and serialization for generic history, existing domain logic for race-specific semantics, and no ad hoc generic DFS replacement.
- Tests: `.venv/bin/python -m unittest tests.test_feature_generators tests.test_feature_program`; catalog replay, library serialization, duplicate DAG elimination, paths/types, deterministic shard continuation, actual novel feature values and train-window variants.
- Success Criteria: two repeated label-free runs produce identical candidate IDs and definitions; novel raw features are materialized and cached safely. The library benchmark proves correctness before corpus-scale execution.
- Checklist:
  - [ ] Measure Featuretools build wall time/peak RSS on fixture, 10,000 representative runner cutoffs and full candidate count; schedule sample builds under the same resource limits.
  - [x] Preserve strict semantics: no approximate cutoff or hand-written generic DFS fallback; sample gate passed, full gate remains explicit.

### Subphase 3.1c: Declarative builder and time-safe cache
- Commit: `feat(features): materialize versioned agent-defined history features`.
- Planned Touch Files: `ima/feature_program.py`, `ima/rich_features.py`, `ima/historical_sources.py`, `ima/research_specs.py`, `scripts/build_feature_program.py`, `tests/test_feature_program.py`.
- Objective: implement FeatureProgram contract and bounded DAG primitives; content-address source/build state; atomic cache writes and readback manifests.
- Tests: `.venv/bin/python -m unittest tests.test_feature_program tests.test_rich_features tests.test_historical_sources`; future invariance, observation-time cutoff, concurrent builds, reproducibility and all-null warnings.
- Success Criteria: a feature absent from the original matrix is actually computed with per-column provenance, not merely renamed or logged as metadata.
- Checklist:
  - [x] Integrate the required Featuretools adapter after its correctness gate; document supported versus domain-specific primitives.
  - [x] Cache deterministic history separately from per-fold learned feature state.

### Subphase 3.2: Learned transforms and cross-pipeline transfer
- Commit: `feat(research): reuse feature programs across compatible pipelines`.
- Planned Touch Files: `ima/research_transforms.py`, `ima/research_executor.py`, `ima/research_model_package.py`, `ima/research_targets.py`, `tests/test_research_executor.py`, `tests/test_research_model_package.py`.
- Objective: integrate feature IDs/datasets and fold-fitted residual/selection state into existing executor/packages; validate compatibility for all five lanes.
- Tests: `.venv/bin/python -m unittest tests.test_research_executor tests.test_research_model_package`; no fitted state outside training, inference replay, target-column exclusion, shared feature tested in two models.
- Success Criteria: transferable definitions preserve identity but each pipeline has its own fitted state/results; ranking scores are never silently treated as probabilities.
- Checklist:
  - [x] Add fixed-model controls and optional race-preserving permutation/learning-curve diagnostics; model coefficients remain learned weights.

### Subphase 3.3: Fold-Local Quality Gates and Feature Screening
- Commit: `feat(features): screen generated candidates within training folds`.
- Planned Touch Files: `ima/feature_screening.py`, `ima/feature_analysis.py`, `ima/research_evaluation.py`, `tests/test_feature_screening.py`.
- Objective: D3/D4 with Feature-engine filters, sklearn selectors and target-aware race scorer adapters. Extend the existing analysis module instead of treating its univariate win-only report as sufficient for all targets.
- Tests: `.venv/bin/python -m unittest tests.test_feature_screening`; train-only masks, exact duplicates including nulls, correlated informative groups, sparse-but-useful features, seeded MI, differing target shortlists, valid temporal inner splits and race-ID preservation.
- Success Criteria: all masks/state carry fit-race hashes; mutating outer score labels does not change the fitted selector for that trial; no generic row-random CV or all-target winner mask exists.
- Checklist:
  - [x] Cap sequential selection to declared shortlisted candidates and charge all screening fits to resource/runtime accounting.
  - [x] Keep baseline control columns unchanged, preserve rejected/deferred feature definitions, and expose reasons per fold.

### Subphase 3.4: Paired Feature Studies and Diagnostic Feedback
- Commit: `feat(research): evaluate feature families with matched controls`.
- Planned Touch Files: `ima/feature_studies.py`, `ima/feature_analysis.py`, `ima/research_evaluation.py`, `ima/research_evidence.py`, `tests/test_feature_studies.py`.
- Objective: D5, fixed-model and equal-budget tuned comparisons, family ablations, race-preserving perturbations, meeting-block sensitivity, explicit statistical selection exposure and target-specific feedback.
- Tests: `.venv/bin/python -m unittest tests.test_feature_studies tests.test_research_evidence`; identical-model zero-delta control, injected useful/noise family, direction checks, paired membership, cohort changes and no holdout access.
- Success Criteria: a machine-readable conclusion links feature candidates to actual matched results and uncertainty; transfer recommendations are hypotheses, not assumed improvements.
- Checklist:
  - [x] Record per-race predictions, failures, costs, correlations and inconclusive results; log no profit claim from feature screening.

### Subphase 3.5: Deterministic CLI Replay and Agent Discovery Interface
- Commit: `feat(agent): direct deterministic feature discovery from evidence`.
- Planned Touch Files: `scripts/discover_features.py`, `ima/openrouter_orchestrator.py`, `ima/research_controller.py`, `ima/research_hypotheses.py`, `ima/mlflow_tracking.py`, `tests/test_feature_discovery_cli.py`, `tests/test_agentic_planner.py`.
- Planned Commands: `.venv/bin/python scripts/discover_features.py enumerate --spec config/feature_discovery_defaults.json --output .tmp/discovery`; `materialize --discovery .tmp/discovery`; `screen --discovery .tmp/discovery --protocol <fixture-protocol>`; `report --discovery .tmp/discovery`; `replay --discovery .tmp/discovery`. Implement these exact subcommands unless repository CLI inspection identifies a conflicting public contract, then document the replacement before coding.
- Objective: D6, end-to-end deterministic execution with LLM disabled, then validated agent requests for extending grids, residual families, source composition, target/model transfer and budgets.
- Tests: `.venv/bin/python -m unittest tests.test_feature_discovery_cli tests.test_agentic_planner`; frozen fixture two-run readback, future-data invariance, exhausted catalog, duplicate request, unsupported primitive and actual second-decision retrieval.
- Success Criteria: deterministic discovery is independently usable from the terminal and agent orchestration consumes its catalog/results. Unsupported definitions are persisted for review, not silently approximated.
- Checklist:
  - [x] Expose generator/tool versions, candidate/rejected/deferred counts and feature family/source evidence in MLflow and the next planner input.

## Phase 4: Continuous Scheduling and Adaptive Decisions
### Subphase 4.1: Measured asynchronous admission
- Commit: `feat(research): replenish resource-aware work across programs`.
- Planned Touch Files: `ima/research_scheduler.py`, `ima/research_controller.py`, `ima/research_search.py`, `ima/research_executor.py`, `tests/test_research_scheduler.py`.
- Objective: long-lived queue with one control-plane ledger writer; dispatch Optuna ask just in time; interleave approved programs using weighted fair scheduling plus age promotion within portfolio policy.
- Tests: `.venv/bin/python -m unittest tests.test_research_scheduler tests.test_research_controller`; injected-clock stragglers, cross-program dispatch, bounded reservations, thread control, failure/restart and resource pause.
- Success Criteria: queue timelines prove no cycle-wide straggler barrier when eligible work exists; actual concurrency and CPU/RAM reservations obey policy.
- Checklist:
  - [ ] Profile pool versus isolated Ray Tune backend; preserve single ask/tell owner. Record choice and measured overhead.
  - [x] Build jobs share the same resource budget; do not allow a dataset build to overlap unrestricted trainers.

### Subphase 4.2: Evidence checkpoints and planner freedom
- Commit: `feat(agent): plan new programs from continuous research feedback`.
- Planned Touch Files: `ima/research_controller.py`, `ima/openrouter_orchestrator.py`, `ima/research_specs.py`, `scripts/optimize.py`, `tests/test_agentic_planner.py`, `tests/test_research_controller.py`.
- Objective: checkpoint planner calls while approved programs continue; allow selected targets, compatible pipelines, diagnostics, features, and varied budgets under operator constraints.
- Tests: focused planner/controller suites; budgets 1/19/64/260 with concurrency 16, empty queue, exhausted program, delayed planner, spend outage, stop/drain, portfolio counts and stale decisions.
- Success Criteria: planner chooses a nonmandatory budget and next action based on a completed result; no fixed bootstrap masquerading as agentic decisions.
- Checklist:
  - [x] Preserve 80/20 allocation with cumulative accounting and rounding; no hardcoded experimental-lane trial minimum tied to worker count.
  - [x] Persist pending/inflight work before dispatch; stop issuing new work immediately on stop and drain safely.

## Phase 5: Scientific Validation and Successor Readiness
### Subphase 5.1: End-to-end canary and adversarial recovery
- Commit: `test(research): prove discovery feedback and recovery end to end`.
- Planned Touch Files: `scripts/run_discovery_canary.py`, `tests/fixtures/discovery/`, `tests/test_discovery_canary.py`, `docs/AGENTIC_DISCOVERY_VALIDATION.md`.
- Tests: deterministic real-user command above; `.venv/bin/python -m unittest discover -s tests`; Armageddon cases; tracking/browser readback; paired baseline report with protected confirmation/holdout excluded.
- Success Criteria: feature creation -> two pipelines -> conclusion -> next evidence-aware decision is replayable; failures do not duplicate scientific outcomes.
- Checklist:
  - [x] Report actual metric outcomes and confidence, even if no improvement; software success is not predictive success.

### Subphase 5.2: Isolated server canary and authorized handoff
- Commit: `docs(ops): document isolated discovery campaign and safe handoff`.
- Planned Touch Files: `config/agentic_discovery_canary.json`, `docs/AGENTIC_DISCOVERY_RUNBOOK.md`, dedicated successor launcher under `deploy/systemd/` only if later authorized.
- Objective: inspect current imaopt layout, stage separate pinned revision/dataset/experiment, and canary within capacity left by v4. Never assume enough spare RAM for simultaneous campaigns.
- Tests: source/revision identity, genuine LLM proposal, completed feature build/two pipelines, next checkpoint, MLflow table and API, shared-service before/after checks, and graceful rollback.
- Success Criteria: server completes repeated feedback without the Mac online. If access/capacity is unavailable, report blocked server verification without claiming deployment.
- Checklist:
  - [x] Obtain rollout authorization: user explicitly requested execution and v5 launch on 2026-10-01, superseding planning-only instructions.
  - [ ] After authorization only, drain v4 at a safe boundary, verify zero workers, start successor, and preserve rollback artifacts. Do not run two unrestricted campaigns.

## Planning Deliverable Readback
- Verify this plan with Megaskill plan checker and generate the read-only dashboard.
- Execution is now authorized. Checked items have implementation/test evidence; rollout and full-build gates remain open until live verification.

## Execution Mapping (2026-10-01)

Actual ownership follows existing repo boundaries instead of introducing every
illustrative file name above. `feature_program.py` owns entity adapters, Featuretools
DFS serialization, sequence/domain definitions and checksummed raw caches;
`feature_residuals.py` owns expanding chronological auxiliary fits;
`feature_screening.py` owns Feature-engine/sklearn fold-local selection;
`feature_studies.py` owns paired scientific comparisons. `research_hypotheses.py`
owns comparison-key champions and SQLite hypothesis memory. `research_v5.py` uses
existing controller/ledger/search/executor/tracking APIs as a single autonomous
controller, not an independent competing loop.

New tests are consolidated in `test_feature_program.py`, `test_discovery_feedback.py`
and `test_research_discovery_state.py`, plus existing executor, controller and
tracking suites. Planned illustrative test/file names are not evidence that those
files exist. CLI replacement is explicit `python -m scripts.discover_features`
with `--dataset`, `--spec`, `--output`; `screen` additionally requires explicit
`--train-races`, `--target`, `--label`. This matches the repo's immutable-input
conventions and avoids implicitly selecting on the full dataset.

Supported feature creation is compositional and typed, not arbitrary Python or
unregistered source invention. Catalog shards contain 64 definition IDs; selection
and residual fitting are separate from raw matrix identity. Optional tsfresh stays
disabled. The existing single-host spawn pool remains the production backend;
isolated Ray compatibility/overhead evidence is required, not a presumed throughput
benefit. Learning curves freeze prior selected feature/transform state and are
explicitly model-training diagnostics rather than independent selection validation.

Proof locations: `docs/AGENTIC_DISCOVERY_BASELINE.md`,
`docs/AGENTIC_DISCOVERY_VALIDATION.md`, `docs/AGENTIC_DISCOVERY_RUNBOOK.md`,
`.mega/discovery-feedback/`, and preserved canary/benchmark artifacts under the
dedicated user's campaign directories. No historical metrics or pinned v4 source
are rewritten; the authorized v4 STOP request is an operational handoff only.
