# Agentic Research Expansion Plan

Status: EXECUTION AUTHORIZED 2026-10-04. User requests the full implementation, a fresh dataset-cleanliness pass, resource efficiency and an isolated V6 campaign. Prepared 2026-10-03; expanded 2026-10-04. Pinned predecessors and other server users/services remain protected; implementation gates below must pass before V6 launch.
Canonical source: this Markdown. Dashboard is a generated read view, not a separate plan.
Dependencies: `HKJC_OFFICIAL_CORPUS_PLAN.md`, `AGENTIC_DISCOVERY_AND_FEEDBACK_PLAN.md`, and `AGENTIC_QUEUE_AND_MEMORY_REPAIR_PLAN.md`. Their existence is not proof of implementation.

## GOAL
Create an auditable discovery loop where the orchestrator creates features and formulas, requests validated successor datasets, composes multi-model pipelines, allocates research independently of worker capacity, and explores paper betting EV, with reproducible comparisons and truthful MLflow records.

## Scope And Approval Boundary
- Planning was delivered first. The 2026-10-04 execution request authorizes implementing all phases and launching V6 after their gates; unchecked items remain outstanding rather than being implicitly complete.
- Future implementation covers official event integration, dataset lifecycle, open feature creation, selection/resource management, composed pipelines, controller evidence, nomenclature and paper EV.
- Out of scope: real-money execution, HKJC credentials, a new application UI, scraping bypasses, unofficial facts substituted for HKJC evidence, unrelated cleanup, other server users/services and guaranteed profit.
- Preserve the 80% Benter / 20% experimental allocation unless explicitly changed. A composed predictive graph is experimental if it has an experimental predictive ancestor; a Benter-only chain stays Benter. Track outer trials, physical component fits and resource consumption separately.
- Do not automatically assume V6 or rename V5. Choose an approved successor ID at rollout; never hotpatch its predecessor's pinned code/data identity.

## Acceptance Criteria
- [ ] Workouts, barrier trials, veterinary/clearance and movement events have exact identity/provenance, temporal classifications and tested historical joins; predictors populate wherever evidence supports them.
- [ ] Unknown/incomplete historical coverage is missing, not zero. Zero event counts require affirmative coverage for that source/entity/interval.
- [ ] Planner DatasetRequests automatically build, validate and atomically publish candidates; failures never replace verified data.
- [ ] Full established history is preserved by default. Original champion replay and common-population comparisons separate real model changes from dataset/window changes.
- [ ] Selection has no hidden 32/12/6 ceilings; integer or all-eligible budgets work with method-aware resource admission and visible actual counts.
- [ ] Deterministic Featuretools discovery and planner-created formulas coexist in a reusable feature registry, beyond the current measurement literals.
- [ ] Planner-created typed model DAGs execute with chronological cross-fitting, calibration/blending, component provenance and reproducible inference packages.
- [ ] Historical race, timed workout and timed barrier-trial speed are available as distinct point-in-time feature families, with training-only condition adjustment, support/uncertainty indicators and no current-race outcome leakage.
- [ ] The existing speed target is activated through an explicit compatible experimental policy, and a conference-inspired homoscedastic/heteroscedastic performance model learns distributions and derives joint race outcomes. Original logit/boosted controls remain available.
- [ ] Ranking-to-win adapters reuse existing race softmax and chronological temperature calibration; time/speed/distribution adapters expose their assumptions and cannot claim a uniquely inferred joint distribution from marginal probabilities.
- [ ] Research decisions continue while workers are occupied; budgets reconcile and eligible tasks refill without execution-wave barriers.
- [ ] MLflow clearly separates actual trial runs, component fits, paid decisions, execution snapshots, datasets and model versions; table USD cost reconciles without duplicate charging.
- [ ] Planner and trace summaries use the same current per-target comparable champion evidence, not mixed-objective best=None or stale scores.
- [ ] Terminal paper research supports win/place/quinella/trio probabilities and EV/fair-price reporting; quote, currency, stake, payout and official-rule semantics are explicit.
- [ ] Missing pre-race quotes produce fair-price-only or explicitly ex-post research, never fabricated historical executable EV.
- [ ] Benter/Thorp-informed paper stake sizing separates expected profit from expected log-bankroll growth, evaluates fractional Kelly and jointly accounts for overlapping tickets, uncertainty, pool impact and official wager increments.
- [ ] Full-history tests, adversarial/restart drills, model replay and measured performance gates pass before approved isolated successor rollout.
- [ ] A second dataset-quality audit classifies whole races and variables by identity, timing, units, outcome consistency, leakage, source coverage and era/source bias. Cleaner subsets may exclude dirty races with explicit exclusions; wider variable availability never excuses fabricated measurements or target leakage.

## Root-Cause Baseline
This planning turn inspected local source, not current server health. Future Phase 1 must refresh deployed evidence before remediation.

Proven local restrictions:
- F1: `ima/feature_discovery_specs.py` has max_selected<=32, max_definitions<=500, depth/window restrictions and a fixed measurement vocabulary. No inspected benchmark establishes 32 as optimal.
- F2: `ima/feature_screening.py` additionally shortlists 32 for embedded selection, 12 for sequential selection and selects at most half of the sequential shortlist. Raising only one validator leaves restrictions intact.
- F3: `scripts/build_official_dataset.py` retains supplementary events but reports events_used_in_historical_features=0 and excludes publication-unverified historical events. Parser records already distinguish event date, publication and capture evidence.
- F4: `ima/research_v5.py` requires immutable dataset/protocol paths and verifies campaign identity. New data cannot safely be swapped into a running identity.
- F5: `_plan` requests min(5, budget, available program slots); planning admission depends on execution-program capacity. Research creation and execution admission are coupled.
- F6: `PipelineRecipe` is single-model with registered stages, not a planner-composed multi-estimator DAG.
- F7: trace naming centers on cycles, while asynchronous programs finish later. Legacy scalar-best fields can be null for mixed objectives. Recent preview improvements exist, so deployed behavior needs verification before claiming a fix is missing.
- F8: `ima/research_targets.py` supports adjusted_finish_time_or_speed as physical speed=distance/finish_time, and secondary regression execution exists. V4/V5 portfolio contracts in `ima/research_controller.py` do not allocate a time/speed target. Existence of executor support is not evidence of a live speed program.
- F9: ranking-to-win conversion already exists in `ima/research_executor.py`: within-race softmax, followed by TemperatureCalibrator fitted on chronological calibration races. Reuse and test it instead of describing the adapter as wholly absent.
- F10: historical speed ratios, trial speed and optional OOF-adjusted speed residual features already exist in `ima/rich_features.py` and `ima/feature_residuals.py`. Expand their source coverage and verify temporal/unit semantics; do not recreate their owner or equate a historical feature with an active future-speed prediction target.

Likely hypotheses: H1 event predictors add signal but coverage/timing may dominate; H2 wide private frames/repeated preparation constrain parallelism; H3 stale summary/outbox watermarks can mislead planning; H4 sufficiently diverse ensembles may improve joint probabilities, but pooling similarly wrong models may not.
Possible causes: mutable historical pages lack publication proof; duplicate observations are not necessarily duplicate physical events; quote history may be absent for exotic pools.
Disproven explanations: Featuretools inherently needs a 32-feature limit; 260 trials must be a multiple of concurrency; win marginals uniquely determine trio probability.
Missing evidence: fresh release/config/ledger/trace exports, availability/coverage by event family, common-population champion replay, matched memory/I/O benchmark and official quote/dividend coverage.
Mutation boundary: plan artifacts only now; future baseline probes read-only; future builders publish immutable candidates; deploy only to approved imaopt paths/services.
Remediation mapping: Phase 2 -> F3/F10/H1; Phase 3 -> F4; Phase 4 -> F1/F2; Phase 5 -> F6/F8/F9/H4; Phase 6 -> F5/F7/H2/H3; Phase 7 -> joint-outcome/quote gaps; Phase 8 -> verification. Sandbox/retention/recovery work is risk reduction, not proof of the original cause.
Not-done: simply raising one ceiling, blanket event-date assumptions called verified, naming-only trace fixes, absent replay or local tests called live deployment.

### Execution Incident: OpenRouter Delivery Receipts (2026-10-04)
- Observed: the isolated 52a actual-data planner call hit its 120-second deadline without a completion, generation ID or cost receipt. Its cost is unknown, not zero; that invocation remains frozen and cannot restart.
- Proven: subsequent noncharged models GETs succeed on default, IPv4 and IPv6 transports through the verified Tailscale route; the approved model exists. A failed POST had no phase telemetry, so connection/TLS versus generation waiting cannot be determined retrospectively. Concurrent V5 usage and absence of a pre-call credit baseline prevent exact attribution from account totals.
- Remediation: use the existing HTTPX bounded transport in streaming-body mode to persist safe phase events and `X-Generation-Id` response headers before buffering the full JSON body. Do not introduce provider pins, host routing changes, guessed costs or automatic paid retries.
- Primary references: [OpenRouter streaming/header contract](https://github.com/OpenRouterTeam/docs/blob/main/api_reference/streaming.mdx) and [generation usage lookup](https://openrouter.ai/docs/api/api-reference/generations/get-generation). The lookup requires a known generation ID; it cannot invent the identifier lost by the previous call.
- Planned Touch Files: `ima/openrouter_transport.py` (new), `ima/openrouter_orchestrator.py`, `ima/research_expansion.py`, `tests/test_openrouter_transport.py` (new), `tests/test_research_expansion_resources.py`.
- Tests: actual localhost HTTP headers followed by delayed-body timeout, successful JSON and cost callback, HTTP failure, scalar JSON rejection, durable receipt integration, and the complete regression suite. No paid calls are required for these tests.
- Commit: `fix(planner): retain early transport and generation receipts`.
- Remaining evidence: reconcile the original provider charge via its generation ID or actual billing record; a fresh successful planner/training gate is still required for live rollout.
- New C1 requalification evidence: the authorized isolated invocation reached TCP at 0.235 seconds and TLS at 3.965 seconds, spent approximately 87.32 seconds in request-body transmission, then failed with ReadError at 91.300 seconds before any response headers. Evidence JSON was 207,352 bytes; that is not yet a measured wire-body size. No generation ID, completion or confirmed USD cost was received. The new invocation is frozen under the requalification rule; do not retry it or label unknown spend as zero.
- New hypothesis, not established cause: request-upload/backpressure or the Tailscale exit/upstream path may be responsible. A successful small GET does not prove a full POST works. Only bounded noncharged synthetic-payload diagnostics, exact local wire-size reconstruction and read-only routing/MTU checks are allowed next; do not disclose research evidence to a public echo endpoint, mutate routing/security, or send another model request to diagnose transport.
- Gate status: C1 full suite/CI, exact Linux fixture and five actual official-data controls with production MLflow readback passed. Paid planner acceptance and continuous V6 remain blocked; V5 stays active and unchanged.

### Memo Delivery and Compact Request Qualification (2026-10-04)
- Updated state: V5 was stopped under explicit user authorization; its interrupted ledger rows remain evidence, not completions. The bounded Luna session completed five fits. Neither proves an unattended OpenRouter campaign.
- Root cause evidence: the earlier 202,674-byte synthetic POST failed before headers without billing. Small GET success did not qualify uploads. Fresh nonbilling probes at 64 KiB passed three times; this does not qualify a larger final request.
- Planned Touch Files: `ima/research_memo.py`, `ima/research_planner_context.py`, `ima/openrouter_orchestrator.py`, `ima/research_expansion.py`, `tests/test_research_memo.py`, `tests/test_research_planner_context.py`, `tests/test_research_expansion_resources.py`.
- Deliver the full committed `docs/V4_V5_RESEARCH_HANDOFF_TO_V6.md` in every V6 evidence snapshot. Hash its exact bytes into evidence identity; require OpenRouter decisions to acknowledge `research_memo_sha256`. Acknowledgment proves delivered version, not human-like comprehension; inspect hypothesis reasoning as well.
- Use reversible columnar predictor metadata and deduplicate schemas only when their definitions are identical. Preserve all predictor eligibility, safety, units and missing-field semantics. Keep full original evidence on disk and in telemetry.
- Tests: exact catalog/schema round trips; unchanged original bundle; full memo delivery; missing/mismatched memo acknowledgment rejection; existing planner, resource and controller regressions.
- Paid gate: measure exact HTTPX body after final release/config/evidence pinning, then qualify that exact byte size with bounded nonbilling synthetic uploads. Do not send research content to diagnostic endpoints. Another pre-header failure freezes paid launch regardless of successful local tests.
- Launch gate: one bounded paid qualification must yield actual response/cost, memo acknowledgment, nonempty executable programs and a durable validated queue. Verify actual training and MLflow readbacks before claiming continuous service; no idle empty-seed launch or paid retry after unknown cost.
- Commit: `fix(planner): deliver pinned research memo with compact evidence`.

### Lane Readiness Repair (2026-10-04)
- Proven live cause: with two completed Benter trials, rounded 80/20 dispatch requires experimental next. Benter reference capacity remains 18, but both preparation slots are occupied by other Benter programs. Experimental readiness is absent, so the fit candidate list is empty before resource admission; empty admission blockers do not mean runnable work exists.
- Preserve dispatch ratios, actual allocations, recipes, chronological protocols and resource ceilings. Prioritize preparation coverage of both lanes, including the next required lane; choose inexpensive initial references before costly discovery. Once lanes are covered, retain fairness so large research programs are not starved. Exhausted or retired ready programs cannot count as usable coverage.
- Planned Touch Files: `ima/research_expansion.py`, `tests/test_research_expansion_resources.py`, `ima/openrouter_orchestrator.py`, `tests/test_research_memo.py`, `config/agentic_research_expansion.json`.
- Tests: two startup slots cover cheap Benter and experimental; restart with Benter covered prioritizes experimental; exhausted ready capacity does not cover a lane; covered-lane fairness remains; independent preparation/fit/resource limits are unchanged.
- Operational evidence: 16,000 total completion tokens caused reasoning exhaustion and truncated JSON. The existing model publicly supports 32,768; use that allowance with medium reasoning, concise output instructions, explicit top-level required identity, extra-column declaration guidance and node-level graph output contracts. Do not reduce trial/program freedom to fix serialization.
- Rollout: immutable scientific identity prohibits hotpatching the pinned release. Preserve its two completed/registered models, original responses, fee receipts and operator-recovery history. Stop only the owned successor predecessor when no fits/uploads/tells are outstanding; use a new pinned V6 campaign with previous V6 results as read-only references plus the full V4/V5 memo.
- Acceptance: exact final-body nonbilling qualification, fresh complete paid decision with memo acknowledgement and reported cost, executable cheap references for both lanes, advancing new training attempts across both lanes, READY models and delivered MLflow traces, effective 24-CPU/100-GB ceilings and no OOM. An active service or preparation-only stall is not success.

## Research
- Built-in options inspected: official parser/snapshot builder, rich historical features, selection, pinned V5 controller, recipe contracts and MLflow tracking; findings are enumerated in the Root-Cause Baseline.
- Off-the-shelf choices: Featuretools/feature-engine for discovery and screening, sklearn/Optuna for fitting/search, Joblib/NumPy for shared arrays, and MLflow/OpenTelemetry conventions for provenance and traces.
- Official standards and domain rules are linked below. New-framework adoption must solve a measured gap; no source establishes that more features or ensembles automatically improve this dataset.

## SOTA, Standards, And Best Practices
Sources checked 2026-10-03; installed SDK versions must be pinned and verified, not inferred from latest documentation.

| Source | Design decision |
|---|---|
| [Featuretools time handling](https://docs.featuretools.com/en/stable/getting_started/handling_time.html), [DFS](https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html) | Reuse EntitySets/time indexes, explicit cutoff inclusivity, training windows, seeds and feature definitions. Library supports unlimited feature generation via max_features=-1; measured admission still bounds physical work. |
| [Featuretools custom primitives](https://docs.featuretools.com/en/latest/getting_started/primitives.html) | Compile validated new measurements into typed transform/aggregation primitives; do not rewrite DFS. |
| [NumExpr expression evaluation](https://numexpr.readthedocs.io/en/latest/user_guide.html) | Optional vectorized expression backend after validation with explicit bindings. Not a sandbox. Use NumPy first when adequate. |
| [Joblib read-only memory maps](https://joblib.readthedocs.io/en/stable/auto_examples/parallel_memmap.html) | Share numeric buffers and cache fold preparation. Retain existing executor/ledger; do not add Ray/Dask without demonstrated distributed need. |
| [Scikit-learn stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) | Reuse estimator APIs, but forward race-grouped OOF artifacts rather than default random/stratified CV or in-sample prefit stacking. |
| [CatBoost uncertainty](https://catboost.ai/docs/en/references/uncertainty) | Existing estimator family offers RMSEWithUncertainty; validate its installed output conventions and use it for conditional performance distributions, not as a guarantee of calibrated joint outcomes. |
| [NGBoost](https://proceedings.mlr.press/v119/duan20a.html) | Optional mature distributional-boosting challenger. Core path uses existing tools first; adding NGBoost requires dependency locking and a measured comparison. |
| [Train, Probit](https://eml.berkeley.edu/books/choice2nd/Ch05_p97-133.pdf), [SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) | Gaussian latent-performance probabilities require explicit identification/covariance and numerical integration or simulation. Reuse SciPy numerical kernels and seeded, convergence-tested draws rather than hand-rolling numerical optimization/random generators. |
| [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [trace FAQ](https://www.mlflow.org/docs/latest/genai/tracing/faq/), [trace UI](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui) | Separate runs/models/input datasets/traces. Use installed supported log_input, metadata/tags and native cost fields, verified in the UI. |
| [OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/), [semantic conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/) | Stable operation names and links for asynchronous work. Racing ima.* attributes are our domain extension, not an industry standard. No new collector required. |
| [HKJC local pools](https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/), [betting rules](https://special.hkjc.com/e-win/en-US/betting-info/racing/betting-rules/) | Pool-specific qualification, payout, scratching and dead-heat rules; no blanket 18% subtraction. Verify effective rules at execution. |

Current official guide payout shares: 82.5% win/place/quinella/quinella-place, 77% trio, 80.5% forecast and 75% tierce. Pool simulation must use applicable local/simulcast rules, not blindly reuse this table.
Local PDFs inspected for the Kelly expansion on 2026-10-03:
- `research/1994-benter.pdf`: Wagering Strategy and Kelly Betting and Pool Size Limitations, printed pages 190-192, particularly single-bet equation (5) on page 191; discusses fractional Kelly, simultaneous pools and reduction of dividends by the bettor's stake. Its reported operation excludes place/show, so our place research is an extension, not a literal replication of its operating strategy.
- `research/KellyCriterion2007.pdf`: Edward O. Thorp, The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market; section 6 on simultaneous bets and section 7.3 on fractional Kelly. The simultaneous-bet discussion requires a full joint distribution, not covariance alone; the fractional discussion addresses estimation uncertainty and overbetting.
- `research/Chance constrained optimization for parimutuel horse race betting.pdf`: Deza, Huang and Metel, arXiv:1503.06535, 2015; finite-horizon chance constraints and payout estimation are optional later research, not a core mixed-integer solver dependency or a promise of HKJC profitability.
- Primary origin: [Kelly (1956), A New Interpretation of Information Rate](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03809.x). Distinguish its expected logarithmic growth criterion from maximizing one-race expected dollars.
Record exact page/equation and assumption mappings in Phase 1. Do not call generic classifier blending or independent place products full Benter replication. Begin ordered-outcome research with a tested Harville/Plackett-Luce baseline, with conditional-choice assumptions and higher-place bias disclosed.
Conference source: `research/benter_what_are_my_odds_conference_transcript.txt`, especially sections 12-16 and Q&A 19.3. It is a user-supplied, edited transcript, not verified audio/slides. It supports researching different performance-distribution widths and winner-likelihood fitting, but does not provide Benter's exact variance/covariance estimator or prove his deployed target was recorded finish time. Call the new family conference-inspired probit, not an exact replication of his proprietary system. The transcript's unresolved professor name must not become an invented algorithm citation.
Rejected approaches: planner eval/exec/import, global supervised synthesis, random CV, independent products of place marginals, final odds/dividends in fundamental inputs, takeout twice, final-dividend-informed ticket selection, hidden caps, campaign identity rewriting and a decorative new UI.

## Dependency and Tooling Preflight
- Inspect `pyproject.toml`, `requirements-research.lock`, `requirements-v5-overlay.lock` and local/remote manifests. Python 3.11-3.13; existing pandas/NumPy/SciPy, sklearn, Optuna, Featuretools/feature-engine/Woodwork, Joblib, Pydantic, psutil and MLflow integrations come first.
- Capture git status/base SHA, actual release SHA, campaign identity, outbox/ledger watermarks, own-unit limits and free disk. Never print API keys or full secret environments.
- Install or repair only needed project-local tooling. Candidate command: `.venv/bin/python -m pip install -e '.[research,features,acquisition,dev]'`, followed by repo lock/version procedure; no incidental blanket upgrades. Add/lock NumExpr only after measured benefit/compatibility.
- Test tooling: unittest, current canaries/benchmarks, PDF extraction, MLflow API, Playwright/Chromium for existing MLflow UI. If selected Python Playwright tooling is absent, install it and run `.venv/bin/python -m playwright install chromium`.
- Server downloads/dependency transfer use approved Tailscale routing, not direct China access. Only imaopt-owned paths/cgroups/services may change.
- Real blockers: missing approval/credentials, host offline, disk exhaustion, incompatible locks, unavailable historical publication evidence or pool quotes. A family-specific data gap must not stop unrelated research.
- No PySR/Julia installation required now. LLM expression proposals plus deterministic DFS meet initial creation requirements. Optional symbolic search later needs a separately validated dependency/resource proposal.

## Architecture And Ownership
```text
Official raw -> observations -> canonical events + availability + coverage
                                       |
Planner DatasetRequest -> immutable builder -> validator -> dataset registry
                                       |                       |
                         historical feature joins        approved successor
                                       |
Deterministic DFS + planner formulas -> reusable feature definition registry
                                       |
                    training-only selection -> shared compact fold artifacts
                                       |
Planner PipelineGraph -> DAG validation -> forward OOF fits -> pooling/calibration
                                       |
                 win/place/ranking/time outputs -> joint order -> paper EV
                                       |
          ledger + Optuna + hypothesis memory -> refreshed evidence -> planner
                        |
            MLflow runs/models/datasets + linked decision/execution traces
```
- Acquisition owns raw/parser evidence, not experiment decisions.
- Dataset registry owns immutable build/validation/promotion; planner requests typed specifications, not file edits.
- Feature registry owns definitions/catalogs reusable across compatible pipelines.
- Executor owns deterministic fit/replay; Optuna tunes parameters inside a planner-chosen hypothesis.
- Controller owns durable decisions/budgets/evidence; scheduler owns admission, not research direction.
- Ledger is authoritative. MLflow is a reconciled observable projection, not another work queue.

## Contract A: Events, Availability And Usable Predictors
Canonical schema: `event_id, physical_event_group_id, source_observation_id, family, horse_id, occurred_at/date, published_at, first_seen_at, valid_from/to, available_at, availability_basis, identity_status, source_url, source_body_sha256, parser_version, typed_values, correction_of, coverage_id`.
- Preserve source observations separately from physical-event equivalence groups. Identical same-day text does not prove one physical workout.
- Exact full horse ID or documented official confirmation only. Name-only movement PDF rows require independently validated official race/saddle mapping, including date and standby context; never fuzzy-join names.
- Availability tiers: verified_point_in_time; observed_prospective; assumed_retrospective. Conflicting/unknown rows stay in evidence, not strict feature matrices.
- Event date is not publication time. Capture of a mutable old page establishes first-known prospective use, not historical publication.
- Strict joins require both occurrence and availability before the race cutoff. Store Asia/Hong_Kong local race dates separately from UTC instants; date-only records use disclosed conservative next-day eligibility. Same-day use requires reliable timestamp evidence.
- Preserve revision history and select the version knowable at cutoff; modern corrections cannot invisibly rewrite historical features.
- Allow a separate assumed-retrospective research dataset with explicit family-specific publication lag assumptions and sensitivity runs. Never mix its results with verified point-in-time champions or present them as race-day-valid historical evidence. This enables exploration without false certainty.
- Coverage contract records family/source/entity-or-population/interval, known complete/incomplete/unknown, collector watermark, missing pages and identity gaps. Global event min/max dates do not prove horse-level completeness. Unknown counts remain NaN; verified complete intervals with no events may yield zero.

| Family | Initial predictors | Restrictions |
|---|---|---|
| Past races | last/rolling/recency-weighted physical speed and speed residuals; same-distance/surface/venue histories; sectional speed when segment length/time are explicitly known; consistency/support | strictly previous races; current finish time/speed only a target/outcome, not a predictor; winner-relative ratios labeled separately |
| Workouts | counts 7/14/30/90d; recency; distance/type/surface summaries; intensity when distance/time exists; deviation from own history | no invented numerical intensity from vague prose; units and unknown timing retained |
| Trials | counts 90/180/365d; recency; individual speed/time; placing percentile; cohort-relative time; distance/surface match; change from previous trials | batch/winner time is not individual horse time; missing/noncomparable timing stays missing |
| Vet/clearance | category counts 30/90/365d; incident/clearance recency; condition state as known at cutoff; official-text category map | future clearance cannot close past state; no unsupported diagnoses or severity |
| Movement | arrival/return recency; completed stay lengths; known days at location by cutoff; travel/rest interactions | future departure/end dates forbidden; truncate ongoing stays at cutoff |
| Interactions | trial recency x distance match; workouts x layoff; clearance x rest; location x venue | dependency provenance, compatible units and train-only learned parameters |

Each output has missingness/coverage companions, resolved/available/usable counts, unique values, era span and exclusion reasons. Raw collection completeness is not predictor completeness.
Race, workout and trial speed have separate source/context semantics: racing near competition intensity, training under chosen effort and trials under another protocol must not be pooled as interchangeable samples. The model may learn their relationship using explicit source/type/condition variables and availability indicators.

## Contract B: Automatic Dataset Lifecycle
DatasetRequest: `request_id, parent_dataset_id, raw_corpus_manifest_id, event_policy_id, race_population_spec, history_window, target_contracts, feature_definition_ids, protocol_id, evaluation_population_id, expected_preserved_keys, rationale, budget, evidence_watermark`.
States: requested -> building -> validating -> verified/rejected -> promotion_requested -> approved -> successor_started. Resume idempotently after crashes.
- Publish immutable runners/events/coverage Parquet, feature artifacts, manifest/lineage/exclusions/temporal audit/validation report under content-addressed paths. Atomic manifest publication; no partial latest pointer.
- Default all established history. Trailing-three-years is an explicit hypothesis with exclusion counts, not a builder shortcut.
- Manifest records ordered row hashes, units, schema/definitions, code/environment/parser/source IDs, temporal tiers, target eligibility, true training/calibration/evaluation dates and protocol IDs.
- Compare champions by target/parameters/metric, evaluation race population, dates and probability basis. Dataset hashes may differ when comparing features on identical score rows; training-data changes must still be disclosed. One hash equality is not the entire comparability test.
- Gate base-key preservation or explicit audited exclusions, exact identity, label/eligibility correctness, chronology, coverage and checksums. The execution request permits a smaller cleaner dataset: exclude malformed whole races with recorded reasons, never selectively drop losing horses or filter on outcome/available workouts. Preserve the original raw corpus and every excluded key; compare era/venue/class/distance/field-size distributions before and after. Missing optional predictors alone are not grounds to exclude a race. Base-only candidates may be valid but cannot claim success integrating events with zero usable events.
- Reproduce reference champions on original data first. Refit on new data and score a fixed common race set; report original reproduction, changed-data/common-population comparison and expanded-population coverage separately.
- Freeze final confirmation races inaccessible to planner selection. Repeated development search can overfit; paired block bootstrap by race/meeting and forward confirmation quantify uncertainty. Do not call every development improvement significant.
- Automatically build/validate. The 2026-10-04 execution request authorizes an isolated V6 successor after the gates below, not silently changing the pinned V5 dataset or stopping another user's service. Subsequent automatic promotions require a recorded operator policy covering availability tiers, audited population changes, evaluation contract, validated replay and resource limits.

## Contract C: Feature Creation And Selection
Two coordinated discovery sources:
1. Deterministic Featuretools catalog over normalized tables, seed expressions, time windows and custom typed primitives; content-equivalent deduplication and training-fold screening.
2. Planner FeatureDefinitions: novel formulas/conditional aggregates/recency functions and source choices, with hypotheses, dependencies, scope, units, missing policy and resource estimates. Not limited to speed_mps/beaten_lengths/carried_weight literals.

FeatureDefinition: `definition_id, schema_version, name, expression_ast, input_refs, source_families, entity_key, time_scope, cutoff_rule, dtype/unit, missing_policy, fit_required, producer, parent_definition_ids, hypothesis_id, validation_report_id`.
- Examples: trial speed minus own prior mean divided by training-safe dispersion; exponentially decayed workouts; known clearance-rest interaction; weather/surface interaction only if official source exists. A new name cannot create nonexistent raw measurements.
- Typed declarative AST validated by Pydantic before lowering to NumPy/NumExpr/Featuretools. No raw Python eval/exec, attribute access, imports, filesystem/network, comprehension or dynamic lookup. Explicit symbol bindings; backend is not the sandbox.
- Arbitrary numeric composition within typed safe semantics is supported, not unrestricted executable code. New operator implementations need isolated reviewed adapter tests; no LLM-generated Python mutates the running release. Optional code-plugin research is a separate later scope.
- Validate dependency closure/cycles, label and temporal taint, units, numeric domains, shape, NaN/inf, overflow and duplicate semantics. Learned normalizers/residuals/weights and supervised formula discovery fit training rows only.
- Safe division/log/clipping behavior is part of the definition. Unknown coverage must not become zero through arithmetic.
- Expand configurable windows/depth/candidate budgets with measured generation admission; remove undisclosed validator ceilings. Rejection/defer reasons stay visible.
- SelectionSpec: `method, requested_feature_budget` (positive integer or null=all eligible), `candidate_shortlist_budget` (integer or null), `fit_time_budget_seconds, memory_budget_gib, proxy_estimator_spec, chronological_inner_protocol, seed`.
- All selectors obey explicit budgets; no embedded32/sequential12/half-shortlist rule. Sequential selection estimates candidate x step x fold cost, then admits/defers/rejects honestly. A request for 64 never silently means six.
- Planner may choose 8/64/128/all eligible, not mandatory larger counts. Resource policy bounds work; more features still need validation. Initial compatibility config can explicitly reproduce legacy16, but never hide a global32 limit.
- Logs distinguish base predictor count, generated candidate count, requested budget, eligible count and actual retained count.
- Cache keys include source/definition/availability IDs, ordered training rows/labels, target, fold boundaries, selector/proxy/transform settings, seed and versions. Learned feature weights live in fitted estimator state, not human-picked constants disguised as training.

## Contract D: Planner-Created Multi-Model Graphs
PipelineGraph: `graph_id, schema_version, nodes, edges, dataset_id, target_contract_id, evaluation_contract_id, output_contract, component_refs, search_spaces, fit_budget, hypothesis_id, evidence_watermark`.
Initial nodes: feature_view, transform, estimator, forward_oof_predict, race_normalize, calibrate, performance_distribution, probabilistic_adapter, weighted_probability_pool, log_probability_pool, meta_estimator, market_blend, rank_distribution, evaluate.
- DAG validation enforces acyclicity, row/target/unit compatibility, explicit fit populations and one declared evaluation output. Scores, win/place probabilities, elapsed seconds, odds and joint distributions are distinct types.
- Initial graph families: Benter -> chronological calibration; Benter+boosted pools; calibrated boosted+ranker stack; heterogeneous feature views -> ensemble; fundamental -> separate market blend; win -> joint order; speed/time distribution -> finish-order simulation.
- Planner chooses constituents, feature views, stage ordering, combination/calibration, target, search space and trial budget. Optuna tunes the declared graph; changing its hypothesis requires a new decision.
- Reference model reuse requires valid training cutoff and input/evaluation contracts. A future-trained champion cannot emit historical OOF predictions; refit the recorded recipe instead.
- Prediction artifacts include race/horse keys, fold/training cutoff, raw/calibrated output, component/graph/features IDs and masks. Train meta-models/calibrators on forward OOF or a disjoint past calibration window, then evaluate later races.
- Initial OOF rows without prior fit are explicitly excluded; do not force expanding-window splits through incompatible cross_val_predict partitions. Deeper stacks need nested chronological cross-fitting or separate safe fit windows at every learned level.
- Fundamental inputs exclude market. Market blend has separate output/metric and quote-availability contract. Within-race win normalization does not make independent place marginals a coherent ranking distribution.
- Existing ranking softmax/temperature conversion is a supported adapter, not a missing capability. Normalization creates a probability vector; chronological calibration assesses its frequency meaning. A learned rank score scale is not automatically a horse-specific variance estimate.
- Reuse compatible prediction/feature artifacts across families. Compare ensemble to constituents/equal-weight controls and eligible market baseline on identical races, with error-correlation/diversity reports.
- A concrete graph trial is one outer MLflow trial run; component/fold fits may be child runs with run_role and parent IDs. They are not extra outer completed trials. Count retries and physical fits separately.
- Registry identity is target/pipeline family; immutable model versions package every node/selector/transformer/calibrator/output contract. Scoped champion aliases must include comparison contract. Retain replayable bests and dependencies; retention of losing components is explicit, never overwriting versions.

## Contract E: Research Freedom And Resources
- Independent knobs: max_trials_per_decision (legacy operator ceiling260), max_new_programs_per_decision, max_pending_programs, max_active_preparations, max_concurrent_fits. None determines the others or requires multiples of worker count.
- PlannerDecision records chosen budget, new allocations, extensions/retirements, dataset/formula/graph requests, unallocated remainder with reason, continuing work assessment and evidence watermark. chosen=allocated_new+allocated_extensions+unallocated exactly.
- Full fit queues do not block research proposal/review. Backlog safety limits produce explicit backpressure, but review/retirement remains available. One intentional proposal is valid; do not force diversity by count.
- A single-trial complex graph is valid. Its many component fits reserve real resources; account physical fit/CPU/memory costs separately from outer budgets and enforce portfolio policy transparently.
- Refill eligible prepared work across programs on every completion, not after all jobs in a wave finish. Preserve lane fairness/aging, including large jobs/preparation; no permanent starvation.
- Implement/reuse queue-memory repair contracts: prepare once, immutable numeric memmaps, training-fold cache, early compact columns, descriptors not pickled wide frames. Audit what exists before creating duplicate owners.
- Measure controller/shared data/builder residency separately from private fit workspaces; memmap pages still consume RAM. Stage/family estimates use conservative cold defaults and measured peaks including OOM samples; monitor PSS/USS, cgroup memory, disk and native thread counts.
- Inspect actual current own-unit limits. Adopt approved near-100 GB process-group policy with explicit GB versus GiB conversion and measured headroom, not 100 GiB by accident. Other services retain their reservation; worker budget is not the cgroup cap.
- Adjust caps progressively after at least30s sustained headroom and representative full folds; do not reflexively halve or force26. Pause admission near critical limits; never kill other users' jobs to increase utilization.
- Expose CPU busy percent separately from load/available logical CPUs percent, memory/cgroup peaks, preparing/pending/running counts, cache hits, throughput and precise admission blockers.
- Job cost estimation: a versioned JobEstimate keys stage, model/graph family, rows, generated/selected features, dtype/sparsity, categorical cardinality, folds, native threads, estimator search settings and cached artifact IDs. Conservative cold estimates precede measured wall time/private-memory p95 plus margin, including failures. Log estimate/actual/error; worker-lifetime ru_maxrss is not a fresh per-job peak.
- Share arrays through Joblib uncompressed dump/load(mmap_mode="r") or NumPy memmap; use Joblib.Memory only for pure, completely keyed preparation. Object arrays are not shared numeric buffers. Locks/atomic manifests protect concurrent misses; no nested Joblib worker pools.
- Reserve shared data once and private preprocessing/fit/simulation workspaces per job; page-cache residency still affects headroom. Fit identical training-fold selection/transforms once. Estimator-only parameter changes may reuse preparation only when proven irrelevant to its cache key.
- Drop unused columns early, persist compact fold arrays and retain categorical vocabulary in manifests; avoid repeated fancy-index/DataFrame copies. Cold generation, warm cache and model-specific threads have distinct cost/admission cases.
- Backfill feasible jobs continuously with fair aging/reservation for large jobs. Expose remaining-work estimates and preparation-versus-fit bottlenecks; scheduling follows measured resources rather than promises of higher utilization.
- Efficiency gate: repeated equal-data/protocol/concurrency cold/warm benchmarks, throughput, private/cgroup peaks, preparation counts, cache hit/wait/miss, disk/page faults and replay parity. A smaller secretly reduced dataset is not evidence of memory optimization.

### Resource Implementation And Measurement Contract
1. Preparation owner: `ima/research_preparation.py` publishes uncompressed Joblib numeric arrays with read-only mappings and an atomic manifest. Persist row keys, feature order, dtype, category vocabulary, shapes and checksums. Workers receive artifact descriptors rather than an entire wide DataFrame. Keep the existing executor; Joblib supplies storage/cache mechanics, not a competing scheduling loop.
2. Cache identity: source content hash, ordered training/calibration/score keys, target labels, availability policy, feature definitions, selection settings, fitted-transform settings, fold dates, seed, implementation revision and dependency versions. Never share fitted objects across different training folds or targets. Estimator hyperparameters can be omitted only when that estimator does not participate in feature selection or preprocessing.
3. Concurrent misses: use the existing file-lock mechanism or a bounded cross-process lock, build into a temporary directory, validate, then publish atomically. A waiting consumer records a cache wait instead of recomputing. Incomplete/corrupt artifacts are rejected and rebuilt under the lock. No untrusted Joblib pickle is accepted. Pin artifacts while jobs use them; eviction cannot delete a running job's inputs.
4. `JobEstimate` owner: `ima/research_resources.py` stores versioned workload fingerprints and measured samples in the campaign ledger. Distinguish feature generation, selection, fit, graph component and simulation. Estimate CPU threads, shared bytes, private peak bytes, preparation/fit wall time and disk bytes. Include native library thread settings and any dense categorical expansion. Use conservative cold estimates, then a high quantile plus margin from representative successful and failed jobs; do not extrapolate linearly from a tiny canary without validation.
5. Measurement: sample the current job's process tree, using USS/PSS where supported and RSS as an explicitly labelled fallback; track cgroup current/peak as the host admission boundary. Retain OOM/censored estimates as lower bounds. A reusable worker's lifetime peak is not the current trial's peak. Record predicted versus measured cost and estimator confidence/sample count.
6. Admission: separately charge controller/preparation residency, shared artifacts once, and private workspaces for every fit. Enforce both configured budget and real available/cgroup headroom. Allocate model-specific threads before the job starts; a model supporting parallel fit can receive multiple threads when beneficial. Do not treat one trial as permanently one core, or assume that adding threads always accelerates it.
7. Continuous dispatch: when a worker finishes, release its reservations and backfill the next feasible ready task across programs immediately. No requirement that chosen trial budget divide by concurrency. Age deferred large jobs and reserve headroom after a bounded wait to prevent starvation. Discovery proposals, pending budget and running workers remain independently bounded.
8. Capacity policy: the user permits up to **100 GB decimal = 93.13 GiB** for IMA on the 128 GB host. Verify host units and the own-user cgroup before setting it. Subtract measured controller/shared/preparation residency and explicit emergency headroom from the fit budget. Probe several caps with at least 30 seconds of representative headroom per increment; no automatic arbitrary halving and no changes to other services.
9. Acceptance benchmark: repeat cold and warm identical workloads at several safe worker/thread caps, preserving row sets, folds and seeds. Report completed trials/hour, p50/p95 stage time, preparation fit count, cache hit/wait/miss, USS/PSS/cgroup peak, page faults/I/O and queue blockers. Require prediction/replay parity, demonstrated warm preparation reuse and no unexplained private-memory or throughput regression before making the optimized path the V6 default. Publish measured savings, not a promised percentage.

- A shared evidence owner serves both planner/traces: compatible latest champions/recipes, negative results, coverage gaps, feature/graph evidence, uncertainty, hypothesis memory, resources/costs and watermark age.
- Hypothesis memory is not just an ADR: hypothesis, expected effect, intervention, comparison contract, supporting/opposing trial IDs, uncertainty, lifecycle and next test. ADRs retain lasting architecture choices; results update research memory.
- Paid zero-proposal outputs must be an intentional no-op/review with reason. Validation failures remain failed/degraded with raw errors/rejections/repair cost, never successful empty discovery.

## Contract F: Standardized Nomenclature And MLflow
| Term | Meaning | Surface |
|---|---|---|
| Campaign | immutable research identity/policy | manifest and experiment/tags |
| Decision | planner direction/allocation checkpoint | finite research.plan trace |
| Program | durable hypothesis/graph/search allocation | program metadata, decision links |
| Trial | concrete graph/config under one protocol | one outer MLflow run |
| Attempt | physical execution or retry | ledger/retry metadata, separate counting |
| Component fit | constituent/fold fit inside trial | child run/span with parent ID |
| Execution snapshot | asynchronous state at a watermark | research.snapshot trace, not paid planning |
| Dataset build | candidate materialization/validation | dataset.build trace and dataset artifact |

- IDs are stable UUID/content IDs; sequence numbers are display aids. ISO-8601 UTC, labeled local conversions. Human-readable names are not identity keys.
- Operation names: research.plan, research.validate, dataset.build, feature.generate, feature.select, pipeline.fit, pipeline.evaluate, research.snapshot, betting.evaluate.
- Experiment example `ima/research/<campaign_id>`; do not renumber historical experiment IDs. Trace names `<campaign_short> | plan D000101`, `<campaign_short> | execution S000042`, `<campaign_short> | dataset B000007`. No ambiguous cycle-only names or long hypothesis paragraphs as names.
- Trial `<target_short> | <pipeline_family> | P000012-T000064`; component `<trial_short> | node <node_id> | fold F02`. No volatile best score in names, no Optuna event mislabeled an LLM decision.
- Tags: schema version plus ima.campaign_id/event_kind/decision_id/program_id/trial_id/attempt_id/run_role/graph_id/dataset_id/feature_set_id/evaluation_contract_id/probability_basis/evidence_watermark/origin_decision_id/status.
- Metric names: fundamental_win.race_log_loss; blended_win.race_log_loss; place.top_3.brier; ranking.ndcg_at_3; speed.mae_mps; speed.rmse_mps; finish_time.mae_seconds; performance_distribution.nll; performance_distribution.crps; performance_distribution.coverage_80; performance_distribution.coverage_95; joint_order.log_loss; final_odds.log_mae; betting.<pool>.realized_roi. Document direction/unit/probability basis; retain old aliases through migration. Distribution NLL is density/unit-dependent and must not be compared numerically to categorical race log loss.
- Trace fields: planner_cost_usd, cost_status, chosen/allocated/unallocated trials, new/continuing programs, preparing/queued/running/completed/failed counts, watermark, best_by_comparison_contract, new_record and run links.
- Actual model-call spans carry installed MLflow-supported LLM token/cost attributes so native Traces USD columns work. Charge once per physical paid call including failed/repair calls. Decision totals reconcile calls; snapshots link and never recharge. Unknown cost=null/unavailable, not0.
- Example snapshot preview: `allocated40/48; 3 new, 9 continuing; fundamental LL=2.15609; new_record=no`. Preview is derived from immutable snapshot, not a stale hand-maintained string.
- Asynchronous work uses finite linked traces/origin metadata, not a root span open for days. Program terminal summaries resolve all allocated slots and trial/run links.
- Multi-objective best is a map. If scalar cannot apply, render `multiple objectives; see per-target bests`, never misleading best=None. Compare only compatible populations/targets/probability bases, not raw score magnitudes across tasks.
- Backfill old tags/names only idempotently with provenance, dry-run and approval; never rewrite old results, charge costs twice or imply old semantics were new.
- Verify installed UI custom-column support. If unavailable, use supported metadata/tags/previews and document limitation; do not promise unsupported table features or build a new UI.

## Contract G: Paper EV And Exotic Outcomes
- Start win/place/quinella/trio; extensible adapters for quinella place/forecast/tierce. No real stakes submitted.
- Quote schema: pool/race/selection/value/convention/unit_stake/currency/quoted_at/close_status/source_hash/rule_version/indicative_or_final/uncertainty.
- Initial "$10" examples mean HK$10 explicitly, configurable currency/stake. Verify official dividend unit, legal minimum and flexi rules per pool/date; do not assume every published dividend is perHK$10.
- Binary ticket with total-return multiple d: EV(s)=s*(p*d-1). Net-profit odds b -> d=b+1; dividend D per unit u -> d=D/u. General settlement: EV=sum_outcomes P(outcome)*gross_return(outcome)-ticket_cost.
- Arithmetic fixture only: p=.20, total-return d=6, HK$10 gives HK$2 expected net. Fair/breakeven d=1/p=5. Not a horse recommendation.
- Published quote/dividend conventions already reflect pool deductions; do not subtract takeout again. Pool-stake simulations instead apply applicable net-pool rules, refunds, dead heats, rounding and minimum-dividend rules.
- Betting place eligibility comes from official declared-starter/withdrawal rule, not always top3. `placing_top_k` research target remains distinct from paid-place contract.
- Quinella=unordered first2; trio=unordered first3; quinella place may have multiple winning pairs; forecast/tierce ordered. Stable ticket keys and rule-aware outcome adapters required.
- Win marginals alone cannot identify joint finishes. Baseline conditional-choice model P(i,j)=p_i*p_j/(1-p_i); sum applicable orders for quinella/trio. Disclose assumptions and test higher-place bias, handle near0/1 stably.
- Experimental alternatives: position-specific corrections/temperatures learned from past orders; listwise strengths; correlated time/speed residual simulation; pooling calibrated joint distributions from several model graphs. A point finish-time estimate alone is not an uncertainty distribution.
- Ordinary no-dead-heat quinella/trio outcome spaces each sum to1; joint-derived place marginals sum to eligible k. Dead heats/nonfinishers use separate actual settlement cases, not fabricated ranks.
- Score joint log loss/Brier where feasible, placing calibration, fair dividend/ticket hit rate, coverage and uncertainty. NDCG is ranking quality, not calibrated betting probability.
- Pre-race timed quotes enable quote-based estimated EV with payout movement uncertainty. Final dividends only allow labeled ex-post settlement of independently selected tickets, not a claim that the final price was available earlier.
- No exotic quotes -> probability/fair-price threshold and data-gap report. Do not invent quinella/trio prices from win odds. Assumed payout scenarios are visibly hypothetical.
- Use observed pool sizes/stakes for pool-impact simulation when available; otherwise disclose absence. Account own stake, correlated tickets, withdrawal/refund and late dividend movement.
- Initial strategies: fixed single ticket, predeclared conservative-EV threshold, fixed-stake portfolio. Threshold fitting uses past windows only. Report stakes/gross returns/netP&L/ROI/hit rate/drawdown/exposure/coverage/confidence separately.
- Planner may create model/distribution/strategy graphs under experimental policy. Never let final-dividend-derived variables become fundamental inputs; noisy profit metrics do not silently replace fundamental-first evaluation.

### Benter, Kelly And Paper Stake Sizing
The stages are probability estimation -> price/payout model -> edge/EV -> portfolio sizing -> settlement. Kelly cannot make an inaccurate probability or missing price reliable.
- Benter single-bet advantage is `p*d-1` under total-return dividend convention d. For one binary ticket, known fixed d>1 and negligible own impact, long-only full Kelly fraction is `max(0, (p*d-1)/(d-1))`, further restricted by exposure limits. For d<=1, invalid p or unknown return convention, reject sizing rather than divide by zero or guess.
- This fraction maximizes `p*log(1+f*(d-1)) + (1-p)*log(1-f)`, not expected dollars or hit rate. Keep EV, expected log growth, stake fraction and risk as separate fields.
- Pure arithmetic fixture: p=.20 and d=6 -> full Kelly .04 of bankroll. A quarter-Kelly research setting gives .01; on a hypothetical HK$1,000 bankroll that is HK$10. This is a test case, not a recommended live bankroll/fraction.
- Evaluate fixed-stake controls, full Kelly as a theoretical benchmark, and configurable fractional Kelly such as .10/.25/.50 of the unconstrained optimum. No fraction is a universally safe default. Fit or choose policy using earlier research windows only; freeze it for later comparison.
- For simultaneous tickets j and exhaustive race/settlement scenarios s, build total-return matrix R[s,j]. Optimize `sum_s P[s]*log(1 + sum_j f[j]*(R[s,j]-1))` with nonnegative f, positive scenario wealth and an explicit cash/exposure reserve. This handles winning together, mutually exclusive bets and shared horse exposure.
- Do not compute Kelly independently for each ticket and add the results, even for separate pools on the same race. Joint outcome probability and actual pool settlement define correlation; a covariance matrix alone is insufficient for exact expected-log optimization.
- Initial solver: existing SciPy constrained optimization with fixed scenario probabilities/payouts, analytic derivatives where straightforward, reproducible tolerances and independent objective/constraint readback. Do not write a numerical optimizer. Constant-payout log utility is concave on its feasible domain; stake-dependent pari-mutuel impact requires a separate model/solver validation, not an unjustified convexity claim.
- Store solver status, residuals, objective, bounds, scenario IDs, probability/quote/model hashes, seed and policy. Failed convergence returns no stake recommendation; zero-stake is a valid optimal/no-edge outcome, not a failure.
- StakePolicy: mode, bankroll/currency, Kelly fraction, per-race/day/ticket exposure caps, minimum cash reserve, wager-unit rules, probability uncertainty policy, payout stress policy, own-impact policy and maximum open exposure. Initial output is paper-only and model selection cannot silently alter operator risk caps.
- Round to legal stake increments conservatively, never round a sub-minimum allocation up automatically. Re-evaluate portfolio utility/exposure after rounding; omit invalid tickets and report omitted edge/capital. In exotic flexi bets, price all ticket combinations and wager fractions under verified rules.
- Test calibration uncertainty using coherent race-level model/parameter draws or posterior/scenario ensembles, plus block-bootstrap historical evaluation. Do not lower each horse's probability independently and then pretend the vector remains a normalized joint model. Fractional Kelly and conservative scenarios mitigate uncertainty; neither guarantees no loss.
- Stress dividend deterioration, probability overconfidence, correlated longshot exposure, minimum stakes, scratches/refunds, dead heats, missing prices and pool impact. Report expected versus realized growth, bankroll path, drawdown and empirical loss/risk thresholds with finite-horizon uncertainty.
- When pool stakes/turnover are observed, fit or simulate own-impact under the official pool rules. Without them, disclose negligible-impact assumption and scenario stresses. Do not apply the simple binary formula to impact-sensitive multi-ticket pools as an exact optimum.
- No timestamped quotes -> no executable stake recommendation. Final-dividend-only data can support labeled hindsight settlement and predeclared hypothetical stress studies, but a strategy sized using those final dividends is hindsight and excluded from deployability claims.
- For huge outcome spaces, enumerate small-field cases exactly first; use seeded scenario sampling only with error/convergence checks and explicit treatment of omitted rare tails. Never silently drop unlikely losing outcomes to inflate growth.
- Optional next research: chance-constrained finite-horizon policy from the local Deza/Huang/Metel paper, with data/solver feasibility gate. Do not install a mixed-integer solver or claim its published gains transfer here without a separate validated experiment.
- Orchestrator may propose distribution/price/portfolio methods and research fractions within policy bounds, using several model pipelines as inputs. It cannot edit live bankroll/risk limits, place wagers, or substitute profit for fundamental model quality without explicit policy change.

## Contract H: Speed History And Conference-Inspired Performance Models

### H1. Historical speed as a feature, separately from a future label
- A timed historical observation requires exact horse identity, distance in metres, elapsed seconds, source/time availability, discipline (race/workout/trial), surface/track, segment or whole-event scope, and timing semantics (individual versus batch/winner). Speed_mps=distance_m/elapsed_seconds with finite positive values only.
- Reuse canonical finish_seconds parsing; do not divide by raw formatted finish_time strings. Quarantine invalid/impossible timings with configurable documented data-quality ranges, retaining raw values and reasons. Missing/untimed workouts supply other training features, not fabricated speed.
- Source-specific histories: previous1/3/5/10 timed observations and explicit time windows; last/mean/median, recency-weighted mean, trend, std/MAD, same-distance/surface/venue summaries, counts, recency and support. Selection of windows remains a planner hypothesis within resource policy.
- Historical variability features describe past observations; they are not automatically the future conditional variance. Use minimum support and shrink sparse histories toward training-population priors, with missingness/support visible. New horses do not get zero variance.
- Course/going/distance/weight/competition adjustments reuse existing AdjustedSpeedHistory where suitable. Fit adjustment models and transformations on earlier training data; historical training residuals are forward cross-fitted. Never learn a track variant from the current race's outcomes or future races and feed it into that same prediction.
- Today's prior-race/workout/trial predictors include only eligible records before its cutoff. Today's observed finish time/speed is a label for evaluation; it becomes history only for later eligible races. Using current race results to build a winner-relative speed feature must be restricted to that historical event before shifting/joining to subsequent races.
- Track race-speed_ratio versus physical m/s and segment speed distinctly. Sectional features require the actual covered segment distance, not today's whole-race distance. Repeated observations and incomplete coverage follow Contract A.
- Model may combine separate source summaries/interactions, not blindly average workout, trial and race speeds. Compare past-races-only, +trials, +workouts, and combined source ablations on identical score races; report strict versus assumed publication tiers separately.

### H2. Activate physical speed and time prediction
- Existing target adjusted_finish_time_or_speed is reused/migrated with explicit units and objective aliases; it currently means physical speed, not proof of a fully condition-adjusted target. Distinguish predicting raw speed with condition features from predicting a train-fitted adjusted residual and reconstructing raw performance.
- Support future speed_mps and optional finish_time_seconds/log_finish_time through explicit target parameters/contracts, rather than one ambiguous metric name. Eligibility is declared at race level; nonfinishers, missing times and dead heats have stated policies, never invented elapsed times or silent field changes for win scoring.
- First point controls: current ridge/hist-gradient/CatBoost regressors; compare conditional simple baselines. Evaluate speed MAE/RMSE in m/s and derived time error in seconds on common valid rows/races; report exclusions and race-weighting.
- Conversion time=distance/speed must use actual race distance and positive support. E[distance/speed] is generally not distance/E[speed]; sample/distribution transformation and point-prediction transformation are distinct outputs. Do not invert a nonpositive Gaussian speed draw and call it a plausible time.
- The successor's experimental policy must actually admit speed/time targets. Merely adding target docs cannot bypass V4/V5's fixed E-lane contracts. Introduce versioned experimental-family scheduling preserving campaign-wide 80/20 allocation while letting the orchestrator choose speed, ranking, placing, odds and composition programs without inventing a new obligatory count for each.

### H3. Learn expected performance and uncertainty
- PerformanceDistribution schema: race/horse keys, model/feature/dataset/evaluation IDs, coordinate (latent strength, physical speed, time or log-time), direction (higher/lower wins), distribution family, location, positive scale/variance, fit/calibration cutoffs, support, uncertainty_kind and optional factor/covariance specification.
- Separate irreducible performance uncertainty (aleatoric) from uncertainty about fitted model parameters (epistemic). Ensembles/different seeds can probe the latter but do not automatically yield calibrated confidence intervals.
- Stage A: homoscedastic point model plus a shared residual distribution fitted on past OOF errors. Stage B: conditional/horse-context scale learned from train/OOF performance data, using shrinkage and a positive variance link. Existing CatBoost RMSEWithUncertainty is the first mature adapter; optional NGBoost is a separately locked challenger.
- Choose positive-support time/speed distributions or a log/latent coordinate with a documented transformation. Gaussian latent performance is not a claim that all physical finish times can be Gaussian; test tail/support behavior and calibration.
- Compare each heteroscedastic variant to the same mean model with shared scale. This isolates variability benefits from changing estimator family/features. Fit scale calibration on a separate chronological window, not future score residuals.
- Score performance density NLL, CRPS via a mature formula/tool, interval coverage/width and probability integral transform diagnostics; compare within the same coordinate/units. Add race-winner log loss, place calibration and joint-order metrics downstream. A smaller time MAE does not establish better win probabilities.

### H4. Conference-inspired race likelihood and joint simulation
- Preserve classical Benter conditional logit and boosted win models as controls. Add clearly named families performance_probit_shared_scale and performance_probit_heteroscedastic in the experimental20% allocation initially; no automatic replacement/reclassification of the Benter80% lane.
- Winner objective: minus mean log P(observed winner | complete pre-race field). Train on race outcomes with properly grouped chronological races; do not optimize independent horse binary loss and call it the same likelihood.
- For win-only latent models, common additive location and global scale are not fully identified by race winners. Fix an explicit location/scale convention, regularize/shrink conditional scale and document parameters. Do not fit unconstrained independent variance parameters per runner, which can game sparse observations.
- Start independent Gaussian latent performance as a labeled baseline. Use existing SciPy integration/optimization: under independent errors, conditional-on-one-performance quadrature/CDF products are a tractable winner-probability approach. Validate positive scales, log-domain tails, gradients or finite-difference stability and quadrature convergence. Numerical approximations cannot silently become zero probability for observed winners.
- Train simulation-based likelihood only with a smooth integration/likelihood approximation validated against quadrature/exact small cases. Raw argmin/argmax Monte Carlo winner counts are discontinuous/noisy and are not a suitable default gradient-training objective.
- Research three fitting regimes explicitly: observed performance distribution likelihood; race-winner likelihood; and an optional weighted joint objective with weights chosen on earlier development data. Compare all under the same race log-loss protocol and report label coverage constraints.
- Joint outcome node draws performances for the complete race and records ordered finishes, producing win/place/quinella/trio/ordered probabilities from the same outcome distribution. For speed higher wins; for time lower wins. Dead heats/nonfinishers and price/settlement remain explicit extensions, not silently discarded cases.
- Next dependency experiment: identifiable structured covariance/pace effects with horse-specific responses. Adding an identical scalar disturbance to every horse leaves its order unchanged and is a negative-control test, not a useful ranking model. Any covariance/factor matrix must be valid and learned from past data; sparse evidence may justify staying independent.
- Reuse NumPy/SciPy RNG and optional scrambled Sobol draws. Log seed, draw count/sequence/version, simulation error/convergence, coordinate and covariance. Increase draws adaptively until a declared probability-error criterion or resource budget; record unmet tolerance. Check across independent scrambles and against numerical winner probabilities. Do not claim ordinary iid binomial error bounds for quasi-Monte Carlo draws.
- For very rare exotic outcomes, use exact low-dimensional/enumerated controls or independently validated tail estimation; zero sampled occurrences do not prove probability zero or infinite fair odds. Simulation complexity/memory are scheduler reservations and artifacts are shared/chunked, not one giant trial-private tensor.
- Race-day repeatability remains fixed models/data/seed/tolerance and deterministic kernels where possible. Research can randomize seeds, features and search directions with recorded lineage; it cannot add unexplained noise to final probabilities.

### H5. Reusable adapters and orchestrator choices
- Rank scores -> existing softmax/chronological temperature -> win probabilities is available now; preserve it. Win probabilities -> Plackett-Luce/Harville is an explicit assumption-based joint-order baseline, not a uniquely recovered finishing distribution.
- Time/speed point prediction -> train/OOF residual distribution -> joint-order node; distributional estimator -> joint-order node directly; legacy win/rank models -> calibrated strengths baseline. Every adapter states lost/inferred information and dependencies.
- The planner can select speed-source feature views, adjustment/window/shrinkage, shared versus conditional scale, distribution family/support, estimator, fitting regime, safe adapters, dependence model, calibration, simulated versus integrated evaluation, ensembles and trial allocations. Optuna tunes allowed parameters inside the chosen graph; deterministic validation controls leakage/units/identification/resources.
- Expose pipeline_family, adapter_id, performance_coordinate, mean_model, variance_model, fit_objective, source_feature_groups, uncertainty_kind, dependency_model, calibration_method and simulation/convergence attributes in MLflow runs/traces/evidence and registered packages. Variance parameters must be packaged for replay, not lost after generating win probabilities.

## Regression Guardrails
- Current edits: this plan/generated scoped dashboard/state/evidence only. Preserve dirty user config/graphs/ops files and queue-memory plan.
- Future damage radius: systemic (data chronology, controller contracts, registry/inference and tracking).
- Branch strategy: current branch is acceptable for documentation-only delivery with no runtime changes. Dedicated branch/worktree `feat/agentic-research-expansion` from approved base before implementation; blocked branch creation is a blocker, not permission to mutate main.
- Protected: source allowlist/robots, raw hashes, exact horse IDs, old keys/history, chronological race grouping, old campaign identity, fundamental-market separation, existing target/replay behavior, restart/outbox idempotency and Cortex/Solar/other users.
- Consumers: acquisition/snapshot timers, dataset CLI, planner contracts, Optuna, executor/inference, MLflow and future race-day operator.
- Proof: focused tests + full regressions + fixtures/adversarial attacks + actual source/model/trace readbacks + full-history memory/performance measurements.
- Commit units: contracts; event normalization; joins; lifecycle; replay; formulas; selection; graph core; ensembles; controller; tracing; outcomes; EV; verification; rollout. Only tested coherent units; dependencies/locks in their owning unit.
- Planned Touch Files below are boundaries, not refactor licenses. Scope expansion requires declaration. The current execution request permits implementation and isolated V6 deployment after verification; unrelated dirty files and other users remain protected.

## Generality Guardrail
- Reuse ResearchLedger, HypothesisMemory, ProgramSearchController, ResourceAdmission, feature content IDs, snapshot verifier, MLflow outbox and existing mature computation tools.
- High recurrence: all feature/event/dataset/graph work shares provenance/cutoff/fit contracts. Add DatasetRequest, FeatureDefinition, PipelineGraph owners, not per-family private optimizer loops.
- Custom code is justified for racing identities/cutoffs, chronological graph integration, coherent finishes and settlement; not generic arrays, optimization, parallelism or a new framework.

## Deterministic Real-User Test
- Entry point: terminal isolated canary and existing MLflow API/UI. No new product UI.
- Fixtures: at least40 chronological meetings with8-12 horses, exact IDs, complete/incomplete event intervals, publication/capture distinctions, corrections, batch/individual trial times and official quote/dividend conventions; recorded planner responses avoid paid tests.
- Workflow: request data -> inspect verified manifest and historical race/workout/trial speed -> approve isolated successor -> generate/select64 eligible generated features -> propose programs with occupied workers -> fit Benter/boosted/ensemble and shared/conditional-scale performance controls -> inspect current champions/USD -> derive joint outcomes and evaluate paper tickets -> restart -> replay model.
- Implemented canary entry point (full execution gate still required): `.venv/bin/python -m scripts.run_research_expansion_canary --planner fixture --output .tmp/research-expansion-canary`. It generates an explicitly synthetic fixture; it does not certify official-data coverage or live paid planning.
- Assertions: no future leakage, honest missingness, preserved history, no hidden truncation/budget loss, safe OOF graph, latest evidence, resource refill, reconciled USD, correct paper arithmetic and no duplicated allocation/cost after restart.
- Record stdout/status/manifests/selection/temporal audits/OOF/ledger/trace API/UI screenshot/resource timelines/settlement/replay diff.

## Fulfillment and Readback Proof
- Planning: checker passes, all requested mechanisms have contracts/phases/tests, generated dashboard readback matches Markdown. Unchecked implementation phases stay unchecked.
- Future: read representative predictor values against source/time evidence; resolve DatasetRequest to validated hashes/rows; reload graph outputs; reconcile traces/budgets/cost/bests to ledger; reproduce EV/settlement from contracts.
- Not-done: only schemas/raw events, unsupported formulas, ignored graph fields, silently truncated selection, dataset dropping keys, incompatible bests, cost only in prose, guessed quotes or deployment without gates.

## Armageddon Mode
- Time: future trial/clearance, same-day undated observation, current profile copied backward, modern correction, global supervised formula, current-race speed or score-window residual used in predictors. Reject or retain missing strict output.
- Identity/coverage: inactive horse name collision, duplicate workout prose, movement standby, conflicting fullIDs, missing family/units, dropped old race/checksum mismatch.
- Formula: dependency target alias, imports/attributes, cyclic/unknown symbol, extreme powers/division, domain error, runaway depth/count and inconsistent units. Narrative word result is not itself leakage.
- Graph: future-trained reference, in-sample stack, row mismatch, cyclic edge, incoherent probabilities, nonpositive physical-speed support, unidentifiable scale, common-offset fake ranking effects, unconverged simulation/zero rare-event probabilities and incomplete mean/variance package.
- Runtime: concurrent cold cache, dead builder/worker, own-unitOOM, stale lock, full backlog, starved large job, paid invalid/timeout/repair response, MLflow outage/restart midallocation.
- Betting: wrong unit/currency, takeout twice, small-fieldplace, stale/finalquote lookahead, deadheat/scratch, duplicate correlated tickets, extreme p, isolated-ticket Kelly oversizing, omitted losing scenarios, rounding above caps and nonconverged sizing.
- Must-fix: leakage, identity/corruption, monetary errors, silent budget loss/duplicate cost, nonreplayable champions or protected-service changes. Performance claims require benchmark, not intuition.

## Ordered State and Dashboard
- Plan `docs/AGENTIC_RESEARCH_EXPANSION_PLAN.md`.
- Scoped ledgers `.mega/agentic-research-expansion/state.jsonl` and `evidence.jsonl`; inspect tool --help for supported path flags.
- Dashboard `.mega/dashboards/agentic-research-expansion.html`; generated from plan/ledgers, open Chromium for review only.
- Record baseline/decisions/tests/atomic commits/failed gates/readback/rollout identity. Plan delivered is not implementation done.

## Phase 1: Baseline
### Subphase 1.1: Evidence and compatibility inventory
- Objective: verify deployment/data/tracking/resource baseline and comparison contracts read-only.
- Inspect: Root-Cause files, existing plans, remote own-user release/config/status/ledger/trace exports, lock manifests and PDFs.
- Planned Touch Files: `docs/AGENTIC_RESEARCH_EXPANSION_VALIDATION.md` (new), `docs/RESEARCH_NOMENCLATURE.md` (new), `tests/fixtures/research_expansion/` (new).
- Commit: `docs(research): record expansion baseline and comparison contracts`.
- Tests: `.venv/bin/python -m unittest tests.test_research_specs tests.test_research_evidence tests.test_mlflow_tracking`; read-only exports; PDF equation/page readback.
- Success Criteria: current identities/counts/recipes/caps, missing evidence and implemented queue-memory pieces distinguished with timestamps.
- Checklist:
  - [ ] Export source/dataset/protocol/SDK identities, champions and tracking watermarks without secrets.
  - [ ] Measure representative current Benter/boosted preparation/fit resources and inspect actual queue behavior.
  - [ ] Extract Benter/joint-order/pari-mutuel assumptions and fixtures; agree target/quote units and nomenclature aliases.
- Handoff: reproducible baseline, protected dirty paths, approved base SHA and explicit unknowns.

## Phase 2: Historical Events
### Subphase 2.1: Normalized identities, availability and coverage
- Objective: implement Contract A without weakening collector trust.
- Planned Touch Files: `ima/historical_events.py` (new), `scrapper/official_corpus.py`, `scripts/build_official_dataset.py`, `tests/test_historical_events.py` (new), `tests/test_official_corpus.py`, `tests/test_official_dataset.py`.
- Inspect: raw/document/collector fixtures, official confirmation logic, snapshot provenance and movementPDF mappings.
- Commit: `feat(data): normalize official events with temporal evidence`.
- Tests: `.venv/bin/python -m unittest tests.test_historical_events tests.test_official_corpus tests.test_official_dataset`.
- Success Criteria: observations/groups/corrections/tiers reproduce; current snapshots do not become historical publication proof.
- Checklist:
  - [ ] Implement typed event/unit/time/coverage schemas and source/physical IDs.
  - [ ] Preserve ambiguity and family-specific tier/exclusion reasons.
  - [ ] Test corrections, name collisions, standby and same-day identical descriptions.
- Handoff: every fixture inclusion/exclusion traceable to hash/time/identity rule.

### Subphase 2.2: Predictor joins and usable coverage
- Objective: populate family predictors safely, including unknown-versus-zero handling.
- Planned Touch Files: `ima/historical_events.py`, `ima/rich_features.py`, `ima/feature_sets.py`, `scripts/build_official_dataset.py`, `scripts/verify_official_snapshot.py`, `tests/test_historical_events.py`, `tests/test_rich_features.py`, `tests/test_official_snapshot_verifier.py`.
- Commit: `feat(features): join official workouts trials vet and movements`.
- Tests: `.venv/bin/python -m unittest tests.test_historical_events tests.test_rich_features tests.test_official_dataset tests.test_official_snapshot_verifier`.
- Success Criteria: available events create correct values; unknown/future events do not; all families have truthful coverage reports and real-source spot checks.
- Checklist:
  - [ ] Replace global date-range availability assumptions with entity/interval/cutoff-aware vectorized joins.
  - [ ] Implement Contract A table and companions; unknown timing/identity remains excluded from strict features.
  - [ ] Test zero/unknown, ongoing stays, future clearance and individual versus batch time.
  - [ ] Benchmark full-history joins; compare representative output rows to raw source.
- Handoff: strict/assumed coverage report by family/era, genuine gaps explicitly retained.

### Subphase 2.3: Source-specific historical speed features
- Objective: make timed prior races, workouts and trials reusable predictors across all compatible pipelines, preserving their different semantics.
- Inspect: `ima/rich_features.py` time parsing/shifted speed ratios/trial speed, `ima/feature_residuals.py` OOF adjustment, parser timing/source fields and coverage/cutoff contracts.
- Planned Touch Files: `ima/historical_events.py`, `ima/speed_features.py` (new thin source-normalization adapter), `ima/rich_features.py`, `ima/feature_residuals.py`, `ima/feature_sets.py`, `ima/feature_program.py`, `scripts/build_official_dataset.py`, `tests/test_speed_features.py` (new), `tests/test_rich_features.py`, `tests/test_feature_residuals.py` (reuse or new after audit).
- Commit: `feat(features): build point-in-time race workout and trial speed histories`.
- Tests: `.venv/bin/python -m unittest tests.test_speed_features tests.test_rich_features tests.test_feature_residuals tests.test_historical_events`; actual-source row readback and timing-unit fixtures.
- Success Criteria: eligible physical speeds and historical consistency/support exist by source, future/current labels are excluded, and residual adjustment uses safe prior/OOF history.
- Checklist:
  - [ ] Normalize explicit distance/time/segment units and individual timing; retain untimed workout availability without fabricated speed.
  - [ ] Reuse existing lag/rolling/residual owners; add source-specific last/recency/window/trend/dispersion/support views and train-safe condition matching.
  - [ ] Test one/no prior observation, new horse shrinkage, future/current race insertion, duplicated capture and misleading batch time.
  - [ ] Run races-only/+trials/+workouts/combined source ablations with matched populations and publication tiers; log coverage and exclusions.
- Handoff: source-to-column lineage plus confirmed historical speed values used by both win and speed prediction recipes.

## Phase 3: Dataset Lifecycle
### Subphase 3.1: Requests, registry and verified builds
- Objective: automatic dataset construction as a durable planner action.
- Planned Touch Files: `ima/dataset_specs.py` (new), `ima/dataset_registry.py` (new), `ima/research_store.py`, `scripts/build_research_dataset.py` (new isolated consumer of immutable official snapshots), `scripts/build_official_dataset.py`, `scripts/refresh_official_snapshot.py`, `scripts/verify_official_snapshot.py`, `tests/test_dataset_registry.py` (new), `tests/test_dataset_registry_adversarial.py` (new independent leakage and lifecycle audit), `tests/test_official_snapshot_refresh.py`.
- Commit: `feat(data): build and validate immutable dataset requests`.
- Tests: `.venv/bin/python -m unittest tests.test_dataset_registry tests.test_official_snapshot_refresh tests.test_official_snapshot_verifier tests.test_research_store`.
- Success Criteria: durable IDs resolve to verified/rejected artifact; partial/crashed/duplicate requests cannot overwrite verified data.
- Checklist:
  - [ ] Implement lifecycle/migrations, atomic hashes/manifests and MLflow dataset inputs.
  - [ ] Validate preserved keys/history, labels/eligibility, coverage/timing/checksums.
  - [ ] Support source-specific partial reports; failure never updates verified pointer.
- Handoff: artifact row/hash readback and interrupted-build recovery evidence.

### Subphase 3.2: Replay and controlled promotion
- Objective: reproducible comparisons before successor handoff.
- Planned Touch Files: `ima/dataset_registry.py`, `ima/research_hypotheses.py`, `ima/research_evaluation.py`, `ima/research_controller.py`, `scripts/replay_dataset_champions.py` (new), `tests/test_dataset_registry.py`, `tests/test_research_evaluation.py`, `tests/test_research_evidence.py`.
- Commit: `feat(research): gate successor data through champion replay`.
- Tests: focused modules and replay CLI on old/new/common race fixture.
- Success Criteria: old results reproduce; changed-data comparisons disclose population/training effects; promotion requires explicit policy/approval.
- Checklist:
  - [ ] Add comparison signatures and per-target shared race masks/exclusions.
  - [ ] Refit canonical Benter/boosted/eligible-market controls; compute paired changes/block uncertainty.
  - [ ] Reject accidental three-year trimming, changed dates or unverified-tier promotion.
  - [ ] Add dry-run successor manifest and approved promotion-policy envelope.
- Handoff: reproducibility/common-population report; existing campaign remains untouched.

## Phase 4: Feature Creation And Selection
### Subphase 4.1: Formula definitions and deterministic catalog
- Objective: support novel measurement formulas not pre-enumerated as literals.
- Planned Touch Files: `ima/feature_definitions.py` (new), `ima/feature_expressions.py` (new), `ima/feature_discovery_specs.py`, `ima/feature_program.py`, `ima/research_specs.py`, `ima/openrouter_orchestrator.py`, `tests/test_feature_expressions.py` (new), `tests/test_research_specs.py`, `tests/test_openrouter_orchestrator.py`.
- Commit: `feat(features): compile reusable planner expression definitions`.
- Tests: `.venv/bin/python -m unittest tests.test_feature_expressions tests.test_research_specs tests.test_openrouter_orchestrator`; AST attack fixtures/custom primitive parity.
- Success Criteria: novel official-data expression executes in multiple model families; unsafe/label-derived definitions reject with structured reasons.
- Checklist:
  - [ ] Implement typed AST, units/temporal dependency validation and safe lowering.
  - [ ] Register content IDs; expose formulas as DFS seeds/custom primitives and reusable catalogs.
  - [ ] Make generation/window/depth limits explicit resources; expose planner actions/cost estimates.
- Handoff: same new definitionID consumed by two pipelines with reproducible matrix hashes.

### Subphase 4.2: Configurable selection and shared preparation
- Objective: eliminate all hidden selector limits and reduce private copying.
- Planned Touch Files: `ima/feature_discovery_specs.py`, `ima/feature_screening.py`, `ima/feature_program.py`, `ima/research_preparation.py` (reuse or new after audit), `ima/research_executor.py`, `tests/test_feature_screening.py` (new), `tests/test_research_preparation.py` (reuse or new), `scripts/benchmark_discovery_features.py`.
- Commit: `feat(features): admit explicit selection budgets against measured resources`.
- Tests: `.venv/bin/python -m unittest tests.test_feature_screening tests.test_research_preparation tests.test_research_executor`; matched cold/warm benchmark.
- Success Criteria: admitted8/32/64/128/all requests behave honestly; sequential64 does not silently become6; cache fit counts and memory savings measured.
- Checklist:
  - [ ] Migrate old max_selected semantics explicitly; remove shortlist32/12 and half-rule.
  - [ ] Add method-aware cost/shortlist/cardinality contracts and structured deferral.
  - [ ] Fit/cache only training/inner folds with full dependency/label/settings keys.
  - [ ] Persist compact read-only numeric artifacts, avoid wide compatibility copies and active-cache eviction.
- Handoff: >32 selection report plus full-history selector/runtime/cache/PSS evidence.

## Phase 5: Pipeline Composition
### Subphase 5.1: Typed DAG and chronological prediction artifacts
- Objective: generalize existing recipe execution, not add a second competing optimizer.
- Planned Touch Files: `ima/pipeline_graph.py` (new), `ima/prediction_store.py` (new), `ima/research_specs.py`, `ima/research_executor.py`, `ima/research_models.py`, `ima/research_search.py`, `tests/test_pipeline_graph.py` (new), `tests/test_prediction_store.py` (new), `tests/test_research_executor.py`.
- Commit: `feat(pipeline): execute graphs with forward OOF artifacts`.
- Tests: graph/store/executor/search tests; future references/cycles/row mismatch; legacy recipe parity.
- Success Criteria: old recipes translate losslessly; two-estimator graph has explicit safe fit scopes and resource accounting.
- Checklist:
  - [ ] Define node/output contracts and adapters, chronological fold execution and reusable predictions.
  - [ ] Deduplicate compatible component jobs; reserve child CPU/RAM and count physical fits separately.
  - [ ] Bind Optuna to frozen graph search spaces, not silent structural mutation.
- Handoff: graph outputs and row-level proof OOF scoring follows each training cutoff.

### Subphase 5.2: Ensembles, market stage and replay packages
- Objective: planner-generated combinations become real inference candidates.
- Planned Touch Files: `ima/pipeline_graph.py`, `ima/research_models.py`, `ima/research_evaluation.py`, `ima/research_model_package.py`, `ima/openrouter_orchestrator.py`, `ima/mlflow_tracking.py`, `tests/test_pipeline_graph.py`, `tests/test_research_model_package.py`, `tests/test_mlflow_tracking.py`.
- Commit: `feat(pipeline): compose calibrated ensembles and version graph models`.
- Tests: pools/stacks/nested-fit leakage controls, constituent comparisons, fundamental-market separation and package reload parity.
- Success Criteria: a planner Benter+boosted graph trains/registers/reloads; old single-model replay unchanged.
- Checklist:
  - [ ] Implement weighted/log pooling and chronological meta-estimator using mature tools.
  - [ ] Add separate point-in-time market blend output with Benter equation/step documentation.
  - [ ] Log outer/component roles, diversity/metric comparisons and all fitted states/dependencies.
- Handoff: real isolated graph run/model links, constituent controls and replay diff.

### Subphase 5.3: Activate speed/time and conditional performance distributions
- Objective: execute the existing speed path in a compatible experimental policy and add learned variability using mature estimator APIs.
- Planned Touch Files: `ima/research_targets.py`, `ima/research_specs.py`, `ima/research_controller.py` (new policy definition only), `ima/research_models.py`, `ima/performance_distributions.py` (new), `ima/research_executor.py`, `ima/research_evaluation.py`, `ima/research_model_package.py`, `ima/pipeline_graph.py`, `pyproject.toml`/research locks (only if new dependency justified), `tests/test_performance_distributions.py` (new), `tests/test_research_targets.py`, `tests/test_research_models.py`, `tests/test_research_executor.py`, `tests/test_research_model_package.py`.
- Commit: `feat(modeling): evaluate future speed and conditional performance uncertainty`.
- Tests: focused target/model/executor/distribution/package tests; shared-scale/conditional-scale controls and raw-time-string unit fixture; new experimental-policy admission while legacy contracts remain unchanged.
- Success Criteria: real speed trial admitted and evaluated, units/support/coverage explicit, shared and conditional distributions calibrated on past data, package reload reproduces mean/scale.
- Checklist:
  - [ ] Resolve finish_time versus finish_seconds input semantics and preserve legacy physical-speed labels/metric aliases through schema migration.
  - [ ] Add clear coordinate/target parameters and nonfinisher/time coverage policies; retain full fields when scoring winner probabilities.
  - [ ] Fit point controls and OOF shared-residual baseline, then CatBoost uncertainty adapter; add NGBoost only as a locked/tested optional challenger.
  - [ ] Test positive support, variance floor/shrinkage, small-history horses, physical versus latent scales, and non-equivalence of inverse means.
  - [ ] Log performance errors/NLL/CRPS/coverage and downstream win metrics separately, using mature distribution scoring formulas/tools rather than new numerical frameworks.
- Handoff: isolated completed speed/distribution run IDs, source-feature ablation, calibration report and registered mean/variance replay.

### Subphase 5.4: Conference-inspired probit and reusable race-outcome adapters
- Objective: train shared/heteroscedastic race likelihoods and compare all compatible existing pipelines through explicit joint-outcome adapters.
- Inspect: transcript sections12-16, existing conditional-logit race likelihood and ranker calibration, SciPy integration facilities and Contract H identification/support rules.
- Planned Touch Files: `ima/performance_probit.py` (new), `ima/performance_distributions.py`, `ima/probabilistic_adapters.py` (new), `ima/rank_distributions.py` (core outcome engine created here; pool-specific extensions Phase7), `ima/pipeline_graph.py`, `ima/research_models.py`, `ima/research_executor.py`, `ima/research_evaluation.py`, `ima/research_model_package.py`, `ima/openrouter_orchestrator.py`, `tests/test_performance_probit.py` (new), `tests/test_probabilistic_adapters.py` (new), `tests/test_rank_distributions.py`, `tests/test_pipeline_graph.py`.
- Commit: `feat(modeling): derive joint race outcomes from performance distributions`.
- Tests: symmetric2/3-horse controls; two-normal analytic win probability; quadrature refinement; location/scale identification; identical-offset rank invariance; deterministic seed/reload; calibration and legacy-ranker parity; independent-scramble convergence and rare-outcome treatment.
- Success Criteria: shared/conditional-scale models fit race-winner likelihood and produce coherent reproducible order probabilities; empirical comparisons isolate scale-model effects, no unsupported claim of proprietary Benter replication.
- Checklist:
  - [ ] Implement positive scale/identification/regularization and numerically stable independent-Gaussian winner integration using mature SciPy kernels/optimizer.
  - [ ] Separate density-fitted, winner-fitted and optional combined objectives; do not train raw Monte Carlo argmax count objectives with ordinary gradients.
  - [ ] Reuse current calibrated ranking adapter and expose point-speed residual and distribution-native adapters with explicit information/assumption contracts.
  - [ ] Add seeded/chunked race-order simulation and convergence reporting; valid dependency/factor models only after independent baseline controls.
  - [ ] Make planner propose the new family/variance/source/adapter/fit regime without changing legacy Benter80% identity or truncating the experimental20% to old fixed targets.
  - [ ] Compare conditional logit, boosted, shared-scale probit and heteroscedastic variants on identical races; package uncertainty/covariance/simulation settings and output contracts.
- Handoff: chronological comparison report, inference/model/run/trace links, quadrature-versus-simulation proof and orchestrator evidence exposing actual mean/variance family choices.

## Phase 6: Controller And Observability
### Subphase 6.1: Independent research queue and current evidence
- Objective: planning freedom without unbounded resource execution.
- Planned Touch Files: `ima/research_expansion.py` (new successor controller), `ima/research_v5.py` (legacy adapter only), `ima/research_store.py`, `ima/research_scheduler.py`, `ima/research_resources.py`, `ima/research_evidence.py`, `ima/research_hypotheses.py`, `ima/openrouter_orchestrator.py`, `ima/optimizer.py`, `tests/test_research_expansion.py` (new), `tests/test_discovery_feedback.py`, `tests/test_research_resources.py`, `tests/test_research_scheduler.py` (reuse or new), `scripts/benchmark_discovery_scheduler.py`.
- Commit: `feat(controller): decouple research decisions from fit capacity`.
- Tests: store/resources/evidence/feedback/expansion with full queue, variable budgets, retirements and unequal durations.
- Success Criteria: busy fits do not block proposals/review; budgets reconcile; fresh champions enter evidence; eligible workers refill fairly.
- Checklist:
  - [ ] Migrate proposal_batch_size naming/precedence; expose independent ceilings and typed action requests.
  - [ ] Queue data/formula/graph work; preparation gates fits, not planning review.
  - [ ] Refresh evidence/hypothesis memory on committed completions; expose staleness.
  - [ ] Integrate measured admission/cap adjustment and cost/restart idempotency.
- Handoff: event timeline/ledger proving planning and refill while workers occupied.

### Subphase 6.2: Names, costs, bests and tracking reconciliation
- Objective: Contract F consistently across CLI/evidence/traces/runs/models.
- Planned Touch Files: `ima/research_telemetry.py` (new), `ima/mlflow_tracking.py`, `ima/research_evidence.py`, `ima/research_expansion.py`, `scripts/enrich_mlflow_research_identity.py`, `tests/test_research_telemetry.py` (new), `tests/test_mlflow_tracking.py`, `tests/test_enrich_mlflow_research_identity.py`, `docs/RESEARCH_NOMENCLATURE.md`.
- Commit: `feat(observability): standardize research identities USD and snapshots`.
- Tests: exports/real isolated MLflow server/UI screenshot, idempotent historical dry-run, SDK-supported cost field readback.
- Success Criteria: finite plan/execution/dataset traces distinguishable; USD/counts/bests/run links match one ledger snapshot.
- Checklist:
  - [ ] Implement shared nomenclature/summary owner and documented aliases.
  - [ ] Link async continuations; per-call cost plus reconciled totals; mixed objectives shown as maps.
  - [ ] Add/reuse outbox retry/reconciliation; optional historical tag backfill never rewrites results.
  - [ ] Verify actual custom-column/native USD capabilities and document unsupported features.
- Handoff: trace table screenshot showing a paid decision and linked later record, with API/ledger reconciliation.

## Phase 7: Joint Outcomes And EV
### Subphase 7.1: Official pools and probability distributions
- Objective: coherent ticket probabilities/rule eligibility before monetary claims.
- Planned Touch Files: `ima/betting_contracts.py` (new), `ima/rank_distributions.py` (reuse Phase5 core; pool-specific adapters), `ima/pipeline_graph.py`, `scrapper/official_corpus.py` (only validated official pool extraction), `tests/test_betting_contracts.py` (new), `tests/test_rank_distributions.py` (extend Phase5 tests), `tests/fixtures/research_expansion/`.
- Commit: `feat(research): add official pool contracts and joint finishes`.
- Tests: exhaustive small-field probabilities/symmetry, extremes, official small-field/deadheat/withdrawal fixtures.
- Success Criteria: correct qualification/units, coherent quinella/trio, no invented prices and explicit assumptions.
- Checklist:
  - [ ] Version effective pool rules and audit actual quote/dividend historical coverage.
  - [ ] Implement stable baseline and past-only position-correction experiments.
  - [ ] Reuse Phase5 performance/rank distribution nodes; extend pool settlement/eligibility and compare joint metrics, avoiding a second independent simulation engine.
- Handoff: probability/qualification report and source evidence for payout conversions.

### Subphase 7.2: Terminal EV, settlement and graph experiments
- Objective: simple honest paper reporting and model-mixing betting research.
- Planned Touch Files: `ima/betting_ev.py` (new), `ima/betting_settlement.py` (new), `scripts/evaluate_betting_ev.py` (new), `ima/research_targets.py`, `ima/research_evidence.py`, `ima/openrouter_orchestrator.py`, `tests/test_betting_ev.py` (new), `tests/test_betting_settlement.py` (new).
- Commit: `feat(research): report paper win place quinella and trio EV`.
- Tests: HK$10 arithmetic/unit conversion; unknown/stale/final quotes; refunds/deadheats; independent ticket selection/backtest no-lookahead.
- Success Criteria: CLI distinguishes quoted/scenario/fair-price/ex-post modes; monetary totals reproduce exact fixtures.
- Checklist:
  - [ ] Implement tested --stake/--currency/--pool/--quote-mode contracts and readable output.
  - [ ] Add chronological strategies/settlement, correlated exposure, coverage/uncertainty and drawdown.
  - [ ] Let planner select model/distribution/strategy graphs in experimental lane; preserve fundamental-first metrics.
- Handoff: source-backed win/place/quinella/trio reports, no live betting.

### Subphase 7.3: Fractional Kelly and correlated portfolio research
- Objective: implement the Benter/Thorp sizing research contract downstream of verified probability and payout contracts, without live betting.
- Inspect: local papers and recorded equation/assumption map, outcome distributions, settlement adapters, exposure and quote eligibility.
- Planned Touch Files: `ima/betting_stakes.py` (new), `ima/betting_contracts.py`, `ima/betting_ev.py`, `ima/betting_settlement.py`, `scripts/evaluate_betting_ev.py`, `ima/pipeline_graph.py`, `ima/research_evidence.py`, `ima/openrouter_orchestrator.py`, `tests/test_betting_stakes.py` (new), `tests/test_betting_ev.py`, `tests/test_betting_settlement.py`.
- Commit: `feat(research): evaluate fractional Kelly and joint ticket portfolios`.
- Tests: `.venv/bin/python -m unittest tests.test_betting_stakes tests.test_betting_ev tests.test_betting_settlement tests.test_rank_distributions`; chronological fixed-stake/fractional/full controls, solver failure, unit rounding and sampled-tail convergence fixtures.
- Success Criteria: single-ticket solver matches analytic Kelly; correlated portfolios satisfy budgets and maximize the checked scenario objective; uncertainty/impact/quote gaps stay explicit; no wagers submitted.
- Checklist:
  - [ ] Implement long-only analytic single-ticket baseline, policy schema and configurable research fractions.
  - [ ] Construct joint scenario returns for overlapping win/place/quinella/trio tickets, including cash, refunds and dead heats.
  - [ ] Use SciPy solver with convergence/constraint checks; test single-ticket reduction, duplicate tickets, mutually exclusive outcomes and jointly winning tickets against exact small-grid controls.
  - [ ] Test p=0/1, no positive edge, d<=1, missing prices, insufficient bankroll, legal increments and post-rounding exposure/growth.
  - [ ] Compare frozen chronological policies under calibration/payout stress and own-impact assumptions; log expected growth, realized bankroll/drawdown and coverage rather than just ROI.
  - [ ] Expose optional portfolio_sizing graph/output action to the planner with policy-bound freedom and clear paper-only status; leave chance-constrained mixed-integer research optional.
- Handoff: equation/page traceability, solver diagnostics, exact arithmetic cases, correlated portfolio report and no-lookahead bankroll backtest.

### Subphase 7.4: Close Planner-To-Paper-Research Integration
- Objective: finish the Phase 7 action boundary identified by the independent 2026-10-04 fulfillment audit. A working standalone EV CLI is not evidence that the planner can request or learn from paper research.
- Proven gap: `PlannerDecision` and `PipelineRecipe` reject betting-policy fields; the controller does not dispatch paper EV/sizing work or return those reports in its evidence. Predictive joint-order graphs are implemented separately.
- Planned Touch Files: `ima/research_betting.py` (new), `ima/research_expansion.py`, `ima/research_telemetry.py`, `ima/openrouter_orchestrator.py`, `ima/research_scheduler.py`, `ima/research_executor.py`, `scripts/run_research_expansion_canary.py`, `tests/test_research_betting.py` (new), `tests/test_research_betting_adversarial.py` (new), `tests/test_research_expansion_betting.py` (new), `tests/test_research_expansion_canary.py` (new), `tests/test_research_executor.py`, `tests/test_research_expansion.py`, `tests/test_research_expansion_resources.py`, `tests/test_research_telemetry.py`.
- Ownership: paper action contract/engine is separate from controller integration; both reuse existing outcome-distribution, EV and Kelly APIs. The controller remains the only durable decision/dispatch owner.
- Commit: `feat(research): dispatch planner-selected paper betting studies`.
- Contract: typed paper-only requests cite current completed attempts and immutable evidence; ensemble inputs must have compatible target, population, probability basis, dataset and protocol. No arbitrary paths, sources, Python, credentials or wagering. Missing quotes produce fair-price-only output; explicitly hypothetical scenario payouts never become historical market observations.
- Integrity repair: freeze the persisted prediction-file SHA256 in execution lineage. Preflight and workers require that checksum in addition to protocol and population checks; older outputs without it are not silently admitted as frozen paper-study inputs. This protects against probability-only edits that leave runner identities unchanged. It does not change model fitting, labels or scores.
- Dispatch: persist accepted actions before asynchronous execution; keep research allocation distinct from fit concurrency and the 80/20 predictive lanes. Bound and advertise action queues, admit work through measured resources, reconcile retries idempotently, and drain pending actions on STOP.
- Feedback: store report, source attempts, assumptions, coverage, quote mode, sizing diagnostics and failure status; include compatible completed paper results in subsequent planner evidence. Link a no-extra-LLM-cost `betting.evaluate` trace to its origin decision.
- Tests: `.venv/bin/python -m unittest tests.test_research_betting tests.test_research_expansion tests.test_research_expansion_resources tests.test_research_telemetry`; then the complete suite and Linux canary.
- Checklist:
  - [ ] Validate typed requests and comparable multi-model probability pooling before any decision mutation.
  - [ ] Reuse joint-order, fair-price, scenario EV and correlated fractional-Kelly machinery without fabricated quote history.
  - [ ] Execute and reconcile queued paper actions durably, with bounded resource admission and clean shutdown/recovery.
  - [ ] Return action results in planner evidence and verify actual origin-linked traces without duplicate cost.
  - [ ] Test mismatched populations, tainted probabilities, missing prices, duplicate/replayed actions, failures and STOP.
- Success Criteria: an accepted agent-selected paper study executes using stored development predictions, writes a replayable report, becomes visible to the next planner decision, and never submits a wager. New live campaign source is pinned only after this integration passes.
- Handoff: exact action schema/API, deterministic report and execution/recovery evidence; original dataset producers and prior campaign identities remain unchanged.

## Phase 8: Verification And Approved Rollout
### Subphase 8.1: Adversarial integration and measured performance
- Objective: prove mechanisms together before replacing a campaign.
- Planned Touch Files: `scripts/run_research_expansion_canary.py` (new), `scripts/benchmark_discovery_scheduler.py`, `scripts/benchmark_discovery_features.py`, `tests/test_research_expansion.py`, `tests/test_research_executor_v6.py` (new integration and model replay gates), `tests/test_research_expansion_resources.py` (new queue/resource recovery gates), `docs/AGENTIC_RESEARCH_EXPANSION_VALIDATION.md`.
- Commit: `test(research): verify expansion discovery replay tracing and EV`.
- Tests: `.venv/bin/python -m unittest discover -s tests`; deterministic CLI above; Armageddon matrix; full-history matched benchmark; actual MLflow UI/API readback.
- Success Criteria: every acceptance criterion has concrete proof, no unresolved must-fix defect or fixture-only behavior claimed live.
- Checklist:
  - [ ] Cover old/new targets/adapters/cache cold/warm; novel feature reused in several graphs; selection>32; timed race/workout/trial source coverage and speed-target admission; shared/conditional scale and rank-adapter parity.
  - [ ] Deterministic replay rtol=1e-10/atol=1e-12; nondeterministic bound measured before comparison, not relaxed to hide regression.
  - [ ] Equal-safe-concurrency repeated benchmark: throughput, cache fit counts, stage/runtime, cgroup peak/PSS/USS/I/O/pagefaults. Warm preparation reuse required; investigate unexplained regression before enabling default.
  - [ ] Restart builder/controller/worker/uploader; reconcile budgets/cost/model versions; verify strict events, packages, refreshed champions and native USD.
- Handoff: complete evidence report, source-specific unresolved limits distinguished from implementation gaps.

### Subphase 8.2: Isolated successor, monitoring and rollback
- Objective: launch only an approved immutable own-user successor and verify ongoing feedback.
- Planned Touch Files: `config/agentic_research_expansion.json` (new), `deploy/systemd/ima-research-expansion-supervisor.service` (new), `deploy/systemd/ima-research-expansion-supervisor` (new), `.github/workflows/research-expansion.yml` (only if existing workflow does not cover the gate), `scripts/optimize.py` (V6 CLI integration), `docs/AGENTIC_RESEARCH_EXPANSION_RUNBOOK.md` (new), `docs/AGENTIC_RESEARCH_EXPANSION_VALIDATION.md`.
- Commit: `docs(ops): deploy verified research successor with rollback`.
- Tests: own-user remote canary, source/data/model readback, multiple real planning/execution windows, resource/outbox/trace checks and rollback dry run.
- Success Criteria: paid planner direction leads to completed graph trials and latest evidence/cost visibility; protected services unchanged.
- Checklist:
  - [ ] Obtain explicit launch/name/policy approval; preserve/cancel predecessor work only as authorized at a safe boundary.
  - [ ] Transfer pinned release/dependencies via Tailscale; immutable campaign paths and backups; no other users/repos/firewall changes.
  - [ ] Start conservatively and increase against sustained measured headroom within approved near100GB envelope.
  - [ ] Verify multiple actual decisions/action types/budgets and a completed composed graph model with tracking/replay.
  - [ ] Rollback stops only successor, preserves all evidence, restores approved prior invocation without identity edits.
  - [ ] Push/merge/deploy as authorized; report local/merged/deployed/live separately; whole implementation final-gate only after evidence.
- Handoff: approved SHA/campaign/run/trace/model links, resource timeline, unchanged protected state and tested rollback.

## Execution Order And Done Rule
Controlled planner requalification: the failed `52a` acceptance root remains frozen with its original unavailable cost, no invented zero and no automatic retry. Its absent generation ID is a historical billing-information gap, not proof that the repaired endpoint is unusable indefinitely. After final-source CI, Linux fixture/readback and official-data control gates, the parent may authorize one new isolated acceptance invocation on the new immutable release, using the production 300-second absolute deadline, credential-free transport phases, early generation-header receipts and full response-cost persistence. Use a fresh root and invocation marker; retain/link the prior unknown-charge incident in the acceptance evidence. If the new call also lacks reported cost, freeze it immediately and do not continue paid cycles. Existing campaign-local unknown-cost admission remains fail-closed; do not transplant or rewrite the failed ledger to bypass that boundary. This is a manually bounded requalification within the user's existing setup authorization, not an automatic paid retry policy or a provider-enforced billing cap.

Official-data startup remediation (2026-10-04): the bounded control remained in preflight with no accepted decisions. `validate_fold` used `DataFrame.iterrows()` on a race-order frame inheriting a large dataset manifest. A metadata-bearing versus metadata-free diagnostic reproduced repeated row-level metadata copies with identical positions. Replace that position-map loop with direct race-ID iteration, preserving race order and leakage checks; verify metadata-bearing folds and immutable Linux startup before further rollout. Planned touch surface: `ima/research_evaluation.py`, `tests/test_research_evaluation.py`. Preserve the slow pinned control and rejected first-launch evidence; do not hotpatch its source or change the protocol.

Official-control resource remediation: default rich seed programs request no feature discovery, but their base-feature preparation was estimated as `feature_generation` rather than `preparation`. At 171782 rows this caused conservative estimates of 7.89/6.34 GiB before controller residency, preventing either job within the isolated 8GB unit; raising that unit to 12GB failed the reserved-host headroom gate while V5 remained active. Classify actual generation (discovery or new feature definitions) separately from base preparation using the existing estimator stages, retain safety margins and measured peaks, and prove actual official fits under the small cap before rollout. Do not reduce all estimates globally, bypass host admission, or alter protected V5 caps. Planned touch surface: `ima/research_expansion.py`, `tests/test_research_expansion_resources.py`.

Implementation guard decisions from the source and adversarial audit: V6 freezes shared entity histories before the meeting, retains validated categorical context separately from numeric formula inputs, and uses explicit pound units for official weight fields. Confirmation races are frozen from the source before filtering and inherited across successors; catalog hashes participate in immutable dataset identity. Unknown planner cost blocks new paid planning. Explicitly delayed result publication and dead-heat races unsupported by the current single-winner likelihood are excluded as whole races with reasons, never silently relabeled; original evidence remains in the official snapshot. These exclusions and publication assumptions must appear in the dataset report and are not claims of complete point-in-time coverage.

1. Baseline -> event normalization/joins/source-specific speed -> dataset lifecycle/replay -> formulas/selection -> graph core/ensembles/speed-distributions/conference-probit -> controller/telemetry -> pool adapters/EV/Kelly -> end-to-end gates -> approved rollout.
2. Contract work may be prepared independently after baseline, but land tested atomic units in dependency order. One controller owns each campaign ledger; no detached proposal/trainer loops without automatic feedback.
3. New modules/CLI options/tests named here are planned, not claimed to exist. Implement and test --help/contracts before invoking planned commands as proof.
4. Audit existing queue-memory/acquisition work before adding owners; preserve and reuse completed pieces, carry open requirements into this execution.
5. Execution is done only when requested mechanisms work and V6 is read back from the approved real system, not merely schemas/tests or a running process. An authentication/deployment blocker must be reported explicitly; it does not turn a local change into a live deployment.
