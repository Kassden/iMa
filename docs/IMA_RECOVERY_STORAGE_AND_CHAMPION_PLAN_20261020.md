# iMa Recovery, Storage Lifecycle And Champion Improvement

## GOAL
Safely recover iMa-owned disk space, prevent recurrent storage exhaustion, and run an evidence-led discovery pipeline that tests useful features and model components against the true champion, with independent later confirmation.

Status: consolidated plan only, 2026-10-20. No server cleanup, code implementation, service changes or campaign rollout are performed by this document. The user's implementation intent is recorded; this planning turn does not override runtime execution restrictions.

## Canonical Scope And Source Reconciliation
This is the SINGLE forward execution plan for cleanup, storage lifecycle and champion improvement. It replaces their separate execution sequences, not their evidence. Do not execute the old plans in parallel or create separate janitors/planners.
- [Champion improvement plan](CHAMPION_GUIDED_RESEARCH_IMPROVEMENT_PLAN_20261018.md) and its [independent review](CHAMPION_GUIDED_RESEARCH_PLAN_REVIEW_20261018.md) remain historical design evidence.
- [Storage audit](IMA_STORAGE_RECLAIM_AUDIT_20261019.md), [audit plan](IMA_STORAGE_RECLAIM_AUDIT_PLAN_20261019.md) and its exact-path manifest under .tmp/ima-storage-audit remain immutable observation/approval history, not an automatically approved purge list.
- The earlier "332 GiB iMa footprint" covered research-v2 only; full readable imaopt home was approximately 694 GiB, including acquisition. Do not repeat the narrower value as total iMa usage.
- Latest verified comparable MLflow best, checked October 20: 2.149415373890447, not the older 2.149418422667394. Run c714445813e441cd822e74dcc62d42b9; attempt-09e8faabf8bcc228e8e4dce7d54b628502a894767ff4b54cfa1c90cec1087d9c. It is a 35% Benter / 65% boosted fundamental pool, recorded October 15; the difference is negligible selected-development progress.
- Fresh execution must resolve the then-current compatible champion and host state. No new live host-health claim is made in this consolidation.

### Obligation Crosswalk
| Earlier work | Consolidated obligation |
| --- | --- |
| Champion plan 1.1 recovery/reference | 1.1 safety/provenance/canary; 2.1 reclaim; 3.1 admission |
| Champion plan 2.1 / 2.2 feature promotion/transfer | 5.1 / 5.2 |
| Champion plan 3.1 / 3.2 components/forward pools | 6.1 / 6.2 |
| Champion plan 4.1 / 4.2 availability/fitness | 7.1 / 7.2 |
| Champion plan 5.1 confirmation | 8.1 |
| Champion plan 6.1 resource/storage optimization | 3.1 / 3.2 / 4.2 |
| Champion plan 6.2 telemetry/rollout | 9.1 / 9.2 |
| Storage audit candidate identification/evidence | Input to 1.1 / 2.1; historical audit completion is NOT cleanup completion |
| Storage hygiene proposals | 3.1 admission; 3.2 bounded cache; 4.1 objects/CSV; 4.2 deltas; 4.3 retention/archive |

## Acceptance Criteria
- [ ] Cleanup touches only explicitly verified imaopt/iMa artifacts; retained models, raw evidence, data, MLflow, cache locks and other users/services are unchanged.
- [ ] Revalidated first-wave candidates are reclaimed with before/after allocated/free-byte receipts and production-package/registry readback. Estimates never substitute for measured reclaimed space.
- [ ] Every artifact-producing worker participates in one atomic filesystem reservation authority; ENOSPC, crash recovery and startup order are tested before unrestricted restart.
- [ ] One shared bounded cache lifecycle evicts only regenerable, unpinned payloads, preserving lock inodes and resume semantics.
- [ ] New immutable snapshot/data versions reuse unchanged objects; feature candidates store parent-plus-delta data; CSV is on-demand in a versioned format, without breaking legacy manifests.
- [ ] Strongest compatible untested feature bundles persist, reach the planner and complete paired fixed-champion comparisons, including negative controls and useful negative results.
- [ ] Pure-component ablations, boosted-primary tuning and forward-trained model pools are actually executed; selection exposure and inference/package lineage are explicit.
- [ ] Fitness predictors have independently reviewed identity, timing, units and usable coverage; unsupported archives remain prospective-only, not backdated.
- [ ] Untouched/prospective confirmation has a locked study/forecast lifecycle, predefined estimand, sample and decision rule; insufficient future data stays pending.
- [ ] MLflow, terminal status and the planner agree on graph identity, comparable best, actual completions, transfers, budget allocation, storage health, native token/USD fields and blockers.
- [ ] Sealed own-user rollout passes failure injection, actual consumer readback and independent review; no guarantee of a numerical or betting-profit improvement is implied.

Functional readiness, reclaimed capacity, development gain and statistically confirmed gain are four different outcomes. Report them separately; no automatic wagering promotion.

## Runtime Contract And Decision Record Policy
Current deliverable is a plan. Preserve completed audit/review evidence and any still-open original obligations; do not replace an unfinished execution goal with this smaller documentation task.

Future implementation task: ima-recovery-storage-champion-20261020. Before implementation, pin actual-outcome assertions for EVERY numbered subphase and register the full goal against this canonical file in execute mode using mega_task.py. Use the real execution session, not a copied historical session ID. The supplied planning session is 01a0c46f-3ae6-79f3-b206-199aa3eff2f6.

Future registration command, only when execution is permitted:
~~~sh
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/milkingthesun/.codex/skills/megaskill/scripts/mega_task.py --root .mega register --id ima-recovery-storage-champion-20261020 --plan docs/IMA_RECOVERY_STORAGE_AND_CHAMPION_PLAN_20261020.md --goal "Safe iMa-only reclaim, bounded storage lifecycle and champion-guided research with independent confirmation" --mode execute --session SESSION_ID --acceptance .mega/ima-recovery-storage-champion-acceptance.json
~~~
Do not run the older champion plan's registration separately. Each obligation requires current linked test AND readback proof, then an explicit done event. Qualitative source, safety and scientific interpretation require independent review. Staged code, passing fixtures or heartbeats cannot certify live fulfillment. Do not claim hook coverage without actual same-session receipts.

Record consequential choices in existing mega_decision.py: containment/cleanup selection, emergency bootstrap, archival authority, cache policy, versioned object identity, comparison eligibility, pool selection exposure, confirmation, billing and handover. Include strongest alternative, assumptions, self-critique, disconfirming test and reversal condition; promote durable accepted choices to docs/adr.

At most three owned delegates: disjoint files, no recursive fan-out, branch changes or child commits. Parent owns integration and live operations. Independent verifier checks original criteria and evidence, not presence of headings. Collect results and actual native closure receipts.

## Root-Cause Baseline
Evidence is dated. October 18/19 host observations require refresh before action; scientific audits describe historical evaluated populations, not current services.

| ID | Observed evidence / classification | Mapped corrective work |
| --- | --- | --- |
| S1 | October 19 filesystem: zero available space; readable imaopt home approximately 694 GiB. Acquisition snapshots approximately 331 GiB and research-v2 approximately 331 GiB. Other users' usage is not fully readable. | Scoped reclaim plus growth control; phases 1-4. Never assign inaccessible remainder to iMa or clean other users. |
| S2 | Audit: 3,573 exact fold payloads = 163,216,367,616 bytes (152.007 GiB); 20 conditional incomplete builds = 1,612,935,168 bytes (1.502 GiB). Stable lock/control metadata excluded; no matching own open mapping at capture. | Revalidate, production canary and lock-aware selected reclaim; phase 2. Observation does not certify future quiescence. |
| S3 | Other caches 5.954 GiB; 315 historical snapshots 324.885 GiB remain conditional. Snapshot hash claims imply an alternative 51.242 GiB dedup opportunity, not independently verified and overlapping snapshot removal. | Archive/reconstruct or qualify immutable reuse; phase 4. No additive or blanket-purge claim. |
| S4 | Old iMa checkout runs LIVE MLflow; audit counted 26,739 runs, 7,449 model versions, 17 experiments. Registry has 1,061 datasets, 1,059 descendants of shared parent. | Protect models/database/datasets; reachability in phases 1/4. Old path names do not mean unused. |
| S5 | Missing equivalence evidence: queried deployed overlay paths did not expose three cache/package modules; local two-fixture cache-eviction tests passed, but no production-package canary. | Discover actual import provenance and verify a real package before cache cleanup; phase 1. |
| S6 | Last observed main unit failed; feature unit repeatedly fails own-cgroup startup preflight. A status file was stale and reported unresolved planner cost. | Own-user startup/recovery, separate stale diagnostics from proven exit cause, reconcile costs; phases 1/9. Do not remove containment checks. |
| R1 | Code catalogue limit 8 plus recency ranking; audit top 15 screened bundles untested as full bundles and absent from planner shortlist. | Durable merit-aware promotion, phase 5. |
| R2 | Cheap Benter screen baseline 2.225279, best screened bundle 2.194613; rich pool best now 2.149415. | Distinguish proxy versus incremental champion merit; phase 5. No assumed transfer. |
| R3 | Audited window: 83 Benter-primary, 12 probit-primary, one boosted-primary; dominant boosted component and weights often fixed. | Ablation/component search and forward pooling, phase 6. Not proof fixed settings are bad. |
| R4 | Frozen data: 171,782 runners, 13,924 races; 110 event/workout/trial columns have no variation. Same 1,502 development races end 2025-05-18. | Supportable fitness signal and independent later evaluation, phases 7/8. |

Likely causes: unbounded derived storage, full snapshot duplication, ineffective feature evidence selection and saturated-component tuning. Possible additional causes: incomplete fitness sources, cohort errors, parameter ranges and convergence. Unknown: every failure's exact timeline, irreducible model loss and archive destination availability. Disconfirmed: outer parameters are not simply ignored; similar but different recipes did not prove non-reproducibility.

Minimum evidence inventory: df/inodes, allocated inode/block inventory, UID and service/cgroup state, actual process identities/mappings/locks, timers, source/package import provenance, SQLite/WAL state, registry/model/MLflow roots, pending tracking/billing and comparison/selection lineage. Missing privileged evidence remains unavailable; it cannot establish safe reclaim.

Database recovery gate: before restarting an affected database writer or attempting any repair, obtain a transaction-consistent WAL-inclusive capture on healthy storage. Prefer SQLite's supported online backup from a valid source connection; if that fails, quiesce only the affected own writer and preserve its matching DB/WAL state using a documented consistent capture procedure. Never copy a live DB and WAL independently and call that a verified backup. Record method, hashes, transaction/capture times, integrity checks on the copy and application-level ledger/model/tracking readback. Preserve originals and isolate failed consumers. A missing healthy capture destination blocks affected writer recovery, not local tests or independently qualified cache-only reclaim that does not modify databases.

Mutation boundary now: local planning documents only. Future recovery initially collects read-only evidence to healthy LOCAL storage. Do not write diagnostics to a full server filesystem or repair a database before preserving available evidence. Every remediation maps to S1-S6/R1-R4 above; generic hardening is risk reduction, not a proven root cause.

## Regression Guardrails
- Planning edit surface: this consolidated plan, source-linked review and generated read view; no implementation or server mutation.
- Future edit surface: exact subphase lists below. No unrelated refactor, metadata churn or old evidence rewriting.
- Server allowlist: imaopt UID/home and verified owned paths under /home/imaopt/research-v2, /home/imaopt/acquisition and /home/imaopt/iMa. Path membership alone never authorizes deletion of protected contents.
- Protected: Cortex/Solar and all other accounts/repos/services; raw HKJC evidence/raw-import/baselines; registered datasets; saved packages/predictions/ledgers/decisions; MLflow DB/WAL/artifacts/environment; confirmation locks; stable cache locks; existing dirty checkout changes; fitted scientific identities.
- No broad sudo, account switching, other-user cleanup, symlink/mount escape, wildcard purge, or age-only "unused" classification. Reject changed path/device/inode/content manifests before action.
- Damage radius: systemic for implementation, small for planning.
- Branch strategy: dedicated implementation worktree/branch feat/ima-storage-champion required, created from a reviewed coherent base. Inventory and selectively carry required dirty patches with provenance; never reset/stash/revert user work.
- Consumers: acquisition, registry, feature worker, planner, executor, Optuna, model loader, MLflow and terminal operator.
- Proof plan: containment tests, actual package/data canaries, exact manifest arithmetic, reservation races, old/new format parity, scientific leakage/identity tests, native MLflow readback and independent live review.
- Atomic commits follow the numbered subphases. No commit/push of unrelated work. No hotpatch under unchanged campaign scientific identity.

## Research
Built-in owners: DatasetRegistry/FeatureLifecycle and existing digest/hash/atomic write helpers; PreparationCache.pin/evict; graph executor, hypotheses/ledgers, Optuna ask/tell, confirmation locks and tracking outboxes.

Primary sources checked October 18-20:
- [Joblib load/memory mapping](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html): compressed arrays cannot be memmapped; keep active working matrices uncompressed and trusted-own-user only.
- [PyArrow datasets](https://arrow.apache.org/docs/python/dataset.html): reuse partitioned Parquet/projection/scanning, not a new file-format engine. Application version/lineage/garbage-collection contracts remain our responsibility.
- [Featuretools time handling](https://docs.featuretools.com/en/stable/getting_started/handling_time.html): cutoff filtering and include_cutoff_time=False; historical publication availability still requires independent evidence.
- [scikit-learn stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html): no in-sample meta-training; default K-fold is not our race-aware chronological protocol.
- [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html): selection can bias evaluation; no claim about the exact magnitude in our model.
- [Optuna 4.2 parallel optimization](https://optuna.readthedocs.io/en/v4.2.0/tutorial/10_key_features/004_distributed.html), [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/): reuse installed-compatible search, constrained weights and run/artifact APIs.

## SOTA, Standards, And Best Practices
Implementation decision: extend existing storage/registry/training owners with atomic reservations, a small shared lifecycle index and versioned readers. Use mature Parquet, Joblib, SQLite/fcntl, Optuna and MLflow mechanisms, not new training or deployment frameworks. Immutable identities do not require physically copying unchanged data.

Reject blanket oldest-first deletion; compressing active memmaps; replacing immutable manifests in place; blindly hardlinking mutable files; interpreting archive moves on the same filesystem as freed space; deleting CSV from legacy manifests; introducing DVC/lakehouse orchestration solely to avoid defining reference contracts; random runner CV; score-label-trained pools; endless tiny tuning for activity; a second competing LLM planner.

## Dependency and Tooling Preflight
Inspect actual Python/dependency versions, disk/inodes, file ownership and deployed import paths before changes. Project Python >=3.11,<3.14; Megaskill tooling Python 3.14 is not the model runtime. Reuse requirements locks where applicable; Featuretools 1.31.0, feature-engine 1.9.4, Woodwork 0.31.0, Optuna >=4.2,<5 and MLflow >=2.17,<4 remain compatibility constraints, not upgrade requests.

No install is needed for planning. For implementation install or repair project-local tools using the supported locked workflow; fallback supported venv command: python -m pip install -e '.[research,features,acquisition,dev]' pytest. Stage large dependencies outside the full server filesystem and transfer via the permitted Tailscale route; do not depend on unrestricted mainland outbound downloads.

Tools: pytest, own-user SSH/systemd, read-only SQLite, /proc metadata, supported .venv, PyArrow/Joblib, MLflow client and plan/dashboard scripts. Browser smoke is required only for generated plan/changed MLflow views, not a new application UI.
Real blockers: SSH auth, unhealthy storage, absent external archive destination/credentials, inaccessible official archives, unverified publication, provider outage or an irreversible-loss boundary. Scope blockers to affected obligations; keep safe independent tests moving.

## Unified Architecture And Storage Contracts
Flow: official raw objects -> immutable snapshot manifest -> availability-safe parent dataset -> feature definitions/deltas -> cheap screen -> durable champion transfers -> component trials -> forward pool -> locked later confirmation -> research registry. A SHARED lifecycle/reservation owner supports all stages; scientific authority remains one strategic planner.

### Artifact Classes
| Class | Retention / removal rule |
| --- | --- |
| Raw evidence, baselines, source manifests, confirmation locks, run/decision/cost metadata | Protected; archive with proven retrieval before any approved local removal. No automatic semantic deletion. |
| Model packages and predictions referenced by MLflow/champions/research history | Retain or verified archival under explicit policy; versioned model history is not a cache. |
| Immutable snapshot/feature objects | Reuse by content identity; remove only after complete root traversal or verified archival. A version limit never overrides a live/historical pin. |
| Fold-preparation payloads | Bounded regenerable cache, LRU by actual access, evict only unpinned artifacts with stable-lock semantics. |
| Discovery/prediction caches and prepared descriptors | Separate reviewed rebuild/resume contract before eviction; not automatically covered by fold-cache classification. |
| Failed .building/scratch | Persist failure receipt, prove no producer/lease/resume/reference, then policy-authorized retirement; age alone is insufficient. |

### Reclaim Contract
Historical first-wave candidate capacity is 164,829,302,784 allocated bytes = 153.509 GiB, NOT guaranteed free space. Refresh the exact-path manifest and preserve actual selected hashes/device/inode/UID, modification signatures, open/lock status, reference reason, operation, authority and measured blocks. All historical approved=false flags remain unchanged; create a new selected execution manifest only when runtime permits execution and existing user authority covers that item.

First-wave scope: revalidated eligible fold payloads and independently qualified incomplete builds. Never remove whole cache roots or their stable .lock/owner/control files. Use existing pin/evict semantics; invalidate/rewarm stale preparation descriptors before resume.

The 5.954 GiB other-cache and 324.885 GiB historical-snapshot categories remain held by default. The 51.242 GiB hash-claimed dedup alternative overlaps snapshot removal and is NOT additive. Do not count the roughly 167 GiB historical CSV copies as extra savings on top of snapshots.

Preserve latest_verified, latest/direct-base and externally referenced snapshot anchors, configured imports/local-snapshot-008 baseline pair and shared raw-import. Discovery showed that raw-import is a real external verification dependency, not a failed build.

### Emergency Bootstrap And Ordering
A full disk creates a dependency cycle: new reservation/index writes need space, but routine workers cannot restart without them.
1. Collect evidence and code/package canary results to healthy local storage or an explicitly owned bounded tmpfs; do not need new server artifacts to authorize a cache eviction.
2. Quiesce only named iMa artifact-producing services/timers after recording their exact states and checking actual fits. Preserve/drain productive fits; never trust stale status or kill unrelated processes.
3. Use a prevalidated, sealed one-shot lock-aware reclaim path on the exact first-wave manifest; write per-item results to local healthy storage, not the full disk. Revalidate immediately before each irreversible step.
4. Measure resulting free blocks/inodes and verify preserved consumers. This is the only recovery exception before the reservation owner exists; it does NOT start training/acquisition.
5. Install/qualify storage reservation and bounded-cache safety before normal writers restart. If canary/equivalence or cleanup conditions fail, stop that affected action; continue local tests/design. Do not erase raw/models to force progress.

Deletion has no magical rollback. Cache rollback is verified reconstruction from preserved sources; data rollback requires proven archive retrieval. State this before action.

### Shared Disk Admission
One atomic reservation authority per output filesystem, used by acquisition, snapshot/feature builds, preparation, OOF, model fits and outboxes. Initial filesystem headroom: max(20 GB decimal, 5% capacity), configurable BEFORE rollout. Check free bytes AND inodes under a filesystem-scoped lock, subtract reserved-but-unwritten output/scratch, then admit or wait.

Reservation fields: job/attempt/stage, parent objects, output classes, byte/inode estimate, maximum scratch, owner UID/PID/start/boot identity, heartbeat, committed output and unconsumed reservation. Unknown jobs get a measured bounded canary reservation, never zero or permanent starvation.
- Monitor consumption; atomically extend before further writes. On failure, pause/checkpoint or safely interrupt only the unqualified job, preserve committed artifacts and report infrastructure—not scientific—failure.
- Temporary artifacts publish atomically only after complete checksum validation. Crash reconciliation verifies actual processes, artifacts and outbox state before releasing leases; expiry alone is not ownership proof.
- Gate newly introduced writers as well as existing ones: an unregistered artifact producer fails admission. Research-client MLflow uploads reserve their duplicate artifact bytes before transfer; the outbox/control reserve must not recursively depend on creating another unbounded reservation. Keep the live MLflow serving process available unless its own protected recovery explicitly requires downtime.
- Simulate competing workers taking the last capacity, underestimated output and another user consuming space. Own budgets cannot guarantee against unrelated disk growth; handle ENOSPC and fail closed without touching other users.
- A minimal diagnostic/control reserve is included in headroom. Diagnostic buffers/backoff cannot create unlimited new files or restart storms.

### Bounded Cache Policy
Initial policy proposal: ONE 50 GiB disk target shared across all qualified iMa fold-cache roots, explicitly independent of the 100 GB RAM budget. Soft-pressure eviction starts above 45 GiB and targets 40 GiB; settings configurable with recorded rationale.
- Record actual cache access separately from creation mtime; cost-aware ordering may retain expensive recent preparations after basic LRU is validated.
- Never evict active/pinned artifacts, pending scheduler leases, needed non-regenerable state or stable lock inodes. If pinned usage exceeds target, disclose over-budget and pause NEW cache-producing admission rather than destroy pinned work.
- Cache hits do not rewrite immutable payloads. Recheck filesystem headroom and reservation safety even below the cache budget.
- Existing uncompressed Joblib read-only arrays remain shared; compress cold archives, not active memmaps. Drop unused columns early and measure private copies/cache reuse.
- Preserve tiny useful probit warm starts unless separately qualified; slow productive trials are not rejected solely to make cache numbers smaller.

### Versioned Snapshot And Feature Storage
Introduce versioned manifest readers/writers without changing existing identities:
- Immutable objects are addressed by cryptographic content digest and include schema/units/producer identity. Write once via verified temporary publication. Separate content identity from semantic recipe/availability/version identity.
- Reuse unchanged raw/normalized/Parquet objects. Snapshot refresh depends on meaningful normalized-input/version changes, not a timer tick alone. Preserve new capture evidence when normalized values are unchanged; skip only unnecessary full materialization, not provenance capture.
- New canonical tabular output is compressed Parquet; CSV is an explicit bounded export with provenance, not automatically duplicated every snapshot. Legacy readers still require every legacy manifest-listed CSV/JSON file; no in-place file removal or manifest rewriting.
- Feature dataset = immutable parent reference + exact race/runner key map + added/replaced delta columns + definition/cutoff/provenance. Resolve only requested columns/fold rows using existing PyArrow/preparation mechanisms. Prevent row drops/reorders, key collisions, undeclared replacement and parent cycles; prove logical/materialized table equality.
- New model packages embed needed inference/replay state or pin immutable objects with validated restore; no silent dependence on evictable caches.
- Preparation keys include actual source/row/label/definition/selection/transform/code/environment dependencies. Model-only hyperparameters stay excluded only when they cannot affect preparation.
- Snapshot/data v1 -> new readers/writers is an additive rollout. Verify old and new schema/values, folds, feature hashes and predictions; no storage-format improvement is allowed to change scientific scores silently.

### Retention, Reference Graph And Archive
Mark protected roots from all own services/timers, current/queued/reserved work, registered datasets/feature lineage, model/package/MLflow references, champions, confirmation locks, raw provenance and pending publication. Reachability and leases are authoritative; a reference count or no grep hit alone is not enough. Serialize sweep versus pin/publication and reconcile after crashes.

Keep newest two verified snapshots PLUS every pinned reference locally unless a validated archival reader can restore them. This is a hot-storage target, not permission to delete unique unarchived history. Retain compact manifests, code hashes, lineage and decisions. New unreferenced immutable objects have an initial seven-day grace; this does not apply retroactively as proof that old snapshots are garbage.

For historical archive candidates: choose an iMa-owned external filesystem/object-store prefix, reserve transfer scratch, verify complete checksums and perform representative restore/readback through actual consumers. Record exact restore recipe, credentials boundary, retrieval cost/latency and archive object/version hashes. No destination or failed restore means HOLD, not delete. The original cleanup request covers qualified derived reclaim; unique historical loss still needs an explicit accepted archival/loss decision. No other-user changes.
A central lifecycle job reports dry-run reclaim and blocked references before acting. Future automatic actions are limited by the qualified class policy; model/raw/history loss is not silently delegated to LRU.

## Scientific Contracts And Planner Freedom
Primary target: fundamental race probabilities with current-race market blending disabled. Preserve ranking, placing, speed/time and odds targets as exploratory options with native labels, adapters and separate comparisons. Historical odds predictors stay explicitly labelled; fundamental does not falsely mean no historical odds information.

One strategic planner chooses hypotheses, models/components, target, typed features/transforms, trial budget, continuation and stopping. Deterministic code owns validation, execution, resources, dedup and accounting. No 80/20 rule, four-model restriction, fixed 260-trial lifetime or budget forced to equal/multiply concurrency. Maximum 26 full fits is only a ceiling; combined main/feature/preparation RAM <=100,000,000,000 bytes. Existing 92/8 partition is initial reference, not forced utilization. Report CPU percentage with denominator separately from load average.

Expensive probit: preserve productive-progress deadlines, initial four-hour policy and comparable measured adjustments. Warm starts only from appropriate training folds. Timeouts are censored lower bounds; profile NumPy/gradient/integration work before adding cores. Do not rebuild unnecessary prep or exclude cold expensive families forever because measurements are missing.

### Comparison And Hypothesis Identity
Run identity binds exact source/parent/child dataset, metric/target/direction, full graph/recipe, scored row and labels hashes, fold/cutoff/availability, component feature routing, code/environment/seed, package/prediction hashes and baseline.
Paired eligibility separately proves same scored population, labels, folds and all undeclared settings. Derived datasets differ only in explicitly declared definitions; verify unchanged parent columns. Do not globally ignore dataset_ref to bypass matched-control failures.

Win metric is mean per-race -log(p_winner), probabilities sum to one. Preserve existing supported dead-heat/exclusion semantics. Lower is better; candidate-minus-control negative means improvement. Retain fold/race values and coverage, not one scalar.

Hypothesis/promotion record: ID, source/rationale, definitions/family, parent/reference, control, changed axes, destination nodes, screen contract/delta/uncertainty/support, redundancy group, transfer state, matched outcome, artifact/run links, rejection reason, next action. Reuse existing research_hypotheses/feature_lifecycle; add a small promotion owner only where needed.
States: built -> screened -> transfer_queued/running -> champion_tested -> promising/rejected/inconclusive -> confirmation_locked -> confirmed/not_confirmed. Infrastructure failures differ from negative scientific results. New champion means new reference-specific evidence; never overwrite old outcomes.

### Discovery And Champion Transfer
Deterministic feature validity/units/availability/coverage checks plus Featuretools/registered typed definitions run independently of model capacity. Planner-authored formulas use the same typed vocabulary; arbitrary generated Python in live workers stays out of scope.

Cheap baseline-v1 screen remains a continuity control, not proof of rich-model merit or universal eligibility. Permit justified direct transfers for interactions/nonlinear effects the linear screen misses.
Rank untransferred candidates within identical screen contracts by ascending paired delta, then interval width when available, then content ID; missing uncertainty is not zero. Select strongest representative per family signature, then fill remaining view from same ranking. Add a small recent lane only if it does not displace the strongest; summarize negative evidence. No recency-only eight-item semantic ceiling.

Durable queue prevents best evidence disappearing. Initial diagnostic panel: three strongest compatible distinct-family COMPLETE bundles, then family/single-feature removals and one null/duplicate control. This is not an ongoing planner cap. Freeze champion graph/hyperparameters/weights and explicitly route added features to Benter, boosted or both. Only declared feature/dataset axes change.

Before proposing more work, inspect per-race loss by field size, venue, distance, class, rest, history support and entropy with counts/uncertainty. Exposed development diagnostics guide falsifiable hypotheses, not independent confirmation. Proposal requires rationale, control, axes, expected failure slice, chosen budget and falsifying outcome. Local semantic preflight rejects invalid paths/axes before paid dispatch, not harmless explanatory words.

### Component Search And Forward Pools
Run Benter-only, boosted-only and old 35/65 controls on identical populations. Log effective node parameters and selected columns, not just outer model_kind.
Tune boosted with Benter fixed, then separately Benter with boosted fixed. Include exact original configs and valid parameter boundaries. Joint search is separately identified. Useful experiments per CPU/disk/cost matter more than near-identical proposal counts; the planner explicitly stops saturated studies.

Within each outer training history generate expanding inner splits of whole meetings; fit transforms/models on strictly earlier meetings. Omit earliest unsupported OOF rows consistently. Cache predictions only with verified fit AND selection lineage. Simplex nonnegative weights sum to one, fitted by mean race log loss using SciPy; compare pure/equal/35-65 controls and allow zero weights.
Outer scored labels cannot fit weights or select meta-regularization. Refit components on permitted outer training, apply frozen weights to later evaluation; dedicated calibration stays separate. Serialize complete graph/packages/adapters and prove deterministic reload.

Two modes:
- development_selected: settings already selected using outer development outcomes. Forward fitting avoids in-sample weight training but score remains selection-exposed.
- nested_selection: features/config/graph selection confined to each outer training history; record selection population/cutoff as well as fit cutoff; reject selection_cutoff >= outer score start.

Initially use inexpensive development_selected diagnostics, then untouched locked confirmation; do not call it an unbiased nested estimate. Later/current-race market blends remain separately named diagnostics, not fundamental progress or automatic EV proof.

### Historical Fitness And Clean Data
Use HKJC official evidence by default. Retain occurrence, verified publication, first observation, correction version and eligibility separately. Capture today does not prove availability before a historical race.
- Prior-race ability: speed, recency trend, within-horse variation, distance/surface/venue comparability and support.
- Workload: counts/recency/type/distance/track and changes over 7/14/30/90 days.
- Trials: placing/field size, explicitly supported individual times, comparable conditions and recency.
- Health/interruption: veterinary incidents/clearances, movement/arrival/acclimatization and layoffs.
- Reliability: coverage/support and sparse-history shrinkage; distinguish performance variability from epistemic uncertainty.

A trial/batch winner time is not every participant's time. Missing coverage is not zero workouts/incidents. Track/class/going adjustments, residual models and shrinkage fit only earlier training; never infer a race's variant from its own outcome. Keep horse-ID and measurement-unit contracts explicit.

For each family publish source/eligible/unresolved counts, availability tiers and varying/nonmissing/support by season/venue/class. Independently compare representative raw pages to parsed rows/features. Include missingness-only controls and cohort-bias checks. Unsupported archives remain prospective-only; continue supportable prior-race work.

Build smaller cleaner successor data if necessary; replay unchanged champion on the SAME new population before comparing. A different population's 2.14 cannot be called absolute improvement over 2.149415. Transfer shared definitions across models while allowing learned weights to differ.

### Untouched Confirmation
Maintain exposure ledger across training/tuning/features/planner/pooling/manual betting research. Already-seen later seasons are retrospective replication. If no trustworthy unused historical slice remains, confirmation waits for future meetings; other qualified research continues.

Default fixed-horizon design: one primary challenger; no interim outcome peeking or optional stopping. Mean paired race-loss difference; resample whole meeting blocks, using total loss difference / resampled race count, not equal meeting means. Two-sided 95% percentile interval, 20,000 resamples, seed 20261018.
Research acceptance: point delta <=-0.001 nats/race AND upper interval endpoint <0. This is an explicit engineering threshold, not profit or proof the TRUE gain exceeds 0.001. Plan sample using exposed development block differences centered/shifted to -0.002, 2,000 pilot simulations; choose smallest fixed meeting count >=30 with >=80% estimated power for that exact criterion. Record Monte Carlo uncertainty/assumptions and freeze before labels. If pilot insufficient, revise BEFORE lock, not after outcomes.

New study-lock schema preserves legacy single-file locks:
1. Freeze fitted challenger/reference packages/weights/definitions, eligible population, no-refit policy, horizon, exclusions, forecast coverage and decision rule before confirmation.
2. Historical inputs may update strictly as of forecast; learned parameters remain frozen. Seal immutable per-race/meeting runners, inputs and predictions at registered deadline, initially five minutes before scheduled post, using timezone-aware instants. Last valid pre-deadline supersession binds; preserve all versions.
3. Cancelled races and post-deadline scratches excluded under frozen rules with reason ledger; missing/late forecasts are operational misses, not outcome-selected exclusions. Less than 95% forecast coverage makes the study operationally unqualified.
4. Store labels separately; planner gets collection/coverage status only until fixed horizon. Aggregate sealed manifests without outcome-dependent inclusion and verify seals predate results.

Every new challenger needs fresh untouched data or a predeclared multiplicity policy. Intervals crossing zero are inconclusive. Secondary targets require native labels/contracts. Application quarantine is a tested information-flow contract, not a security claim that same-user code cannot access files.

## Observability And Naming
MLflow attempt = one graph/model configuration across folds; parent program groups related attempts. Feature build/screen/champion-transfer and confirmation runs have linked hypothesis, definitions, dataset/control/reference IDs. Storage reservations, archive/reclaim batches and cache evictions have separate operational records; do not count them as model trials.

Stable names: campaign/<id>, program/<id>/<hypothesis>, attempt/<id>/<graph-family>, feature/<definition-set>/<stage>, confirmation/<lock>, storage/<operation-id>. Keep old names/history; use linkage tags, not silent migration.

Decision traces show reason for call, true full-graph champion, eligible best by contract and delta, accepted/rejected proposals/reasons, chosen/allocated/held/continuing budget, actual running/completed/failed, transfers, disk reservations/cache pressure and next action/blocker. Heartbeat != planner decision. best=None requires no eligible result and explicit reason.

Native MLflow token/USD fields must populate actual trace columns where provider returns usage, not just a prose string. Preserve request/generation/provider/model IDs, input/output/cached/reasoning tokens, billed vs estimated USD and known/estimated/pending state. Never invent zero or double-bill retries. Reconcile pending accounting before existing safety policy; changes require explicit tests/decision. No new arbitrary spending or local hard context cap.
Compact context is a retrieval-backed view over durable evidence, including controls and safety fields. Trigger paid decisions on meaningful results/decision need, not every pending-fit heartbeat. Planner sees negative results and pending controls, not only optimistic summaries.

## Deterministic Real-User Test
Planned terminal workflow (new CLI surface, not existing commands): scripts/research_storage.py inspect/plan/status; explicit apply only with selected current authority-bound manifest. Extend feature_exploration.py promotions/compare/diagnose; reuse confirmation CLI with versioned study seals.
Fixtures: own/foreign/symlink roots, locked cache, stopped/restarted writer, full disk simulation, duplicate/delta tables, complete chronological races, useful/null/duplicate features, late event, known mixture and future study.
Command: .venv/bin/python -m pytest tests/test_ima_recovery_workflow.py -q (planned test).
User-observable assertions: exact safe candidate totals; foreign/changed/pinned candidates rejected; locks/models/source retained; no admission when reserve insufficient; unchanged snapshot objects reused; no automatic CSV; logical table/prediction parity; strongest untested bundle retained; correct tuned node; no score-label leakage; actual linked MLflow costs and honest negative results.
Real-user readback precedes broad suite. Browser smoke only for generated plan/changed MLflow table. Synthetic success is not proof of real-data gain.

## Fulfillment and Readback Proof
Final user-visible result NOW: one detailed consolidated plan, generated read view and independent substantive review. Required write is local planning artifacts only. Readback asserts source reconciliation, all three workstreams, explicit sequencing and unchecked future execution; file existence alone is not sufficient.
Future fulfillment: selected iMa-only bytes actually reclaimed; preserved consumers verified; storage admission/cache/version readers qualified; actual champion comparisons/ablations/forward pools; supportable fitness data; locked confirmation and live own services.
Evidence bundle: exact before/after source/UID/PID/inode/reference receipts; per-item reclaim + df/inode delta; cache/model canary; backup/restore; old/new format parity; full recipes/predictions/lineage; raw-to-feature audit; OOF selection cutoffs; confirmation exposure/seals/results; MLflow token/USD readback; multiple genuine live cycles; independent review.
Not done if staged only, samples presented as production certification, a created queue without outcomes, unknown archive restore, changed-population gain, or future-label wait hidden as success. Reclaim has no guaranteed number until measured.

## Armageddon Mode
Systemic attack scope: changed manifest/UID/device, symlink/mount escape, open/unlinked files, hardlink accounting, lock-inode recreation, cached descriptor resume, ENOSPC/lost WAL, full-disk bootstrap, crashes mid-delete/index/object publication/archive, pin/GC races, other-user growth, main-down cgroup and restart storms.
Scientific failures: duplicate AST, mismatched population/labels, unknown units, late publication/corrections, horse identity collision, batch time mistaken for individual time, fake OOF, future selection, score-label weights, invalid probabilities/weights and confirmation leakage.
Tracking failures: provider timeout, unknown/double costs, MLflow outage/outbox retry, best=None and heartbeats masquerading as progress.
Must-fix: any unapproved loss, protected-root mutation, false paired/progress claim, unsafe memory/storage overlap, corrupted existing identity, erased negatives or unverified restore. Record actual failure assertions and repaired readbacks. Do not install/test destructive behavior against production for convenience.

## Generality Guardrail
Reuse existing owners: hashes/registry/lifecycle, cache pin/evict, scheduler, hypotheses, graphs, Optuna, packages, confirmation and outboxes. Add missing general storage authority/promotion module, not new per-campaign janitors or competing trainers. Contracts keyed by identity, not hardcoded champion IDs, winning formulas or snapshot dates. Recurrence is high; one-off SQL injection, fake planner responses and filename-only cleanup are rejected.

## Ordered State And Dashboard
Canonical file: docs/IMA_RECOVERY_STORAGE_AND_CHAMPION_PLAN_20261020.md. Generated read view: .mega/dashboards/IMA_RECOVERY_STORAGE_AND_CHAMPION_PLAN_20261020.html. Use mega_plan_dashboard.py over canonical Markdown and exact-plan .mega/state.jsonl/.mega/evidence.jsonl; no hand-maintained HTML status. Historical audits/reviews stay evidence, not another execution queue.
All future implementation phases below start unchecked. The older audit's completed observation steps do not mark cleanup done. Planning-only completion is not execution certification.

## Phase 1: Evidence And Safety
### Subphase 1.1: Scope Lock, Current Health And Real-Package Canary
- Objective: resolve actual own consumers, import provenance, champion and safe mutation boundary before cleanup.
- Inspect: fresh df/inodes/process/mappings/locks, own units/timers/cgroups, audit manifest signatures, registry/MLflow roots, champion predictions/full recipe, source/dependency import hashes and billing/outbox state.
- Allowed edit surface: owned preflight/readback tooling/tests only; no purge or scientific changes.
- Commit: fix(ops): qualify iMa containment and production-package recovery baseline.
- Tests: existing cache-eviction/package tests plus new containment/stale-state/bootstrap canary tests; .venv/bin/python -m pytest tests/test_research_preparation.py tests/test_research_executor_v6.py tests/test_research_recovery.py -q.
- Success Criteria: fresh protected set and champion identity, actual module equivalence or explicit failure, real saved-package inference baseline and unchanged-input replay where possible. Affected SQLite/WAL has a consistent healthy copy plus integrity/application readback before its writer resumes, or a scoped recovery blocker. No other-user operation.
- Planned Touch Files: scripts/replay_dataset_champions.py; new scripts/research_storage.py; new tests/test_research_recovery.py; new docs/IMA_RECOVERY_STORAGE_CHAMPION_VALIDATION_20261020.md.
- Handoff: source/canary/reference and current selected-scope evidence; failure blocks affected cleanup, not local independent tests.
- Checklist:
  - [ ] Preserve current evidence locally, verify safe paths/real PIDs and pin protected source/model/MLflow roots.
  - [ ] Locate deployed imports and reproduce selected real package; resolve current compatible champion and protocol.
  - [ ] Capture affected DB/WAL consistently to healthy storage; verify integrity and consumer readback before writer recovery, preserving originals.
  - [ ] Record quiescence/restore plan and billing/startup failure hypotheses separately from stale status.

## Phase 2: Scoped Reclaim
### Subphase 2.1: Selected First-Wave Reclaim And Readback
- Objective: regain measured iMa disk headroom without removing scientific history.
- Inspect: refreshed per-payload/build signatures, locks/leases/open files, producer state, selected authority, persisted preparation descriptors and source prerequisites.
- Allowed edit surface: iMa-only exact selected manifest and lock-aware reclaim tooling; named own writer/timer quiescence only.
- Commit: feat(storage): apply identity-checked iMa derived reclaim with receipts.
- Tests: synthetic changed-path/UID/hardlink/lock/partial failure; .venv/bin/python -m pytest tests/test_research_storage.py tests/test_research_recovery.py -q; actual per-item/df/package/registry readback.
- Success Criteria: only currently eligible selected payloads/builds removed; stable lock/control and protected artifacts unchanged; exact reclaimed bytes measured. Failed/changed candidates held, not forced to hit 153.509 GiB.
- Planned Touch Files: new ima/research_storage.py; scripts/research_storage.py; ima/research_preparation.py; new tests/test_research_storage.py; validation report; owned ops execution manifest/receipts.
- Handoff: healthy free-space/inode receipt and canary pass; normal writers remain held until phase 3 safety qualifies.
- Checklist:
  - [ ] Quiesce only approved named own producers, preserving/draining real fits; revalidate signatures immediately before each step.
  - [ ] Evict payloads through stable-lock semantics; qualify each incomplete build and keep diagnostic evidence.
  - [ ] Measure actual effects, verify retained packages/data/MLflow and invalidate/rewarm preparation leases as needed.

## Phase 3: Storage Admission And Caches
### Subphase 3.1: Atomic Reservation And Failure Recovery
- Objective: make every artifact-producing worker reserve safe capacity before normal restart.
- Inspect: all writer entrypoints, output/scratch measurement, inode demand, own filesystem/control reserve and cgroup startup.
- Allowed edit surface: shared storage authority and writer/scheduler integration; no data-format replacement.
- Commit: feat(storage): reserve shared filesystem capacity across all iMa workers.
- Tests: two workers racing last capacity, growth underestimate, crash/reconcile, ENOSPC and main-down startup; .venv/bin/python -m pytest tests/test_research_storage.py tests/test_research_resources.py tests/test_research_scheduler.py tests/test_feature_exploration.py -q.
- Success Criteria: acquisition/build/preparation/OOF/fit/outbox paths covered, atomic accounting and explicit blocked statuses; partial outputs never published; no restart storm or unsafe second 100 GB budget.
- Planned Touch Files: ima/research_storage.py; ima/research_resources.py; ima/research_scheduler.py; ima/research_worker_runtime.py; ima/feature_exploration.py; ima/feature_lifecycle.py; ima/dataset_registry.py; ima/research_telemetry.py; scripts/collect_official_corpus.py; scripts/refresh_official_snapshot.py; scripts/audit_official_coverage.py; deploy/systemd/ima-feature-exploration-budget.py; own unit retry policy; corresponding existing/new tests.
- Handoff: reservation qualification ALONE does not authorize normal restart. Keep artifact-producing workers held until 3.2 also passes; permit only explicitly bounded qualification canaries under reservation. Dependent database recovery gates must pass too. Qualified original-format paths can then restart while additive migrations proceed independently.
- Checklist:
  - [ ] Integrate exact shared reservation/extension/reconcile and required headroom; measure real output estimates.
  - [ ] Repair safe own-user budgeting/startup order without removing containment checks.
  - [ ] Prove every writer calls admission; failures wait/back off instead of spawning endless failed work.

### Subphase 3.2: Shared Lock-Aware Cache Lifecycle
- Objective: prevent regenerated fold arrays accumulating without bounds.
- Inspect: preparation hit/miss/pins, scheduler leases, actual access, root ownership, mapped copies and rebuild consumers.
- Allowed edit surface: cache index/retention and preparation/scheduler integration.
- Commit: perf(storage): bound shared fold caches without evicting active state.
- Tests: global 50 GiB policy on small fixture scale, active pins over budget, lock inode retention, cache miss rebuild, crash and package parity; concurrent identical-key preparation builds exactly once; multi-program completion/refill immediately schedules eligible queued work into freed capacity subject to resources, independently of trial-budget size. Command: .venv/bin/python -m pytest tests/test_research_preparation.py tests/test_research_scheduler.py tests/test_research_executor_v6.py -q.
- Success Criteria: one cross-campaign budget, correct LRU/access measurements, pinned-overbudget pauses new production, saved models independent of eviction, one preparation producer per identical key and work-conserving eligible cross-program refill without budget/concurrency coupling.
- Planned Touch Files: ima/research_preparation.py; ima/research_storage.py; ima/research_scheduler.py; ima/research_executor.py; tests/test_research_preparation.py; tests/test_research_scheduler.py; tests/test_research_executor_v6.py.
- Handoff: measured cache/storage/memory performance with read-only sharing intact; other cache classes stay held until independently qualified.
- Checklist:
  - [ ] Reuse pin/evict and stable locks; share control/index, not per-service quotas.
  - [ ] Verify rebuilt arrays/selection/transforms and saved-package predictions match.
  - [ ] Benchmark several worker caps and report CPU %, private/PSS/cgroup memory, disk growth and useful completions/hour.

## Phase 4: Versioned Data Lifecycle
### Subphase 4.1: Immutable Object Reuse And CSV-On-Demand Snapshots
- Objective: preserve history without duplicating unchanged materialization each refresh.
- Inspect: input_digest/normalized change boundary, source capture evidence, snapshot builder/verifier, imported raw blobs and legacy consumers.
- Allowed edit surface: additive snapshot/object manifest writer/readers; no existing snapshot mutation.
- Commit: feat(data): version snapshot manifests with reusable objects and optional exports.
- Tests: repeated unchanged normalized inputs reuse objects; changed raw capture retained; legacy verify passes; missing/corrupt object rejects; atomic publication/row/value parity.
- Success Criteria: new version reuse and export policy proven through build/verify/registry consumers; old CSV-bound manifests remain valid and unmodified.
- Planned Touch Files: scripts/refresh_official_snapshot.py; scripts/build_official_dataset.py; scripts/verify_official_snapshot.py; scrapper/official_corpus.py; ima/dataset_registry.py; new ima/research_objects.py if no existing owner fits; tests/test_official_snapshot_refresh.py; tests/test_official_snapshot_verifier.py; tests/test_official_dataset.py.
- Handoff: parity/space benchmark and explicit new schema compatibility; no claimed historical CSV savings until actual migration/archive.
- Checklist:
  - [ ] Distinguish meaningful normalized/version change from capture-only change while preserving provenance.
  - [ ] Reuse immutable objects and make CSV an explicit bounded export for the new format.
  - [ ] Keep latest_verified and baseline/raw-import semantics; test old/new verifier readback and interruption.

### Subphase 4.2: Parent-Delta Features And Projected Preparation
- Objective: stop copying the parent table for every derived feature dataset.
- Inspect: dataset/feature registry, exact key/label/availability identity, feature routing, package replay and preparation ownership.
- Allowed edit surface: additive registry representation/readers and preparation projection.
- Commit: perf(data): materialize feature deltas against immutable parent datasets.
- Tests: logical/materialized parity, multiple parent layers, key reorder/drop/collision, cycles, missing ancestors, unit/availability replacement and prediction parity.
- Success Criteria: one parent plus delta storage, exact scored populations and existing-model inference; requested columns/folds only loaded without hidden estimator/preprocessing dependency omissions.
- Planned Touch Files: ima/dataset_registry.py; ima/dataset_specs.py; ima/feature_lifecycle.py; ima/research_preparation.py; ima/research_executor.py; ima/research_model_package.py; tests/test_dataset_registry.py; tests/test_dataset_registry_adversarial.py; tests/test_feature_lifecycle_integration.py; new tests/test_dataset_delta.py.
- Handoff: actual bytes, memory/cache/copy benchmark, resolver manifests and old/new prediction equivalence.
- Checklist:
  - [ ] Version representation, preserve original parents and prevent ambiguous keys/replacements.
  - [ ] Resolve projected columns and fold rows through shared preparation; package inference pins/embed state safely.
  - [ ] Reuse across model families without losing independent scientific run identities.

### Subphase 4.3: Reference-Aware Retention And Conditional Archives
- Objective: qualify other caches/history separately and make future lifecycle decisions auditable.
- Inspect: complete own service/registry/MLflow/hypothesis/confirmation/pending roots, leases, object ancestry, archive destination and restore consumers.
- Allowed edit surface: lifecycle mark/sweep/archive tooling; no blanket historical/other-user purge.
- Commit: feat(storage): qualify retained roots and verified archival lifecycle.
- Tests: pin/GC/publication races, grace/lease crash, unreadable/unknown refs, incomplete upload, checksum mismatch, same-filesystem archive and actual restored consumer replay.
- Success Criteria: only unreachable qualified objects or verified archived selected versions eligible; model/raw/unique history protected; archived references recover through explicit compatible restore. No destination means archive obligation held, not unsafe deletion.
- Planned Touch Files: ima/research_storage.py; ima/research_objects.py if introduced; ima/dataset_registry.py; ima/feature_lifecycle.py; scripts/research_storage.py; tests/test_research_storage.py; new tests/test_research_archive.py; validation report.
- Handoff: accepted retention policy, complete dry-run/restore receipts and measured distinct scenario savings. Archive boundary cannot block independent champion tests after core safety passes.
- Checklist:
  - [ ] Mark all protected roots and serialize sweep versus new pins/publication.
  - [ ] Review other caches and each historical snapshot's unique/replay/reference state.
  - [ ] Qualify selected archive destination/restore, retain compact history and apply only independently approved eligible archival changes.

## Phase 5: Champion Feature Research
### Subphase 5.1: Durable Merit-Aware Evidence And Hypothesis Queue
- Objective: route strongest compatible untested feature work into real experiments despite recent-catalogue churn.
- Inspect: screen queue, lifecycle index/outcomes, hypothesis states, graph reference and planner/context admission.
- Allowed edit surface: general promotion/context integration only, not duplicate trainers.
- Commit: feat(features): persist champion-directed promotion and evidence views.
- Tests: >8 recent weak candidates cannot displace older strongest, restart/champion change/dedup/negative evidence and compatible-only selection.
- Success Criteria: actual planner payload contains best untransferred plus complementary and pending evidence; chosen budget independent of worker ceiling, complete control/axis preflight.
- Planned Touch Files: new ima/feature_promotion.py; ima/feature_lifecycle.py; ima/feature_exploration.py; ima/research_lifecycle.py; ima/research_planner_context.py; ima/research_hypotheses.py; tests/test_research_lifecycle.py; tests/test_research_planner_context.py; new tests/test_feature_promotion.py.
- Handoff: exact promoted IDs and downstream evidence receipt, not just a generated candidate list.
- Checklist:
  - [ ] Preserve per-contract merit, support/uncertainty, negative outcomes and reference-specific transfer states.
  - [ ] Implement deterministic family-diverse views plus justified overrides/direct nonlinear tests.
  - [ ] Link every hypothesis to a control/result/next action and avoid paid calls while evidence is unchanged.

### Subphase 5.2: Fixed-Champion Bundle Transfers And Attribution
- Objective: establish incremental feature signal against the actual rich pool.
- Inspect: parent/child logical equality, scored keys/labels/protocol, effective graph/node routing and matched-control guards.
- Allowed edit surface: paired eligibility, promotion execution and reports; no global dataset_ref bypass.
- Commit: feat(features): run fixed-champion complete-bundle comparisons.
- Tests: population/label/schema mismatch rejection, full bundle then ablation/null control, frozen node params/weights and MLflow linkage; test_champion_feature_transfer plus existing lifecycle/discipline/graph integration.
- Success Criteria: actual top compatible bundle tests complete; beneficial/negative/inconclusive results retained with paired race/fold deltas and unchanged control.
- Planned Touch Files: ima/feature_promotion.py; ima/feature_exploration.py; ima/feature_studies.py; ima/experiment_discipline.py; ima/research_telemetry.py; scripts/feature_exploration.py; new tests/test_champion_feature_transfer.py; tests/test_feature_exploration_integration.py; tests/test_experiment_discipline.py.
- Handoff: completed real comparisons and evidence used in the next planner decision; screen gains never labelled champion gains.
- Checklist:
  - [ ] Run selected complete bundles against pinned champion before isolated removals.
  - [ ] Declare exact feature destination and verify unchanged parent values/fold/label populations.
  - [ ] Publish null/negative/inconclusive results and decide adaptation separately.

## Phase 6: Component And Pool Search
### Subphase 6.1: Attribution And Boosted-Primary Studies
- Objective: test component contributions and tune the dominant underexplored boosted component.
- Inspect: actual graph primary binding, model contracts, selected columns/convergence, original settings and slice diagnostics.
- Allowed edit surface: tune-node contracts, effective parameter/feature reporting and studies.
- Commit: feat(research): qualify component controls and boosted-primary optimization.
- Tests: pure Benter/boosted/35-65 controls; only intended node mutates; original/boundary configs; complete budgets survive max concurrency and adaptive study pause.
- Success Criteria: actual component controls/studies and error-slice hypotheses with full attribution, not repeat tuning of an unchanged dominant component.
- Planned Touch Files: ima/pipeline_graph.py; ima/research_specs.py; ima/research_search.py; ima/research_executor.py; ima/pipeline_transparency.py; ima/feature_analysis.py; ima/research_evidence.py; tests/test_pipeline_graph.py; tests/test_research_search.py; new tests/test_component_search.py.
- Handoff: comparable ablations, seeded valid search ranges, per-component cost/resource results and explicit continue/stop decisions.
- Checklist:
  - [ ] Log effective node params/features; tune boosted learning_rate/max_iter/leaves/depth/min_samples_leaf/l2 within contracts.
  - [ ] Keep other component fixed before separately identified joint adaptation.
  - [ ] Audit useful information per experiment and preserve slow-but-productive probit exploration.

### Subphase 6.2: Forward Prediction Registry And Learned Pools
- Objective: learn useful weights cheaply without fit/selection leakage.
- Inspect: graph pools, chronological inner splits, dependency cache, selection cutoff/lineage and package adapter.
- Allowed edit surface: forward prediction lineage and constrained pool integration.
- Commit: feat(research): learn lineage-checked forward fundamental pools.
- Tests: known synthetic mixture, pure/equal/fixed controls, earlier fit and selection cutoffs, reused predictions, corrupt/missing rows and serialized inference.
- Success Criteria: actual weights learned on permitted forward data; outer evaluation labels unavailable to fitting; development_selected explicitly exposed versus nested_selection valid; later comparable scores retained.
- Planned Touch Files: new ima/research_pooling.py; ima/pipeline_graph.py; ima/research_executor.py; ima/research_preparation.py; ima/research_model_package.py; new tests/test_forward_pooling.py; tests/test_research_model_package.py.
- Handoff: immutable prediction lineage, fitted pool package and later-fold diagnostics; no independent-confirmation claim yet.
- Checklist:
  - [ ] Generate expanding whole-meeting prediction blocks and verify fit/selection lineage.
  - [ ] Fit simplex weights with valid zero endpoints and frozen outer protocol.
  - [ ] Persist package/weights/adapter and reject same-score fitting or fake OOF provenance.

## Phase 7: Historical Fitness
### Subphase 7.1: Official Availability And Coverage Recovery
- Objective: establish which raw histories can legitimately feed past/prospective predictors.
- Inspect: cached HKJC corpus, event/horse identity, corrections, publication evidence, individual timing and coverage gaps.
- Allowed edit surface: existing official collectors/parsers/audits, storage-admitted and rate/access compliant.
- Commit: feat(data): qualify historical fitness availability and coverage.
- Tests: existing historical_events/sources, official_coverage/dataset, speed_features and independent raw-record readback.
- Success Criteria: per-family/season exclusion waterfall and supported historical data; unavailable publication stays prospective-only; no alias/name-only joins or fabricated absence.
- Planned Touch Files: scrapper/official_corpus.py; scrapper/historical/results.py; scripts/collect_official_corpus.py; scripts/audit_official_coverage.py; scripts/build_official_dataset.py; ima/historical_events.py; tests/test_historical_events.py; tests/test_official_coverage.py; tests/test_official_dataset.py.
- Handoff: immutable availability/identity/coverage manifest and source-linked sample review; missing backfill does not block other research.
- Checklist:
  - [ ] Reparse verified cache first, then target documented official gaps and retain raw captures.
  - [ ] Separate parser error, true absence, inaccessible page and publication/identity uncertainty.
  - [ ] Capture prospective evidence where historical proof is impossible.

### Subphase 7.2: Clean Fitness Families And Matched Successor Dataset
- Objective: make supported fitness/speed/reliability measurements usable across pipelines.
- Inspect: unit/availability metadata, feature vocabulary, residual normalization/shrinkage, delta resolver and model/target adapters.
- Allowed edit surface: feature builders, typed definitions and immutable successor integration.
- Commit: feat(features): publish clean fitness families with matched champion controls.
- Tests: speed_features/feature_expressions/feature_program/registry adversarial; raw-to-feature checks; family/missingness ablations; native target smoke and unchanged champion on successor population.
- Success Criteria: actual varying supported predictors, no hidden cohort shift, train-only transforms, honest archive limits and per-target comparable results.
- Planned Touch Files: ima/speed_features.py; ima/season_features.py; ima/feature_definitions.py; ima/feature_program.py; ima/dataset_registry.py; ima/dataset_specs.py; scripts/build_research_dataset.py; tests/test_speed_features.py; tests/test_feature_expressions.py; new tests/test_fitness_dataset.py.
- Handoff: versioned definitions/successor, coverage report and matched reference; no raw parent mutation.
- Checklist:
  - [ ] Build ability/workload/trial/health/reliability families independently with support indicators and missingness controls.
  - [ ] Fit adjustments/residuals/shrinkage only on appropriate earlier history.
  - [ ] Test transfers to Benter/boosted/justified heteroscedastic models and retain all native target contracts.

## Phase 8: Independent Confirmation
### Subphase 8.1: Exposure Ledger, Study Seals And Later Evaluation
- Objective: distinguish selected development decimals from genuinely independent gains.
- Inspect: every known campaign/manual-research exposure, package/label history, legacy locks and meeting-block estimand.
- Allowed edit surface: exposure/confirmation contracts and planned CLI sealing; no early label release.
- Commit: feat(research): confirm frozen challengers with prospective meeting seals.
- Tests: selection exposure, same-day timezone deadlines, future runners, supersession, scratches/cancellations/missing forecasts, sample/power lock, block bootstrap and quarantined planner payloads.
- Success Criteria: locked protocol/packages and valid per-meeting forecasts; independent paired report when horizon complete; pending future meetings or inconclusive evidence reported honestly.
- Planned Touch Files: ima/research_confirmation.py; ima/research_hypotheses.py; ima/research_planning.py; scripts/research_confirmation.py; tests/test_research_confirmation.py; new tests/test_confirmation_exposure.py.
- Handoff: immutable lock/seals/exposure history, measured uncertainty and independent interpretation. Secondary-target confirmation cannot reuse win labels under a different name.
- Checklist:
  - [ ] Design exposure/holdout rules BEFORE consuming later data; freeze defaults/sample after development-only power pilot.
  - [ ] Seal actual forecasts before deadlines and keep the reference/packages frozen.
  - [ ] Evaluate only at fixed horizon and publish confirmed/not-confirmed/inconclusive without implying profit.

## Phase 9: Qualification And Rollout
### Subphase 9.1: Unified Telemetry, Native Costs And Real-User Verification
- Objective: make storage/science decisions understandable and prevent false progress.
- Inspect: terminal summaries, MLflow run/trace hierarchy and installed native usage APIs, outbox, request IDs, full graph/contract and lifecycle statuses.
- Allowed edit surface: context/telemetry/readback tooling, no new product UI.
- Commit: fix(research): unify storage and scientific evidence with native trace usage.
- Tests: actual consumer fixture + MLflow readback, best-by-contract/no false None, chosen/allocated/continuing budgets, known/pending cost and token columns, duplicate retry prevention and blocked-storage versus scientific failures.
- Success Criteria: human/operator and planner see matching best/graph/feature results, model/operation counts and native USD/tokens; negative results and pending controls preserved.
- Planned Touch Files: ima/research_telemetry.py; ima/research_summary.py; ima/research_planner_context.py; ima/openrouter_orchestrator.py; scripts/research_storage.py; scripts/feature_exploration.py; tests/test_research_telemetry.py; tests/test_research_planner_context.py; tests/test_openrouter_orchestrator.py; new tests/test_ima_recovery_workflow.py.
- Handoff: deterministic user workflow and real MLflow/native-column receipts, compact context/decision relevance review.
- Checklist:
  - [ ] Link storage operations to protected references without counting them as model fits.
  - [ ] Verify true graph champion, transfers, native token/USD and event-driven paid decisions.
  - [ ] Complete fixture/adversarial workflow and independent consumer review.

### Subphase 9.2: Sealed Own-User Handover And Unattended Qualification
- Objective: deploy only qualified paths, keep rollback and verify actual research/storage behavior.
- Inspect: reviewed branch/CI, sealed file/dependency manifest, current own processes/resources/leases, preservation receipts and rollback availability.
- Allowed edit surface: own release/config/unit integration and validation report only.
- Commit: chore(release): qualify iMa recovery and champion research handover.
- Tests: relevant complete regression suite, sealed real-data canaries, safe drain, restart and multiple genuine live decision/refill/completion cycles; storage/cache pressure failure injection in isolated fixtures.
- Success Criteria: actual reclaim and preserved consumers, core storage guards active, real paired feature/component/pool outcomes and downstream decisions under <=100 GB RAM/max26 fits, healthy headroom and no collateral mutations. Statistical confirmation remains pending until enough future labels exist.
- Planned Touch Files: owned deploy/systemd release manifests/configuration; docs/IMA_RECOVERY_STORAGE_CHAMPION_VALIDATION_20261020.md; task acceptance/proof artifacts; existing rollout tests plus new workflow qualification.
- Handoff: pushed/reviewed coherent release when authorized, sealed operational/scientific identity, rollback, native service/MLflow/readback, independent review and exact unresolved archival/statistical blockers.
- Checklist:
  - [ ] Qualify code/environment and real-data consumers; preserve productive fits and safe original-format restart.
  - [ ] Handover only at verified boundary with new scientific identity for changed code; do not invent version labels automatically.
  - [ ] Prove several real cycles and storage-pressure behavior, close owned agents and certify exact completed obligations.

## Dependency Order And Stop Rules
Critical path: 1.1 -> 2.1 emergency qualified reclaim -> 3.1 mandatory disk admission -> 3.2 cache lifecycle. Only then permit unrestricted original-format workers. Reclaim without guardrails is a checkpoint, not recovery completion.
- Local isolated tests/design for later phases may proceed while SSH or live storage is blocked.
- 4.1/4.2 format changes can be developed in parallel with 5/6 using disjoint owners. Consumers remain on qualified legacy representation until migration parity passes; new shared-object GC is disabled until 4.3 reference/restore safety passes.
- 5.1 -> 5.2 champion transfer, then 6.1 adaptation and 6.2 forward pools. Qualified original data can support these before archive/data-format optimization completes.
- Phase 7 adds real signal; no archival availability proof means continue prospective capture and other valid research.
- Phase 8 exposure design is done BEFORE reserving/revealing later data; evaluation may wait for future meetings without stopping other useful research.
- Phase 9 qualifies integrated paths; archive destination absence blocks only archive removal, not safe derived reclaim or scientific development.
- No guaranteed 153.509 GiB reclaim or lower log loss. Hold changed/ineligible objects; do not erase protected data to hit a target. Stop saturated/invalid work by explicit recorded decision.
- Fresh execution restriction, inaccessible credentials, missing required archival proof, destructive boundary or future-label wait scopes the affected obligation. Do not claim whole execution complete from a completed plan/audit, a passing fixture or a single canary.

