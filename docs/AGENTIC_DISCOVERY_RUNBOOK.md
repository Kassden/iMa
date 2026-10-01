# Agentic Discovery V5 Runbook

## Ownership and Boundaries

V5 runs as `imaopt` on Cortex-server (`100.95.24.121`). Its controller,
process pool, planner thread and tracking thread are one supervised application,
not two unconnected optimization loops. Never run these commands as another app's
user, modify shared Python packages, or restart Cortex/solar services.

- Campaign: `/home/imaopt/research-v2/campaigns/agentic_v5_discovery`
- Pinned source: `/home/imaopt/research-v2/live-releases/ima-v5-current`
- Isolated dependencies: `/home/imaopt/research-v2/v5-dependencies/site-packages`
- User service: `ima-discovery-v5-supervisor.service`
- MLflow experiment: `ima-agentic-v5-discovery`
- Tracking: `http://100.95.24.121:5000` over Tailscale

## What the Agent Chooses

The OpenRouter planner uses `deepseek/deepseek-v4.1-flash` with unpinned provider
routing. It receives current comparable champions, compatible v4 reference
champions, recent hypotheses/outcomes, feature manifests/coverage/selection
reports, active budgets, failed requests and remaining capacity.

It proposes typed feature recipes, selection strategies, registered sources,
transforms, compatible model/target contracts, diagnostic requests, hyperparameter
search spaces, exact-control trials and program retirement. It chooses a decision
budget from 1 through 260; 260 is a ceiling, not a required batch. Program budgets
must sum to no more than that choice. It cannot execute arbitrary generated Python,
invent unavailable external sources, access confirmation holdout labels or promote
models into betting. A new source needs a registered availability adapter first.

## What Code Enforces

- Featuretools DFS enumerates historical aggregates; sequence/domain primitives
  supplement it. Stable recipes and source/builder hashes identify raw matrices.
- Historical race results become available no earlier than the following midnight;
  an explicit later observation delays availability further. Strict cutoffs exclude
  current/same-day results. Feature-engine and sklearn selection fit on outer
  training rows; learned residuals additionally use expanding chronological fits.
- Five lanes: 80% conditional-logit fundamentals; 20% rotating boosted win,
  LambdaRank ranking, CatBoost top-k placing and recorded final-odds regression.
  Portfolio allocation is cumulative, not independently rounded per decision.
- One controller owns ledger writes and Optuna ask/tell. Trials are asked just in
  time; completed slots are replenished without waiting for a cycle's straggler.
  Programs rotate within a lane so fresh probes can share existing large budgets.
- Maximum 12 concurrent trials, each one native CPU thread and a 4 GiB admission
  reservation; aggregate CPU budget 24 and RAM budget 80 GiB. Host reserve: four
  threads/eight GiB. A reservation is not an individual hard RSS limit. The service
  has `MemoryHigh=92G`, `MemoryMax=96G`; dispatch also checks available host memory.
  The 80 GiB admission budget accounts for declared reservations, not measured RSS.
  Initial 16-worker full-history training reached 96 GiB and suffered an OOM
  interruption. All 16 interrupted attempts were automatically requeued; the
  guarded resume uses the same model revision and dataset with 12 workers.
  A subsequent sample measured about 81 GiB and zero memory-pressure stalls at
  12 workers. The cap leaves nominal host headroom, not a guaranteed reservation
  against other apps' growth. The ceiling does not change agent trial budgets.
- Builders execute inside these same worker reservations. Identical raw matrix
  requests share a locked cache; selection/residual fit state is not shared across
  folds or targets. Fresh Featuretools builds may take many minutes.
- No total-trial limit. Planner spend pauses at $5; eight consecutive trial failures
  stop admission. HTTP deadlines are bounded. No silent bootstrap fallback.
- Checkpoints occur after 32 terminal trials, after 600 seconds, at low queue capacity,
  or when the required lane cannot be dispatched. Planning overlaps approved work.

## Where Evidence Lives

`ledger.sqlite` records attempts and upload linkage; `search/` stores Optuna state;
`hypotheses.sqlite` stores append-only research outcomes. `evidence/` freezes planner
inputs; `planner/`, `planner-calls/`, `decisions/` and `decisions.jsonl` preserve
decisions, billed responses and rejected proposals. `queue.jsonl` records resource
reservations. `trials/<attempt>/stage.json` identifies loading, building or training;
that directory also contains predictions, fold reports, diagnostics and packages.
`discovery-cache/` holds checksummed raw feature matrices and serialized catalogs.

MLflow has one training run per attempt, source and generated-matrix dataset inputs,
selection/feature artifacts and registered candidate model versions. Planner traces
show reported USD cost; result checkpoint traces carry no duplicate LLM charge.
Comparisons are partitioned by target, objective, parameters and protocol; there is
no universal score across log loss, NDCG, Brier and odds MAE. Development champions
remain research candidates, not proof of future profitability.

## Operate

```bash
ssh imaopt@100.95.24.121
systemctl --user status ima-discovery-v5-supervisor --no-pager
tail -n 30 /home/imaopt/research-v2/campaigns/agentic_v5_discovery/ops/supervisor.log
```

Read `status.json`, the ledger and timestamps together. An active service alone is
not proof of progress. Check growing completed counts, fresh evidence-aware
decisions, pending tracking/tells, CPU/RSS and MLflow readback.

Graceful stop: create the campaign stop request through the pinned CLI. It stops
new admissions and drains active jobs and uploads before a clean exit.

```bash
CODE=/home/imaopt/research-v2/live-releases/ima-v5-current
PYTHON=/home/imaopt/research-v2/releases/f54c604/.venv/bin/python
env PYTHONPATH=/home/imaopt/research-v2/v5-dependencies/site-packages:$CODE "$PYTHON" -m scripts.optimize stop --campaign /home/imaopt/research-v2/campaigns/agentic_v5_discovery
```

For a crash, systemd restarts this application; the controller recovers interrupted
reservations and reconciles Optuna/MLflow writes. Operator, failure and spend-cap
exits deliberately stay stopped. Do not delete STOP or restart blindly: inspect the
cause, explicitly authorize resumption and preserve campaign identity. Code or
dependency changes require a successor campaign, not a hotpatch of this one.

Rollback preserves v4's source, inputs, registry and STOP evidence. Drain v5 first;
only then, with explicit authorization, clear v4's stop request and enable its old
user service. Never run both campaigns unrestricted.

## Deterministic Discovery Commands

Use the existing explicit immutable-input CLI style rather than a hidden mutable
`--discovery` directory contract. `screen` requires a JSON list of training race IDs.

```bash
.venv/bin/python -m scripts.discover_features enumerate --dataset HISTORY.csv --spec config/feature_discovery_defaults.json --output .tmp/discovery
.venv/bin/python -m scripts.discover_features materialize --dataset HISTORY.csv --spec config/feature_discovery_defaults.json --output .tmp/discovery
.venv/bin/python -m scripts.discover_features screen --dataset HISTORY.csv --spec config/feature_discovery_defaults.json --output .tmp/discovery --train-races TRAIN_IDS.json --target win_probability --label target_win
.venv/bin/python -m scripts.discover_features replay --dataset HISTORY.csv --spec config/feature_discovery_defaults.json --output .tmp/discovery
```

Optional tsfresh is disabled. Catalog shards contain 64 definition IDs; this is a
bounded catalog, not unrestricted arbitrary DAG synthesis or a distributed builder.
Learning-curve diagnostics freeze selected features/transforms and measure model
training sensitivity, not a fresh end-to-end feature-selection curve at each size.
# Full-History Memory Recovery

The 16- and 12-worker waves exceeded the dedicated 96 GiB cap. The authorized
next measurement uses ten workers, `MemoryHigh=96G`, `MemoryMax=100G` (GiB).
The host reports 121.4 GiB usable, leaving about 21.4 GiB outside this cap.
This is an operating limit, not a claim that ten workers have passed all folds.

The unit's `ExecStartPre` runs `v5-memory-backoff.py` from the operational
dependency directory, outside the pinned model release. A new own-unit
`UNIT_RESULT=oom-kill` journal event reduces `max_concurrent_trials` by two;
the persisted transaction makes retry idempotent. One-worker OOM creates a
STOP marker and refuses restart. It never changes model/data/trial budgets,
raises concurrency, or reads other users' units. Inspect
`ops/memory-backoff-state.json` and the own user journal for recovery evidence.
Changing ceilings upward requires operator review of full-fold memory peaks.
The original 4 GiB/job reservation is insufficient for these discovery recipes;
do not mistake admission estimates for observed RSS or total cgroup memory.
