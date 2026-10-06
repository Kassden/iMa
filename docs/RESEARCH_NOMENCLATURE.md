# Research Nomenclature

This document specifies Contract F for the successor. Existing V5 experiment IDs,
run names, metrics and semantics remain historical evidence. Documentation does
not imply every new field or UI column is already implemented.

## Entities And Operations

| Entity | Identity and accounting | Observable surface |
|---|---|---|
| Campaign | Immutable code/data/protocol/environment/policy identity | Manifest, ledger, `ima.campaign_id` |
| Decision | Planner direction/allocation checkpoint, including rejected/repair outcomes | Finite `research.plan` trace |
| Program | Durable hypothesis, graph and allocated search slots | Program metadata and origin decision |
| Trial | One concrete graph/config evaluated under one comparison contract | Outer MLflow run |
| Attempt | Physical execution/retry of a trial | Ledger attempt, retry status/run linkage |
| Component fit | Constituent node/fold fit within an outer trial | Child run/span and parent trial |
| Execution snapshot | State at one committed evidence watermark; no new paid decision | Finite `research.snapshot` trace |
| Dataset build | Candidate construction/validation, never implicit promotion | `dataset.build` trace, manifest/lineage |

Operation names: `research.plan`, `research.validate`, `dataset.build`,
`feature.generate`, `feature.select`, `pipeline.fit`, `pipeline.evaluate`,
`research.snapshot`, `betting.evaluate`. UUID/content IDs are identity keys;
sequences and human labels are display aids. Timestamps are ISO-8601 UTC; label
any Asia/Shanghai rendering as UTC+08:00.

Names: experiment `ima/research/<campaign_id>`; decision
`<campaign_short> | plan D000101`; snapshot
`<campaign_short> | execution S000042`; build
`<campaign_short> | dataset B000007`; trial
`<target_short> | <pipeline_family> | P000012-T000064`; component
`<trial_short> | node <node_id> | fold F02`. Scores and long hypotheses belong in
metadata, not names. An Optuna ask/tell is not a paid LLM decision. A cycle counter
is a legacy loop/checkpoint sequence and cannot count asynchronous completions.

## Tags And Snapshot Contract

Schema/version tags accompany `ima.campaign_id`, `ima.event_kind`,
`ima.decision_id`, `ima.program_id`, `ima.trial_id`, `ima.attempt_id`,
`ima.run_role`, `ima.graph_id`, `ima.dataset_id`, `ima.feature_set_id`,
`ima.evaluation_contract_id`, `ima.probability_basis`, `ima.evidence_watermark`,
`ima.origin_decision_id`, `ima.status`. Finite asynchronous continuation traces
link to their origin; no root span is held open for days.

At one immutable watermark, report chosen/allocated/unallocated trial slots,
new/continuing programs, preparing/queued/running/completed/failed counts,
`planner_cost_usd`, `cost_status`, `best_by_comparison_contract`, `new_record`
and trial/run/model links. Reconcile allocated slots after retirement/restart.
Count outer trials, physical attempts, component fits and paid calls separately.
Derive preview text from that same snapshot. Example:
`allocated40/48; 3 new, 9 continuing; fundamental LL=2.15609; new_record=no`.

## Metrics And Comparability

| Canonical metric | Direction | Unit / probability basis |
|---|---|---|
| `fundamental_win.race_log_loss` | Lower | Categorical race winner, natural log |
| `blended_win.race_log_loss` | Lower | Explicit fundamental/market blend, natural log |
| `place.top_3.brier` | Lower | Runner top-three event probability |
| `ranking.ndcg_at_3` | Higher | Race-group ranking, dimensionless |
| `speed.mae_mps`, `speed.rmse_mps` | Lower | Metres/second |
| `finish_time.mae_seconds` | Lower | Seconds |
| `performance_distribution.nll` | Lower | Density NLL; depends on target units |
| `performance_distribution.crps` | Lower | Same physical units as target |
| `performance_distribution.coverage_80`, `coverage_95` | Near nominal | Empirical interval coverage |
| `joint_order.log_loss` | Lower | Named joint finish-order event and adapter assumptions |
| `final_odds.log_mae` | Lower | Recorded final odds on declared log scale, research target only |
| `betting.<pool>.realized_roi` | Higher | Profit/stake, decimal ratio; explicit settlement/quote mode |

Legacy aliases such as `test_fundamental.race_log_loss` and
`test_blended.race_log_loss` remain legacy keys until an explicit tested adapter
maps them. Do not rewrite old metric values or imply raw aliases are canonical
keys. Coverage is a calibration diagnostic, not a monotonically better score.

Comparison identity includes dataset/eligible race population, temporal folds,
target and parameters, objective/version, probability basis and evaluation
contract. Raw NDCG, odds MAE, density NLL and race log loss cannot compete on one
scalar leaderboard. Density NLL changes under physical unit rescaling. Mixed
objectives render a per-contract map or `multiple objectives; see per-target bests`.
Original champion replay and scoring retrained controls on a changed V6 population
are distinct experiments. A new record requires a compatible prior champion and
the metric's declared direction; never infer it from a stale preview.

## USD And Capability Evidence

Charge once per physical paid model call, including failed and repair calls.
Use stable call IDs for reconciliation; decision totals aggregate those calls;
execution snapshots and uploads link them without charging again. Unknown cost
is null/unavailable; confirmed free cost can be zero. Record provider provenance,
currency USD, token counts and cost status. Allocation/fit counts do not imply cost.

Native USD columns require the installed MLflow-supported LLM span token/cost
attributes, API readback and UI verification. Domain `planner_cost_usd` metadata
alone does not prove native column population. Live baseline is MLflow 3.16.1;
health/version/experiment APIs were verified, native USD/custom columns were not.
If installed UI cannot expose a custom column, retain supported tags/metadata,
previews and artifacts and record that limitation. Do not promise unsupported
columns or implement a new dashboard to conceal it. Any historical name/tag
backfill is idempotent, provenance-bearing, first dry-run and separately assigned;
no historical score, experiment ID, policy meaning or USD total is rewritten.
