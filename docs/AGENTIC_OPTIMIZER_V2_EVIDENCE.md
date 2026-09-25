# Agentic Optimizer V2: Research Evidence And Delivery Record

## GOAL
Review the improvement report and conversation, inspect current code/logs/MLflow, research primary-source prior art, and deliver a decision-complete implementation plan. This research task does not implement that plan.

## Acceptance Criteria
- [x] Report and conversation reviewed; substantive corrections identified.
- [x] Live JSONL, decisions, bounded log and MLflow sample inspected read-only.
- [x] Primary sources compared and linked in AGENTIC_OPTIMIZER_V2_PLAN.md.
- [x] Plan defines contracts, boundaries, subphases, tests and handoff evidence.
- [x] Plan validation and generated-dashboard readback complete.

## Research
See [implementation plan](AGENTIC_OPTIMIZER_V2_PLAN.md) for source links and decisions. Reviewed Karpathy autoresearch, Microsoft RD-Agent, AIDE, Optuna, AutoGluon, scikit-learn, MLflow and OpenRouter official documentation. Research is not a framework benchmark.

## Live Snapshot
Read-only remote inspection at 2026-09-24T09:39:25Z (17:39 China time), through SSH root to 100.95.24.121, executing reads as imaopt in /home/imaopt/iMa. No key contents accessed.
Source: artifacts/agentic-learning/cortex-full-pipeline/trials.jsonl.
- Rows/statuses: 2,048 completed.
- Families: 1,966 logit; 82 boosted.
- Schemas: baseline-v1 for all 2,048.
- Zero fundamental blend weight: 1,006 (49.12%).
- Best: trial-1938, logit-long-c80-balanced-lbfgs-intercept-tol0p0003.
- Best blended race log loss: 2.0118061545903827.
- Same trial fundamental loss: 2.297609926227863.
- Fundamental weight: 0.03012282088199783; market weight: 1.0599410078042995.
- Latest sampled trial: trial-2048, boost-long-lr0p015-iter260-leaf31-l20p3.
- Latest blended loss: 2.011932845005823; fundamental loss: 2.237653105181465; fundamental weight 0.

At the next bounded log read, model-version creation messages reached version 192 at 17:39:31. This is a log observation, not a complete registry inventory. tmux showed ima-full-pipeline created 17:05:45. A session and recent output demonstrate activity, not end-to-end campaign correctness. Counts can advance after this snapshot.

## MLflow API Readback
MlflowClient at http://100.95.24.121:5000 returned:
| Sample | MLflow run ID | Status | Blended loss | Schema |
|---|---|---|---:|---|
| First | e8abcb4086cb44df9db4503816b250f0 | FINISHED | 2.0118672352532454 | baseline-v1 |
| Best | f7471a68d7e04072b4eba2ce05d6ff94 | FINISHED | 2.0118061545903827 | baseline-v1 |
| Latest sampled | aaf8334de99743228708ce7114c261e2 | FINISHED | 2.011932845005823 | baseline-v1 |

All three list context.json, models and run.json artifacts. Metrics agree with sampled JSONL. The best run's parameters include C=80, balanced, lbfgs, intercept=True, tol=0.0003, max_iter=1200. This is three-run reconciliation, not exhaustive auditing or model-load verification.

The last two decisions at 09:14:16Z and 09:28:43Z said continue and voted across test metrics. The latter preferred boost-long-lr0p015-iter180-leaf7-l20p3 with loss about 2.01193282, although the global best logit remains 2.01180615. This supports separating batch winner from global incumbent.

## Code Evidence
Baseline HEAD: f6b560f138b780b9506f9924c65beed31fcb845b; branch audit/run-centric-dashboard-redesign, ahead 21 before documentation; clean initial worktree.
- ima/optimizer.py:60: default primary_metric is test_blended.race_log_loss.
- ima/optimizer.py:229: adaptive sorting uses test blended/fundamental/top-pick metrics.
- ima/optimizer.py:260: adaptive drains long catalogue, then neighborhoods around top 24.
- ima/optimizer.py:376: remote response maps only known available run IDs.
- ima/optimizer.py:401: choose_proposals receives specs/count/config, no completed results.
- ima/optimizer.py:441: voting uses test metrics.
- ima/optimizer.py:500: worker rebuilds baseline full-history dataset per trial.
- ima/optimizer.py:564: results append after batch completion.
- ima/optimizer.py:594: concurrency is CPU-count based.
- ima/openrouter_orchestrator.py: planner_messages contains only catalogue/context instruction, no outcomes; Batch URL uses /api/beta/batches.
- ima/experiments.py: ExperimentSpec contains run_id/kind/parameters; _run_one fits calibrator/blend on validation and reports test metrics.
- ima/data.py:186: chronological split partitions sorted races 70/15/15, not independent calibration and search-score windows.
- ima/mlflow_tracking.py:144: logs model as artifact and registers its file source; no standard flavor packaging in this path.
- ima/feature_sets.py:102: three existing feature schemas; scripts/run_schema_feature_study.py routes rich loading outside optimizer.

Line references describe this inspected commit and can move during implementation.

## Report Review
The earlier report's 1,984-trial count is superseded by this snapshot; its best blended score still matches. Its standalone best is time-sensitive: newer decision data includes fundamental loss approximately 2.23158, better than the earlier 2.23380. No full updated standalone leaderboard was captured, so do not claim this is the current global minimum.

Keep its schema-awareness recommendation. Raise evaluation contamination to first priority. Do not infer blend defects or profitable edge from zero weights. Reject an invented fixed significance threshold. Distinguish a new training objective from algebraic restatements of the current blend. Preserve old artifacts as exploratory history.

## Root-Cause Baseline
- Evidence inventory: current source, remote JSONL logs and three MLflow metric/artifact readbacks recorded above.
- Proven: catalogue-only planner, test-driven selection, narrow schema, batch-delayed persistence.
- Hypotheses: odds dominance/redundant search limit yield; richer information might help.
- Missing evidence: untouched holdout, paired prediction uncertainty and point-in-time availability audit.
- Mutation boundary: read-only server/API; local documentation only.
- Remediation mapping: implementation plan maps each finding to a phase; runtime hardening is explicitly risk reduction.

## SOTA, Standards, And Best Practices
Implementation recommendation is a lightweight agent outer loop and mature Optuna inner loop using current code/MLflow. Karpathy contributes fixed-evaluator discipline, RD-Agent feedback over data/model changes, AIDE candidate lineage. No claim of a universally best framework. Official provider docs, rather than old assumptions, govern Flex/Batch behavior.

## Dependency and Tooling Preflight
No install is needed: Python, Git, SSH and Megaskill scripts are present. Future execution has its own dependency preflight. Real blockers encountered: none for planning. No paid LLM calls or native library installs were necessary.

## Deterministic Real-User Test
Entry point: AGENTIC_OPTIMIZER_V2_PLAN.md. User workflow: hand document to another model. Stable inputs: this snapshot and inspected source revision. User-observable assertions: exact contracts/paths/tests, preserved boundaries, unchecked future phases.
Command: python3 /Users/milkingthesun/.codex/skills/megaskill/scripts/mega_plan_check.py docs/AGENTIC_OPTIMIZER_V2_PLAN.md.

## Fulfillment and Readback Proof
Required write: implementation plan and evidence record; generated dashboard. Readback surface: full Markdown plus dashboard HTML. Expected content: six phases, eleven subphases, explicit pending implementation, findings and source links. Check the actual generated content rather than file existence. This delivery record is separate from the future execution plan, so research completion cannot masquerade as implemented functionality.

## Armageddon Mode
Attack scope: planning errors. Checked stale counts, unsupported live-agent claims, evaluation leakage, wrong Batch route assumption, algebraically duplicate proposed models, unsafe shared-host edits and false registry-load claims. Must-fix: any such ambiguity remaining in final document. No production failure injection performed for a documentation task.

## Generality Guardrail
Existing mechanism: optimizer/MLflow/feature registries. Recurrence is high across future campaigns; the plan extends these owners. No new runtime or application created during research.

## Regression Guardrails
- Delivery gate ledger: .mega/research-v2-delivery-state.jsonl. The pre-existing global .mega/state.jsonl fails event-order validation at line 10; it was preserved. This delivery uses its own ordered ledger and does not claim old ledger repair.
- Planned edit surface: docs/AGENTIC_OPTIMIZER_V2_PLAN.md, docs/AGENTIC_OPTIMIZER_V2_EVIDENCE.md and generated .mega state/evidence/dashboard.
- Protected behaviors: all executable code, running optimizer, shared server services.
- Damage radius: small.
- Branch strategy: current audit branch for documentation only.
- Proof plan: plan lint, readback, scope/diff checks and delivery final gate.

## Phase 1: Research Delivery

### Subphase 1.1: Audit, research and execution plan
- Commit: docs(research): plan evidence-driven agentic optimizer (commit not required by this planning request).
- Tests: both documents pass mega_plan_check.py; generated dashboard content readback; git diff --check.
- Success Criteria: research questions answered with live evidence and runnable future verification contracts.
- Planned Touch Files: docs/AGENTIC_OPTIMIZER_V2_PLAN.md, docs/AGENTIC_OPTIMIZER_V2_EVIDENCE.md.
- Checklist:
  - [x] Inspect report/conversation/code and remote experiment sample.
  - [x] Research primary sources and compare adoption choices.
  - [x] Write decision-complete future implementation plan.
  - [x] Validate/read back artifacts and record final delivery gate.
