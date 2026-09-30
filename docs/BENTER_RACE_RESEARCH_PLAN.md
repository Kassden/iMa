# Race-Aware Benter Research Campaign

Implementation authorized on 2026-09-27. The new campaign is `agentic_v3`; the live v17 campaign and its ledger remain read-only. A separate commit, rollout, and readback are required before declaring completion.

## GOAL
- Train a race-conditional Benter fundamental model, compare it fairly with market-only calibration, and enforce an 80/20 Benter/experimental research allocation in a new campaign.

## Acceptance Criteria
- [ ] Fundamental training minimizes race-mean winner NLL directly; predictions sum to one per race and never use same-race odds as features.
- [ ] Development results separately report fundamental, raw market, calibrated-market-only, and Benter combined loss, plus combination weights and paired incremental value.
- [ ] Several paper-backed experimental programs execute on their own target-specific scoreboards, not merely appear in a prompt. No raw metric from one target is ranked against a different target.
- [ ] For every complete five scheduled trials in an opt-in campaign, four belong to Benter and one to experimental. At any prefix of `n` scheduled trials, Benter count is `round(0.8*n)` (ordinary nearest integer; this series has no half ties). Thus 26 slots give 21/5; exact 4:1 holds at multiples of five. Planner failures and resume cannot alter it.
- [ ] New local and server canaries prove training, allocation, artifact reload, and MLflow readback without altering the live v17 campaign or any shared service.

## Research
- Benter 1994: local `research/1994-benter.pdf`, race-level multinomial logit fundamental model, followed by a second logit combining out-of-sample fundamental and public log probabilities.
- Bolton and Chapman 1986: [original horse-racing multinomial logit](https://pubsonline.informs.org/doi/10.1287/mnsc.32.8.1040).
- Ke et al. 2017: [LightGBM](https://proceedings.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), the efficient boosting engine. Burges 2010: [LambdaMART](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf), the learning-to-rank method. [LightGBM's group-ranking API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html) is the first experimental candidate. These papers show promising results in other domains, **not** evidence of a gain on this racing dataset; ranking scores require separate calibration before they are win probabilities.
- Prokhorenkova et al. 2018: [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), ordered boosting and categorical handling; research backlog, not claimed implemented.
- Chen and Guestrin 2016: [XGBoost](https://doi.org/10.1145/2939672.2939785), scalable boosting; research backlog, not identical to existing sklearn boosted trees.
- So, Woo, and Lee 2025: [horse-racing learning-to-rank comparison](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003266151), 9,140 Korean race records. CatBoost reported NDCG 0.8895 and MAP 0.4204; LightGBM/XGBoost had better hit accuracy in the paper's betting scenarios. This is a ranking signal, **not** a comparable win-log-loss or profit claim for HKJC.
- Sugiura 2026: [leakage-aware temporal horse-race evaluation](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1896192/full), 4,556 future-held-out Japanese races. A matched simpler historical-feature model beat an augmented one on win ROC AUC (0.7543 versus 0.7293) and place ROC AUC (0.7513 versus 0.7164); test feature blocks by ablation rather than assuming richness helps.
- Wilkens 2026: [UK Plackett-Luce full-order study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6860338), a preprint reporting probabilities about as accurate as the market and no measurable lift from LLM-extracted commentary. Its selective ROI claim is marginally significant and not transferable to this dataset; full-order modeling is a later conditional candidate, not a wagering premise.
- [Deep Sets 2017](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) and [Set Transformer 2019](https://proceedings.mlr.press/v97/lee19d.html) offer permutation-aware models of whole fields, but neither establishes a gain on this archive. [TabPFN-2.5](https://arxiv.org/abs/2511.08667) reports strong general tabular results at smaller dataset scales. The initial 20% favors supported CPU training, distinct targets, and immediate leakage-auditable baselines; a race-set neural model is a later candidate after E1/E2/E3 are measured.

### Experimental Portfolio Decision
| ID | Candidate and target | Paper-backed hypothesis | First test and caveat |
| --- | --- | --- | --- |
| E1 | LightGBM LambdaMART, finishing-order ranking | Nonlinear interactions and explicit race grouping may improve ordering beyond linear conditional logit; the 2025 Korean comparison makes this worth testing. | Group by race; use verified finish positions as graded labels, not winner-only labels if the claim is full ranking. Primary NDCG@3 against a race-aware ranking baseline. Separately calibrate scores and measure win log loss against calibrated market; never call raw rank scores probabilities. |
| E2 | CatBoost classifier, top-3 probability | Ordered boosting and categorical handling may improve top-3 forecasts without naive target encoding. | Train a literal finish-position <=3 label and call it top-3, not an HKJC paid-place event until payout rules are encoded. Measure Brier/log loss and calibration against a field-size baseline. Top-3 probabilities do not sum to one. The paper demonstrates the algorithm, not an HKJC gain. |
| E3 | CatBoost regressor, recorded final win odds | Nonlinear categorical and historical interactions may predict the *recorded final odds* as a separate market-forecast target. | Use only pre-race features, never the target race's final odds or derivatives. Evaluate held-out log-odds MAE and rank correlation against a simple pre-race baseline. This is **not** an intra-race odds-movement forecast, which needs timestamped snapshots. |
| Later | Plackett-Luce full finishing-order model | Full order may carry more signal than winner-only labels; UK preprint provides a testable hypothesis. | Admit only after ties, dead heats, non-finishers, and full-order label coverage are audited. Compare to Benter on the same win endpoint and to a full-order baseline on order likelihood; avoid post-race commentary as a same-race feature. |
| Later | XGBoost grouped ranking | An independent boosted implementation may reveal whether gains are model-specific. | Defer until E1 is informative; otherwise this is a near-duplicate experiment. |

The first 20% campaign is a **versioned, fixed portfolio** of E1/E2/E3, not one model. A durable experimental-slot counter assigns these IDs round-robin; five experimental slots in a 26-attempt prefix yield 2/2/1. The planner chooses a hypothesis and bounded search within its assigned ID, not the ID or lane share. A failed prerequisite pauses that ID's slot with a recorded reason; it does not silently become Benter or another experimental target. Later candidates require a human-approved portfolio-version change and bounded canary. Allocation is by durable attempt count, not equal CPU-hours.

## Root-Cause Baseline
- Scope: production-adjacent scientific remediation; preserve live v17 identity and results.
- Proven: current `RaceProbabilityModel.fit` fits per-horse binary loss, normalizing only at inference. On 2026-09-27 the best v17 saved package had fundamental weight 0 and market weight about 1.10; seven of the top eight saved packages had fundamental weight 0. All 48 programs inspected earlier targeted win probability. This explains a market-dominated plateau, not an irreducible mathematical local optimum.
- Likely: the combined objective rewards market recalibration rather than stronger fundamental learning. Possible: historical odds timestamp semantics and source shifts affect comparison. Disproven: the 26-slot batch is a total trial cap, or sklearn binary logit is Benter conditional logit.
- Missing: prospective odds timing, full point-in-time feature audit, untouched holdout. Therefore no profit/live-betting claim.
- Missing evidence is unavailable because the historical archive lacks verified bet-time timestamps and no prospective campaign has yet completed under the new protocol. This uncertainty constrains remediation to supervised research rather than wagering or promotion.
- Mutation boundary: new branch and `agentic_v3` release/campaign only, read-only probes of v17; no shared-service restarts or changes to prior ledgers/MLflow.
- Remediation mapping: horse-wise objective -> conditional race likelihood; zero model weight -> explicit calibrated-market control and incremental metrics; planner target drift -> deterministic allocation; underjustified experimental models -> paper-backed registry. Timeout/rolling work queue is separate risk reduction.
- Not done if a binary classifier is only normalized afterward, allocation exists only in prompt text, or old/new scientific protocols are mixed as equivalent.

## SOTA, Standards, And Best Practices
- Prior art: Benter and Bolton-Chapman race-conditional logit; modern group ranking with LightGBM; CatBoost/XGBoost as documented alternatives.
- Official docs/specs: the linked papers, SciPy optimize, sklearn preprocessing, Optuna ask/tell, and LightGBM group-ranking API are the implementation references.
- Mature libraries: SciPy, sklearn, LightGBM, CatBoost, Optuna, Pydantic, and MLflow. Implementation decision: add conditional likelihood and the three versioned experimental programs, then enforce a two-lane policy with a deterministic experimental portfolio.
- Best practices: immutable dataset/code/protocol campaign identity, train/calibration/score split, deduplicated recipes, matched-race comparisons, and no automatic promotion.
- Use SciPy L-BFGS for the race-conditional likelihood, sklearn train-only sparse preprocessing, Optuna per-program ask/tell, and chronological protected folds. Exclude an intercept that cancels within each race. Define the fundamental probability as `softmax(X beta)` within each eligible race; optimize mean negative log winner probability plus a declared regularizer.
- The second-stage Benter combination is `softmax(alpha * log(p_fundamental) + beta * log(p_market))` by race. Fit `alpha` and `beta` only on later calibration races using fundamental predictions from a model fitted on earlier races. Fit a separate calibrated-market-only `softmax(beta_market * log(p_market))` control on those same calibration races. Freeze both before scoring the next chronological fold.
- Use mature LightGBM group ranking and CatBoost classification/regression for the experimental portfolio. Audit label coverage, sort race groups, and calibrate probabilities on later races. Do not interpret NDCG or raw rank scores as probability quality.
- Reject: sklearn `multi_class="multinomial"` fitted to independent binary horse labels; generated code; uncalibrated ranking scores; comparing unmatched targets; assuming final odds are available at bet placement.

### Execution Contract
- Register `benter_conditional_logit` in the win-probability model contract; leave legacy `logit` unchanged for historical comparability. Register explicit E1 ranking, E2 place, and E3 `recorded_final_win_odds` contracts. E1's optional calibrated win probabilities share held-out races with Benter; E2 and E3 remain separate targets with separate metrics and baselines.
- For the opt-in allocation policy, define `B(n)=round(4*n/5)` for `n` durable attempt reservations. The next slot belongs to Benter exactly when `B(n+1)>B(n)`; otherwise it belongs to experimental. The ledger, not the current process counter, owns `n` across restarts. Failed attempts keep their lane; retries are new attempts only with explicit provenance.
- The controller fills each lane only from registered programs of that lane. For an experimental slot, select the next E1/E2/E3 ID from a durable counter before asking for a proposal. If its program has no capacity, request a bounded proposal for that ID; if the provider fails, record degradation and use an approved same-ID deterministic seed or pause. Never borrow another ID's or lane's slots. Keep program trial budgets independent of concurrency.
- A proposal names `lane`, assigned `experiment_id`, `target_id`, `model_kind`, parent trials, evidence ID, feature schema/transform, search space, and total trial budget. Validate all fields before registration. Do not let the LLM choose lane ratio, arbitrary features, package versions, or evaluator code.
- Persist lane, portfolio version, experiment and target IDs, model contract version, preprocessing version, source paper, dataset/protocol/code identity, target-specific baselines, and applicable combination weights in the attempt, package, MLflow run, and cycle trace. `agentic_v3` uses separate `ima-agentic-v3` MLflow experiment and registered-model namespace. Cross-target raw metrics must not share a leaderboard; cross-campaign score comparisons must state the changed protocol.
- The primary selection gate is paired improvement over calibrated-market-only on protected development races, not blended log loss alone. Report the standalone fundamental score and zero-weight rate. Any candidate with zero model contribution remains a valid negative result, not a promoted winner. The untouched holdout and prospective paper trading stay closed until a later explicit approval.

## Dependency and Tooling Preflight
- Existing: Python venv, SciPy, sklearn, Optuna, MLflow, Pydantic, fixture data, Tailscale SSH. LightGBM is absent locally at baseline; CatBoost availability must be checked.
- Add LightGBM and CatBoost to research dependencies; install project-locally using `uv sync --extra research` or equivalent and verify local/server import. No UI/browser is in scope.
- Install/repair command: `uv sync --extra research` (fallback: `.venv/bin/python -m pip install -e '.[research]'`). Browser/runtime binaries: no browser required for the terminal system; generated plan dashboard gets a static HTML open/readback only.
- Browser smoke: open the generated plan dashboard HTML and confirm that the plan title, phases, and checklist render; no product UI changed.
- Only genuine blockers: incompatible wheel/registry, credentials, or inability to isolate a server release.

## Deterministic Real-User Test
- Entry point: `ima-optimize run`; operator workflow: launch fixture campaign, request status, inspect ranked/comparative JSON and MLflow outputs.
- Stable inputs/fixtures: `tests/fixtures/research_races.csv`, fixed seed and protocol; observable assertions are the lane counts, scores, probability sums, and package reload.
- Run `python -m scripts.optimize run --campaign <temp> --config <fixture>` with fixed seed and bounded trial count; read `status.json`, `decisions.jsonl`, program index, package, and MLflow.
- Assert lane counts after 5, 10, 25, 26, 50, and 52 scheduled attempts (4/1, 8/2, 20/5, 21/5, 40/10, and 42/10 respectively); experimental IDs cycle E1/E2/E3 under failure/resume; race probability sums; finite-difference gradient; shuffled row invariance; invalid winner rejection; no current-race market feature or final-odds leakage; no calibration/score leakage; target-specific labels/metrics; package reload; provider failure; stop/resume; exhausted lane; malformed proposal. Count durable reservations, including failed trials; log successful counts separately so retries cannot bias the allocation report.

## Fulfillment and Readback Proof
- Final result: a versioned research model pipeline and an isolated server campaign. Required write/mutation: commits, new campaign ledger/decisions/packages, and MLflow runs; readback surface: CLI status plus campaign JSON and MLflow API.
- Expected content/counts: four Benter and one experimental trial per complete five, explicit comparator metrics and weights, and no identity change in v17. Evidence: local tests, CLI output, package prediction, server status, and MLflow query.
- Deliver committed code and an isolated completed real-data canary containing Benter plus E1/E2/E3, each with appropriate baselines, metrics, model packages, and MLflow lineage. Read back the same campaign surfaces the operator uses.
- A live process, fixture-only result, unlogged model, reset of v17, or raw rank score called a probability is not fulfillment.

## Armageddon Mode
- Attack scope: systemic model/scientific contract. Failure modes: malformed dataset, dependency outage, provider timeout, interrupted process, tracking outage, and resource pressure. Optimization opportunities: sparse preprocessing and avoiding duplicate full-data reads.
- Systemic risk. Attack empty/single-horse races, multiple/no winners, unseen categories, missing features, unsorted groups, nonfinite logits, batch-size remainders, failures, interrupted tells, provider timeout, exhausted budgets, OOM, and oversubscription.
- Benchmark one real-data fit of each family before scaling. Bound threads/workers by measured memory. Reject any label leakage, wrong group boundaries, allocation drift, or incompatible package load.

## Generality Guardrail
- Existing mechanism: shared research ledger/search/executor/tracking. Recurrence likelihood: high because every future model family needs the same allocation and provenance. General mechanism decision: reuse one model registry and opt-in lane scheduler, no one-off scheduler process.
- Reuse `PipelineRecipe`, `ResearchLedger`, `ProgramSearchController`, protected folds, MLflow outbox/traces, and package loader. One lane mechanism must serve bootstrap, remote planner, and resume; future paper-backed candidates can join the experimental registry.

## Regression Guardrails
- Edit surface: `ima/modeling.py`, `ima/research_models.py`, `ima/research_specs.py`, `ima/research_targets.py`, `ima/research_search.py`, `ima/research_store.py`, `ima/research_controller.py`, `ima/research_executor.py`, `ima/research_evaluation.py`, `ima/research_model_package.py`, `ima/mlflow_tracking.py`, `ima/openrouter_orchestrator.py`, `ima/optimizer.py`, `scripts/optimize.py`, `config/agentic_v3.json`, `pyproject.toml`, focused tests, this plan and `docs/AGENTIC_V3_RUNBOOK.md`. Scope expansions require review.
- Protect old model semantics, package loading, legacy optimizer, fold/label contracts, v17 server campaign, other services, and paper-mode betting.
- Damage: systemic; branch `feat/benter-race-research`. Atomic commits: scientific plan, Benter model, experimental portfolio, allocation, rollout/readback.
- Branch strategy: the dedicated branch was created before plan or code edits because the scientific and scheduler contracts have systemic blast radius.

## Phase 1: Baseline

### Subphase 1.1: Freeze scientific contract
- Commit: `docs(research): specify race-aware Benter allocation` (future implementation turn; not performed now).
- Tests: `mega_plan_check.py docs/BENTER_RACE_RESEARCH_PLAN.md`; read-only v17 identity/status.
- Success Criteria: evidence, controls, allocation denominator, primary sources, and mutation boundary are explicit.
- Planned Touch Files:
  - `docs/BENTER_RACE_RESEARCH_PLAN.md`
- Checklist:
  - [x] Read paper, code, and live results.
  - [x] Validate plan/dashboard; leave uncommitted as requested.

## Phase 2: Model

### Subphase 2.1: Conditional logit and matched baselines
- Commit: `feat(research): train Benter race probabilities`.
- Tests: `.venv/bin/python -m unittest tests.test_research_models tests.test_research_executor tests.test_research_specs` and fixture CLI.
- Success Criteria: correct race likelihood/gradient, valid probabilities, market-only control, incremental metrics, package roundtrip.
- Planned Touch Files:
  - `ima/modeling.py`
  - `ima/research_models.py`
  - `ima/research_specs.py`
  - `ima/research_executor.py`
  - `ima/research_model_package.py`
  - `ima/mlflow_tracking.py`
  - `ima/research_evaluation.py`
  - `tests/test_research_models.py`
  - `tests/test_research_executor.py`
  - `tests/test_research_specs.py`
- Checklist:
  - [ ] Implement Benter and matched controls.
  - [ ] Prove fixture package and leakage guards.

### Subphase 2.2: Paper-backed experimental ranker
- Commit: `feat(research): add LightGBM race-ranker candidate`.
- Tests: focused model/spec/executor suite and bounded real-data benchmark.
- Success Criteria: sorted race groups, graded finishing labels, calibrated race probabilities as a separate endpoint, paper/provenance logging, no superiority claim.
- Planned Touch Files:
  - `pyproject.toml`
  - `ima/modeling.py`
  - `ima/research_specs.py`
  - `ima/research_executor.py`
  - `tests/test_research_models.py`
  - `tests/test_research_executor.py`
  - `tests/test_research_specs.py`
- Checklist:
  - [ ] Install mature dependency and implement grouped ranker.
  - [ ] Run fixture and one real-data fit.

### Subphase 2.3: Place and final-odds programs
- Commit: `feat(research): add distinct place and final-odds targets`.
- Tests: focused target/model/executor suite, leakage fixtures, and one bounded real-data fit per program.
- Success Criteria: E2 uses documented place rules and calibrated place probabilities; E3 predicts recorded final win odds using no same-race final market fields; each has a separate target-specific baseline and metrics.
- Planned Touch Files:
  - `pyproject.toml`
  - `ima/research_targets.py`
  - `ima/research_models.py`
  - `ima/research_specs.py`
  - `ima/research_executor.py`
  - `ima/research_evaluation.py`
  - `tests/test_research_models.py`
  - `tests/test_research_executor.py`
  - `tests/test_research_specs.py`
- Checklist:
  - [ ] Audit place labels, final-odds coverage, and pre-race feature availability.
  - [ ] Add versioned E2/E3 contracts and prove target-specific metrics and package reload.

## Phase 3: Allocation

### Subphase 3.1: Deterministic 80/20 lanes
- Commit: `feat(optimizer): enforce Benter experimental allocation`.
- Tests: `.venv/bin/python -m unittest tests.test_research_search tests.test_research_controller tests.test_agentic_planner tests.test_agentic_optimizer_e2e`.
- Success Criteria: durable ratio and E1/E2/E3 rotation under bootstrap, remote planning, fallback, resume, failures, and 26-slot batches. Budgets remain independent of concurrency.
- Planned Touch Files:
  - `ima/optimizer.py`
  - `ima/research_search.py`
  - `ima/research_store.py`
  - `ima/research_controller.py`
  - `ima/mlflow_tracking.py`
  - `ima/openrouter_orchestrator.py`
  - `scripts/optimize.py`
  - `tests/test_research_search.py`
  - `tests/test_research_controller.py`
  - `tests/test_agentic_planner.py`
  - `tests/test_agentic_optimizer_e2e.py`
- Checklist:
  - [ ] Add opt-in lanes for new campaigns only.
  - [ ] Prove deterministic lane counts, experimental ID rotation, and failure paths.
  - [ ] Replay durable pending reservations before any new slot; pruned duplicate Optuna recipes must not consume executable budget.

## Phase 4: Verification

### Subphase 4.1: Full suite and isolated rollout
- Commit: `docs(research): record Benter rollout and comparison`.
- Tests: full unittest suite, fixture CLI, real-data canary, package reload, MLflow readback, old-campaign status.
- Success Criteria: Benter and all three initial experimental programs produce honest target-specific evidence; v17 remains intact; commit/push/deploy evidence separate.
- Planned Touch Files:
  - `docs/AGENTIC_OPTIMIZER_V2_RUNBOOK.md`
  - `docs/AGENTIC_V3_RUNBOOK.md`
  - `docs/BENTER_RACE_RESEARCH_PLAN.md`
  - `config/agentic_v3.json`
- Checklist:
  - [ ] Verify local code and full suite.
  - [ ] Deploy isolated `agentic_v3` release and canary as `imaopt` while v17 remains read-only; inspect each target, package, MLflow run, and trace.
  - [ ] Start the bounded-resource continuous `agentic_v3` controller only after the canary passes; verify process, natural trial, and resource headroom.
  - [ ] Record limitations and old-campaign readback.
