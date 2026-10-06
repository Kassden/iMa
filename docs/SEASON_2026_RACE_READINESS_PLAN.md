# 2026 Season Evaluation And October 7 Predictions

## GOAL
Audit zero-weight blending, evaluate frozen pre-season-selected models on all completed local HKJC September2026-season races, and deliver October7 Happy Valley probabilities and HKD10 straight-ticket paper results. Never place bets.

## Acceptance Criteria
- Official fixture coverage denominator, hashes, race/runner counts and exclusions; complete nine-race tomorrow card.
- Benter, boosted and existing probability-pool champion tested on identical complete races; probit if executable. Models frozen before September; selection on September results is labelled exploratory.
- Past-only features, no fake outcome or odds for live queries, no model fitting on season test labels.
- WIN, PLACE, QIN, QPL, TRIO, TIERCE, FIRST4 and QUARTET ledgers: HKD10 per straight combination, stakes, gross returns, profit, hit rate and return-on-stake. Boxes charge all combinations; refunds/dead heats explicitly handled or excluded.
- Estimated EV requires a full quoted payout for that specific ticket. Winner-only final dividends cannot supply losing exotic quotes. Separate realized returns, hindsight-priced EV and actionable quoted EV. Unknown remains unknown; no double takeout subtraction.
- Bankroll/reinvestment assumptions explicit. No guarantee of profit or growth.
- Tomorrow ranked horse probabilities, pool combinations and break-even dividends; actual quoted EV only if odds available. No fabricated predictions for unexecutable packages.

## Research
Use repository Benter paper pp187-190, MarketBlend, ResearchModelPackage and existing corrected Plackett-Luce adapters. Official HKJC-only inputs: racecards, local results, fixtures, betting rules.
Sources: https://racing.hkjc.com/en-us/local/information/racecard?racedate=2026%2F10%2F07&Racecourse=HV&RaceNo=1 ; https://special.hkjc.com/e-win/en-US/betting-info/racing/flexi-bet/info/ ; https://campaign.hkjc.com/en/qtt-f4/qtt_flexibet.aspx . Straight HKD10 combinations avoid conditional Flexi eligibility; do not treat a box as one bet.

## Root-Cause Baseline
Evidence inventory: prior blend audit metrics/readbacks, OpenRouter account and HTTP402 logs, live campaign config/state and inspected query-filter code. Remediation mapping: query support addresses dropped live rows; convergence/alignment diagnostics address suspected fitting defects; explicit price/settlement separation addresses hindsight EV bias.
Proven:149 older blends have a=0; stronger recent fundamentals were not blended. OpenRouter returns402, account exhausted; live planner acceptance blocked. Existing rich-feature function drops untimed queries. Frozen packages impose future-date/exact-feature contracts.
Likely: redundant fundamentals. Possible: convergence, alignment or calibration defects; inspect before modifying math. Missing evidence: historic losing-ticket exotic quotes and unpublished tomorrow odds; do not infer these from winner dividends. Mutation boundary: scoped new readiness modules/tests, own-user remote reads and bounded standalone training only if necessary. Preserve all existing fits and unrelated apps. Remedies map query support to dropped rows, independent fitting diagnostics to optimizer suspicion, price/settlement separation to EV bias. Not done with active service alone, incomplete coverage hidden or fabricated data.

## SOTA, Standards, And Best Practices
Reuse pandas/NumPy/SciPy, frozen joblib packages, official parsers, Benter-corrected order probabilities. Use temporal holdout and meeting-cluster uncertainty. Check convex log-softmax blend loss/gradients independently. Reject season-label tuning followed by same-season validation, market payout invented from model fair odds, or publication-time leakage.

### Evidence-Driven Protocol Clarifications
- Older saved packages are research-fold estimators, not full-history final fits. Actual training dates must come from their fitting populations, not feature-context cutoff labels. Fresh final fits of the same fixed recipes train before January14,2026 and preserve older packages separately.
- All78 season fields receive matched archived official declaration inputs. Older calibration PDFs can return404: log this explicitly. Within the predeclared last500 calibration candidates, fit calibration only on whole fields whose current horse ratings are finite. This is an input-quality filter, never a result/log-loss filter; keep the full500 denominator and report the usable count. Do not silently omit season test races.
- Tomorrow public odds are now published. Capture ordinary public WIN/PLACE and exotic listings without login/betting or bypassing restrictions. Verify pool-specific quote units; retain capture/update time, incomplete Top20 lists and possible999 ceilings. Unknown units/capped quotes are not exact EV inputs. Historical losing exotic quotes remain unavailable, so historical returns are realized ROI rather than claimed actionable EV.
- Final verification identified inconsistent tie handling: summary hit rates used an unrounded maximum in result-sorted inputs, while settled tickets used rounded probabilities and numeric horse numbers. Both summaries and tickets must use the same result-independent rounded12/numeric-horse policy. Rerun old and fresh evaluations after the correction; never retain inflated outcome-dependent tie hit rates.

## Dependency and Tooling Preflight
Browser smoke: not applicable; no UI/browser-facing code is changed.
Existing .venv, pytest/unittest, pandas/parquet, joblib/SciPy, HKJC public web and own-user Tailscale SSH. Install or repair commands: no install is needed initially; repair only project-local missing tools. Browser runtime:none; terminal/report surface. Genuine blockers: official outage, absent sources, unpublished quotes, incompatible trusted model environment.

## Deterministic Real-User Test
Entry point: terminal evaluator. Workflow: collect, predict, settle and open report. Stable inputs or fixtures: seeded complete race. Observable assertions: exact costs, predictions and settlements. Command: `.venv/bin/python -m unittest tests.test_season_evaluation`.
Entry:`python -m scripts.evaluate_season_2026 --help` and seeded cached fixture. Known complete race, two probability vectors, official WIN/PLACE/exotic dividends, HKD10 tickets. Assert exact combination counts/stakes/settlements/break-even prices. Read generated markdown/CSV as the user will.

## Fulfillment and Readback Proof
Artifacts:`artifacts/race-readiness-20261007/` rawpages/manifests, frozen candidate provenance, query features, diagnostic, predictions and full paper ledger. Report:`docs/SEASON_2026_EVALUATION_AND_OCT07_PREDICTIONS.md`. Independently reconcile actual rows/coverage, predictions summing1, stake/gross/profit, exclusions and source hashes. Created-only/copied-only/stale or fabricated output is not done.

## Armageddon Mode
Moderate radius. Duplicate horse IDs, scratches, partial fields, dead heats, refunds, currency commas, missing odds/dividends, winner-only quote bias, NaNs, probability normalization, same-day/later leakage, paid-place count and exotic boxes. Quarantine incomplete races for comparative evaluation. Must-fix: silent fabricated values, wrong settlement or feature leakage. Record adversarial fixtures and independent real-ledger readback.

## Generality Guardrail
Reuse acquisition/package/pool primitives, add small shared pre-race-query and season-evaluation owners, not UI/framework. High recurrence for future meetings. Immutable date artifacts are evidence rather than new abstractions.

## Regression Guardrails
- Branch strategy:dedicated `feat/season-2026-race-readiness`; moderate evaluation/live-inference risk.
- Planned edit surface:new collector/features/evaluator/diagnostic and tests; rich_features explicit query mode only if needed; MarketBlend narrowly diagnosed validation only.
- Protected:dirty unrelated files, existing fits/campaigns, credentials, Cortex/Solar services/users, historical identities, no automatic bets.
- Consumers:terminal paper evaluator and frozen packages. Proof:fixture/adjacent regressions, independent verifier and actual official-data artifacts.

## Phase 1: Discovery
### Subphase 1.1: Official Coverage And Frozen Candidates
- Commit:feat(readiness): collect official season results and racecards
- Tests:parser fixtures, source hashes, full fixture denominator and package cutoffs.
- Success Criteria:all meetings enumerated with explicit per-race coverage; candidate list frozen before test scoring.
- Planned Touch Files:
  - `scripts/collect_season_2026.py`
  - `tests/test_season_2026_collection.py`
  - `docs/SEASON_2026_RACE_READINESS_PLAN.md`
- Checklist:
  - [x] Collect and validate official inputs.
  - [x] Freeze candidate provenance.

## Phase 2: Features And Diagnostics
### Subphase 2.1: Point-In-Time Queries And Blend Audit
- Commit:feat(readiness): build past-only queries and diagnose blending
- Tests:current/later outcome invariance, field alignment, independent convex loss/gradient.
- Success Criteria:complete real query predictions; zero optimum verified or narrowly repaired with evidence.
- Planned Touch Files:
  - `ima/season_features.py`
  - `tests/test_season_features.py`
  - `ima/rich_features.py`
  - `ima/modeling.py`
  - `tests/test_modeling.py`
  - `scripts/audit_market_blend.py`
  - `tests/test_modeling_blend_numerics.py`
- Checklist:
  - [x] Build queries without fabricated labels.
  - [x] Run fitting diagnostics and document evidence limits.

### Subphase 2.2: Fresh Fixed-Recipe Final Fits
- Commit:feat(readiness): refit fixed recipes with an independent pre-season calibration holdout
- Tests:strict training/calibration separation, original configuration parity, actual last training dates, immutable package hashes and prediction identity.
- Success Criteria:preserve older packages as benchmarks; fresh fixed Benter/boosted/pool configurations fit only before the first of the predeclared last500 calibration races (January14,2026). Evaluate common-variance probit if it completes within the bounded fit deadline; report any genuine timeout. No current-season labels influence fitting or model selection.
- Planned Touch Files:
  - `scripts/refit_season_2026.py`
  - `tests/test_season_refit.py`
  - `scripts/evaluate_season_2026.py`
- Checklist:
  - [x] Prepare immutable training inputs and freeze fixed configurations.
  - [x] Fit, save and verify genuinely fresh packages without changing live campaigns.
- Discovery:the older flat packages actually fit through March16,2024, while the pool fits through November13,2024. The existing feature-context cutoff is not the actual last training date. This repair is necessary to avoid presenting stale research-fold estimators as current race-day models.

## Phase 3: Season Evaluation
### Subphase 3.1: Frozen Predictions And Fixed-Ticket Settlement
- Commit:feat(readiness): evaluate season models and settlements
- Tests:exact WIN/PLACE/exotic settlement and missing-price behavior, complete comparable populations.
- Success Criteria:real reproducible season ledgers with fixed selection policies, no hindsight ticket selection.
- Planned Touch Files:
  - `ima/season_evaluation.py`
  - `scripts/evaluate_season_2026.py`
  - `scripts/predict_frozen_season.py`
  - `tests/test_season_evaluation.py`
  - `ima/pools.py`
  - `tests/test_pools.py`
  - `tests/test_season_readiness_cli.py`
- Checklist:
  - [ ] Run all executable candidates and matched market benchmark.
  - [ ] Reconcile stakes, gross and quote coverage independently.

## Phase 4: Predictions And Verification
### Subphase 4.1: Tomorrow Readback And Delivery
- Commit:docs(readiness): publish season audit and October7 predictions
- Tests:nine official races, all active runners, normalized finite probabilities, cutoff and payout arithmetic, adjacent regressions.
- Success Criteria:actual model output and quantitative report; genuine missing external inputs explicitly blocked, not hidden.
- Planned Touch Files:
  - `docs/SEASON_2026_EVALUATION_AND_OCT07_PREDICTIONS.md`
  - `docs/OCT07_2026_RANKED_PREDICTIONS.csv`
  - `scripts/report_season_2026.py`
  - `tests/test_season_report.py`
  - `scripts/log_season_readiness.py`
  - `tests/test_season_tracking.py`
  - `scripts/forecast_season_market.py`
  - `tests/test_season_market_forecast.py`
- Checklist:
  - [ ] Produce and independently verify predictions.
  - [ ] Apply only pre-season fitted blending coefficients to the complete current public WIN snapshot for separate forward-looking combined forecasts; never substitute early prices for historical final odds or claim a tested live betting edge.
  - [ ] Publish separate descriptive season-evaluation MLflow records with bound artifact hashes; never relabel them as agentic trials or alter campaign runs.
  - [ ] Commit/push and final fulfillment gate.
