# Preseason Probability Pool Search

## GOAL
Run a cheap deterministic search over existing frozen Benter/boosted predictions, confirm on later preseason races, and deliver audited season comparisons and October7 forecasts without retraining base estimators or placing bets.

## Acceptance Criteria
- Preserve the original season report, model packages, forecasts and all live campaigns.
- Test both original-pool components and standalone Benter/boosted configurations:21 weights,2 methods,7 temperatures=588 configurations.
- Use only finite-rating whole fields from the existing500 reserved preseason races. Split unique meeting dates60/20/20 into search,confirmation,downstream calibration. Never split a meeting between roles.
- Choose one candidate by search race log loss. Adopt only if it beats the fixed60/40 original-pool baseline on later confirmation races. Never choose based on September results.
- Fit market blending and order exponents only in the final disjoint calibration window. Compare baseline and challenger under the same restricted downstream calibration, not only against the previous full-calibration report.
- Keep all78 season races and108 tomorrow runners; exactHKD10 ticket ledgers and quote-unit safeguards. Label season results exploratory because already inspected.
- Publish search, split identities, selected recipe, confirmation/adoption decision, forecasts and metrics to distinct MLflow records. No fake LLM traces/costs.

## Research
Existing weighted_probability_pool,log_probability_pool,TemperatureCalibrator,MarketBlend and settlement owners. Sources:https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier and https://scikit-learn.org/stable/modules/calibration.html . Weighted averaging and temperature calibration are mature tools; no new model framework or complex stacking needed.

## Root-Cause Baseline
Evidence inventory:inspected fitted graph, prior selection manifests, hash-bound prediction readbacks and recorded season metrics. Missing evidence:uncaptured historical losing-ticket exotic quotes are unavailable; constrain EV to verified actual quotes. Remediation mapping:fixed weights map to deterministic search; possible overfitting maps to later confirmation; calibration leakage maps to disjoint downstream fitting.
Proven:the60/40 pool weights are inherited and fixed; original component configurations differ from standalone candidates. Original packages and complete guarded prediction caches exist. Hypothesis:tuned weighting/pooling/temperature improves transfer. Disproven:any claim that prior development improvement guarantees season profit. Missing:component predictions must be extracted from trusted fitted graph; no historical losing exotic quotes. Remediation:cache components once, compare fixed small grid, chronological confirmation, independent downstream calibration. Mutation boundary:new modules/CLIs/tests/report and own-user isolated inference/publication only.

## SOTA, Standards, And Best Practices
Implementation decision:reuse existing mature libraries and official documentation; rejected approaches include expensive refits, unrestricted stacking and season-label optimization.
Reuse NumPy/SciPy stable logsumexp, pandas identity joins, existing race loss, package integrity and pool/settlement primitives. Include0/100 and100/0 endpoints, temperature1 and the original60/40 baseline. No fitting on scored season labels. No random horse-level splits. No forced positive model weight. Do not claim588 independent training runs:these are cached probability combinations.

## Dependency and Tooling Preflight
No install is needed because project dependencies already execute; genuine blockers are SSH outage or original-environment incompatibility.
Existing .venv and own-user Tailscale SSH, original pinned source/environment and fitted packages. No installation needed. If unavailable, test cached standalone candidates and report original-component blocker rather than fabricate. Browser smoke:not applicable; CLI/artifact workflow. Setup commands:existing .venv/bin/python,ssh BatchMode only; no shared services/user changes.

## Deterministic Real-User Test
Entry point:`.venv/bin/python -m scripts.search_season_pool --help`. Seeded complete races with known two-model probabilities; search table, chronological split receipt, selected recipe and normalization. Command:`.venv/bin/python -m unittest discover -s tests -p test_probability_pool_search.py`.

## Fulfillment and Readback Proof
New artifact root:artifacts/pool-search-20261007. Verify all source hashes,588 configurations, disjoint meeting/race roles, frozen choice before season scoring, original-pool reconstruction, all78race ledgers and108runner forecasts. Independently recompute choice, race loss and returns. Read MLflow metrics and artifact hashes after upload; commit/push/merge with CI.

## Armageddon Mode
Attack scope:CLI input validation and numerical edge cases. Must-fix threshold:any wrong identity, temporal leakage or incorrect monetary settlement. Optimization opportunities:cache read-only predictions rather than retrain.
Moderate radius. Duplicate/misaligned identities, NaN/negative probabilities, missing complete fields, unnormalized inputs, invalid weight/temperature, zeros and tiny probabilities, equal-loss tie ordering, altered source hashes, outcomes outside search role, same-day overlap, training/calibration overlap, missing/capped/unverified quote units. Reject corruption; never overwrite existing evidence.

## Generality Guardrail
Existing mechanism:race-aware probability pooling and common settlement helpers. Recurrence:future race meetings need the same bounded search; add a reusable numerical owner without campaign changes.
Small reusable probability combination/search owner; date-specific CLI orchestrates existing acquisition, calibration and settlement. No live campaign or planner redesign, UI, model retraining or automatic bets.

## Regression Guardrails
- Branch strategy: dedicated feat/preseason-pool-search for moderate cross-artifact risk.
Branch:feat/preseason-pool-search. Protected:all existing packages and original report, dirty unrelated files, campaigns,Cortex/Solar users/services. Consumers:terminal research evaluation and descriptive MLflow. Proof:focused/adjacent/full regression tests and independent actual-artifact verification. Atomic units:probability math+tests;component inference+tests;CLI/report/publication; documentation. No unrelated refactors.

## Phase 1: Cached Components
### Subphase 1.1: Trusted Inference
- Commit:feat(research): cache trusted fitted pool component probabilities
- Tests:package/source hashes,training cutoff,component output validation and original60/40 reconstruction.
- Success Criteria:7292rows/587races per component, guarded receipt, no retraining, original evidence unchanged.
- Planned Touch Files:
  - `scripts/predict_pool_components.py`
  - `tests/test_pool_component_inference.py`
  - `docs/PRESEASON_POOL_SEARCH_PLAN.md`
- Checklist:
  - [ ] Validate source dependencies and cached predictions.
  - [ ] Extract and verify original pool components once.

## Phase 2: Search And Confirmation
### Subphase 2.1: Frozen Chronological Search
- Commit:feat(research): search probability pools with chronological confirmation
- Tests:all endpoints,arithmetic/geometric/temperature,normalization,split integrity,selection outcome invariance.
- Success Criteria:588 search rows,one search-selected candidate,later confirmation decision,disjoint downstream calibration.
- Planned Touch Files:
  - `ima/probability_pool_search.py`
  - `tests/test_probability_pool_search.py`
  - `scripts/search_season_pool.py`
  - `tests/test_season_pool_search_cli.py`
- Checklist:
  - [ ] Run deterministic search and freeze candidate.
  - [ ] Confirm or retain baseline without season selection.

## Phase 3: Readback And Delivery
### Subphase 3.1: Evaluation Forecasts And Tracking
- Commit:feat(research): publish confirmed pooling comparison and forecasts
- Tests:exact settlements,allrunner probabilities,market quote guards,MLflow readback and adjacent regressions.
- Success Criteria:78race comparisons,9race forecasts,explicit adoption result,verified artifact publication,CI and merge.
- Planned Touch Files:
  - `scripts/search_season_pool.py`
  - `tests/test_season_pool_search_cli.py`
  - `scripts/report_pool_search.py`
  - `tests/test_pool_search_report.py`
  - `scripts/forecast_season_market.py`
  - `tests/test_season_market_forecast.py`
  - `scripts/log_pool_search.py`
  - `tests/test_pool_search_tracking.py`
  - `docs/PRESEASON_POOL_SEARCH_RESULTS.md`
- Checklist:
  - [ ] Generate comparison and forecasts without changing original output.
  - [ ] Independently verify,publish,commit,push and merge.
