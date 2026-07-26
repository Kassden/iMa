# Multi-Pool Market-Combined Prediction

## GOAL
- Produce auditable live probabilities for standard HKJC single-race pools from market-combined runner strengths and show the complete transformation pipeline in the experiment dashboard.

## Acceptance Criteria
- [x] Live inference returns ranked probabilities and fair odds for WIN, PLACE, QIN, QPL, TRI, TIERCE, FIRST4, and QUARTET.
- [x] Fundamental and timestamped WIN-market probabilities are combined before pool expansion.
- [x] First-time or otherwise unratable runners retain the public-market fallback.
- [x] Dashboard users can compare fundamental, combined, and market metrics explicitly.
- [x] Dashboard contains a responsive pipeline graph with every major data transformation and prediction output.
- [x] Python tests, browser interactions, canvas rendering, and production deployment pass.

## Out Of Scope
- Authenticated wager submission changes.
- Arbitrary pool-specific market blend weights without timestamped historical pool prices and outcomes.
- Claims that result-page final odds were tradable before the race.

## Research
- Existing implementation: `MarketBlend` performs a fitted log-opinion pool over calibrated fundamental and market-implied runner probabilities.
- Existing pool engine: fitted Benter second/third-place exponents expand runner strengths into ordered and unordered combination probabilities.
- Data constraint: the archive has complete final WIN odds but incomplete timestamped historical exotic-pool prices, so pool-specific blend weights cannot yet be fitted without leakage or unsupported assumptions.
- Selected approach: combine timestamp-valid WIN market information at runner level, then derive coherent pool probabilities and compare them with live pool prices downstream.

## Regression Guardrails
- Planned edit surface: `ima/inference.py`, `ima/pools.py`, `ima/experiments.py`, `scripts/predict_pools.py`, `tests/test_operations_inference.py`, `tests/test_modeling.py`, `tests/test_experiments.py`, `docs/model-results/dashboard-template.html`, `docs/MULTI_POOL_MARKET_PIPELINE_PLAN.md`, `public/**`, `README.md`.
- Protected behaviors: race-normalized WIN probabilities, public fallback for unrated runners, fitted order exponents, chronological evaluation, scraper contracts, and Kelly limits.
- Likely consumers: live prediction jobs, paper trading, experiment review, and Vercel dashboard users.
- Damage radius: moderate.
- Proof plan: focused pool tests, full unittest suite, compile/diff checks, generated artifact checks, desktop/mobile browser QA, production HTTP and JSON checks.

## Phase 1: Multi-pool inference

### Subphase 1.1: Expand blended runner strengths into pool combinations
- Commit: add ranked multi-pool live prediction contracts and inference.
- Tests: pool coverage, probability coherence, combination cardinality, fair odds, and market fallback.
- Success Criteria: one live race frame produces ranked outputs for every supported single-race pool.
- Planned Touch Files:
  - `ima/inference.py`
  - `ima/pools.py`
  - `scripts/predict_pools.py`
  - `tests/test_operations_inference.py`
  - `tests/test_modeling.py`
- Checklist:
  - [x] Define canonical supported pool names and aliases.
  - [x] Reuse calibrated market-combined WIN probabilities as order strengths.
  - [x] Emit coherent combination probabilities and fair odds.

## Phase 2: Market-combination visibility

### Subphase 2.1: Expose source-specific experiment metrics
- Commit: publish fundamental, market, and combined probability paths in dashboard data.
- Tests: source selection and complete metric payload.
- Success Criteria: dashboard can switch between model-only and market-combined performance without ambiguity.
- Planned Touch Files:
  - `ima/experiments.py`
  - `tests/test_experiments.py`
- Checklist:
  - [x] Preserve existing fundamental and blended metrics.
  - [x] Add explicit source metadata and timestamp limitations.
  - [x] Keep market benchmark visible for every run.

## Phase 3: Pipeline dashboard

### Subphase 3.1: Visualize transformations and pool outputs
- Commit: add probability-source controls and responsive pipeline graph.
- Tests: required controls, pipeline nodes, rendered canvases, mobile overflow, and downloads.
- Success Criteria: a browser user can trace raw inputs through market blend to every pool output.
- Planned Touch Files:
  - `docs/model-results/dashboard-template.html`
  - `tests/test_experiments.py`
  - `public/index.html`
  - `public/results.json`
  - `public/results.csv`
- Checklist:
  - [x] Add Fundamental, Combined, and Market source selector.
  - [x] Add data-transformation pipeline graph and explanatory legend.
  - [x] Display supported pool outputs and market timestamp warning.

## Phase 4: Verification and deployment

### Subphase 4.1: Prove and publish
- Commit: update operator documentation and generated dashboard assets.
- Tests: full tests, compile, diff, Megaskill checks, browser desktop/mobile QA, Vercel production verification.
- Success Criteria: branch is pushed, PR updated, production dashboard is Ready, and no local server remains.
- Planned Touch Files:
  - `README.md`
  - `docs/MULTI_POOL_MARKET_PIPELINE_PLAN.md`
  - `.mega/evidence.jsonl`
- Checklist:
  - [x] Run focused and full verification.
  - [x] Commit and push atomic changes.
  - [x] Redeploy and verify the production dashboard.
