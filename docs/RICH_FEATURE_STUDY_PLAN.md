# Benter-Inspired Rich Feature Study

## GOAL
- Build and evaluate a leakage-safe rich horse-racing feature set from Benter factor families and repository notebooks, with correlation, significance, redundancy, and out-of-sample contribution rankings.

## Acceptance Criteria
- [ ] Preserve the current 23-feature baseline and evaluate the rich challenger on the identical chronological train, validation, and test races.
- [ ] Implement a versioned rich feature contract derived only from information available before each race.
- [ ] Cover every Benter factor family that the local archive can support and disclose unsupported or partial factors.
- [ ] Incorporate leakage-safe notebook ideas: recent form, speed, distance, weight, odds history, jockey/trainer history, and days since last race.
- [ ] Add auxiliary trackwork, barrier-trial, and sectional features with explicit availability indicators and coverage metrics.
- [ ] Produce pairwise correlation, redundancy flags, target association significance, mutual information, and held-out permutation importance.
- [ ] Rank individual variables and feature families by deterioration in race log loss when permuted.
- [ ] Publish baseline-versus-rich metrics and feature rankings in the browser dashboard.
- [ ] Full tests, compilation, scope checks, Playwright desktop/mobile QA, commits, push, and Vercel production verification pass.

## Out Of Scope
- Treating Benter's unpublished proprietary factor definitions as known.
- Using post-race comments, incidents, dividends, final outcomes, or future horse snapshots as pre-race mutable information.
- Promoting the challenger to automated wagering without forward paper-trading evidence.

## Research
- Built-in options: `ima/data.py`, `ima/modeling.py`, the three feature notebooks, and normalized historical auxiliary tables.
- Paper source: `research/1994-benter.pdf`, especially current condition, past performance, performance adjustments, situational factors, and preference factors.
- Statistical methods: Spearman redundancy matrix, point-biserial association with false-discovery-rate correction, mutual information, and held-out race-log-loss permutation importance.
- Selected pattern: preserve baseline constants; add a named rich feature schema and a separate challenger study before any production promotion.

## Regression Guardrails
- Planned edit surface: `scrapper/historical/results.py`, `ima/historical_sources.py`, `scripts/reparse_official_history.py`, `tests/test_historical.py`, `tests/test_historical_sources.py`, `ima/feature_sets.py`, `ima/rich_features.py`, `ima/feature_analysis.py`, `ima/modeling.py`, `scripts/run_feature_study.py`, `tests/test_rich_features.py`, `tests/test_feature_analysis.py`, `tests/test_modeling.py`, `tests/test_experiments.py`, `docs/model-results/dashboard-template.html`, `public/index.html`, `public/results.json`, `docs/data/HISTORICAL_COVERAGE.md`, `docs/RICH_FEATURE_STUDY_PLAN.md`, regenerated normalized historical outputs, and generated `artifacts/feature-study/*` reports.
- Protected behaviors: baseline feature schema, chronological splitting, calibration, market blending, pool probabilities, first-timer fallback, experiment history, and authenticated execution safeguards.
- Likely consumers: model research, live inference, meeting finalization, dashboard users, and wagering evaluation.
- Damage radius: systemic.
- Proof plan: synthetic point-in-time tests, same-race leakage tests, feature-contract tests, statistical-analysis tests, baseline regression metrics, rich held-out evaluation, full suite, Playwright dashboard QA, and production HTTP verification.
- Atomic units: feature contract; feature engineering; analysis and study runner; dashboard publication.
- Commit plan: one commit for each atomic unit, followed by a documentation/evidence closure commit.

## Phase 1: Feature contract and coverage

### Subphase 1.1: Recover discarded official HKJC race metadata
- Commit: extend the official parser and normalization contract with race class, distance, prize, going, course, lengths behind, and running positions.
- Tests: official fixture parsing, metadata normalization, immutable raw archive reparse, canonical source precedence, and field coverage gains.
- Success Criteria: official HKJC rows retain the richer fields already present in archived source pages without changing raw files.
- Planned Touch Files:
  - `scrapper/historical/results.py`
  - `ima/historical_sources.py`
  - `scripts/reparse_official_history.py`
  - `tests/test_historical.py`
  - `tests/test_historical_sources.py`
  - `docs/data/HISTORICAL_COVERAGE.md`
- Checklist:
  - [x] Parse official race metadata and runner lengths/running positions.
  - [x] Reparse immutable official HTML and rebuild the canonical archive.
  - [x] Verify official coverage and source precedence.

### Subphase 1.2: Define the Benter and notebook feature contract
- Commit: add named baseline and rich schemas plus factor-family provenance and support status.
- Tests: exact feature uniqueness, family membership, baseline compatibility, and unsupported-factor disclosure.
- Success Criteria: every rich variable has a definition, source, availability rule, and Benter/notebook provenance.
- Planned Touch Files:
  - `ima/feature_sets.py`
  - `tests/test_rich_features.py`
- Checklist:
  - [x] Define rich numeric and categorical feature names.
  - [x] Map features to Benter factor families and notebook origins.
  - [x] Keep the baseline schema byte-for-byte compatible.

## Phase 2: Leakage-safe feature engineering

### Subphase 2.1: Build point-in-time rich features
- Commit: implement race-safe horse, jockey, trainer, pair, preference, draw-bias, trackwork, trial, and sectional histories.
- Tests: later outcomes cannot change earlier feature rows; same-race outcomes are excluded; rolling windows and as-of event joins have expected values.
- Success Criteria: canonical rows receive the rich schema without target leakage and feature coverage is measurable.
- Planned Touch Files:
  - `ima/rich_features.py`
  - `ima/modeling.py`
  - `tests/test_rich_features.py`
  - `tests/test_modeling.py`
- Checklist:
  - [x] Implement shifted rolling horse-form and normalized-time features.
  - [x] Implement jockey, trainer, draw, venue, distance, course, and going histories.
  - [x] Implement timestamp-safe auxiliary event features with availability flags.
  - [x] Make model preprocessing accept an explicit named feature schema.

## Phase 3: Model comparison and importance analysis

### Subphase 3.1: Compare models and rank variable contribution
- Commit: add a reproducible baseline-versus-rich study with statistical and permutation reports.
- Tests: deterministic rankings, corrected p-values, correlation symmetry, and positive importance for a synthetic predictive feature.
- Success Criteria: one command produces dataset coverage, model metrics, variable rankings, family rankings, and machine-readable reports.
- Planned Touch Files:
  - `ima/feature_analysis.py`
  - `scripts/run_feature_study.py`
  - `tests/test_feature_analysis.py`
- Checklist:
  - [ ] Calculate correlation and high-redundancy pairs.
  - [ ] Calculate target association, FDR-adjusted significance, and mutual information.
  - [ ] Calculate held-out feature and family permutation importance using race log loss.
  - [ ] Train baseline and rich logistic/boosted challengers on the same race splits.

## Phase 4: Dashboard publication and verification

### Subphase 4.1: Publish transparent feature-study results
- Commit: add feature coverage, model comparison, significance, correlation, and importance views to the dashboard and publish generated assets.
- Tests: dashboard contract tests, desktop/mobile Playwright interactions, chart pixels, no overflow, full suite, Vercel Ready and HTTP 200.
- Success Criteria: a dashboard user can see which variables exist, their coverage, correlation, significance, and held-out contribution without reading code.
- Planned Touch Files:
  - `docs/model-results/dashboard-template.html`
  - `tests/test_experiments.py`
  - `public/index.html`
  - `public/results.json`
  - `artifacts/feature-study/report.json`
  - `artifacts/feature-study/feature-ranking.csv`
  - `artifacts/feature-study/correlation.csv`
  - `docs/RICH_FEATURE_STUDY_PLAN.md`
- Checklist:
  - [ ] Run the full feature study on the 1997-2025 archive.
  - [ ] Render browser tables and charts with methodology notes.
  - [ ] Verify desktop and mobile through Playwright.
  - [ ] Commit, push, deploy, and verify production.