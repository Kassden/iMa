# Technical Pipeline Transparency Dashboard

## GOAL
- Replace the ambiguous pipeline diagram with a simple, technically complete, code-backed representation of historical training and live prediction transformations.

## Acceptance Criteria
- [x] Training and live inference are shown as separate ordered pipelines.
- [x] Every stage exposes exact inputs, operation, outputs, fit scope, and leakage boundary.
- [x] The dashboard lists all 23 current model features and distinguishes collected-but-unused data.
- [x] Stubbed or unavailable historical fields are disclosed explicitly.
- [x] Market probability conversion, calibration, blend formula, fallback, pool expansion, and Kelly logic are represented exactly.
- [x] Desktop and mobile Playwright QA verifies stage navigation, readable details, no overlap, and no horizontal page overflow.
- [ ] Full tests, commit, push, and Vercel production verification pass.

## Out Of Scope
- Changing the actual estimator feature set or retraining models.
- Claiming collected veterinary or trackwork data affects the current baseline when it does not.
- Changing wager execution limits or authenticated submission.

## Research
- Source of truth: `ima/data.py`, `ima/modeling.py`, `ima/inference.py`, `ima/pools.py`, `ima/strategy.py`, `ima/training.py`, `scrapper/contracts.py`, and `scrapper/pipeline.py`.
- Selected pattern: a code-generated pipeline manifest plus an interactive stage explorer, rather than manually duplicated prose in HTML.
- Visual structure: one short numbered overview, separate Training and Live tabs, and one detailed technical panel for the selected stage.

## Regression Guardrails
- Planned edit surface: `ima/pipeline_transparency.py`, `ima/experiments.py`, `tests/test_experiments.py`, `docs/model-results/dashboard-template.html`, `docs/PIPELINE_TRANSPARENCY_PLAN.md`, `public/index.html`, `public/results.json`.
- Protected behaviors: model features, preprocessing, calibration, market blend, fallback, pool formulas, tests, result metrics, and dashboard controls.
- Likely consumers: researchers, model reviewers, paper-trading operators, and dashboard users.
- Damage radius: small for modeling behavior, moderate for dashboard presentation.
- Proof plan: manifest contract tests, full Python suite, Playwright desktop/mobile interaction checks, screenshots, production HTTP/JSON verification.

## Phase 1: Code-backed manifest

### Subphase 1.1: Encode exact training and live transformations
- Commit: add a pipeline manifest generated from current feature constants and implementation contracts.
- Tests: exact feature counts, stage ordering, collected-versus-used disclosure, and required technical fields.
- Success Criteria: dashboard data cannot silently drift from the active 23-feature contract.
- Planned Touch Files:
  - `ima/pipeline_transparency.py`
  - `ima/experiments.py`
  - `tests/test_experiments.py`
- Checklist:
  - [x] Define training stages and live stages separately.
  - [x] Export exact numeric and categorical feature names.
  - [x] Export known stubbed fields and collected-but-unused groups.

## Phase 2: Interactive representation

### Subphase 2.1: Replace the current pipeline layout
- Commit: build a numbered overview and selectable technical stage explorer.
- Tests: tab controls, stage controls, feature table, formula rendering, and leakage labels.
- Success Criteria: each selected stage answers what enters, what happens, what exits, what is fitted, and what may leak.
- Planned Touch Files:
  - `docs/model-results/dashboard-template.html`
  - `tests/test_experiments.py`
- Checklist:
  - [x] Separate Historical training and Live prediction views.
  - [x] Add exact stage detail panel and transformation tables.
  - [x] Show data actually used versus merely collected.

## Phase 3: Verification and deployment

### Subphase 3.1: Generate, inspect, and publish
- Commit: publish regenerated dashboard assets and completed evidence.
- Tests: full suite, compile, diff, Playwright desktop/mobile, production Vercel checks.
- Success Criteria: production dashboard is readable, interactive, and technically consistent with code.
- Planned Touch Files:
  - `public/index.html`
  - `public/results.json`
  - `docs/PIPELINE_TRANSPARENCY_PLAN.md`
  - `.mega/evidence.jsonl`
- Checklist:
  - [x] Regenerate static dashboard from existing experiment results.
  - [x] Verify through Playwright at desktop and mobile sizes.
  - [ ] Commit, push, deploy, and verify production.
