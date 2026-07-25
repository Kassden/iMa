# 1997-2025 Model Experiment Dashboard

## GOAL
- Run parameterized race-aware models on unified 1997-2025 data and publish interactive accuracy graphs and per-run metrics.

## Acceptance Criteria
- [x] Unified dataset spans 1997-06-02 through 2025-12-27 without duplicated 2005 races.
- [x] At least 12 parameterized logit/boosted runs use one fixed chronological split.
- [x] Every run records parameters, Top-1, Top-3, mean winner rank, log loss, Brier, ECE, pseudo-R2, blend weights, and incremental pseudo-R2.
- [x] Results are emitted as CSV, JSON, PNG graphs, and a self-contained interactive HTML dashboard.
- [x] Dashboard filters, metric selection, sorting, table selection, and responsive layouts are browser-verified.
- [x] Full Python tests and artifact integrity checks pass.

## Research
- Built-in options: reuse `ima.data`, `ima.modeling`, calibration, market blend, and chronological race splitting.
- Off-the-shelf options: matplotlib for exported PNGs; self-contained HTML/CSS/JavaScript for zero-install browser interaction.
- Official / standards sources: no external model API; preserve documented final-odds versus timestamped-odds semantics.

## Regression Guardrails
- Planned edit surface: `ima/data.py`, `ima/modeling.py`, `ima/experiments.py`, `scripts/run_experiments.py`, `tests/test_data.py`, `tests/test_experiments.py`, `README.md`, `docs/model-results/**`, generated ignored artifacts.
- Protected behaviors: legacy and canonical single-run training, live scraping, paper trading, model artifact loading, chronological leakage controls, and market timestamp warnings.
- Likely consumers: researchers running offline training, model promotion jobs, and operators reviewing experiment results.
- Damage radius: moderate.
- Proof plan: focused dataset/experiment tests, fixed split assertions, full unit suite, PNG existence, browser interactions, desktop/mobile screenshots, and canvas pixel checks.

## Phase 1: Dataset unification

### Subphase 1.1: Join legacy and canonical timelines
- Commit: add a non-overlapping 1997-2025 point-in-time dataset builder.
- Tests: date bounds, race uniqueness, horse namespace isolation, market normalization, and chronological split boundaries.
- Success Criteria: one frame spans 1997-2025, uses legacy races only before 2005, and validates all race invariants.
- Planned Touch Files:
  - `ima/data.py`
  - `tests/test_data.py`
- Checklist:
  - [x] Prefix legacy race/horse IDs and align feature columns.
  - [x] Concatenate legacy pre-2005 rows with canonical 2005-2025 rows.
  - [x] Validate normalized probabilities and fixed chronological split.

## Phase 2: Experiment runner

### Subphase 2.1: Parameter-grid evaluation
- Commit: add reproducible experiment specifications, execution, ranking, and artifact export.
- Tests: parameter application, deterministic run IDs, metric completeness, stable split reuse, and CSV/JSON output.
- Success Criteria: at least 12 completed runs compare logit and boosted parameters on identical held-out races.
- Planned Touch Files:
  - `ima/modeling.py`
  - `ima/experiments.py`
  - `scripts/run_experiments.py`
  - `tests/test_experiments.py`
- Checklist:
- [x] Make estimator parameters explicit without changing defaults.
- [x] Evaluate calibrated fundamentals and market blends for every run.
- [x] Export ranked machine-readable results and model artifacts.

## Phase 3: Interactive dashboard

### Subphase 3.1: Accuracy visualization surface
- Commit: generate static accuracy graphs and a self-contained interactive dashboard.
- Tests: expected files, embedded run count, required controls, non-empty charts, and responsive layout.
- Success Criteria: browser users can filter model families, switch metrics, sort/select runs, and inspect exact parameters and scores.
- Planned Touch Files:
  - `ima/experiments.py`
  - `docs/model-results/dashboard-template.html`
  - `README.md`
- Checklist:
- [x] Export Top-1/Top-3 and probability-quality PNG graphs.
- [x] Build interactive canvas charts and sortable run table.
- [x] Add CSV/JSON download controls and exact run detail view.

## Phase 4: Verification

### Subphase 4.1: End-to-end proof
- Commit: verify experiments, browser rendering, and regression safety.
- Tests: full unittest suite, compile, diff check, browser desktop/mobile interactions, screenshots, and canvas pixel checks.
- Success Criteria: all acceptance criteria have recorded evidence and no required process remains running.
- Planned Touch Files:
  - `tests/test_experiments.py`
  - `.mega/evidence.jsonl`
- Checklist:
- [x] Run the complete experiment grid.
- [x] Verify dashboard on desktop and mobile.
- [x] Record results and final evidence.
