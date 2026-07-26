# Run-Centric Dashboard Redesign

## GOAL
- Replace the cluttered mixed-chart dashboard with a truthful run-centric analysis workspace while preserving pipeline transparency and variable contribution analysis.

## Acceptance Criteria
- [ ] The default view is `All runs`, with one synchronized run table and comparison chart.
- [ ] A user can select one run from the table or a chart point and every run-specific panel updates from that same run identity.
- [ ] Comparison scope supports individual runs and aggregates by feature schema, model family, and execution batch.
- [ ] Fundamental, market, and combined results are visually and textually distinct; the market benchmark is shown once, not duplicated as if it were one result per run.
- [ ] The latest-notebook-only panel is removed. Notebook runs participate in the same selection and comparison workflow as all other runs.
- [ ] Data pipeline and live prediction pipeline remain available as a dedicated top-level view.
- [ ] Variable contribution, full ranking, family contribution, coverage, and correlation remain available as a dedicated top-level view.
- [ ] Dataset-level feature studies are labeled as dataset/schema analyses and never presented as run-specific calculations unless a run-specific artifact exists.
- [ ] Every chart uses Plotly.js. Native HTML remains acceptable for tables, controls, disclosures, and pipeline text.
- [ ] Every chart has a title, axis units, source/scope note, stable color semantics, exact hover values, and a legend that corresponds to rendered marks.
- [ ] The default desktop view is no taller than 6,000 px at 1440x1000; the default mobile view is no taller than 10,000 px at 390x844.
- [ ] No horizontal overflow occurs at 1440x1000, 1024x768, 768x1024, or 390x844.
- [ ] Existing 34-run history, three feature schemas, eight pool metrics, pipeline stages, simulator disclosure, and feature-study records remain reachable.
- [ ] Production Playwright checks cover selection synchronization, aggregate modes, chart legends, chart data, responsive layout, and console errors.

## Out Of Scope
- Retraining models or changing historical datasets.
- Changing model formulas, calibration, pool probability calculations, or wagering rules.
- Adding authenticated wagering.
- Replacing the static Vercel deployment architecture unless implementation proves the current single-file approach unmaintainable.

## Research
- Built-in option inspected: retain the current static Vercel deployment and vanilla browser runtime, but separate data contract, state, views, and chart modules.
- Off-the-shelf option selected: Plotly.js for every analytical chart because it is already used by the dashboard and supports scatter, bar, heatmap, hover, legends, selection, and responsive SVG rendering.
- Rejected option: React/Next.js migration. The dashboard is a read-only static analytical artifact; adding application rendering, package/build infrastructure, and framework state does not solve the core truthfulness problem.
- Rejected option: keep custom canvas charts. They lack consistent hover/legend behavior and duplicate layout, scale, resize, and interaction code already supplied by Plotly.
- Architecture evidence: refreshed Megatect analysis identified five repo systems, one dependency cycle, `ima/experiments.py` as a high-coupling hub, and moderate overall risk. This supports extracting a narrow dashboard publication contract without distributing the system.
- Pattern fit: modular monolith and clean boundaries fit; microservices should be avoided.
- Render evidence: existing Playwright screenshots at 1440x1000 and 390x844 plus source inspection of `public/index.html`, `scripts/publish_latest_dashboard.py`, `ima/experiments.py`, and `public/results.json`.

## Audit Evidence
- The rendered dashboard is 17,378 px tall on desktop and 26,028 px on mobile.
- The page exposes 26 sections, 12 tables, 10 second-level headings, 16 third-level headings, five selects, four custom canvases, and three Plotly charts.
- `renderNotebookRichResults()` reads `DATA.notebook_full_history.run` directly and therefore cannot follow the selected run.
- Dashboard state is split across `selected`, `source`, `family`, `schema`, `contractSchema`, `studySchema`, `correlationRun`, and `correlationFeature`; related panels can describe different scopes simultaneously.
- `duration_seconds` is offered as a chart metric, but `drawBarChart()` reads it from `sourceMetrics(run)`, where it does not exist.
- Market mode repeats the same global market benchmark for every run, visually implying run-level market observations that are not present.
- The chronological Plotly chart connects points sharing schema/family across separate sweep batches, creating a false continuous progression.
- The Spearman control is labeled by model run even though runs sharing a schema use one shared training-data matrix.
- Feature contribution is schema-level permutation analysis from a selected study model, not a fresh ranking for every run.
- Custom canvas charts lack Plotly-equivalent hover values and use color semantics differently from the Plotly charts.

## Proposed Information Architecture

### Global Shell
- Top-level views: `Runs`, `Pipeline`, `Variables`, `Live simulation`.
- Persist navigable state in URL query parameters: `view`, `run`, `scope`, `source`, `metric`, `schema`, `family`, and `execution`.
- Use one global selected run. No panel may maintain a competing run selection.

### Runs View
- Default scope: `All runs`.
- Primary controls: comparison scope, probability source, metric, feature schema, model family, execution batch, and search.
- Primary chart: run comparison scatter. Default axes are race log loss and Top-1 accuracy; model family uses marker shape, schema uses a restrained color set, and the selected run receives an outline.
- Secondary chart: execution history using actual execution timestamps. Do not connect separate batches. Optional lines may connect only points inside one explicit sweep batch ordered by `sequence`.
- Primary table: compact run index with execution, run name, schema, family, selected metric, Top-1, Top-3, log loss, pseudo-R2, and selection action. Advanced metrics remain in run detail.
- Aggregate modes: `By schema`, `By model family`, and `By execution batch`. Show count, median, interquartile range, best run, and evaluation-window compatibility. Do not silently average runs evaluated on incompatible test windows.

### Selected Run Workspace
- Tabs: `Overview`, `Evaluation`, `Pools`, `Variables`, `Parameters`.
- Overview: identity, data/evaluation window, model family, schema, hyperparameters, and a three-column fundamental/market/combined score comparison.
- Evaluation: Top-1, Top-3, mean winner rank, log loss, Brier, ECE, pseudo-R2, incremental pseudo-R2, calibration parameters, and disagreement analysis.
- Pools: predicted versus observed scatter with a 45-degree reference line, exact hit counts, and pool definitions. Avoid mixed bars and lines for the same values.
- Variables: exact estimator inputs for the run plus a link to the matching schema-level feature study. Clearly state when contribution values come from another representative model.
- Parameters: hyperparameters, temperature, blend weights, place exponents, duration, artifact identity, and raw JSON disclosure.

### Pipeline View
- Preserve the historical training and live prediction segmented control.
- Keep one step navigator and one detail panel with inputs, exact transformation, outputs, fitted state, and leakage boundary.
- Move feature contracts and historical coverage into collapsible supporting panels rather than rendering all fields on initial load.

### Variables View
- Dataset selector: Baseline, Benter rich, Notebook rich.
- Summary: declared variables, numeric/categorical split, zero-coverage count, study model, dataset rows/races, and methodology.
- Plotly horizontal bar: permutation delta log loss with zero reference and confidence/error information when available.
- Searchable/sortable ranking table, initially limited to the top 20 with explicit `Show all`.
- Plotly heatmap for the schema-level Spearman matrix. Label it by dataset/schema, not by run.
- Feature-family contribution and redundancy become tabs or compact secondary tables.
- Coverage and source provenance remain available without dominating the main analytical flow.

### Visual Encoding Contract
- Probability source colors: fundamental `#18794e`, market `#66706b`, combined `#1d4ed8`.
- Model family is encoded by marker shape, not by reusing probability-source colors.
- Feature schema may use color only in all-run comparisons; selected-run metric charts use source colors.
- Positive/negative contribution uses a diverging scale with an explicit zero line.
- Legends are generated from the same trace definitions as the plotted marks.
- Metric labels always include direction: `higher is better` or `lower is better`.

## Data Contract Changes
- Add a normalized published `dashboard.runs[]` record for every persisted run; remove special latest-run rendering paths.
- Add `evaluation_window`, `dataset_id`, and `dataset_hash` so aggregate comparability can be checked.
- Add `feature_study_id` and `feature_study_scope` to link a run to schema-level or run-level feature analysis without implying false specificity.
- Publish market benchmark as one benchmark object per evaluation dataset, not repeated through run rendering.
- Preserve existing `run_history`, `feature_study`, `pipeline_manifest`, `simulator`, and pool metrics during migration; add a compatibility adapter before deleting old fields.

## Target Architecture
- Architecture style: static modular monolith with a clean publication boundary. Do not add microservices, server rendering, or a frontend framework for this redesign.
- Python ownership: experiment execution, metric calculation, feature studies, aggregate compatibility checks, and publication of a versioned dashboard view model.
- Browser ownership: URL-backed selection state, filtering, view composition, table interaction, and Plotly rendering. Browser code must not infer missing model semantics or recompute scientific metrics.
- Proposed Python module: `ima/dashboard_contract.py`, containing typed normalization and validation for runs, benchmarks, feature-study references, evaluation windows, and aggregate groups.
- Proposed static modules: `public/assets/dashboard-state.js`, `dashboard-data.js`, `dashboard-runs.js`, `dashboard-variables.js`, `dashboard-pipeline.js`, `dashboard-live.js`, `dashboard-charts.js`, and `dashboard.css`.
- `public/index.html` becomes a stable shell and fetches `public/results.json`; it no longer embeds the complete result payload or generated application logic.
- `scripts/publish_latest_dashboard.py` becomes an orchestration adapter: load artifacts, build the versioned view model, validate it, write JSON/CSV, and leave the static shell/modules unchanged.
- Dependency direction: scraper/training/analysis -> dashboard contract -> static JSON -> browser views. Browser modules never import or own training policy.
- Decision record: `docs/architecture/decisions/2026-07-27-use-a-static-modular-run-analysis-dashboard.md`.

## Regression Guardrails
- Planned edit surface: `public/index.html`, `public/assets/*`, `docs/model-results/dashboard-template.html` during compatibility migration, `scripts/publish_latest_dashboard.py`, `ima/dashboard_contract.py`, `ima/experiments.py`, `tests/test_dashboard_contract.py`, `tests/test_experiments.py`, and tracked dashboard QA tests/configuration.
- Protected behaviors: static Vercel deployment, all 34 historical runs, source/schema/family filtering, eight pool metrics, pipeline disclosure, feature rankings, Spearman data, paper-simulation warning, and CSV/JSON downloads.
- Likely consumers: dashboard users, Vercel static deployment, experiment publication scripts, retraining/finalization jobs, and tests reading published result contracts.
- Damage radius: moderate. Presentation is broad, but model training and data ingestion remain untouched.
- Proof plan: contract tests, deterministic chart-trace tests, Playwright interaction tests, desktop/mobile screenshots, canvas-pixel checks replaced with Plotly SVG checks, console-error checks, and production HTTP/deployment smoke tests.
- Atomic change units: data adapter; shell/navigation; runs workspace; selected-run detail; variables workspace; pipeline workspace; Plotly migration; responsive cleanup; release evidence.

## Phase 1: Audit And Contracts

### Subphase 1.1: Freeze current behavior and chart semantics
- Commit: `docs(dashboard): record run-centric redesign audit`
- Tests: `python -m unittest tests.test_experiments`; existing Playwright QA against current production.
- Success Criteria: every current panel is classified as keep, merge, move, relabel, or remove; every chart has a documented data source and semantic defect list.
- Planned Touch Files:
  - `docs/RUN_CENTRIC_DASHBOARD_REDESIGN_PLAN.md`
- Checklist:
  - [x] Record structural and rendered-size evidence.
  - [x] Identify mixed-framework and false-scope problems.
  - [ ] Capture a panel disposition matrix before implementation.

### Subphase 1.2: Normalize dashboard publication contract
- Commit: `refactor(dashboard): publish normalized run analysis records`
- Tests: `python -m unittest tests.test_experiments tests.test_feature_analysis`.
- Success Criteria: every run uses one published shape; evaluation-window compatibility and feature-study scope are explicit; old fields remain supported by an adapter.
- Planned Touch Files:
  - `ima/dashboard_contract.py`
  - `scripts/publish_latest_dashboard.py`
  - `ima/experiments.py`
  - `tests/test_dashboard_contract.py`
  - `tests/test_experiments.py`
- Checklist:
  - [ ] Define normalized run, benchmark, aggregate, and feature-study references.
  - [ ] Add compatibility adapter and fixture tests.
  - [ ] Verify 34 run identities remain unique and present.

## Phase 2: Run-Centric Information Architecture

### Subphase 2.1: Build the global shell and single state model
- Commit: `feat(dashboard): add run-centric navigation and state`
- Tests: template structure tests and URL-state Playwright checks.
- Success Criteria: the dashboard opens in `Runs / All runs`; all controls derive from one state object; view/run state is deep-linkable.
- Planned Touch Files:
  - `public/index.html`
  - `public/assets/dashboard-state.js`
  - `public/assets/dashboard-data.js`
  - `public/assets/dashboard.css`
  - `docs/model-results/dashboard-template.html`
  - `tests/test_experiments.py`
- Checklist:
  - [ ] Add Runs, Pipeline, Variables, and Live simulation views.
  - [ ] Replace competing selected-run states with one run identity.
  - [ ] Synchronize URL, controls, chart selection, and table selection.

### Subphase 2.2: Build all-run comparison and aggregate modes
- Commit: `feat(dashboard): add run comparison and category aggregates`
- Tests: deterministic aggregate unit tests and Playwright filter/selection tests.
- Success Criteria: all 34 runs are visible by default; aggregate modes show count, median, IQR, and best run; incompatible evaluation windows are disclosed.
- Planned Touch Files:
  - `public/index.html`
  - `public/assets/dashboard-runs.js`
  - `public/assets/dashboard-charts.js`
  - `public/assets/dashboard.css`
  - `scripts/publish_latest_dashboard.py`
  - `ima/dashboard_contract.py`
  - `tests/test_experiments.py`
- Checklist:
  - [ ] Implement all-runs scatter, compact table, and execution chart.
  - [ ] Implement schema, family, and execution aggregate modes.
  - [ ] Remove misleading market-per-run duplication and cross-batch lines.

### Subphase 2.3: Build selected-run workspace
- Commit: `feat(dashboard): unify selected run analysis`
- Tests: Playwright selection from table and chart; source-switch consistency checks.
- Success Criteria: selecting any run updates Overview, Evaluation, Pools, Variables, and Parameters; no latest-only model panel remains.
- Planned Touch Files:
  - `public/index.html`
  - `public/assets/dashboard-runs.js`
  - `public/assets/dashboard-charts.js`
  - `public/assets/dashboard.css`
  - `tests/test_experiments.py`
- Checklist:
  - [ ] Move notebook runs into the common selected-run path.
  - [ ] Add fundamental/market/combined comparison with stable semantics.
  - [ ] Keep exact variables and parameters behind focused tabs/disclosures.

## Phase 3: Plotly Migration

### Subphase 3.1: Replace all custom charts with Plotly
- Commit: `refactor(charts): standardize dashboard visualizations on Plotly`
- Tests: trace-data assertions, SVG visibility checks, hover-label checks, and zero-console-error Playwright runs.
- Success Criteria: no dashboard `<canvas>` chart remains; all plots use shared layout/config helpers and the visual encoding contract.
- Planned Touch Files:
  - `public/index.html`
  - `public/assets/dashboard-charts.js`
  - `public/assets/dashboard-runs.js`
  - `public/assets/dashboard-variables.js`
  - `public/assets/dashboard.css`
  - `tests/test_experiments.py`
- Checklist:
  - [ ] Replace run bars and scatter.
  - [ ] Replace feature contribution and Spearman matrix.
  - [ ] Replace pool mixed chart with predicted-versus-observed plot.
  - [ ] Standardize legends, hover templates, axes, margins, and responsive behavior.

### Subphase 3.2: Recompose Pipeline and Variables views
- Commit: `feat(dashboard): focus pipeline and feature analysis views`
- Tests: schema switching, pipeline mode/stage navigation, ranking expansion, and heatmap interaction tests.
- Success Criteria: pipeline detail remains complete without rendering every field at once; variable analysis clearly states dataset and model scope.
- Planned Touch Files:
  - `public/index.html`
  - `public/assets/dashboard-pipeline.js`
  - `public/assets/dashboard-variables.js`
  - `public/assets/dashboard-live.js`
  - `public/assets/dashboard-charts.js`
  - `public/assets/dashboard.css`
  - `tests/test_experiments.py`
- Checklist:
  - [ ] Preserve exact transformation and leakage-boundary content.
  - [ ] Add top-20 ranking default and explicit full-table expansion.
  - [ ] Relabel shared correlations as schema-level matrices.
  - [ ] Keep contribution, family, redundancy, coverage, and provenance reachable.

## Phase 4: Verification And Release

### Subphase 4.1: Responsive and semantic QA
- Commit: `test(dashboard): verify run-centric analysis workflows`
- Tests: Python suite plus Playwright at 1440x1000, 1024x768, 768x1024, and 390x844.
- Success Criteria: no overflow, no chart/legend mismatch, no stale panel after run changes, no console errors, and default-page height targets are met.
- Planned Touch Files:
  - `tests/test_experiments.py`
  - `tests/dashboard/dashboard-qa.spec.js`
  - `tests/dashboard/playwright.config.js`
- Checklist:
  - [ ] Assert every plotted trace against the selected source/scope data.
  - [ ] Verify keyboard and mobile selection workflows.
  - [ ] Capture before/after desktop and mobile screenshots.
  - [ ] Run full Python regression suite.

### Subphase 4.2: Publish and verify preview
- Commit: `chore(dashboard): publish run-centric dashboard artifacts`
- Tests: Vercel preview HTTP check and production-like Playwright smoke test.
- Success Criteria: branch preview is reachable, screenshots match the accepted information architecture, and production remains unchanged until explicit approval to merge.
- Planned Touch Files:
  - `public/index.html`
  - `public/results.json`
  - `public/results.csv`
- Checklist:
  - [ ] Publish regenerated static artifacts.
  - [ ] Deploy branch preview.
  - [ ] Verify preview desktop/mobile behavior.
  - [ ] Present preview for approval before merge.
