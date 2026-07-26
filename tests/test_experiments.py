import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.experiments import (
    default_experiment_specs, merge_run_history, render_dashboard, results_frame,
)
from ima.pipeline_transparency import pipeline_manifest
from scripts.publish_latest_dashboard import (
    compact_notebook_history, compact_simulator_report, publish_latest_dashboard,
)
from scripts.run_benter_grid import selected_experiment_specs
from scripts.run_feature_study import matrix_payload, publish_dashboard


class ExperimentTests(unittest.TestCase):
    def test_default_grid_is_unique_and_covers_both_models(self):
        specs = default_experiment_specs()
        self.assertGreaterEqual(len(specs), 12)
        self.assertEqual(len(specs), len({spec.run_id for spec in specs}))
        self.assertEqual({"logit", "boosted"}, {spec.kind for spec in specs})

    def test_rich_grid_can_select_one_named_experiment(self):
        specs = selected_experiment_specs(["boost-lr006-leaf15"])
        self.assertEqual(1, len(specs))
        self.assertEqual("notebook-boost-lr006-leaf15", specs[0].run_id)
        self.assertEqual("boosted", specs[0].kind)
        self.assertEqual(0.06, specs[0].parameters["learning_rate"])

    def test_dashboard_source_contract_names_market_combined_path(self):
        source = Path("ima/experiments.py").read_text(encoding="utf-8")
        self.assertIn('"prediction_sources"', source)
        self.assertIn('"combined"', source)
        self.assertIn('"supported_pools"', source)
        self.assertIn("feature_schema=feature_schema", source)
        self.assertTrue(Path("scripts/run_benter_grid.py").exists())

    def test_pipeline_manifest_matches_active_feature_contract(self):
        manifest = pipeline_manifest()
        contract = manifest["feature_contract"]
        contracts = manifest["feature_contracts"]
        self.assertEqual(23, contract["count"])
        self.assertEqual("baseline-v1", contract["name"])
        self.assertEqual(23, contracts["baseline-v1"]["count"])
        self.assertEqual(81, contracts["benter-rich-v1"]["count"])
        self.assertEqual(166, contracts["notebook-rich-v2"]["count"])
        self.assertEqual(17, len(contract["numeric"]))
        self.assertEqual(6, len(contract["categorical"]))
        self.assertEqual(8, len(manifest["training"]))
        self.assertEqual(9, len(manifest["live"]))
        self.assertIn("trackwork records", manifest["collected_not_consumed_by_baseline"])
        placeholders = {item["field"]: item["reason"] for item in manifest["historical_placeholders"]}
        self.assertIn("anonymized integers", manifest["legacy_age_provenance"]["reason"])
        self.assertIn("obscured", manifest["legacy_age_provenance"]["reason"])
        self.assertNotIn("surface", placeholders)
        self.assertNotIn("prize", placeholders)
        self.assertIn("horse profile page", placeholders["horse_age"])
        self.assertTrue(all("populated" in item["value"] for item in manifest["historical_placeholders"]))
        for stage in [*manifest["training"], *manifest["live"]]:
            self.assertEqual(
                {"id", "name", "purpose", "inputs", "operations", "outputs", "fit_scope", "leakage_boundary"},
                set(stage),
            )

    def test_results_flatten_and_dashboard_embed_runs(self):
        summary = {
            "created_at": "2026-01-01T00:00:00Z",
            "dataset": {"runners": 3, "races": 1},
            "market_test": {"top_pick_win_rate": 0.5, "race_log_loss": 1.0},
            "runs": [{
                "run_id": "demo", "kind": "logit", "parameters": {"C": 0.5},
                "duration_seconds": 1.0, "temperature": 1.0,
                "fundamental_weight": 0.2, "market_weight": 1.0,
                "incremental_pseudo_r2": 0.01,
                "second_place_exponent": 0.8, "third_place_exponent": 0.6,
                "validation": {"top_pick_win_rate": 0.3},
                "test_fundamental": {"top_pick_win_rate": 0.4},
                "test_blended": {"top_pick_win_rate": 0.5},
                "disagreement": [],
            }],
        }
        self.assertEqual("demo", results_frame(summary).iloc[0]["run_id"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.html"
            output = root / "dashboard.html"
            template.write_text("<script>const DATA=__EXPERIMENT_DATA__;</script>", encoding="utf-8")
            render_dashboard(summary, template, output)
            rendered = output.read_text(encoding="utf-8")
        self.assertIn('"run_id":"demo"', rendered)
        self.assertNotIn("__EXPERIMENT_DATA__", rendered)

    def test_run_history_retains_repeated_run_ids_from_different_executions(self):
        first = {"run_id": "demo", "feature_schema": "baseline-v1", "execution_id": "a"}
        second = {"run_id": "demo", "feature_schema": "baseline-v1", "execution_id": "b"}
        history = merge_run_history([first], [second])
        self.assertEqual(2, len(history))
        self.assertEqual(2, len({run["run_key"] for run in history}))

    def test_dashboard_template_contains_interactive_controls(self):
        template = Path("docs/model-results/dashboard-template.html").read_text(encoding="utf-8")
        for marker in (
            'id="family-filter"',
            'id="source-filter"',
            'id="schema-filter"',
            'id="metric-select"',
            'id="sort-select"',
            'id="bar-chart"',
            'id="scatter-chart"',
            'id="progress-chart"',
            'id="progress-note"',
            'id="progress-summary"',
            'id="show-all-runs"',
            'id="schema-contracts"',
            'id="pool-chart"',
            'id="pool-results-body"',
            'id="prediction-pipeline"',
            'id="pipeline-mode"',
            'id="pipeline-stage-nav"',
            'id="pipeline-stage-detail"',
            'id="numeric-features"',
            'id="unused-data"',
            'id="feature-study"',
            'id="feature-importance-chart"',
            'id="feature-ranking-body"',
            'id="correlation-run-select"',
            'id="correlation-feature-select"',
            'id="correlation-matrix-chart"',
            'id="correlation-variable-body"',
            'id="family-ranking-body"',
            'id="redundancy-body"',
            'id="benter-coverage-body"',
            'id="model-comparison-body"',
            'id="auxiliary-results"',
            'id="notebook-full-history"',
            'id="auxiliary-chart"',
            'id="auxiliary-results-body"',
            'id="market-blend-weights"',
            'id="simulator-results"',
            'id="simulator-warning"',
            'id="simulator-recommendations-body"',
            'id="results-body"',
            'href="results.csv"',
            'href="results.json"',
            "__EXPERIMENT_DATA__",
            "Plotly.react",
            "Execution time (points within each batch follow sweep order)",
            "removeAllListeners",
            "run_history",
        ):
            self.assertIn(marker, template)

    def test_feature_study_publication_merges_results_and_renders_dashboard(self):
        summary = {"dataset": {"races": 1}, "runs": [{"run_id": "logit-c005-balanced"}]}
        report = {"selected_rich_model": "benter-rich-v1-boosted", "feature_ranking": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.json"
            template = root / "template.html"
            output = root / "index.html"
            results.write_text(json.dumps(summary), encoding="utf-8")
            template.write_text("<script>const DATA=__EXPERIMENT_DATA__;</script>", encoding="utf-8")
            publish_dashboard(report, results, template, output)
            published = json.loads(results.read_text(encoding="utf-8"))
            rendered = output.read_text(encoding="utf-8")
        self.assertEqual(report, published["feature_study"])
        self.assertEqual("baseline-v1", published["runs"][0]["feature_schema"])
        self.assertEqual(23, published["runs"][0]["feature_count"])
        self.assertEqual(23, len(published["runs"][0]["variables"]))
        self.assertIn('"selected_rich_model":"benter-rich-v1-boosted"', rendered)

    def test_latest_dashboard_publisher_adds_compact_current_artifacts(self):
        summary = {"dataset": {"races": 1}, "runs": []}
        auxiliary = {"dataset": {"features": 166}, "targets": {"position": {}}}
        winner = {"dataset": {"features": 166}, "blend": {"fundamental_weight": 0.1}}
        simulator = {
            "requested_date": "2026-07-27", "displayed_date": "2026-07-26",
            "date_status": "next_available_allowed", "meeting": {"total_cost": 10.0},
            "races": [{
                "race_no": 1,
                "prediction_basis": {"basis": "public_win_market_fallback"},
                "summary": {"recommended_bets": 1},
                "recommendations": [{"pool": "PLACE", "combination": ["2"]}],
                "priced_candidates": [{"large": "payload"}],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in (
                "results.json", "template.html", "index.html", "results.csv",
                "auxiliary.json", "winner.json", "simulator.json",
            )}
            paths["results.json"].write_text(json.dumps(summary), encoding="utf-8")
            paths["template.html"].write_text("<script>const DATA=__EXPERIMENT_DATA__;</script>", encoding="utf-8")
            paths["auxiliary.json"].write_text(json.dumps(auxiliary), encoding="utf-8")
            paths["winner.json"].write_text(json.dumps(winner), encoding="utf-8")
            paths["simulator.json"].write_text(json.dumps(simulator), encoding="utf-8")
            published = publish_latest_dashboard(
                paths["results.json"], paths["template.html"], paths["index.html"],
                paths["results.csv"], paths["auxiliary.json"], paths["winner.json"],
                paths["simulator.json"],
            )
        self.assertEqual(166, published["pipeline_manifest"]["feature_contracts"]["notebook-rich-v2"]["count"])
        self.assertEqual(auxiliary, published["auxiliary_predictions"])
        self.assertEqual(winner, published["notebook_rich_benchmark"])
        self.assertEqual("public_win_market_fallback", published["simulator"]["race"]["prediction_basis"]["basis"])
        self.assertNotIn("priced_candidates", published["simulator"]["race"])

    def test_compact_simulator_handles_no_available_race(self):
        compact = compact_simulator_report({"requested_date": "2026-07-27", "races": []})
        self.assertIsNone(compact["race"])

    def test_compact_notebook_history_keeps_dataset_and_latest_run(self):
        report = {
            "created_at": "2026-07-26T09:22:39+00:00",
            "dataset": {"runners": 271858, "races": 22144},
            "runs": [{"run_id": "notebook-demo", "feature_schema": "notebook-rich-v2"}],
        }
        compact = compact_notebook_history(report)
        self.assertEqual(271858, compact["dataset"]["runners"])
        self.assertEqual("notebook-demo", compact["run"]["run_id"])

    def test_correlation_matrix_payload_is_json_safe_and_labeled(self):
        matrix = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], columns=["a", "b"], index=["a", "b"])
        payload = matrix_payload(matrix)
        self.assertEqual(["a", "b"], payload["features"])
        self.assertIsNone(payload["values"][0][1])


if __name__ == "__main__":
    unittest.main()
