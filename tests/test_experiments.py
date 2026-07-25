import json
import tempfile
import unittest
from pathlib import Path

from ima.experiments import default_experiment_specs, render_dashboard, results_frame
from ima.pipeline_transparency import pipeline_manifest
from scripts.run_feature_study import publish_dashboard


class ExperimentTests(unittest.TestCase):
    def test_default_grid_is_unique_and_covers_both_models(self):
        specs = default_experiment_specs()
        self.assertGreaterEqual(len(specs), 12)
        self.assertEqual(len(specs), len({spec.run_id for spec in specs}))
        self.assertEqual({"logit", "boosted"}, {spec.kind for spec in specs})

    def test_dashboard_source_contract_names_market_combined_path(self):
        source = Path("ima/experiments.py").read_text(encoding="utf-8")
        self.assertIn('"prediction_sources"', source)
        self.assertIn('"combined"', source)
        self.assertIn('"supported_pools"', source)

    def test_pipeline_manifest_matches_active_feature_contract(self):
        manifest = pipeline_manifest()
        contract = manifest["feature_contract"]
        self.assertEqual(23, contract["count"])
        self.assertEqual(17, len(contract["numeric"]))
        self.assertEqual(6, len(contract["categorical"]))
        self.assertEqual(8, len(manifest["training"]))
        self.assertEqual(9, len(manifest["live"]))
        self.assertIn("trackwork records", manifest["collected_not_consumed_by_baseline"])
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

    def test_dashboard_template_contains_interactive_controls(self):
        template = Path("docs/model-results/dashboard-template.html").read_text(encoding="utf-8")
        for marker in (
            'id="family-filter"',
            'id="source-filter"',
            'id="metric-select"',
            'id="sort-select"',
            'id="bar-chart"',
            'id="scatter-chart"',
            'id="progress-chart"',
            'id="prediction-pipeline"',
            'id="pipeline-mode"',
            'id="pipeline-stage-nav"',
            'id="pipeline-stage-detail"',
            'id="numeric-features"',
            'id="unused-data"',
            'id="feature-study"',
            'id="feature-importance-chart"',
            'id="feature-ranking-body"',
            'id="family-ranking-body"',
            'id="redundancy-body"',
            'id="benter-coverage-body"',
            'id="model-comparison-body"',
            'id="results-body"',
            'href="results.csv"',
            'href="results.json"',
            "__EXPERIMENT_DATA__",
        ):
            self.assertIn(marker, template)

    def test_feature_study_publication_merges_results_and_renders_dashboard(self):
        summary = {"dataset": {"races": 1}, "runs": []}
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
        self.assertIn('"selected_rich_model":"benter-rich-v1-boosted"', rendered)


if __name__ == "__main__":
    unittest.main()