import unittest
import tempfile

from pathlib import Path
from unittest import mock

from ima.mlflow_tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    MLflowConfig,
    _model_artifact_source,
    flatten_numeric_metrics,
    log_research_package,
    log_research_package_version,
    research_run_metrics,
    research_run_parameters,
    run_parameters,
)
from ima.research_model_package import ResearchModelPackage
from ima.research_specs import PipelineRecipe
import numpy as np
import pandas as pd


class DummyProbabilityModel:
    def predict_proba(self, frame):
        return np.full(len(frame), 0.5)


class MLflowTrackingTests(unittest.TestCase):
    def test_config_is_disabled_without_uri(self):
        config = MLflowConfig.from_values(tracking_uri=None, experiment_name="demo")
        self.assertFalse(config.enabled)
        self.assertEqual("demo", config.experiment_name)
        self.assertTrue(config.register_models)
        self.assertEqual(DEFAULT_REGISTERED_MODEL_NAME, config.registered_model_name)

    def test_config_can_disable_model_registration_from_env(self):
        with mock.patch.dict("os.environ", {"IMA_MLFLOW_REGISTER_MODELS": "false"}):
            config = MLflowConfig.from_values(tracking_uri="http://mlflow")
        self.assertTrue(config.enabled)
        self.assertFalse(config.register_models)

    def test_model_artifact_source_points_to_logged_joblib(self):
        self.assertEqual(
            "file:///tmp/mlruns/1/abc/artifacts/models/demo.joblib",
            _model_artifact_source(
                "file:///tmp/mlruns/1/abc/artifacts/",
                Path("demo.joblib"),
            ),
        )

    def test_flattens_nested_numeric_metrics(self):
        metrics = flatten_numeric_metrics({
            "test": {"top_pick_win_rate": 0.25, "skip": "text"},
            "pool": {"WIN": {"hit_rate": 0.31, "hits": 12}},
            "flag": True,
        })
        self.assertEqual(0.25, metrics["test.top_pick_win_rate"])
        self.assertEqual(0.31, metrics["pool.WIN.hit_rate"])
        self.assertEqual(12.0, metrics["pool.WIN.hits"])
        self.assertNotIn("flag", metrics)

    def test_run_parameters_include_schema_and_model_params(self):
        params = run_parameters({
            "run_id": "demo",
            "kind": "boosted",
            "feature_schema": "benter-rich-v1",
            "feature_count": 81,
            "parameters": {"learning_rate": 0.03, "class_weight": None},
        })
        self.assertEqual("demo", params["run_id"])
        self.assertEqual("benter-rich-v1", params["feature_schema"])
        self.assertEqual(0.03, params["param.learning_rate"])
        self.assertEqual("None", params["param.class_weight"])

    def test_research_package_logging_is_noop_when_disabled(self):
        self.assertIsNone(log_research_package(Path("."), MLflowConfig(enabled=False)))

    def test_research_run_data_exposes_recipe_and_fold_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            package = ResearchModelPackage(
                DummyProbabilityModel(),
                PipelineRecipe(
                    feature_schema="benter-rich-v1",
                    model={"kind": "logit", "parameters": {"C": 0.25}},
                ),
                protocol_id="protocol-1",
                code_revision="abc123",
            )
            package_dir = package.save(Path(directory) / "package")
            result = {
                "recipe_hash": package.recipe.recipe_hash(),
                "target_kind": "win_probability",
                "objective_name": "development_race_log_loss",
                "objective_value": 2.01,
                "duration_seconds": 12.5,
                "metrics": {
                    "mean_selected_minus_market": -0.002,
                    "dataset_exclusions": {"excluded_rows": 4, "policy": "test"},
                    "folds": [{
                        "fold_id": "fold-001",
                        "selected": {"race_log_loss": 2.01},
                        "raw_market": {"race_log_loss": 2.02},
                    }],
                    "effective_training": [{
                        "fold_id": "fold-001",
                        "rows": 100,
                        "races": 10,
                        "features": ["a", "b"],
                        "available_features": ["a"],
                        "unavailable_features": ["b"],
                    }],
                },
            }
            params = research_run_parameters(
                package_dir, result, attempt_id="attempt-1"
            )
            metrics = research_run_metrics(result)

            self.assertEqual("benter-rich-v1", params["recipe.feature_schema"])
            self.assertEqual("logit", params["recipe.model.kind"])
            self.assertEqual(0.25, params["recipe.model.parameters.C"])
            self.assertEqual(2.01, metrics["objective"])
            self.assertEqual(
                2.01, metrics["fold.fold-001.selected.race_log_loss"]
            )
            self.assertEqual(2.0, metrics["training.fold-001.feature_count"])

    def test_research_package_version_is_idempotent_and_loadable(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = ResearchModelPackage(
                DummyProbabilityModel(),
                PipelineRecipe(),
                protocol_id="protocol-1",
                code_revision="abc123",
            )
            package_dir = package.save(root / "package")
            config = MLflowConfig(
                tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
                experiment_name="agentic-test",
                enabled=True,
                register_models=True,
                registered_model_name="agentic-candidates",
            )
            result = {
                "target_kind": "win_probability",
                "recipe_hash": package.recipe.recipe_hash(),
                "objective_name": "development_race_log_loss",
                "objective_value": 1.0,
                "lineage": {
                    "protocol_id": "protocol-1",
                    "dataset_hash": "dataset-1",
                    "code_revision": "abc123",
                    "environment_hash": "environment-1",
                },
            }
            first = log_research_package_version(
                package_dir, config, attempt_id="attempt-1", result=result
            )
            second = log_research_package_version(
                package_dir, config, attempt_id="attempt-1", result=result
            )
            self.assertEqual(first, second)
            self.assertIn("model_version", first)
            run = mlflow.get_run(first["run_id"])
            self.assertEqual("baseline-v1", run.data.params["recipe.feature_schema"])
            self.assertEqual(1.0, run.data.metrics["objective"])
            loaded = mlflow.pyfunc.load_model(first["registered_model_uri"])
            frame = pd.DataFrame({
                "race_id": ["R1", "R1", "R2", "R2"],
                "field_size": [2, 2, 2, 2],
            })
            np.testing.assert_allclose(loaded.predict(frame), np.full(4, 0.5))


if __name__ == "__main__":
    unittest.main()
