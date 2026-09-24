import unittest

from pathlib import Path
from unittest import mock

from ima.mlflow_tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    MLflowConfig,
    _model_artifact_source,
    flatten_numeric_metrics,
    run_parameters,
)


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


if __name__ == "__main__":
    unittest.main()
