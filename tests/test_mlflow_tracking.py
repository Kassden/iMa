import unittest
import tempfile
import shutil
from types import SimpleNamespace

from pathlib import Path
from unittest import mock

from ima.mlflow_tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    MLflowConfig,
    _model_artifact_source,
    _cycle_result_summary,
    _trace_result_preview,
    _research_model_linkage,
    _writable_code_copy,
    flatten_numeric_metrics,
    log_optimizer_cycle_trace,
    log_optimizer_planner_trace,
    log_research_package,
    log_research_package_version,
    research_identity_tags,
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
    def test_writable_code_copy_preserves_immutable_source_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ima"
            nested = source / "nested"
            nested.mkdir(parents=True)
            module = nested / "module.py"
            module.write_text("VALUE = 42\n")
            module.chmod(0o444)
            nested.chmod(0o555)
            source.chmod(0o555)
            try:
                with _writable_code_copy(source) as staged:
                    self.assertEqual("ima", staged.name)
                    self.assertEqual(module.read_bytes(), (staged / "nested/module.py").read_bytes())
                    for path in (staged, staged / "nested", staged / "nested/module.py"):
                        self.assertTrue(path.stat().st_mode & 0o200)
                    # Model MLflow's copytree + rmtree cleanup, including mode copying.
                    copied = Path(directory) / "mlflow-copy"
                    shutil.copytree(staged, copied)
                    shutil.rmtree(copied)
                self.assertFalse(staged.exists())
                self.assertEqual(0o555, source.stat().st_mode & 0o777)
                self.assertEqual(0o555, nested.stat().st_mode & 0o777)
                self.assertEqual(0o444, module.stat().st_mode & 0o777)
            finally:
                source.chmod(0o755)
                nested.chmod(0o755)

    def test_linkage_checks_backend_run_and_model_status(self):
        client = mock.Mock()
        client.get_run.return_value.info = SimpleNamespace(status="FAILED", experiment_id="1")
        with self.assertRaisesRegex(RuntimeError, "FAILED, not FINISHED"):
            _research_model_linkage(client, "candidates", "run")
        client.search_model_versions.assert_not_called()
        client.get_run.return_value.info.status = "FINISHED"
        client.search_logged_models.return_value = [SimpleNamespace(status="FAILED")]
        with self.assertRaisesRegex(RuntimeError, "no READY logged model"):
            _research_model_linkage(client, "candidates", "run")
        client.search_logged_models.return_value = [SimpleNamespace(status="READY")]
        client.search_model_versions.return_value = [
            SimpleNamespace(name="candidates", status="FAILED_REGISTRATION", version="1")]
        self.assertNotIn("model_version", _research_model_linkage(client, "candidates", "run"))

    def test_large_v6_feature_recipe_is_hashed_in_params_not_truncated_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = PipelineRecipe(schema_version=3,
                extra_numeric_features=tuple(f"measurement_{i}" for i in range(1000)))
            package = ResearchModelPackage(DummyProbabilityModel(), recipe,
                                          protocol_id="p", code_revision="r")
            package_dir = package.save(Path(directory) / "package")
            params = research_run_parameters(package_dir, {}, attempt_id="a")
            value = params["recipe.extra_numeric_features"]
            self.assertTrue(value.startswith("sha256:"))
            self.assertIn("see recipe.json", value)
            self.assertLess(len(value), 5000)

    def test_cycle_trace_preview_keeps_mixed_objectives_separate(self):
        summary = _cycle_result_summary([
            {
                "attempt_id": "win-1", "status": "completed",
                "target_kind": "win_probability",
                "objective_name": "development_fundamental_race_log_loss",
                "objective_value": 2.1636051360490574,
            },
            {
                "attempt_id": "rank-1", "status": "completed",
                "target_kind": "ranking_strength",
                "objective_name": "negative_development_race_ndcg_at_3",
                "objective_value": -0.75,
            },
        ])
        preview = _trace_result_preview(summary)
        self.assertIn("win_probability:development_fundamental_race_log_loss:{}=2.163605", preview)
        self.assertIn("ranking_strength:negative_development_race_ndcg_at_3:{}=-0.750000", preview)
        self.assertNotIn("None=None", preview)

    def test_cycle_trace_preview_has_no_false_best_for_empty_cycle(self):
        self.assertEqual(
            "0/0 completed; no scored trials",
            _trace_result_preview(_cycle_result_summary([])),
        )

    def test_cycle_trace_preview_distinguishes_unchanged_campaign_best(self):
        current = {
            "attempt_id": "current", "status": "completed",
            "target_kind": "win_probability",
            "objective_name": "development_fundamental_race_log_loss",
            "objective_value": 2.18,
        }
        prior = {**current, "attempt_id": "prior", "objective_value": 2.16}
        summary = _cycle_result_summary([current], campaign_results=[prior, current])
        preview = _trace_result_preview(summary)
        self.assertIn("campaign best win_probability=2.160000", preview)
        self.assertIn("cycle best by objective: win_probability", preview)
        self.assertNotIn("NEW campaign best", preview)

    def test_config_is_disabled_without_uri(self):
        with mock.patch.dict("os.environ", {"MLFLOW_TRACKING_URI": ""}):
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
                    "metric_contract_version": 2,
                    "summary": {
                        "selected": {
                            "race_log_loss": {"mean": 2.01, "std": 0.1, "worst": 2.11}
                        }
                    },
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
            self.assertEqual("benter-rich-v1", params["feature_schema"])
            self.assertEqual("logit", params["recipe.model.kind"])
            self.assertEqual("logit", params["model_kind"])
            self.assertEqual("all_history", params["train_window"])
            self.assertEqual("temperature", params["calibration_kind"])
            self.assertEqual("market_softmax", params["blend_kind"])
            self.assertEqual(
                "benter-rich-v1", research_identity_tags(params)["ima.feature_schema"]
            )
            self.assertEqual(0.25, params["recipe.model.parameters.C"])
            self.assertEqual(2, params["metric_contract_version"])
            self.assertEqual(2.01, metrics["objective"])
            self.assertEqual(
                2.01, metrics["summary.selected.race_log_loss.mean"]
            )
            self.assertEqual(
                2.01, metrics["fold.fold-001.selected.race_log_loss"]
            )
            self.assertEqual(2.0, metrics["training.fold-001.feature_count"])

    def test_research_package_version_is_idempotent_and_loadable(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "rich-history.csv.gz"
            dataset_path.write_bytes(b"research dataset fixture")
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
                    "dataset_hash": "a" * 64,
                    "code_revision": "abc123",
                    "environment_hash": "environment-1",
                },
            }
            first = log_research_package_version(
                package_dir, config, attempt_id="attempt-1", result=result,
                dataset_path=dataset_path,
            )
            second = log_research_package_version(
                package_dir, config, attempt_id="attempt-1", result=result,
                dataset_path=dataset_path,
            )
            self.assertEqual(first, second)
            self.assertIn("model_version", first)
            run = mlflow.get_run(first["run_id"])
            self.assertEqual("FINISHED", run.info.status)
            self.assertEqual("baseline-v1", run.data.params["recipe.feature_schema"])
            self.assertEqual("baseline-v1", run.data.params["feature_schema"])
            self.assertEqual("logit", run.data.params["model_kind"])
            self.assertEqual("baseline-v1", run.data.tags["ima.feature_schema"])
            self.assertEqual("logit", run.data.tags["ima.model_kind"])
            self.assertEqual("a" * 64, run.data.tags["ima.dataset_hash"])
            self.assertEqual(1.0, run.data.metrics["objective"])
            self.assertEqual(1, len(run.inputs.dataset_inputs))
            self.assertEqual("rich-history.csv.gz", run.inputs.dataset_inputs[0].dataset.name)
            self.assertEqual("a" * 32, run.inputs.dataset_inputs[0].dataset.digest)
            loaded = mlflow.pyfunc.load_model(first["registered_model_uri"])
            frame = pd.DataFrame({
                "race_id": ["R1", "R1", "R2", "R2"],
                "field_size": [2, 2, 2, 2],
            })
            np.testing.assert_allclose(loaded.predict(frame), np.full(4, 0.5))

    def test_failed_run_with_ready_version_retries_without_rewriting_failure(self):
        import mlflow
        import ima.mlflow_tracking as tracking

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = ResearchModelPackage(DummyProbabilityModel(), PipelineRecipe(),
                                          protocol_id="protocol-1", code_revision="abc123")
            package_dir = package.save(root / "package")
            config = MLflowConfig(tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
                                  experiment_name="failed-retry", enabled=True,
                                  registered_model_name="retry-candidates")
            result = {"target_kind": "win_probability", "recipe_hash": package.recipe.recipe_hash()}
            source = root / "release" / "ima"
            shutil.copytree(Path(tracking.__file__).parent, source,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            directories = [source, *[p for p in source.rglob("*") if p.is_dir()]]
            for path in directories:
                path.chmod(0o555)
            real_log_model = mlflow.pyfunc.log_model

            def fail_after_registration(*args, **kwargs):
                real_log_model(*args, **kwargs)
                raise PermissionError("simulated cleanup failure after registration")

            try:
                with mock.patch.object(tracking, "_writable_code_copy",
                                       side_effect=lambda _: _writable_code_copy(source)):
                    with mock.patch.object(mlflow.pyfunc, "log_model", side_effect=fail_after_registration):
                        with self.assertRaisesRegex(PermissionError, "cleanup failure"):
                            log_research_package_version(package_dir, config,
                                attempt_id="failed-attempt", result=result)
                    client = mlflow.tracking.MlflowClient()
                    experiment = client.get_experiment_by_name(config.experiment_name)
                    failed = client.search_runs([experiment.experiment_id])[0]
                    self.assertEqual("FAILED", failed.info.status)
                    failed_versions = client.search_model_versions(f"run_id = '{failed.info.run_id}'")
                    self.assertEqual("READY", failed_versions[0].status)
                    retry = log_research_package_version(package_dir, config,
                        attempt_id="failed-attempt", result=result)
                    again = log_research_package_version(package_dir, config,
                        attempt_id="failed-attempt", result=result)
                self.assertEqual(retry, again)
                self.assertNotEqual(failed.info.run_id, retry["run_id"])
                self.assertNotEqual(str(failed_versions[0].version), retry["model_version"])
                self.assertEqual("FAILED", client.get_run(failed.info.run_id).info.status)
                successful = client.get_run(retry["run_id"])
                self.assertEqual("FINISHED", successful.info.status)
                self.assertEqual(failed.info.run_id, successful.data.tags["ima.retry_of_run_id"])
                self.assertEqual(2, len(client.search_runs([experiment.experiment_id])))
                self.assertTrue(all(p.stat().st_mode & 0o777 == 0o555 for p in directories))
                loaded = mlflow.pyfunc.load_model(retry["registered_model_uri"])
                np.testing.assert_allclose(loaded.predict(pd.DataFrame({"race_id": ["R", "R"]})), [0.5, 0.5])
            finally:
                for path in directories:
                    path.chmod(0o755)

    def test_optimizer_cycle_trace_links_decision_results_usage_and_cost(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MLflowConfig(
                tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
                experiment_name="trace-test",
                enabled=True,
                register_models=False,
            )
            decision = {
                "cycle": 3,
                "source": "openrouter",
                "evidence_id": "evidence-1",
                "planner_model": "deepseek/test",
                "service_tier": "flex",
                "planner_usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "total_cost_usd": 0.00125,
                    "cost_status": "reported",
                },
                "suggestions": [{
                    "trial_id": "proposal-1",
                    "hypothesis": "A grouped rank objective improves top-three order.",
                    "changed_axes": ["target"],
                    "recipe_hash": "recipe-1",
                    "recipe": {"target": {"kind": "ranking_strength"}},
                }],
            }
            results = [{
                "attempt_id": "attempt-1",
                "proposal_id": "proposal-1",
                "recipe_hash": "recipe-1",
                "target_kind": "ranking_strength",
                "status": "completed",
                "objective_name": "negative_development_race_ndcg_at_3",
                "objective_value": -0.72,
                "duration_seconds": 2.5,
                "metrics": {"summary": {"model": {"race_ndcg_at_3": {"mean": 0.72}}}},
            }, {
                "attempt_id": "attempt-2",
                "proposal_id": "proposal-2",
                "recipe_hash": "recipe-2",
                "target_kind": "win_probability",
                "status": "completed",
                "objective_name": "development_fundamental_race_log_loss",
                "objective_value": 2.1636051360490574,
                "duration_seconds": 3.0,
                "metrics": {"summary": {"model": {"race_log_loss": {"mean": 2.1636051360490574}}}},
            }]
            prior = {
                **results[1], "attempt_id": "prior-win",
                "objective_value": 2.18,
            }
            first = log_optimizer_cycle_trace(
                root, decision, results, config,
                campaign_results=[prior, *results],
            )
            second = log_optimizer_cycle_trace(root, decision, results, config)

            self.assertEqual(first, second)
            experiment = mlflow.get_experiment_by_name("trace-test")
            traces = mlflow.search_traces(
                experiment_ids=[experiment.experiment_id], return_type="list"
            )
            self.assertEqual(1, len(traces))
            trace = mlflow.get_trace(first["trace_id"], flush=True)
            spans = {span.name: span for span in trace.data.spans}
            self.assertIn("NEW campaign best win_probability=2.163605", trace.info.response_preview)
            self.assertIn("win_probability:development_fundamental_race_log_loss:{}=2.163605", trace.info.response_preview)
            self.assertNotIn("None=None", trace.info.response_preview)
            self.assertEqual(
                "attempt-2",
                spans["optimizer-cycle-0003"].outputs["new_campaign_best_by_objective"][
                    "win_probability:development_fundamental_race_log_loss:{}"
                ]["attempt_id"],
            )
            self.assertIn("optimizer-cycle-0003", spans)
            self.assertIn("planner-decision", spans)
            self.assertIn("trial-proposal-1", spans)
            self.assertEqual(
                120,
                spans["planner-decision"].get_attribute(
                    "mlflow.chat.tokenUsage"
                )["total_tokens"],
            )
            self.assertEqual(
                0.00125,
                spans["planner-decision"].get_attribute(
                    "mlflow.llm.cost"
                )["total_cost"],
            )
            self.assertEqual(120, trace.info.token_usage["total_tokens"])
            self.assertEqual(0.00125, trace.info.cost["total_cost"])
            self.assertEqual("reported", first["cost_status"])
            self.assertEqual(1, len(list((root / "traces").glob("cycle-*.json"))))

    def test_planner_trace_is_visible_before_trials_finish(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MLflowConfig(
                tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
                experiment_name="early-planner-trace",
                enabled=True,
                register_models=False,
            )
            decision = {
                "cycle": 49,
                "source": "benter_v3",
                "planner_model": "deepseek/test",
                "planner_usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "total_cost_usd": 0.00125,
                    "cost_status": "reported",
                },
                "suggestions": [],
            }
            first = log_optimizer_planner_trace(root, decision, config)
            self.assertEqual(first, log_optimizer_planner_trace(root, decision, config))
            trace = mlflow.get_trace(first["trace_id"], flush=True)
            self.assertEqual("planner-cycle-0049", trace.info.tags["mlflow.traceName"])
            self.assertEqual(120, trace.info.token_usage["total_tokens"])
            self.assertEqual(0.00125, trace.info.cost["total_cost"])
            self.assertEqual("planner", trace.info.tags["ima.trace_phase"])
            self.assertEqual("0.00125", trace.info.tags["planner_cost_usd"])
            self.assertIn("$0.001250000 USD", trace.info.response_preview)

    def test_optimizer_cycle_trace_marks_missing_cost_unavailable(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MLflowConfig(
                tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
                experiment_name="trace-no-cost",
                enabled=True,
                register_models=False,
            )
            linkage = log_optimizer_cycle_trace(
                root,
                {"cycle": 0, "source": "fixture", "suggestions": []},
                [],
                config,
            )
            trace = mlflow.get_trace(linkage["trace_id"], flush=True)
            planner = next(
                span for span in trace.data.spans if span.name == "planner-decision"
            )
            self.assertEqual("unavailable", linkage["cost_status"])
            self.assertIsNone(linkage["total_cost_usd"])
            self.assertIsNone(planner.get_attribute("mlflow.llm.cost"))
            self.assertIsNone(trace.info.cost)


if __name__ == "__main__":
    unittest.main()
