import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ima.research_executor import RecipeExecutionRequest, execute_recipe
from ima.research_model_package import load_research_package
from ima.research_specs import FUNDAMENTAL_FIRST_PORTFOLIO_VERSION, PipelineRecipe


def dataset(root, races=26):
    rng = np.random.default_rng(73)
    rows = []
    for race in range(races):
        ability = np.array([0.2, 0.5, 0.8])+rng.normal(0, 0.25, 3)
        performance = ability+rng.normal(0, 0.18, 3)
        ranks = np.argsort(np.argsort(-performance))+1
        for horse in range(3):
            date = pd.Timestamp("2020-01-01")+pd.Timedelta(days=race)
            seconds = 72-1.8*performance[horse]+rng.normal(0, 0.15)
            rows.append({"race_id": f"R{race:03}", "date": date, "race_no": 1,
                "horse_no": horse+1, "horse_id": f"H{horse}", "field_size": 3,
                "result": ranks[horse], "target_win": int(ranks[horse] == 1),
                "target_probability": float(ranks[horse] == 1), "market_probability": 1/3,
                "distance": 1200, "finish_seconds": seconds, "finish_time": f"1:{seconds-60:05.2f}",
                "horse_age": 3+horse, "horse_rating": 60+ability[horse]*12,
                "prior_win_rate": ability[horse], "actual_weight": 120+horse,
                "custom_form": ability[horse]**2, "custom_available_at": date-pd.Timedelta(days=1)})
    frame = pd.DataFrame(rows)
    path = root/"dataset.csv"
    frame.to_csv(path, index=False)
    (root/"manifest.json").write_text(json.dumps({"availability_policy": "strict_pre_race",
        "predictor_catalog": [{"name": "custom_form", "eligible": True, "unit": "1",
            "temporal_scope": "historical", "available_at_column": "custom_available_at"}]}))
    return path, frame


FORMULA = {"name": "age_form", "expression_ast": {"op": "multiply", "args": [
    {"op": "ref", "ref": "prior_win_rate"}, {"op": "ref", "ref": "horse_age"}]},
    "input_refs": ["prior_win_rate", "horse_age"], "unit": "1"}
DISCOVERY = {"schema_version": 2, "entities": ["horse"], "measurements": ["speed_mps"],
    "aggregates": ["count", "mean"], "windows_days": [None], "max_definitions": 16,
    "max_selected": None, "selection": "quality", "correlation_threshold": 1.0}


def graph():
    return {"graph_id": "executor-v6", "primary_node_id": "primary", "output_node_id": "market",
        "fundamental_node_id": "calibrated", "n_splits": 2, "nodes": [
            {"node_id": "primary", "kind": "estimator", "parameters": {"model_kind": "benter_conditional_logit"}},
            {"node_id": "boosted", "kind": "estimator", "parameters": {"model_kind": "boosted", "model_parameters": {"max_iter": 8, "min_samples_leaf": 2}}},
            {"node_id": "stack", "kind": "meta_estimator", "inputs": ["primary", "boosted"]},
            {"node_id": "calibrated", "kind": "calibrate", "inputs": ["stack"]},
            {"node_id": "market", "kind": "market_blend", "inputs": ["calibrated"], "output": {"market": True}}]}


class ResearchExecutorV6Tests(unittest.TestCase):
    def run_recipe(self, root, recipe, source, name="trial"):
        return execute_recipe(RecipeExecutionRequest(name, "v6-fixture", 0, recipe, source,
            root/"trials"/name, {"min_train_races": 12, "calibration_races": 3, "score_races": 3, "max_folds": 1},
            code_revision="v6-test", environment_hash="v6-fixture", portfolio_version=FUNDAMENTAL_FIRST_PORTFOLIO_VERSION))

    def assert_reload(self, result, frame):
        self.assertEqual(result.status, "completed", result.error)
        predictions = pd.read_csv(result.artifacts["predictions"])
        score = frame[frame.race_id.isin(predictions.race_id)].copy().reset_index(drop=True)
        package = load_research_package(Path(result.artifacts["package"]))
        actual = package.predict(score)
        expected = predictions["model_probability" if package.recipe.target.kind == "win_probability" else "prediction"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        return package, score, predictions

    def test_basic_formula_extra_and_replay_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            for name, extras in (("basic", {}), ("formula", {"feature_definitions": [FORMULA], "extra_numeric_features": ["custom_form"]})):
                recipe = PipelineRecipe(schema_version=3, model={"kind": "benter_conditional_logit"},
                    blend={"kind": "none"}, **extras)
                result = self.run_recipe(root, recipe, source, name)
                package, score, predictions = self.assert_reload(result, frame)
                self.assertEqual(result.lineage["availability_policy"], "pre_race_synthetic")
                self.assertEqual(result.lineage["probability_basis"], "fundamental")
                self.assertEqual(result.lineage["target_unit"], "1")
                self.assertTrue(result.lineage["evaluation_population_id"].startswith("evaluated-"))
                self.assertTrue((Path(result.artifacts["package"])/"feature-replay.json").exists())
                if name == "formula":
                    score["age_form"] = -10000
                    np.testing.assert_allclose(package.predict(score), predictions.model_probability, rtol=1e-10, atol=1e-12)
                    with self.assertRaisesRegex(ValueError, "declared dataset"):
                        package.predict(score.drop(columns="custom_form"))

    def test_shared_dfs_cold_warm_reports_selector_and_active_pin(self):
        from ima.modeling import RaceProbabilityModel
        from ima.research_preparation import PreparationCache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            recipe = PipelineRecipe(schema_version=3, feature_discovery=DISCOVERY,
                model={"kind": "benter_conditional_logit"}, blend={"kind": "none"})
            original = RaceProbabilityModel.fit
            attempted = []
            def fitted(model, train):
                path = root/"trials"/"cold"/"preparation-fold-001.json"
                if path.exists():
                    preparation = json.loads(path.read_text())
                    attempted.append(PreparationCache(root/"fold-preparation-cache").evict(preparation["artifact_id"]))
                return original(model, train)
            with patch.object(RaceProbabilityModel, "fit", fitted):
                cold = self.run_recipe(root, recipe, source, "cold")
            package, score, predictions = self.assert_reload(cold, frame)
            self.assertTrue(attempted)
            self.assertFalse(any(attempted))
            warm = self.run_recipe(root, recipe.model_copy(update={"model": recipe.model.model_copy(update={"parameters": {"l2": 0.2}})}), source, "warm")
            self.assert_reload(warm, frame)
            cold_report = json.loads((root/"trials"/"cold"/"discovery-selection-fold-001.json").read_text())
            warm_report = json.loads((root/"trials"/"warm"/"discovery-selection-fold-001.json").read_text())
            self.assertEqual(cold_report, warm_report)
            self.assertGreater(cold_report["selected_count"], 0)
            self.assertTrue((root/"trials"/"warm"/"discovery-selector-fold-001.joblib").exists())
            prepared = json.loads((root/"trials"/"warm"/"preparation-fold-001.json").read_text())
            self.assertEqual(prepared["cache_status"], "hit")
            self.assertTrue(PreparationCache(root/"fold-preparation-cache").evict(prepared["artifact_id"]))
            # Package predictions depend on packaged history/definitions, not evicted shared arrays.
            np.testing.assert_allclose(package.predict(score), predictions.model_probability, rtol=1e-10, atol=1e-12)
            live = score.drop(columns=["finish_time", "finish_seconds", "result", "target_win", "target_probability"])
            self.assertTrue(np.isfinite(package.predict(live)).all())

    def test_graph_nested_fit_real_trial_and_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            for name, extras in (("graph", {"feature_definitions": [FORMULA]}), ("graph-dfs", {"feature_discovery": DISCOVERY})):
                recipe = PipelineRecipe(schema_version=3, pipeline_graph=graph(),
                    model={"kind": "benter_conditional_logit", "parameters": {"l2": 0.1}}, **extras)
                self.assertEqual(recipe.calibration.kind, "none")
                self.assertEqual(recipe.blend.kind, "none")
                result = self.run_recipe(root, recipe, source, name)
                package, score, predictions = self.assert_reload(result, frame)
                np.testing.assert_allclose(package.predict_fundamental_proba(score),
                    predictions.model_probability, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(package.predict_selected_proba(score), predictions.selected_probability, rtol=1e-10, atol=1e-12)
                first = pd.to_datetime(score.date).eq(pd.to_datetime(score.date).min())
                raw = score.loc[first].drop(columns=["finish_time", "finish_seconds", "result", "target_win", "target_probability"])
                np.testing.assert_allclose(package.predict_fundamental_proba(raw),
                    predictions.loc[first, "model_probability"], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(package.predict_selected_proba(raw),
                    predictions.loc[first, "selected_probability"], rtol=1e-10, atol=1e-12)
                fits = json.loads(Path(result.artifacts["graph_fits"]).read_text())
                self.assertGreater(fits["physical_fits"], 2)
                for artifact in fits["folds"][0]["oof_artifacts"]:
                    self.assertLess(pd.Timestamp(artifact["training_cutoff"]), pd.Timestamp(artifact["prediction_start"]))

    def test_graph_outer_adapters_are_disabled_or_explicitly_rejected(self):
        specification = {"schema_version": 3, "pipeline_graph": graph(),
                         "model": {"kind": "benter_conditional_logit"}}
        implicit = PipelineRecipe(**specification)
        explicit = PipelineRecipe(**specification, calibration={"kind": "none"}, blend={"kind": "none"})
        self.assertEqual(implicit.recipe_hash(), explicit.recipe_hash())
        for adapter, value in (("calibration", "temperature"), ("blend", "market_softmax")):
            with self.subTest(adapter=adapter), self.assertRaisesRegex(ValueError, "inside graph nodes"):
                PipelineRecipe(**specification, **{adapter: {"kind": value}})

    def test_probit_and_speed_distribution_actual_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            recipes = [
                ("probit", PipelineRecipe(schema_version=3, model={"kind": "gaussian_probit", "parameters": {"max_iter": 100}}, blend={"kind": "none"})),
                ("point-speed", PipelineRecipe(schema_version=3, target={"kind": "adjusted_finish_time_or_speed"},
                    model={"kind": "ridge_regressor"}, calibration={"kind": "none"}, blend={"kind": "none"})),
                ("shared-speed", PipelineRecipe(schema_version=3, target={"kind": "adjusted_finish_time_or_speed"},
                    model={"kind": "ridge_regressor"}, calibration={"kind": "none"}, blend={"kind": "none"}, performance_distribution={"kind": "shared_residual"})),
                ("conditional-speed", PipelineRecipe(schema_version=3, target={"kind": "adjusted_finish_time_or_speed"},
                    model={"kind": "catboost_regressor", "parameters": {"iterations": 30, "depth": 3}},
                    calibration={"kind": "none"}, blend={"kind": "none"}, performance_distribution={"kind": "catboost_uncertainty"})),
            ]
            for name, recipe in recipes:
                with self.subTest(name=name):
                    result = self.run_recipe(root, recipe, source, name)
                    package, score, predictions = self.assert_reload(result, frame)
                    if recipe.target.kind == "adjusted_finish_time_or_speed":
                        np.testing.assert_allclose(package.predict_finish_seconds(score), predictions.prediction_finish_seconds, rtol=1e-10, atol=1e-12)
                        np.testing.assert_allclose(predictions.prediction_speed_mps, predictions.prediction)
                        np.testing.assert_allclose(predictions.label_finish_seconds, score.finish_seconds)
                        np.testing.assert_allclose(predictions.prediction_finish_seconds,
                            score.distance.to_numpy() / predictions.prediction_speed_mps.to_numpy())
                        errors = predictions.prediction_finish_seconds.to_numpy() - score.finish_seconds.to_numpy()
                        diagnostic = result.metrics["folds"][0]["finish_time"]
                        self.assertAlmostEqual(diagnostic["mae"], np.mean(np.abs(errors)))
                        self.assertAlmostEqual(diagnostic["rmse"], np.sqrt(np.mean(errors**2)))
                        self.assertTrue(predictions.finish_time_prediction_valid.all())
                        raw = score.drop(columns=["finish_time", "finish_seconds", "result", "target_win", "target_probability"])
                        np.testing.assert_allclose(package.predict_finish_seconds(raw), predictions.prediction_finish_seconds, rtol=1e-10, atol=1e-12)
                        invalid = score.copy()
                        invalid["distance"] = 0
                        self.assertTrue(np.isnan(package.predict_finish_seconds(invalid)).all())
                        with self.assertRaisesRegex(ValueError, "raw distance"):
                            package.predict_finish_seconds(score.drop(columns="distance"))
                    else:
                        with self.assertRaisesRegex(TypeError, "physical-speed"):
                            package.predict_finish_seconds(score)
                    if recipe.performance_distribution:
                        dist = package.predict_distribution(score)
                        np.testing.assert_allclose(dist.location, predictions.performance_location, rtol=1e-10, atol=1e-12)
                        np.testing.assert_allclose(dist.scale, predictions.performance_scale, rtol=1e-10, atol=1e-12)
                        np.testing.assert_allclose(package.predict_auxiliary_win_proba(score), predictions.performance_win_probability, rtol=1e-10, atol=1e-12)
                        self.assertIn("performance_win", result.metrics["summary"])
                        self.assertIn("distribution", result.metrics["summary"])
                        expected_time = score.distance.to_numpy() * np.exp(-dist.location + 0.5 * dist.scale**2)
                        ratio_time = package.predict_finish_seconds(score)
                        np.testing.assert_allclose(expected_time, ratio_time * np.exp(dist.scale**2))
                        self.assertTrue(np.all(expected_time > ratio_time))
                        self.assertEqual(diagnostic["prediction_basis"], "distance_over_mean_speed_not_mean_finish_time")

    def test_speed_time_readout_invalid_support_is_explicit(self):
        from ima.research_model_package import speed_to_finish_seconds
        distance = np.array([1200, 1200, 1200, -1200, np.nan, 1200, 1200])
        speed = np.array([20, 0, -1, 20, 20, np.inf, np.nan])
        seconds = speed_to_finish_seconds(distance, speed)
        self.assertEqual(seconds[0], 60)
        self.assertTrue(np.isnan(seconds[1:]).all())
        with self.assertRaisesRegex(ValueError, "aligned"):
            speed_to_finish_seconds([1200], [20, 30])

    def test_native_speed_and_rank_graph_labels_and_whole_race_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            specifications = [
                ("speed-native", "ridge_regressor", "performance_distribution", "target_speed", "log_mps", "probabilistic_adapter", {}),
                ("speed-uncertainty", "catboost_regressor", "performance_distribution", "target_speed", "log_mps", "probabilistic_adapter",
                 {"model_parameters": {"iterations": 12, "depth": 3}, "performance_distribution": {"kind": "catboost_uncertainty"}}),
                ("rank-native", "ridge_regressor", "ranking_score", "target_rank_score", "score", "race_normalize", {}),
            ]
            frame.loc[(frame.race_id == "R002") & (frame.horse_no == 1), "finish_seconds"] = np.nan
            frame.to_csv(source,index=False)
            for name, kind, output_kind, label, unit, adapter, parameters in specifications:
                with self.subTest(name=name):
                    spec = {"graph_id": name, "primary_node_id": "native", "output_node_id": "win", "nodes": [
                        {"node_id": "native", "kind": "estimator", "parameters": {"model_kind": kind, **parameters},
                         "output": {"kind": output_kind, "target": label, "unit": unit}},
                        {"node_id": "win", "kind": adapter, "inputs": ["native"]}]}
                    recipe = PipelineRecipe(schema_version=3, pipeline_graph=spec,
                        model={"kind": kind}, calibration={"kind": "none"}, blend={"kind": "none"})
                    result = self.run_recipe(root, recipe, source, name)
                    package, score, predictions = self.assert_reload(result, frame)
                    if label == "target_speed":
                        self.assertEqual(result.metrics["dataset_exclusions"]["graph_untimed_or_unlabelled_races"], 1)
                    else:
                        adapter = package.model.model.output
                        self.assertIsNotNone(adapter.state)
                        self.assertGreater(adapter.state.temperature, 0)
                        report = json.loads(Path(result.artifacts["graph_fits"]).read_text())["folds"][0]
                        self.assertTrue(any(row["fit_kind"] == "ranking_probability_adapter"
                                            and row["fit_scope"] == "forward_oof" for row in report["component_fits"]))
                        np.testing.assert_allclose(package.predict_fundamental_proba(score),
                            predictions.model_probability, rtol=1e-10, atol=1e-12)

    def test_manifest_timing_unknown_bindings_and_thread_budget_fail_closed(self):
        from ima.performance_distributions import CatBoostUncertaintyModel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            recipe = PipelineRecipe(schema_version=3, extra_numeric_features=["unknown_input"])
            result = self.run_recipe(root, recipe, source)
            self.assertEqual(result.status, "failed")
            self.assertIn("predictor catalog", result.error)
            self.assertFalse((root/"trials"/"trial"/"package").exists())
            with patch.dict("os.environ", {"IMA_RESEARCH_THREADS": "0"}):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    CatBoostUncertaintyModel(("prior_win_rate",), parameters={"iterations": 2}).fit(frame, label_column="finish_seconds")
            with patch.dict("os.environ", {"IMA_RESEARCH_THREADS": "2"}):
                model = CatBoostUncertaintyModel(("prior_win_rate",), parameters={"iterations": 2}).fit(frame, label_column="finish_seconds")
                self.assertEqual(model.model.get_params()["thread_count"], 2)

    def test_verified_manifest_and_actual_population_lineage(self):
        from ima.research_telemetry import comparison_key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) | {
                "dataset_id": "official-fixture", "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "predictor_catalog_id": "catalog-fixture", "evaluation_population_id": "declared-development",
                "target_units": {"win_probability": "1", "adjusted_finish_time_or_speed": "m/s"},
            }
            manifest_path.write_text(json.dumps(manifest))
            recipe = PipelineRecipe(schema_version=3, model={"kind": "benter_conditional_logit"}, blend={"kind": "none"})
            first = self.run_recipe(root, recipe, source, "verified")
            self.assert_reload(first, frame)
            self.assertEqual(first.lineage["dataset_id"], "official-fixture")
            self.assertEqual(first.lineage["predictor_catalog_id"], "catalog-fixture")
            self.assertEqual(first.lineage["availability_policy"], "strict_pre_race")
            self.assertEqual(first.lineage["declared_evaluation_population_id"], "declared-development")
            population = json.loads(Path(first.artifacts["evaluation_population"]).read_text())
            self.assertEqual(len(population["rows"]), 9)
            self.assertEqual(population["evaluation_population_id"], first.lineage["evaluation_population_id"])
            second = self.run_recipe(root, recipe.model_copy(update={"extra_numeric_features": ("custom_form",)}), source, "features")
            self.assertEqual(second.status, "completed", second.error)
            self.assertEqual(comparison_key(first.serializable(), recipe.model_dump(mode="json")),
                             comparison_key(second.serializable(), recipe.model_dump(mode="json")))
            changed = frame.copy()
            race = changed.race_id.eq("R015")
            changed.loc[race, "target_win"] = [0, 1, 0] if list(changed.loc[race, "target_win"]) != [0, 1, 0] else [1, 0, 0]
            changed.loc[race, "target_probability"] = changed.loc[race, "target_win"].astype(float)
            changed.to_csv(source, index=False)
            rejected = self.run_recipe(root, recipe, source, "mismatch")
            self.assertEqual(rejected.status, "failed")
            self.assertIn("checksum", rejected.error)
            manifest["dataset_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            third = self.run_recipe(root, recipe, source, "changed-labels")
            self.assertEqual(third.status, "completed", third.error)
            self.assertNotEqual(first.lineage["evaluation_population_id"], third.lineage["evaluation_population_id"])

    def test_registry_manifest_files_catalog_and_comparison_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = dataset(root)
            manifest = {"dataset_id": "dataset-fixture", "status": "verified", "event_policy_id": "strict",
                "files": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
                "feature_catalog": {"prior_win_rate": {"unit": "1", "family": "registered_rich_history",
                    "cutoff_rule": "pre_race_or_previous_race_registered_formula"}},
                "comparison_contract": {"evaluation_population_id": "registry-development",
                    "availability_tier": "strict", "retrospective_lags": {}, "probability_basis": "fundamental_pre_race"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            recipe = PipelineRecipe(schema_version=3, model={"kind": "benter_conditional_logit"}, blend={"kind": "none"})
            result = self.run_recipe(root, recipe, source, "registry")
            package, _, _ = self.assert_reload(result, frame)
            self.assertEqual(package.model.feature_schema.numeric, ("prior_win_rate",))
            self.assertEqual(package.model.feature_schema.categorical, ())
            self.assertEqual(result.lineage["availability_policy"], "strict")
            self.assertEqual(result.lineage["probability_basis"], "fundamental_pre_race")
            self.assertEqual(result.lineage["declared_evaluation_population_id"], "registry-development")

    def test_actual_registry_formula_recipe_recomputes_and_replays(self):
        from ima.dataset_registry import _digest
        from ima.dataset_specs import DatasetRequest
        from ima.feature_definitions import FeatureRegistry
        from ima.research_executor import _v6_feature_frame
        from tests.test_dataset_registry_adversarial import DatasetRegistryAdversarialTests, digest

        fixture = DatasetRegistryAdversarialTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        snapshot, raw_manifest = fixture.fixture()
        registry = fixture.registry
        specification = {"request_id": "executor-parent", "raw_corpus_manifest_id": digest(raw_manifest),
            "rationale": "Actual executor formula lifecycle", "evidence_watermark": "fixture",
            "protocol": {"min_train_races": 2, "calibration_races": 1, "score_races": 1,
                         "max_folds": 2, "final_confirmation_races": 2}}
        registry.submit(DatasetRequest.model_validate(specification))
        parent = registry.build("executor-parent", source_snapshot=snapshot, raw_manifest=raw_manifest)
        keys = {"unit", "dtype", "temporal_scope", "target_tainted", "available_at_column", "source_family"}
        metadata = {name: {key: value for key, value in row.items() if key in keys}
                    for name, row in parent["predictor_catalog"].items()}
        definition = {"name": "body_to_carried_ratio", "input_refs": ["declared_weight", "actual_weight"],
            "expression_ast": {"op": "divide", "args": [{"op": "ref", "ref": "declared_weight"},
                                                          {"op": "ref", "ref": "actual_weight"}]}, "unit": "1"}
        identifier = FeatureRegistry(registry.feature_registry).register(definition, metadata)
        registry.submit(DatasetRequest.model_validate(dict(specification, request_id="executor-formula",
            parent_dataset_id=parent["dataset_id"], feature_definition_ids=[identifier])))
        manifest = registry.build("executor-formula", source_snapshot=snapshot, raw_manifest=raw_manifest)
        path = registry.dataset_path(manifest["dataset_id"])
        source = path / "features.parquet"
        frame = pd.read_parquet(source)
        recipe = PipelineRecipe(schema_version=3, model={"kind": "benter_conditional_logit"},
            feature_definitions=[definition], extra_numeric_features=[definition["name"]], blend={"kind": "none"})
        request = RecipeExecutionRequest("registry-formula", "v6-fixture", 0, recipe, source,
            fixture.root / "trials" / "registry-formula",
            {"min_train_races": 2, "calibration_races": 1, "score_races": 1, "max_folds": 1},
            portfolio_version=FUNDAMENTAL_FIRST_PORTFOLIO_VERSION)
        result = execute_recipe(request)
        package, score, predictions = self.assert_reload(result, frame)
        raw = score.drop(columns=[definition["name"], "dfs_" + identifier])
        np.testing.assert_allclose(package.predict(raw), predictions.model_probability, rtol=1e-10, atol=1e-12)
        tampered = frame.copy()
        tampered[definition["name"]] = -9999
        tampered["dfs_" + identifier] = -9999
        regenerated = _v6_feature_frame(tampered, request)
        np.testing.assert_allclose(regenerated[definition["name"]], frame.declared_weight / frame.actual_weight)
        self.assertNotIn(definition["name"], regenerated.attrs["input_metadata"])
        for field, value in (("definition_id", "0" * 24), ("target_tainted", True)):
            invalid = json.loads(json.dumps(manifest))
            invalid["predictor_catalog"][definition["name"]][field] = value
            (path / "manifest.json").write_text(json.dumps(invalid))
            rejected = execute_recipe(request)
            self.assertEqual(rejected.status, "failed")
            self.assertRegex(rejected.error, "Dataset (predictor catalog identity changed|manifest checksum mismatch|identity payload changed)")
            # Rebinding the sidecar cannot rebind the catalog's content-addressed provenance.
            (path / "manifest.sha256").write_text(digest(path / "manifest.json"))
            rejected = execute_recipe(request)
            self.assertEqual(rejected.status, "failed")
            self.assertRegex(rejected.error, "Dataset (predictor catalog identity changed|identity payload changed)")
            invalid["identity"]["predictor_catalog_sha256"] = _digest(invalid["predictor_catalog"])
            (path / "manifest.json").write_text(json.dumps(invalid))
            (path / "manifest.sha256").write_text(digest(path / "manifest.json"))
            rejected = execute_recipe(request)
            self.assertEqual(rejected.status, "failed")
            self.assertIn("Dataset identity payload changed", rejected.error)
