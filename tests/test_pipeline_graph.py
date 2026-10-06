import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.feature_sets import FeatureSchema
from ima.pipeline_graph import PipelineGraph, fit_graph, fit_pipeline_graph
from ima.prediction_store import PredictionStore


SCHEMA = FeatureSchema("graph-test", ("ability", "context"), ())


class ChronologyGuardEstimator:
    def fit(self, frame):
        self.cutoff = pd.to_datetime(frame.date).max()
        return self

    def predict_proba(self, frame):
        if pd.to_datetime(frame.date).min() <= self.cutoff:
            raise AssertionError("In-sample/future-trained component used for stacking")
        return np.full(len(frame), 0.25)


class ChronologyGuardRankEstimator(ChronologyGuardEstimator):
    prediction_windows = []

    def predict(self, frame):
        start = pd.to_datetime(frame.date).min()
        if start <= self.cutoff:
            raise AssertionError("In-sample/future-trained rank scores used for calibration")
        self.prediction_windows.append((self.cutoff, start))
        return 20 * frame.ability.to_numpy()


def race_frame(dates=30, start=0):
    rng = np.random.default_rng(13+start)
    rows = []
    for day in range(start, start+dates):
        x = rng.normal(size=(4, 2))
        winner = np.argmax(x[:, 0] + rng.normal(scale=0.7, size=4))
        for horse in range(4):
            rows.append({"race_id": f"race-{day}", "horse_id": f"horse-{horse}",
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "ability": x[horse, 0], "context": x[horse, 1],
                "target_win": int(horse == winner), "target_probability": float(horse == winner), "market_probability": 0.25})
    return pd.DataFrame(rows)


def graph_spec(output="pool"):
    return {"graph_id": "benter-boosted", "primary_node_id": "benter", "output_node_id": output,
        "nodes": [
            {"node_id": "benter", "kind": "estimator", "parameters": {"model_kind": "benter_conditional_logit"}},
            {"node_id": "boosted", "kind": "estimator", "parameters": {"model_kind": "boosted", "model_parameters": {"max_iter": 12, "min_samples_leaf": 2}}},
            {"node_id": "pool", "kind": "weighted_probability_pool", "inputs": ["benter", "boosted"], "parameters": {"weights": [0.6, 0.4]}}]}


class PipelineGraphTests(unittest.TestCase):
    def test_real_pool_primary_tuning_and_pickle(self):
        train, calibration, score = race_frame(20), race_frame(5, 20), race_frame(3, 25)
        fitted = fit_graph(graph_spec(), train, calibration, SCHEMA,
                           model_spec={"kind": "benter_conditional_logit", "parameters": {"l2": 0.2}})
        self.assertEqual(fitted.output.parents[0].state.conditional.l2, 0.2)
        p = fitted.predict_fundamental_proba(score)
        expected = 0.6*fitted.output.parents[0].predict(score)+0.4*fitted.output.parents[1].predict(score)
        np.testing.assert_allclose(p, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(p.reshape(-1, 4).sum(axis=1), 1)
        reloaded = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_allclose(reloaded.predict_proba(score), p, rtol=1e-10, atol=1e-12)
        self.assertEqual(fitted.fit_report["physical_fits"], 2)

    def test_stack_forward_proof_cache_and_market(self):
        spec = graph_spec("market")
        spec["fundamental_node_id"] = "calibrated"
        spec["nodes"] += [
            {"node_id": "stack", "kind": "meta_estimator", "inputs": ["benter", "boosted"]},
            {"node_id": "calibrated", "kind": "calibrate", "inputs": ["stack"]},
            {"node_id": "market", "kind": "market_blend", "inputs": ["calibrated"], "output": {"market": True}}]
        train, score = race_frame(32), race_frame(3, 32)
        with tempfile.TemporaryDirectory() as path:
            store = PredictionStore(path)
            fitted = fit_pipeline_graph(spec, train, feature_schema=SCHEMA, prediction_store=store)
            repeated = fit_pipeline_graph(spec, train, feature_schema=SCHEMA, prediction_store=store)
        for artifact in fitted.fit_report["oof_artifacts"]:
            self.assertLess(pd.Timestamp(artifact["training_cutoff"]), pd.Timestamp(artifact["prediction_start"]))
        self.assertGreater(repeated.fit_report["cache_hits"], 0)
        self.assertLess(repeated.fit_report["physical_fits"], fitted.fit_report["physical_fits"])
        np.testing.assert_allclose(fitted.predict_proba(score), repeated.predict_proba(score), rtol=1e-10, atol=1e-12)
        reloaded = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_allclose(fitted.predict_fundamental_proba(score), reloaded.predict_fundamental_proba(score), rtol=1e-10, atol=1e-12)

    def test_graph_contract_errors(self):
        spec = graph_spec()
        spec["nodes"][0]["inputs"] = ["pool"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            PipelineGraph.from_dict(spec).validate()
        spec = graph_spec()
        spec["nodes"][1]["output"] = {"kind": "ranking_score", "unit": "score"}
        with self.assertRaisesRegex(ValueError, "identical"):
            PipelineGraph.from_dict(spec).validate()
        spec = graph_spec()
        spec["component_refs"] = [{"training_cutoff": "2099-01-01"}]
        with self.assertRaisesRegex(ValueError, "references"):
            PipelineGraph.from_dict(spec).validate()
        spec = graph_spec("stack")
        spec["nodes"].append({"node_id": "stack", "kind": "meta_estimator", "inputs": ["pool"], "fit_scope": "in_sample"})
        with self.assertRaisesRegex(ValueError, "In-sample"):
            PipelineGraph.from_dict(spec).validate()
        spec = graph_spec()
        spec["nodes"][0]["output"] = {"unit": "seconds"}
        with self.assertRaisesRegex(ValueError, "unit"):
            PipelineGraph.from_dict(spec).validate()

    def test_duplicates_and_fit_budget(self):
        frame = race_frame(10)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fit_pipeline_graph(graph_spec(), pd.concat([frame, frame]), feature_schema=SCHEMA)
        spec = graph_spec()
        spec["fit_budget"] = 1
        with self.assertRaisesRegex(RuntimeError, "budget"):
            fit_pipeline_graph(spec, frame, feature_schema=SCHEMA)

    def test_equal_component_dedup_and_future_window(self):
        spec = graph_spec()
        spec["nodes"][1]["parameters"] = copy.deepcopy(spec["nodes"][0]["parameters"])
        fitted = fit_pipeline_graph(spec, race_frame(10), feature_schema=SCHEMA)
        self.assertEqual(fitted.fit_report["physical_fits"], 1)
        with self.assertRaisesRegex(ValueError, "strictly follow"):
            fit_graph(spec, race_frame(10), race_frame(3, 5), SCHEMA)

    def test_fundamental_market_ancestry(self):
        spec = graph_spec("market")
        spec["nodes"].append({"node_id": "market", "kind": "market_blend", "inputs": ["pool"], "output": {"market": True}})
        with self.assertRaisesRegex(ValueError, "market ancestor"):
            PipelineGraph.from_dict(spec).validate()

    def test_primary_reachability_and_disconnected_nodes_do_not_fit(self):
        disconnected = graph_spec("benter")
        invalid = copy.deepcopy(disconnected)
        invalid["primary_node_id"] = "boosted"
        with self.assertRaisesRegex(ValueError, "must be used"):
            PipelineGraph.from_dict(invalid).validate()
        with self.assertRaisesRegex(ValueError, "must be used"):
            fit_pipeline_graph(invalid, race_frame(10), feature_schema=SCHEMA)
        fitted = fit_pipeline_graph(disconnected, race_frame(10), feature_schema=SCHEMA)
        self.assertEqual(fitted.fit_report["physical_fits"], 1)
        self.assertEqual({row["node_id"] for row in fitted.fit_report["component_fits"]}, {"benter"})
        self.assertFalse(fitted.fit_report["oof_artifacts"])
        score = race_frame(2, 10)
        np.testing.assert_allclose(fitted.predict_proba(score), fitted.output.state.predict_proba(score))
        fundamental_only = graph_spec("boosted")
        fundamental_only["fundamental_node_id"] = "benter"
        separate = fit_pipeline_graph(fundamental_only, race_frame(10), feature_schema=SCHEMA)
        self.assertEqual(separate.fit_report["physical_fits"], 2)
        self.assertEqual(separate.fundamental.spec.node_id, "benter")
        joint_only = copy.deepcopy(invalid)
        joint_only["joint_node_id"] = "orders"
        joint_only["nodes"].append({"node_id": "orders", "kind": "rank_distribution", "inputs": ["boosted"],
            "output": {"kind": "joint_order", "target": "complete_finish_order", "unit": "probability"}})
        joint = fit_pipeline_graph(joint_only, race_frame(10), feature_schema=SCHEMA)
        self.assertEqual(joint.fit_report["physical_fits"], 2)
        self.assertEqual(joint.joint.parents[0].spec.node_id, "boosted")
        self.assertEqual(set(joint.predict_joint(score)), set(score.race_id.unique()))

    def test_existing_package_save_load_graph_and_cutoff(self):
        from ima.research_model_package import ResearchModelPackage, load_research_package
        from ima.research_specs import PipelineRecipe
        spec = graph_spec("market")
        spec["fundamental_node_id"] = "pool"
        spec["nodes"].append({"node_id": "market", "kind": "market_blend", "inputs": ["pool"], "output": {"market": True}})
        fitted = fit_pipeline_graph(spec, race_frame(15), feature_schema=SCHEMA)
        recipe = PipelineRecipe(schema_version=3, pipeline_graph=spec, model={"kind": "benter_conditional_logit"})
        package = ResearchModelPackage(fitted, recipe, "test-forward-graph", "local-test")
        score = race_frame(3, 15)
        with tempfile.TemporaryDirectory() as directory:
            path = package.save(Path(directory)/"graph-package")
            loaded = load_research_package(path)
            np.testing.assert_allclose(package.predict_proba(score), loaded.predict_proba(score), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(fitted.predict_fundamental_proba(score), loaded.model.predict_fundamental_proba(score), rtol=1e-10, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            fitted.predict_proba(race_frame(3))
        score["field_size"] = 5
        with self.assertRaisesRegex(ValueError, "complete fields"):
            fitted.predict_proba(score)

    def test_predictions_reuse_across_graph_families(self):
        spec = graph_spec("calibrated")
        spec["nodes"].append({"node_id": "calibrated", "kind": "calibrate", "inputs": ["pool"]})
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionStore(directory)
            first = fit_pipeline_graph(spec, race_frame(15), feature_schema=SCHEMA, prediction_store=store)
            other = copy.deepcopy(spec)
            other["graph_id"] = "alternate-log-pool"
            other["nodes"][2]["kind"] = "log_probability_pool"
            other["nodes"][3]["inputs"] = ["benter"]
            second = fit_pipeline_graph(other, race_frame(15), feature_schema=SCHEMA, prediction_store=store)
            # First graph only cached its pool, not leaf predictions. Prime a direct calibration.
            third = copy.deepcopy(other)
            third["graph_id"] = "reusing-benter-in-another-family"
            third["nodes"][2]["parameters"]["weights"] = [0.3, 0.7]
            reused = fit_pipeline_graph(third, race_frame(15), feature_schema=SCHEMA, prediction_store=store)
        self.assertGreater(first.fit_report["physical_fits"], 0)
        self.assertGreater(second.fit_report["physical_fits"], reused.fit_report["physical_fits"])
        self.assertGreater(reused.fit_report["cache_hits"], 0)

    def test_distribution_graph_and_joint_model_support(self):
        from ima.probabilistic_adapters import distribution_to_win
        spec = {"graph_id": "speed-density", "primary_node_id": "speed", "output_node_id": "win",
            "nodes": [
                {"node_id": "speed", "kind": "estimator", "parameters": {"model_kind": "ridge_regressor"},
                 "output": {"kind": "performance_distribution", "target": "speed", "unit": "log_mps"}},
                {"node_id": "win", "kind": "probabilistic_adapter", "inputs": ["speed"]}]}
        train, score = race_frame(20), race_frame(3, 20)
        train["speed"] = np.exp(2.7+0.02*train.ability+0.04*train.context)
        fitted = fit_pipeline_graph(spec, train, feature_schema=SCHEMA)
        dist = fitted.output.parents[0].predict(score)
        np.testing.assert_allclose(fitted.predict_fundamental_proba(score), distribution_to_win(dist, score.race_id))
        self.assertEqual(len(dist.row_keys), len(score))
        np.testing.assert_allclose(fitted.predict_proba(score), pickle.loads(pickle.dumps(fitted)).predict_proba(score), rtol=1e-10, atol=1e-12)

    def test_ranker_and_heterogeneous_feature_views(self):
        spec = {"graph_id": "ranker", "output_node_id": "calibrated", "nodes": [
            {"node_id": "features", "kind": "feature_view", "parameters": {"features": ["ability"]},
             "output": {"kind": "feature_frame", "target": "ranking", "unit": "features"}},
            {"node_id": "ranker", "kind": "estimator", "inputs": ["features"],
             "parameters": {"model_kind": "ridge_regressor"},
             "output": {"kind": "ranking_score", "target": "target_win", "unit": "score"}},
            {"node_id": "win", "kind": "race_normalize", "inputs": ["ranker"]},
            {"node_id": "calibrated", "kind": "calibrate", "inputs": ["win"]}]}
        fitted = fit_pipeline_graph(spec, race_frame(15), feature_schema=SCHEMA)
        score = race_frame(3, 15)
        np.testing.assert_allclose(fitted.predict_proba(score).reshape(-1, 4).sum(axis=1), 1)
        self.assertTrue(fitted.fit_report["oof_populations"])

    def test_advertised_planner_adapter_recipes_validate_and_execute(self):
        from ima.research_planner_examples import adapter_recipe_examples
        from ima.research_specs import PipelineRecipe, validate_recipe

        train, score = race_frame(30), race_frame(3, 30)
        rng = np.random.default_rng(51)
        train["speed"] = np.exp(2.7 + .02*train.ability + .04*train.context
                                + rng.normal(0, .025, len(train)))
        for example in adapter_recipe_examples():
            with self.subTest(model=example["model"]["kind"]):
                recipe = PipelineRecipe.model_validate(example)
                validate_recipe(recipe)
                spec = copy.deepcopy(recipe.pipeline_graph)
                if recipe.model.kind == "catboost_regressor":
                    spec["nodes"][0]["parameters"]["model_parameters"] = {
                        "iterations": 40, "depth": 3, "thread_count": 1}
                fitted = fit_pipeline_graph(spec, train, feature_schema=SCHEMA)
                probabilities = fitted.predict_fundamental_proba(score)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue((probabilities >= 0).all())
                np.testing.assert_allclose(probabilities.reshape(-1, 4).sum(axis=1), 1)
                np.testing.assert_allclose(probabilities,
                    pickle.loads(pickle.dumps(fitted)).predict_fundamental_proba(score))

    def test_rank_adapter_forward_fit_cache_state_and_nested_chronology(self):
        from ima.probabilistic_adapters import ranking_scores_to_probabilities
        spec = {"graph_id": "fitted-rank", "output_node_id": "win", "n_splits": 2, "nodes": [
            {"node_id": "ranker", "kind": "estimator", "parameters": {"model_kind": "ridge_regressor"},
             "output": {"kind": "ranking_score", "target": "target_win", "unit": "score"}},
            {"node_id": "win", "kind": "race_normalize", "inputs": ["ranker"]}]}
        ChronologyGuardRankEstimator.prediction_windows.clear()
        fitted = fit_pipeline_graph(spec, race_frame(24), feature_schema=SCHEMA,
            estimator_factories={"ranker": ChronologyGuardRankEstimator})
        score = race_frame(3, 24)
        before = ranking_scores_to_probabilities(fitted.output.parents[0].predict(score), score.race_id)
        after = fitted.predict_proba(score)
        self.assertGreater(fitted.output.state.temperature, 1.1)
        self.assertFalse(np.allclose(before, after))
        adapter_fits = [r for r in fitted.fit_report["component_fits"] if r["node_id"] == "win"]
        self.assertEqual(len(adapter_fits), 1)
        self.assertEqual(adapter_fits[0]["fit_scope"], "forward_oof")
        self.assertEqual(adapter_fits[0]["rows"], 22 * 4)
        np.testing.assert_allclose(after, pickle.loads(pickle.dumps(fitted)).predict_proba(score))
        nested = copy.deepcopy(spec)
        nested["output_node_id"] = "calibrated"
        nested["nodes"].append({"node_id": "calibrated", "kind": "calibrate", "inputs": ["win"]})
        fit_pipeline_graph(nested, race_frame(24), feature_schema=SCHEMA,
            estimator_factories={"ranker": ChronologyGuardRankEstimator})
        self.assertTrue(ChronologyGuardRankEstimator.prediction_windows)
        self.assertTrue(all(cutoff < start for cutoff, start in ChronologyGuardRankEstimator.prediction_windows))
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionStore(directory)
            first = fit_pipeline_graph(spec, race_frame(24), feature_schema=SCHEMA, prediction_store=store)
            warm = fit_pipeline_graph(spec, race_frame(24), feature_schema=SCHEMA, prediction_store=store)
            self.assertGreater(warm.fit_report["cache_hits"], 0)
            self.assertLess(warm.fit_report["physical_fits"], first.fit_report["physical_fits"])
            np.testing.assert_allclose(first.predict_proba(score), warm.predict_proba(score))
        invalid = copy.deepcopy(spec)
        invalid["nodes"][1]["fit_scope"] = "none"
        with self.assertRaisesRegex(ValueError, "forward_oof"):
            PipelineGraph.from_dict(invalid).validate()
        probability = graph_spec("normalized")
        probability["nodes"].append({"node_id": "normalized", "kind": "race_normalize", "inputs": ["pool"], "fit_scope": "none"})
        normalizer = fit_pipeline_graph(probability, race_frame(10), feature_schema=SCHEMA)
        self.assertIsNone(normalizer.output.state)
        self.assertEqual(normalizer.fit_report["physical_fits"], 2)

    def test_every_nested_ancestor_is_past_and_fit_report_is_json(self):
        spec = graph_spec("calibrated")
        spec["nodes"] += [
            {"node_id": "stack", "kind": "meta_estimator", "inputs": ["benter", "boosted"]},
            {"node_id": "calibrated", "kind": "calibrate", "inputs": ["stack"]}]
        fitted = fit_pipeline_graph(spec, race_frame(30), feature_schema=SCHEMA,
            estimator_factories={"benter": ChronologyGuardEstimator, "boosted": ChronologyGuardEstimator})
        np.testing.assert_allclose(fitted.predict_proba(race_frame(2, 30)), 0.25)
        self.assertIn("physical_fits", json.loads(json.dumps(fitted.fit_report)))
        with self.assertRaisesRegex(ValueError, "history"):
            fit_pipeline_graph(spec, race_frame(3), feature_schema=SCHEMA,
                estimator_factories={"benter": ChronologyGuardEstimator, "boosted": ChronologyGuardEstimator})

    def test_joint_side_endpoint_and_package_state_reload(self):
        spec = graph_spec()
        spec["joint_node_id"] = "orders"
        spec["nodes"].append({"node_id": "orders", "kind": "rank_distribution", "inputs": ["pool"],
            "output": {"kind": "joint_order", "target": "complete_finish_order", "unit": "probability"}})
        fitted = fit_pipeline_graph(spec, race_frame(10), feature_schema=SCHEMA)
        score = race_frame(2, 10)
        outputs = fitted.predict_joint(score)
        p = fitted.predict_fundamental_proba(score)
        for index, race in enumerate(score.race_id.unique()):
            np.testing.assert_allclose(outputs[race].win_probabilities(), p[index*4:index*4+4], rtol=1e-10, atol=1e-12)
        reloaded = pickle.loads(pickle.dumps(fitted))
        self.assertEqual(outputs["race-10"].order_probabilities, reloaded.predict_joint(score)["race-10"].order_probabilities)
