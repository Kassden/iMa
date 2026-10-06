from __future__ import annotations

import json
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts.predict_pool_components import CORE_SOURCE, DEPENDENCIES, run_inference, sha256


class PoolComponentInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        (self.source / "ima").mkdir(parents=True)
        for name in CORE_SOURCE:
            (self.source / "ima" / name).write_text("synthetic pinned source\n")
        implementations = {name: sha256(self.source / "ima" / name) for name in CORE_SOURCE}
        self.directory = root / "trusted" / "pool"
        self.directory.mkdir(parents=True)
        self.recipe = {"model": {"kind": "benter_conditional_logit"}, "target": {"kind": "win_probability"},
            "calibration": {"kind": "none"}, "blend": {"kind": "none"}, "pipeline_graph": {
                "graph_id": "synthetic-pool", "fundamental_node_id": "pool", "output_node_id": "pool", "nodes": [
                    {"node_id": "benter", "kind": "estimator", "parameters": {"model_kind": "benter_conditional_logit"}},
                    {"node_id": "boosted", "kind": "estimator", "parameters": {"model_kind": "boosted"}},
                    {"node_id": "pool", "kind": "weighted_probability_pool", "inputs": ["benter", "boosted"],
                     "parameters": {"weights": [0.6, 0.4]}}]}}
        self.environment = {name: version(name) for name in DEPENDENCIES}
        self.write(self.directory / "manifest.json", {"package_id": "pool-id", "target_kind": "win_probability"})
        self.write(self.directory / "recipe.json", self.recipe)
        self.write(self.directory / "feature-replay.json", {"training_cutoff": "2026-01-11",
            "implementation_hashes": implementations, "dependency_versions": self.environment})
        self.write(self.directory / "refit.json", {"fit_date_max": "2026-01-11"})
        (self.directory / "model.joblib").write_bytes(b"never unpickle synthetic artifact")
        candidate = {"package_id": "pool-id", "recipe": self.recipe, "remote_package": str(self.directory),
            "fit_date_max": "2026-01-11", "training_cutoff": "2026-01-11",
            "hashes": {p.name: sha256(p) for p in self.directory.iterdir()}}
        self.manifest_data = {"environment": self.environment, "implementation_hashes": implementations,
                              "candidates": {"pool": candidate}}
        self.manifest = root / "fresh-candidates.json"
        self.write(self.manifest, self.manifest_data)
        self.frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-14"] * 2 + ["2026-01-18"] * 2),
            "race_id": ["r1", "r1", "r2", "r2"], "race_no": [1, 1, 2, 2],
            "horse_id": ["a", "b", "c", "d"], "horse_no": [1, 2, 1, 2],
            "horse_name": ["A", "B", "C", "D"], "field_size": [2] * 4})
        self.queries = root / "query.parquet"
        self.frame.to_parquet(self.queries, index=False)
        self.components = [np.array([0.8, 0.2, 0.3, 0.7]), np.array([0.4, 0.6, 0.9, 0.1])]
        self.cached = root / "pool.csv"
        cache = self.frame.copy()
        cache["model"] = "pool"
        cache["model_probability"] = 0.6 * self.components[0] + 0.4 * self.components[1]
        cache.to_csv(self.cached, index=False)
        self.receipt = root / "original-readback.json"
        self.receipt_data = {"input_sha256": sha256(self.queries), "models": {"pool": {
            "package_id": "pool-id", "training_cutoff": "2026-01-11", "rows": 4, "races": 2,
            "output_sha256": sha256(self.cached)}}}
        self.write(self.receipt, self.receipt_data)
        self.parents = tuple(SimpleNamespace(spec=SimpleNamespace(node_id=name, kind="estimator",
            output=SimpleNamespace(kind="win_probability"), parameters={"model_kind": kind}),
            predict=Mock(return_value=values)) for name, kind, values in
            zip(("benter", "boosted"), ("benter_conditional_logit", "boosted"), self.components))
        self.graph = SimpleNamespace(fundamental=SimpleNamespace(parents=self.parents,
            spec=SimpleNamespace(node_id="pool", kind="weighted_probability_pool",
                                 output=SimpleNamespace(kind="win_probability"), parameters={"weights": [0.6, 0.4]})),
            _check_cutoff=Mock(), fit_report={"training_cutoff": "2026-01-11"})
        self.package = SimpleNamespace(manifest=lambda: SimpleNamespace(package_id="pool-id"),
            recipe=SimpleNamespace(canonical_payload=lambda: self.recipe),
            feature_context=SimpleNamespace(training_cutoff="2026-01-11"),
            model=SimpleNamespace(model=self.graph, _frame=Mock(side_effect=lambda f: f.copy())),
            _feature_frame=Mock(side_effect=lambda f: f.copy()))
        self.kwargs = dict(manifest=self.manifest, queries=self.queries, output=root / "output",
            trusted_root=root / "trusted", source_root=self.source, pool_predictions=self.cached,
            inference_readback=self.receipt)
        self.runtime = patch("scripts.predict_pool_components._load_runtime", return_value=(self.package,
            lambda values, races: (pd.Series(np.maximum(values, 1e-12)) /
                                   pd.Series(np.maximum(values, 1e-12)).groupby(np.asarray(races)).transform("sum")).to_numpy()))
        self.load = self.runtime.start()
        self.addCleanup(self.runtime.stop)

    @staticmethod
    def write(path, payload):
        path.write_text(json.dumps(payload))

    def test_extract_reconstruct_and_readback_without_fitting(self):
        result = run_inference(**self.kwargs)
        self.assertLessEqual(result["original_pool_max_abs_error"], 1e-12)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["races"], 2)
        self.package._feature_frame.assert_called_once()
        self.package.model._frame.assert_called_once()
        self.graph._check_cutoff.assert_called_once()
        for name, values in zip(("pool_benter", "pool_boosted"), self.components):
            path = self.kwargs["output"] / f"{name}.csv"
            np.testing.assert_array_equal(pd.read_csv(path).model_probability.to_numpy(), values)
            self.assertEqual(result["models"][name]["output_sha256"], sha256(path))
        self.assertTrue(json.loads((self.kwargs["output"] / "readback.json").read_text())["inference_only"])
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            run_inference(**self.kwargs)

    def test_hash_and_environment_reject_before_load(self):
        for path in (self.source / "ima/pipeline_graph.py", self.directory / "model.joblib", self.cached):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    run_inference(**self.kwargs)
                path.write_bytes(original)
        self.manifest_data["environment"]["numpy"] = "not-original"
        self.write(self.manifest, self.manifest_data)
        with self.assertRaisesRegex(ValueError, "environment"):
            run_inference(**self.kwargs)
        self.load.assert_not_called()

    def test_trusted_root_and_symlink_escape_reject_before_load(self):
        with self.assertRaisesRegex(ValueError, "trusted root"):
            run_inference(**{**self.kwargs, "trusted_root": self.source})
        original = (self.directory / "model.joblib").read_bytes()
        outside = self.source / "outside.joblib"
        outside.write_bytes(original)
        (self.directory / "model.joblib").unlink()
        (self.directory / "model.joblib").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside trusted package"):
            run_inference(**self.kwargs)
        self.load.assert_not_called()

    def test_all_candidates_cutoffs_and_fixed_recipe_reject_before_load(self):
        self.manifest_data["candidates"]["other"] = {"fit_date_max": "2026-01-14", "training_cutoff": "2026-01-11"}
        self.write(self.manifest, self.manifest_data)
        with self.assertRaisesRegex(ValueError, "precede"):
            run_inference(**self.kwargs)
        del self.manifest_data["candidates"]["other"]
        self.recipe["pipeline_graph"]["nodes"][-1]["parameters"]["weights"] = [0.5, 0.5]
        self.write(self.manifest, self.manifest_data)
        with self.assertRaisesRegex(ValueError, "0.6/0.4"):
            run_inference(**self.kwargs)
        self.load.assert_not_called()

    def test_incomplete_duplicate_and_cached_misaligned_rows_reject_before_load(self):
        for frame in (self.frame.iloc[:-1], pd.concat([self.frame.iloc[:1], self.frame.iloc[:1], self.frame.iloc[2:]])):
            with self.subTest(rows=len(frame)):
                frame.to_parquet(self.queries, index=False)
                with self.assertRaises(ValueError):
                    run_inference(**self.kwargs)
        self.frame.to_parquet(self.queries, index=False)
        cached = pd.read_csv(self.cached).iloc[::-1]
        cached.to_csv(self.cached, index=False)
        self.receipt_data["models"]["pool"]["output_sha256"] = sha256(self.cached)
        self.write(self.receipt, self.receipt_data)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            run_inference(**self.kwargs)
        self.load.assert_not_called()

    def test_invalid_component_and_pool_reconstruction_write_nothing(self):
        for values in ([np.nan] * 4, [-0.1, 1.1, 0.3, 0.7], [0.8] * 4,
                       [0.5, 0.5, 0.5, 0.5], [0.5, 0.5]):
            with self.subTest(values=values):
                self.parents[0].predict.return_value = np.array(values)
                with self.assertRaises(ValueError):
                    run_inference(**self.kwargs)
                self.assertFalse(self.kwargs["output"].exists())

    def test_loaded_parent_and_cutoff_and_transform_alignment_rejected(self):
        self.graph.fit_report["training_cutoff"] = "2026-01-14"
        with self.assertRaisesRegex(ValueError, "precede"):
            run_inference(**self.kwargs)
        self.graph.fit_report["training_cutoff"] = "2026-01-11"
        self.package._feature_frame.side_effect = lambda frame: frame.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "runner order"):
            run_inference(**self.kwargs)
        self.assertFalse(self.kwargs["output"].exists())

    def test_original_receipt_query_binding_rejects_before_load(self):
        self.receipt_data["input_sha256"] = "other-query"
        self.write(self.receipt, self.receipt_data)
        with self.assertRaisesRegex(ValueError, "input/package mismatch"):
            run_inference(**self.kwargs)
        self.load.assert_not_called()

    def test_signed_refit_cutoff_and_nested_component_cutoff_rejected(self):
        self.write(self.directory / "refit.json", {"fit_date_max": "2026-01-14"})
        self.manifest_data["candidates"]["pool"]["hashes"]["refit.json"] = sha256(self.directory / "refit.json")
        self.write(self.manifest, self.manifest_data)
        with self.assertRaisesRegex(ValueError, "Refit metadata"):
            run_inference(**self.kwargs)
        self.load.assert_not_called()
        self.write(self.directory / "refit.json", {"fit_date_max": "2026-01-11"})
        self.manifest_data["candidates"]["pool"]["hashes"]["refit.json"] = sha256(self.directory / "refit.json")
        self.write(self.manifest, self.manifest_data)
        self.graph.fit_report["component_fits"] = [{"training_cutoff": "2026-01-14"}]
        with self.assertRaisesRegex(ValueError, "Loaded graph fits"):
            run_inference(**self.kwargs)

    def test_real_graph_parent_prediction_and_race_normalization(self):
        from ima.modeling import normalize_by_race
        from ima.pipeline_graph import FittedGraphNode, FittedPipelineGraph, PipelineGraph

        graph = PipelineGraph.from_dict(self.recipe["pipeline_graph"]).validate()
        nodes = {node.node_id: node for node in graph.nodes}
        parents = tuple(FittedGraphNode(nodes[name], state=SimpleNamespace(
            predict_proba=Mock(return_value=values))) for name, values in
            zip(("benter", "boosted"), self.components))
        head = FittedGraphNode(nodes["pool"], parents=parents)
        self.package.model.model = FittedPipelineGraph(graph, head, head, {"training_cutoff": "2026-01-11"})
        self.load.return_value = (self.package, normalize_by_race)
        result = run_inference(**self.kwargs)
        self.assertLessEqual(result["original_pool_max_abs_error"], 1e-12)
        for parent in parents:
            parent.state.predict_proba.assert_called_once()


if __name__ == "__main__":
    unittest.main()
