"""Small synthetic final-refit contracts; no historical/campaign fitting."""
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ima.feature_sets import FeatureSchema
from ima.research_model_package import load_research_package
from ima.research_specs import PipelineRecipe
from scripts import refit_season_2026 as refit
from tests.test_pipeline_graph import race_frame

SCHEMA = FeatureSchema("baseline-v1", ("ability", "context"), ())


def candidates():
    result = {"candidates": {}}
    for family in refit.FAMILIES:
        kind = "benter_conditional_logit" if family == "pool" else family
        parameters = ({"l2": .1, "max_iter": 80} if kind == "benter_conditional_logit" else
                      {"max_iter": 12, "min_samples_leaf": 2} if kind == "boosted" else
                      {"l2": .1, "max_iter": 40, "quadrature_order": 16})
        recipe = {"schema_version": 3, "model": {"kind": kind, "parameters": parameters},
                  "feature_schema": "baseline-v1", "calibration": {"kind": "none"}, "blend": {"kind": "none"}}
        if family == "pool":
            recipe["pipeline_graph"] = {"graph_id": "fixed-synthetic", "primary_node_id": "benter",
                "output_node_id": "pool", "fundamental_node_id": "pool", "nodes": [
                    {"node_id": "benter", "kind": "estimator", "parameters": {
                        "model_kind": "benter_conditional_logit", "model_parameters": parameters}},
                    {"node_id": "boosted", "kind": "estimator", "parameters": {
                        "model_kind": "boosted", "model_parameters": {"max_iter": 12, "min_samples_leaf": 2}}},
                    {"node_id": "pool", "kind": "weighted_probability_pool", "inputs": ["benter", "boosted"],
                     "parameters": {"weights": [.6, .4]}}]}
        result["candidates"][family] = {"recipe": recipe, "attempt": "fixed-fixture"}
    return result


def frames():
    train = race_frame(dates=12)
    train["date"] = pd.Timestamp("2025-12-15") + pd.to_timedelta(train.index // 4, unit="D")
    query = race_frame(dates=3, start=20)
    query["date"] = pd.Timestamp("2026-01-14") + pd.to_timedelta(query.index // 4, unit="D")
    for frame in (train, query):
        frame["race_no"], frame["horse_no"], frame["field_size"] = 1, frame.index % 4 + 1, 4
        frame["horse_name"] = frame.horse_id
    return train, query


class SeasonRefitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.schema_patch = patch.dict("ima.feature_sets.FEATURE_SCHEMAS", {"baseline-v1": SCHEMA})
        self.schema_patch.start()
        self.addCleanup(self.schema_patch.stop)
        self.train, self.query = frames()
        self.frozen = candidates()

    def test_all_four_families_fit_only_prior_rows_preserve_recipe_and_reload(self):
        with threadpool_limits(limits=3):
            for family, candidate in self.frozen["candidates"].items():
                with self.subTest(family=family):
                    unchanged = copy.deepcopy(candidate)
                    package, query, probabilities, report = refit.fit_candidate(
                        self.train, self.query, candidate, code_revision="synthetic-pinned")
                    self.assertEqual(candidate, unchanged)
                    self.assertEqual(package.recipe.recipe_hash(), PipelineRecipe.model_validate(candidate["recipe"]).recipe_hash())
                    self.assertEqual(report["fit_date_max"], "2025-12-26 00:00:00")
                    self.assertEqual(package.feature_context.training_cutoff, report["fit_date_max"])
                    self.assertEqual(report["physical_fits"], 2 if family == "pool" else 1)
                    self.assertFalse(report["calibration_fitted"])
                    self.assertFalse(report["market_blend_fitted"])
                    self.assertNotIn("target_win", query)
                    self.assertNotIn("market_probability", query)
                    path = self.root / family
                    package.save(path)
                    np.testing.assert_allclose(load_research_package(path).predict_proba(query), probabilities,
                                               rtol=0, atol=1e-12)

    def test_cutoff_and_future_rows_rejected_before_any_fit(self):
        for date in ("2026-01-14", "2026-09-06"):
            bad = self.train.copy()
            bad.loc[:3, "date"] = pd.Timestamp(date)
            with self.subTest(date=date), patch("ima.modeling.RaceProbabilityModel.fit") as fitting:
                with self.assertRaisesRegex(ValueError, "strictly BEFORE"):
                    refit.fit_candidate(bad, self.query, self.frozen["candidates"]["boosted"], code_revision="fixture")
                fitting.assert_not_called()

    def test_queries_cannot_contaminate_features_or_transform_fit(self):
        recipe = PipelineRecipe.model_validate(self.frozen["candidates"]["boosted"]["recipe"])
        _, clean, _, _ = refit.prepare_inputs(self.train, self.query, recipe)
        altered = self.query.assign(target_win=1, target_probability=1., market_probability=.999, result=99,
                                    finish_seconds=.01, win_odds=999)
        _, isolated, _, _ = refit.prepare_inputs(self.train, altered, recipe)
        pd.testing.assert_frame_equal(clean, isolated)

    def test_missing_metadata_is_not_zero_filled(self):
        train, query = self.train.copy(), self.query.copy()
        train.loc[0, "context"], query.loc[0, "context"] = np.nan, np.nan
        recipe = PipelineRecipe.model_validate(self.frozen["candidates"]["boosted"]["recipe"])
        prepared, query, _, _ = refit.prepare_inputs(train, query, recipe)
        self.assertTrue(pd.isna(prepared.loc[0, "context"]))
        self.assertTrue(pd.isna(query.loc[0, "context"]))
        with self.assertRaisesRegex(ValueError, "Missing training columns"):
            refit.prepare_inputs(train.drop(columns="context"), query, recipe)

    def test_duplicates_incomplete_races_and_invalid_labels_rejected(self):
        recipe = PipelineRecipe.model_validate(self.frozen["candidates"]["boosted"]["recipe"])
        bad_frames = [pd.concat([self.train, self.train.iloc[:1]]), self.train.iloc[1:],
                      self.train.assign(target_win=0), self.train.assign(target_probability=.25),
                      self.train.assign(ability=np.inf)]
        for bad in bad_frames:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                refit.prepare_inputs(bad, self.query, recipe)

    def test_manifest_rejects_new_blending_or_tuned_pool_weights(self):
        for family, field, value in (("boosted", "blend", {"kind": "market_softmax"}),
                                     ("boosted", "train_window", "trailing_3_years")):
            bad = copy.deepcopy(self.frozen)
            bad["candidates"][family]["recipe"][field] = value
            path = self.root / "manifest.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                refit.load_candidates(path)
        bad = copy.deepcopy(self.frozen)
        bad["candidates"]["pool"]["recipe"]["pipeline_graph"]["nodes"][-1]["parameters"]["weights"] = [.5, .5]
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "weights must remain fixed"):
            refit.load_candidates(path)

    def args(self):
        paths = {name: self.root / (name + ".parquet") for name in ("training", "queries", "history")}
        self.train.to_parquet(paths["training"])
        self.query.to_parquet(paths["queries"])
        self.train.to_parquet(paths["history"])
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.frozen))
        output = self.root / "readiness"
        output.mkdir(exist_ok=True)
        return SimpleNamespace(**paths, manifest=manifest, output=output, provenance=None,
            expected_environment=None, expected_code_hashes=None, worker="boosted", code_revision="synthetic",
            workers=1, threads=3, memory_gb=100., time_limit=.1, families=["boosted"])

    def test_worker_saves_separate_packages_predictions_hashes_and_fit_dates(self):
        args = self.args()
        (args.output / "final-fits").mkdir()
        (args.output / "fresh-predictions").mkdir()
        sentinel = args.output / "old-predictions.csv"
        sentinel.write_text("untouched")
        with patch("resource.setrlimit") as limit:
            refit.worker(args)
        self.assertEqual(limit.call_args.args[1], (100 * 1024**3, 100 * 1024**3))
        report_path = args.output / "final-fits/boosted/refit.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["output_sha256"], refit.sha256(args.output / "fresh-predictions/boosted.csv"))
        self.assertEqual(report["input_sha256"], refit.sha256(args.queries))
        self.assertEqual(report["training_cutoff"], "2025-12-26 00:00:00")
        self.assertEqual(sentinel.read_text(), "untouched")
        refit.write_readback(args, {"boosted": {"status": "completed"}})
        readback = json.loads((args.output / "fresh-predictions/readback.json").read_text())
        self.assertEqual(readback["models"]["boosted"]["package_id"], report["package_id"])
        self.assertEqual(readback["models"]["boosted"]["rows"], len(self.query))
        self.assertEqual(readback["models"]["boosted"]["races"], 3)

    def test_scheduler_timeout_reaps_only_owned_process_and_writes_failure(self):
        args = self.args()
        spawned = []
        real_popen = subprocess.Popen

        def sleeping_worker(*unused, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            spawned.append(child)
            return child

        with patch.object(refit.subprocess, "Popen", side_effect=sleeping_worker):
            self.assertEqual(refit.run(args), 1)
        self.assertIsNotNone(spawned[0].poll())
        status = json.loads((args.output / "final-fits/status.json").read_text())
        self.assertEqual(status["models"]["boosted"]["status"], "timeout")
        self.assertEqual(json.loads((args.output / "fresh-predictions/readback.json").read_text())["models"], {})

    def test_fit_only_then_existing_inference_cli_uses_final_query_hash(self):
        from scripts import predict_frozen_season

        args = self.args()
        args.fit_only = True
        final_queries = args.queries
        args.queries = None
        (args.output / "final-fits").mkdir()
        with patch("resource.setrlimit"):
            refit.worker(args)
        refit.write_readback(args, {"boosted": {"status": "completed"}})
        manifest_path = args.output / "final-fits/fresh-candidates.json"
        fresh = json.loads(manifest_path.read_text())
        self.assertEqual(fresh["candidates"]["boosted"]["training_cutoff"], "2025-12-26 00:00:00")
        self.assertFalse((args.output / "fresh-predictions").exists())
        report = json.loads((args.output / "final-fits/boosted/refit.json").read_text())
        self.assertFalse(report["prediction_readback_performed"])
        command = ["predict_frozen_season.py", "--manifest", str(manifest_path), "--queries", str(final_queries),
                   "--output", str(args.output / "fresh-predictions"),
                   "--trusted-root", str(args.output / "final-fits")]
        with patch.object(sys, "argv", command), redirect_stdout(io.StringIO()):
            predict_frozen_season.main()
        readback = json.loads((args.output / "fresh-predictions/readback.json").read_text())
        self.assertEqual(readback["input_sha256"], refit.sha256(final_queries))
        self.assertEqual(readback["models"]["boosted"]["package_id"], report["package_id"])
        prediction = args.output / "fresh-predictions/boosted.csv"
        self.assertEqual(readback["models"]["boosted"]["output_sha256"], refit.sha256(prediction))

    def test_existing_fresh_directories_are_never_overwritten(self):
        args = self.args()
        path = args.output / "final-fits"
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "must not already exist"):
            refit.run(args)

    def test_snapshot_remains_unchanged_when_parent_main_queries_change(self):
        args = self.args()
        copied = refit.snapshot_inputs(args)
        original_query_hash = refit.sha256(copied.queries)
        args.queries.write_text("parent regenerated main query features")
        self.assertEqual(refit.sha256(copied.queries), original_query_hash)
        self.assertNotEqual(copied.queries, args.queries)
        self.assertEqual(copied.training.parent, args.output / "refit-inputs")
        self.assertTrue((copied.training.parent / "refit_season_2026.py").exists())
        expected = json.loads(copied.expected_code_hashes.read_text())
        self.assertIn("modeling.py", expected)
        self.assertIn("rich_features.py", expected)

    def test_actual_standalone_fit_only_launcher_and_worker(self):
        from ima.feature_sets import BASELINE_SCHEMA

        args = self.args()
        train = self.train.copy()
        rng = np.random.default_rng(42)
        for name in BASELINE_SCHEMA.numeric:
            if name not in train:
                train[name] = rng.normal(size=len(train))
        for name in BASELINE_SCHEMA.categorical:
            train[name] = "UNKNOWN"
        train.to_parquet(args.training)
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if sys.platform == "darwin":
            # macOS rejects the Linux-target address-space limit; simulate only that syscall.
            shim = self.root / "platform-shim"
            shim.mkdir()
            (shim / "sitecustomize.py").write_text("import resource\nresource.setrlimit = lambda *args: None\n")
            env["PYTHONPATH"] = str(shim) + os.pathsep + str(root)
        command = [sys.executable, "-B", str(root / "scripts/refit_season_2026.py"), "--fit-only",
                   "--manifest", str(args.manifest), "--training", str(args.training),
                   "--history", str(args.history), "--output", str(args.output),
                   "--code-revision", "synthetic-pinned", "--families", "boosted",
                   "--workers", "1", "--threads", "3", "--memory-gb", "90", "--time-limit", "30"]
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=45)
        log_path = args.output / "final-fits/boosted.log"
        self.assertEqual(result.returncode, 0, result.stderr + (log_path.read_text() if log_path.exists() else ""))
        report = json.loads((args.output / "final-fits/boosted/refit.json").read_text())
        self.assertIsNotNone(report["input_snapshot"])
        self.assertEqual(report["fit_date_max"], "2025-12-26 00:00:00")
        self.assertTrue((args.output / "final-fits/fresh-candidates.json").exists())
        self.assertFalse((args.output / "fresh-predictions").exists())


if __name__ == "__main__":
    unittest.main()
