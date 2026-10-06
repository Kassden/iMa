from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.log_season_readiness import (EXPERIMENT, FAMILIES, POOLS, VARIANTS,
                                        log_evaluation, model_identity, prepare_logging, sha256)


class SeasonTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evaluation = self.root / "evaluation"
        self.predictions = self.root / "fresh-predictions"
        self.evaluation.mkdir()
        self.predictions.mkdir()
        self.code = self.root / "evaluate.py"
        self.report = self.root / "report.md"
        for path in (self.code, self.report, self.root / "query-features.parquet",
                     self.root / "query-metadata.parquet"):
            path.write_text("synthetic bytes")
        self.write(self.root / "preparation.json", {"query_rows": 7292})
        candidates, models = {}, {}
        for family in FAMILIES:
            output = self.predictions / f"{family}.csv"
            output.write_text("probability\n1\n")
            candidates[family] = {"package_id": family, "fit_date_max": "2026-01-11",
                                  "recipe": {"model": {"kind": family}, "feature_schema": "synthetic-v1",
                                             "target": {"kind": "win_probability"}}}
            models[family] = {"package_id": family, "output_sha256": sha256(output),
                              "probability_sum_min": 1, "probability_sum_max": 1}
            (self.evaluation / f"{family}-tomorrow-runners.csv").write_text("forecast\n")
        self.manifest = self.root / "fresh-candidates.json"
        self.write(self.manifest, {"candidates": candidates})
        self.write(self.predictions / "readback.json", {
            "input_sha256": sha256(self.root / "query-features.parquet"), "models": models})
        self.write(self.evaluation / "provenance.json", {
            "query_features_sha256": sha256(self.root / "query-features.parquet"),
            "inference_readback_sha256": sha256(self.predictions / "readback.json"),
            "evaluation_code_sha256": sha256(self.code)})
        self.write(self.evaluation / "calibration.json", {
            f: {"temperature": 1.2, "blend": {"market_weight": 0.3}}
            for f in (*FAMILIES, "market_final_odds_hindsight")})
        self.summary = {}
        for variant in VARIANTS:
            pools = {pool: {"stake_hkd": 770 if pool == "QPL" else 780,
                            "gross_hkd": 385 if pool == "QPL" else 390,
                            "profit_hkd": -385 if pool == "QPL" else -390,
                            "roi": -0.5, "nsettled": 77 if pool == "QPL" else 78,
                            "nmissing": 1 if pool == "QPL" else 0} for pool in POOLS}
            self.summary[variant] = {"stake_hkd": 6230, "gross_hkd": 3115,
                                     "profit_hkd": -3115, "roi": -0.5, "nsettled": 623,
                                     "nmissing": 1, "by_pool": pools, "price_basis": "paper",
                                     "meeting_cluster_bootstrap_roi_interval_95": [-0.8, -0.2]}
            for suffix in ("tickets", "bankroll", "meetings", "runners"):
                (self.evaluation / f"{variant}-{suffix}.csv").write_text("synthetic ledger\n")
        self.write(self.evaluation / "summary.json", self.summary)
        (self.evaluation / "tomorrow-combinations.csv").write_text("forecast\n")
        raw = self.root / "official" / "raw"
        raw.mkdir(parents=True)
        (raw / "never-upload.json").write_text("raw archive")

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def plan(self):
        return prepare_logging(self.root, self.evaluation, self.predictions,
                               self.manifest, self.report, self.code)

    def test_compact_plan_and_exact_pool_metrics(self):
        for name in ("data-freeze.json", "odds-unit-semantics.json", "odds-quotes.json"):
            self.write(self.root / "official" / name, {"synthetic": True})
        plan = self.plan()
        self.assertEqual(len(plan["children"]), 10)
        metrics = plan["children"]["boosted"]["metrics"]
        self.assertEqual(metrics["stake_hkd"], 6230)
        self.assertEqual(metrics["by_pool.QPL.stake_hkd"], 770)
        self.assertEqual(metrics["roi_ci95_lower"], -0.8)
        self.assertFalse(any("raw" in p.parts for p, _ in plan["artifacts"]))
        evidence = {path.name for path, dest in plan["artifacts"] if dest == "evidence"}
        self.assertTrue({"data-freeze.json", "odds-unit-semantics.json", "odds-quotes.json"} <= evidence)

    def test_optional_forecast_hash_bound_and_directories_rejected(self):
        (self.evaluation / "tomorrow-market-blends.csv").write_text("synthetic blend\n")
        (self.evaluation / "tomorrow-market-blend-combinations.csv").write_text("synthetic combos\n")
        self.write(self.evaluation / "tomorrow-market-blend-provenance.json", {
            "ready_for_report": True,
            "source_sha256": {str(p): sha256(p) for p in (self.root / "query-features.parquet",
                self.evaluation / "summary.json", self.evaluation / "calibration.json",
                self.evaluation / "provenance.json", self.code, self.predictions / "readback.json")},
            "output_sha256": {name: sha256(self.evaluation / name) for name in
                              ("tomorrow-market-blends.csv", "tomorrow-market-blend-combinations.csv")}})
        forecast = self.root / "ranked-forecast.csv"
        forecast.write_text("synthetic ranked quote\n")
        proof = self.root / "ranked-provenance.json"
        self.write(proof, {"query_features_sha256": sha256(self.root / "query-features.parquet"),
                           "output_sha256": {forecast.name: sha256(forecast)}})
        plan = prepare_logging(self.root, self.evaluation, self.predictions, self.manifest,
                               self.report, self.code, forecast_artifacts=(forecast,),
                               forecast_provenance=proof)
        self.assertIn("forecast/ranked-forecast.csv", plan["inventory"])
        self.assertIn("forecast/tomorrow-market-blends.csv", plan["inventory"])
        forecast.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            log_evaluation(plan, None)
        with self.assertRaisesRegex(ValueError, "compact"):
            prepare_logging(self.root, self.evaluation, self.predictions, self.manifest,
                            self.report, self.code, forecast_artifacts=(self.root / "official",))

    def test_ranked_report_companion_is_hash_bound_without_claiming_forecast_proof(self):
        ranked = self.root / "ranked.csv"
        ranked.write_text("report companion\n")
        plan = prepare_logging(self.root, self.evaluation, self.predictions, self.manifest,
                               self.report, self.code, report_artifacts=(ranked,))
        self.assertIn("report/ranked.csv", plan["inventory"])
        ranked.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            log_evaluation(plan, None)

    def test_query_readback_prediction_and_code_tampering_rejected(self):
        for path in (self.root / "query-features.parquet", self.predictions / "readback.json",
                     self.predictions / "boosted.csv", self.code):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    self.plan()
                path.write_bytes(original)

    def test_wrong_stake_pool_totals_and_nonfinite_rejected(self):
        for key, value in (("stake_hkd", 6240), ("gross_hkd", 1), ("race_log_loss", float("nan"))):
            with self.subTest(key=key):
                original = self.summary["boosted"].copy()
                self.summary["boosted"][key] = value
                self.write(self.evaluation / "summary.json", self.summary)
                with self.assertRaises(ValueError):
                    self.plan()
                self.summary["boosted"] = original

    def test_real_mlflow_parent_children_artifacts_and_idempotence(self):
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=f"sqlite:///{self.root / 'tracking.db'}")
        plan = self.plan()
        result = log_evaluation(plan, client, artifact_location=(self.root / "mlartifacts").as_uri())
        self.assertEqual(result, log_evaluation(plan, client))
        experiment = client.get_experiment_by_name(EXPERIMENT)
        runs = client.search_runs([experiment.experiment_id])
        self.assertEqual(len(runs), 11)
        child = client.get_run(result["children"]["boosted"])
        self.assertEqual(child.data.tags["mlflow.parentRunId"], result["parent_run_id"])
        self.assertEqual(child.data.tags["ima.non_agentic"], "true")
        self.assertEqual(child.data.metrics["stake_hkd"], 6230)
        self.assertEqual(child.data.metrics["by_pool.QPL.nmissing"], 1)
        self.assertEqual(child.data.params["package_id"], "boosted")
        self.assertEqual(child.data.params["model_kind"], "boosted")
        self.assertEqual(child.data.tags["ima.model_kind"], "boosted")
        self.assertEqual(child.data.params["feature_schema"], "synthetic-v1")
        self.assertEqual(child.data.params["target_kind"], "win_probability")
        self.assertEqual(child.data.params["fit_date_max"], "2026-01-11")
        self.assertEqual(child.data.params["calibration_application"], "none")
        blend = client.get_run(result["children"]["boosted_market_blend_hindsight"])
        self.assertEqual(blend.data.params["calibration_application"], "market_blend_only")
        calibrated = client.get_run(result["children"]["market_calibrated_hindsight"])
        self.assertEqual(calibrated.data.params["calibration_application"], "temperature_only")
        market = client.get_run(result["children"]["market_final_odds_hindsight"])
        self.assertEqual(market.data.tags["ima.model_kind"], "inverse_final_win_odds")
        paths = {a.path for a in client.list_artifacts(result["parent_run_id"], "source")}
        self.assertIn("source/datafreeze.json", paths)
        self.assertNotIn("official/raw", paths)
        # A completed identity is never rewritten; an incomplete identity is refused.
        client.set_terminated(child.info.run_id, "FAILED")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            log_evaluation(plan, client)
        self.assertEqual(len(client.search_runs([experiment.experiment_id])), 11)

    def test_changed_artifact_rejected_before_tracking_access(self):
        plan = self.plan()
        self.report.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            log_evaluation(plan, None)

    def test_wrong_package_identity_and_changed_unuploaded_query_rejected(self):
        candidate = json.loads(self.manifest.read_text())
        candidate["candidates"]["boosted"]["package_id"] = "other-package"
        self.write(self.manifest, candidate)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.plan()
        candidate["candidates"]["boosted"]["package_id"] = "boosted"
        self.write(self.manifest, candidate)
        plan = self.plan()
        (self.root / "query-features.parquet").write_text("new input")
        with self.assertRaisesRegex(ValueError, "source changed"):
            log_evaluation(plan, None)

    def test_combined_forecast_query_provenance_rejected_when_stale(self):
        forecast = self.root / "combined.csv"
        forecast.write_text("forecast\n")
        proof = self.root / "combined-provenance.json"
        self.write(proof, {"query_features_sha256": "stale",
                           "output_sha256": {forecast.name: sha256(forecast)}})
        with self.assertRaisesRegex(ValueError, "query hash"):
            prepare_logging(self.root, self.evaluation, self.predictions, self.manifest,
                            self.report, self.code, forecast_artifacts=(forecast,),
                            forecast_provenance=proof)
        self.write(proof, {"query_features_sha256": sha256(self.root / "query-features.parquet"),
                           "output_sha256": {forecast.name: "stale"}})
        with self.assertRaisesRegex(ValueError, "output hash"):
            prepare_logging(self.root, self.evaluation, self.predictions, self.manifest,
                            self.report, self.code, forecast_artifacts=(forecast,),
                            forecast_provenance=proof)

    def test_head_identity_not_primary_recipe_kind(self):
        recipe = {"model": {"kind": "benter_conditional_logit"}, "pipeline_graph": {
            "fundamental_node_id": "pool", "output_node_id": "pool", "nodes": [
                {"node_id": "benter", "kind": "estimator", "parameters": {"model_kind": "benter_conditional_logit"}},
                {"node_id": "boosted", "kind": "estimator", "parameters": {"model_kind": "boosted"}},
                {"node_id": "pool", "kind": "weighted_probability_pool", "parameters": {"weights": [0.6, 0.4]}}]}}
        identity = model_identity(recipe)
        self.assertEqual(identity["model_kind"], "weighted_probability_pool")
        self.assertEqual(identity["recipe_model_kind"], "benter_conditional_logit")
        self.assertEqual(json.loads(identity["head_parameters"])["weights"], [0.6, 0.4])
        self.assertEqual(model_identity({"model": {"kind": "gaussian_probit", "parameters": {
            "heteroscedastic": False}}})["distribution_variance"], "homoscedastic")


if __name__ == "__main__":
    unittest.main()
