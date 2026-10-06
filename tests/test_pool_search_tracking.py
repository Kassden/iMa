import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.log_pool_search import (
    EXPERIMENT, VARIANTS, flatten_numeric_metrics, log_bundle, prepare_bundle,
    save_json, sha256, verify_bundle,
)


class FakeClient:
    def __init__(self, fail_variant=None, fail_receipt=False):
        self.experiment = None
        self.runs = {}
        self.writes = []
        self.artifacts = []
        self.fail_variant = fail_variant
        self.fail_receipt = fail_receipt

    def get_experiment_by_name(self, name):
        return self.experiment

    def create_experiment(self, name):
        self.writes.append(("experiment", name))
        self.experiment = SimpleNamespace(experiment_id="experiment-1")
        return self.experiment.experiment_id

    def search_runs(self, experiment_ids, filter_string, **kwargs):
        identity = filter_string.split("'")[1]
        return [r for r in self.runs.values() if r.data.tags.get("ima.pool_search_identity") == identity][:kwargs["max_results"]]

    def create_run(self, experiment_id, tags):
        rid = f"run-{len(self.runs) + 1}"
        run = SimpleNamespace(info=SimpleNamespace(run_id=rid, status="RUNNING", lifecycle_stage="active"),
                              data=SimpleNamespace(tags=dict(tags), metrics={}, params={}))
        self.runs[rid] = run
        self.writes.append(("create", rid))
        return run

    def log_param(self, rid, key, value):
        self.writes.append(("param", rid))
        self.runs[rid].data.params[key] = value

    def log_metric(self, rid, key, value, **kwargs):
        if self.runs[rid].data.tags["ima.variant"] == self.fail_variant:
            raise RuntimeError("Injected logging failure")
        self.writes.append(("metric", rid))
        self.runs[rid].data.metrics[key] = value

    def log_artifact(self, rid, path, artifact_path=None):
        if self.fail_receipt and Path(path).name == "mlflow-receipt.json":
            raise RuntimeError("Injected receipt upload failure")
        self.writes.append(("artifact", rid))
        self.artifacts.append((rid, artifact_path, Path(path).name, sha256(path)))

    def set_terminated(self, rid, status):
        self.writes.append(("status", rid))
        self.runs[rid].info.status = status

    def get_run(self, rid):
        return self.runs[rid]


class PoolSearchTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.root = self.repo / "artifacts/pool-search-20261007"
        self.evaluation = self.root / "evaluation"
        self.evaluation.mkdir(parents=True)
        self.report = self.repo / "report.md"
        self.report.write_text("Descriptive paper results only.\n")
        self.bundle = self.repo / "bundle"
        self.recipe = {"pair": "original_pair", "method": "geometric", "benter_weight": 0.6, "temperature": 1.0}
        save_json(self.root / "search.json", {"grid_count": 588, "baseline_recipe": self.recipe,
                  "candidate_recipe": dict(self.recipe, temperature=0.85)})
        save_json(self.root / "selection.json", {"adopted": True, "baseline_confirmation_loss": 2.1,
                  "candidate_confirmation_loss": 2.05})
        (self.root / "search-grid.csv").write_text("configuration_id,search_race_log_loss\n" +
                                                 "".join(f"c{i},{2 + i / 10000}\n" for i in range(588)))
        (self.root / "splits.csv").write_text("split,race_id\nsearch,R1\nsearch,R2\nconfirmation,R3\ncalibration,R4\n")
        (self.root / "component-cache").mkdir()
        save_json(self.root / "component-cache/readback.json", {"models": {"component": {"rows": 50}}})
        summary = {name: {"race_log_loss": 2.2, "roi": -0.2,
                         "by_pool": {"WIN": {"profit_hkd": -100, "nsettled": 10}},
                         "enabled": True, "unknown": None} for name in VARIANTS}
        save_json(self.evaluation / "summary.json", summary)
        for name in VARIANTS:
            (self.evaluation / f"{name}-tickets.csv").write_text("race_id,stake_hkd\nR1,10\n")
        (self.evaluation / "excluded.parquet").write_text("Do not copy parquet")
        (self.evaluation / "raw").mkdir()
        (self.evaluation / "raw/page.json").write_text("Do not copy raw page")
        self.source = self.repo / "query-features.parquet"
        self.source.write_text("Provenance-only bytes, never load a model")
        self.proof()

    def proof(self):
        outputs = [self.root / name for name in ("search.json", "search-grid.csv", "splits.csv", "selection.json")]
        outputs.extend(p for p in self.evaluation.iterdir() if p.is_file() and p.suffix in {".csv", ".json"})
        if (self.root / "protocol.json").exists():
            outputs.append(self.root / "protocol.json")
        outputs.extend(p for p in (self.root / "predictions").glob("*") if p.suffix in {".csv", ".json"})
        sources = {str(self.source): sha256(self.source),
                   "artifacts/pool-search-20261007/component-cache/readback.json": sha256(self.root / "component-cache/readback.json")}
        sources.update({str(p): sha256(p) for p in (self.root / "component-cache").glob("*.csv")})
        save_json(self.root / "result-provenance.json", {
            "ready_for_report": True,
            "source_sha256": sources,
            "output_sha256": {str(p.relative_to(self.repo)): sha256(p) for p in outputs},
        })

    def prepare(self):
        return prepare_bundle(self.root, self.report, self.bundle, repo_root=self.repo)

    def test_bundle_is_compact_relocatable_and_binds_every_copied_artifact(self):
        plan = self.prepare()
        self.assertEqual(plan["parent"]["params"]["grid_count"], 588)
        self.assertEqual(plan["parent"]["params"]["search_races"], 2)
        self.assertEqual(plan["parent"]["params"]["confirmation_races"], 1)
        self.assertEqual(plan["parent"]["params"]["calibration_races"], 1)
        self.assertEqual(plan["parent"]["metrics"]["search_best_loss"], 2)
        self.assertAlmostEqual(plan["parent"]["metrics"]["confirmation_delta"], -0.05)
        self.assertEqual(plan["children"]["candidate_pool"]["params"]["recipe.temperature"], 0.85)
        self.assertFalse(any("raw" in Path(n).parts or n.endswith(".parquet") for n in plan["artifacts"]))
        self.assertEqual(plan, self.prepare())
        destination = self.repo / "relocated"
        shutil.copytree(self.bundle, destination)
        self.assertEqual(plan, verify_bundle(destination))

    def test_publication_seven_new_finished_runs_and_readback(self):
        plan = self.prepare()
        client = FakeClient()
        receipt = log_bundle(self.bundle, client)
        self.assertEqual(len(client.runs), 7)
        self.assertTrue(all(r.info.status == "FINISHED" for r in client.runs.values()))
        self.assertEqual(receipt, json.loads((self.bundle / "mlflow-receipt.json").read_text()))
        self.assertTrue(receipt["readback_verified"])
        self.assertEqual(client.writes[0], ("experiment", EXPERIMENT))
        parent = client.get_run(receipt["parent_run_id"])
        self.assertEqual(parent.data.metrics["adopted"], 1)
        self.assertEqual(parent.data.tags["ima.pool_search_identity"], plan["identity"])
        for name, rid in receipt["children"].items():
            child = client.get_run(rid)
            self.assertEqual(child.data.tags["mlflow.parentRunId"], parent.info.run_id)
            self.assertEqual(child.data.tags["ima.paper_only"], "true")
            self.assertEqual(child.data.tags["ima.non_agentic"], "true")
            self.assertEqual(child.data.metrics["by_pool.WIN.profit_hkd"], -100)
            self.assertNotIn("enabled", child.data.metrics)
            self.assertEqual(child.data.tags["ima.selected"], "true" if name == "candidate_pool" else "false")
        self.assertEqual(client.get_run(receipt["children"]["candidate_pool_market_blend_hindsight"]).data.params["calibration_application"], "market_blend_only")
        self.assertEqual(client.get_run(receipt["children"]["market_calibrated_hindsight"]).data.params["calibration_application"], "temperature_only")
        self.assertEqual(client.get_run(receipt["children"]["market_final_odds_hindsight"]).data.params["calibration_application"], "none")
        uploaded = {name for rid, _, name, _ in client.artifacts if rid == parent.info.run_id}
        self.assertTrue({Path(n).name for n in plan["artifacts"]} <= uploaded)
        self.assertIn("plan.json", uploaded)

    def test_finished_identity_is_reused_without_existing_mutations(self):
        self.prepare()
        client = FakeClient()
        first = log_bundle(self.bundle, client)
        before = list(client.writes)
        second = log_bundle(self.bundle, client)
        self.assertEqual(client.writes, before)
        self.assertEqual(first["parent_run_id"], second["parent_run_id"])
        self.assertEqual(first["children"], second["children"])
        self.assertTrue(second["reused"])

    def test_collision_incomplete_wrong_metrics_or_parent_never_mutates_existing(self):
        self.prepare()
        for mutation in ("partial", "failed", "metric", "parent", "duplicate"):
            with self.subTest(mutation=mutation):
                client = FakeClient()
                receipt = log_bundle(self.bundle, client)
                child = client.get_run(receipt["children"]["candidate_pool"])
                if mutation == "partial":
                    del client.runs[child.info.run_id]
                elif mutation == "failed":
                    child.info.status = "FAILED"
                elif mutation == "metric":
                    child.data.metrics["race_log_loss"] = 99
                elif mutation == "parent":
                    child.data.tags["mlflow.parentRunId"] = "other"
                else:
                    client.create_run("experiment-1", dict(child.data.tags))
                before = list(client.writes)
                with self.assertRaises(ValueError):
                    log_bundle(self.bundle, client)
                self.assertEqual(client.writes, before)

    def test_failure_marks_only_new_runs_failed_and_retry_refuses_collision(self):
        self.prepare()
        client = FakeClient(fail_variant="candidate_pool")
        unrelated = client.create_run("other", {"ima.variant": "untouched"})
        unrelated.info.status = "FINISHED"
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            log_bundle(self.bundle, client)
        self.assertEqual(unrelated.info.status, "FINISHED")
        self.assertTrue(all(r.info.status == "FAILED" for r in client.runs.values() if r is not unrelated))
        before = list(client.writes)
        with self.assertRaises(ValueError):
            log_bundle(self.bundle, client)
        self.assertEqual(client.writes, before)

    def test_bundle_and_plan_tampering_rejected_before_experiment_creation(self):
        self.prepare()
        path = self.bundle / "search-grid.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"tampering")
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            log_bundle(self.bundle, client)
        self.assertEqual(client.writes, [])
        path.write_bytes(original)
        plan = json.loads((self.bundle / "plan.json").read_text())
        plan["parent"]["metrics"]["search_best_loss"] = 0
        save_json(self.bundle / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "identity"):
            log_bundle(self.bundle, client)
        self.assertEqual(client.writes, [])

    def test_provenance_ready_hashes_grid_summary_and_size_are_gated(self):
        proof_path = self.root / "result-provenance.json"
        original = proof_path.read_text()
        proof = json.loads(original)
        proof["ready_for_report"] = False
        save_json(proof_path, proof)
        with self.assertRaisesRegex(ValueError, "ready_for_report"):
            self.prepare()
        proof_path.write_text(original)
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.proof()
        (self.root / "search-grid.csv").write_text("search_race_log_loss\n2\n")
        self.proof()
        with self.assertRaisesRegex(ValueError, "588"):
            self.prepare()

    def test_external_component_receipt_path_can_be_provenance_only_source(self):
        original = self.root / "component-cache/readback.json"
        receipt = self.repo / "cache/readback.json"
        receipt.parent.mkdir()
        original.rename(receipt)
        search = json.loads((self.root / "search.json").read_text())
        search["component_cache_readback_path"] = str(receipt)
        save_json(self.root / "search.json", search)
        proof = json.loads((self.root / "result-provenance.json").read_text())
        proof["source_sha256"] = {str(self.source): sha256(self.source), str(receipt): sha256(receipt)}
        proof["output_sha256"][str((self.root / "search.json").relative_to(self.repo))] = sha256(self.root / "search.json")
        save_json(self.root / "result-provenance.json", proof)
        self.prepare()
        self.assertEqual(sha256(receipt), sha256(self.bundle / "component-cache/readback.json"))

    def test_finite_metric_flattening_skips_bools_none_and_nonfinite(self):
        self.assertEqual(flatten_numeric_metrics({"loss": 2, "ok": True, "missing": None,
            "inf": float("inf"), "nan": float("nan"), "by_pool": {"WIN": {"roi": -0.1}}}),
            {"loss": 2.0, "by_pool.WIN.roi": -0.1})

    def test_actual_role_schema_and_explicit_component_receipt_take_precedence(self):
        receipt = self.repo / "borrowed-cache/readback.json"
        receipt.parent.mkdir()
        save_json(receipt, {"borrowed_cache": True})
        search = json.loads((self.root / "search.json").read_text())
        search.pop("candidate_recipe")
        search.update(selected_recipe={"pair": "standalone_pair", "method": "geometric",
                                      "benter_weight": 0.15, "temperature": 0.85},
                      component_receipt=str(receipt), search_best_loss=2.0,
                      splits={"search": {"races": 2}, "confirmation": {"races": 1},
                              "calibration": {"races": 1}},
                      policy="already-inspected exploratory evaluation, not untouched confirmation",
                      no_base_model_training=True)
        save_json(self.root / "search.json", search)
        save_json(self.root / "selection.json", {
            "adopted": True, "confirmation_baseline_loss": 2.1,
            "confirmation_candidate_loss": 2.05, "selected_recipe": search["selected_recipe"],
        })
        (self.root / "splits.csv").write_text(
            "race_id,horse_no,horse_id,date,role\nR1,1,H1,2026-01-14,search\n"
            "R1,2,H2,2026-01-14,search\nR2,1,H1,2026-01-18,search\n"
            "R3,1,H1,2026-05-03,confirmation\nR4,1,H1,2026-06-10,calibration\n")
        self.proof()
        proof_path = self.root / "result-provenance.json"
        proof = json.loads(proof_path.read_text())
        proof["source_sha256"][str(receipt)] = sha256(receipt)
        save_json(proof_path, proof)
        plan = self.prepare()
        self.assertEqual(sha256(receipt), sha256(self.bundle / "component-cache/readback.json"))
        self.assertEqual(plan["parent"]["params"]["search_races"], 2)
        params = plan["children"]["candidate_pool"]["params"]
        self.assertEqual(params["recipe.pair"], "standalone_pair")
        self.assertEqual(params["recipe.benter_weight"], 0.15)
        self.assertEqual(params["recipe.temperature"], 0.85)
        self.assertIn("not untouched", plan["parent"]["params"]["policy"])

    def test_overlapping_roles_and_inconsistent_declared_counts_are_rejected(self):
        (self.root / "splits.csv").write_text(
            "race_id,role\nR1,search\nR1,confirmation\nR2,calibration\n")
        self.proof()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.prepare()
        (self.root / "splits.csv").write_text(
            "race_id,role\nR1,search\nR2,confirmation\nR3,calibration\n")
        search = json.loads((self.root / "search.json").read_text())
        search["splits"] = {name: {"races": 2} for name in ("search", "confirmation", "calibration")}
        save_json(self.root / "search.json", search)
        self.proof()
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            self.prepare()

    def test_failed_receipt_upload_does_not_claim_success(self):
        self.prepare()
        client = FakeClient(fail_receipt=True)
        with self.assertRaisesRegex(RuntimeError, "receipt upload"):
            log_bundle(self.bundle, client)
        self.assertTrue(all(r.info.status == "FAILED" for r in client.runs.values()))
        receipt = json.loads((self.bundle / "mlflow-receipt.json").read_text())
        self.assertFalse(receipt["readback_verified"])
        self.assertEqual(receipt["status"], "FAILED")

    def test_unbound_files_and_large_artifacts_are_rejected(self):
        self.prepare()
        (self.bundle / "extra.json").write_text("{}")
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "Unbound"):
            log_bundle(self.bundle, client)
        self.assertEqual(client.writes, [])
        with (self.evaluation / "oversized.csv").open("wb") as stream:
            stream.truncate(10 * 1024 * 1024)
        self.proof()
        with self.assertRaisesRegex(ValueError, "below 10MB"):
            self.prepare()

    def optional_evidence(self):
        save_json(self.root / "protocol.json", {"grid_count": 588})
        for directory, names in (("component-cache", ("pool_benter", "pool_boosted")),
                                 ("predictions", ("baseline_pool", "candidate_pool"))):
            folder = self.root / directory
            folder.mkdir(exist_ok=True)
            models = {}
            for name in names:
                path = folder / f"{name}.csv"
                path.write_text("race_id,horse_no,probability\nR1,1,0.5\n")
                models[name] = {"output_sha256": sha256(path), "rows": 1}
            save_json(folder / "readback.json", {"models": models})
        self.proof()

    def test_optional_reproducibility_evidence_is_copied_hashbound_and_uploaded(self):
        self.optional_evidence()
        plan = self.prepare()
        expected = {"protocol.json", "predictions/readback.json", "predictions/baseline_pool.csv",
                    "predictions/candidate_pool.csv", "component-cache/pool_benter.csv",
                    "component-cache/pool_boosted.csv"}
        self.assertTrue(expected <= plan["artifacts"].keys())
        for name in expected:
            self.assertEqual(plan["artifacts"][name], sha256(self.root / name))
        client = FakeClient()
        receipt = log_bundle(self.bundle, client)
        uploaded = {f"{directory}/{name}" if directory else name
                    for rid, directory, name, _ in client.artifacts if rid == receipt["parent_run_id"]}
        self.assertTrue(expected <= uploaded)

    def test_optional_prediction_csv_requires_declared_and_provenance_hashes_before_copy(self):
        self.optional_evidence()
        receipt = self.root / "component-cache/readback.json"
        original = receipt.read_text()
        payload = json.loads(original)
        payload["models"]["pool_benter"]["output_sha256"] = "0" * 64
        save_json(receipt, payload)
        self.proof()
        with self.assertRaisesRegex(ValueError, "declared output SHA256"):
            self.prepare()
        self.assertFalse(self.bundle.exists())
        receipt.write_text(original)
        self.proof()
        proof_path = self.root / "result-provenance.json"
        proof = json.loads(proof_path.read_text())
        del proof["source_sha256"][str(self.root / "component-cache/pool_benter.csv")]
        save_json(proof_path, proof)
        with self.assertRaisesRegex(ValueError, "source_sha256 provenance"):
            self.prepare()
        self.assertFalse(self.bundle.exists())


if __name__ == "__main__":
    unittest.main()
