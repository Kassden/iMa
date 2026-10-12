"""Real registered-parquet lifecycle, leakage, tracking and consumer-link tests."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ima.dataset_registry import (
    DatasetRegistry, _digest, _keys, _ordered_row_hashes_sha256, _write_json, file_sha256,
)
from ima.dataset_specs import DatasetRequest
from ima.feature_definitions import FeatureDefinition, FeatureRegistry
from ima.feature_lifecycle import (
    FeatureLifecycle, build_feature_candidate, catalog, deterministic_candidate_definitions,
    record_model_outcome, retry_tracking,
)
from scripts.feature_lifecycle import main
from tests.test_dataset_registry_adversarial import source_rows


def ratio(**changes):
    return FeatureDefinition.model_validate({
        "name": "body_to_carried", "input_refs": ["declared_weight", "actual_weight"],
        "expression_ast": {"op": "divide", "args": [
            {"op": "ref", "ref": "declared_weight"}, {"op": "ref", "ref": "actual_weight"}]},
        "unit": "1", **changes,
    })


class FeatureLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        root = Path(cls.fixture.name)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        rows = source_rows()
        rows.to_parquet(snapshot / "runners.parquet", index=False)
        (snapshot / "events.jsonl").write_text("")
        raw = snapshot / "manifest.json"
        _write_json(raw, {"source_policy": "HKJC-only; lifecycle fixture", "rows": len(rows),
            "races": int(rows.race_id.nunique()), "target_only_columns": ["result", "finish_time", "target_win"],
            "files": {name: file_sha256(snapshot / name) for name in ("runners.parquet", "events.jsonl")}})
        registry = DatasetRegistry(root / "registry")
        registry.submit(DatasetRequest(request_id="parent", raw_corpus_manifest_id=file_sha256(raw),
            rationale="Real registry lifecycle fixture", evidence_watermark="fixture",
            protocol={"min_train_races": 2, "calibration_races": 1, "score_races": 1,
                      "max_folds": 2, "final_confirmation_races": 2}))
        cls.original = registry.build("parent", source_snapshot=snapshot, raw_manifest=raw)
        cls.original_path = registry.dataset_path(cls.original["dataset_id"])

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "registry"
        self.registry = DatasetRegistry(self.root)
        self.parent = deepcopy(self.original)
        self.parent_path = self.registry.dataset_path(self.parent["dataset_id"])
        shutil.copytree(self.original_path, self.parent_path)
        self.parent.update(features_path=str(self.parent_path / "features.parquet"),
                           protocol_path=str(self.parent_path / "protocol.json"))
        self._seal()
        self.lifecycle = FeatureLifecycle(self.root)

    def _seal(self):
        _write_json(self.parent_path / "manifest.json", self.parent)
        (self.parent_path / "manifest.sha256").write_text(file_sha256(self.parent_path / "manifest.json") + "\n")

    def _with_inputs(self, values, metadata):
        frame = pd.read_parquet(self.parent_path / "features.parquet")
        for name, value in values.items():
            frame[name] = value
        self.parent["predictor_catalog"].update(metadata)
        self.parent["feature_catalog"] = self.parent["predictor_catalog"]
        self.parent["identity"]["predictor_catalog_sha256"] = _digest(self.parent["predictor_catalog"])
        self.parent["identity"]["fixture_input_sha256"] = _ordered_row_hashes_sha256(frame)
        self.parent["dataset_id"] = "dataset-" + _digest(self.parent["identity"])
        new_path = self.registry.dataset_path(self.parent["dataset_id"])
        shutil.copytree(self.parent_path, new_path)
        self.parent_path = new_path
        frame.to_parquet(self.parent_path / "features.parquet", index=False)
        persisted = pd.read_parquet(self.parent_path / "features.parquet")
        self.parent.update(features_path=str(new_path / "features.parquet"), protocol_path=str(new_path / "protocol.json"),
                           ordered_row_keys_sha256=_digest(_keys(persisted)),
                           ordered_row_hashes_sha256=_ordered_row_hashes_sha256(persisted))
        self.parent["files"]["features.parquet"] = file_sha256(new_path / "features.parquet")
        self._seal()
        self.registry.verify(self.parent["dataset_id"])

    def build(self, definitions=None, **kwargs):
        return build_feature_candidate(self.parent_path / "features.parquet", self.parent_path / "manifest.json",
            self.root, definitions=definitions if definitions is not None else [ratio()], **kwargs)

    def submit(self, definitions=None, **kwargs):
        return self.lifecycle.submit(self.parent_path / "features.parquet", self.parent_path / "manifest.json",
            definitions=definitions if definitions is not None else [ratio()], **kwargs)

    def test_real_successor_preserves_parent_rows_values_protocol_and_confirmation(self):
        before = {p.name: file_sha256(p) for p in self.parent_path.iterdir()}
        with patch.object(DatasetRegistry, "build", side_effect=AssertionError("raw rebuild forbidden")):
            successor = self.build()
        verified = self.registry.verify(successor["dataset_id"])
        frame = pd.read_parquet(verified["features_path"])
        parent_frame = pd.read_parquet(self.parent_path / "features.parquet")
        pd.testing.assert_frame_equal(frame[parent_frame.columns], parent_frame, check_exact=True)
        np.testing.assert_allclose(frame[ratio().feature_id], parent_frame.declared_weight / parent_frame.actual_weight)
        self.assertEqual([ratio().content_id()], verified["feature_definition_ids"])
        self.assertEqual(self.parent["dataset_id"], verified["parent_dataset_id"])
        self.assertEqual("verified", verified["status"])
        self.assertNotEqual(self.parent["dataset_id"], verified["dataset_id"])
        for key in ("rows", "races", "comparison_contract_id", "comparison_contract", "evaluation_population_id",
                    "ordered_row_keys_sha256", "target_eligibility", "event_policy_id"):
            self.assertEqual(self.parent[key], verified[key])
        for name, digest in self.parent["files"].items():
            if name != "features.parquet":
                self.assertEqual(digest, file_sha256(self.registry.dataset_path(verified["dataset_id"]) / name))
        self.assertEqual(before, {p.name: file_sha256(p) for p in self.parent_path.iterdir()})
        self.assertEqual(ratio().content_id(), FeatureRegistry(self.registry.feature_registry).get(ratio().content_id()).content_id())

    def test_deterministic_proposals_reject_unsafe_or_incomplete_catalog(self):
        candidates = deterministic_candidate_definitions(self.parent["predictor_catalog"])
        other = deterministic_candidate_definitions(dict(reversed(list(self.parent["predictor_catalog"].items()))))
        self.assertTrue(candidates)
        self.assertEqual(candidates, other)
        self.assertLessEqual(len(candidates), 4)
        for unsafe in ({"temporal_scope": "current_outcome"}, {"target_tainted": True},
                       {"unit": "mystery_unit"}, {"temporal_scope": "historical"}, {"eligible": False}):
            altered = deepcopy(self.parent["predictor_catalog"])
            altered["actual_weight"].update(unsafe)
            refs = {ref for d in deterministic_candidate_definitions(altered) for ref in d.input_refs}
            self.assertNotIn("actual_weight", refs)
        altered = deepcopy(self.parent["predictor_catalog"])
        altered["declared_weight"].pop("unit")
        self.assertFalse(any("declared_weight" in d.input_refs for d in deterministic_candidate_definitions(altered)))

    def test_finite_missing_and_division_domain_coverage(self):
        rows = len(pd.read_parquet(self.parent_path / "features.parquet"))
        values = np.full(rows, 120.)
        values[:3] = [0., np.inf, np.nan]
        values[3:] += np.arange(rows - 3)
        self._with_inputs({"actual_weight": values}, {})
        successor = self.build()
        frame = pd.read_parquet(successor["features_path"])
        self.assertTrue(frame[ratio().feature_id].iloc[:3].isna().all())
        self.assertFalse(np.isinf(frame[ratio().feature_id]).any())
        stats = successor["feature_build"]["diagnostics"][ratio().content_id()]
        self.assertEqual(3, stats["missing"])
        self.assertEqual(rows - 3, stats["finite"])
        self.assertEqual(0, stats["nonfinite"])

    def test_historical_availability_strict_cutoff_and_missing_timestamp(self):
        frame = pd.read_parquet(self.parent_path / "features.parquet")
        observed = pd.to_datetime(frame.date, utc=True) - pd.Timedelta(days=1)
        observed.iloc[0] = pd.to_datetime(frame.date.iloc[0], utc=True)
        observed.iloc[1] += pd.Timedelta(days=2)
        observed.iloc[2] = pd.NaT
        meta = {"unit": "m/s", "dtype": "float64", "temporal_scope": "historical",
                "target_tainted": False, "available_at_column": "speed_observed_at", "eligible": True}
        self._with_inputs({"historical_measurement": np.arange(len(frame), dtype=float) + 20,
                           "speed_observed_at": observed}, {"historical_measurement": meta})
        definition = FeatureDefinition(name="speed_magnitude", unit="m/s", input_refs=("historical_measurement",),
            expression_ast={"op": "abs", "args": [{"op": "ref", "ref": "historical_measurement"}]})
        successor = self.build([definition])
        output = pd.read_parquet(successor["features_path"])[definition.feature_id]
        self.assertTrue(output.iloc[:3].isna().all())
        np.testing.assert_equal(output.iloc[3:].to_numpy(), np.arange(3, len(frame)) + 20)
        candidates = deterministic_candidate_definitions(self.parent["predictor_catalog"])
        self.assertTrue(candidates)

    def test_hk_midnight_masks_same_local_date_events_before_utc_midnight(self):
        frame = pd.read_parquet(self.parent_path / "features.parquet")
        cutoff = pd.to_datetime(frame.date).dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC")
        observed = cutoff - pd.Timedelta(hours=1)
        observed.iloc[0] = cutoff.iloc[0] + pd.Timedelta(hours=2)
        observed.iloc[1] = cutoff.iloc[1]
        meta = {"unit": "m/s", "dtype": "float64", "temporal_scope": "historical", "target_tainted": False,
                "available_at_column": "speed_observed_at", "eligible": True}
        self._with_inputs({"historical_measurement": np.arange(len(frame), dtype=float) + 10,
                           "speed_observed_at": observed}, {"historical_measurement": meta})
        definition = FeatureDefinition(name="speed_magnitude", unit="m/s", input_refs=("historical_measurement",),
            expression_ast={"op": "abs", "args": [{"op": "ref", "ref": "historical_measurement"}]})
        successor = self.build([definition])
        values = pd.read_parquet(successor["features_path"])[definition.feature_id]
        self.assertTrue(values.iloc[:2].isna().all())
        self.assertTrue(values.iloc[2:].notna().all())

    def test_aware_race_timestamps_preserve_actual_time_and_mixed_offsets(self):
        from ima.feature_lifecycle import _race_cutoffs

        cutoffs = _race_cutoffs(pd.Series(["2024-01-01", "2024-01-01T02:00:00+08:00", "2024-01-01T00:00:00Z"]))
        self.assertEqual(pd.Timestamp("2023-12-31T16:00:00Z"), cutoffs.iloc[0])
        self.assertEqual(pd.Timestamp("2023-12-31T18:00:00Z"), cutoffs.iloc[1])
        self.assertEqual(pd.Timestamp("2024-01-01T00:00:00Z"), cutoffs.iloc[2])
        frame = pd.read_parquet(self.parent_path / "features.parquet")
        aware = pd.to_datetime(frame.date).dt.tz_localize("Asia/Hong_Kong") + pd.Timedelta(hours=12)
        observed = aware.dt.tz_convert("UTC") - pd.Timedelta(hours=1)
        observed.iloc[0] += pd.Timedelta(hours=2)
        meta = {"unit": "1", "dtype": "float64", "temporal_scope": "historical", "target_tainted": False,
                "available_at_column": "support_observed_at", "eligible": True}
        self._with_inputs({"date": aware, "support_observed_at": observed, "historical_support": np.arange(len(frame), dtype=float)},
                          {"historical_support": meta})
        definition = FeatureDefinition(name="support_magnitude", input_refs=("historical_support",),
            expression_ast={"op": "abs", "args": [{"op": "ref", "ref": "historical_support"}]})
        successor = self.build([definition])
        values = pd.read_parquet(successor["features_path"])[definition.feature_id]
        self.assertTrue(pd.isna(values.iloc[0]))
        self.assertTrue(values.iloc[1:].notna().all())

    def test_deterministic_speed_cv_uses_pre_race_history_and_masks_late_observations(self):
        frame = pd.read_parquet(self.parent_path / "features.parquet")
        cutoff = pd.to_datetime(frame.date).dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC")
        observed = cutoff - pd.Timedelta(hours=1)
        observed.iloc[0] = cutoff.iloc[0] + pd.Timedelta(hours=1)
        meta = {"unit": "m/s", "dtype": "float64", "temporal_scope": "historical", "target_tainted": False,
                "available_at_column": "speed_observed_at", "eligible": True}
        self._with_inputs({"speed_observed_at": observed,
                           "past_race_speed_std_5_mps": np.arange(len(frame), dtype=float) + 1,
                           "past_race_speed_mean_5_mps": np.full(len(frame), 15.)},
                          {"past_race_speed_std_5_mps": meta, "past_race_speed_mean_5_mps": meta})
        definitions = deterministic_candidate_definitions(self.parent["predictor_catalog"], limit=8)
        cv = next(d for d in definitions if d.name == "past_race_speed_cv")
        successor = self.build([cv])
        values = pd.read_parquet(successor["features_path"])[cv.feature_id]
        self.assertTrue(pd.isna(values.iloc[0]))
        np.testing.assert_allclose(values.iloc[1:].to_numpy(), (np.arange(1, len(frame)) + 1) / 15.)

    def test_reject_all_missing_and_constant_candidates_without_publication(self):
        for values in ([np.nan] * self.parent["rows"], [120.] * self.parent["rows"]):
            self._with_inputs({"actual_weight": values, "declared_weight": [1000.] * len(values)}, {})
            before = set((self.root / "datasets").iterdir())
            with self.assertRaisesRegex(ValueError, "coverage/variation"):
                self.build()
            self.assertEqual(before, set((self.root / "datasets").iterdir()))
        self.assertEqual(2, catalog(self.root)["counts"]["rejected"])

    def test_reject_labels_unknown_units_future_and_unverified_historical(self):
        for changes in ({"unit": "unknown"}, {"temporal_scope": "future"}, {"target_tainted": True},
                        {"temporal_scope": "historical", "available_at_column": None}):
            self._with_inputs({}, {"actual_weight": {**self.parent["predictor_catalog"]["actual_weight"], **changes}})
            with self.assertRaises(ValueError):
                self.build()
        self._with_inputs({}, {"result": {"unit": "1", "dtype": "int64", "temporal_scope": "pre_race",
                                         "target_tainted": False, "eligible": True}})
        definition = FeatureDefinition(name="copied_label", input_refs=("result",),
            expression_ast={"op": "ref", "ref": "result"})
        with self.assertRaises(ValueError):
            self.build([definition])

    def test_caller_cannot_override_catalog_availability(self):
        supplied = {"actual_weight": {"unit": "kg", "dtype": "float64", "temporal_scope": "pre_race"}}
        with self.assertRaisesRegex(ValueError, "differs from verified catalog"):
            self.build(metadata=supplied)

    def test_units_scale_ast_attack_and_cycles(self):
        with self.assertRaises(ValueError):
            self.build([ratio(unit="lb")])
        with self.assertRaises(ValueError):
            FeatureDefinition.model_validate({**ratio().model_dump(), "expression_ast": {"op": "eval", "ref": "result"}})
        a = FeatureDefinition(name="a", input_refs=("b",), expression_ast={"op": "ref", "ref": "b"})
        b = FeatureDefinition(name="b", input_refs=("a",), expression_ast={"op": "ref", "ref": "a"})
        with self.assertRaisesRegex(ValueError, "cyclic"):
            self.build([a, b])

    def test_parent_and_successor_hash_integrity(self):
        successor = self.build()
        with (self.parent_path / "features.parquet").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        path = self.registry.dataset_path(successor["dataset_id"]) / "features.parquet"
        with path.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.lifecycle.build(successor["feature_build_id"])
        self.assertEqual(1, catalog(self.root)["counts"]["rejected"])

    def test_repeated_build_and_recovered_publication_reuse_one_artifact(self):
        first = self.build()
        second = self.build([ratio(hypothesis="new narrative")])
        self.assertEqual(first["dataset_id"], second["dataset_id"])
        self.assertEqual(first["feature_build_id"], second["feature_build_id"])
        receipt = self.lifecycle.receipt(first["feature_build_id"])
        receipt["status"] = "pending"
        self.lifecycle._save(receipt)
        third = self.lifecycle.build(receipt["build_id"])
        self.assertEqual(first["dataset_id"], third["dataset_id"])
        self.assertEqual(1, catalog(self.root)["counts"]["ready"])

    def test_chained_planner_definitions_registered_and_reused(self):
        first = ratio()
        second = FeatureDefinition(name="log_body_ratio", input_refs=(first.name,),
            parent_definition_ids=(first.content_id(),),
            expression_ast={"op": "log", "args": [{"op": "ref", "ref": first.name}]})
        successor = self.build([second, first])
        frame = pd.read_parquet(successor["features_path"])
        np.testing.assert_allclose(frame[second.feature_id], np.log(frame[first.feature_id]))
        self.assertEqual(sorted([first.content_id(), second.content_id()]), successor["feature_definition_ids"])

    def test_tracking_outage_is_pending_then_published_once_with_real_mlflow(self):
        from mlflow.tracking import MlflowClient

        uri = "sqlite:///" + str(Path(self.temporary.name) / "tracking.db")
        client = MlflowClient(tracking_uri=uri)
        client.create_experiment("ima-feature-builds", artifact_location=str(Path(self.temporary.name) / "mlflow-artifacts"))
        with patch("mlflow.tracking.MlflowClient", side_effect=ConnectionError("offline")):
            successor = self.build(tracking_uri=uri)
        receipt = self.lifecycle.receipt(successor["feature_build_id"])
        self.assertEqual("ready", receipt["status"])
        self.assertEqual("pending", receipt["tracking"]["status"])
        tracking = retry_tracking(self.root, receipt["build_id"], tracking_uri=uri)
        self.assertEqual("published", tracking["status"], tracking.get("error"))
        self.assertTrue(tracking["readback_verified"])
        again = retry_tracking(self.root, receipt["build_id"], tracking_uri=uri)
        self.assertEqual(tracking["run_id"], again["run_id"])
        client = MlflowClient(tracking_uri=uri)
        runs = client.search_runs([tracking["experiment_id"]])
        self.assertEqual(1, len(runs))
        self.assertEqual("feature_build", runs[0].data.tags["stage"])
        self.assertEqual(successor["dataset_id"], runs[0].data.tags["ima.dataset_id"])

    def test_tracking_lost_create_response_recovers_existing_run(self):
        from mlflow.tracking import MlflowClient

        uri = "sqlite:///" + str(Path(self.temporary.name) / "lost-response.db")
        client = MlflowClient(tracking_uri=uri)
        client.create_experiment("ima-feature-builds", artifact_location=str(Path(self.temporary.name) / "mlflow-artifacts"))
        create = client.create_run

        def lost_response(*args, **kwargs):
            create(*args, **kwargs)
            raise ConnectionError("created but response lost")

        with patch.object(client, "create_run", side_effect=lost_response), patch("mlflow.tracking.MlflowClient", return_value=client):
            successor = self.build(tracking_uri=uri)
        tracking = retry_tracking(self.root, successor["feature_build_id"], tracking_uri=uri)
        self.assertEqual("published", tracking["status"], tracking.get("error"))
        self.assertEqual(1, len(client.search_runs([tracking["experiment_id"]])))

    def outcome(self, successor, **changes):
        comparison = successor["comparison_contract"]
        return {"attempt_id": "model-attempt", "model_id": "probit", "dataset_id": successor["dataset_id"],
            "definition_ids": successor["feature_definition_ids"], "score": 2.1,
            "comparison_contract": {"comparison_contract_id": successor["comparison_contract_id"],
                "target_contract": "win_probability", "evaluation_population_id": successor["evaluation_population_id"],
                "protocol_id": comparison["protocol_id"], "score_population_sha256": comparison["score_population_sha256"],
                "probability_basis": comparison["probability_basis"], "metric": "race_log_loss", "direction": "minimize"}, **changes}

    def test_reused_candidate_two_models_and_paired_contract_only_feedback(self):
        successor = self.build()
        outcome = self.outcome(successor)
        baseline = self.outcome(successor, dataset_id=self.parent["dataset_id"], definition_ids=[],
                                attempt_id="baseline-attempt", model_id="baseline", score=2.3)
        first = record_model_outcome(self.root, successor["feature_build_id"], outcome, baseline=baseline)
        again = record_model_outcome(self.root, successor["feature_build_id"], outcome, baseline=baseline)
        self.assertEqual(first, again)
        self.assertTrue(first["paired"])
        self.assertAlmostEqual(0.2, first["improvement"])
        other = self.outcome(successor, model_id="logit", attempt_id="logit-attempt", score=2.2)
        unmatched = deepcopy(baseline)
        unmatched["comparison_contract"]["metric"] = "brier"
        row = record_model_outcome(self.root, successor["feature_build_id"], other, baseline=unmatched)
        self.assertFalse(row["paired"])
        self.assertNotIn("improvement", row)
        feedback = catalog(self.root)
        self.assertEqual(1, feedback["counts"]["ready"])
        observations = feedback["builds"][0]["comparisons"][0]["model_outcomes"]
        self.assertEqual({"probit", "logit"}, {row["outcome"]["model_id"] for row in observations})
        self.assertNotIn("global_score", feedback)

    def test_model_feedback_rejects_wrong_population_dataset_unused_features_and_nan(self):
        successor = self.build()
        invalid = [self.outcome(successor, dataset_id=self.parent["dataset_id"]),
                   self.outcome(successor, definition_ids=[]), self.outcome(successor, score=np.nan)]
        mismatch = self.outcome(successor)
        mismatch["comparison_contract"]["score_population_sha256"] = "another_population"
        invalid.append(mismatch)
        for value in invalid:
            with self.assertRaises(ValueError):
                record_model_outcome(self.root, successor["feature_build_id"], value)

    def test_feedback_attempt_identity_is_immutable_and_baseline_must_be_registered(self):
        successor = self.build()
        outcome = self.outcome(successor)
        record_model_outcome(self.root, successor["feature_build_id"], outcome)
        with self.assertRaisesRegex(ValueError, "already binds"):
            record_model_outcome(self.root, successor["feature_build_id"], {**outcome, "score": 1.8})
        baseline = self.outcome(successor, dataset_id="dataset-" + "0" * 64, definition_ids=[], attempt_id="baseline")
        with self.assertRaises((ValueError, OSError)):
            record_model_outcome(self.root, successor["feature_build_id"],
                                 self.outcome(successor, attempt_id="another-attempt"), baseline=baseline)

    def test_actual_executor_population_and_protocol_hash_control_pairing(self):
        successor = self.build()
        outcome = self.outcome(successor)
        outcome["comparison_contract"].update(
            evaluation_population_id="evaluated-" + "a" * 64, score_population_sha256="a" * 64,
            declared_evaluation_population_id=successor["evaluation_population_id"], protocol_hash="b" * 64,
            target_unit="1", target_parameters={})
        baseline = deepcopy(outcome)
        baseline.update(dataset_id=self.parent["dataset_id"], definition_ids=[], attempt_id="baseline", score=2.3)
        paired = record_model_outcome(self.root, successor["feature_build_id"], outcome, baseline=baseline)
        self.assertTrue(paired["paired"])
        different = deepcopy(outcome)
        different["attempt_id"] = "another-model"
        baseline["comparison_contract"]["protocol_hash"] = "c" * 64
        unpaired = record_model_outcome(self.root, successor["feature_build_id"], different, baseline=baseline)
        self.assertFalse(unpaired["paired"])
        different["attempt_id"] = "invalid-model"
        different["comparison_contract"].pop("declared_evaluation_population_id")
        with self.assertRaisesRegex(ValueError, "Actual scored population"):
            record_model_outcome(self.root, successor["feature_build_id"], different)

    def test_changed_postrace_labels_do_not_change_formula_values(self):
        first = self.build()
        expected = pd.read_parquet(first["features_path"])[ratio().feature_id].copy()
        self._with_inputs({"result": np.arange(self.parent["rows"]) + 50,
                           "target_win": np.full(self.parent["rows"], 999.)}, {})
        second = self.build()
        actual = pd.read_parquet(second["features_path"])[ratio().feature_id]
        pd.testing.assert_series_equal(expected, actual)

    def test_missing_historical_timestamp_binding_and_insufficient_coverage(self):
        frame = pd.read_parquet(self.parent_path / "features.parquet")
        meta = {"unit": "1", "dtype": "float64", "temporal_scope": "historical", "target_tainted": False,
                "available_at_column": "absent_observed_at", "eligible": True}
        self._with_inputs({"history_support": np.arange(len(frame), dtype=float)}, {"history_support": meta})
        definition = FeatureDefinition(name="history_support_magnitude", input_refs=("history_support",),
            expression_ast={"op": "abs", "args": [{"op": "ref", "ref": "history_support"}]})
        with self.assertRaisesRegex(ValueError, "missing_cutoff"):
            self.build([definition])
        values = np.full(len(frame), np.nan)
        values[-2:] = [120., 121.]
        self._with_inputs({"actual_weight": values}, {})
        with self.assertRaisesRegex(ValueError, "coverage/variation"):
            self.build(min_coverage=0.9)
        rejected = [row for row in catalog(self.root)["builds"] if row["diagnostics"]]
        self.assertTrue(rejected)
        self.assertEqual(2, rejected[0]["diagnostics"][ratio().content_id()]["finite"])

    def test_pending_catalog_and_cli_on_registered_parquet(self):
        pending = self.submit()
        self.assertEqual(1, catalog(self.root)["counts"]["pending"])
        cli_args = ["--registry", str(self.root), "propose", "--parent-features", str(self.parent_path / "features.parquet"), "--limit", "1"]
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(0, main(cli_args))
        definitions = json.loads(stream.getvalue())
        self.assertEqual(1, len(definitions))
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(0, main(["--registry", str(self.root), "build", "--parent-features", str(self.parent_path / "features.parquet"), "--limit", "1"]))
        built = json.loads(stream.getvalue())
        self.assertEqual("verified", self.registry.verify(built["dataset_id"])["status"])
        self.assertEqual("pending", self.lifecycle.receipt(pending["build_id"])["status"])

    def test_catalog_exposes_exact_column_refs_and_prioritizes_untasted_latest(self):
        first = self.build()
        record_model_outcome(self.root, first["feature_build_id"], self.outcome(first))
        second = self.build([ratio(name="second_weight_ratio")])
        third = self.build([ratio(name="third_weight_ratio")])
        self.submit([ratio(name="pending_weight_ratio")])
        feedback = catalog(self.root, limit=1)
        self.assertEqual({"pending": 1, "ready": 3, "rejected": 0}, feedback["counts"])
        self.assertEqual(3, feedback["omitted_builds"])
        entry = feedback["builds"][0]
        self.assertEqual(third["feature_build_id"], entry["build_id"])
        self.assertEqual([ratio().feature_id], entry["feature_ids"])
        self.assertEqual("third_weight_ratio", entry["definitions"][0]["name"])
        self.assertEqual("divide", entry["definitions"][0]["expression_ast"]["op"])
        self.assertEqual([ratio().content_id()], entry["formula_report"]["definition_ids"])
        self.assertNotEqual(second["feature_build_id"], entry["build_id"])

    def test_successor_of_successor_preserves_quarantine_and_old_features(self):
        first = self.build()
        definition = FeatureDefinition(name="log_reused_body_ratio", input_refs=(ratio().feature_id,),
            expression_ast={"op": "log", "args": [{"op": "ref", "ref": ratio().feature_id}]})
        second = build_feature_candidate(first["features_path"], Path(first["features_path"]).parent / "manifest.json",
                                         self.root, definitions=[definition])
        before = pd.read_parquet(first["features_path"])
        after = pd.read_parquet(second["features_path"])
        pd.testing.assert_frame_equal(before, after[before.columns], check_exact=True)
        self.assertEqual(first["comparison_contract_id"], second["comparison_contract_id"])
        self.assertEqual(first["files"]["confirmation_keys.json"], second["files"]["confirmation_keys.json"])
        self.assertEqual("verified", self.registry.verify(second["dataset_id"])["status"])

    def test_receipt_definition_tampering_and_disk_retry(self):
        receipt = self.submit()
        with patch.object(self.lifecycle, "_materialize", side_effect=OSError("temporary disk unavailable")):
            with self.assertRaises(OSError):
                self.lifecycle.build(receipt["build_id"])
        self.assertEqual("pending", self.lifecycle.receipt(receipt["build_id"])["status"])
        self.lifecycle.build(receipt["build_id"])
        saved = self.lifecycle.receipt(receipt["build_id"])
        saved["definitions"][0]["input_refs"] = ["result"]
        _write_json(self.lifecycle._path(receipt["build_id"]), saved)
        with self.assertRaisesRegex(ValueError, "definitions changed"):
            self.lifecycle.receipt(receipt["build_id"])


if __name__ == "__main__":
    unittest.main()
