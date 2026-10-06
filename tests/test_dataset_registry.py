import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ima.dataset_registry import DatasetRegistry
from ima.dataset_registry import audit_races
from ima.rich_features import prepare_rich_runner_dataset
from ima.dataset_specs import DatasetProtocolSpec, DatasetRequest
from ima.feature_definitions import FeatureRegistry
from tests.test_dataset_registry_adversarial import source_rows, event, digest


class DatasetRegistryTests(unittest.TestCase):
    def _build_row_hash_fixture(self, root):
        snapshot = root / "snapshot"
        snapshot.mkdir()
        source = source_rows()
        source.to_parquet(snapshot / "runners.parquet", index=False)
        (snapshot / "events.jsonl").write_text(json.dumps(event("row-hash-workout")) + "\n")
        raw = snapshot / "manifest.json"
        raw.write_text(json.dumps({
            "source_policy": "HKJC-only; synthetic row-hash fixture",
            "rows": len(source), "races": int(source.race_id.nunique()),
            "files": {name: digest(snapshot / name) for name in ("runners.parquet", "events.jsonl")},
        }))
        registry = DatasetRegistry(root / "registry")
        registry.submit(DatasetRequest.model_validate({
            "request_id": "row-hash-fixture", "raw_corpus_manifest_id": digest(raw),
            "rationale": "Verify persisted row hashes", "evidence_watermark": "fixture",
            "protocol": {"min_train_races": 2, "calibration_races": 1, "score_races": 1,
                         "max_folds": 2, "final_confirmation_races": 2},
        }))
        manifest = registry.build("row-hash-fixture", source_snapshot=snapshot, raw_manifest=raw)
        return registry, manifest

    def test_signed_nan_build_hashes_actual_parquet_roundtrip(self):
        in_memory = []

        def signed_nan_history(*args, **kwargs):
            frame = prepare_rich_runner_dataset(*args, **kwargs)
            values = frame.last_result.to_numpy(copy=True)
            values[np.isnan(values)] = -np.nan
            frame["last_result"] = values
            in_memory.append(frame.last_result.copy())
            return frame

        with tempfile.TemporaryDirectory() as directory:
            with patch("ima.dataset_registry.prepare_rich_runner_dataset", side_effect=signed_nan_history):
                registry, manifest = self._build_row_hash_fixture(Path(directory))
            persisted = pd.read_parquet(registry.dataset_path(manifest["dataset_id"]) / "features.parquet")
            self.assertTrue(np.signbit(in_memory[-1].loc[in_memory[-1].isna()]).all())
            pd.testing.assert_series_equal(in_memory[-1], persisted.last_result)
            # Equal missing values can have different IEEE NaN payload/sign hashes.
            self.assertFalse(np.array_equal(
                pd.util.hash_pandas_object(in_memory[-1], index=False).to_numpy(),
                pd.util.hash_pandas_object(persisted.last_result, index=False).to_numpy()))
            expected = hashlib.sha256(pd.util.hash_pandas_object(
                persisted, index=False).to_numpy().tobytes()).hexdigest()
            self.assertEqual(expected, manifest["ordered_row_hashes_sha256"])
            self.assertEqual("verified", registry.verify(manifest["dataset_id"])["status"])

    def test_rewritten_declared_row_hash_cannot_bypass_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, manifest = self._build_row_hash_fixture(Path(directory))
            path = registry.dataset_path(manifest["dataset_id"])
            manifest["ordered_row_hashes_sha256"] = "0" * 64
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "manifest.sha256").write_text(digest(path / "manifest.json"))
            with self.assertRaisesRegex(ValueError, "ordered row hash"):
                registry.verify(manifest["dataset_id"])

    def test_rewritten_declared_ordered_keys_cannot_bypass_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, manifest = self._build_row_hash_fixture(Path(directory))
            path = registry.dataset_path(manifest["dataset_id"])
            manifest["ordered_row_keys_sha256"] = "0" * 64
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "manifest.sha256").write_text(digest(path / "manifest.json"))
            with self.assertRaisesRegex(ValueError, "ordered row key hash"):
                registry.verify(manifest["dataset_id"])

    def test_dead_heat_is_quarantined_without_fabricating_a_winner(self):
        source = source_rows()
        source.loc[(source.race_id == "r1") & (source.horse_no == 2), ["result", "finish_time"]] = [1, "1:00.00"]
        clean, exclusions = audit_races(source)
        self.assertNotIn("r1", set(clean.race_id))
        excluded = next(row for row in exclusions if row["race_id"] == "r1")
        self.assertIn("unsupported_dead_heat_single_winner_target", excluded["reasons"])
        self.assertEqual(3, excluded["rows"])
        self.assertEqual(2, source.loc[source.race_id.eq("r1"), "result"].eq(1).sum())

    def test_default_protocol_preserves_history_and_selects_latest_folds(self):
        protocol = DatasetProtocolSpec()
        self.assertEqual("latest", protocol.fold_selection)
        self.assertEqual(500, protocol.final_confirmation_races)

    def test_registered_formula_becomes_reusable_verified_dataset_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            source = source_rows()
            source.to_parquet(snapshot / "runners.parquet", index=False)
            (snapshot / "events.jsonl").write_text(json.dumps(event("prior-workout")) + "\n")
            manifest = {"source_policy": "HKJC-only; synthetic provenance fixture", "rows": len(source),
                        "races": source.race_id.nunique(), "target_only_columns": ["result", "finish_time"],
                        "files": {name: digest(snapshot / name) for name in ("runners.parquet", "events.jsonl")}}
            raw = snapshot / "manifest.json"
            raw.write_text(json.dumps(manifest))
            registry = DatasetRegistry(root / "registry")
            spec = {"request_id": "initial", "raw_corpus_manifest_id": digest(raw),
                    "rationale": "Verify registered formula lifecycle", "evidence_watermark": "fixture",
                    "protocol": {"min_train_races": 2, "calibration_races": 1, "score_races": 1,
                                 "max_folds": 2, "final_confirmation_races": 2}}
            registry.submit(DatasetRequest.model_validate(spec))
            parent = registry.build("initial", source_snapshot=snapshot, raw_manifest=raw)
            parent_path = registry.dataset_path(parent["dataset_id"])
            self.assertEqual(raw.read_bytes(), (parent_path / "raw_manifest.json").read_bytes())
            self.assertEqual(raw.read_bytes(), (parent_path / "source_manifest.json").read_bytes())
            self.assertIsInstance(json.loads((parent_path / "raw_manifest.json").read_text()), dict)
            self.assertEqual("m", parent["predictor_catalog"]["trackwork_last_distance_metres"]["unit"])
            self.assertEqual({"venue", "course", "going", "jockey_key", "trainer_key"},
                             set(parent["eligible_categorical_predictors"]))
            metadata_keys = {"unit", "dtype", "temporal_scope", "target_tainted", "available_at_column", "source_family"}
            metadata = {name: {key: value for key, value in row.items() if key in metadata_keys}
                        for name, row in parent["predictor_catalog"].items()}
            definition = {"name": "body_to_carried_ratio", "input_refs": ["declared_weight", "actual_weight"],
                          "expression_ast": {"op": "divide", "args": [{"op": "ref", "ref": "declared_weight"},
                                                                    {"op": "ref", "ref": "actual_weight"}]}, "unit": "1"}
            identifier = FeatureRegistry(registry.feature_registry).register(definition, metadata)
            registry.submit(DatasetRequest.model_validate(dict(spec, request_id="formula-successor",
                parent_dataset_id=parent["dataset_id"], feature_definition_ids=[identifier])))
            successor = registry.build("formula-successor", source_snapshot=snapshot, raw_manifest=raw)
            path = registry.dataset_path(successor["dataset_id"])
            features = pd.read_parquet(path / "features.parquet")
            np.testing.assert_allclose(features.body_to_carried_ratio, features.declared_weight / features.actual_weight)
            self.assertEqual(identifier, successor["predictor_catalog"]["body_to_carried_ratio"]["definition_id"])
            self.assertEqual(identifier, successor["predictor_catalog"]["dfs_" + identifier]["definition_id"])
            self.assertEqual("verified", registry.verify(successor["dataset_id"])["status"])
            self.assertNotEqual(parent["dataset_id"], successor["dataset_id"])
            saved = path / "raw_manifest.json"
            saved.write_text(json.dumps("trainer_id"))
            successor["files"]["raw_manifest.json"] = digest(saved)
            (path / "manifest.json").write_text(json.dumps(successor))
            (path / "manifest.sha256").write_text(digest(path / "manifest.json"))
            with self.assertRaisesRegex(ValueError, "provenance manifest must be an object"):
                registry.verify(successor["dataset_id"])


if __name__ == "__main__":
    unittest.main()
