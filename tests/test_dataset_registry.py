import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.dataset_registry import DatasetRegistry
from ima.dataset_specs import DatasetProtocolSpec, DatasetRequest
from ima.feature_definitions import FeatureRegistry
from tests.test_dataset_registry_adversarial import source_rows, event, digest


class DatasetRegistryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
