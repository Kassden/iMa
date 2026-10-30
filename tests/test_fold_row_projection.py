import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ima import research_executor as executor
from ima.research_executor import RecipeExecutionRequest, prepare_recipe_folds
from ima.research_specs import PipelineRecipe
from tests.test_research_executor_v6 import dataset


class FoldRowProjectionTests(unittest.TestCase):
    def _parquet_request(self, root, *, recipe=None, row_group_size=9):
        root.mkdir(parents=True, exist_ok=True)
        csv_path, frame = dataset(root)
        parquet_path = root / "features.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, parquet_path, row_group_size=row_group_size)
        csv_path.unlink()
        request = RecipeExecutionRequest("projection", "fold-projection", 0,
            recipe or PipelineRecipe(schema_version=3, blend={"kind": "none"}),
            parquet_path, root / "trials" / "projection",
            {"min_train_races": 12, "calibration_races": 3, "score_races": 3, "max_folds": 1},
            code_revision="projection-test", environment_hash="projection-test")
        return request, frame

    def test_cold_preparation_matches_full_read_and_pushes_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fast, _ = self._parquet_request(root / "fast")
            legacy, _ = self._parquet_request(root / "legacy")
            shutil.copyfile(fast.dataset_path, legacy.dataset_path)
            fast_manifest = fast.dataset_path.parent / "manifest.json"
            legacy_manifest = legacy.dataset_path.parent / "manifest.json"
            shutil.copyfile(fast_manifest, legacy_manifest)

            reads = []
            original_read = pd.read_parquet

            def observed_read(*args, **kwargs):
                value = original_read(*args, **kwargs)
                reads.append((kwargs.get("columns"), kwargs.get("filters"), len(value)))
                return value

            with patch.object(pd, "read_parquet", side_effect=observed_read):
                projected = prepare_recipe_folds(fast)
            with patch.object(executor, "_supports_direct_parquet_fold_pushdown", return_value=False):
                full = prepare_recipe_folds(legacy)

            self.assertEqual(list(full), list(projected))
            for fold_id in full:
                self.assertEqual(full[fold_id].artifact_id, projected[fold_id].artifact_id)
                full_arrays = full[fold_id].load_arrays()
                projected_arrays = projected[fold_id].load_arrays()
                self.assertEqual(set(full_arrays), set(projected_arrays))
                for name in full_arrays:
                    np.testing.assert_array_equal(full_arrays[name], projected_arrays[name])
                    self.assertFalse(projected_arrays[name].flags.writeable)
                self.assertEqual(full[fold_id].manifest()["row_keys"], projected[fold_id].manifest()["row_keys"])
            self.assertGreaterEqual(len(reads), 2)
            population = next(item for item in reads if item[1] is None)
            filtered = [item for item in reads if item[1] is not None]
            self.assertIn("target_win", population[0])
            self.assertNotIn("horse_rating", population[0])
            self.assertTrue(filtered)
            self.assertTrue(all("horse_rating" in cols for cols, _, _ in filtered))
            self.assertTrue(all(rows < population[2] for _, _, rows in filtered))

    def test_two_layer_delta_fold_route_projects_actual_positions_and_matches_full(self):
        from ima.dataset_delta import digest, read_feature_table, write_feature_delta
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fast, source_frame = self._parquet_request(root / "source", row_group_size=9)
            # An invalid label in a late, unselected race must still be seen by the
            # full narrow population pass and excluded exactly as in the full route.
            source_frame.loc[source_frame.race_id == "R025", "target_win"] = np.nan
            pq.write_table(pa.Table.from_pandas(source_frame, preserve_index=False),
                           fast.dataset_path, row_group_size=9)
            registry = root / "registry"
            parent = registry / "datasets" / "dataset-parent"
            parent.mkdir(parents=True)
            shutil.copyfile(fast.dataset_path, parent / "features.parquet")
            parent_manifest = {"dataset_id": "dataset-parent", "features_path": str(parent / "features.parquet"),
                "files": {"features.parquet": digest(parent / "features.parquet")},
                "predictor_catalog": json.loads((fast.dataset_path.parent / "manifest.json").read_text()).get(
                    "predictor_catalog", {})}
            (parent / "manifest.json").write_text(json.dumps(parent_manifest))
            metadata = {"unit": "1", "dtype": "float64", "temporal_scope": "pre_race",
                "target_tainted": False, "eligible": True, "source_family": "validated_formula",
                "availability_proof": "derived from verified fixture values"}
            child = registry / "datasets" / "dataset-child"
            first = source_frame.assign(delta_unused=source_frame.horse_rating / 10)
            write_feature_delta(parent / "manifest.json", child, first, registry_root=registry,
                columns=["delta_unused"], catalog={"delta_unused": metadata})
            child_manifest = {"dataset_id": "dataset-child", "features_path": str(child / "features.delta.json"),
                "files": {"features.delta.json": digest(child / "features.delta.json")},
                "predictor_catalog": parent_manifest["predictor_catalog"]}
            (child / "manifest.json").write_text(json.dumps(child_manifest))
            grandchild = registry / "datasets" / "dataset-grandchild"
            full = first.assign(delta_second=first.delta_unused + 1)
            write_feature_delta(child / "manifest.json", grandchild, full, registry_root=registry,
                columns=["delta_second"], catalog={"delta_second": metadata})
            source = grandchild / "features.delta.json"
            delta_request = replace(fast, dataset_path=source, output_dir=root / "trials" / "delta",
                attempt_id="delta-fold", proposal_id="delta-fold")
            full_request = replace(fast, output_dir=root / "trials" / "full",
                attempt_id="full-fold", proposal_id="full-fold")
            self.assertTrue(executor._supports_direct_parquet_fold_pushdown(delta_request))
            observed_positions = []
            original_reader = read_feature_table
            def observed(path, **kwargs):
                if Path(path) == source and kwargs.get("positions") is not None:
                    observed_positions.extend(kwargs["positions"])
                return original_reader(path, **kwargs)
            with patch("ima.dataset_delta.read_feature_table", side_effect=observed):
                projected = prepare_recipe_folds(delta_request)
            with patch.object(executor, "_supports_direct_parquet_fold_pushdown", return_value=False):
                full_artifacts = prepare_recipe_folds(full_request)
            self.assertTrue(observed_positions)
            self.assertLess(len(observed_positions), len(source_frame))
            self.assertEqual(list(full_artifacts), list(projected))
            for fold_id in full_artifacts:
                expected, actual = full_artifacts[fold_id], projected[fold_id]
                self.assertEqual(expected.manifest()["row_keys"], actual.manifest()["row_keys"])
                self.assertNotIn('R025', {json.loads(key)[0] for group in actual.manifest()["row_keys"].values()
                                          for key in group})
                # Cache identities remain source-hash bound; compare the prepared
                # artifact contents and ordered keys across the logical population.
                a, b = expected.load_arrays(), actual.load_arrays()
                self.assertEqual(set(a), set(b))
                for name in a:
                    np.testing.assert_array_equal(a[name], b[name])

    def test_delta_fold_route_falls_back_for_single_row_group_payloads(self):
        from ima.dataset_delta import digest, write_feature_delta
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request, frame = self._parquet_request(root / "source", row_group_size=10_000)
            registry = root / "registry"
            parent = registry / "datasets" / "parent"
            parent.mkdir(parents=True)
            shutil.copyfile(request.dataset_path, parent / "features.parquet")
            (parent / "manifest.json").write_text(json.dumps({"dataset_id": "parent",
                "features_path": str(parent / "features.parquet"),
                "files": {"features.parquet": digest(parent / "features.parquet")},
                "predictor_catalog": {}}))
            metadata = {"unit": "1", "dtype": "float64", "temporal_scope": "pre_race",
                "target_tainted": False, "eligible": True, "source_family": "validated_formula",
                "availability_proof": "fixture"}
            child = registry / "datasets" / "child"
            write_feature_delta(parent / "manifest.json", child, frame.assign(extra=1.),
                registry_root=registry, columns=["extra"], catalog={"extra": metadata})
            delta_request = replace(request, dataset_path=child / "features.delta.json")
            self.assertFalse(executor._supports_direct_parquet_fold_pushdown(delta_request))

    def test_single_row_group_uses_full_read_and_matches_cached_fold_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            single_group, _ = self._parquet_request(root / "single", row_group_size=10_000)
            full_read, _ = self._parquet_request(root / "full", row_group_size=10_000)
            shutil.copyfile(single_group.dataset_path, full_read.dataset_path)
            shutil.copyfile(single_group.dataset_path.parent / "manifest.json",
                            full_read.dataset_path.parent / "manifest.json")
            self.assertEqual(pq.ParquetFile(single_group.dataset_path).metadata.num_row_groups, 1)

            reads = []
            original_read = pd.read_parquet

            def observed_read(*args, **kwargs):
                value = original_read(*args, **kwargs)
                reads.append((kwargs.get("filters"), len(value)))
                return value

            with patch.object(pd, "read_parquet", side_effect=observed_read):
                projected_request_artifacts = prepare_recipe_folds(single_group)
            with patch.object(executor, "_supports_direct_parquet_fold_pushdown", return_value=False):
                full_request_artifacts = prepare_recipe_folds(full_read)

            self.assertTrue(reads)
            self.assertTrue(all(filters is None for filters, _ in reads))
            self.assertEqual(list(full_request_artifacts), list(projected_request_artifacts))
            for fold_id in full_request_artifacts:
                full_artifact = full_request_artifacts[fold_id]
                actual_artifact = projected_request_artifacts[fold_id]
                self.assertEqual(full_artifact.artifact_id, actual_artifact.artifact_id)
                self.assertEqual(full_artifact.manifest()["row_keys"], actual_artifact.manifest()["row_keys"])
                full_arrays = full_artifact.load_arrays()
                actual_arrays = actual_artifact.load_arrays()
                self.assertEqual(set(full_arrays), set(actual_arrays))
                for name in full_arrays:
                    np.testing.assert_array_equal(full_arrays[name], actual_arrays[name])
                    self.assertFalse(actual_arrays[name].flags.writeable)

    def test_malformed_parquet_footer_fails_closed_without_fallback_read(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _ = self._parquet_request(Path(directory))
            request.dataset_path.write_bytes(request.dataset_path.read_bytes()[:-8])
            with patch.object(pd, "read_parquet") as read:
                with self.assertRaises(Exception):
                    prepare_recipe_folds(request)
            read.assert_not_called()

    def test_invalid_unselected_population_data_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            request, frame = self._parquet_request(Path(directory))
            frame["date"] = frame["date"].astype(str)
            frame.loc[frame.race_id == "R025", "date"] = "not-a-date"
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), request.dataset_path,
                           row_group_size=9)
            with self.assertRaises((ValueError, TypeError)):
                prepare_recipe_folds(request)

    def test_missing_filtered_fold_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _ = self._parquet_request(Path(directory))
            original = pd.read_parquet

            def omit_one(*args, **kwargs):
                frame = original(*args, **kwargs)
                if kwargs.get("filters") is not None and len(frame):
                    return frame.iloc[:-1].copy()
                return frame

            with patch.object(pd, "read_parquet", side_effect=omit_one):
                with self.assertRaisesRegex(ValueError, "population/order"):
                    prepare_recipe_folds(request)

    def test_source_hash_is_rechecked_after_filtered_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _ = self._parquet_request(Path(directory))
            original_hash = executor._hash_file
            calls = 0

            def change_at_final_check(path):
                nonlocal calls
                value = original_hash(path)
                if Path(path) == request.dataset_path:
                    calls += 1
                    if calls == 2:
                        return "changed-during-preparation"
                return value

            with patch.object(executor, "_hash_file", side_effect=change_at_final_check):
                with self.assertRaisesRegex(ValueError, "Dataset changed during fold preparation"):
                    prepare_recipe_folds(request)

    def test_manifest_hash_rejects_tampering_in_unselected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            request, frame = self._parquet_request(Path(directory))
            manifest_path = request.dataset_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"] = {request.dataset_path.name:
                hashlib.sha256(request.dataset_path.read_bytes()).hexdigest()}
            manifest_path.write_text(json.dumps(manifest))
            frame.loc[frame.race_id == "R025", "custom_form"] += 1000
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), request.dataset_path,
                           row_group_size=9)
            with self.assertRaisesRegex(ValueError, "Dataset manifest checksum mismatch"):
                prepare_recipe_folds(request)

    def test_adaptive_and_graph_recipes_stay_on_full_read_path(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _ = self._parquet_request(Path(directory))
            for recipe in (
                request.recipe.model_copy(update={"feature_discovery": {"schema_version": 2,
                    "entities": ["horse"], "measurements": ["speed_mps"], "windows_days": [None]}}),
                request.recipe.model_copy(update={"transforms": [{"kind": "signed_log1p",
                    "parameters": {"columns": ["horse_rating"]}}]}),
                request.recipe.model_copy(update={"feature_definitions": [{"name": "rating_copy",
                    "expression_ast": {"op": "ref", "ref": "horse_rating"},
                    "input_refs": ["horse_rating"], "unit": "1"}]}),
                request.recipe.model_copy(update={"performance_distribution": {"kind": "shared_residual"}}),
                request.recipe.model_copy(update={"pipeline_graph": {"graph_id": "g",
                    "primary_node_id": "p", "output_node_id": "p", "nodes": [
                        {"node_id": "p", "kind": "estimator",
                         "parameters": {"model_kind": "benter_conditional_logit"}}]}}),
            ):
                self.assertFalse(executor._supports_direct_parquet_fold_pushdown(
                    replace(request, recipe=recipe)))


if __name__ == "__main__":
    unittest.main()
