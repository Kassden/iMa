"""Row-preserving feature successors and durable, separately tracked build receipts.

Only a verified DatasetRegistry catalog grants input eligibility. There are no
planner calls, raw acquisition, population selection, or fitted transforms here.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from itertools import combinations
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from .dataset_registry import (
    DatasetRegistry, _digest, _keys, _missing_path_inodes, _ordered_row_hashes_sha256, _write_json, file_sha256,
)
from .feature_definitions import (
    FeatureDefinition, FeatureRegistry, evaluate_formulas, validate_definition,
)
from .feature_expressions import InputMetadata, NumericExpression, unit_dimensions, unit_scale
from .research_targets import SUPPORTED_TARGETS, target_contract, validate_no_forbidden_label_features
from .research_storage import admitted_output,reserved_output_file,storage_writer,as_storage_blocked


EXPERIMENT_NAME = "ima-feature-builds"
_METADATA_KEYS = set(InputMetadata.model_fields)
_UNIT_NAMES = {
    "m", "km", "s", "sec", "seconds", "ms", "day", "days", "kg", "g", "lb",
    "mps", "lengths", "rating",
}


@contextmanager
def _temporary_directory_preserving_error(**kwargs):
    temporary = tempfile.TemporaryDirectory(**kwargs)
    try:
        yield temporary.name
    except BaseException as primary:
        try:
            temporary.cleanup()
        except BaseException as cleanup_error:
            if hasattr(primary, 'add_note'):
                primary.add_note(f"Feature staging cleanup also failed: {cleanup_error!r}")
        raise
    else:
        temporary.cleanup()


def _known_unit(unit: str) -> bool:
    if not isinstance(unit, str):
        return False
    try:
        tree = ast.parse(unit.replace("^", "**"), mode="eval")
        if any(node.id not in _UNIT_NAMES for node in ast.walk(tree) if isinstance(node, ast.Name)):
            return False
        unit_dimensions(unit)
        unit_scale(unit)
        return True
    except (ValueError, SyntaxError, TypeError, OverflowError):
        return False


def _safe_name(name: str) -> None:
    if not isinstance(name, str) or not name.isidentifier() or name.startswith(("target_", "future_", "postrace_", "post_race_")):
        raise ValueError(f"Unsafe feature symbol: {name}")
    for kind in sorted(SUPPORTED_TARGETS):
        validate_no_forbidden_label_features(target_contract(kind), [name])


def _numeric_metadata(catalog: dict, *, excluded=()) -> dict[str, InputMetadata]:
    """Filter explicit declarations; never guess units or availability from names."""
    if not isinstance(catalog, dict):
        raise ValueError("Expected a named, validated numeric catalog")
    result = {}
    for name, row in sorted(catalog.items()):
        if name in excluded or not isinstance(row, dict):
            continue
        if row.get("eligible") is not True or row.get("fit_required", False):
            continue
        if not {"unit", "dtype", "temporal_scope", "target_tainted"} <= row.keys():
            continue
        if not _known_unit(row["unit"]):
            continue
        try:
            _safe_name(name)
            meta = InputMetadata.model_validate({k: v for k, v in row.items() if k in _METADATA_KEYS})
            if meta.target_tainted or meta.temporal_scope in {"future", "current_outcome"}:
                continue
            if meta.temporal_scope == "historical" and not meta.available_at_column:
                continue
            result[name] = meta
        except ValueError:
            continue
    return result


def deterministic_candidate_definitions(catalog: dict, *, limit: int = 4) -> list[FeatureDefinition]:
    """Small deterministic domain formulas from explicit eligible numeric metadata.

    No target values or observed model scores are consulted. Candidate usefulness
    is a hypothesis; coverage and variation are checked during materialization.
    """
    if not 1 <= limit <= 8:
        raise ValueError("Candidate limit must be between 1 and 8")
    metadata = _numeric_metadata(catalog)
    existing = {row.get("definition_id") for row in catalog.values() if isinstance(row, dict)}
    proposals = []

    def add(name, expression, refs, unit, hypothesis):
        if not set(refs) <= metadata.keys():
            return
        for ref in refs:
            row = catalog[ref]
            if row.get("nonmissing") == 0 or row.get("distinct") in {0, 1}:
                return
        definition = FeatureDefinition(
            name=name, expression_ast=expression, input_refs=tuple(refs), unit=unit,
            source_families=tuple(sorted({metadata[ref].source_family for ref in refs})),
            producer="deterministic_feature_lifecycle", hypothesis=hypothesis,
        )
        try:
            validate_definition(definition, metadata)
        except ValueError:
            return
        if definition.content_id() not in existing and name not in catalog:
            existing.add(definition.content_id())
            proposals.append(definition)

    def ref(name):
        return NumericExpression(op="ref", ref=name)

    for name, left, right, op in (
        ("body_to_carried_weight_ratio", "declared_weight", "actual_weight", "divide"),
        ("distance_relative_to_previous", "distance", "last_distance", "divide"),
        ("distance_shift_from_previous", "distance", "last_distance", "subtract"),
        ("recent_speed_deviation", "past_race_speed_last_mps", "past_race_speed_mean_5_mps", "subtract"),
        ("past_race_speed_cv", "past_race_speed_std_5_mps", "past_race_speed_mean_5_mps", "divide"),
        ("workout_to_race_speed_ratio", "workout_speed_last_mps", "past_race_speed_mean_5_mps", "divide"),
    ):
        unit = "1" if op == "divide" else metadata[left].unit if left in metadata else "1"
        add(name, NumericExpression(op=op, args=(ref(left), ref(right))), [left, right], unit,
            "Relative load, distance change, or recent speed may add pre-race signal.")
    for name, meta in metadata.items():
        if meta.unit != "1" or not name.endswith(("_starts", "_count", "_support", "_observations")):
            continue
        clipped = NumericExpression(op="maximum", args=(ref(name), NumericExpression(op="constant", value=0)))
        shifted = NumericExpression(op="add", args=(clipped, NumericExpression(op="constant", value=1)))
        add("log1p_" + name, NumericExpression(op="log", args=(shifted,)), [name], "1",
            "Diminishing historical support may encode experience without a fitted transform.")
        if len(proposals) >= limit:
            break
    if len(proposals) < limit:
        # Compatible-unit contrasts extend the same mechanism to other catalogs.
        groups = {}
        for name, meta in metadata.items():
            groups.setdefault((meta.source_family, tuple(sorted(unit_dimensions(meta.unit).items())), unit_scale(meta.unit)), []).append(name)
        for group in groups.values():
            for left, right in combinations(group, 2):
                add("relative_" + left + "_to_" + right,
                    NumericExpression(op="divide", args=(ref(left), ref(right))), [left, right], "1",
                    "A scale-free contrast between compatible pre-race measurements may add signal.")
                if len(proposals) >= limit:
                    break
            if len(proposals) >= limit:
                break
    return proposals[:limit]


def _parent(features_path, manifest_path):
    features_path, manifest_path = Path(features_path).resolve(), Path(manifest_path).resolve()
    if manifest_path.name != "manifest.json" or features_path.name not in {'features.parquet','features.delta.json'} or features_path.parent!=manifest_path.parent:
        raise ValueError("Parent paths must identify the same registered dataset")
    if manifest_path.parent.parent.name != "datasets":
        raise ValueError("Parent must be an existing registered dataset")
    registry = DatasetRegistry(manifest_path.parent.parent.parent)
    manifest = registry.verify(manifest_path.parent.name)
    if manifest.get("features_path") and Path(manifest["features_path"]).resolve() != features_path:
        raise ValueError("Registered parent features_path does not match")
    required = {"protocol.json", "confirmation_keys.json", "raw_manifest.json", "source_manifest.json", features_path.name}
    if not required <= manifest["files"].keys():
        raise ValueError("Parent lacks protected protocol/provenance artifacts")
    if not manifest.get("comparison_contract_id") or not manifest.get("comparison_contract"):
        raise ValueError("Parent lacks a comparison contract")
    return manifest


def _race_cutoffs(values: pd.Series) -> pd.Series:
    """Naive HK dates mean local midnight; explicit offsets preserve true instants."""
    def instant(value):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Hong_Kong")
        return timestamp.tz_convert("UTC")

    try:
        parsed = pd.to_datetime(values, errors="raise")
    except ValueError:
        return pd.to_datetime(values.map(instant), utc=True, errors="raise")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_dtype(parsed.dtype):
        return parsed.dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC")
    return pd.to_datetime(values.map(instant), utc=True, errors="raise")


def _validate_definitions(definitions, metadata, columns):
    definitions = [d if isinstance(d, FeatureDefinition) else FeatureDefinition.model_validate(d) for d in definitions]
    if not definitions or len(definitions) > 16:
        raise ValueError("A feature build requires 1 to 16 definitions")
    if len({d.name for d in definitions}) != len(definitions) or len({d.content_id() for d in definitions}) != len(definitions):
        raise ValueError("Duplicate names or equivalent definitions")
    pending, completed, ordered = list(definitions), set(), []
    metadata = dict(metadata)
    while pending:
        ready = [d for d in pending if set(d.input_refs) <= metadata.keys() and set(d.parent_definition_ids) <= completed]
        if not ready:
            raise ValueError("Unknown, unavailable, or cyclic feature dependencies")
        for definition in sorted(ready, key=lambda d: d.name):
            _safe_name(definition.name)
            if not _known_unit(definition.unit):
                raise ValueError(f"Unknown output unit: {definition.unit}")
            if definition.name in columns or definition.feature_id in columns:
                raise ValueError(f"Feature already present in parent: {definition.name}")
            for ref in definition.input_refs:
                _safe_name(ref)
            validate_definition(definition, metadata)
            metadata[definition.name] = InputMetadata(unit=definition.unit, dtype=definition.dtype)
            metadata[definition.feature_id] = metadata[definition.name]
            completed.add(definition.content_id())
            pending.remove(definition)
            ordered.append(definition)
    return ordered


class FeatureLifecycle:
    """Filesystem ledger owned by a DatasetRegistry, with lock-serialized retries."""

    def __init__(self, registry_root):
        self.registry = registry_root if isinstance(registry_root, DatasetRegistry) else DatasetRegistry(registry_root)
        self.root = self.registry.root / "feature-lifecycle"
        inodes = _missing_path_inodes(self.root) + sum(
            not (self.root / name).exists() for name in ("builds", "outcomes"))
        with admitted_output(self.root, estimated_bytes=64 * 1024 * inodes,
                             stage='feature_lifecycle_init', inodes=inodes):
            for name in ("builds", "outcomes"):
                (self.root / name).mkdir(parents=True, exist_ok=True)

    def _path(self, build_id):
        if not isinstance(build_id, str) or not re.fullmatch(r"feature-build-[0-9a-f]{64}", build_id):
            raise ValueError("Invalid feature build ID")
        return self.root / "builds" / (build_id + ".json")

    def receipt(self, build_id) -> dict:
        receipt = json.loads(self._path(build_id).read_text())
        if receipt["build_id"] != build_id or "feature-build-" + _digest(receipt["spec"]) != build_id:
            raise ValueError("Feature receipt identity mismatch")
        definitions = [FeatureDefinition.model_validate(d) for d in receipt["definitions"]]
        signature = sorted([{"definition_id": d.content_id(), "name": d.name} for d in definitions], key=lambda d: d["name"])
        if signature != receipt["spec"]["definitions"]:
            raise ValueError("Feature receipt definitions changed")
        return receipt

    def _save(self, receipt):
        receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
        receipt.setdefault("created_at", receipt["updated_at"])
        _write_json(self._path(receipt["build_id"]), receipt)
        self._index(receipt)

    def _index(self, receipt, outcome=None):
        with self.registry._lock("feature-lifecycle-catalog"):
            path = self.root / "catalog-index.json"
            index = json.loads(path.read_text()) if path.exists() else {"builds": {}}
            entry = index["builds"].setdefault(receipt["build_id"], {"outcomes": {}})
            entry.update(status=receipt["status"], created_at=receipt.get("created_at"),
                         updated_at=receipt.get("updated_at"))
            if outcome is not None:
                entry["outcomes"].setdefault(outcome["outcome_id"], {
                    "comparison_id": outcome["comparison_id"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            _write_json(path, index)

    def submit(self, parent_features_path, parent_manifest_path, *, definitions, metadata=None, min_coverage=0.1) -> dict:
        if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be finite and in (0, 1]")
        parent = _parent(parent_features_path, parent_manifest_path)
        definitions = [d if isinstance(d, FeatureDefinition) else FeatureDefinition.model_validate(d) for d in definitions]
        # Narratives do not change semantics; names do change published columns.
        spec = {
            "parent_dataset_id": parent["dataset_id"],
            "parent_manifest_sha256": file_sha256(parent_manifest_path),
            "parent_features_sha256": parent["files"][Path(parent_features_path).name],
            "definitions": sorted([{"definition_id": d.content_id(), "name": d.name} for d in definitions], key=lambda d: d["name"]),
            "min_coverage": min_coverage,
            "code": {name: file_sha256(Path(__file__).parent / name) for name in ("feature_lifecycle.py", "feature_definitions.py", "feature_expressions.py")},
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "pydantic")},
        }
        if os.environ.get('IMA_DATASET_STORAGE_FORMAT')=='delta-v2':
            spec['storage_format']='delta-v2'
            spec['delta_implementation_sha256']=file_sha256(Path(__file__).parent/'dataset_delta.py')
        build_id = "feature-build-" + _digest(spec)
        with self.registry._lock(build_id):
            if self._path(build_id).exists():
                receipt = self.receipt(build_id)
                if metadata is not None:
                    self._metadata(parent, metadata)
                return receipt
            receipt = {
                "schema_version": 1, "build_id": build_id, "spec": spec, "status": "pending",
                "parent_features_path": str(Path(parent_features_path).resolve()),
                "parent_manifest_path": str(Path(parent_manifest_path).resolve()),
                "definitions": [d.model_dump(mode="json") for d in definitions],
                "tracking": {"status": "disabled", "run_id": None},
            }
            try:
                trusted = self._metadata(parent, metadata)
                columns = (json.loads(Path(parent_features_path).read_text())['columns']
                           if Path(parent_features_path).name=='features.delta.json'
                           else parquet.read_schema(parent_features_path).names)
                _validate_definitions(definitions, trusted, columns)
            except ValueError as exc:
                receipt.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
                self._save(receipt)
                raise
            self._save(receipt)
            return receipt

    @staticmethod
    def _metadata(parent, supplied):
        excluded = set(parent.get("target_only_columns", ())) | set(parent.get("market_only_columns", ()))
        metadata = _numeric_metadata(parent["predictor_catalog"], excluded=excluded)
        if supplied is not None:
            for name, row in supplied.items():
                value = row if isinstance(row, InputMetadata) else InputMetadata.model_validate({k: v for k, v in row.items() if k in _METADATA_KEYS})
                if name not in metadata or value != metadata[name]:
                    raise ValueError(f"Supplied metadata differs from verified catalog: {name}")
        return metadata

    @storage_writer(stage='feature_dataset',output=lambda a:a['self'].registry.root.resolve(),
                    estimate=lambda a:2*1024**3)
    def build(self, build_id, *, tracking_uri=None, experiment_name=EXPERIMENT_NAME) -> dict:
        with self.registry._lock(build_id):
            receipt = self.receipt(build_id)
            if receipt["status"] == "rejected":
                raise ValueError(receipt["error"])
            try:
                if receipt["status"] == "ready":
                    manifest = self.registry.verify(receipt["dataset_id"])
                else:
                    receipt["status"] = "pending"
                    self._save(receipt)
                    manifest = self._materialize(receipt)
                    receipt.update(status="ready", dataset_id=manifest["dataset_id"],
                                   diagnostics=manifest["feature_build"]["diagnostics"])
                    receipt.pop("error", None)
                    self._save(receipt)
            except Exception as exc:
                storage = as_storage_blocked(exc, stage='feature_dataset', output=self.registry.root)
                if storage is not None:
                    try:
                        receipt.update(status='blocked_storage', error=str(storage))
                        self._save(receipt)
                    except Exception as publication_error:
                        if hasattr(storage, 'add_note'):
                            storage.add_note(f"Storage block state publication also failed: {publication_error!r}")
                    if storage is exc:
                        raise
                    raise storage from exc
                if isinstance(exc, ValueError):
                    receipt.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
                    self._save(receipt)
                    raise
                if isinstance(exc, OSError):
                    receipt.update(status="pending", error=f"{type(exc).__name__}: {exc}")
                    self._save(receipt)
                    raise
                raise
            if tracking_uri:
                self._track(receipt, manifest, tracking_uri, experiment_name)
            return {**manifest, "feature_build_receipt": deepcopy(receipt)}

    def _materialize(self, receipt):
        if any(file_sha256(Path(__file__).parent / name) != digest for name, digest in receipt["spec"]["code"].items()):
            raise ValueError("Feature implementation changed since submission; resubmit")
        if any(version(name) != value for name, value in receipt["spec"]["dependencies"].items()):
            raise ValueError("Feature dependencies changed since submission; resubmit")
        parent = _parent(receipt["parent_features_path"], receipt["parent_manifest_path"])
        if file_sha256(receipt["parent_manifest_path"]) != receipt["spec"]["parent_manifest_sha256"]:
            raise ValueError("Parent manifest changed since submission")
        from .dataset_delta import read_feature_table, write_feature_delta
        frame = read_feature_table(receipt["parent_features_path"])
        if len(frame) != parent["rows"] or frame.race_id.nunique() != parent["races"] or frame.empty:
            raise ValueError("Parent row/race inventory mismatch")
        metadata = self._metadata(parent, None)
        definitions = _validate_definitions(receipt["definitions"], metadata, frame.columns)
        refs = {ref for d in definitions for ref in d.input_refs} & metadata.keys()
        bindings = frame.copy(deep=False)
        input_diagnostics = {}
        for ref in refs:
            if ref not in frame or not pd.api.types.is_numeric_dtype(frame[ref]):
                raise ValueError(f"Missing or nonnumeric input column: {ref}")
            values = frame[ref].to_numpy(dtype=float, na_value=np.nan)
            input_diagnostics[ref] = {"finite": int(np.isfinite(values).sum()),
                                      "missing": int(np.isnan(values).sum()),
                                      "nonfinite": int(np.isinf(values).sum())}
            bindings[ref] = frame[ref].replace([np.inf, -np.inf], np.nan)
        if "date" not in frame:
            raise ValueError("Parent lacks per-row race cutoffs")
        cutoffs = _race_cutoffs(frame.date)
        if cutoffs.isna().any():
            raise ValueError("Missing per-row race cutoff")
        output, names, report = evaluate_formulas(bindings, definitions, metadata, cutoff=cutoffs)
        for ref in refs:
            output[ref] = frame[ref]
        catalog = deepcopy(parent["predictor_catalog"])
        diagnostics = {}
        receipt["diagnostics"] = diagnostics
        for definition in definitions:
            values = output[definition.feature_id].to_numpy(dtype=float)
            finite = np.isfinite(values)
            stats = {"finite": int(finite.sum()), "missing": int(np.isnan(values).sum()),
                     "nonfinite": int(np.isinf(values).sum()), "total": len(values),
                     "coverage": float(finite.mean()), "distinct": int(pd.Series(values[finite]).nunique())}
            diagnostics[definition.content_id()] = stats
            if stats["coverage"] < receipt["spec"]["min_coverage"] or stats["distinct"] < 2 or stats["nonfinite"]:
                raise ValueError(f"Feature coverage/variation rejected: {definition.name}: {stats}")
            entry = {"unit": definition.unit, "dtype": definition.dtype, "temporal_scope": "pre_race",
                     "eligible": True, "target_tainted": False, "source_family": "validated_formula",
                     "definition_id": definition.content_id(), "input_refs": list(definition.input_refs),
                     "nonmissing": stats["finite"], "missing": stats["missing"], "total": stats["total"],
                     "distinct": stats["distinct"], "coverage": stats["coverage"], "fit_required": False,
                     "availability_proof": "typed formula over verified inputs; strict per-row availability masks"}
            for name in (definition.name, definition.feature_id):
                catalog[name] = dict(entry)
        pd.testing.assert_frame_equal(output.loc[:, frame.columns], frame, check_exact=True)
        identity = deepcopy(parent["identity"])
        identity.update(
            feature_build=receipt["spec"], parent_dataset_id=parent["dataset_id"],
            predictor_catalog_sha256=_digest(catalog),
            eligible_categorical_predictors_sha256=_digest(parent.get("eligible_categorical_predictors")),
        )
        dataset_id = "dataset-" + _digest(identity)
        destination = self.registry.dataset_path(dataset_id)
        if destination.exists():
            return self.registry.verify(dataset_id)
        # Copy protected artifacts, never link mutable inode contents or replay raw history.
        with _temporary_directory_preserving_error(dir=self.registry.root / "staging", prefix="feature-build-") as temporary:
            staging = Path(temporary).resolve() / "datasets" / dataset_id
            staging.mkdir(parents=True)
            delta_format=receipt['spec'].get('storage_format')=='delta-v2'
            replaced = {"features.parquet", "feature-build.json", "feature-catalog.json",
                        "features.delta.json","feature-delta.parquet"}
            if delta_format:
                replaced.update({'runners.parquet','runners.csv','features.csv'})
            for name in parent["files"]:
                if name not in replaced:
                    with (Path(receipt['parent_manifest_path']).parent/name).open('rb') as source, \
                         reserved_output_file(staging/name) as target:
                        shutil.copyfileobj(source,target,length=8*1024**2)
            if delta_format:
                write_feature_delta(receipt['parent_manifest_path'],staging,output,
                    registry_root=self.registry.root,columns=[c for c in output if c not in frame],catalog=catalog)
                feature_name='features.delta.json'
            else:
                with reserved_output_file(staging/'features.parquet') as stream:
                    output.to_parquet(stream,index=False)
                feature_name='features.parquet'
            persisted = read_feature_table(staging / feature_name)
            pd.testing.assert_frame_equal(persisted.loc[:, frame.columns], frame, check_exact=True)
            for name in names:
                np.testing.assert_equal(persisted[name].to_numpy(), output[name].to_numpy())
            payload = {
                "build_id": receipt["build_id"], "parent_dataset_id": parent["dataset_id"],
                "parent_manifest_sha256": receipt["spec"]["parent_manifest_sha256"],
                "definition_ids": report["definition_ids"], "feature_names": list(names),
                "definitions": [d.model_dump(mode="json") for d in definitions],
                "diagnostics": diagnostics, "input_diagnostics": input_diagnostics, "validation": report,
                "preserved_parent_files": {k: v for k, v in parent["files"].items() if k not in replaced},
                "parent_ordered_row_keys_sha256": parent["ordered_row_keys_sha256"],
                "parent_ordered_row_hashes_sha256": parent["ordered_row_hashes_sha256"],
            }
            _write_json(staging / "feature-build.json", payload)
            _write_json(staging / "feature-catalog.json", catalog)
            manifest = deepcopy(parent)
            manifest.update(
                dataset_id=dataset_id, identity=identity, parent_dataset_id=parent["dataset_id"],
                storage_format='ima.dataset.delta.v2' if delta_format else 'ima.dataset.parquet.v1',
                feature_build_id=receipt["build_id"], feature_definition_ids=report["definition_ids"],
                feature_build=payload, features_path=str(destination / feature_name),
                protocol_path=str(destination / "protocol.json"),
                predictor_catalog=catalog, feature_catalog=catalog, predictor_catalog_id="catalog-" + _digest(catalog),
                eligible_numeric_predictors=sorted(catalog), eligible_predictors=sorted(catalog),
                ordered_row_keys_sha256=_digest(_keys(persisted)),
                ordered_row_hashes_sha256=_ordered_row_hashes_sha256(persisted),
                files={p.name: file_sha256(p) for p in staging.iterdir()},
            )
            if manifest["ordered_row_keys_sha256"] != parent["ordered_row_keys_sha256"]:
                raise ValueError("Successor changed ordered parent row keys")
            _write_json(staging / "manifest.json", manifest)
            with reserved_output_file(staging/'manifest.sha256','w') as stream:
                stream.write(file_sha256(staging/'manifest.json')+'\n')
            for name, digest in payload["preserved_parent_files"].items():
                if file_sha256(staging / name) != digest:
                    raise ValueError(f"Parent artifact changed during successor build: {name}")
            DatasetRegistry(temporary).verify(dataset_id)
            feature_registry = FeatureRegistry(self.registry.feature_registry)
            # The lifecycle lock serializes its own immutable definition publication.
            with self.registry._lock("feature-lifecycle-definitions"):
                augmented = dict(metadata)
                for definition in definitions:
                    if (feature_registry.root / (definition.content_id() + ".json")).exists():
                        feature_registry.get(definition.content_id())
                    else:
                        feature_registry.register(definition, augmented)
                    augmented[definition.name] = InputMetadata(unit=definition.unit, dtype=definition.dtype)
                    augmented[definition.feature_id] = augmented[definition.name]
            os.rename(staging, destination)
        return self.registry.verify(dataset_id)

    def _track(self, receipt, manifest, tracking_uri, experiment_name):
        tracking = receipt["tracking"]
        scope = {"tracking_uri": str(tracking_uri), "experiment_name": experiment_name}
        if tracking.get("tracking_uri") and any(tracking.get(k) != v for k, v in scope.items()):
            raise ValueError("Build tracking receipt is bound to another URI/experiment")
        tracking.update(scope)
        if tracking["status"] == "published":
            return
        tracking.update(status="pending", attempts=tracking.get("attempts", 0) + 1)
        self._save(receipt)
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
            else:
                if experiment.lifecycle_stage != "active":
                    raise ValueError("Feature experiment is deleted")
                experiment_id = experiment.experiment_id
            tags = {
                "stage": "feature_build", "ima.stage": "feature_build", "ima.role": "feature_build",
                "ima.feature_build_id": receipt["build_id"], "ima.dataset_id": manifest["dataset_id"],
                "ima.parent_dataset_id": manifest["parent_dataset_id"],
                "ima.dataset_sha256": manifest["files"][Path(manifest['features_path']).name],
                "ima.definition_ids": json.dumps(manifest["feature_definition_ids"]),
                "ima.comparison_contract_id": manifest["comparison_contract_id"],
                "mlflow.runName": receipt["build_id"],
            }
            if not tracking.get("run_id"):
                # Search recovers create_run succeeding before a lost response/receipt.
                runs = client.search_runs([experiment_id], filter_string=f"tags.`ima.feature_build_id` = '{receipt['build_id']}'", max_results=2)
                if len(runs) > 1:
                    raise ValueError("Multiple tracking runs bind one feature build")
                run = runs[0] if runs else client.create_run(experiment_id, tags=tags)
                tracking.update(run_id=run.info.run_id, experiment_id=experiment_id)
                self._save(receipt)
            run_id = tracking["run_id"]
            params = {"definition_ids": json.dumps(manifest["feature_definition_ids"]),
                      "parent_dataset_id": manifest["parent_dataset_id"], "dataset_id": manifest["dataset_id"]}
            metrics = {f"feature.{key}.{field}": float(value) for key, row in manifest["feature_build"]["diagnostics"].items() for field, value in row.items()}
            for key, value in tags.items():
                client.set_tag(run_id, key, value)
            for key, value in params.items():
                client.log_param(run_id, key, value)
            for key, value in metrics.items():
                client.log_metric(run_id, key, value)
            directory = self.registry.dataset_path(manifest["dataset_id"])
            artifacts = ("manifest.json", "feature-build.json", "feature-catalog.json")
            for name in artifacts:
                client.log_artifact(run_id, str(directory / name), artifact_path="feature_build")
            client.set_terminated(run_id, status="FINISHED")
            run = client.get_run(run_id)
            if (run.info.status != "FINISHED" or any(run.data.tags.get(k) != v for k, v in tags.items())
                    or any(run.data.params.get(k) != v for k, v in params.items())
                    or any(run.data.metrics.get(k) != v for k, v in metrics.items())):
                raise RuntimeError("Feature MLflow run readback mismatch")
            with tempfile.TemporaryDirectory(prefix="feature-tracking-readback-") as temporary:
                for name in artifacts:
                    saved = client.download_artifacts(run_id, "feature_build/" + name, temporary)
                    if file_sha256(saved) != file_sha256(directory / name):
                        raise RuntimeError("Feature MLflow artifact readback mismatch")
            tracking.update(status="published", readback_verified=True)
            tracking.pop("error", None)
        except Exception as exc:
            tracking.update(status="pending", readback_verified=False, error=f"{type(exc).__name__}: {exc}")
        self._save(receipt)


def build_feature_candidate(parent_features_path, parent_manifest_path, registry_root, *, definitions,
                            metadata=None, tracking_uri=None, experiment_name=EXPERIMENT_NAME,
                            min_coverage=0.1) -> dict:
    """Publish/reuse a real successor; tracking failure leaves a retryable outbox."""
    lifecycle = FeatureLifecycle(registry_root)
    receipt = lifecycle.submit(parent_features_path, parent_manifest_path, definitions=definitions,
                               metadata=metadata, min_coverage=min_coverage)
    return lifecycle.build(receipt["build_id"], tracking_uri=tracking_uri, experiment_name=experiment_name)


def retry_tracking(registry_root, build_id: str, *, tracking_uri: str,
                   experiment_name=EXPERIMENT_NAME) -> dict:
    lifecycle = FeatureLifecycle(registry_root)
    lifecycle.build(build_id, tracking_uri=tracking_uri, experiment_name=experiment_name)
    return lifecycle.receipt(build_id)["tracking"]


def _outcome(value: dict) -> dict:
    required = {"attempt_id", "model_id", "dataset_id", "definition_ids", "score", "comparison_contract"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError(f"Model outcome requires {sorted(required)}")
    contract_fields = {"comparison_contract_id", "target_contract", "evaluation_population_id", "protocol_id",
                       "score_population_sha256", "metric", "direction", "probability_basis"}
    contract = value["comparison_contract"]
    if not isinstance(contract, dict) or not contract_fields <= contract.keys():
        raise ValueError(f"Outcome comparison contract requires {sorted(contract_fields)}")
    if contract["direction"] not in {"minimize", "maximize"}:
        raise ValueError("Metric direction must be minimize or maximize")
    if isinstance(value["score"], bool) or not math.isfinite(float(value["score"])):
        raise ValueError("Model outcome score must be finite")
    if any(not isinstance(value[k], str) or not value[k] for k in ("attempt_id", "model_id", "dataset_id")):
        raise ValueError("Model outcome identities must be nonempty strings")
    if any(not isinstance(contract[k], str) or not contract[k] for k in contract_fields):
        raise ValueError("Comparison identities must be explicit nonempty strings")
    if not isinstance(value["definition_ids"], (list, tuple)):
        raise ValueError("Model outcome definition_ids must be a sequence")
    return {**deepcopy(value), "score": float(value["score"]), "definition_ids": sorted(set(value["definition_ids"]))}


def record_model_outcome(registry_root, build_id: str, outcome: dict, *, baseline: dict | None = None) -> dict:
    """Link a reused candidate to a model and optionally a strictly matched baseline.

    Unmatched scores remain observations. Even a matched delta is evidence for
    this model comparison, never global or causal feature merit.
    """
    lifecycle = FeatureLifecycle(registry_root)
    outcome = _outcome(outcome)
    baseline = _outcome(baseline) if baseline is not None else None
    with lifecycle.registry._lock(build_id):
        receipt = lifecycle.receipt(build_id)
        if receipt["status"] != "ready":
            raise ValueError("Model outcomes require a ready feature build")
        manifest = lifecycle.registry.verify(receipt["dataset_id"])
        if outcome["dataset_id"] != manifest["dataset_id"]:
            raise ValueError("Model outcome does not consume this successor dataset")
        used = set(outcome["definition_ids"])
        known = {row.get("definition_id") for row in manifest["predictor_catalog"].values()}
        if not used <= known or not used.intersection(manifest["feature_definition_ids"]):
            raise ValueError("Model outcome must use a known candidate definition")
        for observation in (outcome, baseline):
            if observation is None:
                continue
            observed_manifest = manifest
            if observation["dataset_id"] != manifest["dataset_id"]:
                if observation["dataset_id"] == receipt["spec"]["parent_dataset_id"]:
                    observed_manifest = _parent(receipt["parent_features_path"], receipt["parent_manifest_path"])
                else:
                    observed_manifest = lifecycle.registry.verify(observation["dataset_id"])
            observed_definitions = {row.get("definition_id") for row in observed_manifest["predictor_catalog"].values()}
            if not set(observation["definition_ids"]) <= observed_definitions:
                raise ValueError("Baseline names unknown feature definitions")
            contract = observation["comparison_contract"]
            parent_comparison = observed_manifest["comparison_contract"]
            expected = {"comparison_contract_id": observed_manifest["comparison_contract_id"],
                        "protocol_id": parent_comparison["protocol_id"]}
            if contract["evaluation_population_id"].startswith("evaluated-"):
                if (contract.get("declared_evaluation_population_id") != observed_manifest["evaluation_population_id"]
                        or contract["evaluation_population_id"] != "evaluated-" + contract["score_population_sha256"]
                        or not re.fullmatch(r"[0-9a-f]{64}", contract.get("protocol_hash", ""))
                        or not re.fullmatch(r"[0-9a-f]{64}", contract["score_population_sha256"])):
                    raise ValueError("Actual scored population requires declared population and protocol hash")
            else:
                expected.update(evaluation_population_id=observed_manifest["evaluation_population_id"],
                                score_population_sha256=parent_comparison["score_population_sha256"],
                                probability_basis=parent_comparison["probability_basis"])
            if any(contract[k] != v for k, v in expected.items()):
                raise ValueError("Model outcome differs from dataset comparison contract")
        paired = baseline is not None and outcome["comparison_contract"] == baseline["comparison_contract"]
        row = {"build_id": build_id, "outcome": outcome, "baseline": baseline, "paired": paired,
               "comparison_id": _digest(outcome["comparison_contract"]),
               "scope": "model comparison only; no global feature merit"}
        if paired:
            row["score_delta"] = outcome["score"] - baseline["score"]
            if not math.isfinite(row["score_delta"]):
                raise ValueError("Paired model score delta must be finite")
            row["improvement"] = row["score_delta"] * (-1 if outcome["comparison_contract"]["direction"] == "minimize" else 1)
        row["outcome_id"] = "feature-outcome-" + _digest({
            "build_id": build_id, "attempt_id": outcome["attempt_id"],
            "model_id": outcome["model_id"], "comparison_id": row["comparison_id"],
        })
        path = lifecycle.root / "outcomes" / (row["outcome_id"] + ".json")
        if path.exists():
            if json.loads(path.read_text()) != row:
                raise ValueError("Model attempt already binds a different outcome")
        else:
            _write_json(path, row)
        lifecycle._index(receipt, row)
        return row


def catalog(registry_root, *, limit: int = 20) -> dict:
    """Use cheap summaries to bound receipt, parquet and outcome reads by limit.

    Totals describe recorded status; integrity checks cover only the selected
    window. Legacy receipts are hydrated within that same budget, and unindexed
    historical outcomes are disclosed rather than scanned or called untested.
    """
    if not 1 <= limit <= 1000:
        raise ValueError("Catalog limit must be between 1 and 1000")
    lifecycle = FeatureLifecycle(registry_root)
    index_path = lifecycle.root / "catalog-index.json"
    index = json.loads(index_path.read_text())["builds"] if index_path.exists() else {}
    paths = {path.stem: path for path in (lifecycle.root / "builds").glob("*.json")}
    index = {key: row for key, row in index.items() if key in paths}
    outcome_paths = {path.stem: path for path in (lifecycle.root / "outcomes").glob("*.json")}
    indexed_outcomes = {key for row in index.values() for key in row["outcomes"]}
    unindexed_outcomes = len(outcome_paths.keys() - indexed_outcomes)
    counts = {"pending": 0, "ready": 0, "rejected": 0}
    for summary in index.values():
        counts[summary["status"]] += 1
    candidates = []
    for build_id, path in paths.items():
        summary = index.get(build_id)
        created_at = (summary.get("created_at") or "") if summary else datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        priority = (0 if summary and summary["status"] == "ready" and not summary["outcomes"] and not unindexed_outcomes else
                    1 if summary is None else 2 if summary["status"] == "ready" else
                    3 if summary["status"] == "pending" else 4)
        candidates.append((priority, created_at, build_id, path))
    candidates.sort(key=lambda row: (row[1], row[2]), reverse=True)
    candidates.sort(key=lambda row: row[0])
    builds, loaded_outcomes = [], 0
    for _, _, build_id, path in candidates[:limit]:
        receipt = lifecycle.receipt(path.stem)
        summary = index.get(build_id)
        if summary is None:
            lifecycle._index(receipt)
            summary = {"status": receipt["status"], "outcomes": {}}
            index[build_id] = summary
            counts[receipt["status"]] += 1
        status = receipt["status"]
        error = receipt.get("error")
        if status == "ready":
            try:
                manifest = lifecycle.registry.verify(receipt["dataset_id"])
            except (ValueError, OSError) as exc:
                status, error = "rejected", f"integrity: {exc}"
        counts[summary["status"]] -= 1
        counts[status] += 1
        entry = {"build_id": receipt["build_id"], "status": status,
                 "parent_dataset_id": receipt["spec"]["parent_dataset_id"],
                 "definition_ids": [d["definition_id"] for d in receipt["spec"]["definitions"]],
                 "dataset_id": receipt.get("dataset_id"), "tracking": receipt["tracking"],
                 "diagnostics": receipt.get("diagnostics", {}), "comparisons": [],
                 "created_at": receipt.get("created_at"), "updated_at": receipt.get("updated_at"),
                 "model_outcome_count": len(summary["outcomes"]),
                 "model_outcomes_complete": not unindexed_outcomes}
        if error:
            entry["error"] = error
        if status == "ready":
            entry.update(features_path=manifest["features_path"], protocol_path=manifest["protocol_path"],
                         comparison_contract_id=manifest["comparison_contract_id"],
                         evaluation_population_id=manifest["evaluation_population_id"],
                         feature_ids=manifest["feature_build"]["feature_names"],
                         definitions=[{
                             "name": d["name"], "definition_id": FeatureDefinition.model_validate(d).content_id(),
                             "feature_id": FeatureDefinition.model_validate(d).feature_id,
                             "unit": d["unit"], "input_refs": d["input_refs"], "expression_ast": d["expression_ast"],
                         } for d in manifest["feature_build"]["definitions"]],
                         formula_report=manifest["feature_build"]["validation"])
        groups = {}
        refs = sorted(summary["outcomes"].items(), key=lambda item: (item[1]["created_at"], item[0]), reverse=True)
        for outcome_id, ref in refs:
            groups.setdefault(ref["comparison_id"], []).append(outcome_id)
        for comparison_id, outcome_ids in list(groups.items())[:4]:
            group = {"comparison_id": comparison_id, "model_outcomes": [],
                     "omitted_model_outcomes": max(0, len(outcome_ids) - 4)}
            for outcome_id in outcome_ids[:4]:
                row = json.loads(outcome_paths[outcome_id].read_text())
                if row["build_id"] != build_id or row["comparison_id"] != comparison_id or row["outcome_id"] != outcome_id:
                    raise ValueError("Catalog outcome index mismatch")
                loaded_outcomes += 1
                group["comparison_contract"] = row["outcome"]["comparison_contract"]
                group["model_outcomes"].append({k: row[k] for k in ("outcome", "baseline", "paired", "score_delta", "improvement") if k in row})
            entry["comparisons"].append(group)
        entry["omitted_comparisons"] = max(0, len(groups) - 4)
        entry["omitted_model_outcomes"] = len(summary["outcomes"]) - sum(len(group["model_outcomes"]) for group in entry["comparisons"])
        builds.append(entry)
    return {"schema_version": 1, "counts": counts, "builds": builds,
            "counts_scope": "recorded statuses; integrity checked only for selected builds",
            "unindexed_builds": len(paths.keys() - index.keys()), "unindexed_outcomes": unindexed_outcomes,
            "omitted_builds": max(0, len(paths) - len(builds)),
            "omitted_outcomes": max(0, len(outcome_paths) - loaded_outcomes),
            "interpretation": "Scores are grouped by comparison contract; paired deltas apply only to those models. No global feature ranking."}
