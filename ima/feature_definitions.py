"""Content-addressed feature catalog and reproducible formula materialization."""
from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from .feature_discovery_specs import content_id
from .feature_expressions import InputMetadata, NumericExpression, ExpressionError, validate_expression, evaluate_expression, unit_dimensions, unit_scale
from .research_storage import admitted_output, as_storage_blocked, reserved_output_file, storage_writer


_DIRECTORY_BLOCK_ALLOWANCE = 64 * 1024


def _missing_directory_count(path):
    current = Path(path).absolute()
    missing = 0
    while not current.exists():
        missing += 1
        current = current.parent
    return missing


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_definition_file(path, definition_id, root):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink < 1
                or before.st_uid != os.getuid() or before.st_dev != Path(root).stat().st_dev):
            raise ValueError("Unsafe feature definition publication")
        with os.fdopen(descriptor, "r") as stream:
            descriptor = -1
            definition = FeatureDefinition.model_validate(json.load(stream))
            after = os.fstat(stream.fileno())
        current = Path(path).lstat()
        if (before.st_dev, before.st_ino, before.st_mode, before.st_uid) != (
                after.st_dev, after.st_ino, after.st_mode, after.st_uid):
            raise ValueError("Feature definition changed during verification")
        if (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Feature definition path changed during verification")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if definition.content_id() != definition_id:
        raise ValueError("Corrupt registry content ID")
    return definition


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    definition_id: str | None = None
    name: str
    expression_ast: NumericExpression
    input_refs: tuple[str, ...]
    source_families: tuple[str, ...] = ()
    entity_key: str = "horse_id"
    time_scope: Literal["pre_race", "historical"] = "pre_race"
    cutoff_rule: Literal["strict_before"] = "strict_before"
    dtype: Literal["float32", "float64"] = "float64"
    unit: str = "1"
    missing_policy: Literal["propagate"] = "propagate"
    fit_required: bool = False
    producer: str = "planner"
    parent_definition_ids: tuple[str, ...] = ()
    hypothesis_id: str | None = None
    validation_report_id: str | None = None
    hypothesis: str | None = None

    @model_validator(mode="after")
    def valid_definition(self):
        if not self.name.isidentifier():
            raise ValueError("Feature name must be a numeric symbol")
        if self.fit_required:
            raise ValueError("Learned formulas require a training-fold transform adapter")
        if len(self.input_refs) != len(set(self.input_refs)):
            raise ValueError("Duplicate input refs")
        if self.definition_id is not None and self.definition_id != self.content_id():
            raise ValueError("Feature definition content ID mismatch")
        return self

    def content_id(self):
        return content_id(self.model_dump(mode="json",exclude={"definition_id","name","hypothesis_id","hypothesis","validation_report_id","producer"}))

    @property
    def feature_id(self):
        return "dfs_" + self.content_id()


def _publication_bytes(definition):
    value=definition if isinstance(definition,FeatureDefinition) else FeatureDefinition.model_validate(definition)
    return 64*1024+4*len(json.dumps(value.model_dump(mode='json'),sort_keys=True).encode())


class FeatureRegistry:
    def __init__(self, root):
        self.root = Path(root)
        missing = _missing_directory_count(self.root)
        try:
            with admitted_output(self.root, estimated_bytes=_DIRECTORY_BLOCK_ALLOWANCE * missing,
                                 stage='feature_definition_root', inodes=missing):
                self.root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            blocked = as_storage_blocked(exc, stage='feature_definition_root', output=self.root)
            if blocked is None or blocked is exc:
                raise
            raise blocked from exc

    @storage_writer(stage='feature_definition',output=lambda a:a['self'].root,
                    estimate=lambda a:_publication_bytes(a['definition']))
    def register(self, definition, input_metadata):
        definition = definition if isinstance(definition,FeatureDefinition) else FeatureDefinition.model_validate(definition)
        validate_definition(definition,input_metadata)
        admission_enabled = bool(os.environ.get('IMA_STORAGE_ROOT'))
        payload = definition.model_dump(mode="json")
        payload["definition_id"] = definition.content_id()
        destination = self.root / (definition.content_id()+".json")
        definition_id = definition.content_id()
        if destination.exists() or destination.is_symlink():
            if admission_enabled:
                _validate_definition_file(destination, definition_id, self.root)
            else:
                self.get(definition_id)
            return definition_id
        temporary = self.root / (f".{definition_id}.{uuid.uuid4().hex}.tmp")
        identity = None
        created = False
        primary = None
        try:
            with reserved_output_file(temporary, 'x') as stream:
                created = True
                if admission_enabled:
                    info = os.fstat(stream.fileno())
                    identity = (info.st_dev, info.st_ino)
                    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                            or info.st_uid != os.getuid() or info.st_dev != self.root.stat().st_dev):
                        raise ValueError("Unsafe feature definition staging file")
                json.dump(payload,stream,sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if admission_enabled:
                descriptor = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    staged = os.fstat(descriptor)
                    if (identity != (staged.st_dev, staged.st_ino) or not stat.S_ISREG(staged.st_mode)
                            or staged.st_nlink != 1 or staged.st_uid != os.getuid()
                            or staged.st_dev != self.root.stat().st_dev):
                        raise ValueError("Feature definition staging identity changed")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if admission_enabled:
                    _validate_definition_file(destination, definition_id, self.root)
                else:
                    self.get(definition_id)
            else:
                if admission_enabled:
                    published = destination.lstat()
                    if ((published.st_dev, published.st_ino) != identity
                            or not stat.S_ISREG(published.st_mode) or published.st_uid != os.getuid()
                            or published.st_dev != self.root.stat().st_dev):
                        raise ValueError("Feature definition publication identity changed")
                _sync_directory(self.root)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                if created and temporary.exists():
                    staged = temporary.lstat()
                    if admission_enabled and identity is not None and (staged.st_dev, staged.st_ino) != identity:
                        raise ValueError("Feature definition staging identity changed during cleanup")
                    temporary.unlink()
                    _sync_directory(self.root)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                if hasattr(primary, 'add_note'):
                    primary.add_note(f"Feature definition staging cleanup also failed: {cleanup_error!r}")
        return definition_id

    def get(self, definition_id):
        if len(definition_id) != 24 or any(c not in "0123456789abcdef" for c in definition_id):
            raise ValueError("Invalid content ID")
        definition = FeatureDefinition.model_validate(json.loads((self.root/(definition_id+".json")).read_text()))
        if definition.content_id() != definition_id:
            raise ValueError("Corrupt registry content ID")
        return definition


def validate_definition(definition, input_metadata):
    report = validate_expression(definition.expression_ast,input_metadata)
    if set(report["dependencies"]) != set(definition.input_refs):
        raise ExpressionError("dependency_mismatch", definition.name)
    if report["dimensions"] != unit_dimensions(definition.unit) or report["unit_scale"] != unit_scale(definition.unit):
        raise ExpressionError("unit_mismatch", definition.name)
    return report


def evaluate_formulas(frame, definitions, input_metadata, *, cutoff=None):
    """Return a new frame, content-ID numeric names and validation/coverage report.

    input_metadata is an explicit validated dataset catalog, never inferred from names.
    Historical inputs need per-row availability strictly before the supplied cutoffs.
    """
    definitions = [d if isinstance(d,FeatureDefinition) else FeatureDefinition.model_validate(d) for d in definitions]
    metadata = {k:v if isinstance(v,InputMetadata) else InputMetadata.model_validate(v) for k,v in input_metadata.items()}
    output = frame.copy()
    pending = {d.name:d for d in definitions}
    if len(pending) != len(definitions):
        raise ExpressionError("duplicate_definition", "Duplicate formula names")
    if len({d.content_id() for d in definitions}) != len(definitions):
        raise ExpressionError("duplicate_semantics", "Equivalent formula definitions")
    names, reports, completed = [], [], set()
    while pending:
        ready = [d for d in pending.values() if set(d.input_refs) <= metadata.keys() and set(d.parent_definition_ids) <= completed]
        if not ready:
            raise ExpressionError("unknown_or_cyclic_dependency", ",".join(sorted(pending)))
        for definition in sorted(ready,key=lambda d:d.name):
            if definition.name in frame or definition.name in input_metadata:
                raise ExpressionError("duplicate_semantics", definition.name)
            if not set(definition.parent_definition_ids) <= completed:
                raise ExpressionError("parent_dependency_mismatch", definition.name)
            report = validate_definition(definition,metadata)
            bindings = {}
            for ref in definition.input_refs:
                if ref not in output:
                    raise ExpressionError("missing_column", ref)
                if not pd.api.types.is_numeric_dtype(output[ref]):
                    raise ExpressionError("numeric_dtype", ref)
                values = output[ref].to_numpy(dtype=float,copy=True)
                meta = metadata[ref]
                if meta.available_at_column:
                    if cutoff is None or meta.available_at_column not in output:
                        raise ExpressionError("missing_cutoff", ref)
                    observed = pd.to_datetime(output[meta.available_at_column],utc=True,errors="coerce")
                    boundary = pd.to_datetime(cutoff,utc=True,errors="raise")
                    valid = observed.notna() & (observed < boundary)
                    values[~np.asarray(valid)] = np.nan
                bindings[ref] = values
            values = evaluate_expression(definition.expression_ast,bindings,len(output))
            with np.errstate(over="ignore"):
                values = values.astype(definition.dtype)
            values[~np.isfinite(values)] = np.nan
            output[definition.name] = values
            output[definition.feature_id] = values
            metadata[definition.name] = InputMetadata(unit=definition.unit,dtype=definition.dtype)
            metadata[definition.feature_id] = metadata[definition.name]
            completed.add(definition.content_id())
            names.append(definition.feature_id)
            reports.append({**report,"definition_id":definition.content_id(),"feature_id":definition.feature_id,"unit":definition.unit,"dtype":definition.dtype,"coverage":float(np.isfinite(values).mean())})
            del pending[definition.name]
    return output, tuple(names), {"schema_version":1,"definitions":reports,"definition_ids":sorted(completed)}


materialize_formulas = evaluate_formulas
