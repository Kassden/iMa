"""Content-addressed feature catalog and reproducible formula materialization."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from .feature_discovery_specs import content_id
from .feature_expressions import InputMetadata, NumericExpression, ExpressionError, validate_expression, evaluate_expression, unit_dimensions, unit_scale


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


class FeatureRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)

    def register(self, definition, input_metadata):
        definition = definition if isinstance(definition,FeatureDefinition) else FeatureDefinition.model_validate(definition)
        validate_definition(definition,input_metadata)
        payload = definition.model_dump(mode="json")
        payload["definition_id"] = definition.content_id()
        destination = self.root / (definition.content_id()+".json")
        fd, temporary = tempfile.mkstemp(dir=self.root)
        try:
            with os.fdopen(fd,"w") as stream:
                json.dump(payload,stream,sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return definition.content_id()

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
