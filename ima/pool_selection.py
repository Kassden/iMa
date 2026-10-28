"""Prefix-only graph selection with immutable, source-pinned evidence.

This module deliberately does not wire itself into ``pipeline_graph``. Candidate
graphs are fitted through that module only after runtime validation, so importing
this module cannot create a pipeline_graph/research_pooling import cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .feature_sets import FeatureSchema
from .prediction_store import frame_fingerprint
from .research_storage import reserved_output_file, storage_writer


PROTOCOL_KEYS = {
    "schema_version", "selection_cutoff", "validation_start",
    "min_prefix_train_meetings", "min_prefix_validation_meetings",
    "registered_features", "candidate_graphs", "selection_trial_budget",
    "seed", "objective",
}
SOURCE_KEYS = {
    "dataset_hash", "dataset_manifest_hash", "source_revision", "source_code_hash",
    "protocol_hash", "target_contract", "evaluation_population_id",
    "evaluation_population_hash",
}
GRAPH_KEYS = {
    "graph_id", "nodes", "output_node_id", "fundamental_node_id", "primary_node_id",
    "joint_node_id", "date_column", "horse_column", "n_splits", "min_train_dates",
    "schema_version", "dataset_id", "target_contract_id", "evaluation_contract_id",
    "fit_budget", "component_refs", "tune_node_id",
}
NODE_KEYS = {"node_id", "kind", "inputs", "parameters", "output", "fit_scope"}
ALLOWED_NODE_KINDS = {"estimator", "feature_view", "weighted_probability_pool", "log_probability_pool"}
ARTIFACT_KEYS = {
    "schema_version", "selection_id", "selection_sha256", "selection_mode",
    "selection_cutoff", "validation_start", "future_oof_start", "objective",
    "tie_rule", "protocol_sha256", "source", "feature_schema", "feature_schema_sha256",
    "prefix", "candidate_graphs", "candidate_results", "selected_candidate_id",
    "selected_graph", "selected_graph_sha256", "optuna", "scope",
}


class SelectionContractError(ValueError):
    """A malformed, unsupported, or mismatched nested-selection contract."""


@dataclass(frozen=True)
class PrefixSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    dates: pd.Series
    date_column: str
    selection_cutoff: pd.Timestamp
    validation_start: pd.Timestamp


@dataclass(frozen=True)
class SelectionResult:
    candidate_id: str
    graph: dict[str, Any]
    artifact: dict[str, Any]
    artifact_path: Path


@dataclass(frozen=True)
class SelectionArtifact:
    """Validated immutable-content representation for package/readback callers."""

    selection_id: str
    selection_sha256: str
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SelectionArtifact":
        value = _require_mapping(payload, "selection artifact")
        selection_id, digest = _selection_identity(value)
        if value.get("selection_id") != selection_id or value.get("selection_sha256") != digest:
            raise SelectionContractError("selection artifact canonical hash or ID mismatch")
        return cls(selection_id, digest, json.loads(canonical_json_bytes(value)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(self.payload))


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one permitted byte representation for selection evidence."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SelectionContractError(f"Selection payload is not canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items()
            if key not in {"selection_id", "selection_sha256"}}


def _selection_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    digest = _sha256(_identity_payload(payload))
    return "selection-" + digest[:24], digest


def selection_key(payload: Mapping[str, Any]) -> str:
    """Return the full canonical SHA-256, excluding the two identity fields."""
    return _selection_identity(payload)[1]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionContractError(f"{name} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    missing, extra = required - set(value), set(value) - required
    if missing or extra:
        raise SelectionContractError(
            f"{name} keys invalid (missing={sorted(missing)}, extra={sorted(extra)})")


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionContractError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelectionContractError(f"{name} must be a nonempty string")
    return value


def _timestamp_boundary(value: Any, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SelectionContractError(f"{name} must be a valid date boundary") from exc
    if pd.isna(stamp) or stamp != stamp.normalize():
        raise SelectionContractError(f"{name} must be an explicit whole-date boundary")
    return stamp


def _schema_contract(schema: FeatureSchema) -> dict[str, Any]:
    if not isinstance(schema, FeatureSchema):
        raise SelectionContractError("feature_schema must be a fixed FeatureSchema")
    contract = schema.contract()
    if (not schema.features or len(set(schema.features)) != len(schema.features)
            or not schema.numeric or len(set(schema.numeric)) != len(schema.numeric)
            or len(set(schema.categorical)) != len(schema.categorical)
            or set(schema.numeric) & set(schema.categorical)):
        raise SelectionContractError("feature_schema must have unique, nonempty declared columns")
    return contract


def validate_selection_protocol(protocol: Mapping[str, Any], *, feature_schema: FeatureSchema,
                                expected_output: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and normalize the closed JSON protocol without fitting a model.

    ``expected_output`` is the learned pool's declared win-probability contract.
    Candidate graphs must expose that exact output and identical endpoint IDs.
    """
    raw = _require_mapping(protocol, "selection_protocol")
    _exact_keys(raw, PROTOCOL_KEYS, "selection_protocol")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise SelectionContractError("selection_protocol schema_version must equal 1")
    cutoff = _timestamp_boundary(raw["selection_cutoff"], "selection_cutoff")
    validation_start = _timestamp_boundary(raw["validation_start"], "validation_start")
    if validation_start > cutoff:
        raise SelectionContractError("validation_start must be no later than selection_cutoff")
    train_min = _positive_int(raw["min_prefix_train_meetings"], "min_prefix_train_meetings", minimum=3)
    validation_min = _positive_int(raw["min_prefix_validation_meetings"],
                                   "min_prefix_validation_meetings", minimum=2)
    budget = _positive_int(raw["selection_trial_budget"], "selection_trial_budget")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SelectionContractError("seed must be a nonnegative integer")
    if raw["objective"] != "mean_meeting_log_loss":
        raise SelectionContractError("objective must be mean_meeting_log_loss")

    schema_contract = _schema_contract(feature_schema)
    registered = raw["registered_features"]
    if (not isinstance(registered, list) or not registered
            or any(not isinstance(name, str) or not name for name in registered)
            or len(registered) != len(set(registered))):
        raise SelectionContractError("registered_features must be a nonempty unique list")
    if tuple(registered) != tuple(feature_schema.features):
        raise SelectionContractError("registered_features must exactly match the fixed feature schema order")

    candidates = raw["candidate_graphs"]
    if not isinstance(candidates, list) or not candidates:
        raise SelectionContractError("candidate_graphs must be a nonempty list")
    if budget != len(candidates):
        raise SelectionContractError("selection_trial_budget must equal candidate_graphs count in protocol v1")
    candidate_ids: list[str] = []
    normalized_candidates = []
    reference_topology = None
    output_contract = dict(expected_output) if expected_output is not None else {
        "kind": "win_probability", "target": "win_probability", "unit": "probability", "market": False}
    if output_contract != {"kind": "win_probability", "target": "win_probability",
                           "unit": "probability", "market": False}:
        raise SelectionContractError("selection output must be fundamental win_probability")

    # Runtime import only: pipeline_graph imports the learned-pool guard and must
    # never import this module at module initialization time.
    from .pipeline_graph import PipelineGraph

    for index, item in enumerate(candidates):
        candidate = _require_mapping(item, f"candidate_graphs[{index}]")
        _exact_keys(candidate, {"candidate_id", "graph"}, f"candidate_graphs[{index}]")
        candidate_id = _nonempty_string(candidate["candidate_id"], f"candidate_graphs[{index}].candidate_id")
        if candidate_id in candidate_ids:
            raise SelectionContractError("candidate_id values must be unique")
        candidate_ids.append(candidate_id)
        graph_payload = _validate_graph_payload(candidate["graph"], feature_schema,
                                                registered, output_contract, PipelineGraph)
        topology = _graph_topology(graph_payload)
        if reference_topology is None:
            reference_topology = topology
        elif topology != reference_topology:
            raise SelectionContractError("candidate graphs must preserve stable endpoint identities and topology")
        normalized_candidates.append({"candidate_id": candidate_id, "graph": graph_payload})

    return {
        "schema_version": 1,
        "selection_cutoff": cutoff.date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "min_prefix_train_meetings": train_min,
        "min_prefix_validation_meetings": validation_min,
        "registered_features": list(registered),
        "candidate_graphs": normalized_candidates,
        "selection_trial_budget": budget,
        "seed": seed,
        "objective": "mean_meeting_log_loss",
    }


def _validate_graph_payload(value: Any, feature_schema: FeatureSchema, registered: list[str],
                            expected_output: dict[str, Any], PipelineGraph) -> dict[str, Any]:
    from .research_specs import ModelSpec

    graph = _require_mapping(value, "candidate graph")
    unknown_graph_keys = set(graph) - GRAPH_KEYS
    if unknown_graph_keys:
        raise SelectionContractError(f"candidate graph has unsupported keys: {sorted(unknown_graph_keys)}")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise SelectionContractError("candidate graph nodes must be a nonempty list")
    for index, node_value in enumerate(nodes):
        node = _require_mapping(node_value, f"candidate graph node {index}")
        if set(node) - NODE_KEYS:
            raise SelectionContractError(f"candidate graph node has unsupported keys: {sorted(set(node)-NODE_KEYS)}")
        kind = node.get("kind")
        if kind not in ALLOWED_NODE_KINDS:
            raise SelectionContractError(f"unsupported candidate graph node kind: {kind}")
        parameters = _require_mapping(node.get("parameters", {}), f"node {node.get('node_id')} parameters")
        if kind == "estimator":
            if node.get("fit_scope", "forward_oof") != "forward_oof":
                raise SelectionContractError("candidate estimators must declare forward_oof fit_scope")
            if set(parameters) - {"model_kind", "model_parameters", "parameters"}:
                raise SelectionContractError("estimator parameters must use the registered model parameter contract")
            if "model_parameters" in parameters and "parameters" in parameters:
                raise SelectionContractError("estimator cannot specify both model_parameters and parameters")
            model_kind = parameters.get("model_kind", "benter_conditional_logit")
            model_parameters = parameters.get("model_parameters", parameters.get("parameters", {}))
            try:
                ModelSpec(kind=model_kind, parameters=model_parameters)
            except Exception as exc:
                raise SelectionContractError(f"estimator is outside the registered model parameter schema: {exc}") from exc
        elif kind == "feature_view":
            if set(parameters) - {"features"}:
                raise SelectionContractError("feature_view accepts only registered feature columns")
            columns = parameters.get("features", list(feature_schema.features))
            if (not isinstance(columns, (list, tuple)) or not columns
                    or len(columns) != len(set(columns)) or not set(columns) <= set(registered)):
                raise SelectionContractError("feature_view columns must be a nonempty registered feature subset")
        elif kind == "weighted_probability_pool":
            if set(parameters) != {"weights"}:
                raise SelectionContractError("fixed weighted pool requires only explicit weights")
            weights = parameters["weights"]
            if (not isinstance(weights, list) or not weights
                    or len(weights) != len(node.get("inputs", ()))
                    or any(isinstance(w, bool) or not isinstance(w, (int, float))
                           or not np.isfinite(w) or w < 0 for w in weights)
                    or not np.isclose(sum(weights), 1.0, rtol=0, atol=1e-10)):
                raise SelectionContractError("fixed weighted pool weights must be finite, nonnegative and sum to one")
        elif parameters:
            raise SelectionContractError("log_probability_pool has no tunable parameters in protocol v1")
        output = node.get("output", {})
        if not isinstance(output, Mapping):
            raise SelectionContractError("candidate graph node output must be an object")
        if kind == "feature_view":
            if output and (output.get("kind") != "feature_frame" or output.get("unit") != "features"):
                raise SelectionContractError("feature_view nodes must declare a feature_frame output")
        elif output.get("kind", "win_probability") != "win_probability":
            raise SelectionContractError("candidate graph nodes must produce win_probability")
        if isinstance(output, Mapping) and output.get("market", False):
            raise SelectionContractError("market outputs are unsupported in prefix selection")

    try:
        parsed = PipelineGraph.from_dict(graph).validate()
    except Exception as exc:
        raise SelectionContractError(f"candidate graph fails PipelineGraph validation: {exc}") from exc
    if parsed.component_refs:
        raise SelectionContractError("external fitted component references are unsupported")
    if parsed.target_contract_id != "win_probability":
        raise SelectionContractError("candidate graph target_contract_id must be win_probability")
    if parsed.joint_node_id is not None:
        raise SelectionContractError("joint/ranking outputs are unsupported in prefix selection")
    output_node = next(node for node in parsed.nodes if node.node_id == parsed.output_node_id)
    output = asdict(output_node.output)
    if output != expected_output:
        raise SelectionContractError("candidate graphs must have the exact declared win-probability output contract")
    normalized = asdict(parsed)
    normalized["nodes"] = [
        {**asdict(node), "inputs": list(node.inputs), "output": asdict(node.output)}
        for node in parsed.nodes
    ]
    normalized["component_refs"] = list(parsed.component_refs)
    return normalized


def _graph_topology(graph: Mapping[str, Any]) -> dict[str, Any]:
    nodes = graph["nodes"]
    return {
        "graph_configuration": {key: value for key, value in graph.items()
                                 if key not in {"graph_id", "nodes"}},
        "output_node_id": graph["output_node_id"],
        "fundamental_node_id": graph.get("fundamental_node_id"),
        "primary_node_id": graph.get("primary_node_id"),
        "tune_node_id": graph.get("tune_node_id"),
        "date_column": graph["date_column"],
        "horse_column": graph["horse_column"],
        "node_contracts": [{
            "node_id": node["node_id"], "kind": node["kind"],
            "inputs": list(node.get("inputs", ())), "output": node["output"],
            "fit_scope": node["fit_scope"],
        } for node in nodes],
    }


def _source_context(source: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(source, "source_context")
    _exact_keys(value, SOURCE_KEYS, "source_context")
    for key in ("dataset_hash", "source_code_hash", "protocol_hash", "evaluation_population_hash"):
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SelectionContractError(f"source_context.{key} must be a lowercase SHA-256 hex digest")
    if value["dataset_manifest_hash"] is not None and (
            not isinstance(value["dataset_manifest_hash"], str)
            or len(value["dataset_manifest_hash"]) != 64
            or any(c not in "0123456789abcdef" for c in value["dataset_manifest_hash"])):
        raise SelectionContractError("dataset_manifest_hash must be null or a lowercase SHA-256 digest")
    for key in ("source_revision", "evaluation_population_id"):
        _nonempty_string(value[key], f"source_context.{key}")
    if not isinstance(value["target_contract"], Mapping) or not value["target_contract"]:
        raise SelectionContractError("source_context.target_contract must be a nonempty object")
    target = value["target_contract"]
    if target.get("kind") != "win_probability" or target.get("label_column", "target_probability") != "target_probability":
        raise SelectionContractError("source target contract must be win_probability on target_probability")
    # Round-trip rejects non-JSON objects, NaN and mutable custom encoders.
    return json.loads(canonical_json_bytes(value))


def _validate_frame(frame: pd.DataFrame, graph: Mapping[str, Any], *,
                    labels_through: pd.Timestamp) -> pd.Series:
    date_column, horse_column = graph["date_column"], graph["horse_column"]
    required = {date_column, horse_column, "race_id", "field_size", "target_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise SelectionContractError(f"selection source frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise SelectionContractError("selection source frame cannot be empty")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    if dates.isna().any():
        raise SelectionContractError("selection source dates cannot be missing")
    normalized = dates.dt.normalize()
    keys = frame[["race_id", horse_column]]
    if keys.isna().any().any() or frame.duplicated(["race_id", horse_column]).any():
        raise SelectionContractError("selection rows require unique nonmissing race/runner keys")
    if normalized.groupby(frame["race_id"], sort=False).nunique().max() != 1:
        raise SelectionContractError("a race cannot cross whole-date boundaries")
    prefix_mask = normalized <= labels_through
    prefix_frame = frame.loc[prefix_mask]
    labels = pd.to_numeric(prefix_frame["target_probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise SelectionContractError("prefix target probabilities must be finite and nonnegative")
    totals = pd.Series(labels).groupby(prefix_frame["race_id"].to_numpy(), sort=False).sum().to_numpy()
    if not np.allclose(totals, 1.0, rtol=0, atol=1e-8):
        raise SelectionContractError("each complete race must have normalized target probabilities")
    sizes = frame.groupby("race_id", sort=False).size().to_numpy()
    declared = pd.to_numeric(frame.groupby("race_id", sort=False)["field_size"].first(),
                             errors="coerce").to_numpy()
    if not np.array_equal(sizes, declared):
        raise SelectionContractError("prefix selection requires complete race fields")
    return normalized


def _split_prefix(frame: pd.DataFrame, graph: Mapping[str, Any], protocol: Mapping[str, Any],
                  future_oof_start: Any) -> PrefixSplit:
    cutoff = _timestamp_boundary(protocol["selection_cutoff"], "selection_cutoff")
    validation_start = _timestamp_boundary(protocol["validation_start"], "validation_start")
    dates = _validate_frame(frame, graph, labels_through=cutoff)
    future_start = _timestamp_boundary(future_oof_start, "future_oof_start")
    if not cutoff < future_start:
        raise SelectionContractError("future_oof_start must be strictly after selection_cutoff")
    if future_start not in set(dates):
        raise SelectionContractError("future_oof_start must identify a whole meeting date in the source frame")
    if cutoff not in set(dates):
        raise SelectionContractError("selection_cutoff must identify a whole meeting date in the source frame")
    if validation_start not in set(dates):
        raise SelectionContractError("validation_start must identify a whole meeting date in the source frame")
    if dates.max() < future_start:
        raise SelectionContractError("source frame must include chronology through future_oof_start")
    train_mask = dates < validation_start
    validation_mask = (dates >= validation_start) & (dates <= cutoff)
    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()
    train_dates = set(dates.loc[train_mask])
    validation_dates = set(dates.loc[validation_mask])
    if train_dates & validation_dates:
        raise SelectionContractError("training and validation meetings must be date-disjoint")
    if len(train_dates) < protocol["min_prefix_train_meetings"]:
        raise SelectionContractError("prefix training history is below its explicit meeting warmup")
    if len(validation_dates) < protocol["min_prefix_validation_meetings"]:
        raise SelectionContractError("prefix validation history is below its explicit meeting minimum")
    if train.empty or validation.empty:
        raise SelectionContractError("prefix training and validation frames must both be nonempty")
    return PrefixSplit(train, validation, dates, graph["date_column"], cutoff, validation_start)


def _prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = {"target_probability", "target_win", "target_place", "result", "finishing_status",
                 "finish_position", "dividend", "final_odds"}
    return frame.drop(columns=[column for column in forbidden if column in frame]).copy()


def _validate_probability_vector(values: Any, frame: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.shape != (len(frame),) or not np.isfinite(probability).all() or np.any(probability < 0):
        raise SelectionContractError("candidate graph returned invalid win probabilities")
    totals = pd.Series(probability).groupby(frame["race_id"].to_numpy(), sort=False).sum().to_numpy()
    if not np.allclose(totals, 1.0, rtol=0, atol=1e-8):
        raise SelectionContractError("candidate graph probabilities must normalize each complete race")
    return probability


def _meeting_log_loss(probability: np.ndarray, labelled: pd.DataFrame,
                      date_column: str = "date") -> float:
    losses = -labelled["target_probability"].to_numpy(float) * np.log(np.clip(probability, 1e-12, 1.0))
    race_losses = pd.Series(losses).groupby(labelled["race_id"].to_numpy(), sort=False).sum()
    race_dates = labelled.groupby("race_id", sort=False)[date_column].first()
    meeting_losses = race_losses.groupby(pd.to_datetime(race_dates).dt.normalize(), sort=False).mean()
    score = float(meeting_losses.mean())
    if not np.isfinite(score):
        raise SelectionContractError("candidate graph produced a nonfinite validation loss")
    return score


def _validate_selected_graph_for_fit(graph: Mapping[str, Any]) -> None:
    """Guard against any graph feature that can recursively select or fake lineage."""
    for node in graph["nodes"]:
        if node["kind"] not in ALLOWED_NODE_KINDS:
            raise SelectionContractError(f"unsupported candidate node: {node['kind']}")
        if node["kind"] == "feature_view":
            continue
        if node["kind"] == "estimator":
            params = node.get("parameters", {})
            model_kind = params.get("model_kind", "benter_conditional_logit")
            if model_kind in {"gaussian_probit"}:
                raise SelectionContractError("candidate model requires an unsupported custom fitting path")


def _protocol_digest(protocol: Mapping[str, Any]) -> str:
    return _sha256(protocol)


def _prefix_contract(split: PrefixSplit) -> dict[str, Any]:
    train_dates = pd.to_datetime(split.train[split.date_column]).dt.normalize()
    validation_dates = pd.to_datetime(split.validation[split.date_column]).dt.normalize()
    return {
        "training_fingerprint": frame_fingerprint(split.train),
        "validation_fingerprint": frame_fingerprint(split.validation),
        "training_rows": len(split.train), "validation_rows": len(split.validation),
        "training_races": int(split.train.race_id.nunique()),
        "validation_races": int(split.validation.race_id.nunique()),
        "training_meetings": int(train_dates.nunique()),
        "validation_meetings": int(validation_dates.nunique()),
        "training_start": train_dates.min().date().isoformat(),
        "training_end": train_dates.max().date().isoformat(),
        "validation_start": validation_dates.min().date().isoformat(),
        "validation_end": validation_dates.max().date().isoformat(),
    }


def _artifact_payload(*, protocol: dict[str, Any], source: dict[str, Any],
                      feature_schema: FeatureSchema, prefix: dict[str, Any],
                      candidate_results: list[dict[str, Any]], selected_id: str,
                      selected_graph: dict[str, Any], future_oof_start: pd.Timestamp,
                      optuna_version: str) -> dict[str, Any]:
    graph_by_id = {candidate["candidate_id"]: candidate["graph"]
                   for candidate in protocol["candidate_graphs"]}
    payload = {
        "schema_version": 1,
        "selection_mode": "nested_selection",
        "selection_cutoff": protocol["selection_cutoff"],
        "validation_start": protocol["validation_start"],
        "future_oof_start": future_oof_start.date().isoformat(),
        "objective": protocol["objective"],
        "tie_rule": "minimum_loss_then_lexicographic_candidate_id",
        "protocol_sha256": _protocol_digest(protocol),
        "source": source,
        "feature_schema": _schema_contract(feature_schema),
        "feature_schema_sha256": _sha256(_schema_contract(feature_schema)),
        "prefix": prefix,
        "candidate_graphs": [{"candidate_id": c["candidate_id"],
                              "graph": graph_by_id[c["candidate_id"]],
                              "graph_sha256": _sha256(graph_by_id[c["candidate_id"]])}
                             for c in protocol["candidate_graphs"]],
        "candidate_results": candidate_results,
        "selected_candidate_id": selected_id,
        "selected_graph": selected_graph,
        "selected_graph_sha256": _sha256(selected_graph),
        "optuna": {"version": optuna_version, "sampler": "GridSampler",
                   "seed": protocol["seed"], "trial_budget": protocol["selection_trial_budget"],
                   "trials": len(candidate_results)},
        "scope": {"selection_scope": "outer_training_prefix_development",
                  "independent_confirmation": False, "promotion_allowed": False},
    }
    selection_id, digest = _selection_identity(payload)
    payload["selection_id"], payload["selection_sha256"] = selection_id, digest
    return payload


def _validate_artifact_payload(payload: Mapping[str, Any], *, protocol: dict[str, Any],
                               feature_schema: FeatureSchema, source: dict[str, Any],
                               split: PrefixSplit, future_oof_start: pd.Timestamp) -> dict[str, Any]:
    value = _require_mapping(payload, "selection artifact")
    _exact_keys(value, ARTIFACT_KEYS, "selection artifact")
    if value["schema_version"] != 1 or value["selection_mode"] != "nested_selection":
        raise SelectionContractError("unsupported selection artifact schema/mode")
    if (value["objective"] != protocol["objective"]
            or value["tie_rule"] != "minimum_loss_then_lexicographic_candidate_id"):
        raise SelectionContractError("selection artifact objective or tie rule mismatch")
    expected_id, expected_digest = _selection_identity(value)
    if value["selection_sha256"] != expected_digest or value["selection_id"] != expected_id:
        raise SelectionContractError("selection artifact canonical hash or ID mismatch")
    if value["protocol_sha256"] != _protocol_digest(protocol):
        raise SelectionContractError("selection artifact recipe/protocol mismatch")
    if value["source"] != source:
        raise SelectionContractError("selection artifact source pins mismatch")
    schema_contract = _schema_contract(feature_schema)
    if value["feature_schema"] != schema_contract or value["feature_schema_sha256"] != _sha256(schema_contract):
        raise SelectionContractError("selection artifact feature schema mismatch")
    expected_prefix = _prefix_contract(split)
    if value["prefix"] != expected_prefix:
        raise SelectionContractError("selection artifact actual prefix frame fingerprints mismatch")
    if (value["selection_cutoff"] != protocol["selection_cutoff"]
            or value["validation_start"] != protocol["validation_start"]
            or value["future_oof_start"] != future_oof_start.date().isoformat()):
        raise SelectionContractError("selection artifact cutoff/source chronology mismatch")
    expected_candidates = [{"candidate_id": c["candidate_id"], "graph": c["graph"],
                            "graph_sha256": _sha256(c["graph"])} for c in protocol["candidate_graphs"]]
    if value["candidate_graphs"] != expected_candidates:
        raise SelectionContractError("selection artifact candidate recipe/graph mismatch")
    ids = [item.get("candidate_id") for item in value["candidate_results"]
           if isinstance(item, Mapping)]
    expected_ids = [item["candidate_id"] for item in protocol["candidate_graphs"]]
    if len(ids) != len(value["candidate_results"]) or sorted(ids) != sorted(expected_ids):
        raise SelectionContractError("selection artifact trial set is incomplete or duplicated")
    candidate_hashes = {item["candidate_id"]: item["graph_sha256"] for item in expected_candidates}
    trial_numbers = []
    for result in value["candidate_results"]:
        expected_result_keys = {
            "candidate_id", "trial_number", "mean_meeting_log_loss", "training_fingerprint",
            "validation_fingerprint", "training_cutoff", "validation_start", "graph_sha256",
            "component_fits", "effective_parameters", "ordered_feature_columns",
            "dependency_versions",
        }
        if set(result) != expected_result_keys:
            raise SelectionContractError("selection candidate result fields are invalid")
        if (isinstance(result["trial_number"], bool) or not isinstance(result["trial_number"], int)
                or result["trial_number"] < 0):
            raise SelectionContractError("selection candidate trial number is invalid")
        trial_numbers.append(result["trial_number"])
        score = result.get("mean_meeting_log_loss")
        if (isinstance(score, bool) or not isinstance(score, (int, float, np.number))
                or not np.isfinite(score)):
            raise SelectionContractError("selection candidate score must be a finite number")
        if (result.get("training_fingerprint") != expected_prefix["training_fingerprint"]
                or result.get("validation_fingerprint") != expected_prefix["validation_fingerprint"]
                or result.get("graph_sha256") != candidate_hashes.get(result["candidate_id"])
                or result.get("training_cutoff") != expected_prefix["training_end"]
                or result.get("validation_start") != expected_prefix["validation_start"]):
            raise SelectionContractError("selection candidate result has invalid prefix lineage/score")
    if sorted(trial_numbers) != list(range(len(expected_ids))):
        raise SelectionContractError("selection candidate Optuna trial numbers are not a complete unique sequence")
    winner = min(value["candidate_results"],
                 key=lambda item: (float(item["mean_meeting_log_loss"]), item["candidate_id"]))
    if value["selected_candidate_id"] != winner["candidate_id"]:
        raise SelectionContractError("selection artifact winner does not match declared objective/tie rule")
    selected = next((c["graph"] for c in protocol["candidate_graphs"]
                     if c["candidate_id"] == winner["candidate_id"]), None)
    if (selected is None or value["selected_graph"] != selected
            or value["selected_graph_sha256"] != _sha256(selected)):
        raise SelectionContractError("selection artifact selected graph mismatch")
    if value["scope"] != {"selection_scope": "outer_training_prefix_development",
                           "independent_confirmation": False, "promotion_allowed": False}:
        raise SelectionContractError("selection artifact scientific scope flags are invalid")
    optuna = value["optuna"]
    if (optuna.get("sampler") != "GridSampler" or optuna.get("seed") != protocol["seed"]
            or optuna.get("trial_budget") != protocol["selection_trial_budget"]
            or optuna.get("trials") != len(expected_ids) or not optuna.get("version")):
        raise SelectionContractError("selection artifact Optuna trial contract mismatch")
    return value


def validate_selection_artifact(payload: Mapping[str, Any], *, protocol: Mapping[str, Any],
                                feature_schema: FeatureSchema, source_context: Mapping[str, Any],
                                source_frame: pd.DataFrame, future_oof_start: Any,
                                expected_output: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read-only validation against expected inputs; this function never fits."""
    normalized = validate_selection_protocol(protocol, feature_schema=feature_schema,
                                             expected_output=expected_output)
    candidates = normalized["candidate_graphs"]
    split = _split_prefix(source_frame, candidates[0]["graph"], normalized, future_oof_start)
    source = _source_context(source_context)
    return _validate_artifact_payload(payload, protocol=normalized, feature_schema=feature_schema,
                                      source=source, split=split,
                                      future_oof_start=_timestamp_boundary(future_oof_start, "future_oof_start"))


def _strict_json_loads(data: bytes) -> dict[str, Any]:
    def pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SelectionContractError(f"duplicate JSON key in selection artifact: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda constant: (_ for _ in ()).throw(
                               SelectionContractError(f"nonfinite JSON constant: {constant}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionContractError(f"invalid selection artifact JSON: {exc}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise SelectionContractError("selection artifact is not in canonical byte form")
    return value


def load_selection_artifact(path: str | Path, *, protocol: Mapping[str, Any],
                            feature_schema: FeatureSchema, source_context: Mapping[str, Any],
                            source_frame: pd.DataFrame, future_oof_start: Any,
                            expected_output: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load and validate a saved selection artifact without fitting or mutating."""
    artifact_path = Path(path)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise SelectionContractError("selection artifact path must be an existing regular file")
    payload = _strict_json_loads(artifact_path.read_bytes())
    return validate_selection_artifact(payload, protocol=protocol, feature_schema=feature_schema,
        source_context=source_context, source_frame=source_frame,
        future_oof_start=future_oof_start, expected_output=expected_output)


@storage_writer(stage="nested_pool_selection_artifact",
                output=lambda args: Path(args["path"]).parent,
                estimate=lambda args: max(4096, len(canonical_json_bytes(args["payload"])) * 2))
def write_selection_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish canonical bytes once; identical retries are idempotent."""
    destination = Path(path)
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise SelectionContractError("selection artifact parent must be an existing real directory")
    data = canonical_json_bytes(payload)
    # Validate identity shape before writing; full expected-source validation is
    # performed by the caller before and after publication.
    normalized = _strict_json_loads(data)
    expected_id, expected_digest = _selection_identity(normalized)
    if normalized.get("selection_id") != expected_id or normalized.get("selection_sha256") != expected_digest:
        raise SelectionContractError("refusing to write selection artifact with invalid identity")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                          dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    primary = None
    try:
        with reserved_output_file(temporary, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != data:
                raise SelectionContractError("immutable selection artifact path already has different bytes")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"selection artifact temporary cleanup also failed: {cleanup!r}")


def select_prefix_graph(protocol: Mapping[str, Any], *, data: pd.DataFrame,
                        feature_schema: FeatureSchema, source_context: Mapping[str, Any],
                        future_oof_start: Any, artifact_path: str | Path,
                        expected_output: Mapping[str, Any] | None = None) -> SelectionResult:
    """Select one candidate graph on a frozen prefix using the shared graph fitter.

    Only prefix train rows reach ``fit_pipeline_graph``. Validation predictions
    are requested from the fitted graph after its cutoff, with label columns
    removed; labels are used only by the equal-meeting loss after prediction.
    """
    normalized = validate_selection_protocol(protocol, feature_schema=feature_schema,
                                             expected_output=expected_output)
    candidates = normalized["candidate_graphs"]
    split = _split_prefix(data, candidates[0]["graph"], normalized, future_oof_start)
    source = _source_context(source_context)
    prefix = _prefix_contract(split)
    future_start = _timestamp_boundary(future_oof_start, "future_oof_start")
    schema_contract = _schema_contract(feature_schema)
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]

    try:
        import optuna
        from optuna.samplers import GridSampler
    except Exception as exc:  # pragma: no cover - environment/setup failure
        raise RuntimeError("Optuna is required for executable nested pool selection") from exc

    from .pipeline_graph import fit_pipeline_graph

    study = optuna.create_study(direction="minimize",
                                sampler=GridSampler({"candidate_id": candidate_ids},
                                                    seed=normalized["seed"]))
    result_by_id: dict[str, dict[str, Any]] = {}

    def objective(trial):
        candidate_id = trial.suggest_categorical("candidate_id", candidate_ids)
        if candidate_id in result_by_id:
            raise SelectionContractError("Optuna GridSampler repeated a candidate in protocol v1")
        candidate = next(item for item in candidates if item["candidate_id"] == candidate_id)
        graph = candidate["graph"]
        _validate_selected_graph_for_fit(graph)
        fitted = fit_pipeline_graph(graph, split.train, feature_schema=feature_schema,
                                    seed=normalized["seed"])
        validation_input = _prediction_frame(split.validation)
        probabilities = _validate_probability_vector(fitted.predict_proba(validation_input),
                                                    split.validation)
        loss = _meeting_log_loss(probabilities, split.validation, graph["date_column"])
        training_cutoff = pd.to_datetime(split.train[graph["date_column"]]).max().normalize()
        validation_start = pd.to_datetime(split.validation[graph["date_column"]]).min().normalize()
        if training_cutoff >= validation_start:
            raise SelectionContractError("candidate fit did not precede prefix validation")
        stable_fit_records = []
        for record in fitted.fit_report["component_fits"]:
            stable_fit_records.append({key: value for key, value in record.items()
                                       if key != "wall_seconds"})
        candidate_result = {
            "candidate_id": candidate_id,
            "trial_number": int(trial.number),
            "mean_meeting_log_loss": loss,
            "training_fingerprint": prefix["training_fingerprint"],
            "validation_fingerprint": prefix["validation_fingerprint"],
            "training_cutoff": training_cutoff.date().isoformat(),
            "validation_start": validation_start.date().isoformat(),
            "graph_sha256": _sha256(graph),
            "component_fits": stable_fit_records,
            "effective_parameters": [record.get("effective_parameters", {})
                                     for record in fitted.fit_report["component_fits"]],
            "ordered_feature_columns": [record.get("features", [])
                                        for record in fitted.fit_report["component_fits"]],
            "dependency_versions": fitted.fit_report.get("dependency_versions", {}),
        }
        result_by_id[candidate_id] = candidate_result
        trial.set_user_attr("candidate_graph_sha256", candidate_result["graph_sha256"])
        trial.set_user_attr("training_fingerprint", candidate_result["training_fingerprint"])
        trial.set_user_attr("validation_fingerprint", candidate_result["validation_fingerprint"])
        return loss

    study.optimize(objective, n_trials=normalized["selection_trial_budget"], n_jobs=1,
                   show_progress_bar=False, catch=())
    if set(result_by_id) != set(candidate_ids) or len(study.trials) != len(candidate_ids):
        raise SelectionContractError("Optuna did not evaluate each declared candidate exactly once")
    results = sorted(result_by_id.values(), key=lambda item: item["candidate_id"])
    winner = min(results, key=lambda item: (item["mean_meeting_log_loss"], item["candidate_id"]))
    graph_by_id = {candidate["candidate_id"]: candidate["graph"] for candidate in candidates}
    payload = _artifact_payload(protocol=normalized, source=source, feature_schema=feature_schema,
        prefix=prefix, candidate_results=results, selected_id=winner["candidate_id"],
        selected_graph=graph_by_id[winner["candidate_id"]], future_oof_start=future_start,
        optuna_version=optuna.__version__)
    validated = _validate_artifact_payload(payload, protocol=normalized, feature_schema=feature_schema,
        source=source, split=split, future_oof_start=future_start)
    path = write_selection_artifact(artifact_path, validated)
    reread = load_selection_artifact(path, protocol=normalized, feature_schema=feature_schema,
        source_context=source, source_frame=data, future_oof_start=future_start,
        expected_output=expected_output)
    return SelectionResult(reread["selected_candidate_id"], reread["selected_graph"], reread, path)
