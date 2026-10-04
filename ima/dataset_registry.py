"""Durable requests, whole-race auditing, and atomic immutable dataset builds.

Publication verifies a candidate. It never promotes a campaign or changes V5.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import pandas as pd

from .dataset_specs import DatasetRequest
from .feature_sets import RICH_SCHEMA, NOTEBOOK_RICH_SCHEMA
from .historical_events import HORSE_ID, FAMILY_WINDOWS, build_event_features, normalize_events, official_source
from .rich_features import prepare_rich_runner_dataset
from .speed_features import build_speed_features, finish_seconds, speed_feature_catalog


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _keys(frame):
    return [(str(r), str(h)) for r, h in zip(frame["race_id"], frame["horse_id"])]


def _distribution(frame):
    races = frame.drop_duplicates("race_id").copy()
    races["era"] = races["date"].astype(str).str[:4]
    races["field_size"] = races["race_id"].map(frame.groupby("race_id").size())
    return {column: {str(k): int(v) for k, v in races[column].fillna("unknown").value_counts().items()}
            for column in ("era", "venue", "race_class", "distance", "field_size") if column in races}


def audit_races(source):
    """Exclude malformed whole races; missing optional predictors never select rows."""
    required = {"date", "venue", "race_id", "race_no", "horse_id", "horse_no", "result", "distance", "source", "source_url", "source_body_hash"}
    if required - set(source):
        raise ValueError(f"Missing source columns: {sorted(required - set(source))}")
    exclusions, accepted = [], []
    repeated = source.duplicated(["date", "horse_id"], keep=False)
    repeated_races = set(source.loc[repeated, "race_id"])
    for race_id, group in source.groupby("race_id", sort=False, dropna=False):
        reasons = []
        if race_id in repeated_races:
            reasons.append("duplicate_horse_meeting_identity")
        runner_numbers = pd.to_numeric(group["horse_no"], errors="coerce")
        if not (np.isfinite(runner_numbers) & runner_numbers.ge(1) & runner_numbers.mod(1).eq(0)).all():
            reasons.append("missing_or_invalid_runner_number")
        if not group["venue"].isin({"ST", "HV"}).all():
            reasons.append("unresolved_local_venue")
        if pd.isna(race_id) or group["horse_id"].duplicated().any() or group["horse_no"].duplicated().any():
            reasons.append("duplicate_or_missing_race_runner_identity")
        if not group["horse_id"].map(lambda h: isinstance(h, str) and bool(HORSE_ID.fullmatch(h))).all():
            reasons.append("unresolved_exact_horse_identity")
        if not group["source"].eq("official:hkjc-results").all() or not group["source_url"].map(official_source).all():
            reasons.append("nonofficial_source")
        if not group["source_body_hash"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
            reasons.append("missing_source_body_identity")
        for column in ("date", "venue", "distance", "race_no", "course", "race_class"):
            if column in group and group[column].nunique(dropna=False) != 1:
                reasons.append(f"inconsistent_race_{column}")
        dates = pd.to_datetime(group["date"], errors="coerce")
        if dates.isna().any():
            reasons.append("invalid_race_date")
        if "outcome_available_at" in group:
            explicit = group["outcome_available_at"].notna()
            available = pd.to_datetime(group["outcome_available_at"], utc=True, errors="coerce")
            assumed = dates.dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC") + pd.Timedelta(days=1)
            if (explicit & (available.isna() | available.gt(assumed))).any():
                reasons.append("delayed_result_publication_requires_asof_history_rebuild")
        distance = pd.to_numeric(group["distance"], errors="coerce")
        if not (np.isfinite(distance) & distance.gt(0)).all():
            reasons.append("invalid_distance_metres")
        result = pd.to_numeric(group["result"], errors="coerce")
        nonfinish = group.get("finishing_status", pd.Series("FINISHED", index=group.index)).isin({"PU", "UR", "FE", "DNF", "DISQ", "TNP"})
        if result.eq(1).sum() == 0:
            reasons.append("missing_winner")
        if result.eq(1).sum() > 1:
            reasons.append("unsupported_dead_heat_single_winner_target")
        if ((result.isna() & ~nonfinish) | (result.notna() & (result.lt(1) | result.gt(len(group)) | result.mod(1).ne(0)))).any():
            reasons.append("invalid_finish_order_or_nonfinisher")
        # Raw ties remain preserved upstream; current conditional-logit fitting needs one winner.
        if "finish_time" in group:
            times = group["finish_time"].map(finish_seconds)
            measured = times.notna() & result.notna()
            for _, placed in pd.DataFrame({"position": result[measured], "time": times[measured]}).groupby("position", sort=True):
                if placed["time"].max() - placed["time"].min() > .02:
                    reasons.append("inconsistent_deadheat_times")
            ordered = pd.DataFrame({"position": result[measured], "time": times[measured]}).sort_values("position")
            if ordered["time"].diff().lt(-.02).any():
                reasons.append("individual_time_finish_order_conflict")
        if len(group) < 2:
            reasons.append("incomplete_field")
        if reasons:
            exclusions.append({"race_id": str(race_id), "reasons": sorted(set(reasons)), "runner_keys": _keys(group), "rows": len(group), "category": "quality"})
        else:
            accepted.extend(group.index)
    return source.loc[accepted].copy().reset_index(drop=True), exclusions


def _catalog(features, event_columns, speed_columns, *, target_only=()):
    # Existing registered schemas supply the allowlist, not all numeric columns.
    base = set(RICH_SCHEMA.numeric) | set(NOTEBOOK_RICH_SCHEMA.numeric)
    unsafe_terms = ("market", "rating", "age", "profile", "stakes", "sectional", "trial", "trackwork", "veterinary", "movement", "injury", "fracture", "surgery")
    base = {name for name in base if not any(term in name for term in unsafe_terms)}
    columns = (base | set(event_columns) | set(speed_columns)) & set(features)
    current_dependencies = {
        "draw": {"draw", "draw_bias_starts", "draw_bias_win_rate", "draw_bias_top3_rate"},
        "actual_weight": {"actual_weight", "carried_weight_rank", "carried_weight_change", "carried_weight_change_rank", "poly_weight_change_distance"},
        "declared_weight": {"declared_weight", "body_weight_rank", "body_weight_change", "body_weight_change_pct", "weight_change_per_day", "poly_weight_change_distance"},
        "distance": {"distance", "distance_change", "distance_band_starts", "distance_band_win_rate", "distance_band_top3_rate", "distance_experience_rank"},
    }
    tainted = set(target_only)
    for name in target_only:
        tainted.update(current_dependencies.get(name, ()))
        tainted.update(column for column in columns if column.startswith("poly_") and name in column)
        if name == "distance":
            tainted.update(column for column in columns if "same_distance" in column)
    columns -= tainted
    catalog = {}
    speed = speed_feature_catalog()
    for name in sorted(columns):
        if not pd.api.types.is_numeric_dtype(features[name]):
            continue
        catalog[name] = speed.get(name, {"family": "official_events" if name in event_columns else "registered_rich_history",
            "unit": "days" if "days" in name else "indicator" if name.endswith(("available", "coverage")) else "source_specific",
            "cutoff_rule": "occurrence_and_availability_before_cutoff" if name in event_columns else "pre_race_or_previous_race_registered_formula",
            "fit_required": False}) | {"nonmissing": int(features[name].notna().sum()), "total": len(features),
            "distinct": int(features[name].nunique(dropna=True)), "missing": int(features[name].isna().sum())}
        unit = "1"
        if name in speed:
            unit = speed[name]["unit"]
            if unit in {"count", "indicator"}:
                unit = "1"
        elif name.endswith("_days") or "days_since" in name:
            unit = "days"
        elif "weight" in name and not any(term in name for term in ("pct", "rank")):
            unit = "lb/days" if name == "weight_change_per_day" else "lb"
        elif name == "distance" or name == "last_distance" or name == "distance_change" or name.startswith("total_distance_"):
            unit = "m"
        elif "finish_time" in name:
            unit = "s"
        elif "lengths_behind" in name:
            unit = "lengths"
        unit = {"poly_distance_sq": "m^2", "poly_draw_distance": "m",
                "poly_weight_change_distance": "lb*m", "prize": "HKD"}.get(name, unit)
        catalog[name].update(unit=unit, temporal_scope="pre_race", target_tainted=False,
                             eligible=True, dtype="float64", source_family=catalog[name]["family"],
                             availability_proof="materialized cutoff-safe event views or completed prior-meeting history")
    return catalog


class DatasetRegistry:
    def __init__(self, root, *, feature_registry=None):
        self.root = Path(root).resolve()
        self.feature_registry = Path(feature_registry) if feature_registry else self.root / "feature-definitions"
        for name in ("requests", "datasets", "locks", "staging"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self, request_id):
        with (self.root / "locks" / f"{request_id}.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _state_path(self, request_id):
        if not isinstance(request_id, str) or not __import__("re").fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request_id):
            raise ValueError("Invalid request ID")
        return self.root / "requests" / f"{request_id}.json"

    def get(self, request_id):
        return json.loads(self._state_path(request_id).read_text())

    def _transition(self, state, status, **fields):
        state.update(fields, status=status, updated_at=datetime.now(timezone.utc).isoformat())
        state.setdefault("history", []).append({"status": status, "at": state["updated_at"]})
        _write_json(self._state_path(state["request"]["request_id"]), state)

    def submit(self, request):
        request = request if isinstance(request, DatasetRequest) else DatasetRequest.model_validate(request)
        path = self._state_path(request.request_id)
        with self._lock(request.request_id):
            if path.exists():
                state = self.get(request.request_id)
                if state["request_fingerprint"] != request.fingerprint():
                    raise ValueError("Request ID already binds a different immutable specification")
                return state
            state = {"request": request.model_dump(mode="json"), "request_fingerprint": request.fingerprint()}
            self._transition(state, "requested")
            return state

    def dataset_path(self, dataset_id):
        if not isinstance(dataset_id, str) or not __import__("re").fullmatch(r"dataset-[0-9a-f]{64}", dataset_id):
            raise ValueError("Invalid dataset ID")
        return self.root / "datasets" / dataset_id

    def verify(self, dataset_id):
        path = self.dataset_path(dataset_id)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["dataset_id"] != dataset_id or manifest["status"] != "verified":
            raise ValueError("Unverified dataset identity")
        if "dataset-" + _digest(manifest["identity"]) != dataset_id:
            raise ValueError("Dataset identity payload changed")
        if manifest["identity"].get("predictor_catalog_sha256") != _digest(manifest["predictor_catalog"]):
            raise ValueError("Dataset predictor catalog identity changed")
        if manifest["identity"].get("eligible_categorical_predictors_sha256") != _digest(manifest.get("eligible_categorical_predictors")):
            raise ValueError("Dataset categorical predictor identity changed")
        checksum = path / "manifest.sha256"
        if not checksum.is_file() or checksum.read_text().strip() != file_sha256(path / "manifest.json"):
            raise ValueError("Dataset manifest checksum mismatch")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or file_sha256(path / name) != digest:
                raise ValueError(f"Dataset checksum mismatch: {name}")
        return manifest

    def _confirmation_races(self, source, request):
        """The registry, not each planner request, owns the protected race set."""
        parent_races = set()
        if request.parent_dataset_id:
            self.verify(request.parent_dataset_id)
            parent_keys = json.loads((self.dataset_path(request.parent_dataset_id) / "confirmation_keys.json").read_text())
            parent_races.update(str(key[0]) for key in parent_keys)
        order = source[["race_id", "date"]].drop_duplicates("race_id").copy()
        order["date"] = pd.to_datetime(order["date"], errors="coerce")
        order = order.dropna(subset=["date"]).sort_values(["date", "race_id"], kind="stable")
        requested = set(request.protocol.final_confirmation_race_ids) | parent_races
        if request.protocol.final_confirmation_start:
            requested.update(order.loc[order.date.ge(pd.Timestamp(request.protocol.final_confirmation_start)), "race_id"].astype(str))
        count = request.protocol.final_confirmation_races
        if count:
            if len(order) <= count:
                raise ValueError("Insufficient races after final confirmation quarantine")
            boundary = order.iloc[-count]["date"]
            requested.update(order.loc[order.date.ge(boundary), "race_id"].astype(str))
        path = self.root / "protected-confirmation-races.json"
        with self._lock("confirmation-policy"):
            if path.exists():
                protected = set(json.loads(path.read_text())["race_ids"])
            else:
                protected = set(requested)
                _write_json(path, {"race_ids": sorted(protected), "policy": "operator seed; never released by dataset requests"})
        return protected | requested

    def build(self, request_id, *, source_snapshot, raw_manifest):
        """Resume a submitted request. Failures never overwrite a verified artifact.

        raw_manifest must identify the acquisition manifest by SHA256 or its
        declared manifest_id. Raw blobs are verified when locally provided;
        their absence is explicitly reported as delegated upstream raw replay.
        """
        self._state_path(request_id)
        with self._lock(request_id):
            state = self.get(request_id)
            request = DatasetRequest.model_validate(state["request"])
            if state["status"] == "verified":
                return self.verify(state["dataset_id"])
            self._transition(state, "building", error=None)
            started = time.monotonic()
            try:
                snapshot, raw_path = Path(source_snapshot).resolve(), Path(raw_manifest).resolve()
                source_manifest = json.loads((snapshot / "manifest.json").read_text())
                raw = json.loads(raw_path.read_text())
                raw_hash = file_sha256(raw_path)
                identifiers = {raw_hash, raw.get("manifest_id"), raw.get("raw_corpus_manifest_id"), raw.get("snapshot_id")} if isinstance(raw, dict) else {raw_hash}
                if request.raw_corpus_manifest_id not in identifiers:
                    raise ValueError("Raw manifest identity does not match DatasetRequest")
                if "HKJC-only" not in source_manifest.get("source_policy", ""):
                    raise ValueError("Source snapshot is not official-only")
                source_hashes = {}
                for name in ("runners.parquet", "events.jsonl"):
                    expected = source_manifest.get("files", {}).get(name)
                    if not expected or file_sha256(snapshot / name) != expected:
                        raise ValueError(f"Source snapshot checksum mismatch: {name}")
                    source_hashes[name] = expected
                source = pd.read_parquet(snapshot / "runners.parquet")
                if len(source) != source_manifest.get("rows") or source["race_id"].nunique() != source_manifest.get("races"):
                    raise ValueError("Source snapshot row/race inventory mismatch")
                protected_races = self._confirmation_races(source, request)
                code = {name: file_sha256(Path(__file__).parent / name) for name in ("dataset_specs.py", "dataset_registry.py", "historical_events.py", "speed_features.py", "rich_features.py", "data.py", "feature_sets.py", "feature_definitions.py", "feature_expressions.py", "research_targets.py", "research_evaluation.py")}
                identity = {"request": request.fingerprint(), "source_manifest": file_sha256(snapshot / "manifest.json"), "source_files": source_hashes, "raw_manifest": raw_hash, "code": code,
                            "protected_confirmation_races": sorted(protected_races),
                            "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "pydantic")}}
                dataset_id = "dataset-" + _digest(identity)
                destination = self.dataset_path(dataset_id)
                if destination.exists():
                    manifest = self.verify(dataset_id)
                    self._transition(state, "verified", dataset_id=dataset_id, artifact_path=str(destination))
                    return manifest
                staging = self.root / "staging" / dataset_id
                if staging.exists():
                    # This is our lock-owned incomplete build, never a published candidate.
                    shutil.rmtree(staging)
                staging.mkdir()
                confirmation_keys = _keys(source.loc[source["race_id"].isin(protected_races)])
                clean, exclusions = audit_races(source)
                population = request.race_population_spec
                keep = pd.Series(True, index=clean.index)
                dates = pd.to_datetime(clean["date"])
                if population.start_date:
                    keep &= dates.ge(pd.Timestamp(population.start_date))
                if population.end_date:
                    keep &= dates.le(pd.Timestamp(population.end_date))
                if population.venues:
                    keep &= clean["venue"].isin(population.venues)
                if request.history_window:
                    keep &= dates.ge(dates.max() - pd.Timedelta(days=request.history_window))
                for race_id, group in clean.loc[~keep].groupby("race_id", sort=False):
                    exclusions.append({"race_id": str(race_id), "reasons": ["explicit_population_hypothesis"], "runner_keys": _keys(group), "rows": len(group), "category": "population"})
                clean = clean.loc[keep].sort_values(["date", "race_no", "race_id", "horse_no"], kind="stable").reset_index(drop=True)
                confirmation = clean["race_id"].isin(protected_races)
                for race_id, group in clean.loc[confirmation].groupby("race_id", sort=False):
                    exclusions.append({"race_id": str(race_id), "reasons": ["final_confirmation_quarantine"], "runner_keys": _keys(group), "rows": len(group), "category": "confirmation"})
                clean = clean.loc[~confirmation].reset_index(drop=True)
                if clean.empty:
                    raise ValueError("No clean development races")
                # Rebuild after whole-race exclusion: legacy source features may have partial fields.
                features = prepare_rich_runner_dataset(clean, strict_before_meeting=True)
                missing = set(_keys(clean)) - set(_keys(features))
                if missing:
                    dropped_races = {r for r, _ in missing}
                    for race_id, group in clean.loc[clean["race_id"].isin(dropped_races)].groupby("race_id"):
                        exclusions.append({"race_id": str(race_id), "reasons": ["legacy_rich_feature_contract_rejected_whole_race"], "runner_keys": _keys(group), "rows": len(group), "category": "builder_contract"})
                    clean = clean.loc[~clean["race_id"].isin(dropped_races)].reset_index(drop=True)
                    features = prepare_rich_runner_dataset(clean, strict_before_meeting=True)
                if set(_keys(clean)) != set(_keys(features)) or len(clean) != len(features):
                    raise ValueError("Feature builder changed clean population")
                clean = clean.set_index(["race_id", "horse_id"]).loc[list(zip(features["race_id"], features["horse_id"]))].reset_index()
                with (snapshot / "events.jsonl").open() as stream:
                    events = normalize_events(json.loads(line) for line in stream if line.strip())
                coverage = raw.get("event_coverage", []) if isinstance(raw, dict) else []
                event_features = build_event_features(features, events, coverage, policy=request.event_policy_id, retrospective_lags=request.retrospective_lags)
                speeds = build_speed_features(features, events, policy=request.event_policy_id, retrospective_lags=request.retrospective_lags)
                # Disable the legacy global-date coverage owners, then replace with audited joins.
                legacy = [c for c in features if c.startswith(("trackwork_", "veterinary_", "movements_", "movement_", "injury_events", "fracture_events", "surgery_events", "sectionals_"))]
                legacy += [c for c in ("trials_90d", "days_since_trial", "last_trial_placing", "last_trial_speed", "days_since_trackwork", "days_since_veterinary", "days_since_movement", "days_since_hk_arrival", "last_sectional_time", "last_sectional_position", "barrier_available") if c in features]
                features = features.drop(columns=list(set(legacy)), errors="ignore")
                features = pd.concat([features.drop(columns=event_features.columns, errors="ignore"), event_features, speeds], axis=1)
                numeric = features.select_dtypes(include="number").columns
                nonfinite = {name: int(np.isinf(features[name].to_numpy(float)).sum()) for name in numeric}
                nonfinite = {name: count for name, count in nonfinite.items() if count}
                features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
                catalog = _catalog(features, event_features.columns, speeds.columns,
                                   target_only=source_manifest.get("target_only_columns", ()))
                categorical = [name for name in ("venue", "course", "going", "jockey_key", "trainer_key")
                    if name in features and name not in source_manifest.get("target_only_columns", ())]
                for raw, derived in (("jockey_id", "jockey_key"), ("trainer_id", "trainer_key")):
                    if raw in source_manifest.get("target_only_columns", ()) and derived in categorical:
                        categorical.remove(derived)
                formula_report = {"definitions": [], "definition_ids": []}
                if request.feature_definition_ids:
                    from .feature_definitions import FeatureRegistry, evaluate_formulas
                    definitions = [FeatureRegistry(self.feature_registry).get(identifier) for identifier in request.feature_definition_ids]
                    metadata_keys = {"unit", "dtype", "temporal_scope", "target_tainted", "available_at_column", "source_family"}
                    metadata = {name: {key: value for key, value in entry.items() if key in metadata_keys} for name, entry in catalog.items()}
                    features, _, formula_report = evaluate_formulas(features, definitions, metadata, cutoff=pd.to_datetime(features.date, utc=True))
                    for definition in definitions:
                        for name in (definition.name, definition.feature_id):
                            catalog[name] = {"unit": definition.unit, "dtype": definition.dtype, "temporal_scope": "pre_race",
                                "eligible": True, "target_tainted": False, "source_family": "validated_formula",
                                "definition_id": definition.content_id(), "input_refs": list(definition.input_refs),
                                "nonmissing": int(features[name].notna().sum()), "missing": int(features[name].isna().sum()),
                                "total": len(features), "distinct": int(features[name].nunique(dropna=True))}
                identity["predictor_catalog_sha256"] = _digest(catalog)
                identity["eligible_categorical_predictors_sha256"] = _digest(categorical)
                dataset_id = "dataset-" + _digest(identity)
                destination = self.dataset_path(dataset_id)
                if destination.exists():
                    manifest = self.verify(dataset_id)
                    self._transition(state, "verified", dataset_id=dataset_id, artifact_path=str(destination))
                    return manifest
                excluded_keys = {tuple(key) for row in exclusions for key in row["runner_keys"]}
                preserved = set(_keys(clean))
                source_keys = set(_keys(source))
                if source_keys - preserved - excluded_keys:
                    raise ValueError("Unaudited population loss")
                if set(request.expected_preserved_keys) - preserved - excluded_keys:
                    raise ValueError("Expected preserved keys absent from source or exclusions")
                self._transition(state, "validating", dataset_id=dataset_id)
                from .research_evaluation import build_expanding_folds
                spec = request.protocol
                folds = build_expanding_folds(features, min_train_races=spec.min_train_races, calibration_races=spec.calibration_races,
                    score_races=spec.score_races, max_folds=spec.max_folds, whole_meeting_boundaries=spec.whole_meeting_boundaries,
                    fold_selection=spec.fold_selection)
                if not folds:
                    raise ValueError("No valid chronological protocol folds")
                score_keys = sorted({race for fold in folds for race in fold.score_race_ids})
                comparison = {"availability_tier": request.event_policy_id, "retrospective_lags": request.retrospective_lags,
                    "protocol_id": request.protocol_id, "protocol_spec": spec.model_dump(mode="json"), "evaluation_population_id": request.evaluation_population_id,
                    "score_population_sha256": _digest(score_keys), "probability_basis": "fundamental_pre_race", "targets": list(request.target_contracts)}
                if request.budget.get("wall_seconds") and time.monotonic() - started > request.budget["wall_seconds"]:
                    raise ValueError("Dataset build exceeded explicit wall_seconds budget")
                clean.to_parquet(staging / "runners.parquet", index=False)
                features.to_parquet(staging / "features.parquet", index=False)
                persisted_events = events.copy()
                persisted_events["typed_values"] = persisted_events["typed_values"].map(lambda x: json.dumps(x, sort_keys=True, default=str))
                persisted_events.to_parquet(staging / "events.parquet", index=False)
                pd.DataFrame(coverage).to_parquet(staging / "coverage.parquet", index=False)
                _write_json(staging / "request.json", request.model_dump(mode="json"))
                _write_json(staging / "raw_manifest.json", raw)
                _write_json(staging / "source_manifest.json", source_manifest)
                _write_json(staging / "exclusions.json", exclusions)
                _write_json(staging / "confirmation_keys.json", confirmation_keys)
                _write_json(staging / "formula-report.json", formula_report)
                from .research_targets import apply_target_contract, target_contract
                target_eligibility = {}
                target_kinds = {"win": "win_probability", "top3": "placing_top_k", "position": "ranking_strength",
                                "speed": "adjusted_finish_time_or_speed", "finish_time": "adjusted_finish_time_or_speed"}
                for name in request.target_contracts:
                    try:
                        labeled = apply_target_contract(features, target_contract(target_kinds[name]))
                        eligible_races = sorted(labeled.race_id.unique().tolist())
                        target_eligibility[name] = {"eligible_races": len(eligible_races), "eligible_rows": len(labeled),
                            "excluded_race_ids": sorted(set(features.race_id) - set(eligible_races)),
                            "contract": target_kinds[name], "finish_time_basis": "distance_metres / physical_speed_mps" if name == "finish_time" else None}
                    except ValueError as exc:
                        target_eligibility[name] = {"eligible_races": 0, "eligible_rows": 0, "error": str(exc), "contract": target_kinds[name]}
                _write_json(staging / "target-eligibility.json", target_eligibility)
                _write_json(staging / "protocol.json", {"protocol_id": request.protocol_id, "spec": spec.model_dump(mode="json"), "folds": [fold.__dict__ for fold in folds]})
                event_audit = {"observations": len(events), "tiers": events["availability_tier"].value_counts().to_dict(), "families": events["family"].value_counts().to_dict(),
                    "identity_unresolved": int(events["horse_id"].isna().sum()), "usable_runner_event_views": int(sum(event_features[f"{family}_usable_observations"].sum() for family in FAMILY_WINDOWS)),
                    "unknown_counts_remain_missing": True, "date_only_eligibility": "next Hong Kong local day; strictly before race cutoff",
                    "raw_blob_verification": "delegated_to_hash_verified_official_snapshot; raw blobs not replayed by this build", "coverage_records": len(coverage)}
                report = {"valid": True, "source_rows": len(source), "source_races": int(source["race_id"].nunique()), "rows": len(clean), "races": int(clean["race_id"].nunique()),
                    "excluded_races": len(exclusions), "excluded_rows": sum(e["rows"] for e in exclusions), "unaccounted_source_keys": 0,
                    "distributions_before": _distribution(source), "distributions_after": _distribution(clean), "confirmation_rows_quarantined": len(confirmation_keys),
                    "confirmation_labels_published": False, "fold_count": len(folds), "build_seconds": time.monotonic() - started,
                    "numeric_nonfinite_replacements": nonfinite}
                _write_json(staging / "validation.json", report)
                _write_json(staging / "temporal_audit.json", event_audit)
                manifest = {"dataset_id": dataset_id, "status": "verified", "request_id": request_id, "identity": identity,
                    "parent_dataset_id": request.parent_dataset_id, "source_snapshot": str(snapshot), "raw_corpus_manifest_id": request.raw_corpus_manifest_id,
                    "rows": len(features), "races": int(features["race_id"].nunique()), "date_min": str(clean["date"].min()), "date_max": str(clean["date"].max()),
                    "event_policy_id": request.event_policy_id, "availability_tier": request.event_policy_id,
                    "comparison_contract_id": "comparison-" + _digest(comparison), "comparison_contract": comparison,
                    "eligible_numeric_predictors": sorted(catalog), "eligible_predictors": sorted(catalog),
                    "feature_catalog": catalog, "predictor_catalog": catalog, "predictor_catalog_id": "catalog-" + _digest(catalog),
                    "eligible_categorical_predictors": categorical,
                    "availability_policy": request.event_policy_id, "evaluation_population_id": request.evaluation_population_id,
                    "features_path": str(destination / "features.parquet"),
                    "protocol_path": str(destination / "protocol.json"), "target_only_columns": source_manifest.get("target_only_columns", []),
                    "market_only_columns": source_manifest.get("market_only_columns", []), "ordered_row_keys_sha256": _digest(_keys(features)),
                    "ordered_row_hashes_sha256": hashlib.sha256(pd.util.hash_pandas_object(features, index=False).to_numpy().tobytes()).hexdigest(),
                    "event_audit": event_audit, "validation": report, "promotion": "candidate_only; explicit operator policy required",
                    "target_eligibility": target_eligibility,
                    "unsupported_requirements": ["raw blobs are not locally replayed by this registry", "source rating/profile publication is unverified; dependent predictors excluded", "condition residuals require parent train-only fitted transforms", "explicit delayed race-result publication excludes that race from development history; delayed-label asof rebuild not yet supported", "dead-heat races excluded by the current single-winner training contract; original tied outcomes preserved upstream"],
                    "files": {path.name: file_sha256(path) for path in staging.iterdir() if path.is_file()}}
                _write_json(staging / "manifest.json", manifest)
                (staging / "manifest.sha256").write_text(file_sha256(staging / "manifest.json") + "\n")
                # A complete manifest and every payload become visible together.
                os.rename(staging, destination)
                manifest = self.verify(dataset_id)
                self._transition(state, "verified", dataset_id=dataset_id, artifact_path=str(destination))
                return manifest
            except Exception as exc:
                self._transition(state, "rejected", error=f"{type(exc).__name__}: {exc}")
                raise
