"""Loadable research model packages with lineage manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .research_specs import PipelineRecipe, is_fundamental_first_portfolio


def speed_to_finish_seconds(distance_m, speed_mps):
    """Physical ratio readout, not E[finish time] for a speed distribution."""
    distance, speed = np.asarray(distance_m, float), np.asarray(speed_mps, float)
    if distance.shape != speed.shape or distance.ndim != 1:
        raise ValueError("Distance and speed must be aligned one-dimensional arrays")
    valid = np.isfinite(distance) & (distance > 0) & np.isfinite(speed) & (speed > 0)
    seconds = np.full(speed.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(distance, speed, out=seconds, where=valid)
    seconds[~np.isfinite(seconds) | (seconds <= 0)] = np.nan
    return seconds


@dataclass
class FeatureReplayContext:
    """Self-contained past-only feature inputs, independent of mutable trial caches."""

    definitions: tuple[dict, ...] = ()
    input_metadata: dict = field(default_factory=dict)
    required_extra_features: tuple[str, ...] = ()
    discovery_spec: Any = None
    discovery_features: tuple[str, ...] = ()
    history: pd.DataFrame | None = None
    training_cutoff: str | None = None
    dependency_versions: dict = field(default_factory=dict)
    implementation_hashes: dict = field(default_factory=dict)
    dataset_manifest_hash: str | None = None

    def contract(self):
        return {"schema_version": 1, "definitions": self.definitions,
            "input_metadata": self.input_metadata, "required_extra_features": self.required_extra_features,
            "discovery_spec": self.discovery_spec.model_dump(mode="json") if self.discovery_spec else None,
            "discovery_features": self.discovery_features, "history_rows": 0 if self.history is None else len(self.history),
            "history_hash": _frame_hash(self.history) if self.history is not None else None,
            "training_cutoff": self.training_cutoff, "dependency_versions": self.dependency_versions,
            "implementation_hashes": self.implementation_hashes,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "query_history_policy": "Query observations may affect only later queries after next-day availability; current outcomes never enter current predictors"}

    def transform(self, frame):
        from importlib.metadata import version
        for name, expected in self.dependency_versions.items():
            if version(name) != expected:
                raise ValueError(f"Feature replay dependency changed: {name}")
        for name, expected in self.implementation_hashes.items():
            path = Path(__file__).parent/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Feature replay implementation changed: {name}")
        output = frame.copy()
        if self.training_cutoff:
            dates = pd.to_datetime(output["date"], errors="raise")
            if dates.isna().any() or (dates <= pd.Timestamp(self.training_cutoff)).any():
                raise ValueError("V6 package queries must follow the frozen training cutoff")
        missing = set(self.required_extra_features)-set(output)
        if missing:
            raise ValueError(f"Inference missing declared dataset predictors: {sorted(missing)}")
        for name, metadata in self.input_metadata.items():
            available = metadata.get("available_at_column")
            if available and name in output:
                if available not in output:
                    raise ValueError(f"Inference missing predictor availability: {available}")
                observed = pd.to_datetime(output[available], utc=True, errors="coerce")
                valid = observed.notna() & observed.lt(pd.to_datetime(output.date, utc=True))
                output.loc[~valid,name] = np.nan
        synthesis_metadata = dict(self.input_metadata)
        if self.definitions:
            from .feature_definitions import evaluate_formulas
            # Derived columns are regenerated from bindings, never trusted as caller predictions.
            from .feature_definitions import FeatureDefinition
            derived = {d["name"] for d in self.definitions}
            derived.update(FeatureDefinition.model_validate(d).feature_id for d in self.definitions)
            output = output.drop(columns=list(derived & set(output)))
            output, _, _ = evaluate_formulas(output, self.definitions, self.input_metadata,
                                             cutoff=pd.to_datetime(output.date, utc=True))
            for definition in self.definitions:
                parsed = FeatureDefinition.model_validate(definition)
                metadata = {"unit": parsed.unit, "dtype": parsed.dtype, "temporal_scope": "pre_race"}
                synthesis_metadata[parsed.name] = metadata
                synthesis_metadata[parsed.feature_id] = metadata
        if self.discovery_spec and self.discovery_features:
            from .feature_program import synthesize
            if self.history is None:
                raise ValueError("Discovery inference requires its packaged historical observations")
            required = {"date", "race_id", "horse_no", "horse_id", "distance"}
            required.update(f"{entity}_id" for entity in self.discovery_spec.entities)
            if "carried_weight" in self.discovery_spec.measurements or self.discovery_spec.domain_history:
                required.add("actual_weight")
            if "beaten_lengths" in self.discovery_spec.measurements:
                required.add("lengths_raw")
            if set(self.discovery_spec.measurements)-{"speed_mps", "beaten_lengths", "carried_weight"}:
                required.update(set(self.discovery_spec.measurements)-{"speed_mps", "beaten_lengths", "carried_weight"})
            missing = required-set(output)
            if missing:
                raise ValueError(f"Discovery inference missing source bindings: {sorted(missing)}")
            output["date"] = pd.to_datetime(output.date)
            # Pre-race queries may be untimed; their labels are not fabricated.
            if "finish_seconds" not in output:
                from .data import _finish_time_seconds
                output["finish_seconds"] = _finish_time_seconds(output.finish_time) if "finish_time" in output else np.nan
            history = self.history.copy()
            if "observed_at" in history and "observed_at" not in output:
                output["observed_at"] = output.date.dt.normalize()+pd.Timedelta(days=1)
            history.attrs = {}
            output.attrs = {}
            joined = pd.concat([history, output], ignore_index=True)
            if joined.duplicated(["race_id", "horse_no"]).any():
                raise ValueError("Discovery query overlaps packaged past runner keys")
            matrix, _ = synthesize(joined, self.discovery_spec,
                cutoff_positions=np.arange(len(history), len(joined)), input_metadata=synthesis_metadata)
            absent = set(self.discovery_features)-set(matrix)
            if absent:
                raise ValueError(f"Packaged discovery definitions cannot be regenerated: {sorted(absent)}")
            output = output.drop(columns=list(set(matrix.columns) & set(output))).reset_index(drop=True)
            output = pd.concat([output, matrix[list(self.discovery_features)].reset_index(drop=True)], axis=1)
        return output


def _frame_hash(frame):
    identity = hashlib.sha256(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    identity.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return identity.hexdigest()


@dataclass(frozen=True)
class ResearchPackageManifest:
    package_id: str
    recipe_hash: str
    protocol_id: str
    code_revision: str
    model_kind: str
    target_kind: str
    prediction_method: str
    portfolio_version: str | None = None
    created_by: str = "ima-research-v2"


@dataclass
class ResearchModelPackage:
    model: Any
    recipe: PipelineRecipe
    protocol_id: str
    code_revision: str
    portfolio_version: str | None = None
    feature_context: FeatureReplayContext | None = None

    def manifest(self) -> ResearchPackageManifest:
        recipe_hash = self.recipe.recipe_hash()
        identity = {
                "recipe_hash": recipe_hash,
                "protocol_id": self.protocol_id,
                "code_revision": self.code_revision,
                "model_kind": self.recipe.model.kind,
        }
        if self.portfolio_version is not None:
            identity["portfolio_version"] = self.portfolio_version
        context = getattr(self, "feature_context", None)
        if context is not None:
            identity["feature_replay_contract"] = context.contract()
        package_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return ResearchPackageManifest(
            package_id=package_id,
            recipe_hash=recipe_hash,
            protocol_id=self.protocol_id,
            code_revision=self.code_revision,
            model_kind=self.recipe.model.kind,
            target_kind=self.recipe.target.kind,
            prediction_method=(
                "predict_fundamental_proba"
                if self.recipe.target.kind == "win_probability"
                and is_fundamental_first_portfolio(self.portfolio_version)
                else "predict_proba" if self.recipe.target.kind == "win_probability" else "predict"
            ),
            portfolio_version=self.portfolio_version,
            created_by="ima-research-v6" if self.recipe.schema_version == 3 else "ima-research-v3" if self.portfolio_version else "ima-research-v2",
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind != "win_probability":
            raise TypeError("predict_proba is only valid for win_probability packages")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Packaged model does not expose predict_proba")
        predict = (
            self.model.predict_fundamental_proba
            if is_fundamental_first_portfolio(self.portfolio_version)
            else self.model.predict_proba
        )
        probabilities = np.asarray(predict(self._feature_frame(frame)), dtype=float)
        _validate_complete_races(frame, probabilities)
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind == "win_probability":
            return self.predict_proba(frame)
        if not hasattr(self.model, "predict"):
            raise TypeError("Packaged model does not expose predict")
        predictions = np.asarray(self.model.predict(self._feature_frame(frame)), dtype=float)
        if len(frame) != len(predictions) or not np.isfinite(predictions).all():
            raise ValueError("Packaged predictions must be finite and match input rows")
        return predictions

    def predict_fundamental_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind != "win_probability":
            raise TypeError("Fundamental probabilities require a win_probability package")
        if not hasattr(self.model, "predict_fundamental_proba"):
            raise TypeError("Packaged model does not expose fundamental probabilities")
        probabilities = np.asarray(
            self.model.predict_fundamental_proba(self._feature_frame(frame)), dtype=float
        )
        _validate_complete_races(frame, probabilities)
        return probabilities

    def predict_selected_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Inspect the selected graph/recipe output without changing portfolio defaults."""
        if self.recipe.target.kind != "win_probability":
            raise TypeError("Selected probabilities require a win_probability package")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Packaged model does not expose selected probabilities")
        probabilities = np.asarray(self.model.predict_proba(self._feature_frame(frame)), dtype=float)
        _validate_complete_races(frame, probabilities)
        return probabilities

    def predict_finish_seconds(self, frame: pd.DataFrame) -> np.ndarray:
        """Return distance / predicted physical speed, with NaN for invalid ratios.

        For distribution packages this is distance / E[speed], not E[finish time].
        Raw distance is in metres; learned feature transforms must not change it.
        """
        if self.recipe.target.kind != "adjusted_finish_time_or_speed":
            raise TypeError("Finish-time readout requires a physical-speed package")
        if "distance" not in frame:
            raise ValueError("Finish-time readout requires raw distance in metres")
        distance = pd.to_numeric(frame["distance"], errors="coerce").to_numpy(float)
        return speed_to_finish_seconds(distance, self.predict(frame))

    def predict_auxiliary_win_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind not in {"ranking_strength", "adjusted_finish_time_or_speed"}:
            raise TypeError("Auxiliary win probabilities require ranking or performance packages")
        probabilities = np.asarray(self.model.predict_win_proba(self._feature_frame(frame)), dtype=float)
        _validate_complete_races(frame, probabilities)
        return probabilities

    def _feature_frame(self, frame):
        context = getattr(self, "feature_context", None)
        return context.transform(frame) if context is not None else frame

    def predict_distribution(self, frame):
        if not hasattr(self.model, "predict_distribution"):
            raise TypeError("Packaged model does not expose a performance distribution")
        return self.model.predict_distribution(self._feature_frame(frame))

    def predict_joint(self, frame):
        if not hasattr(self.model, "predict_joint"):
            raise TypeError("Packaged model does not expose joint finish orders")
        return self.model.predict_joint(self._feature_frame(frame))

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        joblib.dump(self, path / "model.joblib")
        (path / "manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (path / "recipe.json").write_text(
            json.dumps(self.recipe.canonical_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        context = getattr(self, "feature_context", None)
        if context is not None:
            (path / "feature-replay.json").write_text(json.dumps(context.contract(), indent=2, sort_keys=True), encoding="utf-8")
        return path


def load_research_package(path: Path) -> ResearchModelPackage:
    package = joblib.load(path / "model.joblib")
    if not isinstance(package, ResearchModelPackage):
        raise TypeError("Loaded artifact is not a ResearchModelPackage")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["package_id"] != package.manifest().package_id:
        raise ValueError("Package manifest does not match loaded model")
    if package.recipe.schema_version == 3 and getattr(package, "feature_context", None) is not None:
        stored = json.loads((path / "feature-replay.json").read_text())
        if stored != json.loads(json.dumps(package.feature_context.contract())):
            raise ValueError("Package feature replay contract does not match loaded state")
    return package


def prediction_readback(package_path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    package = load_research_package(package_path)
    probabilities = package.predict_proba(frame)
    totals = pd.Series(probabilities).groupby(frame["race_id"]).sum()
    return {
        "package_id": package.manifest().package_id,
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "probability_sum_min": float(totals.min()),
        "probability_sum_max": float(totals.max()),
    }


def _validate_complete_races(frame: pd.DataFrame, probabilities: np.ndarray) -> None:
    if len(frame) != len(probabilities):
        raise ValueError("Prediction length does not match input rows")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Packaged probabilities must be finite and in [0,1]")
    if "field_size" in frame.columns:
        observed = frame.groupby("race_id")["race_id"].transform("size")
        declared = pd.to_numeric(frame["field_size"], errors="coerce")
        if not observed.eq(declared).all():
            raise ValueError("Input contains incomplete races relative to declared field_size")
    totals = pd.Series(probabilities).groupby(frame["race_id"].to_numpy()).sum()
    if not np.allclose(totals.to_numpy(), 1.0):
        raise ValueError("Packaged probabilities must sum to one by race")
