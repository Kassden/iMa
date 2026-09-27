"""Loadable research model packages with lineage manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .research_specs import FUNDAMENTAL_FIRST_PORTFOLIO_VERSION, PipelineRecipe


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
                and self.portfolio_version == FUNDAMENTAL_FIRST_PORTFOLIO_VERSION
                else "predict_proba" if self.recipe.target.kind == "win_probability" else "predict"
            ),
            portfolio_version=self.portfolio_version,
            created_by="ima-research-v3" if self.portfolio_version else "ima-research-v2",
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind != "win_probability":
            raise TypeError("predict_proba is only valid for win_probability packages")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Packaged model does not expose predict_proba")
        predict = (
            self.model.predict_fundamental_proba
            if self.portfolio_version == FUNDAMENTAL_FIRST_PORTFOLIO_VERSION
            else self.model.predict_proba
        )
        probabilities = np.asarray(predict(frame), dtype=float)
        _validate_complete_races(frame, probabilities)
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind == "win_probability":
            return self.predict_proba(frame)
        if not hasattr(self.model, "predict"):
            raise TypeError("Packaged model does not expose predict")
        predictions = np.asarray(self.model.predict(frame), dtype=float)
        if len(frame) != len(predictions) or not np.isfinite(predictions).all():
            raise ValueError("Packaged predictions must be finite and match input rows")
        return predictions

    def predict_auxiliary_win_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe.target.kind != "ranking_strength":
            raise TypeError("Auxiliary win probabilities require a ranking package")
        probabilities = np.asarray(self.model.predict_win_proba(frame), dtype=float)
        _validate_complete_races(frame, probabilities)
        return probabilities

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
        return path


def load_research_package(path: Path) -> ResearchModelPackage:
    package = joblib.load(path / "model.joblib")
    if not isinstance(package, ResearchModelPackage):
        raise TypeError("Loaded artifact is not a ResearchModelPackage")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["package_id"] != package.manifest().package_id:
        raise ValueError("Package manifest does not match loaded model")
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
    if "field_size" in frame.columns:
        observed = frame.groupby("race_id")["race_id"].transform("size")
        declared = pd.to_numeric(frame["field_size"], errors="coerce")
        if not observed.eq(declared).all():
            raise ValueError("Input contains incomplete races relative to declared field_size")
    totals = pd.Series(probabilities).groupby(frame["race_id"]).sum()
    if not np.allclose(totals.to_numpy(), 1.0):
        raise ValueError("Packaged probabilities must sum to one by race")
