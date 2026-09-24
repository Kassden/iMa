"""MLflow tracking helpers for iMa experiment runs.

The rest of the repo can import this module without requiring MLflow at import
time. A missing MLflow package is only an error when tracking/export is invoked.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_NAME = "ima-racing"
DEFAULT_REGISTERED_MODEL_NAME = "ima-racing-candidates"


class MLflowUnavailableError(RuntimeError):
    """Raised when MLflow-backed behavior is requested but MLflow is unavailable."""


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str | None = None
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    enabled: bool = False
    register_models: bool = True
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME

    @classmethod
    def from_values(
        cls,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        register_models: bool | None = None,
        registered_model_name: str | None = None,
    ) -> "MLflowConfig":
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        name = experiment_name or os.environ.get("IMA_MLFLOW_EXPERIMENT") or DEFAULT_EXPERIMENT_NAME
        register = register_models
        if register is None:
            register = os.environ.get("IMA_MLFLOW_REGISTER_MODELS", "1").lower() not in {
                "0", "false", "no", "off",
            }
        model_name = (
            registered_model_name
            or os.environ.get("IMA_MLFLOW_REGISTERED_MODEL")
            or DEFAULT_REGISTERED_MODEL_NAME
        )
        return cls(
            tracking_uri=uri,
            experiment_name=name,
            enabled=bool(uri),
            register_models=register,
            registered_model_name=model_name,
        )


def _mlflow():
    try:
        import mlflow  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when dependency missing.
        raise MLflowUnavailableError(
            "MLflow tracking was requested, but the mlflow package is not installed. "
            "Install project dependencies with python3.11 -m pip install -e ."
        ) from exc
    return mlflow


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_. /-]+", "_", value).strip(" ._/")[:240] or "value"


def flatten_numeric_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        name = _safe_key(f"{prefix}.{key}" if prefix else str(key))
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[name] = float(value)
        elif isinstance(value, dict):
            metrics.update(flatten_numeric_metrics(value, name))
    return metrics


def run_parameters(run: dict[str, Any]) -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {
        "run_id": run.get("run_id", ""),
        "kind": run.get("kind", ""),
        "feature_schema": run.get("feature_schema", "baseline-v1"),
        "feature_count": int(run.get("feature_count") or len(run.get("variables", [])) or 0),
    }
    for key, value in (run.get("parameters") or {}).items():
        params[f"param.{_safe_key(str(key))}"] = value if isinstance(value, int | float | bool) else str(value)
    return params


def _model_artifact_source(artifact_uri: str, model_path: Path) -> str:
    return f"{artifact_uri.rstrip('/')}/models/{model_path.name}"


def _register_model_version(
    mlflow,
    run: dict[str, Any],
    model_path: Path,
    config: MLflowConfig,
    run_id: str,
    artifact_uri: str,
) -> None:
    if not config.register_models or not model_path.is_file():
        return
    client = mlflow.tracking.MlflowClient()
    name = config.registered_model_name
    try:
        client.get_registered_model(name)
    except Exception:
        try:
            client.create_registered_model(name)
        except Exception:
            client.get_registered_model(name)
    version = client.create_model_version(
        name=name,
        source=_model_artifact_source(artifact_uri, model_path),
        run_id=run_id,
    )
    run["mlflow_registered_model_name"] = name
    run["mlflow_model_version"] = str(version.version)
    tags = {
        "ima.run_id": str(run.get("run_id", "")),
        "ima.run_key": str(run.get("run_key") or run.get("run_id", "")),
        "ima.kind": str(run.get("kind", "")),
        "ima.feature_schema": str(run.get("feature_schema", "baseline-v1")),
        "ima.race_log_loss": str(run.get("test_blended", {}).get("race_log_loss", "")),
        "ima.top_pick_win_rate": str(run.get("test_blended", {}).get("top_pick_win_rate", "")),
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)


def log_experiment_run(
    run: dict[str, Any],
    dataset: dict[str, Any],
    market_test: dict[str, Any],
    model_path: Path,
    config: MLflowConfig,
) -> str | None:
    if not config.enabled:
        return None
    mlflow = _mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    tags = {
        "ima.run_key": str(run.get("run_key") or run.get("run_id")),
        "ima.execution_id": str(run.get("execution_id", "")),
        "ima.feature_schema": str(run.get("feature_schema", "baseline-v1")),
        "ima.kind": str(run.get("kind", "")),
    }
    with mlflow.start_run(run_name=str(run.get("run_id", "ima-run"))) as active:
        mlflow.set_tags(tags)
        mlflow.log_params(run_parameters(run))
        metric_payload = {
            "duration_seconds": run.get("duration_seconds"),
            "temperature": run.get("temperature"),
            "fundamental_weight": run.get("fundamental_weight"),
            "market_weight": run.get("market_weight"),
            "place_market_weight": run.get("place_market_weight"),
            "incremental_pseudo_r2": run.get("incremental_pseudo_r2"),
            "second_place_exponent": run.get("second_place_exponent"),
            "third_place_exponent": run.get("third_place_exponent"),
            "validation": run.get("validation", {}),
            "test_fundamental": run.get("test_fundamental", {}),
            "test_blended": run.get("test_blended", {}),
            "pool_metrics": run.get("pool_metrics", {}),
        }
        mlflow.log_metrics(flatten_numeric_metrics(metric_payload))
        mlflow.log_dict({"dataset": dataset, "market_test": market_test}, "context.json")
        if model_path.is_file():
            mlflow.log_artifact(str(model_path), artifact_path="models")
            try:
                _register_model_version(
                    mlflow,
                    run,
                    model_path,
                    config,
                    active.info.run_id,
                    active.info.artifact_uri,
                )
            except Exception as exc:  # pragma: no cover - depends on MLflow server behavior.
                run["mlflow_model_registry_error"] = str(exc)
        run["mlflow_run_id"] = active.info.run_id
        run["mlflow_experiment_id"] = active.info.experiment_id
        mlflow.log_dict(run, "run.json")
        return active.info.run_id


def log_research_package(
    package_dir: Path,
    config: MLflowConfig,
    *,
    run_name: str = "research-package",
    tags: dict[str, str] | None = None,
) -> str | None:
    if not config.enabled:
        return None
    mlflow = _mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({
            "ima.package_id": str(manifest.get("package_id", "")),
            "ima.recipe_hash": str(manifest.get("recipe_hash", "")),
            "ima.protocol_id": str(manifest.get("protocol_id", "")),
            **(tags or {}),
        })
        mlflow.log_params({
            "package_id": str(manifest.get("package_id", "")),
            "recipe_hash": str(manifest.get("recipe_hash", "")),
            "protocol_id": str(manifest.get("protocol_id", "")),
            "model_kind": str(manifest.get("model_kind", "")),
        })
        mlflow.log_artifacts(str(package_dir), artifact_path="research_package")
        return active.info.run_id


def log_research_package_version(
    package_dir: Path,
    config: MLflowConfig,
    *,
    attempt_id: str,
    result: dict[str, Any],
) -> dict[str, str] | None:
    """Idempotently log and register a loadable target-aware research package."""
    if not config.enabled:
        return None
    mlflow = _mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.set_experiment(config.experiment_name)
    client = mlflow.tracking.MlflowClient()
    target_kind = str(result.get("target_kind", "win_probability"))
    model_name = f"{config.registered_model_name}-{target_kind.replace('_', '-')}"
    existing = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.`ima.attempt_id` = '{attempt_id}'",
        max_results=2,
    )
    if len(existing) > 1:
        raise RuntimeError(f"Duplicate MLflow runs for attempt {attempt_id}")
    if existing:
        return _research_model_linkage(client, model_name, existing[0].info.run_id)

    class ResearchPyFuncModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            from ima.research_model_package import load_research_package
            self.package = load_research_package(Path(context.artifacts["package"]))

        def predict(self, context, model_input, params=None):
            if self.package.recipe.target.kind == "win_probability":
                return self.package.predict_proba(model_input)
            return self.package.predict(model_input)

    tags = {
        "ima.attempt_id": attempt_id,
        "ima.recipe_hash": str(result.get("recipe_hash", "")),
        "ima.target_kind": target_kind,
        "ima.protocol_id": str(result.get("lineage", {}).get("protocol_id", "")),
        "ima.dataset_hash": str(result.get("lineage", {}).get("dataset_hash", "")),
        "ima.code_revision": str(result.get("lineage", {}).get("code_revision", "")),
        "ima.environment_hash": str(result.get("lineage", {}).get("environment_hash", "")),
    }
    with mlflow.start_run(run_name=attempt_id, tags=tags) as active:
        mlflow.log_params({
            "attempt_id": attempt_id,
            "recipe_hash": str(result.get("recipe_hash", "")),
            "target_kind": target_kind,
            "objective_name": str(result.get("objective_name", "")),
        })
        if result.get("objective_value") is not None:
            mlflow.log_metric("objective", float(result["objective_value"]))
        mlflow.log_dict(result, "result.json")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=ResearchPyFuncModel(),
            artifacts={"package": str(package_dir)},
            registered_model_name=model_name if config.register_models else None,
        )
        run_id = active.info.run_id
    linkage = _research_model_linkage(client, model_name, run_id)
    if config.register_models and "model_version" not in linkage:
        raise RuntimeError(f"MLflow did not register a model version for {attempt_id}")
    return linkage


def _research_model_linkage(client, model_name: str, run_id: str) -> dict[str, str]:
    versions = list(client.search_model_versions(f"run_id = '{run_id}'"))
    matching = [version for version in versions if version.name == model_name]
    linkage = {
        "run_id": run_id,
        "model_name": model_name,
        "model_uri": f"runs:/{run_id}/model",
    }
    if matching:
        linkage["model_version"] = str(matching[0].version)
        linkage["registered_model_uri"] = f"models:/{model_name}/{matching[0].version}"
    return linkage


def import_runs_from_results(
    results_path: Path,
    tracking_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    models_root: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    config = MLflowConfig(tracking_uri=tracking_uri, experiment_name=experiment_name, enabled=True)
    dataset = payload.get("dataset", {})
    market = payload.get("market_test", {})
    logged = []
    for run in payload.get("run_history", payload.get("runs", [])):
        run = dict(run)
        model_path = Path("")
        if models_root:
            candidate = models_root / f"{run['run_id']}.joblib"
            if candidate.exists():
                model_path = candidate
        logged.append(log_experiment_run(run, dataset, market, model_path, config))
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "runs_logged": len([item for item in logged if item]),
    }
