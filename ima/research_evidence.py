"""Campaign evidence summaries for agentic optimizer research."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PRIMARY_METRIC = "test_blended.race_log_loss"


@dataclass(frozen=True)
class JsonlReadResult:
    rows: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def metric_value(payload: dict[str, Any], dotted: str) -> float | None:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool):
        return None
    return float(current) if isinstance(current, int | float) else None


def read_jsonl(path: Path) -> JsonlReadResult:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return JsonlReadResult(rows, errors)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({
                "line": line_number,
                "error": str(exc),
                "prefix": line[:160],
            })
            continue
        if not isinstance(payload, dict):
            errors.append({
                "line": line_number,
                "error": "JSONL row is not an object",
                "prefix": line[:160],
            })
            continue
        rows.append(payload)
    return JsonlReadResult(rows, errors)


def _counter_dict(values: list[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def _trial_summary(row: dict[str, Any], primary_metric: str) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {
        "trial_id": row.get("trial_id"),
        "run_id": row.get("run_id"),
        "status": row.get("status"),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "output_dir": row.get("output_dir"),
        "primary_metric": metric_value(metrics, primary_metric),
        "kind": metrics.get("kind"),
        "feature_schema": metrics.get("feature_schema"),
        "fundamental_weight": metrics.get("fundamental_weight"),
        "market_weight": metrics.get("market_weight"),
        "mlflow_run_id": metrics.get("mlflow_run_id"),
        "mlflow_experiment_id": metrics.get("mlflow_experiment_id"),
        "mlflow_model_version": metrics.get("mlflow_model_version"),
        "test_blended": metrics.get("test_blended"),
        "test_fundamental": metrics.get("test_fundamental"),
        "validation": metrics.get("validation"),
    }


def _best_trial(rows: list[dict[str, Any]], primary_metric: str) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        value = metric_value(metrics, primary_metric)
        if value is not None:
            scored.append((value, row))
    if not scored:
        return None
    reverse = not primary_metric.endswith("loss") and not primary_metric.endswith("error")
    return sorted(scored, key=lambda item: item[0], reverse=reverse)[0][1]


def _completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "completed"]


def summarize_campaign(
    campaign_dir: Path,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> dict[str, Any]:
    trials_path = campaign_dir / "trials.jsonl"
    decisions_path = campaign_dir / "decisions.jsonl"
    trials = read_jsonl(trials_path)
    decisions = read_jsonl(decisions_path)
    completed = _completed_rows(trials.rows)
    best = _best_trial(trials.rows, primary_metric)
    first = completed[0] if completed else None
    latest = completed[-1] if completed else None
    metrics = [
        row.get("metrics") for row in completed
        if isinstance(row.get("metrics"), dict)
    ]
    fundamental_weights = [
        float(row["fundamental_weight"])
        for row in metrics
        if isinstance(row.get("fundamental_weight"), int | float)
    ]
    zero_fundamental_weight = sum(abs(value) < 1e-12 for value in fundamental_weights)
    schemas = _counter_dict([row.get("feature_schema") for row in metrics])
    kinds = _counter_dict([row.get("kind") for row in metrics])
    statuses = _counter_dict([row.get("status") for row in trials.rows])
    selected = {
        "first_completed": _trial_summary(first, primary_metric) if first else None,
        "best_primary": _trial_summary(best, primary_metric) if best else None,
        "latest_completed": _trial_summary(latest, primary_metric) if latest else None,
    }
    primary_values = [
        metric_value(row, primary_metric)
        for row in metrics
    ]
    primary_values = [value for value in primary_values if value is not None]
    return {
        "campaign_dir": str(campaign_dir),
        "files": {
            "trials": str(trials_path),
            "decisions": str(decisions_path),
            "trials_exists": trials_path.exists(),
            "decisions_exists": decisions_path.exists(),
        },
        "trial_rows": len(trials.rows),
        "decision_rows": len(decisions.rows),
        "corrupt_rows": {
            "trials": trials.errors,
            "decisions": decisions.errors,
        },
        "statuses": statuses,
        "completed_trials": len(completed),
        "families": kinds,
        "schemas": schemas,
        "fundamental_weight": {
            "observed": len(fundamental_weights),
            "zero_count": zero_fundamental_weight,
            "zero_fraction": (
                zero_fundamental_weight / len(fundamental_weights)
                if fundamental_weights else None
            ),
            "min": min(fundamental_weights) if fundamental_weights else None,
            "max": max(fundamental_weights) if fundamental_weights else None,
        },
        "primary_metric": {
            "path": primary_metric,
            "label": "legacy_dev" if primary_metric.startswith("test_") else "development",
            "count": len(primary_values),
            "min": min(primary_values) if primary_values else None,
            "max": max(primary_values) if primary_values else None,
            "best_is_lower": primary_metric.endswith("loss") or primary_metric.endswith("error"),
        },
        "selected_trials": selected,
        "latest_decision": decisions.rows[-1] if decisions.rows else None,
        "notes": [
            "Metrics named test_* are historical development metrics after optimizer reuse.",
            "This summary is read-only and does not verify model loadability.",
        ],
    }


def mlflow_sample_readback(
    selected_trials: dict[str, Any],
    tracking_uri: str,
    client: Any | None = None,
) -> dict[str, Any]:
    if client is None:
        try:
            import mlflow  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency.
            return {"enabled": False, "error": f"mlflow unavailable: {exc}"}
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

    samples: dict[str, Any] = {}
    for sample_name, trial in selected_trials.items():
        if not isinstance(trial, dict):
            samples[sample_name] = {"status": "missing_trial"}
            continue
        run_id = trial.get("mlflow_run_id")
        if not run_id:
            samples[sample_name] = {"status": "missing_mlflow_run_id"}
            continue
        try:
            run = client.get_run(str(run_id))
            artifacts = client.list_artifacts(str(run_id))
        except Exception as exc:
            samples[sample_name] = {
                "status": "readback_failed",
                "run_id": run_id,
                "error": str(exc),
            }
            continue
        samples[sample_name] = {
            "status": getattr(run.info, "status", None),
            "run_id": run_id,
            "experiment_id": getattr(run.info, "experiment_id", None),
            "artifact_paths": sorted(getattr(item, "path", "") for item in artifacts),
            "metric_count": len(getattr(run.data, "metrics", {}) or {}),
            "param_count": len(getattr(run.data, "params", {}) or {}),
        }
    return {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "samples": samples,
    }
