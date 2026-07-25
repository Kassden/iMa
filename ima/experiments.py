from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .data import RaceSplits, chronological_race_split, validate_runner_dataset
from .feature_sets import BASELINE_SCHEMA, FeatureSchema
from .modeling import (
    MarketBlend, RaceProbabilityModel, TemperatureCalibrator, disagreement_report,
    evaluate_probabilities, incremental_pseudo_r2,
)
from .pipeline_transparency import pipeline_manifest
from .pools import SUPPORTED_POOLS, fit_order_exponents


@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    kind: str
    parameters: dict


def default_experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec("logit-c005-balanced", "logit", {"C": 0.05, "class_weight": "balanced"}),
        ExperimentSpec("logit-c010-balanced", "logit", {"C": 0.10, "class_weight": "balanced"}),
        ExperimentSpec("logit-c050-balanced", "logit", {"C": 0.50, "class_weight": "balanced"}),
        ExperimentSpec("logit-c100-balanced", "logit", {"C": 1.00, "class_weight": "balanced"}),
        ExperimentSpec("logit-c200-balanced", "logit", {"C": 2.00, "class_weight": "balanced"}),
        ExperimentSpec("logit-c050-unweighted", "logit", {"C": 0.50, "class_weight": None}),
        ExperimentSpec("boost-lr003-leaf15", "boosted", {"learning_rate": 0.03, "max_leaf_nodes": 15}),
        ExperimentSpec("boost-lr003-leaf31", "boosted", {"learning_rate": 0.03, "max_leaf_nodes": 31}),
        ExperimentSpec("boost-lr006-leaf15", "boosted", {"learning_rate": 0.06, "max_leaf_nodes": 15}),
        ExperimentSpec("boost-lr006-leaf31", "boosted", {"learning_rate": 0.06, "max_leaf_nodes": 31}),
        ExperimentSpec("boost-lr006-leaf63", "boosted", {"learning_rate": 0.06, "max_leaf_nodes": 63}),
        ExperimentSpec("boost-lr010-leaf15", "boosted", {"learning_rate": 0.10, "max_leaf_nodes": 15}),
        ExperimentSpec("boost-lr010-leaf31", "boosted", {"learning_rate": 0.10, "max_leaf_nodes": 31}),
        ExperimentSpec("boost-lr010-leaf63", "boosted", {"learning_rate": 0.10, "max_leaf_nodes": 63}),
        ExperimentSpec(
            "boost-lr006-leaf31-l2zero", "boosted",
            {"learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 0.0},
        ),
        ExperimentSpec(
            "boost-lr006-leaf31-l2three", "boosted",
            {"learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 3.0},
        ),
    ]


def _run_one(
    spec: ExperimentSpec,
    splits: RaceSplits,
    models_dir: Path,
    feature_schema: FeatureSchema = BASELINE_SCHEMA,
) -> dict:
    started = time.perf_counter()
    model = RaceProbabilityModel(
        kind=spec.kind, parameters=spec.parameters, feature_schema=feature_schema,
    ).fit(splits.train)
    validation_raw = model.predict_proba(splits.validation)
    calibrator = TemperatureCalibrator.fit(validation_raw, splits.validation)
    validation = calibrator.transform(validation_raw, splits.validation["race_id"])
    order_frame = splits.validation.assign(model_probability=validation)
    exponents = fit_order_exponents(order_frame, "model_probability")
    blend = MarketBlend.fit(
        validation,
        splits.validation["market_probability"].to_numpy(),
        splits.validation,
    )
    fundamental = calibrator.transform(model.predict_proba(splits.test), splits.test["race_id"])
    blended = blend.transform(
        fundamental,
        splits.test["market_probability"].to_numpy(),
        splits.test["race_id"],
    )
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "blend": blend,
        "order_exponents": exponents,
        "experiment": asdict(spec),
        "feature_schema": feature_schema.name,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, models_dir / f"{spec.run_id}.joblib")
    return {
        "run_id": spec.run_id,
        "kind": spec.kind,
        "parameters": spec.parameters,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "temperature": calibrator.temperature,
        "fundamental_weight": blend.fundamental_weight,
        "market_weight": blend.market_weight,
        "second_place_exponent": exponents.second,
        "third_place_exponent": exponents.third,
        "validation": evaluate_probabilities(validation, splits.validation),
        "test_fundamental": evaluate_probabilities(fundamental, splits.test),
        "test_blended": evaluate_probabilities(blended, splits.test),
        "incremental_pseudo_r2": incremental_pseudo_r2(
            blended, splits.test["market_probability"].to_numpy(), splits.test
        ),
        "disagreement": disagreement_report(
            fundamental,
            splits.test["market_probability"].to_numpy(),
            blended,
            splits.test,
        ),
    }


def results_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for run in summary.get("run_history", summary["runs"]):
        row = {
            "run_key": run.get("run_key"),
            "execution_id": run.get("execution_id"),
            "run_id": run["run_id"],
            "kind": run["kind"],
            "parameters": json.dumps(run["parameters"], sort_keys=True),
            "duration_seconds": run["duration_seconds"],
            "temperature": run["temperature"],
            "fundamental_weight": run["fundamental_weight"],
            "market_weight": run["market_weight"],
            "incremental_pseudo_r2": run["incremental_pseudo_r2"],
            "second_place_exponent": run["second_place_exponent"],
            "third_place_exponent": run["third_place_exponent"],
        }
        for scope in ("validation", "test_fundamental", "test_blended"):
            for metric, value in run[scope].items():
                row[f"{scope}_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def merge_run_history(*collections: list[dict], default_execution_id: str | None = None) -> list[dict]:
    """Merge persisted experiment executions without collapsing repeated run IDs."""
    merged: dict[str, dict] = {}
    for collection in collections:
        for source in collection:
            run = dict(source)
            execution_id = run.get("execution_id") or default_execution_id or "legacy-import"
            run["execution_id"] = execution_id
            run_key = run.get("run_key") or (
                f"{execution_id}|{run.get('feature_schema', 'baseline-v1')}|{run['run_id']}"
            )
            run["run_key"] = run_key
            merged[run_key] = run
    return sorted(
        merged.values(),
        key=lambda run: (run.get("execution_id", ""), run.get("sequence", 0), run["run_id"]),
    )


def _save_accuracy_graph(frame: pd.DataFrame, market: dict, output: Path) -> None:
    plot = frame.sort_values("test_fundamental_top_pick_win_rate")
    height = max(6, len(plot) * 0.42)
    fig, axis = plt.subplots(figsize=(12, height))
    colors = plot["kind"].map({"logit": "#18794e", "boosted": "#b54708"})
    axis.barh(plot["run_id"], plot["test_fundamental_top_pick_win_rate"] * 100, color=colors)
    axis.axvline(market["top_pick_win_rate"] * 100, color="#1d4ed8", linewidth=2, label="Market")
    axis.set_xlabel("Top-1 winner accuracy (%)")
    axis.set_title("1997-2025 training: held-out winner accuracy by run")
    axis.grid(axis="x", color="#d9dde3", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_probability_graph(frame: pd.DataFrame, market: dict, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))
    for kind, color, marker in (("logit", "#18794e", "o"), ("boosted", "#b54708", "s")):
        group = frame[frame["kind"].eq(kind)]
        axis.scatter(
            group["test_fundamental_race_log_loss"],
            group["test_fundamental_top_pick_win_rate"] * 100,
            color=color, marker=marker, s=70, label=kind.title(), alpha=0.9,
        )
    axis.scatter(
        [market["race_log_loss"]], [market["top_pick_win_rate"] * 100],
        color="#1d4ed8", marker="*", s=220, label="Market", zorder=5,
    )
    axis.set_xlabel("Race log loss (lower is better)")
    axis.set_ylabel("Top-1 winner accuracy (%)")
    axis.set_title("Accuracy versus probability quality")
    axis.grid(color="#d9dde3", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_dashboard(summary: dict, template_path: Path, output_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    payload = json.dumps(summary, separators=(",", ":")).replace("</", "<\\/")
    output_path.write_text(
        template.replace("__EXPERIMENT_DATA__", payload),
        encoding="utf-8",
    )


def run_experiments(
    frame: pd.DataFrame,
    output_dir: Path,
    template_path: Path,
    specs: list[ExperimentSpec] | None = None,
    feature_schema: FeatureSchema = BASELINE_SCHEMA,
) -> dict:
    validate_runner_dataset(frame)
    splits = chronological_race_split(frame)
    specs = specs or default_experiment_specs()
    market = evaluate_probabilities(splits.test["market_probability"].to_numpy(), splits.test)
    execution_id = datetime.now(timezone.utc).isoformat()
    runs = []
    for sequence, spec in enumerate(specs):
        run = _run_one(spec, splits, output_dir / "models", feature_schema)
        run.update({
            "execution_id": execution_id,
            "run_key": f"{execution_id}|{feature_schema.name}|{spec.run_id}",
            "sequence": sequence,
            "feature_schema": feature_schema.name,
            "feature_count": len(feature_schema.features),
            "variables": list(feature_schema.features),
        })
        runs.append(run)
    runs.sort(key=lambda run: run["test_fundamental"]["race_log_loss"])
    summary = {
        "created_at": execution_id,
        "dataset": {
            "runners": len(frame),
            "races": int(frame["race_id"].nunique()),
            "date_min": frame["date"].min().date().isoformat(),
            "date_max": frame["date"].max().date().isoformat(),
            "train_races": int(splits.train["race_id"].nunique()),
            "validation_races": int(splits.validation["race_id"].nunique()),
            "test_races": int(splits.test["race_id"].nunique()),
            "test_date_min": splits.test["date"].min().date().isoformat(),
            "test_date_max": splits.test["date"].max().date().isoformat(),
        },
        "market_test": market,
        "prediction_sources": {
            "fundamental": "Calibrated horse and race feature model without current odds.",
            "market": "Probability implied by the available WIN market odds.",
            "combined": "Fitted multiplicative blend of calibrated fundamental and market probabilities.",
            "historical_market_limit": (
                "Historical experiments use final WIN odds; live decisions must use odds observed "
                "before the wager timestamp."
            ),
        },
        "supported_pools": list(SUPPORTED_POOLS),
        "pipeline_manifest": pipeline_manifest(),
        "runs": runs,
        "run_history": runs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame_results = results_frame(summary)
    frame_results.to_csv(output_dir / "results.csv", index=False)
    _save_accuracy_graph(frame_results, market, output_dir / "top_pick_accuracy.png")
    _save_probability_graph(frame_results, market, output_dir / "accuracy_vs_logloss.png")
    render_dashboard(summary, template_path, output_dir / "dashboard.html")
    return summary
