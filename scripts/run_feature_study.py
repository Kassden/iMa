from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ima.data import build_full_history_dataset, chronological_race_split
from ima.experiments import render_dashboard
from ima.feature_analysis import association_report, correlation_report, permutation_importance_report
from ima.feature_sets import (
    BASELINE_SCHEMA, BENTER_COVERAGE, FEATURE_FAMILIES, FEATURE_ORIGINS, RICH_SCHEMA,
)
from ima.modeling import (
    MarketBlend, RaceProbabilityModel, TemperatureCalibrator, evaluate_probabilities,
    incremental_pseudo_r2,
)
from ima.rich_features import load_full_rich_history


def data_provenance(report: dict, canonical_path: Path) -> dict:
    canonical_sources = pd.read_csv(canonical_path, usecols=["source"])["source"].value_counts()
    study_rows = int(report["dataset"]["runners"])
    canonical_rows = int(canonical_sources.sum())
    legacy_rows = max(0, study_rows - canonical_rows)
    roles = {
        "official:hkjc-results": "Authoritative official result pages",
        "kaggle:mexwell-hkjc": "Gap fill where official archive pages are absent",
        "kaggle:jeffreymuller-2013-2020": "Gap fill where higher-priority sources are absent",
        "third-party:swords-2008-2009": "Small residual gap fill",
    }
    rows = []
    for source, count in canonical_sources.items():
        rows.append({
            "source": source,
            "role": roles.get(source, "Documented canonical gap fill"),
            "runner_rows": int(count),
            "share": float(count / study_rows),
        })
    if legacy_rows:
        rows.append({
            "source": "kaggle:gdaley-hkracing",
            "role": "Pre-2005 extension used only before official canonical coverage",
            "runner_rows": legacy_rows,
            "share": float(legacy_rows / study_rows),
        })
    return {
        "policy": (
            "HKJC is authoritative wherever an archived official result page exists. "
            "Lower-priority archives fill missing races only, and source is retained per runner."
        ),
        "official_canonical_share": float(
            canonical_sources.get("official:hkjc-results", 0) / canonical_rows
        ),
        "sources": rows,
    }


def publish_dashboard(report: dict, results_path: Path, template_path: Path, output_path: Path) -> None:
    summary = json.loads(results_path.read_text(encoding="utf-8"))
    summary["feature_study"] = report
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_dashboard(summary, template_path, output_path)


def evaluate_candidate(kind: str, schema, splits) -> tuple[dict, dict]:
    model = RaceProbabilityModel(kind=kind, feature_schema=schema).fit(splits.train)
    validation_raw = model.predict_proba(splits.validation)
    calibrator = TemperatureCalibrator.fit(validation_raw, splits.validation)
    validation = calibrator.transform(validation_raw, splits.validation["race_id"])
    blend = MarketBlend.fit(
        validation, splits.validation["market_probability"].to_numpy(), splits.validation,
    )
    fundamental = calibrator.transform(model.predict_proba(splits.test), splits.test["race_id"])
    combined = blend.transform(
        fundamental, splits.test["market_probability"].to_numpy(), splits.test["race_id"],
    )
    report = {
        "name": f"{schema.name}-{kind}", "schema": schema.name, "kind": kind,
        "feature_count": len(schema.features), "temperature": calibrator.temperature,
        "fundamental_weight": blend.fundamental_weight, "market_weight": blend.market_weight,
        "validation": evaluate_probabilities(validation, splits.validation),
        "test_fundamental": evaluate_probabilities(fundamental, splits.test),
        "test_combined": evaluate_probabilities(combined, splits.test),
        "incremental_pseudo_r2": incremental_pseudo_r2(
            combined, splits.test["market_probability"].to_numpy(), splits.test,
        ),
    }
    return report, {"model": model, "calibrator": calibrator, "blend": blend}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Benter-inspired rich feature study")
    parser.add_argument("--legacy-runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--legacy-races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/processed/historical/runners.csv.gz"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/historical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature-study"))
    parser.add_argument("--permutation-repeats", type=int, default=2)
    parser.add_argument("--dashboard-results", type=Path, default=Path("public/results.json"))
    parser.add_argument(
        "--dashboard-template", type=Path,
        default=Path("docs/model-results/dashboard-template.html"),
    )
    parser.add_argument("--dashboard-output", type=Path, default=Path("public/index.html"))
    parser.add_argument(
        "--publish-only", action="store_true",
        help="Publish an existing feature-study report without retraining models.",
    )
    args = parser.parse_args()

    if args.publish_only:
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        report["data_provenance"] = data_provenance(report, args.canonical)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        publish_dashboard(report, args.dashboard_results, args.dashboard_template, args.dashboard_output)
        return 0

    rich = load_full_rich_history(
        args.legacy_runs, args.legacy_races, args.canonical,
        args.processed / "trackwork.csv.gz", args.processed / "barrier-trials.csv.gz",
        args.processed / "sectionals.csv.gz",
    )
    baseline = build_full_history_dataset(args.legacy_runs, args.legacy_races, args.canonical)
    shared_races = set(rich["race_id"].unique())
    baseline = baseline[baseline["race_id"].isin(shared_races)].copy()
    rich = rich[rich["race_id"].isin(set(baseline["race_id"].unique()))].copy()
    baseline_splits = chronological_race_split(baseline)
    rich_splits = chronological_race_split(rich)
    split_ids = lambda split: tuple(split["race_id"].drop_duplicates())
    for baseline_split, rich_split in zip(
        (baseline_splits.train, baseline_splits.validation, baseline_splits.test),
        (rich_splits.train, rich_splits.validation, rich_splits.test),
    ):
        if split_ids(baseline_split) != split_ids(rich_split):
            raise ValueError("Baseline and rich study race splits differ")

    candidates = []
    artifacts = {}
    for schema, splits in ((BASELINE_SCHEMA, baseline_splits), (RICH_SCHEMA, rich_splits)):
        for kind in ("logit", "boosted"):
            report, artifact = evaluate_candidate(kind, schema, splits)
            candidates.append(report)
            artifacts[report["name"]] = artifact
    rich_candidates = [row for row in candidates if row["schema"] == RICH_SCHEMA.name]
    selected = min(rich_candidates, key=lambda row: row["validation"]["race_log_loss"])
    selected_model = artifacts[selected["name"]]["model"]

    associations = association_report(rich_splits.train, RICH_SCHEMA)
    matrix, redundant = correlation_report(rich_splits.train, RICH_SCHEMA.numeric)
    feature_groups = {feature: (feature,) for feature in RICH_SCHEMA.features}
    feature_importance = permutation_importance_report(
        selected_model, rich_splits.test, feature_groups, args.permutation_repeats,
    ).rename(columns={"name": "feature"})
    family_importance = permutation_importance_report(
        selected_model, rich_splits.test, FEATURE_FAMILIES, args.permutation_repeats,
    ).rename(columns={"name": "family"})
    ranking = feature_importance.merge(associations, on="feature", how="left")
    feature_family = {
        feature: family for family, features in FEATURE_FAMILIES.items() for feature in features
    }
    ranking["family"] = ranking["feature"].map(feature_family)
    ranking["rank"] = range(1, len(ranking) + 1)

    args.output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output / "feature-ranking.csv", index=False)
    matrix.to_csv(args.output / "correlation.csv")
    redundant.to_csv(args.output / "redundant-pairs.csv", index=False)
    family_importance.to_csv(args.output / "family-ranking.csv", index=False)
    joblib.dump(artifacts[selected["name"]], args.output / "selected-rich-model.joblib")
    coverage = (rich[list(RICH_SCHEMA.features)].notna().mean()).to_dict()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "runners": len(rich), "races": int(rich["race_id"].nunique()),
            "date_min": rich["date"].min().date().isoformat(),
            "date_max": rich["date"].max().date().isoformat(),
            "train_races": int(rich_splits.train["race_id"].nunique()),
            "validation_races": int(rich_splits.validation["race_id"].nunique()),
            "test_races": int(rich_splits.test["race_id"].nunique()),
        },
        "methodology": {
            "association": "Training-only point-biserial or Cramer's V with Benjamini-Hochberg FDR correction.",
            "correlation": "Training-only Spearman correlation; pairs flagged at absolute rho >= 0.80.",
            "importance": "Held-out increase in race log loss after deterministic global permutation.",
            "selection": "Rich model family selected only by validation race log loss.",
        },
        "schemas": {
            BASELINE_SCHEMA.name: {"features": list(BASELINE_SCHEMA.features)},
            RICH_SCHEMA.name: {"features": list(RICH_SCHEMA.features)},
        },
        "benter_coverage": list(BENTER_COVERAGE),
        "feature_origins": {key: list(value) for key, value in FEATURE_ORIGINS.items()},
        "coverage": coverage,
        "candidates": candidates,
        "selected_rich_model": selected["name"],
        "feature_ranking": ranking.to_dict(orient="records"),
        "family_ranking": family_importance.to_dict(orient="records"),
        "redundant_pairs": redundant.to_dict(orient="records"),
    }
    report["data_provenance"] = data_provenance(report, args.canonical)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    publish_dashboard(report, args.dashboard_results, args.dashboard_template, args.dashboard_output)
    print(json.dumps({
        "selected": selected["name"], "top_features": ranking.head(10)["feature"].tolist(),
        "candidates": candidates,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())