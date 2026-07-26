from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ima.data import build_full_history_dataset, chronological_race_split
from ima.feature_analysis import association_report, correlation_report, permutation_importance_report
from ima.feature_sets import (
    BASELINE_SCHEMA, FEATURE_FAMILIES, FEATURE_SCHEMAS, NOTEBOOK_FEATURE_FAMILIES,
    NOTEBOOK_RICH_SCHEMA, RICH_SCHEMA,
)
from ima.rich_features import load_full_rich_history
from scripts.run_feature_study import matrix_payload, publish_dashboard


def feature_families(schema_name: str) -> dict:
    if schema_name == BASELINE_SCHEMA.name:
        return {"baseline": BASELINE_SCHEMA.features}
    if schema_name == RICH_SCHEMA.name:
        return FEATURE_FAMILIES
    if schema_name == NOTEBOOK_RICH_SCHEMA.name:
        return NOTEBOOK_FEATURE_FAMILIES
    raise ValueError(f"Unsupported feature schema: {schema_name}")


def legacy_benter_study(report: dict) -> dict:
    return {
        "created_at": report.get("created_at"),
        "dataset": report.get("dataset", {}),
        "selected_model": report.get("selected_rich_model"),
        "coverage": report.get("coverage", {}),
        "feature_ranking": report.get("feature_ranking", []),
        "family_ranking": report.get("family_ranking", []),
        "redundant_pairs": report.get("redundant_pairs", []),
        "methodology": report.get("methodology", {}),
        "inactive_features": [],
    }


def load_dataset(args, schema_name: str):
    if schema_name == BASELINE_SCHEMA.name:
        return build_full_history_dataset(args.legacy_runs, args.legacy_races, args.canonical)
    return load_full_rich_history(
        args.legacy_runs, args.legacy_races, args.canonical,
        args.processed / "trackwork.csv.gz", args.processed / "barrier-trials.csv.gz",
        args.processed / "sectionals.csv.gz", args.processed / "horse-snapshots.csv.gz",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate schema-specific variable ranking and correlation")
    parser.add_argument("--schema", choices=sorted(FEATURE_SCHEMAS), required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--legacy-runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--legacy-races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/processed/historical/runners.csv.gz"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/historical"))
    parser.add_argument("--study-report", type=Path, default=Path("artifacts/feature-study/report.json"))
    parser.add_argument("--dashboard-results", type=Path, default=Path("public/results.json"))
    parser.add_argument("--dashboard-template", type=Path, default=Path("docs/model-results/dashboard-template.html"))
    parser.add_argument("--dashboard-output", type=Path, default=Path("public/index.html"))
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument(
        "--refresh-coverage-only", action="store_true",
        help="Refresh full/training coverage in an existing schema study without recomputing analysis.",
    )
    args = parser.parse_args()

    schema = FEATURE_SCHEMAS[args.schema]
    frame = load_dataset(args, schema.name)
    splits = chronological_race_split(frame)
    coverage = frame[list(schema.features)].notna().mean().to_dict()
    training_coverage = splits.train[list(schema.features)].notna().mean().to_dict()
    inactive = [feature for feature, value in training_coverage.items() if value == 0]
    if args.refresh_coverage_only:
        report = json.loads(args.study_report.read_text(encoding="utf-8"))
        schema_report = report["schema_studies"][schema.name]
        schema_report["coverage"] = coverage
        schema_report["training_coverage"] = training_coverage
        schema_report["inactive_features"] = inactive
        args.study_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        publish_dashboard(report, args.dashboard_results, args.dashboard_template, args.dashboard_output)
        print(json.dumps({"schema": schema.name, "inactive_features": inactive}, indent=2))
        return 0
    artifact = joblib.load(args.model_artifact)
    model = artifact["model"]

    associations = association_report(splits.train, schema)
    matrix, redundant = correlation_report(splits.train, schema.numeric)
    groups = {feature: (feature,) for feature in schema.features}
    families = feature_families(schema.name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping features without any observed values")
        importance = permutation_importance_report(
            model, splits.test, groups, repeats=args.permutation_repeats,
        ).rename(columns={"name": "feature"})
        family_importance = permutation_importance_report(
            model, splits.test, families, repeats=args.permutation_repeats,
        ).rename(columns={"name": "family"})
    ranking = importance.merge(associations, on="feature", how="left")
    family_by_feature = {
        feature: family for family, features in families.items() for feature in features
    }
    ranking["family"] = ranking["feature"].map(family_by_feature)
    ranking["rank"] = range(1, len(ranking) + 1)
    report = json.loads(args.study_report.read_text(encoding="utf-8"))
    studies = report.setdefault("schema_studies", {})
    studies.setdefault(RICH_SCHEMA.name, legacy_benter_study(report))
    studies[schema.name] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "runners": len(frame), "races": int(frame["race_id"].nunique()),
            "date_min": frame["date"].min().date().isoformat(),
            "date_max": frame["date"].max().date().isoformat(),
            "train_races": int(splits.train["race_id"].nunique()),
            "validation_races": int(splits.validation["race_id"].nunique()),
            "test_races": int(splits.test["race_id"].nunique()),
        },
        "selected_model": args.model_name,
        "coverage": coverage,
        "training_coverage": training_coverage,
        "inactive_features": inactive,
        "feature_ranking": ranking.to_dict(orient="records"),
        "family_ranking": family_importance.to_dict(orient="records"),
        "redundant_pairs": redundant.to_dict(orient="records"),
        "methodology": {
            "association": "Training-only point-biserial or Cramer's V with FDR correction.",
            "correlation": "Training-only Spearman matrix sampled at up to 100,000 runners.",
            "importance": f"Held-out race-log-loss permutation contribution with {args.permutation_repeats} repeat(s).",
        },
    }
    report.setdefault("schemas", {})[schema.name] = {"features": list(schema.features)}
    report.setdefault("correlation_matrices", {})[schema.name] = matrix_payload(matrix)
    args.study_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    publish_dashboard(report, args.dashboard_results, args.dashboard_template, args.dashboard_output)
    print(json.dumps({
        "schema": schema.name,
        "selected_model": args.model_name,
        "features": len(schema.features),
        "inactive_features": inactive,
        "top_features": ranking.head(15)["feature"].tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
