"""Propose, build, inspect and retry features from registered server parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.feature_lifecycle import (
    EXPERIMENT_NAME, FeatureLifecycle, _parent, build_feature_candidate, catalog,
    deterministic_candidate_definitions, record_model_outcome, retry_tracking,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("propose", "build"):
        command = commands.add_parser(name)
        command.add_argument("--parent-features", type=Path, required=True)
        command.add_argument("--parent-manifest", type=Path)
        command.add_argument("--limit", type=int, default=4)
        if name == "build":
            command.add_argument("--definitions", type=Path, help="JSON list of typed FeatureDefinition objects; absent uses deterministic candidates")
            command.add_argument("--min-coverage", type=float, default=0.1)
            command.add_argument("--submit-only", action="store_true")
            command.add_argument("--tracking-uri")
            command.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    command = commands.add_parser("catalog")
    command.add_argument("--limit", type=int, default=20)
    command = commands.add_parser("retry-tracking")
    command.add_argument("--build-id", required=True)
    command.add_argument("--tracking-uri", required=True)
    command.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    command = commands.add_parser("outcome")
    command.add_argument("--build-id", required=True)
    command.add_argument("--outcome", type=Path, required=True)
    command.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            result = catalog(args.registry, limit=args.limit)
        elif args.command == "retry-tracking":
            result = retry_tracking(args.registry, args.build_id, tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)
        elif args.command == "outcome":
            result = record_model_outcome(args.registry, args.build_id, json.loads(args.outcome.read_text()),
                                          baseline=json.loads(args.baseline.read_text()) if args.baseline else None)
        else:
            manifest_path = args.parent_manifest or args.parent_features.parent / "manifest.json"
            parent = _parent(args.parent_features, manifest_path)
            definitions = json.loads(args.definitions.read_text()) if args.command == "build" and args.definitions else deterministic_candidate_definitions(parent["predictor_catalog"], limit=args.limit)
            if args.command == "propose":
                result = [definition.model_dump(mode="json") for definition in definitions]
            elif args.submit_only:
                result = FeatureLifecycle(args.registry).submit(args.parent_features, manifest_path,
                    definitions=definitions, min_coverage=args.min_coverage)
            else:
                result = build_feature_candidate(args.parent_features, manifest_path, args.registry,
                    definitions=definitions, min_coverage=args.min_coverage,
                    tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
