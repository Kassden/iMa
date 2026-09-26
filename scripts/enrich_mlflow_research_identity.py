"""Expose important research recipe identity fields in the MLflow runs table."""

from __future__ import annotations

import argparse
import time

from mlflow.entities import Param, RunTag
from mlflow.tracking import MlflowClient

try:
    from ima.mlflow_tracking import RESEARCH_IDENTITY_PARAM_PATHS
except ImportError:  # Allows an operational rollout beside an older immutable release.
    RESEARCH_IDENTITY_PARAM_PATHS = {
        "feature_schema": "recipe.feature_schema",
        "model_kind": "recipe.model.kind",
        "train_window": "recipe.train_window",
        "calibration_kind": "recipe.calibration.kind",
        "blend_kind": "recipe.blend.kind",
    }


def identity_updates(params: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Build missing short params and corresponding filterable tags."""
    aliases = {
        alias: params[nested]
        for alias, nested in RESEARCH_IDENTITY_PARAM_PATHS.items()
        if alias not in params and nested in params
    }
    tag_values = {
        f"ima.{alias}": params.get(alias, params.get(nested, ""))
        for alias, nested in RESEARCH_IDENTITY_PARAM_PATHS.items()
        if params.get(alias, params.get(nested)) is not None
    }
    return aliases, tag_values


def enrich_once(
    client: MlflowClient,
    experiment_name: str,
    *,
    include_tags: bool = False,
    progress_every: int = 100,
) -> dict[str, int]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = []
    page_token = None
    while True:
        page = client.search_runs(
            [experiment.experiment_id],
            max_results=1000,
            page_token=page_token,
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            break
    updated = 0
    params_added = 0
    tags_added = 0
    for run in runs:
        aliases, desired_tags = identity_updates(run.data.params)
        tags = {}
        if include_tags:
            tags = {
                key: value
                for key, value in desired_tags.items()
                if run.data.tags.get(key) != value
            }
        if not aliases and not tags:
            continue
        client.log_batch(
            run.info.run_id,
            params=[Param(key, value) for key, value in aliases.items()],
            tags=[RunTag(key, value) for key, value in tags.items()],
        )
        updated += 1
        params_added += len(aliases)
        tags_added += len(tags)
        if progress_every and updated % progress_every == 0:
            print(
                {"runs_updated": updated, "params_added": params_added, "tags_added": tags_added},
                flush=True,
            )
    return {
        "runs_scanned": len(runs),
        "runs_updated": updated,
        "params_added": params_added,
        "tags_added": tags_added,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add concise recipe identity params and tags to MLflow research runs"
    )
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default="ima-agentic-v2")
    parser.add_argument("--interval-seconds", type=float, default=0)
    parser.add_argument(
        "--include-tags",
        action="store_true",
        help="Also backfill filter tags; new logger versions add them automatically",
    )
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    while True:
        print(enrich_once(client, args.experiment, include_tags=args.include_tags), flush=True)
        if args.interval_seconds <= 0:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
