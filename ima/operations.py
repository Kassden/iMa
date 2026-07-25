"""Post-meeting finalization and guarded retraining orchestration."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from scrapper.historical.results import collect_meeting

from .registry import ModelRegistry
from .training import train_and_report


def validate_meeting(races: list[dict]) -> None:
    if not races:
        raise ValueError("No official races were collected for the meeting")
    race_numbers = [race["race_no"] for race in races]
    if len(race_numbers) != len(set(race_numbers)):
        raise ValueError("Duplicate race numbers in meeting")
    for race in races:
        runners = race.get("runners") or []
        if not runners:
            raise ValueError(f"Race {race['race_no']} has no runners")
        winners = [runner for runner in runners if runner["place"] == 1]
        if not winners:
            raise ValueError(f"Race {race['race_no']} has no official winner")


def finalize_meeting(
    race_date: str,
    venue: str,
    output_dir: Path,
    runs_path: Path,
    races_path: Path,
    artifact_dir: Path,
    registry_dir: Path,
    collector: Callable[..., list[dict]] = collect_meeting,
    trainer: Callable[..., dict] = train_and_report,
) -> dict:
    meeting_dir = output_dir / race_date.replace("/", "-") / venue
    official = collector(race_date, venue, meeting_dir / "official")
    validate_meeting(official)
    canonical = json.dumps(official, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(canonical).hexdigest()
    version = f"{race_date.replace('/', '')}-{venue}-{checksum[:10]}"
    model_dir = artifact_dir / version
    report = trainer(runs_path, races_path, model_dir)
    registry = ModelRegistry(registry_dir)
    registry_path = registry.register(
        version,
        report,
        [str(path) for path in sorted(model_dir.glob("*"))],
    )
    manifest = {
        "meeting_version": version,
        "race_date": race_date,
        "venue": venue,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "official_checksum": checksum,
        "race_count": len(official),
        "runner_count": sum(len(race["runners"]) for race in official),
        "model_registry_record": str(registry_path),
        "recommended_champion": report["recommended_champion"],
        "promotion_status": "awaiting_metric_comparison_and_operator_approval",
        "canonical_enrichment_status": (
            "official outcomes preserved; feature-complete racecard/horse snapshots are required "
            "before these rows enter the training table"
        ),
    }
    manifest_path = meeting_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
