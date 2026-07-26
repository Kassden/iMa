"""Normalize provider data and emit model-compatible race snapshots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    HISTORY_REQUIRED_COLUMNS,
    LIVE_REQUIRED_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    ModelRow,
    NormalizedRunner,
)
from .browser_odds import BrowserOddsClient
from .history import HistoricalFeatureStore
from .hkjc import HKJCClient
from .horse_pages import HorsePageClient, HorsePageData


def _number(value: Any) -> float | None:
    if value in (None, "", "---"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _odds_by_runner(pools: Iterable[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float], str | None]:
    win: dict[str, float] = {}
    place: dict[str, float] = {}
    latest_update: str | None = None
    for pool in pools:
        odds_type = pool.get("oddsType")
        if odds_type not in {"WIN", "PLA"}:
            continue
        if pool.get("lastUpdateTime") and (
            latest_update is None or pool["lastUpdateTime"] > latest_update
        ):
            latest_update = pool["lastUpdateTime"]
        target = win if odds_type == "WIN" else place
        for node in pool.get("oddsNodes") or []:
            combination = str(node.get("combString", "")).strip()
            value = _number(node.get("oddsValue"))
            if combination and value is not None:
                target[combination] = value
    return win, place, latest_update


def normalize_snapshot(
    payload: dict[str, Any],
    race_no: int,
    received_at: str | None = None,
) -> list[NormalizedRunner]:
    received_at = received_at or datetime.now(timezone.utc).isoformat()
    meetings = payload.get("data", {}).get("raceMeetings") or []
    if not meetings:
        raise ValueError("Snapshot contains no race meetings")

    for meeting in meetings:
        race = next(
            (item for item in meeting.get("races") or [] if _integer(item.get("no")) == race_no),
            None,
        )
        if race is None:
            continue
        win_odds, place_odds, odds_updated_at = _odds_by_runner(meeting.get("pmPools") or [])
        field_size = _integer(race.get("wageringFieldSize")) or len(race.get("runners") or [])
        normalized: list[NormalizedRunner] = []
        for runner in race.get("runners") or []:
            horse = runner.get("horse") or {}
            horse_no = str(runner.get("no") or runner.get("saddleClothNo") or "").strip()
            normalized.append(
                NormalizedRunner(
                    race_id=str(race.get("id") or f"{meeting.get('id')}:{race_no}"),
                    race_no=race_no,
                    race_date=str(meeting.get("date") or ""),
                    venue_code=str(meeting.get("venueCode") or ""),
                    post_time=race.get("postTime"),
                    race_status=str(race.get("status") or meeting.get("status") or ""),
                    field_size=field_size,
                    horse_no=horse_no,
                    horse_id=str(horse.get("code") or horse.get("id") or runner.get("id") or ""),
                    horse_page_id=str(horse.get("id") or ""),
                    horse_name=str(runner.get("name_en") or ""),
                    runner_status=str(runner.get("status") or ""),
                    jockey_id=(runner.get("jockey") or {}).get("code"),
                    trainer_id=(runner.get("trainer") or {}).get("code"),
                    declared_weight=_number(runner.get("currentWeight")),
                    actual_weight=_number(runner.get("handicapWeight")),
                    draw=_integer(runner.get("barrierDrawNumber")),
                    win_odds=win_odds.get(horse_no, _number(runner.get("winOdds"))),
                    place_odds=place_odds.get(horse_no),
                    race_class=str(race.get("claCode") or race.get("raceClass_en") or "") or None,
                    horse_rating=_number(runner.get("currentRating")),
                    horse_gear=runner.get("gearInfo"),
                    last6run=runner.get("last6run"),
                    odds_updated_at=odds_updated_at,
                    received_at=received_at,
                )
            )
        return normalized
    raise ValueError(f"Race {race_no} was not found in the snapshot")


def build_model_rows(
    runners: Iterable[NormalizedRunner],
    history: HistoricalFeatureStore | None = None,
    horse_pages: dict[str, HorsePageData] | None = None,
) -> list[ModelRow]:
    rows: list[ModelRow] = []
    for runner in runners:
        values = {name: None for name in MODEL_FEATURE_COLUMNS}
        values.update(
            {
                "horse_no_y": runner.field_size,
                "horse_id": runner.horse_id,
                "jockey_id": runner.jockey_id,
                "trainer_id": runner.trainer_id,
                "declared_weight": runner.declared_weight,
                "actual_weight": runner.actual_weight,
                "draw": runner.draw,
                "win_odds": runner.win_odds,
                "place_odds": runner.place_odds,
                "race_class": runner.race_class,
                "horse_rating": runner.horse_rating,
                "horse_gear": runner.horse_gear,
            }
        )
        if history:
            values.update(
                {
                    key: value
                    for key, value in history.features_for(runner.horse_id, runner.race_date).items()
                    if key in values
                }
            )
        page_data = (horse_pages or {}).get(runner.horse_page_id)
        if page_data:
            values.update(
                {
                    key: value
                    for key, value in page_data.model_features(runner.race_date).items()
                    if value is not None
                }
            )
        required = (*LIVE_REQUIRED_COLUMNS, *HISTORY_REQUIRED_COLUMNS)
        missing = [name for name in required if values.get(name) is None]
        active = runner.runner_status.upper() not in {"SCRATCHED", "WITHDRAWN", "RESERVE"}
        rows.append(
            ModelRow(
                race_id=runner.race_id,
                horse_no=runner.horse_no,
                values=values,
                missing_features=missing,
                prediction_ready=active and not missing,
            )
        )
    return rows


@dataclass
class ScrapeResult:
    raw_path: Path
    runners_path: Path
    model_path: Path
    report_path: Path
    horse_details_path: Path
    runner_count: int
    prediction_ready_count: int


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def write_snapshot(
    payload: dict[str, Any],
    runners: list[NormalizedRunner],
    model_rows: list[ModelRow],
    output_dir: Path,
    horse_pages: dict[str, HorsePageData] | None = None,
) -> ScrapeResult:
    if not runners:
        raise ValueError("Cannot write an empty race snapshot")
    first = runners[0]
    stamp = _safe_token(first.received_at.replace("+00:00", "Z"))
    stem = f"{_safe_token(first.race_date)}_{_safe_token(first.venue_code)}_R{first.race_no}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{stem}.raw.json"
    runners_path = output_dir / f"{stem}.runners.json"
    model_path = output_dir / f"{stem}.model.csv"
    report_path = output_dir / f"{stem}.report.json"
    horse_details_path = output_dir / f"{stem}.horse-details.json"

    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    runners_path.write_text(
        json.dumps([asdict(runner) for runner in runners], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    horse_details_path.write_text(
        json.dumps(
            {key: value.serializable() for key, value in (horse_pages or {}).items()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fieldnames = ["race_id", "horse_no_x", *MODEL_FEATURE_COLUMNS, "prediction_ready", "missing_features"]
    with model_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.csv_row() for row in model_rows)

    report = {
        "race_id": first.race_id,
        "race_date": first.race_date,
        "venue_code": first.venue_code,
        "race_no": first.race_no,
        "received_at": first.received_at,
        "odds_updated_at": first.odds_updated_at,
        "runner_count": len(runners),
        "prediction_ready_count": sum(row.prediction_ready for row in model_rows),
        "scratched_or_inactive": [
            runner.horse_no
            for runner in runners
            if runner.runner_status.upper() in {"SCRATCHED", "WITHDRAWN", "RESERVE"}
        ],
        "missing_features_by_runner": {
            row.horse_no: row.missing_features for row in model_rows if row.missing_features
        },
        "horse_page_errors": {
            key: value.errors for key, value in (horse_pages or {}).items() if value.errors
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return ScrapeResult(
        raw_path=raw_path,
        runners_path=runners_path,
        model_path=model_path,
        report_path=report_path,
        horse_details_path=horse_details_path,
        runner_count=len(runners),
        prediction_ready_count=report["prediction_ready_count"],
    )


def scrape_race(
    race_no: int,
    output_dir: Path,
    race_date: str | None = None,
    venue_code: str | None = None,
    endpoint: str | None = None,
    history: HistoricalFeatureStore | None = None,
    scrape_horse_pages: bool = True,
    provider: str = "browser",
    include_all_pools: bool = False,
) -> ScrapeResult:
    if provider == "browser":
        client = BrowserOddsClient(include_all_pools=include_all_pools)
    elif provider == "graphql":
        client = HKJCClient(endpoint=endpoint) if endpoint else HKJCClient()
    else:
        raise ValueError(f"Unknown odds provider: {provider}")
    payload = client.fetch_race_snapshot(race_no, race_date, venue_code)
    runners = normalize_snapshot(payload, race_no)
    horse_pages: dict[str, HorsePageData] = {}
    if scrape_horse_pages:
        page_client = HorsePageClient()
        for runner in runners:
            if runner.runner_status.upper() in {"SCRATCHED", "WITHDRAWN", "RESERVE"}:
                continue
            if not runner.horse_page_id or not runner.horse_page_id.upper().startswith("HK_"):
                continue
            try:
                horse_pages[runner.horse_page_id] = page_client.fetch_all(runner.horse_page_id)
            except RuntimeError as exc:
                failed = HorsePageData(horse_page_id=runner.horse_page_id, errors=[str(exc)])
                horse_pages[runner.horse_page_id] = failed
    model_rows = build_model_rows(runners, history, horse_pages)
    return write_snapshot(payload, runners, model_rows, output_dir, horse_pages)
