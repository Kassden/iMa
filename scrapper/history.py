"""Historical feature lookup matching the legacy notebook definitions."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


@dataclass
class HistoricalFeatureStore:
    by_horse: dict[str, list[dict[str, str]]]
    live_to_history_id: dict[str, str]

    @classmethod
    def from_csv(
        cls,
        runs_path: Path,
        races_path: Path,
        identity_map_path: Path | None = None,
    ) -> "HistoricalFeatureStore":
        with races_path.open(newline="", encoding="utf-8-sig") as handle:
            races = {row["race_id"]: row for row in csv.DictReader(handle)}

        by_horse: dict[str, list[dict[str, str]]] = defaultdict(list)
        with runs_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                race = races.get(row["race_id"], {})
                row["race_date"] = race.get("date", "")
                row["distance"] = race.get("distance", "")
                by_horse[row["horse_id"]].append(row)
        for rows in by_horse.values():
            rows.sort(key=lambda row: (row["race_date"], _int(row["race_id"]) or -1))

        identity_map: dict[str, str] = {}
        if identity_map_path and identity_map_path.exists():
            with identity_map_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    identity_map[row["live_horse_code"]] = row["historical_horse_id"]
        return cls(dict(by_horse), identity_map)

    def features_for(self, live_horse_id: str, before_date: str) -> dict[str, Any]:
        historical_id = self.live_to_history_id.get(live_horse_id, live_horse_id)
        rows = [row for row in self.by_horse.get(historical_id, []) if row["race_date"] < before_date]
        if not rows:
            return {}

        previous = rows[-1]
        results = [_int(row.get("result")) for row in rows]
        valid_results = [result for result in results if result is not None]
        previous_distance = _float(previous.get("distance"))
        previous_race_id = previous.get("race_id")
        # Distance lives in races.csv in the source data and is not repeated in runs.csv.
        # The pipeline adds it when a richer history source is available.
        finish_time = _float(previous.get("finish_time"))
        speed = None
        if previous_distance and finish_time:
            speed = round(previous_distance / finish_time, 4)

        return {
            "horse_age": _int(previous.get("horse_age")),
            "horse_country": previous.get("horse_country") or None,
            "horse_type": previous.get("horse_type") or None,
            "last_speed": speed,
            "avg_2last": (
                sum(valid_results[-2:]) / 2 if len(valid_results) >= 2 else None
            ),
            "won_odds": sum(result == 1 for result in valid_results),
            "second_count": sum(result == 2 for result in valid_results),
            "third_count": sum(result == 3 for result in valid_results),
            "exp": len(rows),
            "raced": 1,
            "fin_time": finish_time,
            "prev_dist": previous_distance,
            "cum_avg_prev_resu": (
                sum(valid_results) / len(valid_results) if valid_results else None
            ),
            "prev_resu": valid_results[-1] if valid_results else None,
            "prev_odds": _float(previous.get("win_odds")),
            "prev_wt": _float(previous.get("actual_weight")),
            "prev_declar_wt": _float(previous.get("declared_weight")),
            "history_horse_id": historical_id,
            "history_last_race_id": previous_race_id,
        }
