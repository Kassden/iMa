"""Resumable date-range acquisition for the official HKJC result archive."""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .results import fetch_results, parse_results


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  race_date TEXT NOT NULL,
  venue TEXT NOT NULL,
  status TEXT NOT NULL,
  race_count INTEGER NOT NULL DEFAULT 0,
  runner_count INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  checked_at TEXT NOT NULL,
  PRIMARY KEY (race_date, venue)
);
"""


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


class BulkArchiveCollector:
    def __init__(
        self,
        output_dir: Path,
        workers: int = 4,
        request_delay: float = 0.25,
        max_races: int = 12,
        request_retries: int = 3,
        retry_backoff: float = 0.5,
    ):
        self.output_dir = output_dir
        self.workers = workers
        self.request_delay = request_delay
        self.max_races = max_races
        self.request_retries = request_retries
        self.retry_backoff = retry_backoff
        output_dir.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(output_dir / "checkpoint.sqlite", check_same_thread=False)
        self.database.executescript(SCHEMA)

    def _fetch_with_retry(self, display_date: str, venue: str, race_no: int):
        for attempt in range(self.request_retries):
            try:
                return fetch_results(display_date, venue, race_no)
            except OSError:
                if attempt + 1 == self.request_retries:
                    raise
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise RuntimeError("unreachable")

    def pending(self, start: date, end: date) -> list[tuple[str, str]]:
        completed = {
            (row[0], row[1])
            for row in self.database.execute(
                "SELECT race_date, venue FROM scans "
                "WHERE status IN ('complete', 'no_meeting', 'archive_unavailable')"
            )
        }
        return [
            (day.isoformat(), venue)
            for day in date_range(start, end)
            for venue in ("ST", "HV")
            if (day.isoformat(), venue) not in completed
        ]

    def _collect(self, iso_date: str, venue: str) -> dict:
        display_date = iso_date.replace("-", "/")
        races = []
        for race_no in range(1, self.max_races + 1):
            html, source_url = self._fetch_with_retry(display_date, venue, race_no)
            try:
                runners = parse_results(html)
            except ValueError:
                if race_no == 1:
                    return {"race_date": iso_date, "venue": venue, "status": "no_meeting"}
                break
            race_dir = self.output_dir / "raw" / iso_date / venue
            race_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(race_dir / f"R{race_no}.html.gz", "wt", encoding="utf-8") as handle:
                handle.write(html)
            races.append({
                "race_date": iso_date,
                "venue": venue,
                "race_no": race_no,
                "source_url": source_url,
                "runners": [asdict(runner) for runner in runners],
            })
            if self.request_delay:
                time.sleep(self.request_delay)
        normalized = self.output_dir / "normalized" / iso_date
        normalized.mkdir(parents=True, exist_ok=True)
        (normalized / f"{venue}.json").write_text(json.dumps(races, indent=2), encoding="utf-8")
        return {
            "race_date": iso_date,
            "venue": venue,
            "status": "complete",
            "race_count": len(races),
            "runner_count": sum(len(race["runners"]) for race in races),
        }

    def run(self, start: date, end: date, limit: int | None = None) -> dict:
        targets = [
            (day.isoformat(), venue)
            for day in date_range(start, end)
            for venue in ("ST", "HV")
        ]
        return self.run_targets(targets, limit)

    def run_targets(
        self,
        targets: list[tuple[str, str]],
        limit: int | None = None,
        known_meetings: bool = False,
    ) -> dict:
        completed = {
            (row[0], row[1])
            for row in self.database.execute(
                "SELECT race_date, venue FROM scans "
                "WHERE status IN ('complete', 'no_meeting', 'archive_unavailable')"
            )
        }
        pending = sorted(set(targets) - completed)
        if limit is not None:
            pending = pending[:limit]
        counts = {"complete": 0, "no_meeting": 0, "archive_unavailable": 0, "error": 0}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._collect, day, venue): (day, venue) for day, venue in pending}
            for future in as_completed(futures):
                day, venue = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"race_date": day, "venue": venue, "status": "error", "error": str(exc)}
                if known_meetings and result["status"] == "no_meeting":
                    result["status"] = "archive_unavailable"
                counts[result["status"]] += 1
                self.database.execute(
                    "INSERT OR REPLACE INTO scans VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        day, venue, result["status"], result.get("race_count", 0),
                        result.get("runner_count", 0), result.get("error"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self.database.commit()
        return {"attempted": len(pending), **counts, **self.coverage()}

    def coverage(self) -> dict:
        row = self.database.execute(
            "SELECT COUNT(*), SUM(status='complete'), SUM(race_count), "
            "SUM(runner_count), SUM(status='error') FROM scans"
        ).fetchone()
        return {
            "scanned_venue_dates": int(row[0] or 0),
            "meetings": int(row[1] or 0),
            "races": int(row[2] or 0),
            "runners": int(row[3] or 0),
            "errors_total": int(row[4] or 0),
        }
