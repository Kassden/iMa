"""Official HKJC local result archive parser and collector."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scrapper.horse_pages import table_rows


RESULTS_URL = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"


@dataclass(frozen=True)
class HistoricalResult:
    place: int
    horse_no: int
    horse_name: str
    horse_code: str
    jockey: str
    trainer: str
    actual_weight: float
    declared_weight: float
    draw: int
    lengths_behind: str
    running_position: str
    finish_time: str
    win_odds: float


@dataclass(frozen=True)
class HistoricalRace:
    race_class: str | None
    distance: int | None
    prize: float | None
    going: str | None
    course: str | None
    race_name: str | None
    runners: list[HistoricalResult]


def _money(value: str) -> float | None:
    match = re.search(r"(?:HK\$|\$)\s*([\d,]+(?:\.\d+)?)", value)
    return float(match.group(1).replace(",", "")) if match else None


def parse_race(html: str) -> HistoricalRace:
    rows = table_rows(html)
    race_class = None
    distance = None
    prize = None
    going = None
    course = None
    race_name = None
    for index, row in enumerate(rows):
        if not row:
            continue
        details = re.match(r"^(.+?)\s+-\s+(\d+)M(?:\s+-\s+.*)?$", row[0].strip())
        if not details:
            continue
        race_class = details.group(1).strip()
        distance = int(details.group(2))
        if len(row) >= 3 and row[1].strip().lower().startswith("going"):
            going = row[2].strip() or None
        if index + 1 < len(rows):
            next_row = rows[index + 1]
            race_name = (next_row[0].strip() or None) if next_row else None
            if len(next_row) >= 3 and next_row[1].strip().lower().startswith("course"):
                course = next_row[2].strip() or None
        if index + 2 < len(rows):
            prize = _money(rows[index + 2][0])
        break

    results = []
    for row in rows:
        placing = re.match(r"(\d+)", row[0]) if row else None
        if len(row) < 12 or not placing or not row[1].isdigit():
            continue
        horse_match = re.match(r"(.+?)\s*\([A-Z]*([A-Z]\d{3})\)\s*$", row[2])
        horse_name = horse_match.group(1).strip() if horse_match else row[2].strip()
        horse_code = horse_match.group(2) if horse_match else ""
        try:
            results.append(HistoricalResult(
                place=int(placing.group(1)), horse_no=int(row[1]), horse_name=horse_name,
                horse_code=horse_code, jockey=row[3], trainer=row[4],
                actual_weight=float(row[5]), declared_weight=float(row[6]), draw=int(row[7]),
                lengths_behind=row[8], running_position=row[9], finish_time=row[10],
                win_odds=float(row[11]),
            ))
        except ValueError:
            continue
    if not results:
        raise ValueError("No result runners found in HKJC archive page")
    return HistoricalRace(
        race_class=race_class, distance=distance, prize=prize, going=going,
        course=course, race_name=race_name, runners=results,
    )


def parse_results(html: str) -> list[HistoricalResult]:
    return parse_race(html).runners


def fetch_results(race_date: str, venue: str, race_no: int, timeout: float = 20.0) -> tuple[str, str]:
    query = urlencode({"RaceDate": race_date, "Racecourse": venue, "RaceNo": race_no})
    url = f"{RESULTS_URL}?{query}"
    request = Request(url, headers={"User-Agent": "iMa-historical-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace"), response.geturl()


def collect_result(race_date: str, venue: str, race_no: int, output_dir: Path) -> dict:
    html, source_url = fetch_results(race_date, venue, race_no)
    race = parse_race(html)
    token = f"{race_date.replace('/', '-')}_{venue}_R{race_no}"
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{token}.html"
    normalized_path = normalized_dir / f"{token}.json"
    if not raw_path.exists():
        raw_path.write_text(html, encoding="utf-8")
    record = {
        "race_date": race_date,
        "venue": venue,
        "race_no": race_no,
        "source_url": source_url,
        "race_class": race.race_class,
        "distance": race.distance,
        "prize": race.prize,
        "going": race.going,
        "course": race.course,
        "race_name": race.race_name,
        "runners": [asdict(result) for result in race.runners],
    }
    normalized_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def collect_meeting(
    race_date: str,
    venue: str,
    output_dir: Path,
    max_races: int = 12,
    delay_seconds: float = 0.5,
) -> list[dict]:
    import time

    races = []
    for race_no in range(1, max_races + 1):
        try:
            races.append(collect_result(race_date, venue, race_no, output_dir))
        except ValueError:
            if race_no == 1:
                return []
            break
        if delay_seconds:
            time.sleep(delay_seconds)
    return races
