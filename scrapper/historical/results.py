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


def parse_results(html: str) -> list[HistoricalResult]:
    results = []
    for row in table_rows(html):
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
    return results


def fetch_results(race_date: str, venue: str, race_no: int, timeout: float = 20.0) -> tuple[str, str]:
    query = urlencode({"RaceDate": race_date, "Racecourse": venue, "RaceNo": race_no})
    url = f"{RESULTS_URL}?{query}"
    request = Request(url, headers={"User-Agent": "iMa-historical-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace"), response.geturl()


def collect_result(race_date: str, venue: str, race_no: int, output_dir: Path) -> dict:
    html, source_url = fetch_results(race_date, venue, race_no)
    results = parse_results(html)
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
        "runners": [asdict(result) for result in results],
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
