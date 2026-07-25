"""Scrape HKJC horse profile, form, trackwork, veterinary, and movement pages."""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://racing.hkjc.com/en-us/local/information"
ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", " ", fragment, flags=re.IGNORECASE)
    return " ".join(html.unescape(TAG_RE.sub(" ", fragment)).split())


def table_rows(source: str) -> list[list[str]]:
    return [[_text(cell) for cell in CELL_RE.findall(row)] for row in ROW_RE.findall(source)]


def _number(value: Any) -> float | None:
    if value in (None, "", "-", "---"):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _placing(value: str) -> int | None:
    match = re.match(r"\s*0*(\d+)", value)
    return int(match.group(1)) if match else None


def _finish_seconds(value: str) -> float | None:
    parts = value.strip().split(".")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 100
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class FormRecord:
    race_index: str
    placing: int | None
    date: str
    course: str
    distance: float | None
    going: str
    race_class: str
    draw: int | None
    rating: float | None
    trainer: str
    jockey: str
    lengths_behind: str
    win_odds: float | None
    actual_weight: float | None
    running_position: str
    finish_time: float | None
    declared_weight: float | None
    gear: str


@dataclass
class HorsePageData:
    horse_page_id: str
    horse_name: str = ""
    brand_code: str = ""
    country: str | None = None
    age: int | None = None
    colour: str | None = None
    sex: str | None = None
    trainer: str | None = None
    owner: str | None = None
    current_rating: float | None = None
    wins: int | None = None
    seconds: int | None = None
    thirds: int | None = None
    starts: int | None = None
    form_records: list[FormRecord] = field(default_factory=list)
    trackwork: list[dict[str, str]] = field(default_factory=list)
    veterinary: list[dict[str, str]] = field(default_factory=list)
    movements: list[dict[str, str]] = field(default_factory=list)
    source_urls: dict[str, str] = field(default_factory=dict)
    fetched_at: str = ""
    errors: list[str] = field(default_factory=list)

    def model_features(self, before_date: str | None = None) -> dict[str, Any]:
        cutoff = datetime.strptime(before_date, "%Y-%m-%d").date() if before_date else None
        valid = []
        for record in self.form_records:
            if record.placing is None:
                continue
            record_date = datetime.strptime(record.date, "%d/%m/%y").date()
            if cutoff is None or record_date < cutoff:
                valid.append(record)
        previous = valid[0] if valid else None
        prior_results = [record.placing for record in valid if record.placing is not None]
        return {
            "horse_age": self.age,
            "horse_country": self.country,
            "horse_type": self.sex,
            "last_speed": (
                round(previous.distance / previous.finish_time, 4)
                if previous and previous.distance and previous.finish_time
                else None
            ),
            "avg_2last": sum(prior_results[:2]) / 2 if len(prior_results) >= 2 else None,
            "won_odds": self.wins if self.wins is not None else sum(value == 1 for value in prior_results),
            "second_count": self.seconds if self.seconds is not None else sum(value == 2 for value in prior_results),
            "third_count": self.thirds if self.thirds is not None else sum(value == 3 for value in prior_results),
            "exp": self.starts if self.starts is not None else len(valid),
            "raced": int(bool(valid)),
            "fin_time": previous.finish_time if previous else None,
            "prev_dist": previous.distance if previous else None,
            "cum_avg_prev_resu": sum(prior_results) / len(prior_results) if prior_results else None,
            "prev_resu": previous.placing if previous else None,
            "prev_odds": previous.win_odds if previous else None,
            "prev_wt": previous.actual_weight if previous else None,
            "prev_declar_wt": previous.declared_weight if previous else None,
        }

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_features"] = self.model_features()
        return result


def parse_profile_and_form(source: str, horse_page_id: str) -> HorsePageData:
    rows = table_rows(source)
    data = HorsePageData(horse_page_id=horse_page_id)
    title_match = re.search(r'<span\b[^>]*class="title_text"[^>]*>(.*?)</span>', source, re.I | re.S)
    if title_match:
        title = _text(title_match.group(1))
        match = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", title)
        data.horse_name = match.group(1) if match else title
        data.brand_code = match.group(2) if match else ""

    profile = {row[0]: row[2] for row in rows if len(row) == 3 and row[1] == ":"}
    country_age = profile.get("Country of Origin / Age", "").split("/")
    if country_age:
        data.country = country_age[0].strip() or None
    if len(country_age) > 1:
        data.age = _integer(country_age[1])
    colour_sex = profile.get("Colour / Sex", "").split("/")
    if colour_sex:
        data.colour = colour_sex[0].strip() or None
    if len(colour_sex) > 1:
        data.sex = colour_sex[1].strip() or None
    data.trainer = profile.get("Trainer")
    data.owner = profile.get("Owner")
    data.current_rating = _number(profile.get("Current Rating"))
    starts = [_integer(value) for value in profile.get("No. of 1-2-3-Starts*", "").split("-")]
    if len(starts) == 4:
        data.wins, data.seconds, data.thirds, data.starts = starts

    for row in rows:
        if len(row) < 18 or not re.fullmatch(r"\d+", row[0]):
            continue
        if not re.fullmatch(r"\d{2}/\d{2}/\d{2}", row[2]):
            continue
        data.form_records.append(
            FormRecord(
                race_index=row[0], placing=_placing(row[1]), date=row[2], course=row[3],
                distance=_number(row[4]), going=row[5], race_class=row[6], draw=_integer(row[7]),
                rating=_number(row[8]), trainer=row[9], jockey=row[10], lengths_behind=row[11],
                win_odds=_number(row[12]), actual_weight=_number(row[13]), running_position=row[14],
                finish_time=_finish_seconds(row[15]), declared_weight=_number(row[16]), gear=row[17],
            )
        )
    return data


def _records(source: str, headers: tuple[str, ...]) -> list[dict[str, str]]:
    records = []
    for row in table_rows(source):
        if len(row) == len(headers) and row != list(headers) and re.fullmatch(r"\d{2}/\d{2}/\d{4}", row[0]):
            records.append(dict(zip(headers, row)))
    return records


class HorsePageClient:
    def __init__(self, timeout_seconds: float = 20.0, delay_seconds: float = 0.2):
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds

    def _fetch(self, path: str, horse_page_id: str) -> tuple[str, str]:
        url = f"{BASE_URL}/{path}?{urlencode({'horseid': horse_page_id})}"
        request = Request(url, headers={"User-Agent": "iMa-live-race-collector/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                source = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"horse page request failed for {url}: {exc}") from exc
        if "No information." in source:
            raise RuntimeError(f"horse page returned no information for {horse_page_id}")
        time.sleep(self.delay_seconds)
        return url, source

    def fetch_all(self, horse_page_id: str) -> HorsePageData:
        fetched_at = datetime.now().astimezone().isoformat()
        profile_url, profile_source = self._fetch("horse", horse_page_id)
        data = parse_profile_and_form(profile_source, horse_page_id)
        data.fetched_at = fetched_at
        data.source_urls["profile_and_form"] = profile_url
        resources = (
            ("trackwork", "trackworkresult", ("date", "type", "track", "workouts", "gear")),
            ("veterinary", "ovehorse", ("date", "details", "passed_date")),
            ("movements", "movementrecords", ("from", "to", "arrival_date")),
        )
        for name, path, headers in resources:
            try:
                url, source = self._fetch(path, horse_page_id)
                data.source_urls[name] = url
                if name == "trackwork":
                    data.trackwork = _records(source, headers)
                elif name == "veterinary":
                    data.veterinary = _records(source, headers)
                else:
                    movement_rows = []
                    for row in table_rows(source):
                        if len(row) == 3 and re.fullmatch(r"\d{2}/\d{2}/\d{4}", row[2]):
                            movement_rows.append(dict(zip(headers, row)))
                    data.movements = movement_rows
            except RuntimeError as exc:
                data.errors.append(str(exc))
        return data


def write_horse_page_fixture(data: HorsePageData, path: Path) -> None:
    path.write_text(json.dumps(data.serializable(), indent=2, sort_keys=True), encoding="utf-8")
