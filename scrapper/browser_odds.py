"""Browser-backed HKJC odds provider using the public rendered odds app."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


DEFAULT_ODDS_URL = "https://bet.hkjc.com/en/racing/wp"
DEFAULT_CHROME_PATHS = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
)


class BrowserOddsError(RuntimeError):
    """Rendered odds could not be collected or parsed."""


class _IdTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str | None] = []
        self.text: dict[str, list[str]] = {}
        self.links: dict[str, list[str]] = {}
        self.document_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        self.stack.append(element_id)
        if element_id:
            self.text.setdefault(element_id, [])
        href = attributes.get("href")
        if href:
            for parent_id in reversed(self.stack):
                if parent_id:
                    self.links.setdefault(parent_id, []).append(href)
                    break

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if not value:
            return
        self.document_text.append(value)
        for element_id in reversed(self.stack):
            if element_id:
                self.text.setdefault(element_id, []).append(value)
                break

    def value(self, element_id: str) -> str | None:
        value = " ".join(self.text.get(element_id, [])).strip()
        return value or None


def _number(value: str | None) -> float | None:
    if not value or value.strip().upper() in {"---", "SCR", "RFD", "-"}:
        return None
    match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None


def _money(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", value)
    return float(match.group(1).replace(",", "")) if match else None


def _horse_page_id(hrefs: list[str]) -> str:
    for href in hrefs:
        for pattern in (r"(?:HorseNo|horseNo|horseno)=([^&#]+)", r"/horses?/([^/?#]+)"):
            match = re.search(pattern, href)
            if match:
                return match.group(1)
    return ""


def _infer_date(text: str) -> str:
    match = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text)
    if not match:
        return datetime.now(timezone.utc).date().isoformat()
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def parse_rendered_odds(
    html: str,
    race_no: int,
    race_date: str | None = None,
    venue_code: str | None = None,
    received_at: str | None = None,
) -> dict[str, Any]:
    parser = _IdTextParser()
    parser.feed(html)
    received_at = received_at or datetime.now(timezone.utc).isoformat()
    document = " ".join(parser.document_text)
    race_date = race_date or _infer_date(document)
    venue_code = venue_code or parser.value("venue_S1") or "HKJC"
    runner_indexes = sorted(
        {
            int(match.group(1))
            for element_id in parser.text
            if (match := re.fullmatch(rf"runnerNo_{race_no}_(\d+)", element_id))
        }
    )
    if not runner_indexes:
        raise BrowserOddsError(f"Rendered page contains no runners for race {race_no}")

    runners: list[dict[str, Any]] = []
    win_nodes: list[dict[str, Any]] = []
    place_nodes: list[dict[str, Any]] = []
    for index in runner_indexes:
        suffix = f"{race_no}_{index}"
        horse_no = parser.value(f"runnerNo_{suffix}") or str(index)
        horse_name = parser.value(f"horseName_{suffix}") or ""
        page_id = _horse_page_id(parser.links.get(f"horseName_{suffix}", []))
        win_text = parser.value(f"odds_WIN_{suffix}")
        win = _number(win_text)
        place = _number(parser.value(f"odds_PLA_{suffix}"))
        status = "SCRATCHED" if win_text and "SCR" in win_text.upper() else "RUNNER"
        runners.append(
            {
                "id": page_id or f"browser:{race_date}:{venue_code}:{race_no}:{horse_no}",
                "no": horse_no,
                "saddleClothNo": horse_no,
                "status": status,
                "name_en": horse_name,
                "barrierDrawNumber": _number(parser.value(f"draw_{suffix}")),
                "handicapWeight": _number(parser.value(f"handicapWt_{suffix}")),
                "currentWeight": _number(parser.value(f"declaredWt_{suffix}")),
                "currentRating": _number(parser.value(f"rating_{suffix}")),
                "gearInfo": parser.value(f"gear_{suffix}"),
                "winOdds": win,
                "horse": {"id": page_id, "code": page_id},
                "jockey": {"code": parser.value(f"jockey_{suffix}"), "name_en": parser.value(f"jockey_{suffix}")},
                "trainer": {"code": parser.value(f"trainer_{suffix}"), "name_en": parser.value(f"trainer_{suffix}")},
            }
        )
        if win is not None:
            win_nodes.append({"combString": horse_no, "oddsValue": win})
        if place is not None:
            place_nodes.append({"combString": horse_no, "oddsValue": place})

    meeting_id = f"browser:{race_date}:{venue_code}"
    return {
        "data": {"raceMeetings": [{
            "id": meeting_id,
            "venueCode": venue_code,
            "date": race_date,
            "status": "SELLING",
            "races": [{
                "id": f"{meeting_id}:R{race_no}",
                "no": race_no,
                "status": "SELLING",
                "postTime": None,
                "wageringFieldSize": len(runners),
                "runners": runners,
            }],
            "pmPools": [
                {"oddsType": "WIN", "lastUpdateTime": received_at, "oddsNodes": win_nodes},
                {"oddsType": "PLA", "lastUpdateTime": received_at, "oddsNodes": place_nodes},
            ],
        }]},
        "provider": {
            "kind": "browser",
            "source_url": DEFAULT_ODDS_URL,
            "received_at": received_at,
            "available_pool_ids": sorted(
                element_id.removeprefix("poolInv")
                for element_id in parser.text
                if element_id.startswith("poolInv")
            ),
            "pool_turnover": {
                element_id.removeprefix("poolInv"): amount
                for element_id in parser.text
                if element_id.startswith("poolInv")
                if (amount := _money(parser.value(element_id))) is not None
            },
        },
    }


@dataclass
class BrowserOddsClient:
    url: str = DEFAULT_ODDS_URL
    chrome_path: Path | None = None
    timeout_seconds: float = 35.0
    virtual_time_ms: int = 15_000
    include_all_pools: bool = False

    def _chrome(self) -> Path:
        if self.chrome_path and self.chrome_path.exists():
            return self.chrome_path
        for candidate in DEFAULT_CHROME_PATHS:
            if candidate.exists():
                return candidate
        raise BrowserOddsError("Google Chrome or Chromium was not found")

    def render(self) -> str:
        with tempfile.TemporaryDirectory(prefix="ima-hkjc-") as profile:
            command = [
                str(self._chrome()), "--headless=new", "--disable-gpu", "--no-first-run",
                "--disable-background-networking", f"--user-data-dir={profile}",
                f"--virtual-time-budget={self.virtual_time_ms}", "--dump-dom", self.url,
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                )
                html = completed.stdout
            except subprocess.TimeoutExpired as exc:
                html = exc.stdout or ""
                if isinstance(html, bytes):
                    html = html.decode("utf-8", errors="replace")
            if "runnerNo_" not in html or "odds_WIN_" not in html:
                raise BrowserOddsError("Chrome did not render the expected HKJC odds table")
            return html

    def fetch_race_snapshot(
        self,
        race_no: int,
        race_date: str | None = None,
        venue_code: str | None = None,
    ) -> dict[str, Any]:
        if self.include_all_pools:
            from .browser_pools import InteractivePoolClient

            html, pool_pages = InteractivePoolClient(
                url=self.url, chrome_path=self.chrome_path
            ).collect()
            payload = parse_rendered_odds(html, race_no, race_date, venue_code)
            payload["provider"]["pool_pages"] = pool_pages
            return payload
        return parse_rendered_odds(self.render(), race_no, race_date, venue_code)
