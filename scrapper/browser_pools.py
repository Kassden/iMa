"""Interactive browser collection for HKJC exotic and multi-race pools."""

from __future__ import annotations

import json
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from websocket import create_connection

from .browser_odds import DEFAULT_CHROME_PATHS, DEFAULT_ODDS_URL, BrowserOddsError


POOL_TABS = {
    "WPQ": "wpq",
    "FCT": "fct",
    "TCE": "tce",
    "TRI": "tri",
    "FF": "ff",
    "QTT": "qtt",
    "DBL": "dbl",
    "TBL": "tbl",
    "DT": "dt",
    "TT": "tt",
    "6UP": "6up",
    "JKC": "jkc",
    "TNC": "tnc",
    "JTC": "jtcombo",
}
TOP_POOL_SHAPES = {"TCE": 3, "TRI": 3, "FF": 4, "QTT": 4}


def parse_top_combinations(text: str, pool: str) -> list[dict[str, object]]:
    places = TOP_POOL_SHAPES.get(pool.upper())
    if places is None:
        return []
    combination_pattern = re.compile(rf"^\d+(?:-\d+){{{places - 1}}}$")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results = []
    for index, line in enumerate(lines[:-1]):
        if not combination_pattern.fullmatch(line):
            continue
        try:
            odds = float(lines[index + 1])
        except ValueError:
            continue
        results.append({"combination": line.split("-"), "decimal_odds": odds})
    unique = {}
    for result in results:
        unique[tuple(result["combination"])] = result
    return list(unique.values())[:20]


def parse_rectangular_matrix(
    text: str,
    header_marker: str,
    end_marker: str,
) -> list[dict[str, object]]:
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if header_marker in line)
        end = next(index for index in range(start + 1, len(lines)) if end_marker in lines[index])
        header_line = next(line for line in lines[start + 1:end] if line.startswith("\t1\t"))
    except StopIteration:
        return []
    columns = [item for item in header_line.split("\t") if item.isdigit()]
    results = []
    for line in lines[start + 1:end]:
        cells = line.split("\t")
        if not cells or not cells[0].isdigit():
            continue
        row_runner = cells[0]
        for column_runner, value in zip(columns, cells[1:]):
            if not value or not re.fullmatch(r"\d+(?:\.\d+)?", value):
                continue
            results.append({
                "combination": [column_runner, row_runner],
                "decimal_odds": float(value),
            })
    return results


def parse_dom_odds(elements: list[dict[str, str]]) -> list[dict[str, object]]:
    results = []
    for element in elements:
        match = re.fullmatch(r"qb_([A-Z0-9-]+)_([0-9_]+)", element.get("id", ""))
        if not match:
            continue
        try:
            odds = float(element.get("text", ""))
        except ValueError:
            continue
        results.append({
            "pool": match.group(1),
            "combination": match.group(2).split("_"),
            "decimal_odds": odds,
        })
    return results


def parse_leg_runners(elements: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    legs: dict[str, list[dict[str, object]]] = {}
    for element in elements:
        match = re.fullmatch(r"runnerNo_(\d+)_(\d+)", element.get("id", ""))
        if not match:
            continue
        race_no = match.group(1)
        legs.setdefault(race_no, []).append({
            "runner_no": element.get("runner_no", ""),
            "horse_name": element.get("horse_name", ""),
            "win_odds": float(element["win_odds"]) if re.fullmatch(r"\d+(?:\.\d+)?", element.get("win_odds", "")) else None,
        })
    return legs


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class InteractivePoolClient:
    url: str = DEFAULT_ODDS_URL
    chrome_path: Path | None = None
    page_wait_seconds: float = 30.0
    pool_wait_seconds: float = 2.0

    def _chrome(self) -> Path:
        if self.chrome_path and self.chrome_path.exists():
            return self.chrome_path
        for candidate in DEFAULT_CHROME_PATHS:
            if candidate.exists():
                return candidate
        raise BrowserOddsError("Google Chrome or Chromium was not found")

    def collect(self) -> tuple[str, dict[str, dict[str, object]]]:
        port = _free_port()
        with tempfile.TemporaryDirectory(prefix="ima-hkjc-cdp-") as profile:
            process = subprocess.Popen(
                [
                    str(self._chrome()), "--headless=new", "--disable-gpu", "--no-first-run",
                    f"--remote-debugging-port={port}", "--remote-allow-origins=*",
                    f"--user-data-dir={profile}", self.url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            websocket = None
            try:
                tabs = None
                for _ in range(80):
                    try:
                        tabs = json.load(urlopen(f"http://127.0.0.1:{port}/json", timeout=1))
                        break
                    except Exception:
                        time.sleep(0.25)
                if not tabs:
                    raise BrowserOddsError("Chrome DevTools endpoint did not become ready")
                page = next(item for item in tabs if item.get("type") == "page")
                websocket = create_connection(page["webSocketDebuggerUrl"], timeout=20)
                sequence = 0

                def evaluate(expression: str) -> object:
                    nonlocal sequence
                    sequence += 1
                    websocket.send(json.dumps({
                        "id": sequence,
                        "method": "Runtime.evaluate",
                        "params": {"expression": expression, "returnByValue": True},
                    }))
                    while True:
                        message = json.loads(websocket.recv())
                        if message.get("id") == sequence:
                            return message.get("result", {}).get("result", {}).get("value")

                initial_html = ""
                deadline = time.monotonic() + self.page_wait_seconds
                while time.monotonic() < deadline:
                    initial_html = str(evaluate("document.documentElement.outerHTML") or "")
                    if "runnerNo_" in initial_html:
                        break
                    time.sleep(1.0)
                if "runnerNo_" not in initial_html:
                    raise BrowserOddsError("Interactive Chrome session did not render runners")
                pages: dict[str, dict[str, object]] = {}
                for pool, tab_id in POOL_TABS.items():
                    clicked = evaluate(
                        f"document.getElementById({json.dumps(tab_id)}) ? "
                        f"(document.getElementById({json.dumps(tab_id)}).click(), true) : false"
                    )
                    if not clicked:
                        continue
                    time.sleep(self.pool_wait_seconds)
                    text = str(evaluate(
                        "document.getElementById('rcOddsTable')?.innerText || ''"
                    ) or "")
                    top_text = str(evaluate(
                        "document.getElementById('qTTOddsTableCollapse')?.innerText || ''"
                    ) or "")
                    dom_odds_json = str(evaluate(
                        "JSON.stringify(Array.from(document.querySelectorAll('[id^=\\\"qb_\\\"]'))"
                        ".map(e => ({id:e.id,text:e.innerText.trim()})))"
                    ) or "[]")
                    dom_odds = parse_dom_odds(json.loads(dom_odds_json))
                    leg_json = str(evaluate(
                        "JSON.stringify(Array.from(document.querySelectorAll('[id^=\\\"runnerNo_\\\"]'))"
                        ".map(e => { const s=e.id.replace('runnerNo_',''); return {"
                        "id:e.id,runner_no:e.innerText.trim(),"
                        "horse_name:document.getElementById('horseName_'+s)?.innerText.trim()||'',"
                        "win_odds:document.getElementById('odds_WIN_'+s)?.innerText.trim()||''}; }))"
                    ) or "[]")
                    update_match = re.search(r"Last Update:\s*([^\n]+)", text)
                    pages[pool] = {
                        "text": text,
                        "top_combinations": parse_top_combinations(top_text, pool),
                        "normalized_combinations": dom_odds,
                        "legs": parse_leg_runners(json.loads(leg_json)),
                        "updated_at_display": update_match.group(1).strip() if update_match else None,
                        "availability": (
                            "unavailable" if "not available" in text.lower()
                            else "not_started" if "not yet started" in text.lower()
                            else "selling"
                        ),
                    }
                    if pool == "FCT":
                        pages[pool]["matrix_combinations"] = parse_rectangular_matrix(
                            text, "1st\u00a0Horse", "Forecast Method"
                        )
                    elif pool == "DBL":
                        pages[pool]["matrix_combinations"] = parse_rectangular_matrix(
                            text, "1st\u00a0Leg", "Guide to Odds Change"
                        )
                return initial_html, pages
            finally:
                if websocket is not None:
                    websocket.close()
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
