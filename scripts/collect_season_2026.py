"""Independent, bounded acquisition of official 2026/27 HKJC season records.

Only this script's output directory is written. Cached response bytes and their
request metadata are retained for deterministic offline replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit

import requests
from scrapy.http import HtmlResponse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrapper.official_corpus import DATE_LIST, HORSE_ID, WITHDRAWN, extract_tables, parse_document

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/race-readiness-20261007/official"
BASE = "https://racing.hkjc.com"
START, END, CARD_DATE = "2026-09-01", "2026-10-06", "2026-10-07"
POOLS = ("WIN", "PLACE", "QIN", "QPL", "TRIO", "TIERCE", "FIRST4", "QUARTET")
POOL_NAMES = {"WIN": "WIN", "PLACE": "PLACE", "QUINELLA": "QIN",
              "QUINELLA PLACE": "QPL", "TRIO": "TRIO", "TIERCE": "TIERCE",
              "FIRST 4": "FIRST4", "FIRST4": "FIRST4", "QUARTET": "QUARTET"}
POOL_SIZE = dict(WIN=1, PLACE=1, QIN=2, QPL=2, TRIO=3, TIERCE=3, FIRST4=4, QUARTET=4)
ORDERED = {"TIERCE", "QUARTET"}
GOING_NAMES = {"FIRM", "GOOD TO FIRM", "GOOD", "GOOD TO YIELDING", "YIELDING",
               "YIELDING TO SOFT", "SOFT", "HEAVY", "FAST", "WET FAST", "GOOD TO SLOW",
               "SLOW", "WET SLOW", "SEALED"}
VERSION = "season-2026-v1"
UNIT_RULES = "https://www.hkjc.com/english/betting/template_betting_rule_files/pdf/7-Horse_Race_Illustration_Eng_Jan23.pdf"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean(selector):
    return " ".join(" ".join(selector.xpath(".//text()").getall()).split())


def visible(response):
    return "\n".join(" ".join(t.split()) for t in response.xpath(
        "//body//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::select)]"
    ).getall() if t.strip())


def official_url(url):
    parts = urlsplit(url)
    if (parts.scheme != "https" or (parts.hostname != "racing.hkjc.com" and url != UNIT_RULES)
            or parts.username or parts.password or parts.port not in (None, 443)):
        raise ValueError(f"Outside official public racing host: {url}")
    return url


def race_url(kind, day, venue, number, legacy=False, archive=False):
    route = "LocalResults" if kind == "results" else "RaceCard"
    path = (f"/racing/information/English/Racing/{route}.aspx" if legacy else
            f"/en-us/local/information/{'localresults' if kind == 'results' else 'racecard'}")
    if archive and not legacy and kind == "results":
        path = "/en-us/local/information/archive/localresults"
    return BASE + path + "?" + urlencode({"RaceDate": day.replace("-", "/"),
                                         "Racecourse": venue, "RaceNo": number})


class Fetcher:
    """One sequential request stream, zero retries; cache also records failures."""

    def __init__(self, output=OUTPUT, max_requests=150, delay=0.6, timeout=20, offline=False):
        self.output = Path(output)
        self.max_requests, self.delay, self.timeout = max_requests, delay, timeout
        self.offline, self.requests_made, self.last_request = offline, 0, 0.0
        self.deadline = None
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "iMa-official-season-research/1.0"
        self.sources = {}

    def fetch(self, url):
        official_url(url)
        token = hashlib.sha256(url.encode()).hexdigest()
        meta_path = self.output / "raw" / (token + ".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["source_url"] != url:
                raise ValueError("Cache URL mismatch")
            body = (self.output / meta["raw_path"]).read_bytes() if meta.get("raw_path") else b""
            if meta.get("sha256") != hashlib.sha256(body).hexdigest():
                raise ValueError("Cache content hash mismatch")
        else:
            if self.offline:
                raise ValueError(f"Offline cache miss: {url}")
            if self.requests_made >= self.max_requests:
                raise ValueError("Request budget exhausted")
            pause = max(0, self.delay - (time.monotonic() - self.last_request))
            if self.deadline is not None and time.monotonic() + pause >= self.deadline:
                raise ValueError("Acquisition wall-clock deadline reached")
            time.sleep(pause)
            self.requests_made += 1
            self.last_request = time.monotonic()
            meta = {"source_url": url, "retrieved_at": now(), "http_status": None,
                    "final_url": url, "redirects": [], "error": None}
            body = b""
            try:
                current = url
                for _ in range(4):
                    request_timeout = self.timeout
                    if self.deadline is not None:
                        remaining = self.deadline - time.monotonic()
                        if remaining <= 0:
                            raise ValueError("Acquisition wall-clock deadline reached")
                        request_timeout = min(request_timeout, remaining)
                    reply = self.session.get(current, timeout=request_timeout, allow_redirects=False)
                    meta.update(http_status=reply.status_code, final_url=reply.url,
                                content_type=reply.headers.get("Content-Type", ""))
                    if reply.is_redirect:
                        next_url = official_url(urljoin(current, reply.headers["Location"]))
                        meta["redirects"].append({"url": current, "status": reply.status_code,
                                                  "location": next_url})
                        if self.requests_made >= self.max_requests:
                            raise ValueError("Request budget exhausted during redirect")
                        if self.deadline is not None and time.monotonic() + self.delay >= self.deadline:
                            raise ValueError("Acquisition wall-clock deadline reached during redirect")
                        time.sleep(self.delay)
                        self.requests_made += 1
                        self.last_request = time.monotonic()
                        current = next_url
                        continue
                    body = reply.content
                    break
                else:
                    raise ValueError("Too many redirects")
            except (requests.RequestException, ValueError, KeyError) as exc:
                meta["error"] = f"{type(exc).__name__}: {exc}"
            meta["sha256"] = hashlib.sha256(body).hexdigest()
            meta["byte_count"] = len(body)
            suffix = ".pdf" if body.startswith(b"%PDF-") else ".html"
            meta["raw_path"] = "raw/" + token + "-" + meta["sha256"] + suffix
            raw_path = self.output / meta["raw_path"]
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(body)
            write_json(meta_path, meta)
        self.sources[url] = meta
        if meta["error"] or meta["http_status"] != 200:
            raise ValueError(f"Official access failure: {meta['http_status']} {url}: {meta['error']}")
        response = HtmlResponse(url=meta["final_url"], body=body, encoding="utf-8")
        if not body.startswith(b"%PDF-") and re.search(r"access denied|request rejected|verify you are human|captcha", visible(response), re.I):
            raise ValueError(f"Official access challenge: {url}; raw={meta['raw_path']}")
        return response, meta


def parse_fixture(response, year, month):
    text = visible(response)
    if not re.search(rf"\b{month}\s*/\s*{year}\b", text):
        raise ValueError("Displayed fixture month/year mismatch")
    meetings = []
    for cell in response.css("td.calendar"):
        venues = cell.xpath('.//img[@alt="ST" or @alt="HV"]/@alt').getall()
        if not venues:
            continue
        day = cell.css("span.f_fs14::text").get("").strip()
        if not day.isdigit() or len(venues) != 1:
            raise ValueError("Ambiguous fixture day or local venue")
        race_day = date(year, month, int(day)).isoformat()
        count = sum(int(n) for n in re.findall(r"\b\d{3,4}\((\d+)\)", clean(cell)))
        meetings.append({"race_date": race_day, "venue": venues[0],
                         "scheduled_race_count": count or None, "fixture_text": clean(cell),
                         "status": "scheduled"})
    cancellations = []
    for match in re.finditer(r"race meeting originally scheduled for\s+(?:\w+,?\s+)?"
                            r"(\d{1,2}\s+\w+\s+\d{4})\s+at\s+(Sha Tin|Happy Valley)"
                            r"\s+Racecourse has been cancelled", text, re.I):
        cancellations.append({"race_date": datetime.strptime(match[1], "%d %B %Y").date().isoformat(),
                              "venue": "ST" if match[2].lower() == "sha tin" else "HV",
                              "status": "cancelled", "evidence": match[0]})
    for meeting in meetings:
        if any((m["race_date"], m["venue"]) == (meeting["race_date"], meeting["venue"])
               for m in cancellations):
            meeting["status"] = "cancelled"
    if not meetings:
        raise ValueError("No official local fixtures found")
    return {"meetings": meetings, "cancellations": cancellations}


def race_numbers(response, day, venue, kind, current):
    numbers = {current}
    for href in response.css("a::attr(href)").getall():
        target = urlsplit(urljoin(response.url, href))
        route = target.path.lower().rsplit("/", 1)[-1].removesuffix(".aspx")
        if route != ("localresults" if kind == "results" else "racecard"):
            continue
        query = {k.lower(): v for k, v in parse_qsl(target.query)}
        if (query.get("racedate", "").replace("/", "-") == day
                and query.get("racecourse", "").upper() == venue
                and query.get("raceno", "").isdigit()):
            numbers.add(int(query["raceno"]))
    if max(numbers) > 12 or sorted(numbers) != list(range(1, max(numbers) + 1)):
        raise ValueError("Noncontiguous or excessive official race navigation")
    return sorted(numbers)


def expanded_rows(table):
    """Expand HTML rowspans/colspans before reading dividend continuations."""
    pending = {}
    for index, row in enumerate(table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")):
        grid, col = {}, 0
        for column, (remaining, value) in list(pending.items()):
            grid[column] = value
            if remaining <= 1:
                del pending[column]
            else:
                pending[column] = (remaining - 1, value)
        original = []
        for cell in row.xpath("./th|./td"):
            value = clean(cell)
            original.append(value)
            while col in grid:
                col += 1
            span, height = int(cell.attrib.get("colspan", 1)), int(cell.attrib.get("rowspan", 1))
            if not 1 <= span <= 30 or not 1 <= height <= 100:
                raise ValueError("Excessive dividend table spans")
            for offset in range(span):
                grid[col + offset] = value
                if height > 1:
                    pending[col + offset] = (height - 1, value)
            col += span
        yield index, [grid.get(i, "") for i in range(max(grid, default=-1) + 1)], original


def dividend_value(raw):
    value = raw.strip()
    # Large flexi dividends sometimes explicitly publish a smaller stake unit.
    value = re.sub(r"\s*/\s*(?:HK)?\$\s*\d+(?:\.\d+)?$", "", value)
    if re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        number = Decimal(value.replace(",", ""))
        return "payable", str(number), float(number)
    if re.search(r"refund", value, re.I):
        return "refund", None, None
    if re.search(r"not\s*(?:won|payable)|no\s*(?:winning|winner)|unclaimed", value, re.I):
        return "nonpayable", None, None
    if value.upper() in {"", "-", "--", "---", "N/A", "NOT AVAILABLE"}:
        return "unavailable", None, None
    return "unparsed", None, None


def parse_dividends(response):
    pools = {p: {"status": "unavailable", "ordered": p in ORDERED,
                  "records": [], "reason": "pool_not_published_in_dividend_table"} for p in POOLS}
    evidence = []
    for table_index, table in enumerate(response.css("table")):
        rows = list(expanded_rows(table))
        header = next((i for i, row, _ in rows if row[:3] ==
                       ["Pool", "Winning Combination", "Dividend (HK$)"]), None)
        if header is None:
            continue
        for row_index, cells, original in rows:
            if row_index <= header or len(cells) != 3:
                continue
            label, combination, raw = cells
            pool = POOL_NAMES.get(label.upper())
            evidence.append({"table_index": table_index, "row_index": row_index,
                             "cells": original, "expanded_cells": cells})
            if not pool:
                continue
            status, decimal, number = dividend_value(raw)
            unit_match = re.search(r"/\s*(?:HK)?\$\s*(\d+(?:\.\d+)?)$", raw)
            unit = Decimal(unit_match[1]) if unit_match else Decimal("10")
            if unit <= 0:
                status, decimal, number = "unparsed", None, None
            horses = ([int(n) for n in combination.split(",")]
                      if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", combination) else [])
            if status == "payable" and (len(horses) != POOL_SIZE[pool]
                    or len(set(horses)) != len(horses) or any(n < 1 or n > 99 for n in horses)):
                status = "unparsed"
            if status == "unparsed":
                decimal = number = None
            record = {"winning_combination": combination, "combination": horses,
                      "combination_key": ",".join(map(str, horses if pool in ORDERED else sorted(horses))),
                      "dividend_hkd": number, "dividend_decimal": decimal,
                      "dividend_raw": raw, "status": status, "currency": "HKD",
                      "unit_stake_hkd": float(unit) if unit > 0 else None,
                      "unit_stake_decimal": str(unit) if unit > 0 else None,
                      "unit_provenance": "explicit_source_suffix" if unit_match else "HKJC_standard_unit",
                      "unit_rule_source_url": UNIT_RULES,
                      "dividend_per_hkd_10_decimal": str(Decimal(decimal) * Decimal("10") / unit) if decimal else None,
                      "table_index": table_index, "row_index": row_index}
            pools[pool]["records"].append(record)
    for pool in pools.values():
        if pool["records"]:
            statuses = {r["status"] for r in pool["records"]}
            pool["status"] = next(iter(statuses)) if len(statuses) == 1 else "mixed"
            pool["reason"] = None
    return pools, evidence


def parse_result(response, day, venue, number):
    # Verify against the requested context as well as the existing parser's final URL.
    from scrapper.official_corpus import verify_race_context
    verify_race_context(response, {"racedate": day, "racecourse": venue, "raceno": str(number)})
    parsed = parse_document(response)
    if parsed["status"] != "fetched_parsed" or not parsed.get("race", {}).get("runners"):
        raise ValueError(parsed.get("parse_error", "Missing parsed result runners"))
    record = parsed["race"]
    runner_rows = []
    for table in response.css("table"):
        rows = table.xpath("./tr|./thead/tr|./tbody/tr")
        if not rows or [clean(c) for c in rows[0].xpath("./th|./td")][:2] != ["Pla.", "Horse No."]:
            continue
        for row in rows[1:]:
            cells = [clean(c) for c in row.xpath("./td")]
            if len(cells) >= 11 and cells[1].isdigit():
                runner_rows.append(cells)
    by_no = {r["horse_no"]: r for r in record["runners"]}
    unparsed = []
    for cells in runner_rows:
        runner = by_no.get(int(cells[1]))
        if runner:
            runner.update(place_raw=cells[0], dead_heat="DH" in cells[0].upper(), source_cells=cells)
        else:
            unparsed.append({"horse_no": int(cells[1]), "finishing_status": cells[0], "source_cells": cells})
    if len(runner_rows) != len(by_no) + len(unparsed):
        raise ValueError("Duplicate result runner rows")
    record.update(kind="results", race_id=f"{day}_{venue}_R{number}",
                  unparsed_or_withdrawn_runners=unparsed, result_rows=runner_rows,
                  dead_heat=any(r["dead_heat"] for r in by_no.values()))
    record["dividends"], record["dividend_rows"] = parse_dividends(response)
    record["published_at"] = None
    record["availability_note"] = "Historical results captured now; publication time is not inferred."
    return record


def parse_card(response, day, venue, number):
    text = visible(response)
    contexts = re.findall(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*"
                          r"(\w+ \d{1,2}, \d{4}),\s*(Happy Valley|Sha Tin),\s*(\d{1,2}:\d{2})", text)
    if len(set(contexts)) != 1:
        raise ValueError("Missing or ambiguous displayed racecard date/venue")
    displayed, location, off_time = contexts[0]
    if (datetime.strptime(displayed, "%B %d, %Y").date().isoformat() != day
            or ("HV" if location == "Happy Valley" else "ST") != venue):
        raise ValueError("Displayed racecard date/venue mismatch")
    titles = re.findall(r"^Race (\d+)\s*-\s*(.+)$", text, re.M)
    if len(titles) != 1 or int(titles[0][0]) != number:
        raise ValueError("Displayed racecard number mismatch")
    runners, headers, raw_rows = [], [], []
    for table in response.css("table"):
        rows = table.xpath("./tr|./thead/tr|./tbody/tr")
        if not rows:
            continue
        labels = [clean(c) for c in rows[0].xpath("./th|./td")]
        if labels[:2] != ["Horse No.", "Last 6 Runs"] or "Horse" not in labels or "Brand No." not in labels:
            continue
        if headers:
            raise ValueError("Ambiguous starter tables")
        headers = labels
        for row in rows[1:]:
            cells = [clean(c) for c in row.xpath("./td")]
            if not cells or not cells[0].isdigit():
                continue
            if len(cells) != len(headers):
                raise ValueError("Racecard column count mismatch")
            values = dict(zip(headers, cells))
            ids = set(HORSE_ID.findall(row.get()))
            if len(ids) != 1:
                raise ValueError("Missing or ambiguous full racecard horse ID")
            identity = ids.pop().upper()
            if identity.rsplit("_", 1)[-1] != values["Brand No."]:
                raise ValueError("Racecard brand/full horse ID mismatch")
            def numeric(label):
                try:
                    return float(Decimal(values.get(label, "" ).replace(",", "")))
                except InvalidOperation:
                    return None
            runners.append({"horse_no": int(cells[0]), "horse_name": values["Horse"],
                            "horse_page_id": identity, "horse_code": values["Brand No."],
                            "jockey": re.sub(r"\s*\(-\d+\)\s*$", "", values.get("Jockey", "")),
                            "jockey_raw": values.get("Jockey"), "trainer": values.get("Trainer"),
                            "draw": numeric("Draw"), "weight": numeric("Wt."),
                            "rating": numeric("Rtg."), "declared_weight": numeric("Horse Wt. (Declaration)"),
                            "age": numeric("Age"), "over_weight": numeric("Over Wt."),
                            "apprentice_allowance_lbs": (int(m[1]) if (m := re.search(r"\(-\s*(\d+)\)", values.get("Jockey", ""))) else None),
                            "net_carried_weight_lbs": (numeric("Wt.") - (int(m[1]) if (m := re.search(r"\(-\s*(\d+)\)", values.get("Jockey", ""))) else 0) + (numeric("Over Wt.") or 0)) if numeric("Wt.") is not None and (not values.get("Over Wt.") or numeric("Over Wt.") is not None) else None,
                            "weight_note": "Declared Wt. minus published jockey claim plus published probable overweight; blank claim/overweight means none declared. Not post-race actual weight.",
                            "gear": values.get("Gear"), "last_six_runs": values.get("Last 6 Runs"),
                            "source_values": values})
            raw_rows.append(cells)
    if not runners or len({r["horse_no"] for r in runners}) != len(runners):
        raise ValueError("No starters or duplicate horse numbers")
    distance = re.search(r"\b(\d{3,4})\s*(?:Metres|M)\b", text, re.I)
    race_class = re.search(r"\bClass\s+[1-5]\b", text, re.I)
    conditions = re.search(r"^(Turf|All Weather Track),\s*(?:(.*?)\s*,\s*)?(\d{3,4})M(?:,\s*(.+))?$", text, re.M | re.I)
    prize_match = re.search(r"Prize Money:\s*((?:HK)?\$)\s*([\d,]+(?:\.\d+)?)", text)
    surface = conditions[1].upper() if conditions else None
    course_configuration = conditions[2] if conditions else None
    going_raw = conditions[4] if conditions else None
    going = going_raw.upper() if going_raw and going_raw.upper() in GOING_NAMES else going_raw
    return {"kind": "racecard", "race_id": f"{day}_{venue}_R{number}",
            "race_date": day, "venue": venue, "race_no": number, "race_name": titles[0][1],
            "scheduled_time_local": off_time, "timezone": "Asia/Hong_Kong",
            "distance": int(distance[1]) if distance else None,
            "surface": surface, "course_configuration": course_configuration,
            "course": (surface + (" - " + course_configuration if course_configuration else "")) if surface else None,
            "going": going, "going_raw": going_raw,
            "conditions_raw": conditions[0] if conditions else None,
            "prize": float(Decimal(prize_match[2].replace(",", ""))) if prize_match else None,
            "prize_decimal": str(Decimal(prize_match[2].replace(",", ""))) if prize_match else None,
            "prize_currency": ("HKD" if prize_match and prize_match[1] == "HK$" else None),
            "prize_currency_symbol": prize_match[1] if prize_match else None,
            "prize_currency_note": "Plain $ symbol retained; currency code only emitted for explicit HK$.",
            "race_class": race_class[0] if race_class else None, "runners": runners,
            "starter_headers": headers, "starter_rows": raw_rows,
            "tables": extract_tables(response), "published_at": None}


def collect_race(fetcher, kind, day, venue, number, archive=False):
    attempts = []
    parser = parse_result if kind == "results" else parse_card
    for legacy in (False, True):
        url = race_url(kind, day, venue, number, legacy)
        if archive and not legacy and day < "2026-07-05":
            token = hashlib.sha256(url.encode()).hexdigest()
            if not (fetcher.output / "raw" / (token + ".json")).exists():
                url = race_url(kind, day, venue, number, archive=True)
        try:
            response, source = fetcher.fetch(url)
            record = parser(response, day, venue, number)
            record.update(schema_version=VERSION, source=source, acquisition_attempts=attempts)
            return record, race_numbers(response, day, venue, kind, number)
        except ValueError as exc:
            attempt = {"source_url": url, "error": str(exc)}
            attempts.append(attempt)
            print(json.dumps({"event": "source_failure", **attempt}), flush=True)
    raise ValueError(json.dumps(attempts))


def coverage_manifest(meetings, races, cards, errors, census, fixtures, sources):
    expected = {(m["race_date"], m["venue"], n) for m in meetings
                if m["status"] != "cancelled" for n in m.get("race_numbers", [])}
    actual = {(r["race_date"], r["venue"], r["race_no"]) for r in races}
    missing = sorted(expected - actual)
    unresolved = [m for m in meetings if m["status"] != "cancelled" and not m.get("race_numbers")]
    pool_gaps, pool_states = [], {p: Counter() for p in POOLS}
    for race in races:
        for pool, data in race["dividends"].items():
            pool_states[pool][data["status"]] += 1
            if data["status"] in {"unavailable", "unparsed"} or any(
                    r["status"] in {"unavailable", "unparsed"} for r in data["records"]):
                pool_gaps.append({"race_id": race["race_id"], "pool": pool,
                                  "status": data["status"], "reason": data["reason"]})
    cards_complete = {(c["race_date"], c["venue"], c["race_no"]) for c in cards} == {
        (CARD_DATE, "HV", n) for n in range(1, 10)}
    fixture_ok = len(fixtures) == 2 and bool(census)
    result_complete = bool(meetings) and fixture_ok and not unresolved and not missing and expected == actual
    parse_gaps = [g for g in pool_gaps if g["status"] != "unavailable"]
    has_dividend_tables = all(r["dividend_rows"] for r in races)
    known = {m["race_date"] for m in meetings}
    return {"schema_version": VERSION, "generated_at": now(),
            "scope": {"results_start": START, "results_end": END,
                      "racecard_date": CARD_DATE, "racecard_venue": "HV", "expected_racecards": 9},
            "complete": result_complete and cards_complete and not parse_gaps and has_dividend_tables and not errors,
            "results_complete": result_complete, "racecards_complete": cards_complete,
            "dividends_complete": result_complete and not parse_gaps and has_dividend_tables,
            "all_requested_pools_published": result_complete and not pool_gaps,
            "dividend_completeness_definition": "Every published requested-pool row parsed, with explicit unavailable state for absent pools; does not imply every pool was offered or payable.",
            "meeting_count": sum(m["status"] == "completed" for m in meetings),
            "expected_result_races": len(expected), "collected_result_races": len(races),
            "result_runners": sum(len(r["runners"]) for r in races),
            "collected_racecards": len(cards), "racecard_runners": sum(len(c["runners"]) for c in cards),
            "dividend_records": sum(len(p["records"]) for r in races for p in r["dividends"].values()),
            "pool_status_counts": {p: dict(v) for p, v in pool_states.items()},
            "missing_result_races": [list(x) for x in missing], "unresolved_meetings": unresolved,
            "pool_gaps": pool_gaps, "errors": errors,
            "cancellations": [m for m in meetings if m["status"] == "cancelled"],
            "date_list_candidates": census,
            "date_list_exclusions": [{**c, "reason": "not_in_official_local_fixture_calendar; venue_unresolved; not assumed local"}
                                     for c in census if c["date"] not in known],
            "denominator_evidence": "Both monthly official local fixtures, cancellation notices, and contiguous race navigation on verified result pages",
            "fixture_sources": fixtures, "sources": list(sources.values()),
            "publication_time_policy": "retrieved_at is capture time; published_at remains null",
            "payment_policy": "Only officially published winning combinations are listed; absent combinations are not synthesized as zero or nonpayable.",
            "files": {"races": "races.json", "racecards": "racecards.json", "meetings": "meetings.json",
                      "schema": "schema.json", "raw": "raw/"}}


def api_schema():
    return {"schema_version": VERSION, "encoding": "UTF-8 JSON", "races.json": "Array of result race objects",
            "racecards.json": "Array of pre-meeting racecard objects", "identity": {
                "race_id": "YYYY-MM-DD_VENUE_RN", "race_date": "ISO date", "venue": "ST or HV", "race_no": "integer"},
            "race": {"runners": "Full field from official result table; nonfinishers retained; raw statuses preserved",
                     "unparsed_or_withdrawn_runners": "Additional raw result rows not recognized by upstream parser",
                     "dead_heat": "true when a displayed placing contains DH",
                     "dividends": "Object keyed WIN PLACE QIN QPL TRIO TIERCE FIRST4 QUARTET",
                     "dividend_rows": "All dividend rows, including unrequested pools, with DOM table/row indexes",
                     "source": "source_url, final_url, retrieved_at (UTC), sha256 of exact bytes, raw_path, HTTP status"},
            "pool": {"status": "payable | refund | nonpayable | unavailable | unparsed | mixed",
                     "ordered": "true for TIERCE and QUARTET", "records": "Array; preserves multiple/dead-heat winning combinations",
                     "reason": "Explicit absence reason, or null"},
            "dividend_record": {"combination": "Array of horse numbers in published order",
                                "combination_key": "Comma joined; sorted only for unordered pools",
                                "winning_combination": "Exact normalized published combination text",
                                "dividend_hkd": "Published numeric HKD amount or null; never inferred from odds",
                                "dividend_decimal": "Exact decimal string without thousands separators or null",
                                "dividend_raw": "Published text including refund/nonpayable markers",
                                "unit_stake_hkd": "Explicit /$N suffix, else HKJC standard HK$10 unit (see rule source)",
                                "unit_provenance": "explicit_source_suffix | HKJC_standard_unit",
                                "dividend_per_hkd_10_decimal": "Exact dividend normalized to HK$10; null for nonnumeric outcomes",
                                "status": "payable | refund | nonpayable | unavailable | unparsed",
                                "currency": "HKD", "table_index": "Zero based DOM table index",
                                "row_index": "Zero based row in table"},
            "unit_policy": "dividend_hkd/decimal retain published amounts; unit_stake_hkd tracks explicit smaller units; dividend_per_hkd_10_decimal normalizes separately",
            "unit_rule_source_url": UNIT_RULES,
            "unit_rule_cached_evidence": "dividend_units.json contains the official PDF URL, raw bytes hash/path and extracted unit evidence",
            "racecard": "surface, course_configuration, course, going (recognized names uppercased), going_raw, conditions_raw, prize, prize_decimal, prize_currency (null for plain $), prize_currency_symbol; missing source fields remain null",
            "racecard_runner": "horse_no, horse_name, horse_page_id, horse_code, jockey (terminal (-n) allowance removed only), jockey_raw, trainer, draw, weight, rating, age, declared_weight, over_weight, apprentice_allowance_lbs (null if absent), net_carried_weight_lbs (declared Wt minus published claim plus probable overweight; not post-race actual), gear, last_six_runs, source_values (all published columns)",
            "season-runner-enrichment.json": "Flat exact-date PDF declaration rows; race_id, race_date, date, venue, race_no, horse_no, horse_page_id, horse_id, horse_code, horse_name, rating, horse_rating, gear, horse_age, age, horse_country, country, published_at (PDF Data as at or null), source, pdf_page, pdf_row_raw, pdf_identity_raw",
            "calibration-runner-enrichment.json": "Same flat schema as season enrichment, separately scoped by calibration-race-census.csv; coverage manifest must be checked before use",
            "coverage_manifest.json": "Completeness booleans, expected/actual counts, explicit missing races/pools, source inventory, cancellations and excluded candidate dates",
            "horse_histories.json": "Optional array by horse_page_id: form_records include ISO race_date, result link/identity, original form fields and source_cells",
            "history_coverage_manifest.json": "Separate horse-page, prior-start and baseline coverage; never equates season collection with full calendar-year history",
            "history-races.json": "Optional Jan-Jul 2026 missing full-race fields; same result schema as races.json",
            "history-race-coverage.json": "Seven monthly fixtures and race-navigation denominator, read-only baseline hash, new/missing race IDs, request/time accounting",
            "data-freeze.json": "Final immutable-input file hashes and freeze ID, written after coverage and source replay validation",
            "replay": "python scripts/collect_season_2026.py --offline",
            "scope": "No remote writes, services, authentication, anti-bot evasion, or betting"}


def reparse_cards(output=OUTPUT):
    """Refresh only card normalization from captured bytes; zero network requests."""
    output = Path(output)
    previous = json.loads((output / "racecards.json").read_text())
    cards, fetcher = [], Fetcher(output, offline=True)
    for old in previous:
        response, source = fetcher.fetch(old["source"]["source_url"])
        card = parse_card(response, old["race_date"], old["venue"], old["race_no"])
        card.update(schema_version=VERSION, source=source, acquisition_attempts=old.get("acquisition_attempts", []))
        cards.append(card)
    if {(c["race_date"], c["venue"], c["race_no"]) for c in cards} != {(CARD_DATE, "HV", n) for n in range(1, 10)}:
        raise ValueError("Cannot publish a partial card replay")
    write_json(output / "racecards.json", cards)
    write_json(output / "schema.json", api_schema())
    fetcher.session.close()
    return {"racecards": len(cards), "runners": sum(len(c["runners"]) for c in cards), "requests": 0}


def parse_pdf_card_entries(pages, day):
    """Recognize dated declaration rows in the official bilingual form guide."""
    entries = []
    for page_no, text in enumerate(pages, 1):
        lines = text.splitlines()
        header = re.search(r"\bRACE\s+(\d{1,2})\b", "\n".join(lines[:4]), re.I)
        primary = re.match(r"B\d+\s*\n\s*(\d{1,2})\s*\n", text)
        page_race_no = int(header[1]) if header else int(primary[1]) if primary else None
        for index, line in enumerate(lines):
            row = re.fullmatch(r"\s*(\d+)\s+(\d{1,2}/\d{1,2}/\d{2})\s+(.+?)\s+"
                               r"(ST|HV|AWT)\s+(.+?)\s+(\d{3,4})\s+(#?\d{2,3})\s+"
                               r"(?:\((\d+)\)\s*)?(.*?)\s*", line)
            if not row or datetime.strptime(row[2], "%d/%m/%y").date().isoformat() != day:
                continue
            following = [line for line in lines[index + 1:index + 12] if line.strip()]
            if len(following) < 2:
                continue
            brand = re.search(r"\b([A-Z])0*(\d{1,3})\s*$", following[0])
            name = re.match(r"(.+?)\s*\(([A-Z]{2,4})\)\s+(\d+)\s", following[1])
            if not brand or not name:
                continue
            jockey_draw = re.fullmatch(r"(-?\d*[A-Za-z][A-Za-z ]*?)\s+(\d(?:\s*\d)?)\s*(.*)", row[9])
            gear = jockey_draw[3].strip() if jockey_draw else row[9].strip()
            if gear and not re.fullmatch(r"[A-Z0-9/+-]+", gear):
                continue
            horse_no = None
            for offset in range(index + 3, min(index + 45, len(lines) - 1)):
                footer = re.match(r"\s*([A-Z])0*(\d{1,3})(?:\s.*)?$", lines[offset])
                if footer and footer[1] == brand[1] and int(footer[2]) == int(brand[2]):
                    remaining = [v.strip() for v in lines[offset + 1:offset + 4] if v.strip()]
                    number = re.fullmatch(r"(\d{1,2})(?:\s+[F#%]+)?", remaining[0]) if remaining else None
                    if number:
                        horse_no = int(number[1])
                        break
            if horse_no is None:
                footer_numbers = []
                for block_line in lines[index + 3:index + 45]:
                    if re.match(r"\s*(?:\d+/\d+\s+\*?\d|W\([vx]\)|\d+\s+\d{1,2}/\d{1,2}/)", block_line, re.I):
                        break
                    number = re.fullmatch(r"\s*(\d{1,2})(?:\s+[F#%]+)?\s*", block_line)
                    if number:
                        footer_numbers.append(int(number[1]))
                if len(footer_numbers) == 1:
                    horse_no = footer_numbers[0]
            entries.append({"official_race_index": int(row[1]), "race_date": day,
                "pdf_race_no": page_race_no, "pdf_horse_no": horse_no,
                "venue": "ST" if row[4] == "AWT" else row[4],
                "distance": int(row[6]), "horse_code": brand[1] + brand[2].zfill(3),
                "horse_name": name[1].strip(), "horse_country": name[2], "horse_age": int(name[3]),
                "country": name[2], "age": int(name[3]), "horse_rating": int(row[8]) if row[8] else None, "rating": int(row[8]) if row[8] else None,
                "rating_status": "declared" if row[8] else "source_blank",
                "gear": gear, "gear_status": "declared" if gear else "source_blank",
                "card_weight_raw": row[7], "jockey_code_raw": jockey_draw[1] if jockey_draw else None,
                "draw": int(jockey_draw[2].replace(" ", "")) if jockey_draw else None,
                "pdf_page": page_no, "pdf_line_index": index, "pdf_row_raw": line,
                "pdf_identity_raw": following[:2]})
    return entries


def enrich_season(output=OUTPUT, max_requests=16, delay=1, timeout=20, offline=False,
                  calibration=False, max_seconds=2700):
    from io import BytesIO
    from pypdf import PdfReader
    output = Path(output)
    races = json.loads((output / "races.json").read_text())
    scope = "calibration" if calibration else "season"
    if calibration:
        import pandas as pd
        census = pd.read_csv(ROOT / "artifacts/race-readiness-20261007/calibration-race-census.csv")
        baseline = pd.read_parquet(ROOT / ".tmp/season-official-history.parquet")
        recovered = json.loads((output / "history-races.json").read_text())
        recovered_by_key = {(r["race_date"], r["venue"], r["race_no"]): r for r in recovered}
        races = []
        for target in census.itertuples():
            key = (target.date, target.venue, int(target.race_no))
            if key in recovered_by_key:
                race = dict(recovered_by_key[key])
            else:
                rows = baseline[(baseline.date == target.date) & (baseline.venue == target.venue) & (baseline.race_no == target.race_no)]
                if rows.empty:
                    raise ValueError(f"No full result identity rows for calibration race {key}")
                race = {"race_date": target.date, "venue": target.venue, "race_no": int(target.race_no),
                        "distance": int(rows.iloc[0].distance), "runners": [
                            {"horse_page_id": r.horse_id, "horse_code": r.horse_id.rsplit("_", 1)[-1],
                             "horse_no": int(r.horse_no), "horse_name": r.horse_name} for r in rows.itertuples()]}
            race["race_id"] = target.race_id
            races.append(race)
    fetcher = Fetcher(output, max_requests, delay, timeout, offline)
    fetcher.deadline = time.monotonic() + max_seconds
    records, gaps, errors = [], [], []
    meetings = sorted({(r["race_date"], r["venue"]) for r in races})
    completed = 0
    attempted = 0
    def checkpoint():
        captured_ids = {r["race_id"] for r in records}
        missing_ids = [r["race_id"] for r in races if sum(x["race_id"] == r["race_id"] for x in records) != len(r["runners"])]
        rated_complete = [r["race_id"] for r in races if r["race_id"] not in missing_ids
                          and all(x["horse_rating"] is not None for x in records if x["race_id"] == r["race_id"])]
        manifest = {"schema_version": VERSION, "generated_at": now(),
            "source_kind": "official archived dated meeting form-guide PDF",
            "expected_meetings": len(meetings), "captured_meetings": completed,
            "attempted_meetings": attempted, "acquisition_finished": attempted == len(meetings),
            "expected_races": len(races), "expected_result_runners": sum(len(r["runners"]) for r in races),
            "captured_races": len(captured_ids), "missing_or_partial_races": missing_ids,
            "whole_field_finite_rating_race_ids": rated_complete,
            "whole_field_finite_rating_races": len(rated_complete),
            "source_blank_ratings": sum(r["horse_rating"] is None for r in records),
            "enriched_runners": len(records), "complete": completed == len(meetings) and not gaps and not errors,
            "gaps": gaps, "errors": errors, "sources": list(fetcher.sources.values()),
            "requests_made_this_run": fetcher.requests_made, "request_delay_seconds": delay,
            "join_keys": ["race_date", "venue", "race_no", "horse_page_id"],
            "verification": "Exact PDF date, venue, distance, unique horse code and normalized name; result official index for season, printed page race number and horse footer number for calibration",
            "publication_policy": "Archived declaration row for that meeting; capture time is not the original publication timestamp",
            "scope": scope, "request_budget": max_requests, "max_seconds": max_seconds,
            "result_mutation": False, "output": f"{scope}-runner-enrichment.json"}
        write_json(output / f"{scope}-runner-enrichment.json", records)
        write_json(output / f"{scope}-enrichment-coverage.json", manifest)
        return manifest
    def name_key(value):
        return " ".join(value.upper().replace(chr(0x2019), "'").split())
    for day, venue in meetings:
        attempted += 1
        url = BASE + "/racing/content/PDF/RaceCard/" + day.replace("-", "") + "_starter_all.pdf"
        try:
            _, source = fetcher.fetch(url)
            body = (output / source["raw_path"]).read_bytes()
            if not body.startswith(b"%PDF-"):
                raise ValueError("Archived form guide is not PDF")
            text_path = output / f"formguide_text_{day}.json"
            cached_text = json.loads(text_path.read_text()) if text_path.exists() else None
            if cached_text and cached_text["source"]["sha256"] == source["sha256"]:
                pages = cached_text["pages"]
            else:
                reader = PdfReader(BytesIO(body))
                if reader.is_encrypted or not 1 <= len(reader.pages) <= 150:
                    raise ValueError("Encrypted or excessive form-guide pages")
                pages = [page.extract_text() or "" for page in reader.pages]
            entries = parse_pdf_card_entries(pages, day)
            publication = re.findall(r"Data as at:\s*(\d{1,2}:\d{2} on \d{1,2} [A-Za-z]{3} \d{4})", "\n".join(pages))
            published_at = None
            if len(set(publication)) == 1:
                from zoneinfo import ZoneInfo
                published_at = datetime.strptime(publication[0], "%H:%M on %d %b %Y").replace(tzinfo=ZoneInfo("Asia/Hong_Kong")).isoformat()
            write_json(output / f"formguide_text_{day}.json", {"source": source, "pages": pages})
            for race in [r for r in races if r["race_date"] == day and r["venue"] == venue]:
                indexes = []
                if not calibration:
                    result_page, _ = fetcher.fetch(race["source"]["source_url"])
                    indexes = re.findall(r"^RACE\s+" + str(race["race_no"]) + r"\s+\((\d+)\)$", visible(result_page), re.M | re.I)
                    if len(indexes) != 1:
                        raise ValueError("Official result race index missing or ambiguous")
                for runner in race["runners"]:
                    candidates = [e for e in entries if (e["official_race_index"] == int(indexes[0]) if indexes else e["pdf_race_no"] == race["race_no"])
                                  and e["venue"] == venue and e["distance"] == race["distance"]
                                  and (e["pdf_horse_no"] == runner["horse_no"] if calibration else e["pdf_horse_no"] in (None, runner["horse_no"]))
                                  and e["horse_code"] == runner["horse_code"]
                                  and name_key(e["horse_name"]) == name_key(runner["horse_name"])]
                    if len(candidates) != 1:
                        gaps.append({"race_id": race["race_id"], "horse_page_id": runner["horse_page_id"],
                                     "horse_name": runner["horse_name"], "candidate_count": len(candidates)})
                        continue
                    records.append(candidates[0] | {"race_id": race["race_id"], "race_no": race["race_no"],
                        "date": day,
                        "horse_no": runner["horse_no"], "horse_page_id": runner["horse_page_id"],
                        "horse_id": runner["horse_page_id"], "source": source, "published_at": published_at,
                        "publication_raw": publication[0] if published_at else None,
                        "rating_source_url": source["source_url"], "rating_source_body_hash": source["sha256"]})
            completed += 1
            print(json.dumps({"event": f"{scope}_pdf_enrichment", "race_date": day,
                              "entries": len(entries), "enriched_total": len(records), "gaps": len(gaps)}), flush=True)
        except Exception as exc:
            errors.append({"race_date": day, "source_url": url, "error": str(exc)})
            print(json.dumps({"event": "pdf_enrichment_error", **errors[-1]}), flush=True)
        checkpoint()
    result = checkpoint()
    fetcher.session.close()
    return result


def cache_unit_rules(fetcher):
    from io import BytesIO
    from pypdf import PdfReader
    _, source = fetcher.fetch(UNIT_RULES)
    body = (fetcher.output / source["raw_path"]).read_bytes()
    if not body.startswith(b"%PDF-"):
        raise ValueError("Official dividend rules did not return a PDF")
    page = PdfReader(BytesIO(body)).pages[0].extract_text() or ""
    if page.count("Unit Bet $10") < 8:
        raise ValueError("Official rules do not verify the eight standard unit amounts")
    write_json(fetcher.output / "dividend_units.json", {
        "source": source, "standard_unit_hkd": {p: 10 for p in POOLS},
        "evidence": "Unit Bet $10", "pdf_page": 1,
        "explicit_source_suffix_overrides_standard": True,
        "qpl_absence_context": "Official rules terminate QPL with fewer than seven starters; absence is preserved as unavailable, not a synthetic refund.",
        "validation": "All eight requested pools have Unit Bet $10 entries on page 1"})


def parse_horse_history(response, horse_id):
    from scrapper.official_corpus import parse_official_profile, displayed_brand_matches
    profile = parse_official_profile(response, horse_id)
    if not profile.brand_code or not displayed_brand_matches(profile.brand_code, horse_id):
        raise ValueError("Displayed form horse brand does not verify requested full ID")
    parsed = profile.serializable()["form_records"]
    by_key = {(r["race_index"], r["date"]): r for r in parsed}
    forms, unparsed = [], []
    for table in response.css("table"):
        rows = table.xpath("./tr|./thead/tr|./tbody/tr")
        if not rows or [clean(c) for c in rows[0].xpath("./th|./td")][:3] != ["Race Index", "Pla.", "Date"]:
            continue
        for row in rows[1:]:
            cells = [clean(c) for c in row.xpath("./td")]
            if len(cells) < 18 or not re.fullmatch(r"\d{2}/\d{2}/(?:\d{2}|\d{4})", cells[2]):
                continue
            form_date = datetime.strptime(cells[2], "%d/%m/%y" if len(cells[2]) == 8 else "%d/%m/%Y").date().isoformat()
            if form_date > END:
                continue
            item = dict(by_key.get((cells[0], cells[2]), {}))
            if not item:
                unparsed.append({"race_date": form_date, "source_cells": cells})
                continue
            link = next((urljoin(response.url, h) for h in row.xpath("./td[1]//a/@href").getall()
                         if "localresults" in h.lower()), None)
            query = {k.lower(): v for k, v in parse_qsl(urlsplit(link).query)} if link else {}
            if link and query.get("racedate", "").replace("/", "-") != form_date:
                raise ValueError("Form date and result link disagree")
            venue = query.get("racecourse") or cells[3].split("/")[0].strip()
            number = int(query["raceno"]) if query.get("raceno", "").isdigit() else None
            item.update(horse_page_id=horse_id, horse_id=horse_id, race_date=form_date,
                        venue=venue if venue in {"ST", "HV"} else None, race_no=number,
                        race_id=f"{form_date}_{venue}_R{number}" if venue in {"ST", "HV"} and number else None,
                        result_source_url=link, source_cells=cells, placing_raw=cells[1],
                        dead_heat="DH" in cells[1].upper(), published_at=None)
            item["finishing_status"] = "WITHDRAWN" if cells[1].upper() in WITHDRAWN else (
                "FINISHED" if item["placing"] is not None else cells[1].upper())
            forms.append(item)
    keys = [(r["race_date"], r["race_index"]) for r in forms]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate horse form rows")
    if not forms and profile.starts != 0:
        raise ValueError("No recognized horse form records; not assumed no previous starts")
    forms.sort(key=lambda r: (r["race_date"], r["race_index"]), reverse=True)
    starts_count = sum(r["finishing_status"] != "WITHDRAWN" for r in forms)
    return {"horse_page_id": horse_id, "horse_id": horse_id, "horse_name": profile.horse_name,
            "reported_local_starts": profile.starts, "form_records": forms,
            "form_record_count": len(forms), "unparsed_form_rows": unparsed,
            "captured_start_count": starts_count,
            "withdrawn_form_count": len(forms) - starts_count,
            "oldest_form_date": forms[-1]["race_date"] if forms else None,
            "latest_form_date": forms[0]["race_date"] if forms else None,
            "reported_starts_match_captured": profile.starts == starts_count,
            "history_complete": profile.starts is not None and profile.starts == starts_count and not unparsed,
            "profile_attributes_policy": "Current profile values are not historical race features",
            "status": "captured_form" if forms else "source_reports_zero_local_starts"}


def collect_horse_histories(output=OUTPUT, max_requests=150, delay=0.6, timeout=20, offline=False,
                           history_path=ROOT / ".tmp/season-official-history.parquet"):
    output = Path(output)
    cards = json.loads((output / "racecards.json").read_text())
    entrants = {r["horse_page_id"]: r for c in cards for r in c["runners"]}
    histories, errors = [], []
    fetcher = Fetcher(output, max_requests, delay, timeout, offline)
    baseline = None
    baseline_info = {"path": str(history_path), "status": "not_inspected", "complete_history": False}
    if history_path.exists():
        try:
            import pandas as pd
            baseline = pd.read_parquet(history_path)
            baseline_info.update(status="read_only", rows=len(baseline),
                calendar_2026_races=int(baseline[baseline["date"].between("2026-01-01", END)][["date", "venue", "race_no"]].drop_duplicates().shape[0]),
                september_races=int(baseline[baseline["date"].between(START, "2026-09-30")][["date", "venue", "race_no"]].drop_duplicates().shape[0]),
                jan_jul_races=int(baseline[baseline["date"].between("2026-01-01", "2026-07-31")][["date", "venue", "race_no"]].drop_duplicates().shape[0]))
        except (ImportError, ValueError, OSError, KeyError) as exc:
            baseline_info.update(status="unavailable", error=str(exc))

    def checkpoint():
        missing = sorted(set(entrants) - {h["horse_page_id"] for h in histories})
        manifest = {"generated_at": now(), "schema_version": VERSION,
                    "scope": "Official Show All form pages for Oct7 HV declared starters; prior local starts only",
                    "expected_horses": len(entrants), "captured_horses": len(histories),
                    "form_records": sum(h["form_record_count"] for h in histories),
                    "page_capture_complete": not missing and not errors,
                    "history_complete": not missing and not errors and all(h["history_complete"] for h in histories),
                    "calendar_2026_all_race_history_complete": False,
                    "missing_horses": missing, "errors": errors,
                    "horse_history_gaps": [{"horse_page_id": h["horse_page_id"], "reported_local_starts": h["reported_local_starts"],
                                            "captured_form_records": h["form_record_count"], "captured_starts": h.get("captured_start_count"),
                                            "unparsed_rows": len(h["unparsed_form_rows"])}
                                           for h in histories if not h["history_complete"]],
                    "recent_six_gaps": [{"horse_page_id": h["horse_page_id"], "expected_card_recent_runs": h["expected_card_recent_runs"],
                                         "captured_form_records": h["form_record_count"]} for h in histories
                                        if h["form_record_count"] < h["expected_card_recent_runs"]],
                    "baseline": baseline_info, "sources": list(fetcher.sources.values()),
                    "requests_made_this_run": fetcher.requests_made, "request_budget": max_requests,
                    "request_delay_seconds": delay,
                    "jan_jul_backfill_estimate": {"method": "Planning estimate; exact fixture/navigation census not yet collected",
                        "existing_distinct_races": baseline_info.get("jan_jul_races"),
                        "estimated_missing_race_pages": [150, 220], "additional_monthly_fixture_pages": 7,
                        "additional_meeting_navigation_pages": 56,
                        "bounded_batches": "2 batches, at most 150 new requests each; reuse verified raw hashes and fetch only missing races",
                        "estimated_minutes": [10, 20], "llm_credits_required": 0,
                        "status": "not_executed; no full-year completeness claim"}}
        write_json(output / "horse_histories.json", histories)
        write_json(output / "history_coverage_manifest.json", manifest)
        return manifest

    for horse_id, entrant in sorted(entrants.items()):
        url = BASE + "/en-us/local/information/horse?" + urlencode({"horseid": horse_id, "Option": 1})
        try:
            response, source = fetcher.fetch(url)
            record = parse_horse_history(response, horse_id)
            record["source"] = source
            record["expected_card_recent_runs"] = len([x for x in (entrant.get("last_six_runs") or "").split("/") if re.fullmatch(r"\d+|PU|UR|FE|TNP|DISQ", x)])
            if baseline is not None:
                prior = baseline[baseline["horse_id"] == horse_id]
                keys = {(r.date, r.venue, int(r.race_no)) for r in prior.itertuples()}
                recovered = [r for r in record["form_records"] if r["race_no"] is not None and
                             (r["race_date"], r["venue"], r["race_no"]) not in keys]
                record["baseline_rows_for_horse"] = len(prior)
                record["form_records_missing_from_baseline"] = len(recovered)
                record["missing_baseline_race_ids"] = [r["race_id"] for r in recovered]
            histories.append(record)
            print(json.dumps({"event": "horse_history", "horse_id": horse_id,
                              "records": record["form_record_count"], "history_complete": record["history_complete"]}), flush=True)
        except ValueError as exc:
            errors.append({"horse_page_id": horse_id, "source_url": url, "error": str(exc)})
            print(json.dumps({"event": "history_failure", **errors[-1]}), flush=True)
        checkpoint()
    result = checkpoint()
    fetcher.session.close()
    return result


def fill_history(output=OUTPUT, max_requests=400, delay=1.0, timeout=20, offline=False,
                 max_seconds=1800, history_path=ROOT / ".tmp/season-official-history.parquet"):
    """Repair missing Jan-Jul race IDs using full opponent fields, never form-only rows."""
    import pandas as pd
    output = Path(output)
    started = time.monotonic()
    fetcher = Fetcher(output, max_requests, delay, timeout, offline)
    fetcher.deadline = started + max_seconds
    baseline = pd.read_parquet(history_path)
    scoped = baseline[baseline["date"].between("2026-01-01", "2026-07-31")]
    baseline_keys = {(r.date, r.venue, int(r.race_no)) for r in
                     scoped[["date", "venue", "race_no"]].drop_duplicates().itertuples()}
    baseline_hash = hashlib.sha256(history_path.read_bytes()).hexdigest()
    prior_requests, prior_elapsed = 0, 0.0
    previous_path = output / "history-race-coverage.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous.get("baseline", {}).get("sha256") == baseline_hash:
            prior_requests = previous.get("cumulative_requests", previous.get("requests_made_this_run", 0))
            prior_elapsed = previous.get("cumulative_elapsed_seconds", previous.get("elapsed_seconds", 0))
    races, meetings, fixtures, errors = [], [], [], []
    stop_reason = None

    def checkpoint():
        expected = {(m["race_date"], m["venue"], n) for m in meetings
                    if m["status"] != "cancelled" for n in m.get("race_numbers", [])}
        acquired = {(r["race_date"], r["venue"], r["race_no"]) for r in races}
        covered = acquired | baseline_keys
        missing = sorted(expected - covered)
        unresolved = [m for m in meetings if m["status"] != "cancelled" and not m.get("race_numbers")]
        manifest = {"schema_version": VERSION, "generated_at": now(),
            "scope": {"start": "2026-01-01", "end": "2026-07-31", "venue": ["ST", "HV"],
                      "kind": "full official result fields plus requested dividends"},
            "complete": len(fixtures) == 7 and bool(expected) and not missing and not unresolved and not errors and stop_reason is None,
            "baseline": {"path": str(history_path), "sha256": baseline_hash,
                         "rows": len(baseline), "jan_jul_race_ids": len(baseline_keys),
                         "policy": "Read only; existing official race IDs counted as baseline coverage, full original fields not independently re-audited here"},
            "fixture_months_captured": len(fixtures), "fixture_sources": fixtures,
            "meetings": meetings, "expected_meetings": sum(m["status"] != "cancelled" for m in meetings),
            "enumerated_meetings": sum(bool(m.get("race_numbers")) for m in meetings),
            "expected_races": len(expected), "covered_races": len(expected & covered),
            "baseline_races_in_denominator": len(expected & baseline_keys),
            "new_races": len(races), "new_runners": sum(len(r["runners"]) for r in races),
            "missing_races": [list(k) for k in missing], "unresolved_meetings": unresolved,
            "cancellations": [m for m in meetings if m["status"] == "cancelled"],
            "errors": errors, "stopped_reason": stop_reason,
            "requests_made_this_run": fetcher.requests_made, "request_budget": max_requests,
            "prior_checkpoint_requests": prior_requests,
            "cumulative_requests": prior_requests + fetcher.requests_made,
            "cumulative_elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
            "request_delay_seconds": delay, "max_seconds": max_seconds,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "sources": list(fetcher.sources.values()),
            "output": "history-races.json", "record_schema": "Same as races.json; all runners from each newly captured race",
            "not_claimed": "No all-years completeness, no form-only opponent reconstruction, and no independent validation of baseline race fields",
            "ready_for_freeze": len(fixtures) == 7 and not missing and not unresolved and not errors and stop_reason is None}
        write_json(output / "history-races.json", sorted(races, key=lambda r: (r["race_date"], r["venue"], r["race_no"])))
        write_json(output / "history-race-coverage.json", manifest)
        return manifest

    def stopped():
        nonlocal stop_reason
        if time.monotonic() - started >= max_seconds:
            stop_reason = "wall_clock_deadline_reached"
        elif fetcher.requests_made >= max_requests:
            stop_reason = "request_budget_reached"
        return stop_reason is not None

    checkpoint()
    for month in range(1, 8):
        if stopped():
            break
        url = BASE + "/en-us/local/information/fixture?" + urlencode({"CalMonth": f"{month:02}", "CalYear": 2026})
        try:
            response, source = fetcher.fetch(url)
            fixture = parse_fixture(response, 2026, month)
            fixtures.append(source)
            write_json(output / f"fixture_2026_{month:02}.json", fixture | {"source": source})
            for meeting in fixture["meetings"]:
                if "2026-01-01" <= meeting["race_date"] <= "2026-07-31":
                    meetings.append(meeting | {"fixture_source": source})
            for cancelled in fixture["cancellations"]:
                if "2026-01-01" <= cancelled["race_date"] <= "2026-07-31" and not any(
                        (m["race_date"], m["venue"]) == (cancelled["race_date"], cancelled["venue"]) for m in meetings):
                    meetings.append(cancelled | {"fixture_source": source})
            print(json.dumps({"event": "history_fixture", "month": month,
                              "meetings": len(fixture["meetings"])}), flush=True)
        except ValueError as exc:
            errors.append({"kind": "history_fixture", "source_url": url, "error": str(exc)})
            print(json.dumps({"event": "history_fixture_failure", **errors[-1]}), flush=True)
        checkpoint()
    # Visit recent meetings first so an interrupted bounded pass still repairs recent features.
    meetings.sort(key=lambda m: (m["race_date"], m["venue"]), reverse=True)
    for meeting in meetings:
        if meeting["status"] == "cancelled":
            continue
        if stopped():
            break
        day, venue = meeting["race_date"], meeting["venue"]
        try:
            first, numbers = collect_race(fetcher, "results", day, venue, 1, archive=True)
            meeting["race_numbers"] = numbers
            meeting["result_navigation_source"] = first["source"]
            meeting["fixture_count_matches_results"] = meeting.get("scheduled_race_count") == len(numbers)
            if (day, venue, 1) not in baseline_keys:
                races.append(first)
            for number in numbers[1:]:
                if (day, venue, number) in baseline_keys:
                    continue
                if stopped():
                    break
                try:
                    record, other_numbers = collect_race(fetcher, "results", day, venue, number, archive=True)
                    if other_numbers != numbers:
                        raise ValueError("Prior meeting race navigation changed")
                    races.append(record)
                except ValueError as exc:
                    errors.append({"kind": "history_result", "race_date": day, "venue": venue,
                                   "race_no": number, "error": str(exc)})
                checkpoint()
            meeting["status"] = "enumerated"
            print(json.dumps({"event": "history_meeting", "race_date": day, "venue": venue,
                              "race_count": len(numbers), "new_races_total": len(races),
                              "requests": fetcher.requests_made,
                              "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
        except ValueError as exc:
            meeting["status"] = "unresolved"
            errors.append({"kind": "history_meeting", "race_date": day, "venue": venue, "error": str(exc)})
            print(json.dumps({"event": "history_meeting_failure", **errors[-1]}), flush=True)
        checkpoint()
    result = checkpoint()
    fetcher.session.close()
    return result


def freeze_output(output=OUTPUT):
    """Hash the completed snapshots and verify every referenced original source."""
    output = Path(output)
    manifests = {name: json.loads((output / name).read_text()) for name in (
        "coverage_manifest.json", "history_coverage_manifest.json", "history-race-coverage.json")}
    enrichment = output / "season-enrichment-coverage.json"
    if enrichment.exists():
        manifests[enrichment.name] = json.loads(enrichment.read_text())
        if not manifests[enrichment.name]["complete"]:
            raise ValueError("Cannot freeze incomplete season rating enrichment")
    calibration = output / "calibration-enrichment-coverage.json"
    if calibration.exists():
        manifests[calibration.name] = json.loads(calibration.read_text())
        if not manifests[calibration.name].get("acquisition_finished"):
            raise ValueError("Cannot freeze calibration acquisition while still running")
    if (not manifests["coverage_manifest.json"]["complete"]
            or not manifests["history_coverage_manifest.json"]["page_capture_complete"]
            or not manifests["history-race-coverage.json"]["complete"]):
        raise ValueError("Cannot freeze incomplete season, horse-page or full-race acquisition")
    baseline = manifests["history-race-coverage.json"]["baseline"]
    if hashlib.sha256(Path(baseline["path"]).read_bytes()).hexdigest() != baseline["sha256"]:
        raise ValueError("Original read-only baseline changed during acquisition")
    sources = {}
    referenced = [json.loads((output / "dividend_units.json").read_text())["source"]]
    weight_verification = output / "card-weight-verification.json"
    if weight_verification.exists():
        evidence = json.loads(weight_verification.read_text())
        referenced.extend((evidence["source"], evidence["method_evidence"]["source"]))
    referenced.extend(source for manifest in manifests.values() for source in manifest.get("sources", []))
    datasets = ["races.json", "racecards.json", "horse_histories.json", "history-races.json"]
    if enrichment.exists():
        datasets.append("season-runner-enrichment.json")
    if calibration.exists():
        datasets.append("calibration-runner-enrichment.json")
    for filename in datasets:
        for record in json.loads((output / filename).read_text()):
            referenced.append(record["source"])
    for source in referenced:
        raw = (output / source["raw_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Source hash mismatch before freeze: {source['raw_path']}")
        sources[source["raw_path"]] = source["sha256"]
    files = {}
    for path in sorted(output.glob("*.json")):
        if path.name == "data-freeze.json":
            continue
        body = path.read_bytes()
        files[path.name] = {"sha256": hashlib.sha256(body).hexdigest(), "byte_count": len(body)}
    freeze_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    freeze = {"schema_version": VERSION, "frozen_at": now(), "freeze_id": freeze_id,
              "calibration_enrichment_complete": manifests.get("calibration-enrichment-coverage.json", {}).get("complete"),
              "calibration_whole_field_finite_rating_races": manifests.get("calibration-enrichment-coverage.json", {}).get("whole_field_finite_rating_races"),
              "files": files, "source_hashes": sources, "source_count": len(sources),
              "original_baseline_verified_unchanged": True, "baseline": baseline,
              "season_result_races": manifests["coverage_manifest.json"]["collected_result_races"],
              "racecards": manifests["coverage_manifest.json"]["collected_racecards"],
              "horse_pages": manifests["history_coverage_manifest.json"]["captured_horses"],
              "new_history_races": manifests["history-race-coverage.json"]["new_races"],
              "jan_jul_combined_race_id_coverage": manifests["history-race-coverage.json"]["expected_races"],
              "scope_note": "Jan-Jul baseline plus repaired race IDs; complete Sep-Oct6 results and Oct7 cards. Does not claim all-years history or validate downstream models."}
    write_json(output / "data-freeze.json", freeze)
    return freeze


def collect(output=OUTPUT, max_requests=150, delay=0.6, timeout=20, offline=False):
    output = Path(output)
    fetcher = Fetcher(output, max_requests, delay, timeout, offline)
    races, cards, meetings, errors, fixtures, census = [], [], [], [], [], []
    write_json(output / "schema.json", api_schema())

    def checkpoint():
        write_json(output / "races.json", races)
        write_json(output / "racecards.json", cards)
        write_json(output / "meetings.json", meetings)
        manifest = coverage_manifest(meetings, races, cards, errors, census, fixtures, fetcher.sources)
        manifest["requests_made_this_run"] = fetcher.requests_made
        manifest["request_budget"] = max_requests
        manifest["request_delay_seconds"] = delay
        write_json(output / "coverage_manifest.json", manifest)
        return manifest

    for number in range(1, 10):
        try:
            record, navigation = collect_race(fetcher, "racecard", CARD_DATE, "HV", number)
            if navigation != list(range(1, 10)):
                raise ValueError(f"Official racecard navigation is not nine races: {navigation}")
            cards.append(record)
            print(json.dumps({"event": "racecard", "race_id": record["race_id"], "runners": len(record["runners"])}), flush=True)
        except ValueError as exc:
            errors.append({"kind": "racecard", "race_no": number, "error": str(exc)})
        checkpoint()

    for month in (9, 10):
        url = BASE + "/en-us/local/information/fixture?" + urlencode({"CalMonth": f"{month:02}", "CalYear": 2026})
        try:
            response, source = fetcher.fetch(url)
            fixture = parse_fixture(response, 2026, month)
            fixtures.append(source)
            write_json(output / f"fixture_2026_{month:02}.json", fixture | {"source": source})
            for meeting in fixture["meetings"]:
                if START <= meeting["race_date"] <= END:
                    meetings.append(meeting | {"fixture_source": source})
            for cancelled in fixture["cancellations"]:
                if START <= cancelled["race_date"] <= END and not any(
                        (m["race_date"], m["venue"]) == (cancelled["race_date"], cancelled["venue"]) for m in meetings):
                    meetings.append(cancelled | {"fixture_source": source})
        except ValueError as exc:
            errors.append({"kind": "fixture", "source_url": url, "error": str(exc)})
    try:
        cache_unit_rules(fetcher)
    except (ValueError, ImportError, OSError) as exc:
        errors.append({"kind": "dividend_units", "source_url": UNIT_RULES, "error": str(exc)})
    try:
        response, source = fetcher.fetch(BASE + DATE_LIST + "?lang=en-us")
        document = parse_document(response)
        if document["status"] != "fetched_parsed":
            raise ValueError(document.get("parse_error", "Missing date list"))
        census = [c for c in document["date_candidates"] if START <= c["date"] <= END]
        write_json(output / "official_dates.json", {"date_candidates": census, "source": source})
        for entry in census:
            for venue in entry["venues"]:
                if not any((m["race_date"], m["venue"]) == (entry["date"], venue) for m in meetings):
                    meetings.append({"race_date": entry["date"], "venue": venue, "status": "scheduled",
                                     "scheduled_race_count": None, "date_list_source": source})
    except ValueError as exc:
        errors.append({"kind": "date_list", "error": str(exc)})
    meetings.sort(key=lambda m: (m["race_date"], m["venue"]))
    checkpoint()

    for meeting in meetings:
        if meeting["status"] == "cancelled":
            continue
        day, venue = meeting["race_date"], meeting["venue"]
        try:
            first, numbers = collect_race(fetcher, "results", day, venue, 1)
            meeting["race_numbers"] = numbers
            meeting["race_count"] = len(numbers)
            meeting["result_source"] = first["source"]
            meeting["fixture_count_matches_results"] = meeting.get("scheduled_race_count") == len(numbers)
            races.append(first)
            for number in numbers[1:]:
                try:
                    record, other_numbers = collect_race(fetcher, "results", day, venue, number)
                    if other_numbers != numbers:
                        raise ValueError("Race navigation changed within meeting")
                    races.append(record)
                except ValueError as exc:
                    errors.append({"kind": "results", "race_date": day, "venue": venue,
                                   "race_no": number, "error": str(exc)})
                checkpoint()
            meeting["status"] = "completed" if sum(r["race_date"] == day and r["venue"] == venue for r in races) == len(numbers) else "partial"
            print(json.dumps({"event": "meeting", "date": day, "venue": venue,
                              "races": len(numbers), "status": meeting["status"]}), flush=True)
        except ValueError as exc:
            meeting["status"] = "unresolved"
            errors.append({"kind": "meeting", "race_date": day, "venue": venue, "error": str(exc)})
        checkpoint()
    races.sort(key=lambda r: (r["race_date"], r["venue"], r["race_no"]))
    manifest = checkpoint()
    fetcher.session.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-requests", type=int, default=150)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--horse-history", action="store_true", help="Recover Show All horse form pages after the priority season collection")
    parser.add_argument("--fill-history", action="store_true", help="Fill missing Jan-Jul 2026 full race fields against read-only parent history")
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--freeze", action="store_true", help="Verify source hashes and freeze completed local artifacts")
    parser.add_argument("--reparse-cards", action="store_true", help="Refresh only cached racecards without network or history mutation")
    parser.add_argument("--enrich-season", action="store_true", help="Recover exact-date season ratings/gear from official archived meeting PDFs")
    parser.add_argument("--enrich-calibration", action="store_true", help="Enrich calibration census from dated official meeting PDFs")
    args = parser.parse_args()
    if sum((args.horse_history, args.fill_history, args.freeze, args.reparse_cards, args.enrich_season, args.enrich_calibration)) > 1:
        parser.error("Choose one collection or freeze mode")
    ceiling = 600 if args.enrich_calibration else 400 if args.fill_history else 200
    minimum_delay = 1.0 if args.fill_history or args.enrich_calibration else 0.5
    if not 1 <= args.max_requests <= ceiling or args.delay < minimum_delay or not 1 <= args.timeout <= 30 or not 1 <= args.max_seconds <= (2700 if args.enrich_calibration else 1800):
        parser.error(f"Require 1..{ceiling} requests, delay >= {minimum_delay} seconds, timeout 1..30 seconds, deadline <= 1800 seconds")
    if args.freeze:
        result = freeze_output(args.output)
        print(json.dumps({k: result[k] for k in ("freeze_id", "source_count", "season_result_races", "new_history_races")}))
        return 0
    if args.reparse_cards:
        print(json.dumps(reparse_cards(args.output)))
        return 0
    if args.enrich_season or args.enrich_calibration:
        result = enrich_season(args.output, args.max_requests, args.delay, args.timeout, args.offline,
                               args.enrich_calibration, args.max_seconds)
        print(json.dumps({k: result[k] for k in ("complete", "enriched_runners", "gaps", "errors")}))
        return 0 if result["complete"] else 2
    if args.fill_history:
        result = fill_history(args.output, args.max_requests, args.delay, args.timeout, args.offline, args.max_seconds)
        print(json.dumps({k: result[k] for k in ("complete", "expected_meetings", "expected_races", "new_races", "missing_races", "errors", "stopped_reason")}))
        return 0 if result["complete"] else 2
    if args.horse_history:
        result = collect_horse_histories(args.output, args.max_requests, args.delay, args.timeout, args.offline)
        print(json.dumps({k: result[k] for k in ("page_capture_complete", "history_complete", "expected_horses", "captured_horses", "form_records", "errors")}))
        return 0 if result["page_capture_complete"] else 2
    result = collect(args.output, args.max_requests, args.delay, args.timeout, args.offline)
    print(json.dumps({k: v for k, v in result.items() if k in {
        "complete", "meeting_count", "collected_result_races", "collected_racecards", "dividend_records", "errors", "pool_gaps"}}, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
