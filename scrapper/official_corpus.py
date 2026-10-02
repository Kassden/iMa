"""Scoped, resumable acquisition of public HKJC racing records."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sqlite3
from io import BytesIO
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit

import scrapy
from lxml import etree
from scrapy.exceptions import IgnoreRequest

from scrapper.historical.results import parse_race
from scrapper.horse_pages import parse_profile_and_form, _finish_seconds

ROOT = "https://racing.hkjc.com/en-us/local/information/"
HOST = "racing.hkjc.com"
HORSE_ID = re.compile(r"HK_\d{4}_[A-Z]\d{3,4}", re.I)
ROUTES = {
    "horse", "otherhorse", "horselist", "horseformername", "horsepedigree",
    "trackworkresult", "trackworksearch", "trackworkhorsesearch", "localtrackwork", "trackworkonedayresult",
    "ovehorse", "veterinaryrecord", "veterinaryrecords", "localvetrecord",
    "movementrecords", "horsemovementrecords", "btresult", "btresultsearch",
    "localresults", "results", "racingcalendar", "fixtures", "racecard",
    "racingincidentreport", "incidentreport", "comments", "formline",
    "sectional", "sectionaltime", "displaysectionaltime", "trackwork", "horseform", "newhorse",
    "racereportfull", "racereportext", "corunning", "trackworkotherresult", "oveotherhorse",
}
PARSER_VERSION = "official-corpus-v14"
MOVEMENT_PDF = "/general/-/media/Sites/JCRW/Page/content/conghua.pdf"
DAILY_JSON = re.compile(r"^/racing/information/json/TrackworkOneDayRecords/(\d{8})1E\.aspx$")
DATE_LIST = "/racing/information/json/DateList/LocalResults.aspx"
NONFINISHERS = {"PU", "UR", "FE", "DNF", "DISQ", "TNP"}
WITHDRAWN = {"WX", "WV", "WX-A", "WV-A", "WXNR", "W", "SCR", "WD"}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_url(value: str, base: str = ROOT) -> str | None:
    try:
        parts = urlsplit(urljoin(base, value))
    except ValueError:
        return None
    if parts.scheme != "https" or parts.hostname != HOST or parts.username or parts.password:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port not in (None, 443):
        return None
    path = parts.path
    if any(p in {".", ".."} for p in unquote(path).split("/")):
        return None
    if path == MOVEMENT_PDF:
        query = dict(parse_qsl(parts.query))
        if set(query) - {"rev", "sc_lang"} or (query.get("rev") and not re.fullmatch(r"[a-f0-9]{32}", query["rev"])):
            return None
        if query.get("sc_lang", "en-US") != "en-US":
            return None
        return urlunsplit(("https", HOST, path, urlencode(sorted(query.items())), ""))
    if path == DATE_LIST:
        if parse_qsl(parts.query) != [("lang", "en-us")]:
            return None
        return urlunsplit(("https", HOST, path, "lang=en-us", ""))
    json_match = DAILY_JSON.fullmatch(path)
    if json_match:
        try:
            datetime.strptime(json_match[1], "%Y%m%d")
        except ValueError:
            return None
        query = dict(parse_qsl(parts.query))
        if set(query) != {"PageNum"} or not query["PageNum"].isdigit() or not 1 <= int(query["PageNum"]) <= 5000:
            return None
        return urlunsplit(("https", HOST, path, urlencode(query), ""))
    if not (path.lower().startswith("/en-us/local/information/") or
            path.lower().startswith("/racing/information/english/")):
        return None
    route = path.rsplit("/", 1)[-1].lower().removesuffix(".aspx")
    if route not in ROUTES:
        return None
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if not k.lower().startswith("utm_") and k.lower() != "b_cid")
    return urlunsplit(("https", HOST, path, urlencode(query), ""))


def family(url: str) -> str:
    if urlsplit(url).path == MOVEMENT_PDF:
        return "movements"
    if urlsplit(url).path == DATE_LIST:
        return "race_census"
    if DAILY_JSON.fullmatch(urlsplit(url).path):
        return "trackwork"
    route = urlsplit(url).path.rsplit("/", 1)[-1].lower().removesuffix(".aspx")
    if route in {"horse", "otherhorse", "horseform"}:
        return "horse"
    if route in {"localresults", "results"}:
        return "results"
    if "trackwork" in route:
        return "trackwork"
    if "vet" in route or route in {"ovehorse", "oveotherhorse"}:
        return "veterinary"
    if "movement" in route:
        return "movements"
    if route.startswith("bt"):
        return "barrier_trials"
    if "sectional" in route:
        return "sectionals"
    if "incident" in route or route in {"comments", "racereportfull", "racereportext", "corunning"}:
        return "incidents"
    return "discovery"


def extract_tables(response):
    return [
        [[" ".join(" ".join(cell.xpath(".//text()").getall()).split())
          for cell in row.xpath("./th|./td")] for row in table.xpath(".//tr")]
        for table in response.css("table")
    ]


def event_date(value):
    for pattern in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), pattern).date().isoformat()
        except ValueError:
            pass
    return None


def verify_race_context(response, query):
    visible = "\n".join(" ".join(text.split()) for text in response.xpath(
        "//body//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::select)]").getall())
    meetings = re.findall(r"Race Meeting:\s*(\d{1,2}/\d{1,2}/\d{4})\s+(Sha Tin|Happy Valley)", visible, re.I)
    contexts = {(event_date(d), "ST" if venue.lower() == "sha tin" else "HV") for d, venue in meetings}
    if len(contexts) != 1:
        raise ValueError("Displayed race date/venue missing or ambiguous")
    date, venue = contexts.pop()
    requested_date = query.get("racedate")
    if requested_date and datetime.strptime(requested_date.replace("/", "-"), "%Y-%m-%d").date().isoformat() != date:
        raise ValueError("Requested race date/venue not verified in displayed meeting header")
    if query.get("racecourse") and query["racecourse"].upper() != venue:
        raise ValueError("Requested race date/venue not verified in displayed meeting header")
    numbers = {int(n) for n in re.findall(r"^RACE\s+(\d+)\s+\(\d+\)$", visible, re.M | re.I)}
    if len(numbers) != 1:
        raise ValueError("Displayed race number missing or ambiguous")
    number = numbers.pop()
    if query.get("raceno") and int(query["raceno"]) != number:
        raise ValueError(f"Requested race number {query['raceno']} differs from displayed {number}")
    return date, venue, number


def result_parser_input(response):
    candidates = response.xpath("//table/tr[1][count(td|th)=11] | //table/thead/tr[1][count(td|th)=11] | //table/tbody/tr[1][count(td|th)=11]")
    if not any(" ".join(row.xpath("./td[1]/text()|./th[1]/text()").getall()).strip() == "Pla." for row in candidates):
        return response.text
    root = deepcopy(response.selector.root)
    for table in root.xpath(".//table"):
        rows = table.xpath("./tr|./thead/tr|./tbody/tr")
        if not rows:
            continue
        header = [" ".join("".join(c.itertext()).split()) for c in rows[0].xpath("./td|./th")]
        if len(header) != 11 or header[:2] != ["Pla.", "Horse No."] or header[9:] != ["Finish Time", "Win Odds"]:
            continue
        # Older official tables omit running positions, not an entire runner.
        for index, row in enumerate(rows):
            cells = row.xpath("./td|./th")
            if len(cells) == 11:
                cell = etree.Element("td")
                cell.text = "Running Position" if index == 0 else ""
                row.insert(list(row).index(cells[9]), cell)
    return etree.tostring(root, encoding="unicode", method="html")


def optional_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def parse_official_profile(response, horse_id):
    data = parse_profile_and_form(response.text, horse_id)
    title = response.css("span.title_text").xpath("string()").get("")
    title = " ".join(title.split())
    match = re.fullmatch(r"(.+?)\s*\(([A-Z]{1,2}\d{3,4})\)\s*(?:\((?:Retired|Deregistered)\))?", title, re.I)
    if match:
        data.horse_name, data.brand_code = match[1], match[2].upper()
    return data


def displayed_brand_matches(brand, horse_id):
    expected = horse_id.rsplit("_", 1)[-1].upper()
    # Official old cards/profile titles use B/C cycle prefixes, e.g. CA018 -> A018.
    # This validates an alias for one full linked ID; it never merges horses by brand.
    return brand.upper() in {expected, "B" + expected, "C" + expected}


def parse_events(response, kind, query):
    """Extract only recognized event tables; unknown/no-info is not zero events."""
    schemas = {
        "trackwork": ("Date", "Type", "Racecourse/Track", "Workouts", "Gear"),
        "movements": ("From", "To", "Arrival Date"),
    }
    events, recognized, batch, batch_context = [], False, None, {}
    for table in response.css("table"):
        rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
        if not rows:
            continue
        texts = [[" ".join(" ".join(c.xpath(".//text()").getall()).split())
                  for c in row.xpath("./th|./td")] for row in rows]
        header = texts[0]
        if header and header[0].startswith("Batch "):
            batch = header[0]
            batch_context = {}
            match = re.fullmatch(r"Batch\s+(\d+)\s*-\s*(SHA TIN|HAPPY VALLEY|CONGHUA)\s+(.+?)\s*-\s*(\d+)m", batch, re.I)
            if match:
                venue_name = match[2].upper()
                batch_context.update(batch_no=int(match[1]), trial_venue_name=venue_name,
                    venue={"SHA TIN": "ST", "HAPPY VALLEY": "HV"}.get(venue_name, "unknown"),
                    track=match[3].upper(), distance_metres=int(match[4]))
            summary = " ".join(" ".join(row) for row in texts)
            going = re.search(r"Going:\s*(.*?)\s*Time:", summary, re.I)
            time = re.search(r"\bTime:\s*(\d+\.\d{2}\.\d{2})", summary, re.I)
            splits = re.search(r"Sectional Time:\s*((?:\d+(?:\.\d+)?\s*)+)", summary, re.I)
            if going:
                batch_context["going"] = going[1].strip()
            if time:
                batch_context["batch_winner_time_seconds"] = _finish_seconds(time[1])
            if splits:
                batch_context["batch_sectional_seconds"] = [float(v) for v in splits[1].split()]
        schema = schemas.get(kind)
        if kind == "veterinary" and header and header[0].lower() == "date" and any("detail" in c.lower() for c in header):
            schema = tuple(header)
        if kind == "veterinary" and header == ["Horse No.", "Horse Name", "Date", "Details", "Passed On"]:
            schema = tuple(header)
        if kind == "barrier_trials" and header[:4] == ["Horse", "Jockey", "Trainer", "Draw"]:
            schema = tuple(header)
        if not schema or tuple(header) != schema:
            continue
        recognized = True
        previous_horse = None
        for index, cells in enumerate(texts[1:], 1):
            if len(cells) != len(header):
                continue
            values = dict(zip(header, cells))
            date = event_date(values.get("Date", values.get("Arrival Date", query.get("date", ""))))
            horse = query.get("horseid")
            if kind == "veterinary" and header[0] == "Horse No.":
                ids = set(HORSE_ID.findall(rows[index].get()))
                if len(ids) == 1:
                    previous_horse = ids.pop()
                elif cells[0] or cells[1]:
                    previous_horse = None
                horse = previous_horse
            if kind == "barrier_trials":
                ids = HORSE_ID.findall(rows[index].get())
                horse = ids[0] if len(set(ids)) == 1 else None
            if not date or not horse or not HORSE_ID.fullmatch(horse):
                continue
            events.append({"horse_id": horse.upper(), "event_date": date,
                "published_at": None, "availability_status": "historical_publication_unverified",
                "values": values, "batch": batch, "table_index": len(events)})
            if kind == "barrier_trials":
                events[-1].update(batch_context)
                seconds = _finish_seconds(values.get("Time", ""))
                events[-1]["trial_finish_seconds"] = seconds if seconds and math.isfinite(seconds) and seconds > 0 else None
            passed = event_date(values.get("Passed On", values.get("Passed Date", "")))
            if kind == "veterinary" and passed:
                events.append({"horse_id": horse.upper(), "event_date": passed,
                    "event_family": "veterinary_clearance", "published_at": None,
                    "availability_status": "historical_publication_unverified",
                    "values": {"related_event_date": date, "clearance_date": passed}})
    return events, recognized


def parse_sectionals(response, query):
    """Keep split-time components separate; never infer identity from horse names."""
    visible = "\n".join(" ".join(t.split()) for t in response.xpath(
        "//body//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::select)]").getall())
    date = event_date(query.get("racedate", ""))
    meetings = re.findall(r"Meeting Date:\s*(\d{1,2}/\d{1,2}/\d{4}),\s*(Sha Tin|Happy Valley)", visible, re.I)
    if len(set(meetings)) != 1 or event_date(meetings[0][0]) != date:
        raise ValueError("Sectional meeting date not verified in displayed content")
    venue = "ST" if meetings[0][1].lower() == "sha tin" else "HV"
    number = int(query.get("raceno", "0"))
    if {int(n) for n in re.findall(r"^Race\s+(\d+)$", visible, re.M | re.I)} != {number}:
        raise ValueError("Sectional race number mismatch or multi-race page")
    tables = [t for t in response.css("table") if t.xpath("./thead/tr[1]/td[1]/text()").get("").strip() == "Finishing Order"]
    if len(tables) != 1:
        raise ValueError("Sectional runner table missing or ambiguous")
    events = []
    for row in tables[0].xpath("./tbody/tr"):
        cells = row.xpath("./td")
        if len(cells) != 10:
            raise ValueError("Unexpected sectional runner column count")
        ids = set(HORSE_ID.findall(cells[2].get()))
        if len(ids) != 1:
            raise ValueError("Sectional runner full identity missing or ambiguous")
        splits = []
        for index, cell in enumerate(cells[3:9], 1):
            time = cell.xpath("./p[2]/text()").get("").strip()
            if not time:
                continue
            seconds = float(time)
            position = cell.xpath("./p[1]/span/text()").get("").strip()
            margin = " ".join(cell.xpath("./p[1]/i/text()").getall()).strip()
            if seconds <= 0 or not position.isdigit():
                raise ValueError("Invalid sectional seconds or running position")
            splits.append({"section": index, "seconds": seconds, "position": int(position),
                           "margin_behind_leader_raw": margin,
                           "subsplit_seconds": [float(v.strip()) for v in cell.css(".color_blue2 span::text").getall() if v.strip()]})
        if not splits:
            raise ValueError("No actual sectional splits")
        saddle = cells[1].xpath("string()").get().strip()
        if not saddle.isdigit():
            raise ValueError("Sectional saddle number missing")
        events.append({"horse_id": ids.pop().upper(), "event_date": date, "venue": venue,
            "race_no": number, "event_family": "sectionals", "published_at": None,
            "availability_status": "historical_publication_unverified",
            "values": {"horse_no": int(saddle), "finish_order_raw": cells[0].xpath("string()").get().strip(),
                       "finish_time_raw": cells[9].xpath("string()").get().strip(), "splits": splits}})
    if not events:
        raise ValueError("Empty sectional runner table")
    return events


def parse_movement_text(text):
    if "HORSES STABLED IN CONGHUA SINCE LAST START" not in text:
        raise ValueError("Unrecognized Conghua movement report")
    headers = re.findall(r"(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\s+(\d{1,2} [A-Z]+ \d{4}) RACE MEETING", text)
    meetings = set()
    for weekday, value in headers:
        date = datetime.strptime(value, "%d %B %Y")
        if date.strftime("%A").upper() != weekday:
            raise ValueError("Movement meeting weekday mismatch")
        meetings.add(date.date().isoformat())
    if len(meetings) != 1:
        raise ValueError("Missing or conflicting movement meeting date")
    if not re.search(r"Race\s+Horse Number\s+Horse Name\s+Arrived in Conghua\s+Returned to HK", text):
        raise ValueError("Movement column schema mismatch")
    meeting = meetings.pop()
    events, race, previous = [], None, None
    for line in text.splitlines():
        line = " ".join(line.split())
        dates = re.fullmatch(r"(.*?)\s*(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", line)
        if not dates:
            if re.search(r"\d{2}/\d{2}/\d{4}", line):
                raise ValueError("Unparsed movement date row")
            continue
        prefix, arrival, returned = dates.groups()
        if prefix:
            identity = re.fullmatch(r"(?:(\d{1,2})\s+)?(Standby\s+\d{1,2}|\d{1,2})\s+(.+)", prefix)
            if not identity:
                raise ValueError("Unparsed movement identity row")
            if identity[1]:
                race = int(identity[1])
            if not race or not 1 <= race <= 20:
                raise ValueError("Missing movement race context")
            name = re.search(r"([A-Z][A-Z0-9 '&().-]*)$", identity[3])
            if not name:
                raise ValueError("Missing movement horse name")
            standby = identity[2].startswith("Standby")
            previous = {"race_no": race, "horse_no": None if standby else int(identity[2]),
                        "standby_no": int(identity[2].split()[-1]) if standby else None,
                        "horse_name": name[1].strip()}
        if not previous:
            raise ValueError("Orphan movement continuation")
        arrival, returned = event_date(arrival), event_date(returned)
        if not arrival or not returned or not arrival <= returned <= meeting:
            raise ValueError("Movement stay dates inconsistent with meeting")
        events.append(previous | {"horse_id": None, "meeting_date": meeting, "venue": "unknown",
            "event_date": returned, "arrived_in_conghua": arrival, "returned_to_hk": returned,
            "event_family": "movements", "identity_status": "meeting_race_saddle_name; verified_card_join_required",
            "published_at": None, "availability_status": "historical_publication_unverified",
            "values": {"source_row": line, "meeting_date": meeting, **previous,
                       "arrived_in_conghua": arrival, "returned_to_hk": returned},
            "table_index": len(events)})
    if not events:
        raise ValueError("No parsed movement stays")
    return events


def parse_document(response) -> dict:
    kind = family(response.url)
    if urlsplit(response.url).path == MOVEMENT_PDF:
        record = {"family": kind, "parser_version": PARSER_VERSION,
                  "horse_ids": [], "tables": [], "status": "fetched_unparsed"}
        try:
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
            if not response.body.startswith(b"%PDF-") or len(response.body) > 5 * 1024 * 1024:
                raise ValueError("Invalid or excessive movement PDF")
            try:
                reader = PdfReader(BytesIO(response.body))
            except PyPdfError as exc:
                raise ValueError(f"Invalid movement PDF: {exc}") from exc
            if reader.is_encrypted or not 1 <= len(reader.pages) <= 20:
                raise ValueError("Encrypted or excessive movement PDF pages")
            texts = []
            for page in reader.pages:
                contents = page.get_contents()
                if contents is None:
                    raise ValueError("Missing PDF page content stream")
                if len(contents.get_data()) > 5 * 1024 * 1024:
                    raise ValueError("Excessive PDF content stream")
                texts.append(page.extract_text() or "")
            record.update(events=parse_movement_text("\n".join(texts)), status="fetched_parsed")
        except (ValueError, TypeError, ImportError) as exc:
            record["parse_error"] = str(exc)
        return record
    source = response.text
    if urlsplit(response.url).path == DATE_LIST:
        record = {"family": "race_census", "parser_version": PARSER_VERSION,
                  "horse_ids": [], "tables": [], "status": "fetched_unparsed"}
        try:
            entries = json.loads(source)["MeetingDateList"]
            if not isinstance(entries, list) or not entries:
                raise ValueError("Missing official result-date candidates")
            dates = []
            for entry in entries:
                date = datetime.fromisoformat(entry["Key"]).date().isoformat()
                venues = entry["Value"]
                if venues is not None and (not isinstance(venues, dict) or
                        any(v not in {"ST", "HV"} for v in venues)):
                    raise ValueError("Unrecognized official venue schema")
                dates.append({"date": date, "venues": sorted(venues or {}),
                    "status": "venue_confirmed" if venues else "venue_unresolved"})
            record.update(status="fetched_parsed", date_candidates=dates,
                denominator_status="candidate_dates_only; not an exhaustive historical meeting census")
        except (ValueError, KeyError, TypeError) as exc:
            record["parse_error"] = str(exc)
        return record
    json_match = DAILY_JSON.fullmatch(urlsplit(response.url).path)
    if json_match:
        record = {"family": "trackwork", "parser_version": PARSER_VERSION,
                  "horse_ids": [], "tables": [], "status": "fetched_unparsed"}
        try:
            data = json.loads(source)
            records = data["Records"]
            page = int(dict(parse_qsl(urlsplit(response.url).query))["PageNum"])
            if not isinstance(records, list) or type(data.get("next")) is not int or type(data.get("page")) is not int or data["page"] != page:
                raise ValueError("Invalid official daily JSON pagination/schema")
            if data["next"] and not page < data["next"] <= 5000:
                raise ValueError("Nonadvancing or excessive daily JSON pagination")
            date = datetime.strptime(json_match[1], "%Y%m%d").date().isoformat()
            events = []
            for values in records:
                if not isinstance(values, dict) or not isinstance(values.get("Horse"), str) or not values["Horse"] or not isinstance(values.get("Type"), str) or not values["Type"]:
                    raise ValueError("Invalid daily workout row")
                events.append({"horse_id": None, "horse_name": values["Horse"], "event_date": date,
                    "identity_status": "official_name_only; full_identity_confirmation_required",
                    "published_at": None, "availability_status": "historical_publication_unverified",
                    "date_provenance": "official_daily_resource_path", "values": values})
            record.update(events=events, identity_unresolved_count=len(events), next_page=data["next"],
                          status="fetched_parsed" if events else "parsed_empty_table")
        except (ValueError, KeyError, TypeError) as exc:
            record["parse_error"] = str(exc)
        return record
    query = {k.lower(): v for k, v in parse_qsl(urlsplit(response.url).query)}
    record = {"family": kind, "parser_version": PARSER_VERSION,
              "horse_ids": sorted(set(HORSE_ID.findall(source))),
              "tables": extract_tables(response), "status": "fetched_unparsed"}
    if kind == "sectionals":
        try:
            record.update(events=parse_sectionals(response, query), status="fetched_parsed")
        except (ValueError, TypeError) as exc:
            record["parse_error"] = str(exc)
        return record
    if kind in {"trackwork", "veterinary", "movements", "barrier_trials"}:
        if query.get("horseid"):
            profile = parse_official_profile(response, query["horseid"])
            if profile.brand_code and not displayed_brand_matches(profile.brand_code, query["horseid"]):
                record["parse_error"] = "Displayed event horse differs from requested identity"
                return record
        events, recognized = parse_events(response, kind, query)
        if query.get("horseid") and profile.horse_name and profile.brand_code:
            for event in events:
                event["horse_name"] = profile.horse_name
                event["identity_evidence"] = "requested_full_id_and_displayed_brand"
        if kind == "barrier_trials":
            requested = event_date(query.get("date", ""))
            displayed = response.css("select option[selected]::text").getall()
            visible = " ".join(response.xpath("//body//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::select)]").getall())
            verified = requested and (requested in [event_date(v) for v in displayed] or
                datetime.fromisoformat(requested).strftime("%d/%m/%Y") in visible)
            if not verified:
                record["parse_error"] = "Requested trial date not verified in displayed content"
                return record
        record["events"] = events
        if recognized:
            record["status"] = "fetched_parsed" if events else "parsed_empty_table"
        elif "No information is found" in source:
            record["status"] = "source_reports_no_information"
    if kind == "horse" and query.get("horseid"):
        data = parse_official_profile(response, query["horseid"])
        if data.brand_code and not displayed_brand_matches(data.brand_code, query["horseid"]):
            record["parse_error"] = "Displayed horse brand differs from requested full identity"
        elif data.horse_name or data.form_records:
            attributes = {}
            for table in record["tables"]:
                for row in table:
                    if len(row) == 3 and row[1] == ":" and row[0]:
                        attributes.setdefault(row[0], set()).add(row[2])
            record["horse"] = data.serializable() | {"canonical_brand": query["horseid"].rsplit("_", 1)[-1].upper(),
                "profile_attributes": {label: next(iter(values)) for label, values in attributes.items() if len(values) == 1},
                "profile_attribute_conflicts": {label: sorted(values) for label, values in attributes.items() if len(values) > 1},
                "profile_attributes_policy": "source labels at capture; snapshot-only unless separately validated for historical use",
                "registration_status_at_capture": "inactive" if re.search(r"\((?:Retired|Deregistered)\)", response.css("span.title_text").xpath("string()").get(""), re.I) else "unspecified"}
            record["status"] = "fetched_parsed"
    if kind == "results":
        try:
            from dataclasses import asdict
            date, venue, race_no = verify_race_context(response, query)
            race = parse_race(result_parser_input(response))
            if race.distance is None:
                raise ValueError("Race distance missing from displayed race conditions")
            record["race"] = asdict(race) | {"race_date": date,
                "venue": venue, "race_no": race_no, "runners": []}
            for table in response.css("table"):
                rows = table.xpath("./tr|./thead/tr|./tbody/tr")
                if not rows:
                    continue
                header = [" ".join(" ".join(c.xpath(".//text()").getall()).split()) for c in rows[0].xpath("./th|./td")]
                if len(header) not in (11, 12) or header[:2] != ["Pla.", "Horse No."]:
                    continue
                for row in rows[1:]:
                    cells = [" ".join(" ".join(c.xpath(".//text()").getall()).split()) for c in row.xpath("./td")]
                    if len(header) == 11 and len(cells) == 11:
                        cells.insert(9, "")
                    if len(cells) < 12 or not cells[1].isdigit():
                        continue
                    status = cells[0].upper()
                    placing = re.fullmatch(r"(\d+)(?:\s*DH)?", status)
                    if not placing and status not in NONFINISHERS:
                        continue
                    ids = set(HORSE_ID.findall(row.xpath("./td")[2].get()))
                    horse = ids.pop().upper() if len(ids) == 1 else None
                    runner = {"place": int(placing[1]) if placing else None,
                        "finishing_status": "FINISHED" if placing else status,
                        "horse_no": int(cells[1]),
                        "horse_name": " ".join(row.xpath("./td")[2].css("a::text").get("").split()) or cells[2],
                        "horse_page_id": horse, "horse_code": horse.rsplit("_", 1)[-1] if horse else None,
                        "jockey": cells[3], "trainer": cells[4], "actual_weight": optional_number(cells[5]),
                        "declared_weight": optional_number(cells[6]), "draw": optional_number(cells[7]),
                        "lengths_behind": cells[8], "running_position": cells[9],
                        "finish_time": cells[10] if placing or status == "DISQ" else None,
                        "win_odds": optional_number(cells[11])}
                    record["race"]["runners"].append(runner)
            record["status"] = "fetched_parsed"
            incidents = []
            for table in response.css("table"):
                rows = table.xpath("./tr|./thead/tr|./tbody/tr")
                if not rows:
                    continue
                header = [" ".join(" ".join(c.xpath(".//text()").getall()).split()) for c in rows[0].xpath("./th|./td")]
                if header != ["Pla.", "Horse No.", "Horse", "Incident"]:
                    continue
                for row in rows[1:]:
                    cells = row.xpath("./td")
                    ids = HORSE_ID.findall(row.get())
                    if len(cells) == 4 and len(set(ids)) == 1:
                        incidents.append({"horse_id": ids[0].upper(), "event_date": date,
                            "published_at": None, "availability_status": "historical_publication_unverified",
                            "event_family": "incidents", "values": {"details": " ".join(" ".join(cells[3].xpath(".//text()").getall()).split())}})
            record["events"] = incidents
        except (ValueError, KeyError) as exc:
            record["parse_error"] = str(exc)
    return record


class ScopeMiddleware:
    def process_request(self, request):
        if request.url == f"https://{HOST}/robots.txt":
            return None
        if canonical_url(request.url) is None:
            raise IgnoreRequest("Outside public HKJC racing allowlist")


class CorpusSpider(scrapy.Spider):
    name = "hkjc_official_corpus"
    allowed_domains = [HOST]

    def __init__(self, seeds, output, retry_scope_errors=False, collector_revision="unknown", **kwargs):
        super().__init__(**kwargs)
        self.seed_urls = seeds
        self.retry_scope_errors = retry_scope_errors
        self.collector_revision = collector_revision
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "inventory.sqlite")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
              url TEXT PRIMARY KEY, status TEXT, family TEXT, body_hash TEXT,
              fetched_at TEXT, metadata_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS links (
              source TEXT, destination TEXT, method TEXT,
              PRIMARY KEY(source,destination,method));
            CREATE TABLE IF NOT EXISTS captures (
              id INTEGER PRIMARY KEY AUTOINCREMENT, requested_url TEXT, source_url TEXT,
              status TEXT, body_hash TEXT, fetched_at TEXT, metadata_json TEXT);
        """)

    async def start(self):
        pending_streams = self.db.execute("""
            SELECT links.source, links.destination FROM links
            LEFT JOIN pages ON pages.url = links.destination
            WHERE pages.url IS NULL AND links.method IN
              ('official_json_next','verified_public_script_endpoint')
        """).fetchall()
        for source, destination in pending_streams:
            if canonical_url(destination) and DAILY_JSON.fullmatch(urlsplit(destination).path):
                yield scrapy.Request(destination, callback=self.parse, errback=self.failed,
                    dont_filter=True, priority=200000, meta={"discovered_from": source})
        if self.retry_scope_errors:
            for url, in self.db.execute("SELECT url FROM pages WHERE status IN ('request_error','scope_denied')"):
                if canonical_url(url):
                    yield scrapy.Request(url, callback=self.parse, errback=self.failed, dont_filter=True, priority=10000)
        for url in self.seed_urls:
            safe = canonical_url(url)
            if safe:
                pdf = urlsplit(safe).path == MOVEMENT_PDF
                urgent = (urlsplit(safe).path == DATE_LIST or DAILY_JSON.fullmatch(urlsplit(safe).path)
                          or urlsplit(safe).path.endswith("/trackworkonedayresult"))
                captured = self.db.execute("SELECT 1 FROM pages WHERE url=?", (safe,)).fetchone()
                query = {k.lower(): v for k, v in parse_qsl(urlsplit(safe).query)}
                meeting_lead = family(safe) == "results" and query.get("racedate") and not query.get("raceno")
                trial_lead = family(safe) == "barrier_trials" and event_date(query.get("date", ""))
                # Fresh high-impact seeds must not wait behind an older persisted frontier.
                yield scrapy.Request(safe, callback=self.parse, errback=self.failed,
                    priority=100000 if urgent or pdf else 90000 if trial_lead else 75000 if meeting_lead else 50000 if family(safe) == "results" else 0,
                    dont_filter=bool(pdf or (urgent or trial_lead or meeting_lead) and not captured))

    def failed(self, failure):
        url = failure.request.url
        record = {"source_url": url, "status": "scope_denied" if isinstance(failure.value, IgnoreRequest) else "request_error", "error": str(failure.value),
                  "family": family(url), "fetched_at": now()}
        self.store(record)

    def store(self, record):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?)",
                (record["source_url"], record["status"], record["family"],
                 record.get("body_hash"), record["fetched_at"], json.dumps(record)))
            self.db.execute("INSERT INTO captures(requested_url,source_url,status,body_hash,fetched_at,metadata_json) VALUES (?,?,?,?,?,?)",
                (record.get("requested_url",record["source_url"]),record["source_url"],record["status"],
                 record.get("body_hash"),record["fetched_at"],json.dumps(record)))
        counts = dict(self.db.execute("SELECT status,count(*) FROM pages GROUP BY status"))
        report = {"updated_at": now(), "pages": sum(counts.values()), "statuses": counts,
                  "families": dict(self.db.execute("SELECT family,count(*) FROM pages GROUP BY family"))}
        temporary = self.output / "status.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(self.output / "status.json")

    def parse(self, response):
        if canonical_url(response.url) is None:
            return
        digest = hashlib.sha256(response.body).hexdigest()
        blob = self.output / "raw" / f"{digest}.html.gz"
        blob.parent.mkdir(exist_ok=True)
        if not blob.exists():
            temporary = blob.with_suffix(".tmp")
            with gzip.open(temporary, "wb") as handle:
                handle.write(response.body)
            temporary.replace(blob)
        original_url = response.meta.get("redirect_urls", [response.request.url])[0]
        record = {"source_url": response.url, "requested_url": original_url,
                  "requested_family": family(original_url),
                  "redirect_chain": response.meta.get("redirect_urls", []) + [response.url],
                  "referrer": response.request.meta.get("discovered_from"),
                  "official_source": "official:hkjc", "http_status": response.status,
                  "collector_revision": self.collector_revision,
                  "fetched_at": now(), "body_hash": digest, "raw_path": str(blob),
                  "response_dates": {key: response.headers.get(key, b"").decode(errors="replace")
                                     for key in ("Date", "Last-Modified")},
                  "content_type": response.headers.get("Content-Type", b"").decode(errors="replace")}
        if response.status != 200:
            record.update(status="denied" if response.status in (401,403,429) else "http_error",
                          family=family(response.url))
            self.store(record)
            if response.status in (401,403,429):
                self.crawler.engine.close_spider(self, reason="access_or_rate_denied")
            return
        allowed_pdf = urlsplit(response.url).path == MOVEMENT_PDF and "pdf" in record["content_type"].lower()
        if not allowed_pdf and not any(value in record["content_type"].lower() for value in ("html", "json")):
            record.update(status="unsupported_content", family=family(response.url))
            self.store(record)
            return
        record.update(parse_document(response))
        url_hash = hashlib.sha256(response.url.encode()).hexdigest()[:20]
        normalized = self.output / "documents" / f"{url_hash}-{digest}-{PARSER_VERSION}.json"
        normalized.parent.mkdir(exist_ok=True)
        if normalized.exists():
            previous = json.loads(normalized.read_text())
            record["first_fetched_at"] = previous.get("first_fetched_at", previous["fetched_at"])
        else:
            record["first_fetched_at"] = record["fetched_at"]
        first_capture = self.db.execute(
            "SELECT min(fetched_at) FROM captures WHERE source_url=? AND body_hash=?",
            (response.url, digest)).fetchone()[0]
        if first_capture:
            record["first_fetched_at"] = min(record["first_fetched_at"], first_capture)
        temporary = normalized.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(normalized)
        self.store(record)
        if allowed_pdf:
            return
        if urlsplit(response.url).path == DATE_LIST:
            for entry in record.get("date_candidates", []):
                if entry["date"] >= datetime.now(timezone.utc).date().isoformat():
                    continue
                for venue in entry["venues"]:
                    target = ROOT + "localresults?" + urlencode({"RaceDate": entry["date"].replace("-", "/"),
                        "Racecourse": venue, "RaceNo": 1})
                    with self.db:
                        self.db.execute("INSERT OR IGNORE INTO links VALUES (?,?,?)", (response.url,target,"official_date_venue"))
                    yield scrapy.Request(target, callback=self.parse, errback=self.failed,
                        meta={"discovered_from": response.url})
            return
        if DAILY_JSON.fullmatch(urlsplit(response.url).path):
            if record.get("next_page"):
                parts = urlsplit(response.url)
                target = urlunsplit(parts._replace(query=urlencode({"PageNum": record["next_page"]})))
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO links VALUES (?,?,?)", (response.url,target,"official_json_next"))
                yield scrapy.Request(target, callback=self.parse, errback=self.failed,
                                     priority=200000, meta={"discovered_from":response.url})
            return
        candidates = [(href,"link") for href in response.css("a::attr(href)").getall()]
        if urlsplit(response.url).path.endswith("/trackworkonedayresult"):
            query = {k.lower():v for k,v in parse_qsl(urlsplit(response.url).query)}
            date = event_date(query.get("oneday", ""))
            if date:
                path = "/racing/information/json/TrackworkOneDayRecords/" + date.replace("-", "") + "1E.aspx"
                if path in response.text:
                    candidates.append(("https://" + HOST + path + "?PageNum=1", "verified_public_script_endpoint"))
        for value in response.css("option::attr(value)").getall():
            if HORSE_ID.fullmatch(value):
                candidates.append((ROOT + "horse?" + urlencode({"horseid":value}), "option"))
            elif value.startswith(("/", "https://")):
                candidates.append((value,"option"))
            elif record["family"] == "barrier_trials" and event_date(value):
                candidates.append((ROOT + "archive/btresult?" + urlencode({"Date":value}), "date_option"))
        for horse in record["horse_ids"]:
            for route in ("horse", "otherhorse", "trackworkresult", "ovehorse", "movementrecords"):
                candidates.append((ROOT + route + "?" + urlencode({"horseid":horse}),"horse_id"))
        if response.meta.get("depth", 0) >= 8:
            candidates = []
        for target, method in candidates:
            target = canonical_url(target,response.url)
            if target is None:
                continue
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO links VALUES (?,?,?)", (response.url,target,method))
            target_query = {k.lower(): v for k, v in parse_qsl(urlsplit(target).query)}
            meeting = record.get("race", {}).get("race_date")
            sibling = meeting and event_date(target_query.get("racedate", "")) == meeting and family(target) in {"results", "sectionals"}
            yield scrapy.Request(target, callback=self.parse, errback=self.failed,
                                 priority=100000 if method == "verified_public_script_endpoint" else 65000 if sibling else 0,
                                 meta={"discovered_from":response.url})

    def closed(self, reason):
        summary = json.loads((self.output / "status.json").read_text()) if (self.output / "status.json").exists() else {}
        (self.output / "closed.json").write_text(json.dumps(summary | {"reason":reason,"closed_at":now()},indent=2))
        self.db.close()
