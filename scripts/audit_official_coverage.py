"""Read-only coverage audit of captured official documents and immutable snapshots."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from scrapy.http import HtmlResponse

from scrapper.official_corpus import DAILY_JSON, PARSER_VERSION, canonical_url, event_date, family, now, parse_document


FAMILIES = ("results", "horse", "trackwork", "barrier_trials", "veterinary",
            "veterinary_clearance", "movements", "sectionals", "incidents")


def observed_race_numbers(response, race):
    numbers = {race["race_no"]}
    for href in response.css("a::attr(href)").getall():
        url = canonical_url(href, response.url)
        if not url or family(url) != "results":
            continue
        query = {k.lower(): v for k, v in parse_qsl(urlsplit(url).query)}
        number = query.get("raceno", "")
        if (number.isdigit() and int(number) > 0 and event_date(query.get("racedate", "")) == race["race_date"]
                and query.get("racecourse", "").upper() == race["venue"]):
            numbers.add(int(number))
    return sorted(numbers)


def load_documents(roots, reparse=False):
    for root in roots:
        for path in (root / "documents").glob("*.json"):
            document = json.loads(path.read_text())
            if reparse:
                url = canonical_url(document["source_url"])
                if not url:
                    raise ValueError(f"Nonofficial corpus source: {path}")
                raw = root / "raw" / f"{document['body_hash']}.html.gz"
                body = gzip.open(raw, "rb").read()
                if hashlib.sha256(body).hexdigest() != document["body_hash"]:
                    raise ValueError(f"Corrupt raw blob: {raw}")
                response = HtmlResponse(url, body=body, encoding="utf8")
                document.update(parse_document(response))
                if document.get("status") == "fetched_parsed" and document.get("race"):
                    document["observed_race_numbers"] = observed_race_numbers(response, document["race"])
            yield document


def season(date):
    year = int(date[:4])
    start = year if int(date[5:7]) >= 9 else year - 1
    return f"{start}-{start + 1}"


def audit(documents, snapshot=None):
    # Keep one latest capture per URL; old parser revisions are not extra coverage.
    latest = {}
    for document in documents:
        key = document["source_url"]
        if key not in latest or document.get("fetched_at", "") > latest[key].get("fetched_at", ""):
            latest[key] = document
    groups = defaultdict(Counter)
    streams = defaultdict(dict)
    census, gaps, meetings = {}, [], {}
    for document in latest.values():
        family = document["family"]
        url = document["source_url"]
        query = {k.lower(): v for k, v in parse_qsl(urlsplit(url).query)}
        race = document.get("race", {})
        date = race.get("race_date")
        venue = race.get("venue", "unknown")
        if document["status"] == "fetched_parsed" and race:
            meeting = meetings.setdefault((date, venue), {"observed": set(), "parsed": set(), "sources": set()})
            meeting["parsed"].add(race["race_no"])
            meeting["observed"].update(document.get("observed_race_numbers", [race["race_no"]]))
            meeting["sources"].add(url)
        # A requested event date is not sufficient evidence for accepted coverage.
        group = groups[(family, season(date) if date else "unknown", venue, "all")]
        group["fetched_pages"] += 1
        group[document["status"]] += 1
        for entry in document.get("date_candidates", []):
            census[entry["date"]] = entry
        match = DAILY_JSON.fullmatch(urlsplit(url).path)
        if match:
            page = int(query["pagenum"])
            streams[match[1]][page] = document
        for event in document.get("events", []):
            event_family = event.get("event_family", family)
            event_group = groups[(event_family, season(event["event_date"]), event.get("venue", "unknown"), "all")]
            event_group["parsed_events"] += 1
            if event.get("horse_id"):
                event_group["identity_resolved_events"] += 1
            else:
                event_group["identity_unresolved_events"] += 1
            if event.get("published_at"):
                event_group["publication_evidenced_events"] += 1
            else:
                event_group["publication_unknown_events"] += 1
        if document["status"] in {"fetched_unparsed", "scope_denied", "request_error", "denied", "http_error"}:
            gaps.append({"family": family, "source_url": url, "reason": document["status"],
                         "details": document.get("parse_error", document.get("error"))})
    pagination = []
    for date, pages in sorted(streams.items()):
        cursor, visited = 1, set()
        complete = False
        while cursor in pages and cursor not in visited:
            visited.add(cursor)
            row = pages[cursor]
            if row["status"] not in {"fetched_parsed", "parsed_empty_table"} or "next_page" not in row:
                break
            cursor = row["next_page"]
            if cursor == 0:
                complete = True
                break
        item = {"date": date, "pages_captured": len(pages), "chain_pages": len(visited),
                "complete": complete, "next_missing_or_invalid_page": None if complete else cursor,
                "rows": sum(len(p.get("events", [])) for p in pages.values()),
                "completeness_scope": "captured pagination chain only; source can publish later updates"}
        pagination.append(item)
        if not complete:
            gaps.append({"family": "trackwork", "date": date,
                         "reason": "incomplete_official_json_pagination", "page": cursor})
    if snapshot:
        manifest = json.loads((snapshot / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Snapshot hash mismatch: {name}")
        summary = {k: manifest[k] for k in ("races", "rows", "horses", "date_min", "date_max",
                   "feature_rows", "feature_races", "events_used_in_historical_features")}
        summary["feature_attrition_rows"] = summary["rows"] - summary["feature_rows"]
        summary["benter_gaps"] = manifest["benter_gaps"]
    else:
        summary = None
    for family in FAMILIES:
        if not any(key[0] == family for key in groups):
            gaps.append({"family": family, "reason": "no_captured_family_documents_or_events"})
    matrix = [{"family": f, "season": s, "venue": v, "cohort": c,
               "expected_count": None, "denominator_status": "official_universe_not_established",
               **dict(counts)} for (f, s, v, c), counts in sorted(groups.items())]
    meeting_coverage = []
    for (date, venue), meeting in sorted(meetings.items()):
        missing = sorted(meeting["observed"] - meeting["parsed"])
        meeting_coverage.append({"date": date, "venue": venue,
            "source_observed_race_numbers": sorted(meeting["observed"]),
            "parsed_race_numbers": sorted(meeting["parsed"]), "missing_observed_race_numbers": missing,
            "source_urls": sorted(meeting["sources"]),
            "scope": "observed official meeting navigation only; not exhaustive historical fixture census"})
        if missing:
            gaps.append({"family": "results", "date": date, "venue": venue,
                         "reason": "missing_source_observed_meeting_races", "race_numbers": missing})
    return {"updated_at": now(), "parser_version": PARSER_VERSION,
            "policy": "read-only; no guessed venue/identity/publication; no optimizer mutation",
            "snapshot": summary, "coverage": matrix, "daily_streams": pagination,
            "meeting_coverage": meeting_coverage,
            "date_candidates": list(sorted(census.values(), key=lambda r: r["date"])),
            "date_census_status": "not exhaustive; null venue is unresolved, not no meeting",
            "gaps": gaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reparse-cached", action="store_true", help="Hash-check and reparse raw without modifying captured documents")
    args = parser.parse_args()
    documents = load_documents(args.corpus, args.reparse_cached)
    report = audit(documents, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "groups": len(report["coverage"]),
                      "gaps": len(report["gaps"]), "daily_streams": report["daily_streams"]}, indent=2))


if __name__ == "__main__":
    main()
