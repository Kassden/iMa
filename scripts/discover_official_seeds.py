"""Bounded public-browser discovery for controls absent from server-rendered HTML."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from parsel import Selector

from scrapper.official_corpus import HORSE_ID, ROOT, canonical_url, event_date, family, now


def result_selector_seeds(options, observed, today):
    parts = urlsplit(observed)
    query = dict(parse_qsl(parts.query))
    if not canonical_url(observed) or family(observed) != "results" or set(query) != {"racedate"}:
        raise ValueError("Result selector did not navigate to the verified date-only route")
    seeds = []
    selected_date = event_date(query["racedate"])
    observed_option = False
    for option in options:
        value = json.loads(option["value"])
        date = event_date(value.get("date", ""))
        venue = value.get("venue")
        if not date or venue not in ("", "ST", "HV"):
            raise ValueError("Unexpected official result selector schema")
        observed_option |= date == selected_date and venue == ""
        if date >= today:
            continue
        parameters = {"racedate": date.replace("-", "/")}
        if venue:
            parameters["Racecourse"] = venue
        seeds.append(urlunsplit(parts._replace(query=urlencode(parameters))))
    if not observed_option:
        raise ValueError("Observed navigation does not match a displayed selector option")
    return list(dict.fromkeys(seeds))


def trial_selector_seeds(options, observed, today):
    parts = urlsplit(observed)
    query = dict(parse_qsl(parts.query))
    if not canonical_url(observed) or family(observed) != "barrier_trials" or set(query) != {"Date"}:
        raise ValueError("Trial selector did not navigate to the verified dated route")
    dates = [event_date(o["value"]) for o in options]
    if not all(dates) or event_date(query["Date"]) not in dates:
        raise ValueError("Trial navigation differs from displayed options")
    return list(dict.fromkeys(urlunsplit(parts._replace(query=urlencode({"Date": date.replace("-", "/")})))
                              for date in dates if date <= today))


def archive_links(profiles, cutoff):
    edges = {}
    inspected = 0
    for path in sorted((profiles / "normalized").glob("*.json")):
        profile = json.loads(path.read_text())
        horse = profile.get("horse_page_id", "")
        if not HORSE_ID.fullmatch(horse):
            continue
        dates = [event_date(f["date"]) for f in profile.get("form_records", [])]
        if not any(date and date < cutoff for date in dates):
            continue
        raw = profiles / "raw" / f"{profile['horse_page_id']}.html.gz"
        if not raw.exists():
            continue
        source = profile.get("source_urls", {}).get("profile_and_form")
        if not source or canonical_url(source) is None:
            continue
        if {k.lower():v for k,v in parse_qsl(urlsplit(source).query)}.get("horseid", "").upper() != horse.upper():
            continue
        body = gzip.open(raw, "rb").read()
        inspected += 1
        for href in Selector(body.decode("utf8")).css("a::attr(href)").getall():
            url = canonical_url(href, source)
            if not url or family(url) != "results":
                continue
            query = {k.lower():v for k,v in parse_qsl(urlsplit(url).query)}
            date = event_date(query.get("racedate", ""))
            if not date or date >= cutoff:
                continue
            edges.setdefault(url, {"destination": url, "date": date, "source_url": source,
                "source_body_hash": hashlib.sha256(body).hexdigest(),
                "horse_id": profile["horse_page_id"], "method": "actual_official_form_race_href"})
    ordered = sorted(edges.values(), key=lambda e: (e["date"], e["destination"]))
    return [e["destination"] for e in ordered], {"captured_at": now(), "inspected_profiles": inspected,
        "before_date": cutoff, "edges": ordered, "method": "cached official raw anchor discovery; no URL guessing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("trackwork", "archive-links", "results", "barrier-trials"), default="trackwork")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--before-date", default="2008-04-02")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Discovery outputs are immutable; choose a new output")
    if args.kind == "archive-links":
        if not args.profiles or not event_date(args.before_date):
            parser.error("archive-links requires --profiles and an ISO --before-date")
        seeds, evidence = archive_links(args.profiles, event_date(args.before_date))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(seeds, indent=2))
        args.output.with_suffix(".evidence.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps({"seed_count": len(seeds), "profiles": evidence["inspected_profiles"]}), flush=True)
        return
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        if args.kind in {"results", "barrier-trials"}:
            source = ROOT + ("localresults" if args.kind == "results" else "btresult")
            page.goto(source, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_function("document.querySelector('#selectId')?.options.length > 0", timeout=30000)
            options = page.locator("#selectId option").evaluate_all("els => els.map(e => ({value:e.value,text:e.text}))")
            today = datetime.now(timezone.utc).date().isoformat()
            if args.kind == "results":
                chosen = next(o for o in options if event_date(json.loads(o["value"])["date"]) < today and
                              json.loads(o["value"])["venue"] == "")
            else:
                chosen = min((o for o in options if event_date(o["value"]) and event_date(o["value"]) <= today),
                             key=lambda o: event_date(o["value"]))
            page.locator("#selectId").select_option(chosen["value"])
            page.locator("#submitBtn").click()
            page.wait_for_url("**/" + ("localresults" if args.kind == "results" else "btresult") + "?**", timeout=30000)
            observed = canonical_url(page.url)
            seeds = (result_selector_seeds if args.kind == "results" else trial_selector_seeds)(options, observed or "", today)
            evidence = {"captured_at": now(), "source_url": source, "observed_navigation": observed,
                "selected_option": chosen, "options": options,
                "method": "actual public date-selector Search click; no guessed venue",
                "policy": "Selector dates are leads; verify displayed date and source context on every fetched page"}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(seeds, indent=2))
            args.output.with_suffix(".evidence.json").write_text(json.dumps(evidence, indent=2))
            browser.close()
            print(json.dumps({"seed_count": len(seeds), "evidence": str(args.output.with_suffix('.evidence.json'))}), flush=True)
            return
        source = ROOT + "trackworksearch"
        page.goto(source, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function("document.querySelector('#oneDay')?.options.length > 0", timeout=30000)
        options = page.locator("#oneDay option").evaluate_all("els => els.map(e => ({value:e.value,text:e.text}))")
        options = [o for o in options if event_date(o["value"])]
        if not options:
            raise ValueError("No official dated trackwork options rendered")
        page.locator("#oneDay").select_option(options[0]["value"])
        page.locator(".submit").click()
        page.wait_for_url("**/trackworkonedayresult?**", timeout=30000)
        observed = canonical_url(page.url)
        if not observed:
            raise ValueError("Public control navigated outside official racing allowlist")
        parts = urlsplit(observed)
        query = dict(parse_qsl(parts.query))
        if query.get("OneDay") != options[0]["value"]:
            raise ValueError("Observed public control does not match selected date")
        seeds = [urlunsplit(parts._replace(query=urlencode(query | {"OneDay": o["value"]}))) for o in options]
        evidence = {"captured_at": now(), "source_url": source, "observed_navigation": observed,
                    "method": "rendered public date selector and actual Search click", "options": options,
                    "seeds": seeds, "historical_retention_claim": "Only displayed dates proven; older retention not inferred"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(seeds, indent=2))
        args.output.with_suffix(".evidence.json").write_text(json.dumps(evidence, indent=2))
        browser.close()
    print(json.dumps({"seed_count": len(seeds), "evidence": str(args.output.with_suffix('.evidence.json'))}), flush=True)


if __name__ == "__main__":
    main()
