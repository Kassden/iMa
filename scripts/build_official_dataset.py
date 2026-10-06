"""Build a new official-only research snapshot from replayable HKJC raw pages."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from importlib.metadata import version, PackageNotFoundError
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
from scrapy.http import HtmlResponse

from ima.rich_features import prepare_rich_runner_dataset
from scrapper.official_corpus import HORSE_ID, NONFINISHERS, PARSER_VERSION, ROOT, WITHDRAWN, canonical_url, event_date, now, parse_document


def read_verified(path, url):
    safe = canonical_url(url)
    if not safe:
        raise ValueError("Source outside official public racing allowlist")
    body = gzip.open(path, "rb").read()
    response = HtmlResponse(url=safe, body=body, encoding="utf-8")
    record = parse_document(response)
    return record | {"source_url": safe, "raw_path": str(path.resolve()),
                     "body_hash": hashlib.sha256(body).hexdigest()}


def race_rows(document):
    if document.get("status") != "fetched_parsed" or "race" not in document:
        raise ValueError(document.get("parse_error", "Race not parsed"))
    race = document["race"]
    runners = race["runners"]
    if len({r.get("horse_page_id") for r in runners}) != len(runners) or len({r["horse_no"] for r in runners}) != len(runners):
        raise ValueError("Duplicate runner identity or saddle number")
    if not any(r["place"] == 1 for r in runners):
        raise ValueError("No winner")
    if any(not HORSE_ID.fullmatch(r.get("horse_page_id") or "") for r in runners):
        raise ValueError("Unresolved full horse identity; whole race quarantined")
    result_tables = [t for t in document["tables"] if t and t[0][:2] == ["Pla.", "Horse No."] and len(t[0]) in (11, 12)]
    if len(result_tables) != 1:
        raise ValueError("Ambiguous result table")
    starters = [r for r in result_tables[0][1:] if len(r) >= len(result_tables[0][0]) and r[1].isdigit()
                and r[0].upper() not in WITHDRAWN]
    if len(starters) != len(runners):
        raise ValueError("Incomplete runner parsing or nonfinishers; whole race quarantined")
    race_id = f"HKJC:{race['race_date']}:{race['venue']}:R{race['race_no']}"
    rows = []
    for runner in runners:
        rows.append({"date": race["race_date"], "race_id": race_id,
            "race_no": race["race_no"], "venue": race["venue"],
            "distance": race["distance"], "course": race["course"],
            "going": race["going"], "race_class": race["race_class"], "prize": race["prize"],
            "horse_id": runner["horse_page_id"].upper(), "horse_no": runner["horse_no"],
            "horse_name": runner["horse_name"], "result": runner["place"],
            "finishing_status": runner.get("finishing_status", "FINISHED"),
            "win_odds": runner["win_odds"], "finish_time": runner["finish_time"],
            "actual_weight": runner["actual_weight"], "declared_weight": runner["declared_weight"],
            "draw": runner["draw"], "lengths_behind": runner["lengths_behind"],
            "running_position": runner["running_position"], "jockey_name": runner["jockey"],
            "trainer_name": runner["trainer"], "source": "official:hkjc-results",
            "source_url": document["source_url"], "source_body_hash": document["body_hash"],
            "horse_rating": None, "gear": None})
    return rows


def merge_form(ratings, horse, form, document):
    date = event_date(form["date"])
    if not date:
        return
    key = (horse.upper(), date)
    value = {"rating": form.get("rating"), "gear": form.get("gear"),
             "source_url": document["source_url"], "body_hash": document["body_hash"]}
    previous = ratings.get(key, {})
    if previous.get("conflict"):
        return
    if pd.notna(previous.get("rating")) and pd.notna(value["rating"]) and previous["rating"] != value["rating"]:
        ratings[key] = {"rating": None, "gear": None, "conflict": True}
    elif pd.notna(value["rating"]) or not previous:
        ratings[key] = value


def workout_signature(event):
    values = event["values"]
    name = event.get("horse_name") or values.get("Horse")
    fields = (name, values.get("Type"), values.get("Racecourse/Track", values.get("Racecourse_Track")),
              values.get("Workouts"), values.get("Gear"))
    if not all(isinstance(v, str) for v in fields) or not all(fields[:4]):
        return None
    return (event["event_date"], *(" ".join(v.upper().split()) for v in fields))


def resolve_workout_identities(events):

    evidence = {}
    for event in events:
        key = workout_signature(event) if event["family"] == "trackwork" else None
        if key and event.get("horse_id") and event.get("identity_evidence") == "requested_full_id_and_displayed_brand":
            evidence.setdefault(key, {})[event["horse_id"]] = event
    resolved = 0
    for event in events:
        if event["family"] != "trackwork" or event.get("horse_id"):
            continue
        matches = evidence.get(workout_signature(event), {})
        if len(matches) != 1:
            event["identity_status"] = "ambiguous_official_confirmation" if matches else "official_confirmation_missing"
            continue
        horse, confirmed = next(iter(matches.items()))
        event.update(horse_id=horse, identity_status="confirmed_against_official_horse_workout",
                     identity_confirmation={"source_url": confirmed["source_url"],
                                            "source_body_hash": confirmed["source_body_hash"]})
        resolved += 1
    return resolved


def group_workout_observations(events):
    groups = Counter()
    for event in events:
        if event["family"] != "trackwork":
            continue
        signature = workout_signature(event)
        horse = event.get("horse_id")
        if not signature or not horse or not HORSE_ID.fullmatch(horse):
            continue
        # Preserve source observations; equal official descriptions need not prove one physical event.
        key = json.dumps([horse, signature], ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.sha256(key.encode()).hexdigest()
        event["workout_equivalence_key"] = digest
        groups[digest] += 1
    return {"resolved_equivalence_groups": len(groups),
            "resolved_source_observations": sum(groups.values()),
            "repeated_source_observations": sum(count - 1 for count in groups.values()),
            "policy": "Exact full-ID/date/name/type/track/workout/gear equivalence; source observations retained; not proof of unique physical events"}


def merge_country(countries, horse, country, document):
    if not country:
        return
    value = {"country": country.strip().upper(), "source_url": document["source_url"],
             "source_body_hash": document["body_hash"]}
    previous = countries.get(horse)
    if previous and (previous.get("conflict") or previous["country"] != value["country"]):
        countries[horse] = {"country": None, "conflict": True}
    else:
        countries[horse] = value


def build(args):
    if args.output.exists():
        raise ValueError("Immutable snapshot already exists; choose a new output path")
    staging = args.output.with_name(args.output.name + ".building")
    if staging.exists():
        raise ValueError(f"Preserved unfinished build exists: {staging}")
    staging.mkdir(parents=True)
    exclusions, counts, records, lineage, identities, ratings, events = [], Counter(), {}, {}, set(), {}, {}
    profiles, countries = {}, {}
    code_hashes = {}
    (staging / "code").mkdir()
    for path in (Path(__file__), Path("scrapper/official_corpus.py"), Path("ima/rich_features.py"),
                 Path("ima/data.py"), Path("ima/feature_sets.py"),
                 Path("scrapper/historical/results.py"), Path("scrapper/horse_pages.py")):
        body = path.read_bytes()
        code_hashes[str(path)] = hashlib.sha256(body).hexdigest()
        (staging / "code" / path.name).write_bytes(body)
    if args.base_snapshot:
        manifest = json.loads((args.base_snapshot / "manifest.json").read_text())
        if manifest.get("parser_version") != PARSER_VERSION:
            raise ValueError("Base snapshot parser differs; full raw replay required to revalidate race identity")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((args.base_snapshot / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Base snapshot hash mismatch: {name}")
        base = pd.read_parquet(args.base_snapshot / "runners.parquet")
        if not base["source"].eq("official:hkjc-results").all():
            raise ValueError("Base snapshot contains nonofficial runners")
        for race_id, group in base.groupby("race_id", sort=False):
            records[race_id] = group.to_dict("records")
        identities.update(base["horse_id"])
        for row in base.to_dict("records"):
            ratings[(row["horse_id"], row["date"])] = {"rating": row.get("horse_rating"), "gear": row.get("gear"),
                "source_url": row.get("rating_source_url"), "body_hash": row.get("rating_source_body_hash")}
        lineage.update({r["body_hash"]: r for r in json.loads((args.base_snapshot / "lineage.json").read_text())})
        counts["base_races"] = len(records)
    # Cached normalized records identify the source URL; values are replayed from raw HTML.
    cached_candidates = []
    if args.base_snapshot:
        for excluded in json.loads((args.base_snapshot / "exclusions.json").read_text()):
            path = Path(excluded["path"])
            if path.name.startswith("R") and path.name.endswith(".html.gz"):
                cached_candidates.append({"race_date": path.parent.parent.name, "venue": path.parent.name,
                    "race_no": int(path.name.split(".")[0][1:]), "source_url": ROOT + "localresults?" +
                    f"RaceDate={path.parent.parent.name.replace('-', '/')}&Racecourse={path.parent.name}&RaceNo={path.name.split('.')[0][1:]}"})
    else:
        for normalized in sorted(args.archive.glob("normalized/*/*.json")):
            cached_candidates.extend(json.loads(normalized.read_text()))
    for cached in cached_candidates:
            path = args.archive / "raw" / cached["race_date"].replace("/", "-") / cached["venue"] / f"R{cached['race_no']}.html.gz"
            counts["cached_race_candidates"] += 1
            try:
                document = read_verified(path, cached["source_url"])
                rows = race_rows(document)
                records[rows[0]["race_id"]] = rows
                lineage[document["body_hash"]] = {k: document[k] for k in ("source_url", "raw_path", "body_hash")}
                identities.update(r["horse_id"] for r in rows)
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
                exclusions.append({"path": str(path), "reason": str(exc)})
            if counts["cached_race_candidates"] % 500 == 0:
                print(json.dumps(dict(counts) | {"accepted_races": len(records), "excluded": len(exclusions)}), flush=True)
    for normalized in sorted(args.profiles.glob("normalized/*.json")):
        cached = json.loads(normalized.read_text())
        horse = cached.get("horse_page_id", "")
        url = cached.get("source_urls", {}).get("profile_and_form")
        path = args.profiles / "raw" / f"{horse}.html.gz"
        if not url or not HORSE_ID.fullmatch(horse) or not path.exists():
            counts["profile_missing_raw_or_identity"] += 1
            continue
        try:
            document = read_verified(path, url)
            if document.get("status") != "fetched_parsed" or "horse" not in document:
                counts["profiles_unparsed"] += 1
                exclusions.append({"path": str(path), "reason": document.get("parse_error", "Profile not parsed")})
                continue
            profile = document.get("horse", {})
            profiles[(horse, document["body_hash"])] = profile | {"source_url": document["source_url"],
                "source_body_hash": document["body_hash"], "capture_time_status": "unknown_legacy_capture_time",
                "mutable_attributes_policy": "snapshot_only; never substituted into historical races"}
            merge_country(countries, horse, profile.get("country"), document)
            lineage[document["body_hash"]] = {k: document[k] for k in ("source_url", "raw_path", "body_hash")}
            for form in profile.get("form_records", []):
                merge_form(ratings, horse, form, document)
            counts["profiles_replayed"] += 1
            counts["form_records_replayed"] += len(profile.get("form_records", []))
        except (ValueError, OSError) as exc:
            exclusions.append({"path": str(path), "reason": str(exc)})
    for corpus in args.corpus:
        for path in sorted((corpus / "documents").glob("*.json")):
            cached = json.loads(path.read_text())
            portable_raw = corpus / "raw" / f"{cached['body_hash']}.html.gz"
            raw = portable_raw if portable_raw.exists() else Path(cached["raw_path"])
            document = read_verified(raw, cached["source_url"])
            if document["body_hash"] != cached["body_hash"]:
                raise ValueError(f"Corrupt corpus blob: {raw}")
            lineage[document["body_hash"]] = {k: document[k] for k in ("source_url", "raw_path", "body_hash")} | {
                "fetched_at": cached["fetched_at"], "first_fetched_at": cached.get("first_fetched_at", cached["fetched_at"]),
                "collector_revision": cached.get("collector_revision", "legacy_capture_without_collector_manifest")}
            for form in document.get("horse", {}).get("form_records", []):
                merge_form(ratings, document["horse"]["horse_page_id"], form, document)
            if "horse" in document:
                horse = document["horse"]["horse_page_id"]
                profiles[(horse, document["body_hash"])] = document["horse"] | {
                    "source_url": document["source_url"], "source_body_hash": document["body_hash"],
                    "available_at": cached.get("first_fetched_at", cached["fetched_at"]),
                    "mutable_attributes_policy": "snapshot_only; never substituted into historical races"}
                merge_country(countries, horse, document["horse"].get("country"), document)
            if document.get("family") == "results":
                try:
                    rows = race_rows(document)
                    records[rows[0]["race_id"]] = rows
                    identities.update(r["horse_id"] for r in rows)
                except ValueError as exc:
                    exclusions.append({"path": str(raw), "reason": str(exc)})
            for event in document.get("events", []):
                event = event | {"family": event.get("event_family", document["family"]), "source_url": document["source_url"],
                    "source_body_hash": document["body_hash"], "fetched_at": cached["fetched_at"],
                    "available_at": cached.get("first_fetched_at", cached["fetched_at"]),
                    "availability_evidence": "first_observed_public_capture; not historical publication proof"}
                key = json.dumps([event["family"], event["horse_id"], event["event_date"], event.get("batch"), event["values"]], sort_keys=True)
                previous = events.get(key)
                if previous and previous["available_at"] < event["available_at"]:
                    event["available_at"] = previous["available_at"]
                    event["first_capture_evidence"] = previous.get("first_capture_evidence", {
                        "source_url": previous["source_url"], "source_body_hash": previous["source_body_hash"]})
                events[key] = event
    rows = []
    for race in records.values():
        for row in race:
            form = ratings.get((row["horse_id"], row["date"]), {})
            row.update(horse_rating=form.get("rating"), gear=form.get("gear"),
                rating_source_url=form.get("source_url"), rating_source_body_hash=form.get("body_hash"))
            country = countries.get(row["horse_id"], {})
            row.update(horse_country=country.get("country"), country_source_url=country.get("source_url"),
                       country_source_body_hash=country.get("source_body_hash"))
            rows.append(row)
    if not rows:
        raise ValueError("No verified official races; refusing empty snapshot")
    source = pd.DataFrame(rows).sort_values(["date", "race_id", "horse_no"]).reset_index(drop=True)
    source.to_parquet(staging / "runners.parquet", index=False)
    source.to_csv(staging / "runners.csv", index=False)
    pd.DataFrame(profiles.values()).to_json(staging / "profiles.jsonl", orient="records", lines=True)
    print(f"Preparing rich historical features for {len(source)} official runners", flush=True)
    features = prepare_rich_runner_dataset(source)
    # No publication-verified event inputs exist yet. Preserve unknown, not zero workouts.
    unknown = [c for c in features if c.startswith(("trackwork_", "veterinary_", "movements_", "injury_events", "fracture_events", "surgery_events"))]
    unknown += [c for c in ("trials_90d", "days_since_trial", "last_trial_placing", "last_trial_speed",
        "days_since_trackwork", "days_since_veterinary", "days_since_movement", "days_since_hk_arrival") if c in features]
    for column in set(unknown):
        if not column.endswith("_available"):
            features[column] = float("nan")
    features.to_parquet(staging / "features.parquet", index=False)
    features.to_csv(staging / "features.csv", index=False)
    newly_resolved_workouts = resolve_workout_identities(list(events.values()))
    workout_groups = group_workout_observations(list(events.values()))
    pd.DataFrame(events.values()).to_json(staging / "events.jsonl", orient="records", lines=True)
    imported = args.output.parent / "raw-import"
    imported.mkdir(exist_ok=True)
    for digest, record in lineage.items():
        destination = imported / f"{digest}.html.gz"
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(record["raw_path"], temporary)
            if hashlib.sha256(gzip.open(temporary, "rb").read()).hexdigest() != digest:
                raise ValueError(f"Source changed while preserving raw blob: {record['raw_path']}")
            temporary.replace(destination)
        elif hashlib.sha256(gzip.open(destination, "rb").read()).hexdigest() != digest:
            raise ValueError(f"Corrupt imported raw blob: {destination}")
        record["original_raw_path"] = record["raw_path"]
        record["raw_path"] = str(destination.resolve())
        record["capture_time_status"] = "recorded_public_capture; not historical publication proof" if record.get("fetched_at") else "unknown_legacy_capture_time; replay is not publication proof"
    (staging / "lineage.json").write_text(json.dumps(list(lineage.values()), indent=2))
    (staging / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    seeds = [ROOT + "otherhorse?horseid=" + h for h in sorted(identities)]
    (staging / "horse_seeds.json").write_text(json.dumps(seeds, indent=2))
    coverage = {c: {"nonmissing": int(features[c].notna().sum()), "total": len(features),
                    "distinct": int(features[c].nunique(dropna=True))} for c in features}
    event_counts = Counter(event["family"] for event in events.values())
    try:
        pdf_version = version("pypdf")
    except PackageNotFoundError:
        pdf_version = None
    manifest = {"created_at": now(), "source_policy": "HKJC-only; raw replay; no third-party values",
        "parser_version": PARSER_VERSION, "code_hashes": code_hashes,
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "scrapy", "parsel", "lxml")} | {"pypdf": pdf_version},
        "base_snapshot": str(args.base_snapshot) if args.base_snapshot else None,
        "nonfinishers": int(source.get("finishing_status", pd.Series(dtype=str)).isin(NONFINISHERS).sum()),
        "rows": len(source), "races": source["race_id"].nunique(), "horses": source["horse_id"].nunique(),
        "feature_rows": len(features), "feature_races": features["race_id"].nunique(),
        "date_min": source["date"].min(), "date_max": source["date"].max(),
        "counts": dict(counts), "excluded_races_or_pages": len(exclusions),
        "profile_snapshots": len(profiles),
        "country_conflicts": sum(v.get("conflict", False) for v in countries.values()),
        "immutable_profile_predictors": {"horse_country": "official country of origin; conflicts quarantined"},
        "mutable_profile_policy": "age/sex/trainer/current rating/model_features retained only in profiles.jsonl snapshots, not copied into historical runners",
        "events_collected": len(events), "events_used_in_historical_features": 0,
        "daily_workouts_identity_confirmed": newly_resolved_workouts,
        "workout_observation_groups": workout_groups,
        "events_identity_unresolved": sum(not e.get("horse_id") for e in events.values()),
        "events_by_family": dict(event_counts),
        "target_only_columns": ["result", "finishing_status", "finish_time", "finish_seconds", "speed_ratio_raw", "lengths_raw", "late_gain_raw"],
        "market_only_columns": ["win_odds", "place_odds", "market_raw", "market_probability", "place_market_probability"],
        "feature_contract": "Select predictors through a registered pre-race feature schema, never all numeric columns; target_* and current-race outcomes are not fundamental predictors",
        "availability_policy": "Event publication dates unverified: excluded from historical training",
        "features": coverage, "by_year": source.groupby(source["date"].str[:4]).size().to_dict(),
        "benter_gaps": ["proprietary adjustments not reproduced", "historical workout/trial/vet publication unverified",
            "current profile age not substituted into history", "no exhaustive official fixture denominator yet"],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir() if p.is_file()}}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.rename(staging, args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("features", "files")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--base-snapshot", type=Path, help="Hash-verified official snapshot; retry its quarantined races and add newly captured pages")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
