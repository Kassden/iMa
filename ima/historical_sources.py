"""Normalize and reconcile official and third-party historical archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd


CANONICAL_COLUMNS = (
    "race_date", "venue", "race_no", "horse_no", "horse_id", "horse_page_id", "horse_name", "result",
    "win_odds", "actual_weight", "declared_weight", "draw", "finish_time", "going",
    "rating", "jockey_id", "jockey_name", "trainer_id", "trainer_name", "distance",
    "course", "race_class", "prize", "horse_age", "horse_country", "horse_colour",
    "horse_type", "gear", "horse_age_reference_source", "horse_age_reference_year",
    "horse_age_reference_value", "horse_age_year_offset", "horse_age_identity_method", "lengths_behind",
    "running_position", "source",
)

POOL_NAMES = {
    "win": "WIN", "pla": "PLACE", "place": "PLACE", "quinella": "QIN",
    "qpl": "QPL", "tierce": "TCE", "trio": "TRI", "first4": "FF",
    "quartet": "QTT", "double": "DBL",
}


def extract_horse_identity(values: pd.Series) -> pd.DataFrame:
    """Split display names while tolerating archive prefixes such as CE332."""
    horse = values.astype("string").str.extract(r"^(.*)\([A-Z]*([A-Z]\d{3})\)$")
    horse[0] = horse[0].str.strip().fillna(values)
    return horse


def normalize_mexwell(path: Path, start_year: int = 2005, end_year: int = 2012) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    source = source[source["race_country"].eq("HK") & source["race_no"].gt(0)].copy()
    source["race_date"] = pd.to_datetime(source["race_date"]).dt.normalize()
    source = source[source["race_date"].dt.year.between(start_year, end_year)]
    frame = pd.DataFrame({
        "race_date": source["race_date"],
        "venue": source["race_location"].replace({"H": "HV"}),
        "race_no": source["race_no"],
        "horse_no": source["horse_no"],
        "horse_id": source["horse_id"],
        "horse_name": source["horse_name"],
        "result": source["final_placing"],
        "win_odds": source["winning_odds"],
        "actual_weight": source["actual_weight"],
        "declared_weight": source["on_date_weight"],
        "draw": source["draw"],
        "finish_time": source["finish_time"],
        "going": source["going"],
        "rating": source["rating"],
        "jockey_id": source["jockey_id"],
        "jockey_name": source["jockey_name"],
        "trainer_id": source["trainer_id"],
        "trainer_name": source["trainer_name"],
        "distance": source["distance"],
        "course": source["course"],
        "race_class": source["race_class"],
        "gear": source["gears"].replace({"": "NONE", "--": "NONE"}),
        "source": "kaggle:mexwell-hkjc",
    })
    return frame.reindex(columns=CANONICAL_COLUMNS)


def normalize_2013_2020(races_path: Path, performances_path: Path) -> pd.DataFrame:
    races = pd.read_csv(races_path, low_memory=False)
    performances = pd.read_csv(performances_path, low_memory=False)
    source = performances.merge(races, left_on="race_id", right_on="id", validate="many_to_one")
    frame = pd.DataFrame({
        "race_date": pd.to_datetime(source["date"]),
        "venue": source["location"].replace({"Sha Tin": "ST", "Happy Valley": "HV"}),
        "race_no": source["race_no"],
        "horse_no": source["number"],
        "horse_id": source["horse_id"],
        "horse_name": None,
        "result": source["place"],
        "win_odds": source["win_odds"],
        "actual_weight": source["actual_weight"],
        "declared_weight": source["declared_weight"],
        "draw": source["draw"],
        "finish_time": source["finish_time"],
        "going": source["track_condition"],
        "rating": None,
        "jockey_id": source["jockey_id"],
        "jockey_name": None,
        "trainer_id": source["trainer_id"],
        "trainer_name": None,
        "distance": source["distance"],
        "course": source["track_type"],
        "race_class": source["horse_class"],
        "source": "kaggle:jeffreymuller-2013-2020",
    })
    return frame.reindex(columns=CANONICAL_COLUMNS)


def normalize_2008_2009(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name="Sheet1")
    horse = extract_horse_identity(source["Horse"])
    frame = pd.DataFrame({
        "race_date": pd.to_datetime(source["Date (dd/mm/yyyy)"], dayfirst=True),
        "venue": source["Racecourse"].replace({"Sha Tin": "ST", "Happy Valley": "HV"}),
        "race_no": source["Race_no"],
        "horse_no": source["Horse No."],
        "horse_id": horse[1],
        "horse_name": horse[0],
        "result": source["Plc."],
        "win_odds": source["Win Odds"],
        "actual_weight": source["Actual Wt."],
        "declared_weight": source["Declar. Horse Wt."],
        "draw": source["Draw"],
        "finish_time": source["Finish Time"],
        "going": source["Going"],
        "rating": None,
        "jockey_id": None,
        "jockey_name": source["Jockey"],
        "trainer_id": None,
        "trainer_name": source["Trainer"],
        "distance": pd.to_numeric(
            source["Distance"].astype("string").str.replace("M", "", regex=False),
            errors="coerce",
        ),
        "course": source["Course"],
        "race_class": source["Race_class"],
        "source": "third-party:swords-2008-2009",
    })
    return frame.reindex(columns=CANONICAL_COLUMNS)


def iter_mexwell_odds(path: Path, chunksize: int = 5_000) -> Iterator[pd.DataFrame]:
    """Yield long-form WIN/PLACE snapshots without loading millions of rows at once."""
    columns = ["race_date", "race_location", "race_country", "race_no", "data", "capture_time", "last_updated"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["race_country"].eq("HK")]
        records = []
        for row in chunk.itertuples(index=False):
            try:
                pools = json.loads(row.data)
            except (TypeError, json.JSONDecodeError):
                continue
            for source_pool, prices in pools.items():
                pool = POOL_NAMES.get(str(source_pool).lower(), str(source_pool).upper())
                if not isinstance(prices, dict):
                    continue
                for horse_no, odds in prices.items():
                    records.append({
                        "race_date": pd.Timestamp(row.race_date).normalize(),
                        "venue": "HV" if row.race_location == "H" else row.race_location,
                        "race_no": row.race_no,
                        "captured_at": row.capture_time,
                        "pool": pool,
                        "horse_no": pd.to_numeric(horse_no, errors="coerce"),
                        "odds": pd.to_numeric(odds, errors="coerce"),
                        "source_updated_at": row.last_updated,
                        "timestamp_semantics": "source-naive timestamp; timezone unverified",
                        "source": "kaggle:mexwell-hkjc",
                    })
        if records:
            yield pd.DataFrame.from_records(records)


def _mexwell_venue_map(races_path: Path) -> dict[tuple[str, int], str]:
    races = pd.read_csv(races_path, low_memory=False)
    races = races[races["race_country"].eq("HK")].copy()
    races["race_date"] = pd.to_datetime(races["race_date"]).dt.strftime("%Y-%m-%d")
    races["venue"] = races["race_location"].replace({"H": "HV"})
    return {(row.race_date, int(row.race_no)): row.venue for row in races.itertuples()}


def normalize_mexwell_dividends(path: Path, races_path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    source = source[source["race_country"].eq("HK")]
    venues = _mexwell_venue_map(races_path)
    records = []
    for row in source.itertuples(index=False):
        race_date = pd.Timestamp(row.race_date).normalize()
        try:
            pools = json.loads(row.dividends)
        except (TypeError, json.JSONDecodeError):
            continue
        for source_pool, payouts in pools.items():
            if not payouts:
                continue
            pool = POOL_NAMES.get(str(source_pool).lower(), str(source_pool).upper())
            for payout in payouts:
                combination = [str(value).strip() for value in payout.get("combination", [])]
                records.append({
                    "race_date": race_date,
                    "venue": venues.get((race_date.strftime("%Y-%m-%d"), int(row.race_no))),
                    "race_no": row.race_no,
                    "pool": pool,
                    "combination": json.dumps(combination, separators=(",", ":")),
                    "combination_key": "-".join(map(str, combination)),
                    "dividend": pd.to_numeric(payout.get("dividend"), errors="coerce"),
                    "observed_at": row.last_updated,
                    "timestamp_semantics": "post-race dividend",
                    "source": "kaggle:mexwell-hkjc",
                })
    return pd.DataFrame.from_records(records)


def read_archive_csv(path: Path, member: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return pd.read_csv(handle, low_memory=False, **kwargs)


def normalize_gdaley_final_odds(path: Path, start_year: int = 2005) -> pd.DataFrame:
    races = read_archive_csv(path, "races.csv")[["race_id", "date", "venue", "race_no"]]
    runs = read_archive_csv(path, "runs.csv")
    source = runs.merge(races, on="race_id", validate="many_to_one")
    source["race_date"] = pd.to_datetime(source["date"])
    source = source[source["race_date"].dt.year.ge(start_year)]
    win = source[
        ["race_date", "venue", "race_no", "horse_no", "horse_id", "win_odds"]
    ].rename(columns={"win_odds": "odds"})
    win["pool"] = "WIN"
    place = source[
        ["race_date", "venue", "race_no", "horse_no", "horse_id", "place_odds"]
    ].rename(columns={"place_odds": "odds"})
    place["pool"] = "PLACE"
    frame = pd.concat([win, place], ignore_index=True).dropna(subset=["odds"])
    frame["captured_at"] = None
    frame["timestamp_semantics"] = "final/result odds; observation time unavailable"
    frame["source"] = "kaggle:gdaley-hkracing"
    return frame


def normalize_gdaley_dividends(path: Path, start_year: int = 2005) -> pd.DataFrame:
    source = read_archive_csv(path, "races.csv")
    source["race_date"] = pd.to_datetime(source["date"])
    source = source[source["race_date"].dt.year.ge(start_year)]
    records = []
    for row in source.itertuples(index=False):
        for pool, count in (("WIN", 2), ("PLACE", 4)):
            prefix = pool.lower()
            for index in range(1, count + 1):
                selection = getattr(row, f"{prefix}_combination{index}")
                dividend = getattr(row, f"{prefix}_dividend{index}")
                if pd.isna(selection) or pd.isna(dividend):
                    continue
                selection = str(int(selection)) if float(selection).is_integer() else str(selection)
                records.append({
                    "race_date": row.race_date,
                    "venue": row.venue,
                    "race_no": row.race_no,
                    "pool": pool,
                    "combination": json.dumps([selection]),
                    "combination_key": selection,
                    "dividend": dividend,
                    "observed_at": None,
                    "timestamp_semantics": "post-race dividend; observation time unavailable",
                    "source": "kaggle:gdaley-hkracing",
                })
    return pd.DataFrame.from_records(records)


def normalize_hrosebaby_trackwork(path: Path) -> pd.DataFrame:
    source = read_archive_csv(path, "trackwork.csv")
    return pd.DataFrame({
        "horse_id": source["horse_code"],
        "horse_name": source["horse"],
        "event_date": pd.to_datetime(source["date"]),
        "work_type": source["type"],
        "track": source["track"],
        "workout": source["workouts"],
        "gear": source["gear"],
        "source": "kaggle:hrosebaby-experts",
    })


def normalize_hrosebaby_barriers(path: Path) -> pd.DataFrame:
    source = read_archive_csv(path, "barrier.csv")
    horse = extract_horse_identity(source["horse"])
    return pd.DataFrame({
        "horse_id": horse[1],
        "horse_name": horse[0],
        "event_date": pd.to_datetime(source["date"]),
        "venue": source["venue"].replace({"SHA TIN": "ST", "HAPPY VALLEY": "HV"}),
        "trial_no": source["raceno"],
        "placing": source["plc"],
        "draw": source["draw"],
        "distance": source["distance"],
        "going": source["going"],
        "course": source["course"],
        "finish_time": source["time"],
        "result": source["result"],
        "comment": source["comment"],
        "source": "kaggle:hrosebaby-experts",
    })


def normalize_hrosebaby_comments(path: Path) -> pd.DataFrame:
    source = read_archive_csv(path, "comments.csv")
    return pd.DataFrame({
        "race_date": pd.to_datetime(source["date"]),
        "race_no": source["raceno"],
        "horse_no": source["horseno"],
        "placing": source["plc"],
        "gear": source["gear"],
        "comment": source["comment"],
        "comment_zh": source["comment_ch"],
        "source": "kaggle:hrosebaby-experts",
    })


def normalize_lantanacamara_incidents(path: Path) -> pd.DataFrame:
    source = read_archive_csv(path, "race-result-race.csv")
    return pd.DataFrame({
        "race_date": pd.to_datetime(source["race_date"]),
        "venue": source["race_course"].replace({"Sha Tin": "ST", "Happy Valley": "HV"}),
        "race_no": source["race_number"],
        "course": source["track"],
        "race_class": source["race_class"],
        "distance": source["race_distance"],
        "going": source["track_condition"],
        "incident_report": source["incident_report"],
        "source": "kaggle:lantanacamara-2014-2017",
    })


def normalize_datasetlabs_racecards(path: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if "/race_card/" not in member or not member.endswith(".csv"):
                continue
            with archive.open(member) as handle:
                frames.append(pd.read_csv(handle, low_memory=False))
    source = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "race_date": pd.to_datetime(source["race_date"]),
        "venue": source["race_course"],
        "race_no": pd.to_numeric(source["race_no"].astype("string").str.extract(r"(\d+)")[0]),
        "horse_id": source["horse_id"],
        "rating": pd.to_numeric(source["rtg"], errors="coerce"),
        "draw": pd.to_numeric(source["draw"], errors="coerce"),
        "trainer_name": source["trainer"],
        "jockey_name": source["jockey"],
        "actual_weight": pd.to_numeric(source["act_wt"], errors="coerce"),
        "declared_weight": pd.to_numeric(source["declar_horse_wt"], errors="coerce"),
        "gear": source["gear"],
        "distance": pd.to_numeric(
            source["distance"].astype("string").str.replace("M", "", regex=False),
            errors="coerce",
        ),
        "course": source["race_track"],
        "race_class": source["race_class"],
        "going": source["going"],
        "timestamp_semantics": "point-in-time race card; exact capture time unavailable",
        "source": "kaggle:datasetlabs-2024-sample",
    })


def normalize_datasetlabs_barriers(path: Path) -> pd.DataFrame:
    member = "hk-horse_racing-data-kaggle-demo_2024/barrier_trail_2024.csv"
    source = read_archive_csv(path, member)
    return pd.DataFrame({
        "horse_id": source["horse_id"],
        "horse_name": source["horse"],
        "event_date": pd.to_datetime(source["barrier_trial_date"]),
        "venue": source["race_course"],
        "trial_no": source["batch_no"],
        "placing": pd.to_numeric(source["result"], errors="coerce"),
        "draw": source["draw"],
        "distance": source["distance"],
        "going": source["going"],
        "course": source["race_track"],
        "finish_time": source["finish_time"],
        "result": source["result"],
        "comment": source["comment"],
        "source": "kaggle:datasetlabs-2024-sample",
    })


def normalize_2008_2009_dividends(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name="Sheet1")
    source = source[source["Place_dividend"].notna() & source["Horse No."].notna()].copy()
    return pd.DataFrame({
        "race_date": pd.to_datetime(source["Date (dd/mm/yyyy)"], dayfirst=True),
        "venue": source["Racecourse"].replace({"Sha Tin": "ST", "Happy Valley": "HV"}),
        "race_no": source["Race_no"],
        "pool": "PLACE",
        "combination": source["Horse No."].map(lambda value: json.dumps([int(value)])),
        "combination_key": source["Horse No."].astype(int).astype(str),
        "dividend": pd.to_numeric(source["Place_dividend"], errors="coerce"),
        "observed_at": None,
        "timestamp_semantics": "post-race dividend; observation time unavailable",
        "source": "third-party:swords-2008-2009",
    })


def normalize_mexwell_sectionals(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    source = source[source["race_country"].eq("HK")]
    records = []
    for row in source.itertuples(index=False):
        try:
            positions = json.loads(row.sections) if pd.notna(row.sections) else []
            times = json.loads(row.sections_time) if pd.notna(row.sections_time) else []
        except (TypeError, json.JSONDecodeError):
            continue
        for index in range(max(len(positions), len(times))):
            position = positions[index] if index < len(positions) else {}
            timing = times[index] if index < len(times) else {}
            records.append({
                "race_date": pd.Timestamp(row.race_date).normalize(),
                "race_no": row.race_no,
                "horse_id": row.horse_id,
                "horse_no": row.horse_no,
                "horse_name": row.horse_name,
                "section_index": index + 1,
                "section_time": pd.to_numeric(timing.get("time"), errors="coerce"),
                "position": pd.to_numeric(position.get("placing"), errors="coerce"),
                "lengths_behind": position.get("lbw"),
                "source_updated_at": row.last_updated,
                "source": "kaggle:mexwell-hkjc",
            })
    return pd.DataFrame.from_records(records)


def normalize_mexwell_horses(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    source = source[source["horse_country"].eq("HK")].copy()
    source = source.rename(columns={"last_updated": "snapshot_at"})
    source["source"] = "kaggle:mexwell-hkjc"
    return source


def normalize_2008_2009_incidents(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name="Sheet2")
    return pd.DataFrame({
        "race_date": pd.to_datetime(source["Date (dd/mm/yyyy)"], dayfirst=True),
        "venue": source["Racecourse"].replace({"Sha Tin": "ST", "Happy Valley": "HV"}),
        "race_no": source["Race_no"],
        "course": source["Course"],
        "race_class": source["Race_class"],
        "distance": pd.to_numeric(
            source["Distance"].astype("string").str.replace("M", "", regex=False),
            errors="coerce",
        ),
        "going": source["Going"],
        "incident_report": source["Incident_report"],
        "source": "third-party:swords-2008-2009",
    })


def normalize_official_archive(root: Path) -> pd.DataFrame:
    records = []
    for path in root.glob("normalized/*/*.json"):
        for race in json.loads(path.read_text(encoding="utf-8")):
            for runner in race["runners"]:
                records.append({
                    "race_date": pd.Timestamp(race["race_date"]),
                    "venue": race["venue"],
                    "race_no": race["race_no"],
                    "horse_no": runner["horse_no"],
                    "horse_id": runner["horse_code"],
                    "horse_page_id": runner.get("horse_page_id"),
                    "horse_name": runner["horse_name"],
                    "result": runner["place"],
                    "win_odds": runner["win_odds"],
                    "actual_weight": runner["actual_weight"],
                    "declared_weight": runner["declared_weight"],
                    "draw": runner["draw"],
                    "finish_time": runner["finish_time"],
                    "going": race.get("going"),
                    "rating": None,
                    "jockey_id": None,
                    "jockey_name": runner["jockey"],
                    "trainer_id": None,
                    "trainer_name": runner["trainer"],
                    "distance": race.get("distance"),
                    "course": race.get("course"),
                    "race_class": race.get("race_class"),
                    "prize": race.get("prize"),
                    "gear": runner.get("gear"),
                    "lengths_behind": runner.get("lengths_behind"),
                    "running_position": runner.get("running_position"),
                    "source": "official:hkjc-results",
                })
    return pd.DataFrame(records, columns=CANONICAL_COLUMNS)


def enrich_horse_profiles(
    frame: pd.DataFrame,
    profiles_path: Path,
    form_path: Path,
    age_references: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join official profile attributes without leaking current form into old races."""
    enriched = frame.copy()
    if not profiles_path.exists():
        return enriched

    profiles = pd.read_csv(profiles_path, low_memory=False)
    if profiles.empty or "horse_page_id" not in profiles:
        return enriched
    static_columns = [
        "horse_page_id", "horse_id", "horse_country", "horse_colour", "horse_type",
        "horse_age_at_capture", "capture_year",
    ]
    profiles = profiles.reindex(columns=static_columns).drop_duplicates(
        "horse_page_id", keep="last"
    )
    profiles["profile_year"] = pd.to_numeric(
        profiles["horse_page_id"].astype("string").str.extract(r"HK_(\d{4})_")[0],
        errors="coerce",
    )
    enriched["_row_id"] = range(len(enriched))
    enriched = enriched.merge(
        profiles, on="horse_page_id", how="left", suffixes=("", "_profile")
    )
    enriched["horse_age_reference_source"] = enriched[
        "horse_age_reference_source"
    ].astype("string")
    enriched["horse_age_identity_method"] = enriched[
        "horse_age_identity_method"
    ].astype("string")
    for column in ("horse_country", "horse_colour", "horse_type"):
        profile_column = f"{column}_profile"
        if profile_column in enriched:
            enriched[column] = enriched[column].where(
                enriched[column].notna(), enriched[profile_column]
            )
            enriched = enriched.drop(columns=profile_column)

    race_year = pd.to_datetime(enriched["race_date"], errors="coerce").dt.year
    captured_age = pd.to_numeric(enriched.pop("horse_age_at_capture"), errors="coerce")
    capture_year = pd.to_numeric(enriched.pop("capture_year"), errors="coerce")
    projected_age = captured_age - (capture_year - race_year)
    projected_age = projected_age.where(projected_age.between(1, 20))
    existing_age = pd.to_numeric(enriched["horse_age"], errors="coerce")
    direct_reference = existing_age.isna() & projected_age.notna()
    enriched["horse_age"] = existing_age.where(existing_age.notna(), projected_age)
    enriched.loc[direct_reference, "horse_age_reference_source"] = (
        "official:hkjc-horse-profile"
    )
    enriched.loc[direct_reference, "horse_age_reference_year"] = capture_year[direct_reference]
    enriched.loc[direct_reference, "horse_age_reference_value"] = captured_age[direct_reference]
    enriched.loc[direct_reference, "horse_age_year_offset"] = (
        race_year[direct_reference] - capture_year[direct_reference]
    )
    enriched.loc[direct_reference, "horse_age_identity_method"] = "horse-page-id-exact"

    references = profiles.rename(columns={
        "horse_age_at_capture": "age_at_reference",
        "capture_year": "reference_year",
    })[[
        "horse_page_id", "horse_id", "profile_year", "age_at_reference", "reference_year",
    ]].dropna(subset=["age_at_reference", "reference_year"])
    references["reference_priority"] = 2
    references["reference_source"] = "official:hkjc-horse-profile"
    references["reference_identity_method"] = "horse-page-id-exact"
    if age_references is not None and not age_references.empty:
        snapshots = age_references.copy()
        if "horse_country" in snapshots:
            snapshots = snapshots[snapshots["horse_country"].eq("HK")]
        snapshots["age_at_reference"] = pd.to_numeric(snapshots["age"], errors="coerce")
        snapshots["reference_year"] = pd.to_datetime(
            snapshots["snapshot_at"], errors="coerce"
        ).dt.year
        snapshots = snapshots.dropna(subset=["horse_id", "age_at_reference", "reference_year"])
        snapshots["_reference_id"] = range(len(snapshots))
        identity_candidates = snapshots[[
            "_reference_id", "horse_id", "age_at_reference", "reference_year",
        ]].merge(
            profiles[["horse_page_id", "horse_id", "profile_year"]].dropna(),
            on="horse_id", how="inner",
        )
        identity_candidates = identity_candidates[
            identity_candidates["profile_year"].le(identity_candidates["reference_year"])
            & identity_candidates["reference_year"].sub(
                identity_candidates["profile_year"]
            ).le(12)
        ]
        identity_candidates = identity_candidates.sort_values("profile_year").drop_duplicates(
            "_reference_id", keep="last"
        )
        snapshot_references = identity_candidates[[
            "horse_page_id", "horse_id", "profile_year", "age_at_reference", "reference_year",
        ]].copy()
        snapshot_references["reference_priority"] = 1
        snapshot_references["reference_source"] = "kaggle:mexwell-hkjc-horse-snapshot"
        snapshot_references["reference_identity_method"] = "horse-code-profile-cycle"
        references = pd.concat([references, snapshot_references], ignore_index=True)

    missing_age = enriched["horse_age"].isna()
    if missing_age.any() and not references.empty:
        age_rows = enriched.loc[
            missing_age, ["_row_id", "horse_page_id", "horse_id", "race_date"]
        ].copy()
        age_rows["race_year"] = pd.to_datetime(age_rows["race_date"], errors="coerce").dt.year
        direct = age_rows.dropna(subset=["horse_page_id"]).merge(
            references, on="horse_page_id", how="inner", suffixes=("", "_reference")
        )
        direct["identity_method"] = direct["reference_identity_method"]
        fallback = age_rows[age_rows["horse_page_id"].isna()].merge(
            references, on="horse_id", how="inner", suffixes=("", "_reference")
        )
        fallback["identity_method"] = "horse-code-profile-cycle"
        fallback = fallback[
            fallback["profile_year"].le(fallback["race_year"])
            & fallback["race_year"].sub(fallback["profile_year"]).le(12)
        ]
        age_candidates = pd.concat([direct, fallback], ignore_index=True)
        if not age_candidates.empty:
            age_candidates["projected_age"] = age_candidates["age_at_reference"] + (
                age_candidates["race_year"] - age_candidates["reference_year"]
            )
            age_candidates = age_candidates[age_candidates["projected_age"].between(2, 20)]
            age_candidates["reference_distance"] = (
                age_candidates["race_year"] - age_candidates["reference_year"]
            ).abs()
            age_candidates = age_candidates.sort_values(
                ["reference_distance", "reference_priority"], ascending=[True, False]
            ).drop_duplicates("_row_id", keep="first")
            age_values = age_candidates.set_index("_row_id")["projected_age"]
            enriched.loc[missing_age, "horse_age"] = enriched.loc[
                missing_age, "_row_id"
            ].map(age_values).to_numpy()
            selected = age_candidates.set_index("_row_id")
            selected_rows = enriched["_row_id"].map(selected["projected_age"]).notna()
            enriched.loc[selected_rows, "horse_age_reference_source"] = enriched.loc[
                selected_rows, "_row_id"
            ].map(selected["reference_source"]).to_numpy()
            enriched.loc[selected_rows, "horse_age_reference_year"] = enriched.loc[
                selected_rows, "_row_id"
            ].map(selected["reference_year"]).to_numpy()
            enriched.loc[selected_rows, "horse_age_reference_value"] = enriched.loc[
                selected_rows, "_row_id"
            ].map(selected["age_at_reference"]).to_numpy()
            enriched.loc[selected_rows, "horse_age_year_offset"] = enriched.loc[
                selected_rows, "_row_id"
            ].map(selected["race_year"] - selected["reference_year"]).to_numpy()
            enriched.loc[selected_rows, "horse_age_identity_method"] = enriched.loc[
                selected_rows, "_row_id"
            ].map(selected["identity_method"]).to_numpy()

    missing_static = enriched[["horse_country", "horse_colour", "horse_type"]].isna().any(axis=1)
    fallback_rows = enriched.loc[
        missing_static & enriched["horse_id"].notna(), ["_row_id", "horse_id", "race_date"]
    ].copy()
    if not fallback_rows.empty:
        fallback_rows["race_year"] = pd.to_datetime(
            fallback_rows["race_date"], errors="coerce"
        ).dt.year
        fallback_profiles = profiles.dropna(subset=["horse_id", "profile_year"])[
            ["horse_id", "profile_year", "horse_country", "horse_colour", "horse_type"]
        ]
        candidates = fallback_rows.merge(fallback_profiles, on="horse_id", how="inner")
        candidates = candidates[
            candidates["profile_year"].le(candidates["race_year"])
            & candidates["race_year"].sub(candidates["profile_year"]).le(12)
        ]
        candidates = candidates.sort_values("profile_year").drop_duplicates(
            "_row_id", keep="last"
        )
        if not candidates.empty:
            candidate_values = candidates.set_index("_row_id")
            row_ids = enriched["_row_id"]
            for column in ("horse_country", "horse_colour", "horse_type"):
                fallback = row_ids.map(candidate_values[column])
                enriched[column] = enriched[column].where(enriched[column].notna(), fallback)

    if form_path.exists():
        form = pd.read_csv(form_path, low_memory=False)
        if not form.empty and {"horse_page_id", "race_date", "horse_gear"}.issubset(form.columns):
            form["race_date"] = pd.to_datetime(form["race_date"], errors="coerce").dt.normalize()
            form["horse_gear"] = form["horse_gear"].replace({"": "NONE", "--": "NONE"})
            gear_by_code = None
            if "horse_id" in form.columns:
                gear_by_code = (
                    form.dropna(subset=["horse_id", "race_date"])
                    .drop_duplicates(["horse_id", "race_date"], keep="last")
                    [["horse_id", "race_date", "horse_gear"]]
                )
            form_by_page = (
                form.dropna(subset=["horse_page_id", "race_date"])
                .drop_duplicates(["horse_page_id", "race_date"], keep="last")
                [["horse_page_id", "race_date", "horse_gear"]]
            )
            enriched["race_date"] = pd.to_datetime(enriched["race_date"]).dt.normalize()
            enriched = enriched.merge(form_by_page, on=["horse_page_id", "race_date"], how="left")
            enriched["gear"] = enriched["gear"].combine_first(enriched.pop("horse_gear"))

            missing_gear = enriched["gear"].isna() & enriched["horse_id"].notna()
            if missing_gear.any() and gear_by_code is not None:
                fallback_gear = enriched.loc[
                    missing_gear, ["_row_id", "horse_id", "race_date"]
                ].merge(gear_by_code, on=["horse_id", "race_date"], how="left")
                gear_values = fallback_gear.set_index("_row_id")["horse_gear"]
                enriched.loc[missing_gear, "gear"] = enriched.loc[
                    missing_gear, "_row_id"
                ].map(gear_values).to_numpy()

    enriched["gear"] = enriched["gear"].fillna("NONE")

    return enriched.reindex(columns=CANONICAL_COLUMNS)


def reconcile_sources(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    combined = pd.concat(prepared, ignore_index=True).reindex(columns=CANONICAL_COLUMNS)
    combined["race_date"] = pd.to_datetime(combined["race_date"]).dt.normalize()
    combined["horse_no"] = pd.to_numeric(combined["horse_no"], errors="coerce")
    combined = combined.dropna(subset=["race_date", "venue", "race_no"])
    early_archive = combined["race_date"].dt.year.le(2012)
    combined["_runner_key"] = None
    combined.loc[early_archive, "_runner_key"] = combined.loc[early_archive, "horse_id"].map(
        lambda value: f"id:{value}" if pd.notna(value) else None
    )
    missing_key = combined["_runner_key"].isna()
    combined.loc[missing_key, "_runner_key"] = combined.loc[missing_key, "horse_no"].map(
        lambda value: f"no:{int(value)}" if pd.notna(value) else None
    )
    missing_key = combined["_runner_key"].isna()
    combined.loc[missing_key, "_runner_key"] = combined.loc[missing_key, "horse_id"].map(
        lambda value: f"id:{value}" if pd.notna(value) else None
    )
    combined = combined.dropna(subset=["_runner_key"])
    priority = {
        "official:hkjc-results": 3,
        "third-party:swords-2008-2009": 2.5,
        "kaggle:mexwell-hkjc": 2,
        "kaggle:jeffreymuller-2013-2020": 1,
    }
    combined["_priority"] = combined["source"].map(priority).fillna(0)
    keys = ["race_date", "venue", "race_no", "_runner_key"]
    duplicates = combined[combined.duplicated(keys, keep=False)].sort_values(keys + ["_priority"])
    canonical = (
        combined.sort_values("_priority")
        .drop_duplicates(keys, keep="last")
        .drop(columns=["_priority", "_runner_key"])
        .sort_values(["race_date", "venue", "race_no", "horse_no", "horse_id"], na_position="last")
        .reset_index(drop=True)
    )
    return canonical, duplicates.drop(columns=["_priority", "_runner_key"])


def validate_canonical(frame: pd.DataFrame) -> dict:
    race_keys = ["race_date", "venue", "race_no"]
    winners = (
        frame.assign(_winner=pd.to_numeric(frame["result"], errors="coerce").eq(1))
        .groupby(race_keys)["_winner"]
        .sum()
    )
    return {
        "rows": len(frame),
        "races": int(frame.groupby(race_keys).ngroups),
        "date_min": frame["race_date"].min().date().isoformat(),
        "date_max": frame["race_date"].max().date().isoformat(),
        "races_without_winner": int(winners.eq(0).sum()),
        "dead_heat_races": int(winners.gt(1).sum()),
        "missingness": {column: float(frame[column].isna().mean()) for column in CANONICAL_COLUMNS},
        "rows_by_source": frame["source"].value_counts().to_dict(),
        "races_by_year": (
            frame.assign(year=frame["race_date"].dt.year)
            .groupby("year").apply(lambda group: group.groupby(race_keys).ngroups, include_groups=False)
            .to_dict()
        ),
    }
