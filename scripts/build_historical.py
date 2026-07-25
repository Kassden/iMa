from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from ima.historical_sources import (
    enrich_horse_profiles, iter_mexwell_odds, normalize_2008_2009, normalize_2008_2009_dividends,
    normalize_2008_2009_incidents, normalize_2013_2020, normalize_datasetlabs_barriers,
    normalize_datasetlabs_racecards, normalize_gdaley_dividends,
    normalize_gdaley_final_odds, normalize_hrosebaby_barriers, normalize_hrosebaby_comments,
    normalize_hrosebaby_trackwork, normalize_lantanacamara_incidents, normalize_mexwell,
    normalize_mexwell_dividends, normalize_mexwell_horses, normalize_mexwell_sectionals,
    normalize_official_archive, reconcile_sources, validate_canonical,
)


def write_odds_snapshots(source: Path, destination: Path) -> dict:
    rows = 0
    snapshots = set()
    races = set()
    date_min = None
    date_max = None
    with gzip.open(destination, "wt", encoding="utf-8", newline="") as handle:
        first = True
        for chunk in iter_mexwell_odds(source):
            chunk.to_csv(handle, index=False, header=first)
            first = False
            rows += len(chunk)
            snapshots.update(
                zip(
                    chunk["race_date"].astype(str), chunk["venue"],
                    chunk["race_no"], chunk["captured_at"],
                )
            )
            races.update(zip(chunk["race_date"].astype(str), chunk["venue"], chunk["race_no"]))
            chunk_min = chunk["race_date"].min()
            chunk_max = chunk["race_date"].max()
            date_min = chunk_min if date_min is None else min(date_min, chunk_min)
            date_max = chunk_max if date_max is None else max(date_max, chunk_max)
    return {
        "rows": rows, "snapshots": len(snapshots), "races": len(races),
        "date_min": date_min.date().isoformat() if date_min is not None else None,
        "date_max": date_max.date().isoformat() if date_max is not None else None,
    }


def frame_report(frame: pd.DataFrame, date_column: str | None = None) -> dict:
    report = {"rows": len(frame)}
    if "source" in frame:
        report["rows_by_source"] = frame["source"].value_counts().to_dict()
    if date_column and date_column in frame and not frame.empty:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        report["date_min"] = dates.min().date().isoformat() if dates.notna().any() else None
        report["date_max"] = dates.max().date().isoformat() if dates.notna().any() else None
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reconciled iMa historical runner archive")
    parser.add_argument("--third-party", type=Path, default=Path("data/historical/third-party"))
    parser.add_argument("--official", type=Path, default=Path("data/historical/hkjc-2005-2025"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/historical"))
    args = parser.parse_args()
    frames = [
        normalize_mexwell(args.third_party / "mexwell/performances.csv"),
        normalize_2008_2009(args.third_party / "0809/08-09 season.xlsx"),
        normalize_2013_2020(
            args.third_party / "2013-2020/races.csv",
            args.third_party / "2013-2020/performances.csv",
        ),
    ]
    official = normalize_official_archive(args.official)
    if not official.empty:
        frames.append(official)
    canonical, conflicts = reconcile_sources(frames)
    canonical = enrich_horse_profiles(
        canonical,
        args.output / "horse-profiles.csv.gz",
        args.output / "horse-form.csv.gz",
    )
    report = validate_canonical(canonical)
    profile_path = args.output / "horse-profiles.csv.gz"
    form_path = args.output / "horse-form.csv.gz"
    profile_count = (
        len(pd.read_csv(profile_path, usecols=["horse_page_id"]))
        if profile_path.exists() else 0
    )
    form_count = (
        len(pd.read_csv(form_path, usecols=["horse_page_id"]))
        if form_path.exists() else 0
    )
    report["profile_enrichment"] = {
        "source": "official:hkjc-horse-profile-and-form",
        "profiles": profile_count,
        "form_records": form_count,
        "runner_coverage": {
            "horse_age": float(canonical["horse_age"].notna().mean()),
            "horse_country": float(canonical["horse_country"].notna().mean()),
            "horse_type": float(canonical["horse_type"].notna().mean()),
            "horse_gear": float(canonical["gear"].notna().mean()),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(args.output / "runners.csv.gz", index=False, compression="gzip")
    conflicts.to_csv(args.output / "source-conflicts.csv.gz", index=False, compression="gzip")
    additional = args.third_party / "additional"
    dividends = pd.concat([
        normalize_mexwell_dividends(
            args.third_party / "mexwell/all_dividends.csv",
            args.third_party / "mexwell/races.csv",
        ),
        normalize_2008_2009_dividends(args.third_party / "0809/08-09 season.xlsx"),
        normalize_gdaley_dividends(additional / "gdaley-hkracing.zip"),
    ], ignore_index=True)
    sectionals = normalize_mexwell_sectionals(args.third_party / "mexwell/sectional_times.csv")
    horses = normalize_mexwell_horses(args.third_party / "mexwell/horses.csv")
    incidents = pd.concat([
        normalize_2008_2009_incidents(args.third_party / "0809/08-09 season.xlsx"),
        normalize_lantanacamara_incidents(additional / "lantanacamara-2014-2017.zip"),
    ], ignore_index=True)
    final_odds = normalize_gdaley_final_odds(additional / "gdaley-hkracing.zip")
    trackwork = normalize_hrosebaby_trackwork(additional / "hrosebaby-experts.zip")
    barriers = pd.concat([
        normalize_hrosebaby_barriers(additional / "hrosebaby-experts.zip"),
        normalize_datasetlabs_barriers(additional / "datasetlabs-2024-sample.zip"),
    ], ignore_index=True)
    comments = normalize_hrosebaby_comments(additional / "hrosebaby-experts.zip")
    racecards = normalize_datasetlabs_racecards(additional / "datasetlabs-2024-sample.zip")
    dividends.to_csv(args.output / "dividends.csv.gz", index=False, compression="gzip")
    sectionals.to_csv(args.output / "sectionals.csv.gz", index=False, compression="gzip")
    horses.to_csv(args.output / "horse-snapshots.csv.gz", index=False, compression="gzip")
    incidents.to_csv(args.output / "incidents.csv.gz", index=False, compression="gzip")
    final_odds.to_csv(args.output / "legacy-final-odds.csv.gz", index=False, compression="gzip")
    trackwork.to_csv(args.output / "trackwork.csv.gz", index=False, compression="gzip")
    barriers.to_csv(args.output / "barrier-trials.csv.gz", index=False, compression="gzip")
    comments.to_csv(args.output / "runner-comments.csv.gz", index=False, compression="gzip")
    racecards.to_csv(args.output / "racecards.csv.gz", index=False, compression="gzip")
    odds_report = write_odds_snapshots(
        args.third_party / "mexwell/live_odds.csv", args.output / "odds-snapshots.csv.gz"
    )
    report["supplemental_tables"] = {
        "odds_snapshots": odds_report,
        "dividends": frame_report(dividends, "race_date"),
        "sectionals": frame_report(sectionals, "race_date"),
        "horse_snapshots": frame_report(horses, "snapshot_at"),
        "incidents": frame_report(incidents, "race_date"),
        "legacy_final_odds": frame_report(final_odds, "race_date"),
        "trackwork": frame_report(trackwork, "event_date"),
        "barrier_trials": frame_report(barriers, "event_date"),
        "runner_comments": frame_report(comments, "race_date"),
        "racecards": frame_report(racecards, "race_date"),
    }
    (args.output / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
