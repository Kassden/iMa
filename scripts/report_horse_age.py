from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Report canonical horse-age coverage and provenance")
    parser.add_argument(
        "--runners", type=Path, default=Path("data/processed/historical/runners.csv.gz")
    )
    args = parser.parse_args()

    columns = [
        "race_date", "horse_age", "horse_age_reference_source",
        "horse_age_reference_year", "horse_age_reference_value", "horse_age_year_offset",
        "horse_age_identity_method",
    ]
    frame = pd.read_csv(args.runners, usecols=columns, low_memory=False)
    frame["race_year"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.year
    known = frame[frame["horse_age"].notna()].copy()
    report = {
        "rows": len(frame),
        "known_age_rows": len(known),
        "coverage": float(len(known) / len(frame)),
        "age_distribution": known["horse_age"].value_counts().sort_index().to_dict(),
        "reference_sources": known["horse_age_reference_source"].value_counts().to_dict(),
        "identity_methods": known["horse_age_identity_method"].value_counts().to_dict(),
        "reference_years": known["horse_age_reference_year"].value_counts().sort_index().to_dict(),
        "year_coverage": frame.groupby("race_year")["horse_age"].apply(
            lambda values: float(values.notna().mean())
        ).to_dict(),
        "year_offset_distribution": known["horse_age_year_offset"].value_counts().sort_index().to_dict(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
