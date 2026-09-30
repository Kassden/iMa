"""Build an immutable v4 rich-history snapshot with race-dated form ratings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from ima.historical_sources import enrich_horse_profiles
from ima.rich_features import load_full_rich_history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_v4_dataset(source: Path, output: Path) -> dict:
    flat = (source / "runs.csv").exists()
    inputs = ({
        "runs": source / "runs.csv",
        "races": source / "races.csv",
        "runners": source / "runners.csv.gz",
        "profiles": source / "horse-profiles.csv.gz",
        "form": source / "horse-form.csv.gz",
    } if flat else {
        "runs": source / "track/hkracing 2/runs.csv",
        "races": source / "track/hkracing 2/races.csv",
        "runners": source / "data/processed/historical/runners.csv.gz",
        "profiles": source / "data/processed/historical/horse-profiles.csv.gz",
        "form": source / "data/processed/historical/horse-form.csv.gz",
    })
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    if output.exists():
        raise FileExistsError(f"Immutable dataset already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical_path = output.with_name(output.name.replace(".csv.gz", "-canonical.csv.gz"))
    if canonical_path.exists():
        raise FileExistsError(f"Immutable canonical snapshot already exists: {canonical_path}")
    canonical = pd.read_csv(inputs["runners"], low_memory=False)
    canonical = enrich_horse_profiles(canonical, inputs["profiles"], inputs["form"])
    official = canonical["source"].eq("official:hkjc-results")
    recent = pd.to_datetime(canonical["race_date"]).dt.year.ge(2021)
    cohort = canonical.loc[official & recent]
    rating_coverage = float(pd.to_numeric(cohort["rating"], errors="coerce").notna().mean())
    if rating_coverage < 0.85:
        raise ValueError(f"Recent official rating coverage too low: {rating_coverage:.3f}")
    canonical.to_csv(canonical_path, index=False, compression="gzip")
    frame = load_full_rich_history(inputs["runs"], inputs["races"], canonical_path)
    if frame["race_id"].nunique() < 20_000 or len(frame) < 250_000:
        raise ValueError("Rich history is unexpectedly incomplete")
    temporary = output.with_name(output.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": 1,
        "dataset": str(output),
        "dataset_sha256": _sha256(output),
        "canonical_sha256": _sha256(canonical_path),
        "sources": {name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in inputs.items()},
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "recent_official_rating_coverage": rating_coverage,
        "rating_rule": "exact horse_page_id and race_date match to race-dated official form row",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_v4_dataset(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
