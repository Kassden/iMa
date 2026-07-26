from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scrapper.horse_pages import HorsePageClient


HORSE_ID_RE = re.compile(r"horse\?horseid=([^&\" ]+)", re.I)


def discover_horse_page_ids(raw_root: Path) -> list[str]:
    horse_ids = set()
    for path in raw_root.rglob("R*.html.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            horse_ids.update(HORSE_ID_RE.findall(handle.read()))
    return sorted(horse_ids)


def fetch_one(horse_page_id: str, output: Path, retries: int) -> tuple[str, str | None]:
    raw_path = output / "raw" / f"{horse_page_id}.html.gz"
    normalized_path = output / "normalized" / f"{horse_page_id}.json"
    if raw_path.exists() and normalized_path.exists():
        return horse_page_id, None
    for attempt in range(retries + 1):
        try:
            data, source = HorsePageClient(delay_seconds=0).fetch_profile(horse_page_id)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            if not raw_path.exists():
                with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
                    handle.write(source)
            if not normalized_path.exists():
                temporary = normalized_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(data.serializable(), indent=2), encoding="utf-8")
                temporary.replace(normalized_path)
            return horse_page_id, None
        except Exception as exc:  # Network and malformed legacy pages are recorded for retry.
            if attempt == retries:
                return horse_page_id, str(exc)
            time.sleep(1.5 * (attempt + 1))
    return horse_page_id, "unreachable"


def write_tables(output: Path, processed: Path) -> dict:
    profiles = []
    forms = []
    for path in sorted((output / "normalized").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles.append({
            "horse_page_id": data["horse_page_id"],
            "horse_id": data.get("brand_code", "")[-4:] or None,
            "horse_name": data.get("horse_name"),
            "horse_country": data.get("country"),
            "horse_age_at_capture": data.get("age"),
            "horse_colour": data.get("colour"),
            "horse_type": data.get("sex"),
            "fetched_at": data.get("fetched_at"),
            "source_url": data.get("source_urls", {}).get("profile_and_form"),
        })
        for record in data.get("form_records", []):
            forms.append({
                "horse_page_id": data["horse_page_id"],
                "horse_id": data.get("brand_code", "")[-4:] or None,
                "race_date": pd.to_datetime(record.get("date"), dayfirst=True, errors="coerce"),
                "horse_gear": record.get("gear"),
                "rating": record.get("rating"),
                "declared_weight": record.get("declared_weight"),
                "actual_weight": record.get("actual_weight"),
            })
    profile_frame = pd.DataFrame(profiles)
    form_frame = pd.DataFrame(forms)
    if not profile_frame.empty:
        captured = pd.to_datetime(profile_frame["fetched_at"], errors="coerce", utc=True)
        profile_frame["capture_year"] = captured.dt.year
    processed.mkdir(parents=True, exist_ok=True)
    profile_frame.to_csv(processed / "horse-profiles.csv.gz", index=False, compression="gzip")
    form_frame.to_csv(processed / "horse-form.csv.gz", index=False, compression="gzip")
    return {"profiles": len(profile_frame), "form_records": len(form_frame)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill official HKJC horse profile and form pages")
    parser.add_argument("--raw-results", type=Path, default=Path("data/historical/hkjc-2005-2025/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/historical/horse-profiles"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/historical"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    horse_ids = discover_horse_page_ids(args.raw_results)
    if args.limit:
        horse_ids = horse_ids[:args.limit]
    completed = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_one, horse_id, args.output, args.retries): horse_id
            for horse_id in horse_ids
        }
        for future in as_completed(futures):
            horse_id, error = future.result()
            completed += 1
            if error:
                errors.append({"horse_page_id": horse_id, "error": error})
            if completed % 100 == 0 or completed == len(horse_ids):
                print(json.dumps({"completed": completed, "total": len(horse_ids), "errors": len(errors)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    report = {"requested": len(horse_ids), "errors": len(errors), **write_tables(args.output, args.processed)}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
