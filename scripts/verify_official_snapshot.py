"""Independently read back snapshot hashes, probability fields and sampled raw rosters."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from parsel import Selector

from scrapper.official_corpus import HORSE_ID, WITHDRAWN, canonical_url


def check_baseline_roster(runners, baseline_runners, baseline_manifest):
    manifest = json.loads(baseline_manifest.read_text())
    if manifest.get("source_policy") != "HKJC-only; raw replay; no third-party values":
        raise ValueError("Baseline is not an official-only snapshot")
    if hashlib.sha256(baseline_runners.read_bytes()).hexdigest() != manifest["files"]["runners.parquet"]:
        raise ValueError("Baseline runner hash mismatch")
    baseline = pd.read_parquet(baseline_runners)
    if not baseline["source"].eq("official:hkjc-results").all() or baseline.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Invalid baseline source or duplicate runner")
    keys = pd.MultiIndex.from_frame(baseline[["race_id", "horse_id"]])
    missing = keys.difference(pd.MultiIndex.from_frame(runners[["race_id", "horse_id"]]))
    if len(missing):
        raise ValueError(f"Verified baseline runner attrition: {len(missing)} missing keys; sample {list(missing[:5])}")
    return {"baseline_runner_keys_retained": len(baseline),
            "baseline_races_retained": int(baseline["race_id"].nunique())}


def verify(snapshot, baseline_runners=None, baseline_manifest=None):
    if bool(baseline_runners) != bool(baseline_manifest):
        raise ValueError("Baseline runners and manifest must be supplied together")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Snapshot artifact hash mismatch: {name}")
    for name, digest in manifest["code_hashes"].items():
        if hashlib.sha256((snapshot / "code" / Path(name).name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Captured code hash mismatch: {name}")
    runners = pd.read_parquet(snapshot / "runners.parquet")
    features = pd.read_parquet(snapshot / "features.parquet")
    baseline_report = check_baseline_roster(runners, baseline_runners, baseline_manifest) if baseline_runners else {}
    if not runners["source"].eq("official:hkjc-results").all():
        raise ValueError("Nonofficial runner source")
    if not runners["horse_id"].map(lambda v: bool(HORSE_ID.fullmatch(v))).all():
        raise ValueError("Unresolved horse identity")
    if runners.duplicated(["race_id", "horse_id"]).any() or runners.duplicated(["race_id", "horse_no"]).any():
        raise ValueError("Duplicate race runner")
    for column in ("market_probability", "target_probability"):
        if not np.allclose(features.groupby("race_id")[column].sum().to_numpy(), 1):
            raise ValueError(f"Race-normalized probabilities fail: {column}")
    feature_counts = features.groupby("race_id").size()
    source_counts = runners.groupby("race_id").size().reindex(feature_counts.index)
    if not feature_counts.equals(source_counts):
        raise ValueError("Feature preparation retained a partial race")
    if features.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate feature runner")
    lineage = {r["body_hash"]: r for r in json.loads((snapshot / "lineage.json").read_text())}
    meetings = runners[["date", "venue"]].drop_duplicates().sort_values(["date", "venue"])
    sampled = pd.concat([group.iloc[np.linspace(0, len(group) - 1, min(6, len(group)), dtype=int)]
                         for _, group in meetings.groupby("venue")]).sort_values(["date", "venue"])
    receipts = []
    verified_horses = set()
    for _, meeting in sampled.iterrows():
        group = runners[(runners["date"] == meeting["date"]) & (runners["venue"] == meeting["venue"])]
        group = group[group["race_id"] == group["race_id"].iloc[0]].sort_values("horse_no")
        source = group.iloc[0]
        if not canonical_url(source["source_url"]):
            raise ValueError("Sample URL outside official scope")
        record = lineage[source["source_body_hash"]]
        body = gzip.open(record["raw_path"], "rb").read()
        if hashlib.sha256(body).hexdigest() != source["source_body_hash"]:
            raise ValueError("Sample raw hash mismatch")
        selector = Selector(body.decode("utf8"))
        tables = []
        for table in selector.css("table"):
            rows = table.xpath("./tr|./thead/tr|./tbody/tr")
            if rows and len(rows[0].xpath("./td|./th")) in (11, 12) and rows[0].xpath("./td[1]/text()|./th[1]/text()").get("").strip() == "Pla.":
                tables.append(rows)
        if len(tables) != 1:
            raise ValueError("Sample result table ambiguous")
        roster = {}
        for row in tables[0][1:]:
            cells = row.xpath("./td")
            if len(cells) not in (11, 12):
                continue
            status = cells[0].xpath("string()").get().strip().upper()
            saddle = cells[1].xpath("string()").get().strip()
            if not saddle.isdigit() or status in WITHDRAWN:
                continue
            ids = set(HORSE_ID.findall(cells[2].get()))
            if len(ids) != 1:
                raise ValueError("Sample source identity ambiguous")
            roster[int(saddle)] = ids.pop().upper()
        if roster != dict(zip(group["horse_no"], group["horse_id"])):
            raise ValueError(f"Sample raw roster differs: {source['race_id']}")
        verified_horses.update(roster.values())
        receipts.append({"race_id": source["race_id"], "runners": len(roster),
                         "body_hash": source["source_body_hash"]})
    horses = sorted(verified_horses)[:20]
    return {"snapshot": str(snapshot), **baseline_report, "races": int(runners["race_id"].nunique()),
            "runners": len(runners), "feature_races": int(features["race_id"].nunique()),
            "feature_attrition_rows": len(runners) - len(features), "raw_meeting_samples": receipts,
            "identity_samples": horses, "verdict": "passed; sampled raw roster readback, not exhaustive website coverage"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-runners", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    report = verify(args.snapshot, args.baseline_runners, args.baseline_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
