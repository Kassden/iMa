"""Forward-only tomorrow market blends from frozen inference and calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ima.modeling import MarketBlend
from ima.pools import CombinationProbability, OrderExponents, rank_pool_combinations
from scripts.evaluate_season_2026 import quote_lookup, ticket_quote

ROOT = Path(__file__).resolve().parents[1] / "artifacts/race-readiness-20261007"
FAMILIES = ("benter_conditional_logit", "boosted", "gaussian_probit", "pool")
KEYS = ["race_id", "horse_no", "horse_id"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_quote_sources(quotes, official_dir):
    """Check parsed prices against their retained public response, not a new fetch."""
    sources = {}
    for quote in quotes.values():
        filename = quote["source_file"]
        if Path(filename).name != filename:
            raise ValueError("Quote source path escapes official directory")
        if filename not in sources:
            sources[filename] = read_json(official_dir / filename)
        event = sources[filename][quote["source_event_index"]]
        variables = event["request_body"]["variables"]
        day, venue, number = quote["meeting_date"], quote["venue"], int(quote["race_no"])
        if (variables.get("date"), variables.get("venueCode"), int(variables.get("raceNo", -1))) != (day, venue, number):
            raise ValueError("Quote request date/venue/race identity mismatch")
        if event.get("http_status") != 200 or urlsplit(quote["source_url"]).hostname != "info.cld.hkjc.com" or event["source_url"] != quote["source_url"]:
            raise ValueError("Quote source is not a successful official response")
        pool_id = quote["pool_id"]
        odds_types = {"WIN": "WIN", "PLACE": "PLA", "QIN": "QIN", "QPL": "QPL", "TRIO": "TRI", "TIERCE": "TCETop", "FIRST4": "FF", "QUARTET": "QTTTop"}
        if quote["official_odds_type"] != odds_types[quote["pool"]]:
            raise ValueError("Quote pool label differs from official odds type")
        if not re.fullmatch(r"MTG_" + day.replace("-", "") + r"_\d{4}" + re.escape(quote["official_odds_type"]) + r"\d+", pool_id):
            raise ValueError("Returned quote pool date/type identity mismatch")
        pools = [p for meeting in event["response"]["data"]["raceMeetings"] for p in meeting["pmPools"]
                 if p["id"] == pool_id and p["oddsType"] == quote["official_odds_type"]
                 and p["leg"]["races"] == [number]]
        if len(pools) != 1:
            raise ValueError("Returned quote pool/race identity mismatch")
        nodes = [node for node in pools[0]["oddsNodes"] if node["combString"] == quote["combination"]]
        if len(nodes) != 1 or str(nodes[0]["oddsValue"]) != str(quote["odds_value_raw"]):
            raise ValueError("Quote value differs from official raw response")
        try:
            numeric = float(quote["odds_value_raw"])
        except ValueError:
            numeric = None
        if numeric != quote.get("odds_value_numeric"):
            raise ValueError("Parsed numeric quote differs from raw value")
        capture = pd.Timestamp(quote["retrieved_at_utc"])
        if capture.tzinfo is None or capture >= pd.Timestamp(day, tz="Asia/Hong_Kong") + pd.Timedelta(hours=18, minutes=35):
            raise ValueError("Quote capture is not pre-off with an explicit timezone")
    return [official_dir / name for name in sources]


def build_forecast(frames, calibration, scores, quotes, units, *, families=FAMILIES):
    if isinstance(families, str):
        raise ValueError("Families must be a nonempty collection of unique names")
    families = tuple(families)
    if not families or any(not isinstance(name, str) or not name.strip() for name in families) or len(set(families)) != len(families):
        raise ValueError("Families must be a nonempty collection of unique names")
    if set(families) != set(frames):
        raise ValueError("Family names must match frame dictionary keys")
    win_unit = units.get("WIN", {})
    if win_unit.get("displayed_odds_to_D10_factor") != 10 or win_unit.get("displayed_odds_basis") != "gross_return_multiple_per_HKD1":
        raise ValueError("WIN displayed odds basis is not verified gross multiples")
    runners, combinations = [], []
    population = None
    for family in families:
        frame = frames[family].copy()
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate tomorrow runner identity")
        keys = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
        if population is not None and keys != population:
            raise ValueError("Family runner populations differ")
        population = keys
        weights = calibration[family]["blend"]
        if not all(np.isfinite(v) and v >= 0 for v in weights.values()):
            raise ValueError("Invalid frozen blend coefficients")
        blend = MarketBlend(**weights)
        order = OrderExponents(**scores[family + "_market_blend_hindsight"]["order_exponents"])
        if not all(np.isfinite(v) and v > 0 for v in asdict(order).values()):
            raise ValueError("Invalid frozen order exponents")
        for race_id, race in frame.groupby("race_id", sort=True):
            race = race.sort_values("horse_no").copy()
            ids = race.horse_no.astype(int).astype(str).tolist()
            day = str(pd.Timestamp(race.date.iloc[0]).date())
            venue, number = race.venue.iloc[0], int(race.race_no.iloc[0])
            if race_id != f"HKJC:{day}:{venue}:R{number}" or race[["date", "venue", "race_no"]].drop_duplicates().shape[0] != 1:
                raise ValueError("Runner race identity mismatch")
            win_keys = {(day, venue, number, "WIN", (runner,)) for runner in ids}
            available = {key for key in quotes if key[:4] == (day, venue, number, "WIN")}
            if available != win_keys:
                raise ValueError("WIN quote population is not the full runner field")
            pfund = race.model_probability.to_numpy(dtype=float)
            if not np.isfinite(pfund).all() or (pfund <= 0).any() or not np.isclose(pfund.sum(), 1, atol=1e-8):
                raise ValueError("Fundamental field probabilities are invalid")
            raw = []
            for runner in ids:
                quote = quotes[(day, venue, number, "WIN", (runner,))]
                price = ticket_quote(CombinationProbability("WIN", (runner,), 0), race, "WIN", quotes, units)
                if price["quote_status"] != "indicative_snapshot_not_final":
                    raise ValueError("WIN field contains a capped, missing or unit-unverified quote")
                raw.append(float(quote["odds_value_numeric"]))
            pmarket = 1 / np.asarray(raw)
            pmarket /= pmarket.sum()
            combined = blend.transform(pfund, pmarket, race.race_id)
            rankings = rank_pool_combinations(ids, combined, exponents=order)
            pplace = {ticket.runners[0]: ticket.probability for ticket in rankings["PLACE"]}
            ranks = {ticket.runners[0]: rank for rank, ticket in enumerate(rankings["WIN"], 1)}
            for position, row in enumerate(race.itertuples()):
                runner = str(int(row.horse_no))
                price = ticket_quote(CombinationProbability("WIN", (runner,), float(combined[position])), race, "WIN", quotes, units)
                runners.append({"family": family, "model": family, "date": day, "venue": venue,
                    "race_id": race_id, "race_no": number, "horse_no": int(runner), "horse_id": row.horse_id,
                    "horse_name": row.horse_name, "rank": ranks[runner], "pcombined": float(combined[position]),
                    "combinedp": float(combined[position]), "pwin": float(combined[position]),
                    "fundp": float(pfund[position]), "marketp": float(pmarket[position]),
                    "pfund": float(pfund[position]), "pmarket": float(pmarket[position]),
                    "rawWINquote": quotes[(day, venue, number, "WIN", (runner,))]["odds_value_raw"],
                    "pplace": pplace[runner], **price})
            for pool, tickets in rankings.items():
                for rank, ticket in enumerate(tickets[:3], 1):
                    combinations.append({"family": family, "model": family, "date": day, "venue": venue,
                        "race_id": race_id, "race_no": number, "pool": "TRIO" if pool == "TRI" else pool,
                        "rank": rank, "combination": "/".join(ticket.runners), "probability": ticket.probability,
                        "stake_hkd_per_combination": 10, "break_even_dividend_hkd_per_10": 10 / ticket.probability,
                        **ticket_quote(ticket, race, pool, quotes, units)})
    return (pd.DataFrame(runners).sort_values(["family", "race_no", "rank"]).reset_index(drop=True),
            pd.DataFrame(combinations).sort_values(["family", "race_no", "pool", "rank"]).reset_index(drop=True))


def forecast(root=ROOT):
    root = Path(root)
    evaluation, official = root / "evaluation", root / "official"
    provenance_path = evaluation / "provenance.json"
    provenance = read_json(provenance_path)
    prediction_dir = Path(provenance["prediction_dir"])
    if not prediction_dir.exists():
        prediction_dir = root / prediction_dir.name
    receipt_path = prediction_dir / "readback.json"
    receipt = read_json(receipt_path)
    features, metadata_path = root / "query-features.parquet", root / "query-metadata.parquet"
    if sha(features) != provenance["query_features_sha256"] or sha(features) != receipt["input_sha256"] or sha(receipt_path) != provenance["inference_readback_sha256"]:
        raise ValueError("Evaluation or inference receipt is stale for current query features")
    helper_path = Path(__file__).with_name("evaluate_season_2026.py")
    if sha(helper_path) != provenance["evaluation_code_sha256"]:
        raise ValueError("Evaluation helper code changed; wait for final evaluation provenance")
    if set(receipt["models"]) != set(FAMILIES):
        raise ValueError("Trusted receipt does not contain exactly the four frozen families")
    metadata = pd.read_parquet(metadata_path)
    calibration_dates = pd.to_datetime(metadata.loc[metadata.cohort.eq("calibration"), "date"])
    if calibration_dates.empty or not calibration_dates.lt(pd.Timestamp("2026-09-01")).all():
        raise ValueError("Blend calibration cohort is not strictly preseason")
    if any(pd.Timestamp(model["training_cutoff"]) >= calibration_dates.min() for model in receipt["models"].values()):
        raise ValueError("Frozen training cutoff overlaps preseason calibration")
    tomorrow = metadata[metadata.cohort.eq("tomorrow")].copy()
    tomorrow["date"] = pd.to_datetime(tomorrow.date)
    if len(tomorrow) != 108 or tomorrow.duplicated(KEYS).any() or not tomorrow.date.eq(pd.Timestamp("2026-10-07")).all() or not tomorrow.venue.eq("HV").all() or set(tomorrow.race_no) != set(range(1, 10)):
        raise ValueError("Expected exactly 108 Oct7 HV starters across nine races")
    calibration_path, summary_path = evaluation / "calibration.json", evaluation / "summary.json"
    frames = {}
    inputs = [features, metadata_path, provenance_path, receipt_path, calibration_path, summary_path, helper_path,
              official / "odds-quotes.json", official / "odds-unit-semantics.json"]
    for family in FAMILIES:
        prediction_path = prediction_dir / f"{family}.csv"
        if sha(prediction_path) != receipt["models"][family]["output_sha256"]:
            raise ValueError("Frozen prediction hash differs from receipt")
        predictions = pd.read_csv(prediction_path)
        if predictions.duplicated(KEYS).any() or set(map(tuple, predictions[KEYS].itertuples(index=False, name=None))) != set(map(tuple, metadata[KEYS].itertuples(index=False, name=None))):
            raise ValueError("Prediction population differs from query metadata")
        expected = tomorrow.merge(predictions[[*KEYS, "model_probability"]], on=KEYS, validate="one_to_one")
        live_path = evaluation / f"{family}-tomorrow-runners.csv"
        live = pd.read_csv(live_path, parse_dates=["date"])
        actual = live.merge(expected[KEYS + ["model_probability"]], on=KEYS, suffixes=("", "_trusted"), validate="one_to_one")
        if len(live) != 108 or len(actual) != 108 or not np.allclose(actual.model_probability, actual.model_probability_trusted, rtol=1e-12, atol=1e-14):
            raise ValueError("Tomorrow CSV differs from trusted frozen inference")
        if not live.date.eq(pd.Timestamp("2026-10-07")).all() or not live.venue.eq("HV").all():
            raise ValueError("Tomorrow CSV date/venue differs from official query")
        identities = live.merge(tomorrow[KEYS + ["horse_name", "race_no"]], on=KEYS, suffixes=("", "_official"), validate="one_to_one")
        if not identities.horse_name.eq(identities.horse_name_official).all() or not identities.race_no.eq(identities.race_no_official).all():
            raise ValueError("Tomorrow CSV runner names or race numbers differ from official query")
        frames[family] = live
        inputs.extend((prediction_path, live_path))
    quotes, units = quote_lookup(official)
    inputs.extend(verify_quote_sources(quotes, official))
    before = {str(path): sha(path) for path in inputs}
    runners, combinations = build_forecast(frames, read_json(calibration_path), read_json(summary_path), quotes, units)
    if len(runners) != 432 or len(combinations) != 864:
        raise ValueError("Incomplete forecast output population")
    if before != {str(path): sha(path) for path in inputs}:
        raise ValueError("Source inputs changed while forecasting")
    outputs = [evaluation / "tomorrow-market-blends.csv", evaluation / "tomorrow-market-blend-combinations.csv"]
    runners.to_csv(outputs[0], index=False)
    combinations.to_csv(outputs[1], index=False)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": before,
        "ready_for_report": True,
        "evaluation_code_sha256_at_evaluation": provenance["evaluation_code_sha256"],
        "shared_helper_module_sha256_at_forecast": before[str(helper_path)],
        "evaluation_code_matches_current_helper_module": provenance["evaluation_code_sha256"] == before[str(helper_path)],
        "output_sha256": {str(path): sha(path) for path in outputs}, "runner_rows": len(runners),
        "combination_rows": len(combinations), "families": list(FAMILIES),
        "runner_probability_contract": {"combinedp": "frozen market-blended WIN probability", "pwin": "alias of combinedp", "pplace": "top paid positions probability using frozen blended order exponents", "fundp": "trusted frozen fundamental WIN probability", "marketp": "within-race normalized inverse current WIN odds", "pcombined": "alias of combinedp", "pfund": "alias of fundp", "pmarket": "alias of marketp"},
        "calibration": read_json(calibration_path),
        "order_exponents": {family: read_json(summary_path)[family + "_market_blend_hindsight"]["order_exponents"] for family in FAMILIES},
        "policy": "Forward-only frozen preseason blend; within-race normalized inverse current WIN odds. No fitting, historical quote substitution, services or wagers. Snapshot prices are indicative, not final dividends; missing/unverified/capped ticket prices have no point EV."}
    (evaluation / "tomorrow-market-blend-provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = forecast(args.root)
    print(json.dumps({key: report[key] for key in ("runner_rows", "combination_rows", "families")}))


if __name__ == "__main__":
    main()
