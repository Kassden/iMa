"""Official event evidence and cutoff-safe views; capture is not publication."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

HORSE_ID = re.compile(r"HK_\d{4}_[A-Z]\d{3,4}", re.I)
FAMILY_WINDOWS = {"trackwork": (7, 14, 30, 90), "barrier_trials": (90, 180, 365),
                  "veterinary": (30, 90, 365), "veterinary_clearance": (30, 90, 365),
                  "movements": (30, 90, 365), "incidents": (30, 90, 365)}


class HistoricalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    physical_event_group_id: str
    source_observation_id: str
    family: str
    horse_id: str | None
    occurred_at: str | None
    occurred_date: str | None
    occurrence_eligible_at: str | None
    published_at: str | None
    first_seen_at: str | None
    valid_from: str | None = None
    valid_to: str | None = None
    available_at: str | None
    availability_basis: str
    availability_tier: Literal["verified_point_in_time", "observed_prospective", "unknown"]
    identity_status: str
    source_url: str | None
    source_body_sha256: str | None
    parser_version: str
    typed_values: dict[str, Any]
    correction_of: str | None = None
    coverage_id: str | None = None
    physical_identity_verified: bool = False
    exclusion_reasons: tuple[str, ...] = ()


class CoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coverage_id: str
    family: str
    source_url: str
    horse_id: str | None = None
    population_horse_ids: tuple[str, ...] = ()
    start_at: str
    end_at: str
    available_at: str
    status: Literal["complete", "incomplete", "unknown"] = "unknown"
    collector_watermark: str
    missing_pages: tuple[str, ...] = ()
    identity_gaps: int = Field(default=0, ge=0)
    publication_gaps: int = Field(default=0, ge=0)
    physical_identity_gaps: int = Field(default=0, ge=0)


def _present(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and value != ""


def instant(value, *, date_only_next_day=False):
    """Naive official instants are Hong Kong local; return UTC Timestamp."""
    if not _present(value):
        return pd.NaT
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Hong_Kong")
    if date_only_next_day and len(str(value)) == 10:
        stamp += pd.Timedelta(days=1)
    return stamp.tz_convert("UTC")


def _iso(value, **kwargs):
    stamp = instant(value, **kwargs)
    return None if pd.isna(stamp) else stamp.isoformat()


def official_source(url):
    if not isinstance(url, str):
        return False
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname in {"racing.hkjc.com", "racing.hkjc.com.hk", "www.hkjc.com", "hkjc.com", "campaign.hkjc.com"} and parsed.username is None


def normalize_events(records) -> pd.DataFrame:
    """Preserve ambiguous observations; never resolve a name to a full horse ID."""
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")
    normalized = []
    for raw in records:
        if isinstance(raw, HistoricalEvent):
            normalized.append(raw.model_dump())
            continue
        if "occurrence_eligible_at" in raw and "typed_values" in raw:
            raw = dict(raw)
            if isinstance(raw["typed_values"], str):
                raw["typed_values"] = json.loads(raw["typed_values"])
            normalized.append(HistoricalEvent.model_validate(raw).model_dump())
            continue
        values = raw.get("typed_values") or raw.get("values") or {}
        if isinstance(values, str):
            values = json.loads(values)
        values = dict(values)
        for name in ("distance_metres", "trial_finish_seconds", "workout_finish_seconds", "time_seconds", "distance", "distance_unit", "time_unit", "time_basis", "batch_winner_time_seconds", "venue", "track", "arrived_in_conghua", "returned_to_hk"):
            if _present(raw.get(name)):
                values[name] = raw[name]
        family = raw.get("event_family") or raw.get("family") or "unknown"
        horse = str(raw.get("horse_id") or "").upper()
        exact = bool(HORSE_ID.fullmatch(horse))
        identity = (raw.get("identity_status") or "exact_full_id") if exact else "unresolved_identity"
        if any(term in identity for term in ("ambiguous", "unresolved", "missing", "name_only", "required")):
            exact = False
        url = raw.get("source_url")
        digest = raw.get("source_body_sha256") or raw.get("source_body_hash") or raw.get("body_hash")
        reasons = []
        provenance = official_source(url) and isinstance(digest, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", digest))
        if not provenance:
            reasons.append("unverified_official_source")
        if not exact:
            reasons.append("unresolved_identity")
        occurred = raw.get("occurred_at")
        date = raw.get("occurred_date") or raw.get("event_date")
        try:
            eligibility = _iso(occurred or date, date_only_next_day=not _present(occurred))
            published = _iso(raw.get("published_at"), date_only_next_day=True)
            seen = _iso(raw.get("first_seen_at") or raw.get("available_at") or raw.get("first_fetched_at") or raw.get("fetched_at"), date_only_next_day=True)
        except (ValueError, TypeError):
            eligibility = published = seen = None
            reasons.append("invalid_event_time")
        verified = raw.get("publication_verified") is True or raw.get("availability_status") == "verified_point_in_time"
        tier, available, basis = "unknown", None, "publication_and_capture_unknown"
        if published and verified and provenance:
            tier, available, basis = "verified_point_in_time", published, "official_publication_evidence"
        elif seen and provenance:
            tier, available, basis = "observed_prospective", seen, "first_observed_public_capture; not historical publication proof"
        if not eligibility:
            reasons.append("occurrence_unknown")
        observation = raw.get("source_observation_id") or hashlib.sha256(json.dumps(
            [url, digest, family, horse, date, occurred, raw.get("batch"), raw.get("table_index"), values], sort_keys=True, default=str).encode()).hexdigest()
        event_id = raw.get("event_id") or observation
        group = raw.get("physical_event_group_id") or event_id
        normalized.append(HistoricalEvent(event_id=event_id, physical_event_group_id=group,
            source_observation_id=observation, family=family, horse_id=horse if exact else None,
            occurred_at=_iso(occurred) if _present(occurred) and eligibility else None,
            occurred_date=str(date) if _present(date) else None, occurrence_eligible_at=eligibility,
            published_at=published, first_seen_at=seen, available_at=available,
            availability_basis=basis, availability_tier=tier, identity_status=identity,
            source_url=url, source_body_sha256=digest, parser_version=raw.get("parser_version") or "legacy_unspecified",
            typed_values=values, correction_of=raw.get("correction_of"), coverage_id=raw.get("coverage_id"),
            valid_from=_iso(raw.get("valid_from")), valid_to=_iso(raw.get("valid_to")),
            physical_identity_verified=bool(raw.get("physical_identity_verified", False)),
            exclusion_reasons=tuple(reasons)).model_dump())
    by_id = {e["event_id"]: e for e in normalized}
    for event in normalized:
        current, visited = event, set()
        while current.get("correction_of"):
            if current["event_id"] in visited:
                raise ValueError("Event correction cycle")
            visited.add(current["event_id"])
            parent = by_id.get(current["correction_of"])
            if parent is None or parent["horse_id"] != event["horse_id"] or parent["family"] != event["family"]:
                event["exclusion_reasons"] = (*event["exclusion_reasons"], "correction_parent_missing_or_incompatible")
                break
            current = parent
        event["physical_event_group_id"] = current["physical_event_group_id"]
    columns = list(HistoricalEvent.model_fields)
    frame = pd.DataFrame(normalized, columns=columns)
    if not frame.empty:
        identities = pd.DataFrame({"source_observation_id": frame["source_observation_id"],
            "payload": [json.dumps(row, sort_keys=True, default=str) for row in frame.to_dict("records")]})
        conflicting = identities.groupby("source_observation_id")["payload"].nunique().gt(1)
        if conflicting.any():
            raise ValueError("Conflicting source observation identities")
        frame = frame.drop_duplicates("source_observation_id").reset_index(drop=True)
    return frame


def runner_cutoffs(runners):
    if "cutoff_at" in runners:
        return [instant(value) for value in runners["cutoff_at"]]
    return [instant(str(value)[:10]) for value in runners["date"]]


def event_views(events, *, policy="strict", retrospective_lags=None):
    if policy not in {"strict", "assumed_retrospective"}:
        raise ValueError("Unknown availability policy")
    events = normalize_events(events)
    lags = retrospective_lags or {}
    if policy == "strict" and lags:
        raise ValueError("Strict joins cannot apply retrospective lags")
    views = {}
    for row in events.to_dict("records"):
        if row["exclusion_reasons"] or not row["horse_id"] or not row["occurrence_eligible_at"]:
            continue
        if not official_source(row["source_url"]) or not re.fullmatch(r"[0-9a-fA-F]{64}", row["source_body_sha256"] or ""):
            continue
        occurrence = instant(row["occurrence_eligible_at"])
        available = instant(row["available_at"])
        tier = row["availability_tier"]
        if policy == "strict" and tier == "unknown":
            continue
        if policy == "assumed_retrospective" and tier != "verified_point_in_time":
            if row["family"] not in lags:
                continue
            lag = float(lags[row["family"]])
            if not np.isfinite(lag) or lag < 0:
                raise ValueError("Retrospective lag must be finite and nonnegative")
            available = occurrence + pd.Timedelta(days=lag)
            tier = "assumed_retrospective"
        if pd.isna(available):
            continue
        row = dict(row, _occurred=instant(row["occurred_at"] or row["occurred_date"]),
                   _eligible=max(occurrence, available, instant(row["valid_from"]) if row["valid_from"] else occurrence),
                   _available=available, _tier=tier)
        views.setdefault((row["horse_id"], row["family"]), []).append(row)
    for key in views:
        views[key].sort(key=lambda r: (r["_eligible"], r["event_id"]))
    return views


def select_as_of(rows, cutoff):
    versions = {}
    for row in rows:
        if row["_eligible"] >= cutoff:
            break
        key = row["physical_event_group_id"]
        if key not in versions or row["_available"] > versions[key]["_available"]:
            versions[key] = row
    # valid_to is applied only on an already knowable observation, never by a future correction.
    return sorted((row for row in versions.values() if not row["valid_to"] or instant(row["valid_to"]) >= cutoff),
                  key=lambda r: (r["_occurred"], r["event_id"]))


def build_event_features(runners, events, coverage=(), *, policy="strict", retrospective_lags=None):
    """Return index-aligned numeric predictors. Counts require affirmative completeness."""
    views = event_views(events, policy=policy, retrospective_lags=retrospective_lags)
    records = [r if isinstance(r, CoverageRecord) else CoverageRecord.model_validate(r) for r in coverage]
    coverage_by_horse = {}
    for record in records:
        if not official_source(record.source_url):
            continue
        for horse in ((record.horse_id,) if record.horse_id else record.population_horse_ids):
            coverage_by_horse.setdefault((horse, record.family), []).append(record)
    names = []
    for family, windows in FAMILY_WINDOWS.items():
        names.extend([f"{family}_available", f"{family}_usable_observations", f"days_since_{family}", f"{family}_last_distance_metres"])
        for window in windows:
            names.extend([f"{family}_{window}d", f"{family}_{window}d_coverage", f"{family}_{window}d_observed_count"])
    names += ["veterinary_condition_open", "movement_completed_stay_days", "movement_known_stay_days", "movement_days_since_hk_return", "last_trial_placing"]
    output = np.full((len(runners), len(names)), np.nan)
    positions = {name: i for i, name in enumerate(names)}
    cutoffs = runner_cutoffs(runners)
    for index, (horse, cutoff) in enumerate(zip(runners["horse_id"], cutoffs)):
        if pd.isna(cutoff):
            raise ValueError("Unknown race cutoff")
        selected = {}
        for family, windows in FAMILY_WINDOWS.items():
            rows = select_as_of(views.get((horse, family), ()), cutoff)
            selected[family] = rows
            output[index, positions[f"{family}_available"]] = float(bool(rows))
            output[index, positions[f"{family}_usable_observations"]] = len(rows)
            if rows:
                output[index, positions[f"days_since_{family}"]] = (cutoff - rows[-1]["_occurred"]).total_seconds() / 86400
                distance = rows[-1]["typed_values"].get("distance_metres")
                if isinstance(distance, (float, int)) and distance > 0:
                    output[index, positions[f"{family}_last_distance_metres"]] = distance
            for window in windows:
                start = cutoff - pd.Timedelta(days=window)
                count = sum(row["_occurred"] >= start for row in rows)
                complete = any(r.status == "complete" and not r.missing_pages and not r.identity_gaps
                               and not r.publication_gaps and not r.physical_identity_gaps
                               and instant(r.available_at) < cutoff and instant(r.start_at) <= start
                               and instant(r.end_at) >= cutoff for r in coverage_by_horse.get((horse, family), ()))
                output[index, positions[f"{family}_{window}d_coverage"]] = float(complete)
                output[index, positions[f"{family}_{window}d_observed_count"]] = count
                if complete:
                    output[index, positions[f"{family}_{window}d"]] = count
        vet, clearance = selected["veterinary"], selected["veterinary_clearance"]
        if vet:
            output[index, positions["veterinary_condition_open"]] = float(not clearance or clearance[-1]["_occurred"] < vet[-1]["_occurred"])
        movement = selected["movements"]
        if movement:
            values = movement[-1]["typed_values"]
            arrival = instant(values.get("arrived_in_conghua") or values.get("Arrival Date"))
            returned = instant(values.get("returned_to_hk"))
            if pd.notna(arrival) and arrival < cutoff:
                known_end = min(returned, cutoff) if pd.notna(returned) else cutoff
                if known_end >= arrival:
                    output[index, positions["movement_known_stay_days"]] = (known_end - arrival).total_seconds() / 86400
                if pd.notna(returned) and arrival <= returned < cutoff:
                    output[index, positions["movement_completed_stay_days"]] = (returned - arrival).total_seconds() / 86400
                    output[index, positions["movement_days_since_hk_return"]] = (cutoff - returned).total_seconds() / 86400
        trials = selected["barrier_trials"]
        if trials:
            value = trials[-1]["typed_values"].get("placing", trials[-1]["typed_values"].get("Placing"))
            try:
                output[index, positions["last_trial_placing"]] = float(value)
            except (ValueError, TypeError):
                pass
    return pd.DataFrame(output, index=runners.index, columns=names)
