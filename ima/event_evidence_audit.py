"""Read-only, evidence-aware summaries for historical event rows."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import pandas as pd

_HORSE_ID = re.compile(r"HK_\d{4}_[A-Z]\d{3,4}", re.IGNORECASE)
_PHYSICAL_EVIDENCE_FIELDS = (
    "physical_identity_evidence",
    "physical_grouping_evidence",
    "physical_event_evidence",
    "physical_identity_source",
    "physical_grouping_source",
)
_AUDITED_FIELDS = (
    "horse_id", "identity_status", "source_observation_id", "source_url",
    "source_body_sha256", "event_id", "physical_event_group_id",
    "physical_identity_verified", *_PHYSICAL_EVIDENCE_FIELDS,
    "published_at", "publication_verified", "availability_tier", "first_seen_at",
    "available_at", "first_fetched_at", "fetched_at", "occurred_at", "occurred_date",
)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    if isinstance(value, (float,)) and pd.isna(value):
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _valid_timestamp(value: Any) -> bool:
    if not _present(value):
        return False
    try:
        return not pd.isna(pd.Timestamp(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _rows(events: Any) -> list[dict[str, Any]]:
    if isinstance(events, pd.DataFrame):
        return events.to_dict("records")
    if isinstance(events, Mapping):
        raise TypeError("events must be an iterable of row mappings, not one mapping")
    try:
        values = list(events)
    except TypeError as exc:
        raise TypeError("events must be a dataframe or iterable of row mappings") from exc
    rows = []
    for index, row in enumerate(values):
        if hasattr(row, "model_dump"):
            row = row.model_dump()
        if not isinstance(row, Mapping):
            raise TypeError(f"event row {index} must be a mapping")
        rows.append(dict(row))
    return rows


def _new_counts() -> Counter:
    return Counter({
        "observations": 0,
        "horse_id_exact": 0,
        "horse_id_unresolved_or_missing": 0,
        "source_observation_id_present": 0,
        "source_link_present": 0,
        "source_body_hash_present": 0,
        "physical_group_id_present": 0,
        "physical_identity_verified_true": 0,
        "physical_identity_flag_false_unknown": 0,
        "physical_identity_flag_absent_unknown": 0,
        "physical_identity_flag_malformed_unknown": 0,
        "physical_evidence_present": 0,
        "publication_timestamp_present": 0,
        "publication_timestamp_valid": 0,
        "publication_verification_true": 0,
        "publication_availability_verified": 0,
        "first_capture_timestamp_present": 0,
        "first_capture_timestamp_valid": 0,
        "effective_availability_timestamp_present": 0,
        "effective_availability_timestamp_valid": 0,
        "occurrence_timestamp_present": 0,
        "occurrence_timestamp_valid": 0,
    })


def audit_event_evidence(events: Any) -> dict[str, Any]:
    """Summarize evidence without treating defaults or captures as historical proof.

    Accepts raw rows, normalized ``HistoricalEvent`` rows, or a DataFrame. The
    summary reports field availability and strict boolean evidence separately;
    it does not normalize or mutate event data.
    """
    rows = _rows(events)
    overall = _new_counts()
    families: dict[str, Counter] = defaultdict(_new_counts)
    field_availability = Counter()
    source_rows: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = row.get("family", row.get("event_family", "unknown"))
        family = str(family) if _present(family) else "unknown"
        groups = (overall, families[family])
        for key in _AUDITED_FIELDS:
            if key in row and _present(row[key]):
                field_availability[key] += 1

        horse = row.get("horse_id")
        exact_horse = isinstance(horse, str) and bool(_HORSE_ID.fullmatch(horse.strip()))
        source_id = row.get("source_observation_id")
        source_url = row.get("source_url")
        source_hash = row.get("source_body_sha256", row.get("source_body_hash", row.get("body_hash")))
        physical_id = row.get("physical_event_group_id")
        flag = row.get("physical_identity_verified")
        physical_evidence = any(_present(row.get(name)) for name in _PHYSICAL_EVIDENCE_FIELDS)
        publication = row.get("published_at")
        publication_verified = row.get("publication_verified") is True
        normalized_publication_verified = row.get("availability_tier") == "verified_point_in_time"
        capture = next((row.get(name) for name in ("first_seen_at", "first_fetched_at", "fetched_at") if _present(row.get(name))), None)
        effective_availability = row.get("available_at")
        occurrence = next((row.get(name) for name in ("occurred_at", "occurred_date", "event_date")
                           if _present(row.get(name))), None)

        for counts in groups:
            counts["observations"] += 1
            counts["horse_id_exact" if exact_horse else "horse_id_unresolved_or_missing"] += 1
            counts["source_observation_id_present"] += int(_present(source_id))
            counts["source_link_present"] += int(_present(source_url))
            counts["source_body_hash_present"] += int(_present(source_hash))
            counts["physical_group_id_present"] += int(_present(physical_id))
            counts["physical_evidence_present"] += int(physical_evidence)
            if flag is True:
                counts["physical_identity_verified_true"] += 1
            elif flag is False:
                # False is the HistoricalEvent schema default. Without separate
                # evidence it means unassessed/unknown, not identity rejected.
                counts["physical_identity_flag_false_unknown"] += 1
            elif flag is None:
                counts["physical_identity_flag_absent_unknown"] += 1
            else:
                counts["physical_identity_flag_malformed_unknown"] += 1
            counts["publication_timestamp_present"] += int(_present(publication))
            counts["publication_timestamp_valid"] += int(_valid_timestamp(publication))
            counts["publication_verification_true"] += int(publication_verified)
            counts["publication_availability_verified"] += int(
                _present(publication) and (publication_verified or normalized_publication_verified)
            )
            counts["first_capture_timestamp_present"] += int(_present(capture))
            counts["first_capture_timestamp_valid"] += int(_valid_timestamp(capture))
            counts["effective_availability_timestamp_present"] += int(_present(effective_availability))
            counts["effective_availability_timestamp_valid"] += int(_valid_timestamp(effective_availability))
            counts["occurrence_timestamp_present"] += int(_present(occurrence))
            counts["occurrence_timestamp_valid"] += int(_valid_timestamp(occurrence))
        if _present(source_id) and _present(source_url):
            source_rows[str(source_id)].add(str(source_url))

    def unpack(counter: Counter) -> dict[str, int]:
        return {key: int(value) for key, value in counter.items()}

    return {
        "observations": len(rows),
        "denominators": {"event_rows": len(rows), "source_linked_rows": sum(bool(_present(r.get("source_url"))) for r in rows)},
        "summary": unpack(overall),
        "field_availability": {key: {"present": int(field_availability[key]), "denominator": len(rows)} for key in _AUDITED_FIELDS},
        "source_linkage": {
            "rows_with_source_observation_id_and_url": sum(
                _present(r.get("source_observation_id")) and _present(r.get("source_url")) for r in rows
            ),
            "distinct_source_observation_ids_linked": len(source_rows),
            "distinct_source_urls_linked": len({url for urls in source_rows.values() for url in urls}),
        },
        "physical_grouping": {
            "verified_flag_true": overall["physical_identity_verified_true"],
            "explicit_false_unknown_not_rejected": overall["physical_identity_flag_false_unknown"],
            "flag_absent_unknown": overall["physical_identity_flag_absent_unknown"],
            "malformed_flag_unknown": overall["physical_identity_flag_malformed_unknown"],
            "separate_evidence_payload_present": overall["physical_evidence_present"],
            "interpretation": "Only literal true counts as verified; false/default without evidence is unknown, not rejected.",
        },
        "availability": {
            "publication_timestamp_present": overall["publication_timestamp_present"],
            "publication_timestamp_valid": overall["publication_timestamp_valid"],
            "publication_verification_true": overall["publication_verification_true"],
            "verified_publication_available": overall["publication_availability_verified"],
            "first_capture_timestamp_present": overall["first_capture_timestamp_present"],
            "first_capture_timestamp_valid": overall["first_capture_timestamp_valid"],
            "effective_availability_timestamp_present": overall["effective_availability_timestamp_present"],
            "effective_availability_timestamp_valid": overall["effective_availability_timestamp_valid"],
            "occurrence_timestamp_present": overall["occurrence_timestamp_present"],
            "occurrence_timestamp_valid": overall["occurrence_timestamp_valid"],
            "interpretation": "First capture establishes observation by that time; it never retrospectively validates publication.",
        },
        "groups_by_event_family": {key: unpack(families[key]) for key in sorted(families)},
    }
