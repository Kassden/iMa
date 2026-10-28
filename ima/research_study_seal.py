"""Manifest-bound prospective study seals layered over temporal confirmation.

The seal records an operator supplied exposure audit and scientific identities.
It cannot prove that unlisted manual research never saw a label.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .research_confirmation import (
    _canonical_json,
    _sha256,
    _write_immutable,
    _utc_today,
    create_lock,
    evaluate_lock,
    _read_lock,
    _population,
)


SEAL_SCHEMA = "ima.research_study_seal.v1"
_REQUIRED_IDENTIFIERS = ("split_id", "recipe_id", "dataset_id", "code_id", "environment_id")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _exposure_inventory(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("sources"), list) or not raw["sources"]:
        raise ValueError("An explicit non-empty exposure source inventory is required")
    scope = _required_text(raw.get("scope"), "exposure inventory scope")
    sources, held = [], []
    names = set()
    for source in raw["sources"]:
        if not isinstance(source, Mapping):
            raise ValueError("Exposure source entries must be objects")
        name = _required_text(source.get("name"), "exposure source name")
        path_text = _required_text(source.get("path"), f"{name} path")
        expected = _required_text(source.get("sha256"), f"{name} sha256")
        status = source.get("status")
        if name in names:
            raise ValueError(f"Duplicate exposure source name: {name}")
        names.add(name)
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid SHA-256 for exposure source {name}")
        path = Path(path_text).expanduser().resolve()
        if status not in {"resolved", "unresolved"}:
            raise ValueError(f"Exposure source {name} status must be resolved or unresolved")
        if status == "unresolved":
            held.append(f"exposure_source_unresolved:{name}")
        else:
            if not path.is_file():
                raise FileNotFoundError(f"Exposure source is unreadable: {name}: {path}")
            if _sha256(path) != expected:
                raise ValueError(f"Exposure source hash mismatch: {name}")
        sources.append({"name": name, "path": str(path), "sha256": expected, "status": status})

    known = raw.get("known_exposures", [])
    if not isinstance(known, list):
        raise ValueError("known_exposures must be a list")
    source_index = {source["name"]: source for source in sources}
    keys = []
    for row in known:
        if not isinstance(row, Mapping):
            raise ValueError("Known exposure entries must be objects")
        day = row.get("date")
        try:
            if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Known exposure date must be YYYY-MM-DD") from exc
        race = _required_text(row.get("race_id"), "known exposure race_id")
        runner = row.get("horse_no")
        if runner is not None:
            runner = _required_text(str(runner), "known exposure horse_no")
        source_name = _required_text(row.get("source_name"), "known exposure source_name")
        source = source_index.get(source_name)
        if source is None:
            raise ValueError(f"Known exposure references an uninventoried source: {source_name}")
        if row.get("source_sha256") != source["sha256"]:
            raise ValueError(f"Known exposure source hash does not match inventory: {source_name}")
        keys.append({"date": day, "race_id": race, "horse_no": runner,
                     "source_name": source_name, "source_sha256": source["sha256"]})
    if len({(k["date"], k["race_id"], k["horse_no"]) for k in keys}) != len(keys):
        raise ValueError("Duplicate known exposure key")
    normalized = {"scope": scope, "sources": sources,
                  "known_exposures": sorted(keys, key=lambda k: (k["date"], k["race_id"], k["horse_no"] or "")),
                  "history_completeness_claimed": False}
    return normalized, held


def create_study_seal(manifest: Mapping[str, Any], lock_path: str | Path,
                      seal_path: str | Path) -> Path:
    """Validate a caller-frozen manifest and publish its prospective study seal.

    Required manifest fields: study_id, selection_cutoff, candidate and baseline
    (each identity and prediction_csv), objective, split_id, recipe_id,
    dataset_id, code_id, environment_id, and exposure_inventory.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("Study manifest must be an object")
    study_id = _required_text(manifest.get("study_id"), "study_id")
    cutoff = _required_text(manifest.get("selection_cutoff"), "selection_cutoff")
    identifiers = {key: _required_text(manifest.get(key), key) for key in _REQUIRED_IDENTIFIERS}
    objective = _required_text(manifest.get("objective"), "objective")
    inventory, held = _exposure_inventory(manifest.get("exposure_inventory"))

    sides = {}
    for side in ("candidate", "baseline"):
        value = manifest.get(side)
        if not isinstance(value, Mapping):
            raise ValueError(f"{side} must contain identity and prediction_csv")
        prediction = Path(_required_text(value.get("prediction_csv"), f"{side} prediction_csv")).expanduser().resolve()
        if not prediction.is_file():
            raise FileNotFoundError(f"{side} prediction artifact is unreadable: {prediction}")
        identity = value.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{side} identity must be an object")
        sides[side] = {"path": prediction, "identity": dict(identity), "sha256": _sha256(prediction)}

    # Apply the exposure quarantine before create_lock publishes anything.
    cutoff_day = date.fromisoformat(cutoff)
    if cutoff_day.isoformat() != cutoff:
        raise ValueError("selection_cutoff must be an ISO date (YYYY-MM-DD)")
    candidate_rows, candidate_races = _population(
        sides["candidate"]["path"], cutoff, prospective=True, lock_date=_utc_today())
    baseline_rows, baseline_races = _population(
        sides["baseline"]["path"], cutoff, prospective=True, lock_date=_utc_today())
    if candidate_rows != baseline_rows or candidate_races != baseline_races:
        raise ValueError("Candidate and baseline runner populations differ")
    exposed_races = {(row["date"], row["race_id"]) for row in inventory["known_exposures"]}
    exposed_runners = {(row["date"], row["race_id"], row["horse_no"])
                       for row in inventory["known_exposures"] if row["horse_no"] is not None}
    if ({tuple(row) for row in candidate_races} & exposed_races
            or {tuple(row) for row in candidate_rows} & exposed_runners):
        raise ValueError("Prospective test population overlaps known exposed race/date selections")

    lock = Path(lock_path).expanduser().resolve()
    create_lock(sides["candidate"]["path"], sides["baseline"]["path"], lock,
                candidate_identity=sides["candidate"]["identity"],
                baseline_identity=sides["baseline"]["identity"],
                selection_cutoff=cutoff, objective=objective, prospective=True)
    locked = _read_lock(lock)
    if any(_sha256(sides[side]["path"]) != sides[side]["sha256"] for side in sides):
        raise ValueError("Prediction artifact changed while sealing study")
    for source in inventory["sources"]:
        if source["status"] == "resolved" and _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Exposure source changed while sealing study: {source['name']}")
    payload = {
        "schema": SEAL_SCHEMA,
        "study_id": study_id,
        "status": "qualificationheld" if held else "sealed_awaiting_future_labels",
        "qualification_holds": held,
        "exposure_inventory": inventory,
        "population_identity": {
            "date_hash": hashlib.sha256(_canonical_json(locked["population"]["dates"])).hexdigest(),
            "runner_keys_sha256": locked["population"]["runner_keys_sha256"],
            "race_count": locked["population"]["race_count"],
            "runner_count": locked["population"]["runner_count"],
        },
        "candidate": {"identity": locked["candidate"]["identity"], "path": locked["candidate"]["path"],
                      "sha256": locked["candidate"]["sha256"]},
        "baseline": {"identity": locked["baseline"]["identity"], "path": locked["baseline"]["path"],
                     "sha256": locked["baseline"]["sha256"]},
        "objective": locked["objective"], "selection_cutoff": locked["selection_cutoff"],
        "split_id": identifiers["split_id"], "recipe_id": identifiers["recipe_id"],
        "dataset_id": identifiers["dataset_id"], "code_id": identifiers["code_id"],
        "environment_id": identifiers["environment_id"],
        "scientific_identity_authority": "caller_asserted; not independently verified against training artifacts",
        "lock_path": str(lock), "lock_sha256": _sha256(lock),
        "scope": "Operator-supplied exposure inventory; absence of overlap does not establish complete history or feature/training independence.",
    }
    payload["seal_id"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return _write_immutable(Path(seal_path), payload)


def evaluate_study_seal(seal_path: str | Path, result_path: str | Path,
                        labels_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate only when the fixed horizon has elapsed and labels are available."""
    path = Path(seal_path).expanduser().resolve()
    seal = json.loads(path.read_text(encoding="utf-8"))
    seal_id = seal.pop("seal_id", None)
    if seal.get("schema") != SEAL_SCHEMA or not isinstance(seal_id, str):
        raise ValueError("Unsupported or malformed study seal")
    if hashlib.sha256(_canonical_json(seal)).hexdigest() != seal_id:
        raise ValueError("Study seal integrity check failed")
    seal["seal_id"] = seal_id
    if seal["status"] == "qualificationheld":
        return {"status": "qualificationheld", "holds": list(seal["qualification_holds"]),
                "seal_id": seal_id}
    lock = Path(seal["lock_path"])
    if not lock.is_file() or _sha256(lock) != seal["lock_sha256"]:
        raise ValueError("Sealed confirmation lock is missing or corrupted")
    locked = _read_lock(lock)
    for side in ("candidate", "baseline"):
        artifact = Path(seal[side]["path"])
        if not artifact.is_file() or _sha256(artifact) != seal[side]["sha256"]:
            raise ValueError(f"Sealed {side} prediction artifact is missing or corrupted")
    for source in seal["exposure_inventory"]["sources"]:
        if source["status"] == "resolved":
            source_path = Path(source["path"])
            if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
                raise ValueError(f"Sealed exposure source is missing or corrupted: {source['name']}")
    if max(date.fromisoformat(day) for day in locked["population"]["dates"]) > _utc_today():
        return {"status": "scheduled", "reason": "future_locked_dates_have_not_elapsed", "seal_id": seal_id}
    if labels_path is None:
        return {"status": "pending_labels", "reason": "future_labels_not_available", "seal_id": seal_id}
    evaluate_lock(lock, result_path, labels_path=labels_path)
    return {"status": "evaluated", "result_path": str(Path(result_path).resolve()), "seal_id": seal_id}
