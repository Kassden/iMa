"""Compare frozen predictions using the required CSV columns date, race_id, horse_no, target_win, model_probability.

The selection cutoff is operator-declared. Locking existing predictions does not
verify prospective independence, and results are not automatic promotion evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from contextlib import closing

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from .research_storage import admitted_output, reserved_output_file
from .research_sqlite import admitted_sqlite, native_sqlite_connection


OBJECTIVE = "win_race_log_loss"
CONTRACT = {
    "date_column": "date",
    "race_column": "race_id",
    "runner_column": "horse_no",
    "label_column": "target_win",
    "probability_column": "model_probability",
    "required_forecast_columns": ["date", "race_id", "horse_no", "model_probability"],
    "retrospective_label_column": "target_win",
    "prospective_label_columns": ["date", "race_id", "horse_no", "target_win"],
    "optional_population_columns": ["field_size"],
    "population_key": ["date", "race_id", "horse_no"],
    "probability_bounds": [0.0, 1.0],
    "race_probability_sum": 1.0,
    "race_probability_sum_atol": 1e-10,
    "winner_count_per_race": 1,
    "place_metrics": "unsupported_without_locked_native_place_labels_and_contract",
}
LOCK_SCHEMA = "ima.temporal_confirmation.lock.v1"
RESULT_SCHEMA = "ima.temporal_confirmation.result.v1"
MIN_BOOTSTRAP_RESAMPLES = 100
MAX_BOOTSTRAP_RESAMPLES = 1_000_000
_IMMUTABLE_DIRECTORY_RESERVE_BYTES = 64 * 1024
_IMMUTABLE_METADATA_RESERVE_BYTES = 256 * 1024


class ExposureLedger:
    """Append-only operator/import exposure receipts across targets and campaigns.

    This records known exposure, not proof that unimported manual work never saw
    a label. Callers must reconcile all prior research before sealing a study.
    """
    def __init__(self, path):
        self.path=Path(path).absolute()

    def _write(self, statement, parameters):
        # The encoded row/schema allowance includes SQLite page and index
        # overhead; the helpers add existing-file and rollback peaks.
        growth=max(16 * 1024, len(_canonical_json(parameters)) + 16 * 1024)
        if os.environ.get("IMA_STORAGE_ROOT"):
            connection=native_sqlite_connection(
                self.path, estimated_growth_bytes=growth,
                stage="research_confirmation_exposure",
            )
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE IF NOT EXISTS exposures ("
                                   "receipt_id TEXT PRIMARY KEY, first_date TEXT NOT NULL, "
                                   "last_date TEXT NOT NULL, payload TEXT NOT NULL)")
                connection.execute(statement, parameters)
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
            return
        with admitted_sqlite(
                self.path, estimated_growth_bytes=growth,
                stage="research_confirmation_exposure") as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS exposures ("
                               "receipt_id TEXT PRIMARY KEY, first_date TEXT NOT NULL, "
                               "last_date TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute(statement, parameters)

    def record(self, *, first_date, last_date, campaign, purpose, evidence):
        first,last=_cutoff(first_date),_cutoff(last_date)
        if first>last or not campaign or not purpose:
            raise ValueError('Exposure needs ordered dates, campaign and purpose')
        evidence=Path(evidence).resolve()
        payload={'first_date':first,'last_date':last,'campaign':campaign,'purpose':purpose,
                 'evidence_sha256':_sha256(evidence),'evidence':str(evidence)}
        receipt_id=hashlib.sha256(_canonical_json(payload)).hexdigest()
        encoded=_canonical_json(payload).decode("utf-8")
        self._write("INSERT OR IGNORE INTO exposures VALUES (?,?,?,?)",
                    (receipt_id,first,last,encoded))
        return receipt_id

    def overlapping(self, dates):
        # A missing ledger is an empty read; never initialize storage here.
        if not self.path.is_file():
            return []
        uri=self.path.as_uri()+"?mode=ro"
        with closing(sqlite3.connect(uri,uri=True)) as connection:
            rows=connection.execute(
                "SELECT first_date,last_date,payload FROM exposures ORDER BY receipt_id"
            ).fetchall()
        return [json.loads(payload) for first,last,payload in rows
                if any(first<=day<=last for day in dates)]




def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_today() -> date:
    return _utc_now().date()


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("created_at_utc must be a UTC ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc must be a UTC ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at_utc must include UTC timezone")
    return parsed.astimezone(timezone.utc)


def _validate_prediction_probabilities(frame: pd.DataFrame) -> None:
    probabilities = pd.to_numeric(frame["model_probability"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("model_probability must be finite and between zero and one")
    work = frame[["date", "race_id"]].copy()
    work["probability"] = probabilities
    for race_key, race in work.groupby(["date", "race_id"], sort=True):
        if not math.isclose(float(race["probability"].sum()), 1.0, rel_tol=0,
                            abs_tol=CONTRACT["race_probability_sum_atol"]):
            raise ValueError(f"Race {race_key} probabilities must sum to one")


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path).resolve()
    content = _canonical_json(payload)
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable artifact conflict: {path}")
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return path
    if path.exists() or path.is_symlink():
        raise ValueError(f"Immutable artifact conflict: {path}")
    missing_directories = 0
    parent = path.parent
    while not parent.exists():
        if parent.is_symlink():
            raise ValueError("Immutable artifact parent is a symlink")
        missing_directories += 1
        parent = parent.parent
    temporary = path.parent / f".{path.name}.{os.urandom(12).hex()}.tmp"
    estimated_bytes=(len(content)
                     + missing_directories * _IMMUTABLE_DIRECTORY_RESERVE_BYTES
                     + _IMMUTABLE_METADATA_RESERVE_BYTES)
    published = False
    try:
        with admitted_output(
                path.parent, estimated_bytes=estimated_bytes,
                stage="research_confirmation_immutable",
                inodes=missing_directories + 1):
            # Admission happens before creating even the output directory.
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.parent.resolve() != path.parent or path.parent.is_symlink():
                raise ValueError("Immutable artifact parent changed or became a symlink")
            with reserved_output_file(temporary, "xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
                published = True
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"Immutable artifact conflict: {path}")
            if published:
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _identity(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} identity must be a non-empty JSON object")
    result = dict(value)
    for field in ("candidate_id", "revision"):
        item = result.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} identity requires a non-empty {field}")
    try:
        _canonical_json(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} identity must contain finite JSON values") from exc
    return result


def _cutoff(value: str | date) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError("selection_cutoff must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("selection_cutoff must be an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != value:
        raise ValueError("selection_cutoff must be an ISO date (YYYY-MM-DD)")
    return value


def _population(
    path: Path, cutoff: str, *, prospective: bool = False, lock_date: date | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    # Lock-time parsing reads only dates and population metadata, never labels or scores.
    header = set(pd.read_csv(path, nrows=0).columns)
    if prospective and "target_win" in header:
        raise ValueError("Prospective forecast CSV must not contain target_win at lock time")
    required = set(CONTRACT["population_key"])
    missing = required - header
    if missing:
        raise ValueError(f"Prediction CSV missing required population columns: {sorted(missing)}")
    columns = CONTRACT["population_key"] + [
        column for column in CONTRACT["optional_population_columns"] if column in header
    ]
    frame = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Prediction file has no rows: {path}")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
    if (dates <= cutoff).any():
        raise ValueError("Every confirmation prediction date must be strictly after selection_cutoff")
    if prospective and (lock_date is None or (dates <= lock_date.isoformat()).any()):
        raise ValueError("Prospective forecast race dates must be strictly after the current UTC date")
    for column in ("race_id", "horse_no"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Population key {column} contains an empty value")
    keys = pd.DataFrame({"date": dates, "race_id": frame["race_id"], "horse_no": frame["horse_no"]})
    if keys.duplicated(CONTRACT["population_key"]).any():
        raise ValueError(f"Duplicate runner keys in {path}")
    if "field_size" in frame:
        _validate_field_sizes(frame, dates)
    rows = sorted([list(row) for row in keys.itertuples(index=False, name=None)])
    race_pairs = sorted({(row[0], row[1]) for row in rows})
    races = [[day, race] for day, race in race_pairs]
    return rows, races


def _validate_field_sizes(frame: pd.DataFrame, dates: pd.Series) -> None:
    field_sizes = pd.to_numeric(frame["field_size"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(field_sizes).all() or (field_sizes <= 0).any() or (field_sizes % 1 != 0).any():
        raise ValueError("field_size must contain positive finite integers")
    race_rows = pd.DataFrame({"date": dates, "race_id": frame["race_id"],
                              "field_size": field_sizes})
    for race_key, race in race_rows.groupby(["date", "race_id"], sort=True):
        declared = set(race["field_size"].astype(int))
        if len(declared) != 1 or len(race) != next(iter(declared)):
            raise ValueError(f"Partial or inconsistent field_size for race {race_key}")


def create_lock(
    candidate_path: str | Path,
    baseline_path: str | Path,
    lock_path: str | Path,
    *,
    candidate_identity: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    selection_cutoff: str | date,
    objective: str = OBJECTIVE,
    contract: Mapping[str, Any] | None = None,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 20_000,
    random_seed: int = 20261008,
    prospective: bool = False,
) -> Path:
    """Freeze a retrospective comparison or future-dated, label-free forecasts."""
    if not isinstance(prospective, bool):
        raise ValueError("prospective must be a boolean")
    if objective != OBJECTIVE:
        raise ValueError(f"Unsupported objective: {objective}")
    selected_contract = dict(CONTRACT if contract is None else contract)
    if selected_contract != CONTRACT:
        raise ValueError("Unsupported confirmation contract")
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float))
            or not math.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("confidence_level must be between zero and one")
    if (isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int)
            or not MIN_BOOTSTRAP_RESAMPLES <= bootstrap_resamples <= MAX_BOOTSTRAP_RESAMPLES):
        raise ValueError(f"bootstrap_resamples must be an integer from {MIN_BOOTSTRAP_RESAMPLES} to {MAX_BOOTSTRAP_RESAMPLES}")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    candidate = Path(candidate_path).resolve()
    baseline = Path(baseline_path).resolve()
    if not candidate.is_file() or not baseline.is_file():
        raise FileNotFoundError("Candidate and baseline prediction CSVs must exist")
    cutoff = _cutoff(selection_cutoff)
    candidate_identity_value = _identity(candidate_identity, "candidate")
    baseline_identity_value = _identity(baseline_identity, "baseline")
    if candidate_identity_value == baseline_identity_value:
        raise ValueError("Candidate and baseline identities must be distinct")
    candidate_hash, baseline_hash = _sha256(candidate), _sha256(baseline)
    output = Path(lock_path).resolve()
    if prospective and output.exists():
        existing = _read_lock(output)
        expected_inference = {
            "method": "scipy_paired_percentile_bootstrap",
            "resampling_unit": "race",
            "confidence_level": float(confidence_level),
            "resamples": int(bootstrap_resamples),
            "random_seed": random_seed,
            "assumption": "Races are independent exchangeable units; within-race runner dependence is preserved.",
        }
        if (existing["mode"] == "prospective"
                and existing["candidate"]["identity"] == candidate_identity_value
                and existing["baseline"]["identity"] == baseline_identity_value
                and existing["candidate"]["sha256"] == candidate_hash
                and existing["baseline"]["sha256"] == baseline_hash
                and existing["selection_cutoff"] == cutoff
                and existing["objective"] == objective
                and existing["contract"] == selected_contract
                and existing["inference"] == expected_inference):
            if _sha256(candidate) != candidate_hash or _sha256(baseline) != baseline_hash:
                raise ValueError("Prediction file changed while replaying the prospective lock")
            return output
    created_at_utc = _utc_now().astimezone(timezone.utc) if prospective else None
    utc_today = created_at_utc.date() if created_at_utc else None
    if prospective and date.fromisoformat(cutoff) > utc_today:
        raise ValueError("In prospective mode selection_cutoff must be on or before the current UTC date")
    candidate_rows, candidate_races = _population(
        candidate, cutoff, prospective=prospective, lock_date=utc_today,
    )
    baseline_rows, baseline_races = _population(
        baseline, cutoff, prospective=prospective, lock_date=utc_today,
    )
    if candidate_rows != baseline_rows:
        raise ValueError("Candidate and baseline runner populations differ")
    if candidate_races != baseline_races:
        raise ValueError("Candidate and baseline race populations differ")
    if len(candidate_races) < 2:
        raise ValueError("Temporal confirmation requires at least two races")
    if prospective:
        forecast_columns = CONTRACT["required_forecast_columns"]
        candidate_forecast = pd.read_csv(candidate, usecols=forecast_columns, dtype={
            "date": str, "race_id": str, "horse_no": str,
        })
        baseline_forecast = pd.read_csv(baseline, usecols=forecast_columns, dtype={
            "date": str, "race_id": str, "horse_no": str,
        })
        _validate_prediction_probabilities(candidate_forecast)
        _validate_prediction_probabilities(baseline_forecast)
    if _sha256(candidate) != candidate_hash or _sha256(baseline) != baseline_hash:
        raise ValueError("Prediction file changed while the lock was being created")

    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "mode": "prospective" if prospective else "retrospective",
        "candidate": {"identity": candidate_identity_value,
                      "path": str(candidate), "sha256": candidate_hash},
        "baseline": {"identity": baseline_identity_value,
                     "path": str(baseline), "sha256": baseline_hash},
        "objective": objective,
        "contract": selected_contract,
        "selection_cutoff": cutoff,
        "population": {
            "runner_count": len(candidate_rows), "race_count": len(candidate_races),
            "dates": sorted({row[0] for row in candidate_rows}),
            "races": candidate_races,
            "runner_keys_sha256": hashlib.sha256(_canonical_json(candidate_rows)).hexdigest(),
        },
        "inference": {"method": "scipy_paired_percentile_bootstrap",
                      "resampling_unit": "race", "confidence_level": float(confidence_level),
                      "resamples": int(bootstrap_resamples), "random_seed": random_seed,
                      "assumption": "Races are independent exchangeable units; within-race runner dependence is preserved."},
    }
    if prospective:
        payload["created_at_utc"] = created_at_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    payload["lock_id"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return _write_immutable(output, payload)


def _read_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Unsupported or malformed confirmation lock")
    lock_id = lock.get("lock_id")
    unsigned = {key: value for key, value in lock.items() if key != "lock_id"}
    if not isinstance(lock_id, str) or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != lock_id:
        raise ValueError("Confirmation lock integrity check failed")
    if lock.get("objective") != OBJECTIVE or lock.get("contract") != CONTRACT:
        raise ValueError("Confirmation lock objective or contract is unsupported")
    try:
        mode = lock["mode"]
        if mode not in {"prospective", "retrospective"}:
            raise ValueError("Invalid confirmation mode")
        _identity(lock["candidate"]["identity"], "candidate")
        _identity(lock["baseline"]["identity"], "baseline")
        if lock["candidate"]["identity"] == lock["baseline"]["identity"]:
            raise ValueError("Candidate and baseline identities must be distinct")
        for side in ("candidate", "baseline"):
            source = lock[side]
            if not Path(source["path"]).is_absolute() or not _is_sha256(source["sha256"]):
                raise ValueError(f"Invalid {side} source binding in lock")
        cutoff = date.fromisoformat(_cutoff(lock["selection_cutoff"]))
        cutoff_text = cutoff.isoformat()
        lock_date = None
        if mode == "prospective":
            created_at = _utc_timestamp(lock["created_at_utc"])
            lock_date = created_at.date()
            if cutoff > lock_date:
                raise ValueError("Prospective lock cutoff cannot be after its UTC creation date")
        population = lock["population"]
        race_count = population["race_count"]
        runner_count = population["runner_count"]
        if (isinstance(race_count, bool) or not isinstance(race_count, int) or race_count < 2
                or isinstance(runner_count, bool) or not isinstance(runner_count, int)
                or runner_count < race_count):
            raise ValueError("Invalid lock population counts; at least two races are required")
        dates = population["dates"]
        races = population["races"]
        digest = population["runner_keys_sha256"]
        if (not isinstance(dates, list) or not dates or dates != sorted(set(dates))
                or any(_cutoff(day) <= cutoff_text for day in dates)
                or (mode == "prospective" and any(
                    _cutoff(day) <= lock_date.isoformat() for day in dates))
                or not isinstance(races, list) or len(races) != race_count
                or races != sorted(races) or len({tuple(race) for race in races}) != race_count
                or any(not isinstance(race, list) or len(race) != 2 or not all(
                    isinstance(item, str) and item for item in race) or _cutoff(race[0]) <= cutoff_text
                    or (mode == "prospective" and _cutoff(race[0]) <= lock_date.isoformat())
                    for race in races)
                or sorted({race[0] for race in races}) != dates
                or not _is_sha256(digest)):
            raise ValueError("Invalid lock temporal population")
        inference = lock["inference"]
        confidence = inference["confidence_level"]
        resamples, seed = inference["resamples"], inference["random_seed"]
        if (inference.get("method") != "scipy_paired_percentile_bootstrap"
                or inference.get("resampling_unit") != "race"
                or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not math.isfinite(confidence) or not 0 < confidence < 1
                or isinstance(resamples, bool) or not isinstance(resamples, int)
                or not MIN_BOOTSTRAP_RESAMPLES <= resamples <= MAX_BOOTSTRAP_RESAMPLES
                or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                or inference.get("assumption") != "Races are independent exchangeable units; within-race runner dependence is preserved."):
            raise ValueError("Invalid lock inference settings")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed confirmation lock") from exc
    return lock


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read_labels(
    path: Path, lock: Mapping[str, Any], *, prospective: bool, evaluation_date: date,
) -> pd.DataFrame:
    required = CONTRACT["prospective_label_columns"]
    frame = pd.read_csv(path, usecols=required, dtype={
        "date": str, "race_id": str, "horse_no": str,
    })
    if frame.empty:
        raise ValueError("Labels CSV has no rows")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
    frame["date"] = dates
    if frame[["race_id", "horse_no"]].astype(str).apply(
            lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Labels CSV contains an empty population key")
    keys = frame[CONTRACT["population_key"]]
    if keys.duplicated(CONTRACT["population_key"]).any():
        raise ValueError("Labels CSV contains duplicate runner keys")
    ordered = sorted([list(row) for row in keys.itertuples(index=False, name=None)])
    population = lock["population"]
    if (len(ordered) != population["runner_count"]
            or hashlib.sha256(_canonical_json(ordered)).hexdigest() != population["runner_keys_sha256"]):
        raise ValueError("Labels population does not exactly match the locked forecast population")
    if prospective:
        locked_date = _utc_timestamp(lock["created_at_utc"]).date()
        if (dates <= locked_date.isoformat()).any():
            raise ValueError("Prospective labels must be for race dates after the lock date")
        if (dates > evaluation_date.isoformat()).any():
            raise ValueError("Prospective labels include a race date that has not passed in UTC")
    labels = pd.to_numeric(frame["target_win"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("target_win must contain finite binary labels")
    frame["target_win"] = labels
    race_sums = frame.groupby(["date", "race_id"], sort=True)["target_win"].sum()
    if not race_sums.eq(1).all():
        raise ValueError("Each labels race must contain exactly one winner")
    return frame


def _scored_losses(
    path: Path, expected_hash: str, lock: Mapping[str, Any],
    labels: pd.DataFrame | None = None,
) -> dict[tuple[str, str], float]:
    if _sha256(path) != expected_hash:
        raise ValueError(f"Prediction file hash differs from lock: {path}")
    frame = pd.read_csv(path, dtype={"date": str, "race_id": str, "horse_no": str})
    required = set(CONTRACT["required_forecast_columns"])
    if labels is None:
        required.add("target_win")
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction CSV missing required columns: {sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
    if (dates <= lock["selection_cutoff"]).any():
        raise ValueError("Confirmation population overlaps selection_cutoff")
    keys = pd.DataFrame({"date": dates, "race_id": frame["race_id"].astype(str),
                         "horse_no": frame["horse_no"].astype(str)})
    if keys.duplicated(CONTRACT["population_key"]).any():
        raise ValueError(f"Duplicate runner keys in {path}")
    if "field_size" in frame:
        _validate_field_sizes(frame, dates)
    ordered = sorted([list(row) for row in keys.itertuples(index=False, name=None)])
    population = lock["population"]
    if (len(ordered) != population["runner_count"]
            or hashlib.sha256(_canonical_json(ordered)).hexdigest() != population["runner_keys_sha256"]):
        raise ValueError("Prediction runner population differs from lock")

    probabilities = pd.to_numeric(frame["model_probability"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("model_probability must be finite and between zero and one")
    if labels is None:
        labels = pd.to_numeric(frame["target_win"], errors="raise").to_numpy(dtype=float)
    else:
        left = frame[CONTRACT["population_key"]].sort_values(
            CONTRACT["population_key"]
        ).reset_index(drop=True)
        right = labels[CONTRACT["population_key"]].sort_values(
            CONTRACT["population_key"]
        ).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("Labels population differs from forecast runner keys")
        frame = frame.sort_values(CONTRACT["population_key"]).reset_index(drop=True)
        probabilities = pd.to_numeric(frame["model_probability"], errors="raise").to_numpy(dtype=float)
        labels = pd.to_numeric(labels.sort_values(CONTRACT["population_key"])["target_win"],
                               errors="raise").to_numpy(dtype=float)
        dates = frame["date"].astype(str)
        keys = frame[CONTRACT["population_key"]].copy()
        keys["date"] = dates
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("target_win must contain finite binary labels")
    work = pd.DataFrame({"date": dates, "race_id": keys["race_id"], "horse_no": keys["horse_no"],
                         "label": labels, "probability": probabilities})
    grouped = work.groupby(["date", "race_id"], sort=True, dropna=False)
    losses: dict[tuple[str, str], float] = {}
    for race_key, race in grouped:
        if int(race["label"].sum()) != 1:
            raise ValueError(f"Race {race_key} must have exactly one winner")
        if not math.isclose(float(race["probability"].sum()), 1.0, rel_tol=0,
                            abs_tol=CONTRACT["race_probability_sum_atol"]):
            raise ValueError(f"Race {race_key} probabilities must sum to one")
        winner_probability = float(race.loc[race["label"].eq(1), "probability"].iloc[0])
        if winner_probability <= 0:
            raise ValueError(f"Race {race_key} winner probability must be positive for finite log loss")
        losses[(str(race_key[0]), str(race_key[1]))] = -math.log(winner_probability)
    if _sha256(path) != expected_hash:
        raise ValueError(f"Prediction file changed during evaluation: {path}")
    return losses


def evaluate_lock(
    lock_path: str | Path,
    result_path: str | Path,
    labels_path: str | Path | None = None,
) -> Path:
    """Evaluate one locked pair; deterministic replays are idempotent."""
    lock_file = Path(lock_path).resolve()
    lock_file_hash = _sha256(lock_file)
    lock = _read_lock(lock_file)
    prospective = lock["mode"] == "prospective"
    if prospective and labels_path is None:
        raise ValueError("Prospective evaluation requires a separate --labels CSV")
    if labels_path is not None and not Path(labels_path).is_file():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")
    labels_file = Path(labels_path).resolve() if labels_path is not None else None
    labels_file_hash = _sha256(labels_file) if labels_file is not None else None
    external_labels = (
        _read_labels(labels_file, lock, prospective=prospective, evaluation_date=_utc_today())
        if labels_file is not None else None
    )
    candidate_path, baseline_path = Path(lock["candidate"]["path"]), Path(lock["baseline"]["path"])
    candidate_losses = _scored_losses(candidate_path, lock["candidate"]["sha256"], lock, external_labels)
    baseline_losses = _scored_losses(baseline_path, lock["baseline"]["sha256"], lock, external_labels)
    if candidate_losses.keys() != baseline_losses.keys():
        raise ValueError("Candidate and baseline race populations differ")

    if external_labels is None:
        candidate_labels = pd.read_csv(candidate_path, usecols=CONTRACT["population_key"] + ["target_win"],
                                       dtype={"date": str, "race_id": str, "horse_no": str})
        baseline_labels = pd.read_csv(baseline_path, usecols=CONTRACT["population_key"] + ["target_win"],
                                      dtype={"date": str, "race_id": str, "horse_no": str})
        join_keys = CONTRACT["population_key"]
        left = candidate_labels.sort_values(join_keys).reset_index(drop=True)
        right = baseline_labels.sort_values(join_keys).reset_index(drop=True)
        left_labels = pd.to_numeric(left["target_win"], errors="raise").to_numpy(dtype=float)
        right_labels = pd.to_numeric(right["target_win"], errors="raise").to_numpy(dtype=float)
        if (not left[join_keys].equals(right[join_keys])
                or not np.array_equal(left_labels, right_labels)):
            raise ValueError("Candidate and baseline labels differ")

    race_keys = sorted(candidate_losses)
    candidate_values = np.asarray([candidate_losses[key] for key in race_keys], dtype=float)
    baseline_values = np.asarray([baseline_losses[key] for key in race_keys], dtype=float)
    delta = candidate_values - baseline_values

    def mean_difference(candidate_sample: np.ndarray, baseline_sample: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.mean(candidate_sample - baseline_sample, axis=axis)

    interval = bootstrap(
        (candidate_values, baseline_values), mean_difference, paired=True,
        vectorized=True, n_resamples=lock["inference"]["resamples"],
        confidence_level=lock["inference"]["confidence_level"], method="percentile",
        rng=np.random.default_rng(lock["inference"]["random_seed"]), batch=256,
    ).confidence_interval
    lower, upper = float(interval.low), float(interval.high)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Bootstrap confidence interval endpoints must be finite and ordered")
    status = (
        "confirmed_temporal_improvement" if prospective and upper < 0
        else "temporal_comparison_improvement" if upper < 0
        else "provisional"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "mode": lock["mode"],
        "lock_id": lock["lock_id"],
        "status": status,
        "interpretation": (
            "prospective predictions were frozen before the race dates; feature and training leakage are not verified"
            if status == "confirmed_temporal_improvement"
            else "temporal comparison improvement; prospective preselection independence is unverified"
            if status == "temporal_comparison_improvement"
            else "inconclusive; improvement not established by this comparison"
        ),
        "provenance": {
            "status": ("prospective_predictions_frozen_before_race_dates" if prospective
                       else "retrospective_comparison_only"),
            "selection_cutoff_source": "operator_declared",
            "feature_training_leakage_verified": False,
            "automatic_promotion_evidence": False,
            "promotion_requires": "separate_upstream_artifact_and_dataset_provenance",
        },
        "comparison": {
            "candidate_identity": lock["candidate"]["identity"],
            "baseline_identity": lock["baseline"]["identity"],
            "candidate_mean_race_log_loss": float(candidate_values.mean()),
            "baseline_mean_race_log_loss": float(baseline_values.mean()),
            "mean_candidate_minus_baseline": float(delta.mean()),
            "confidence_level": lock["inference"]["confidence_level"],
            "paired_race_bootstrap_interval": [lower, upper],
            "race_count": len(race_keys),
            "resamples": lock["inference"]["resamples"],
            "random_seed": lock["inference"]["random_seed"],
            "resampling_assumption": lock["inference"]["assumption"],
        },
        "temporal_population": {
            "selection_cutoff_exclusive": lock["selection_cutoff"],
            "lock_created_at_utc": lock.get("created_at_utc"),
            "dates": lock["population"]["dates"],
            "runner_count": lock["population"]["runner_count"],
            "race_count": lock["population"]["race_count"],
            "runner_keys_sha256": lock["population"]["runner_keys_sha256"],
        },
        "objective": OBJECTIVE,
        "place_metrics": {"status": "unsupported", "reason": CONTRACT["place_metrics"]},
        "scope": "Descriptive evidence for this locked comparison only. The operator-declared cutoff does not prove prospective preselection independence. Population completeness is checked only when optional field_size is supplied. This is not guaranteed betting edge evidence or automatic promotion evidence.",
        "input_sha256": {"candidate": lock["candidate"]["sha256"],
                         "baseline": lock["baseline"]["sha256"]},
        "labels_sha256": labels_file_hash,
    }
    if (_sha256(candidate_path) != lock["candidate"]["sha256"]
            or _sha256(baseline_path) != lock["baseline"]["sha256"]):
        raise ValueError("Prediction file changed before result write")
    if _sha256(lock_file) != lock_file_hash:
        raise ValueError("Confirmation lock changed during evaluation")
    if labels_file is not None and _sha256(labels_file) != labels_file_hash:
        raise ValueError("Labels file changed during evaluation")
    return _write_immutable(Path(result_path), result)
