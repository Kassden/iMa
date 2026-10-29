"""Fixed-horizon prospective meeting study with immutable forecast versions.

This module deliberately keeps collection status label-free. A study is eligible
for evaluation only after its locked horizon has elapsed and every race has a
terminal outcome. Package identity is derived from verified package manifests
and artifact bytes; a caller-supplied model name is never treated as proof.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from .research_confirmation import ExposureLedger, _canonical_json, _sha256, _write_immutable

LOCK_SCHEMA = "ima.research_meeting_study.lock.v1"
VERSION_SCHEMA = "ima.research_meeting_forecast.version.v1"
RESULT_SCHEMA = "ima.research_meeting_study.result.v1"
CALENDAR_SCHEMA = "ima.official_fixture_calendar_capture.v1"
SCHEDULE_SCHEMA = "ima.official_meeting_card.v1"
DEADLINE_MINUTES = 5
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20261018
POWER_SIMULATIONS = 2_000
POWER_TARGET = 0.80
POWER_SHIFT = -0.002
MIN_MEETINGS = 30
MIN_COVERAGE = 0.95
_KEY = ("race_id", "horse_no")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _add_content_hash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["content_sha256"] = _digest_bytes(_canonical_json(payload))
    return payload


def _verify_content_hash(payload: Mapping[str, Any], label: str) -> None:
    declared = payload.get("content_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "content_sha256"}
    if declared != _digest_bytes(_canonical_json(unhashed)):
        raise ValueError(f"Immutable {label} content hash mismatch")


def _json_file(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    source = Path(path).absolute()
    if source.is_symlink():
        raise ValueError(f"Expected regular JSON file, got symlink: {source}")
    p = source.resolve(strict=True)
    if not p.is_file():
        raise ValueError(f"Expected regular JSON file: {p}")
    raw = p.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return p, value, _digest_bytes(raw)


def _instant(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp with timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp with timezone") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return result.astimezone(timezone.utc)


def _scheduled_post(value: Any) -> datetime:
    """Parse a published race post time expressed in Hong Kong's +08:00 zone."""
    if not isinstance(value, str):
        raise ValueError("scheduled_post must include the Hong Kong timezone (+08:00)")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("scheduled_post must include the Hong Kong timezone (+08:00)") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(hours=8):
        raise ValueError("scheduled_post must include the Hong Kong timezone (+08:00)")
    return result.astimezone(timezone.utc)


def _is_label_field(name: Any) -> bool:
    field = str(name).strip().lower()
    forbidden = {"target", "target_win", "winner", "winners", "win", "won",
                 "finish_position", "finish_time", "finish_seconds", "result", "results",
                 "label", "actual_position", "actual_time", "dividend", "payout"}
    return field in forbidden or field.startswith(("target_", "label_", "result_", "winner_", "outcome_"))


def _contains_label_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_is_label_field(key) or _contains_label_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_label_key(item) for item in value)
    return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _weighted_ratio(differences: np.ndarray, race_counts: np.ndarray, axis: int | None = None):
    return np.sum(differences, axis=axis) / np.sum(race_counts, axis=axis)


def _rowwise_linear_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    """NumPy's linear quantile for each row, using partition rather than full sort."""
    position = (values.shape[1] - 1) * quantile
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    ordered = np.partition(values, (lower, upper), axis=1)
    fraction = position - lower
    return ordered[:, lower] * (1 - fraction) + ordered[:, upper] * fraction


def _ratio_percentile_upper_below_zero(numerator_sums: np.ndarray,
                                       denominator_sums: np.ndarray,
                                       quantile: float = .975) -> np.ndarray:
    """Evaluate the exact sign of a rowwise linear percentile of ratios."""
    if (numerator_sums.ndim != 2 or numerator_sums.shape[1] < 1
            or denominator_sums.shape != numerator_sums.shape
            or not np.isfinite(numerator_sums).all()
            or not np.isfinite(denominator_sums).all()
            or (denominator_sums <= 0).any()
            or not 0 <= quantile <= 1):
        raise ValueError("Bootstrap ratios require aligned finite sums and positive denominators")
    ratios = numerator_sums / denominator_sums
    return _rowwise_linear_quantile(ratios, quantile) < 0


def _bootstrap_upper_sign_rank(numerator_sums: np.ndarray,
                               quantile: float = .975) -> tuple[np.ndarray, np.ndarray]:
    """Resolve non-boundary rows from signs; mark rows needing exact ratios."""
    if (numerator_sums.ndim != 2 or numerator_sums.shape[1] < 1
            or not np.isfinite(numerator_sums).all() or not 0 <= quantile <= 1):
        raise ValueError("Bootstrap numerator sums must be a non-empty finite 2D array")
    position = (numerator_sums.shape[1] - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    negative_counts = np.count_nonzero(numerator_sums < 0, axis=1)
    result = negative_counts > upper
    boundary = (lower < upper) & (negative_counts == upper)
    return result, boundary


def _owned_regular_file(path: Path, *, root: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"Trusted package artifact is a symlink: {path}")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if (not resolved.is_relative_to(root) or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o022):
        raise ValueError(f"Package artifact is outside the owned trusted boundary: {path}")
    return _sha256(resolved)


def _package_hashes(directory: Path, trusted_root: Path) -> dict[str, str | None]:
    required = ("model.joblib", "manifest.json", "recipe.json", "feature-replay.json")
    hashes = {name: _owned_regular_file(directory / name, root=trusted_root) for name in required}
    return hashes


def _manifest_matches(loaded: Mapping[str, Any], saved: Mapping[str, Any]) -> bool:
    # Older native writers omit this optional empty field. Keep the original
    # manifest bytes/hash pinned while comparing its documented default.
    expected = dict(saved)
    expected.setdefault("selection_references", [])
    return dict(loaded) == expected


def _verify_package(path: str | Path, trusted_root: str | Path,
                    expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load the native research package only inside an explicit owned trust root."""
    directory_arg = Path(path).absolute()
    trust_arg = Path(trusted_root).absolute()
    if directory_arg.is_symlink() or trust_arg.is_symlink():
        raise ValueError("Trusted package and root must not be symlinks")
    directory, root = directory_arg.resolve(strict=True), trust_arg.resolve(strict=True)
    root_info, package_info = root.stat(), directory.stat()
    if (not directory.is_dir() or not directory.is_relative_to(root) or directory == root
            or root_info.st_uid != os.getuid() or package_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) & 0o022
            or stat.S_IMODE(package_info.st_mode) & 0o022):
        raise ValueError("Research package must be owned by this user inside trusted_package_root")
    hashes = _package_hashes(directory, root)
    if expected is not None and (
            expected.get("package_dir") != str(directory)
            or expected.get("trusted_root") != str(root)
            or expected.get("artifact_sha256") != hashes):
        raise ValueError("Native package artifact hashes changed after study lock")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not all(
            isinstance(manifest.get(name), str) and manifest[name]
            for name in ("package_id", "recipe_hash", "protocol_id", "code_revision",
                         "model_kind", "target_kind", "prediction_method")):
        raise ValueError("Package does not contain a native ResearchPackageManifest")
    if manifest["target_kind"] != "win_probability":
        raise ValueError("Meeting confirmation requires native win_probability packages")
    from .research_model_package import load_research_package
    try:
        package = load_research_package(directory)
    except Exception as exc:
        raise ValueError(f"Native research package loader failed: {type(exc).__name__}: {exc}") from exc
    loaded_manifest = json.loads(json.dumps(package.manifest().__dict__))
    if not _manifest_matches(loaded_manifest, manifest):
        raise ValueError("Native package loader manifest readback mismatch")
    if package.recipe.target.kind != "win_probability":
        raise ValueError("Loaded package recipe is not a win_probability target")
    feature_context = getattr(package, "feature_context", None)
    if feature_context is None:
        raise ValueError("Prospective confirmation requires a native package feature replay context")
    training_cutoff = getattr(feature_context, "training_cutoff", None)
    if not isinstance(training_cutoff, str) or not training_cutoff:
        raise ValueError("Native package feature replay context must declare its frozen training cutoff")
    identity = {
        "package_dir": str(directory), "trusted_root": str(root),
        "manifest": manifest, "manifest_sha256": hashes["manifest.json"],
        "artifact_sha256": hashes,
        "feature_replay": {"training_cutoff": training_cutoff,
                            "contract_sha256": _digest_bytes(_canonical_json(feature_context.contract()))},
        "loader_receipt": {
            "loader": "ima.research_model_package.load_research_package",
            "package_id": manifest["package_id"], "recipe_hash": manifest["recipe_hash"],
            "protocol_id": manifest["protocol_id"], "code_revision": manifest["code_revision"],
            "feature_replay_sha256": hashes["feature-replay.json"],
        },
    }
    if expected is not None and identity != dict(expected):
        raise ValueError("Native package files or loader identity changed after study lock")
    if _package_hashes(directory, root) != hashes:
        raise ValueError("Native package files changed while loading")
    return identity


def _verify_package_identity(identity: Mapping[str, Any]) -> None:
    directory = Path(identity["package_dir"])
    root = Path(identity["trusted_root"])
    if directory.is_symlink() or root.is_symlink():
        raise ValueError("Pinned native package path or trusted root became a symlink")
    if directory.resolve(strict=True) != directory or root.resolve(strict=True) != root:
        raise ValueError("Pinned native package path changed after study lock")
    if _package_hashes(directory, root) != identity["artifact_sha256"]:
        raise ValueError("A pinned native research package artifact changed")


def _package_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Fingerprint fitted bytes as package_id alone may be shared by refits."""
    return _digest_bytes(_canonical_json({
        "manifest_sha256": identity["manifest_sha256"],
        "artifact_sha256": identity["artifact_sha256"],
    }))


def _read_calendar(path: str | Path, source_readback_path: str | Path,
                   cutoff: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p, calendar, digest = _json_file(path)
    if calendar.get("schema") != CALENDAR_SCHEMA:
        raise ValueError("Unsupported published meeting calendar schema")
    if calendar.get("timezone") != "Asia/Hong_Kong":
        raise ValueError("Official fixture calendar must declare Asia/Hong_Kong")
    readback_path, readback, readback_hash = _json_file(source_readback_path)
    if (readback.get("read_only") is not True
            or calendar.get("source_readback_sha256") != readback_hash
            or not isinstance(calendar.get("source_readback_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", calendar["source_readback_sha256"])
            or not isinstance(readback.get("pages"), list)):
        raise ValueError("Official calendar must bind its pinned read-only raw source readback")
    pages: dict[str, dict[str, Any]] = {}
    for page in readback["pages"]:
        if not isinstance(page, Mapping):
            raise ValueError("Raw calendar readback pages must be objects")
        url, metadata = page.get("url"), page.get("metadata")
        if not isinstance(url, str) or not isinstance(metadata, Mapping) or url in pages:
            raise ValueError("Raw calendar readback needs unique source URLs and metadata")
        if (metadata.get("official_source") != "official:hkjc"
                or metadata.get("http_status") != 200
                or metadata.get("source_url") != url
                or metadata.get("requested_url") != url
                or page.get("body_sha256") != metadata.get("body_hash")
                or page.get("raw_path") != metadata.get("raw_path")
                or not isinstance(page.get("raw_path"), str)
                or not Path(page["raw_path"]).is_absolute()
                or not re.fullmatch(r"[0-9a-f]{64}", str(page.get("body_sha256", "")))
                or not isinstance(page.get("tables"), list) or not page["tables"]):
            raise ValueError("Raw calendar page is not a matching successful official HKJC readback")
        pages[url] = dict(page)
    dates = calendar.get("meetings")
    if not isinstance(dates, list) or not dates:
        raise ValueError("Official calendar must contain its published meeting dates")
    normalized, seen, observations, source_urls = [], set(), [], set()
    for item in dates:
        if not isinstance(item, Mapping):
            raise ValueError("Official calendar meeting rows must be objects")
        value = item.get("meeting_date")
        try:
            day = date.fromisoformat(value).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("Calendar meeting dates must be YYYY-MM-DD") from exc
        if day <= cutoff or day <= _now().astimezone(timezone(timedelta(hours=8))).date().isoformat() or day in seen:
            raise ValueError("Calendar dates must be unique and strictly after selection cutoff")
        source_url = item.get("source_url")
        parsed = urlparse(source_url) if isinstance(source_url, str) else None
        query = parse_qs(parsed.query, keep_blank_values=True) if parsed else {}
        month_values, year_values = query.get("calmonth", []), query.get("calyear", [])
        if (parsed is None or parsed.scheme != "https" or parsed.netloc != "racing.hkjc.com"
                or parsed.path != "/en-us/local/information/fixture"
                or set(query) != {"calmonth", "calyear"}
                or len(month_values) != 1 or len(year_values) != 1
                or not re.fullmatch(r"0[1-9]|1[0-2]", month_values[0])
                or not re.fullmatch(r"20\d{2}", year_values[0])
                or int(year_values[0]) != date.fromisoformat(day).year
                or int(month_values[0]) != date.fromisoformat(day).month):
            raise ValueError("Calendar source URL must be the exact zero-padded official month page")
        body_hash, raw_path = item.get("source_body_sha256"), item.get("source_raw_path")
        page = pages.get(source_url)
        metadata = page.get("metadata", {}) if page else {}
        if (not isinstance(body_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", body_hash)
                or page is None or page.get("body_sha256") != body_hash
                or page.get("raw_path") != raw_path
                or not isinstance(raw_path, str) or not Path(raw_path).is_absolute()):
            raise ValueError("Calendar meeting does not match its raw official source page hash/path")
        observed = _instant(item.get("observed_at"), "calendar source observed_at")
        if observed != _instant(metadata.get("fetched_at"), "calendar page fetched_at"):
            raise ValueError("Calendar observation time does not match its pinned page receipt")
        if item.get("race_ids") not in (None, []) or item.get("post_times") not in (None, []):
            raise ValueError("Calendar lock may include dates and venues only; race cards bind at seal time")
        venue = item.get("venue")
        icons = item.get("session_icons")
        if (not isinstance(venue, str) or not venue.strip()
                or not isinstance(icons, list) or not icons
                or venue not in icons
                or any(not isinstance(icon, str) or not icon.strip() for icon in icons)):
            raise ValueError("Published meeting rows must retain venue and source session icons")
        seen.add(day)
        observations.append(observed)
        source_urls.add(source_url)
        normalized.append({"date": day, "venue": venue, "session_icons": list(icons),
                           "source_url": source_url, "source_body_sha256": body_hash,
                           "source_raw_path": raw_path, "observed_at": observed.isoformat()})
    normalized.sort(key=lambda item: item["date"])
    latest_observed = max(observations)
    if latest_observed.astimezone(timezone(timedelta(hours=8))).date().isoformat() > cutoff:
        raise ValueError("Official calendar source was first observed after the selection cutoff")
    if _sha256(readback_path) != readback_hash or _sha256(p) != digest:
        raise ValueError("Official calendar source evidence changed while reading")
    return {"path": str(p), "sha256": digest,
            "source_url": "https://racing.hkjc.com/en-us/local/information/fixture",
            "source_urls": sorted(source_urls), "latest_source_observed_at": latest_observed.isoformat(),
            "schema": CALENDAR_SCHEMA, "timezone": "Asia/Hong_Kong",
            "source_readback_path": str(readback_path),
            "source_readback_sha256": readback_hash, "meeting_dates": normalized}, normalized


def _development_blocks(path: str | Path, candidate_package_sha256: str,
                        reference_package_sha256: str) -> tuple[list[dict[str, Any]], str]:
    p = Path(path).resolve(strict=True)
    digest = _sha256(p)
    blocks = []
    with p.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"meeting_id", "date", "races", "difference_sum",
                    "candidate_package_sha256", "reference_package_sha256"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("Development blocks require meeting keys, paired difference and both locked package hashes")
        for row in reader:
            if (row["candidate_package_sha256"] != candidate_package_sha256
                    or row["reference_package_sha256"] != reference_package_sha256):
                raise ValueError("Development blocks do not match the locked candidate/reference package pair")
            try:
                day = date.fromisoformat(row["date"]).isoformat()
                races = int(row["races"]); total = float(row["difference_sum"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Malformed development block") from exc
            if not row["meeting_id"] or races <= 0 or not math.isfinite(total):
                raise ValueError("Development blocks need finite differences and positive race counts")
            blocks.append({"meeting_id": row["meeting_id"], "date": day,
                           "races": races, "difference_sum": total})
    if len({x["meeting_id"] for x in blocks}) != len(blocks) or len(blocks) < MIN_MEETINGS:
        raise ValueError(f"Power pilot needs at least {MIN_MEETINGS} unique development meeting blocks")
    if _sha256(p) != digest:
        raise ValueError("Development blocks changed while reading")
    return blocks, digest


def _exposure_inventory(path: str | Path, ledger_path: str | Path,
                        cutoff: str) -> dict[str, Any]:
    source, inventory, digest = _json_file(path)
    ledger = Path(ledger_path).resolve(strict=True)
    if (inventory.get("schema") != "ima.research_exposure_inventory.v1"
            or inventory.get("status") != "complete"
            or inventory.get("unresolved_sources") != []
            or inventory.get("selection_cutoff") != cutoff
            or inventory.get("ledger_path") != str(ledger)
            or inventory.get("ledger_sha256") != _sha256(ledger)):
        raise ValueError("Complete source-bound exposure inventory is required before locking")
    sources = inventory.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Exposure inventory must identify the inspected research sources")
    for item in sources:
        if (not isinstance(item, Mapping) or not isinstance(item.get("scope"), str)
                or not item["scope"].strip() or not isinstance(item.get("evidence"), str)):
            raise ValueError("Exposure inventory source scope/evidence is invalid")
        evidence = Path(item["evidence"]).resolve(strict=True)
        if not evidence.is_file() or _sha256(evidence) != item.get("evidence_sha256"):
            raise ValueError("Exposure inventory source evidence is missing or changed")
    return {"path": str(source), "sha256": digest, "inventory": inventory,
            "qualification": "attested inspected scopes; unresolved sources hold locking"}


def _assert_no_new_overlap(lock: Mapping[str, Any]) -> None:
    audit = lock["development"]["known_exposure_audit"]
    inventory = audit.get("inventory", {})
    inventory_path = Path(inventory.get("path", ""))
    if not inventory_path.is_file() or _sha256(inventory_path) != inventory.get("sha256"):
        raise ValueError("Locked exposure inventory is missing or changed")
    dates = [meeting["date"] for meeting in lock["horizon"]["meetings"]]
    overlap = ExposureLedger(audit["ledger_path"]).overlapping(dates)
    for item in overlap:
        evidence = Path(item["evidence"])
        if not evidence.is_file() or _sha256(evidence) != item.get("evidence_sha256"):
            raise ValueError("Known exposure evidence is missing or no longer matches its ledger hash")
    if overlap:
        raise ValueError("Known exposure now overlaps locked confirmation dates")


def _power_pilot(blocks: list[dict[str, Any]], *, max_meetings: int,
                 simulations: int = POWER_SIMULATIONS,
                 inner_resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    if simulations != POWER_SIMULATIONS or inner_resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("Pilot requires 2,000 simulations with the locked 20,000-resample interval")
    races = np.asarray([b["races"] for b in blocks], dtype=float)
    raw_diffs = np.asarray([b["difference_sum"] for b in blocks], dtype=float)
    constant_development_rate = bool(np.all(raw_diffs / races == raw_diffs[0] / races[0]))
    # Shift every development block by race count so the pooled estimand is -0.002.
    if constant_development_rate:
        # Exact zero-variance shortcut: a common rate remains common after shift.
        diffs = POWER_SHIFT * races
    else:
        diffs = raw_diffs + (POWER_SHIFT - float(raw_diffs.sum() / races.sum())) * races
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reports = []
    max_meetings = min(int(max_meetings), 200)
    for n in range(MIN_MEETINGS, max_meetings + 1):
        successes = 0
        for offset in range(0, simulations, 8):
            batch = min(8, simulations - offset)
            indices = rng.integers(0, len(blocks), size=(batch, n))
            sample_d, sample_r = diffs[indices], races[indices]
            point = sample_d.sum(axis=1) / sample_r.sum(axis=1)
            # Apply the locked meeting-block percentile method to each pilot study.
            if constant_development_rate:
                # Every possible block resample has the same ratio; this is exact.
                upper = point.copy()
            else:
                # Draw categorical meeting indices directly. This is the same
                # whole-block bootstrap law as multinomial counts, without the
                # much slower per-category multinomial sampler.
                sampled_indices = rng.integers(0, n, size=(batch, inner_resamples, n),
                                               dtype=np.uint8 if n <= 256 else np.uint16)
                selected_d = sample_d[np.arange(batch)[:, None, None], sampled_indices]
                bootstrap_difference_sums = selected_d.sum(axis=2)
                upper_below_zero, boundary = _bootstrap_upper_sign_rank(
                    bootstrap_difference_sums, .975)
                # Only when the two order statistics straddle zero can the
                # positive race-count denominators affect the interpolated
                # endpoint's sign. Materialize those denominators for that
                # small boundary subset and apply the exact locked quantile.
                boundary_rows = np.flatnonzero(boundary)
                if len(boundary_rows):
                    boundary_indices = sampled_indices[boundary_rows]
                    selected_r = sample_r[
                        boundary_rows[:, None, None], boundary_indices]
                    bootstrap_race_sums = selected_r.sum(axis=2)
                    upper_below_zero[boundary_rows] = _ratio_percentile_upper_below_zero(
                        bootstrap_difference_sums[boundary_rows], bootstrap_race_sums)
            if constant_development_rate:
                upper_below_zero = point < 0
            successes += int(np.count_nonzero((point <= -.001) & upper_below_zero))
        estimate = successes / simulations
        # Wilson 95% interval makes pilot Monte Carlo error visible.
        z = 1.959963984540054; den = 1 + z*z/simulations
        center = (estimate + z*z/(2*simulations))/den
        half = z*math.sqrt(estimate*(1-estimate)/simulations + z*z/(4*simulations**2))/den
        reports.append({"meetings": n, "successes": successes, "simulations": simulations,
                        "power": estimate, "mc_95": [max(0, center-half), min(1, center+half)]})
        if estimate >= POWER_TARGET:
            return {"status": "powered", "selected_meetings": n, "target_power": POWER_TARGET,
                    "shifted_estimand": POWER_SHIFT, "inner_resamples": inner_resamples,
                    "method": "meeting-block percentile bootstrap with independent categorical resampling; total loss difference / total races",
                    "assumptions": [
                        "development meeting blocks are representative and exchangeable with the future horizon",
                        "the same per-race shift is applied to every development block to target -0.002",
                        "power is conditional on this empirical block distribution and does not account for model or protocol drift",
                        "the Wilson interval describes Monte Carlo error across 2,000 pilot simulations, not uncertainty in those assumptions",
                    ],
                    "simulations": reports}
    return {"status": "power_insufficient", "selected_meetings": None,
            "maximum_meetings_tested": max_meetings, "target_power": POWER_TARGET,
            "shifted_estimand": POWER_SHIFT, "inner_resamples": inner_resamples,
            "method": "meeting-block percentile bootstrap with independent categorical resampling; total loss difference / total races",
            "assumptions": [
                "development meeting blocks are representative and exchangeable with the future horizon",
                "the same per-race shift is applied to every development block to target -0.002",
                "power is conditional on this empirical block distribution and does not account for model or protocol drift",
                "the Wilson interval describes Monte Carlo error across 2,000 pilot simulations, not uncertainty in those assumptions",
            ],
            "simulations": reports}


def create_meeting_study(manifest: Mapping[str, Any], output: str | Path) -> Path:
    """Verify sources, audit known exposure, run the development power pilot, lock."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Study manifest must be an object")
    study_id = manifest.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise ValueError("study_id is required")
    required = {"selection_cutoff", "calendar", "calendar_source_readback",
                "trusted_package_root", "candidate_package",
                "reference_package", "development_blocks", "exposure_ledger",
                "exposure_inventory"}
    if not required <= set(manifest):
        raise ValueError(f"Study manifest missing fields: {sorted(required - set(manifest))}")
    try:
        cutoff = date.fromisoformat(manifest["selection_cutoff"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("selection_cutoff must be an ISO calendar date") from exc
    calendar, meetings = _read_calendar(manifest["calendar"],
                                         manifest["calendar_source_readback"], cutoff)
    candidate = _verify_package(manifest["candidate_package"], manifest["trusted_package_root"])
    reference = _verify_package(manifest["reference_package"], manifest["trusted_package_root"])
    if candidate["artifact_sha256"] == reference["artifact_sha256"]:
        raise ValueError("Candidate and reference must be distinct verified packages")
    if candidate["manifest"]["protocol_id"] != reference["manifest"]["protocol_id"]:
        raise ValueError("Candidate and reference packages must share the frozen protocol_id")
    for package in (candidate, reference):
        try:
            trained_through = date.fromisoformat(package["feature_replay"]["training_cutoff"][:10])
        except (TypeError, ValueError) as exc:
            raise ValueError("Native package training cutoff must be an ISO date or timestamp") from exc
        if trained_through.isoformat() > cutoff:
            raise ValueError("Native package training cutoff must not exceed the frozen selection cutoff")
    candidate_package_sha256 = _package_identity_sha256(candidate)
    reference_package_sha256 = _package_identity_sha256(reference)
    blocks, blocks_hash = _development_blocks(
        manifest["development_blocks"], candidate_package_sha256, reference_package_sha256)
    if any(block["date"] > cutoff for block in blocks):
        raise ValueError("Power-pilot development blocks must be entirely before selection cutoff")
    dates = [m["date"] for m in meetings]
    ledger = ExposureLedger(manifest["exposure_ledger"])
    exposures = ledger.overlapping(dates)
    for item in exposures:
        evidence = Path(item["evidence"])
        if not evidence.is_file() or _sha256(evidence) != item.get("evidence_sha256"):
            raise ValueError("Known exposure evidence is missing or no longer matches its ledger hash")
    if exposures:
        raise ValueError("Known exposure overlaps confirmation schedule; hold study: " +
                         ", ".join(sorted({f"{x['campaign']}:{x['purpose']}" for x in exposures})))
    inventory = _exposure_inventory(manifest["exposure_inventory"], manifest["exposure_ledger"], cutoff)
    pilot = _power_pilot(blocks, max_meetings=int(manifest.get("pilot_max_meetings", 60)))
    if pilot["status"] != "powered":
        report = {"schema": "ima.research_meeting_power_report.v1", "study_id": study_id,
                  "selection_cutoff": cutoff, "calendar_sha256": calendar["sha256"],
                  "candidate_package": candidate, "reference_package": reference,
                  "development_blocks_sha256": blocks_hash, "power_pilot": pilot,
                  "lock_created": False,
                  "decision": "power_insufficient; revise design before any confirmation lock"}
        return _write_immutable(Path(output).with_name(Path(output).name + ".power-insufficient.json"),
                                _add_content_hash(report))
    if len(meetings) < pilot["selected_meetings"]:
        raise ValueError("Published calendar has fewer dates than the frozen powered horizon")
    meetings = meetings[:pilot["selected_meetings"]]
    if (_sha256(Path(calendar["path"])) != calendar["sha256"]
            or _sha256(Path(calendar["source_readback_path"])) != calendar["source_readback_sha256"]
            or _sha256(Path(manifest["development_blocks"])) != blocks_hash):
        raise ValueError("A study source changed during lock")
    if (_verify_package(manifest["candidate_package"], manifest["trusted_package_root"]) != candidate
            or _verify_package(manifest["reference_package"], manifest["trusted_package_root"]) != reference):
        raise ValueError("Fitted package changed during the development power pilot")
    if ledger.overlapping(dates):
        raise ValueError("Known exposure ledger changed to overlap confirmation dates during the pilot")
    if (_exposure_inventory(manifest["exposure_inventory"], manifest["exposure_ledger"], cutoff)
            != inventory):
        raise ValueError("Exposure inventory changed during the development power pilot")
    payload = {"schema": LOCK_SCHEMA, "study_id": study_id, "created_at": _now().isoformat(),
               "selection_cutoff": cutoff, "horizon": {"meeting_count": len(meetings), "meetings": meetings},
               "candidate": candidate, "reference": reference, "calendar": calendar,
               "development": {"path": str(Path(manifest["development_blocks"]).resolve()),
                               "sha256": blocks_hash, "blocks": blocks,
                               "paired_package_ids": {
                                   "candidate": candidate["manifest"]["package_id"],
                                   "reference": reference["manifest"]["package_id"]},
                               "paired_package_sha256": {
                                   "candidate": candidate_package_sha256,
                                   "reference": reference_package_sha256},
                               "known_exposure_audit": {"ledger_path": str(Path(manifest["exposure_ledger"]).resolve()),
                                                         "ledger_sha256": _sha256(Path(manifest["exposure_ledger"]).resolve())
                                                         if Path(manifest["exposure_ledger"]).is_file() else None,
                                                         "overlapping_records": [],
                                                         "inventory": inventory,
                                                         "result": "no_known_overlap_on_locked_meeting_dates"}},
               "power_pilot": pilot,
               "contract": {"objective": "paired_win_race_log_loss_difference_nats_per_race",
                            "primary_comparison": "one_verified_candidate_package_vs_one_verified_reference_package",
                            "forecast_generation": "native_research_package_predict_proba_on_sha256_pinned_parquet_features",
                            "csv_forecast_import": "held; no operator-supplied probability rows are accepted",
                            "horizon_population": "powered meeting dates locked from published calendar; official cards and per-race posts bound at sealing",
                            "refit_after_lock": False,
                            "acceptance": {"delta_lte": -0.001, "upper_95_lt": 0},
                            "confidence": .95, "bootstrap": "meeting_block_percentile",
                            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                            "deadline_minutes_before_post": DEADLINE_MINUTES,
                            "minimum_forecast_coverage": MIN_COVERAGE,
                            "exclusions": "postdeadline_scratches_and_cancelled_races_only_with_frozen_reason_ledger",
                            "optional_stopping": False}}
    return _write_immutable(Path(output), _add_content_hash(payload))


def _read_official_card(path: str | Path, meeting_date: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    card_path, card, digest = _json_file(path)
    if card.get("schema") != SCHEDULE_SCHEMA or card.get("date") != meeting_date:
        raise ValueError("Official meeting card does not match the locked calendar date")
    meeting_id = card.get("meeting_id")
    if not isinstance(meeting_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", meeting_id):
        raise ValueError("Official meeting card needs a valid meeting_id")
    if not isinstance(card.get("source_url"), str) or not card["source_url"].startswith("https://"):
        raise ValueError("Official meeting card must identify its HTTPS source")
    published = _instant(card.get("published_at"), "card published_at")
    races = card.get("races")
    if not isinstance(races, list) or not races:
        raise ValueError("Official meeting card must contain its actual races and scheduled posts")
    normalized, seen = [], set()
    for race in races:
        if not isinstance(race, Mapping):
            raise ValueError("Official meeting card race entries must be objects")
        race_id = race.get("race_id")
        raw_post = race.get("scheduled_post")
        post = _scheduled_post(raw_post)
        if (not isinstance(race_id, str) or not race_id or race_id in seen
                or not isinstance(raw_post, str) or raw_post[:10] != meeting_date):
            raise ValueError("Official card race IDs and Hong Kong scheduled dates must be valid and unique")
        seen.add(race_id)
        normalized.append({"race_id": race_id, "scheduled_post": post.isoformat(),
                           "deadline": (post - timedelta(minutes=DEADLINE_MINUTES)).isoformat()})
    if _sha256(card_path) != digest:
        raise ValueError("Official meeting card changed while reading")
    return {"path": str(card_path), "sha256": digest, "source_url": card["source_url"],
            "published_at": published.isoformat(), "meeting_id": meeting_id,
            "date": meeting_date, "races": normalized}, normalized


def _read_feature_input(path: str | Path, expected: set[tuple[str, str]], meeting_date: str):
    source_arg = Path(path).absolute()
    if source_arg.is_symlink():
        raise ValueError("Pinned feature input must not be a symlink")
    source = source_arg.resolve(strict=True)
    source_info = source.stat()
    if (not stat.S_ISREG(source_info.st_mode) or source_info.st_uid != os.getuid()
            or source_info.st_nlink != 1 or stat.S_IMODE(source_info.st_mode) & 0o022):
        raise ValueError("Pinned feature input must be a non-group/other-writable regular file owned by this user")
    digest = _sha256(source)
    if source.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Native package replay input must be a Parquet feature frame")
    frame = pd.read_parquet(source)
    if any(_is_label_field(name) for name in frame.columns):
        raise ValueError("Pinned feature input must be label-free")
    required = {"race_id", "horse_no", "date"}
    if not required <= set(frame.columns):
        raise ValueError(f"Pinned feature input is missing required runner keys: {sorted(required-set(frame.columns))}")
    frame = frame.copy()
    keys = list(zip(frame["race_id"].astype(str), frame["horse_no"].astype(str)))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Pinned feature input runner population does not match the official card runners")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.date.astype(str)
    if not dates.eq(meeting_date).all():
        raise ValueError("Pinned feature input contains rows outside the locked meeting date")
    race_keys = frame["race_id"].astype(str)
    counts = race_keys.groupby(race_keys).size().to_dict()
    if "field_size" in frame:
        declared = pd.to_numeric(frame["field_size"], errors="coerce")
        if not declared.eq(race_keys.map(counts)).all():
            raise ValueError("Pinned feature input field_size does not match the sealed runner population")
    else:
        frame["field_size"] = race_keys.map(counts).astype(int)
    if _sha256(source) != digest:
        raise ValueError("Pinned feature input changed while reading")
    return source, digest, frame


def _package_predictions(identity: Mapping[str, Any], frame: pd.DataFrame) -> list[dict[str, str]]:
    directory, root = Path(identity["package_dir"]), Path(identity["trusted_root"])
    if _package_hashes(directory, root) != identity["artifact_sha256"]:
        raise ValueError("Pinned native package changed before inference")
    from .research_model_package import load_research_package
    package = load_research_package(directory)
    manifest = json.loads(json.dumps(package.manifest().__dict__))
    if not _manifest_matches(manifest, identity["manifest"]):
        raise ValueError("Loaded model package does not match the locked native manifest")
    probabilities = np.asarray(package.predict_proba(frame), dtype=float)
    if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
        raise ValueError("Native package returned invalid prediction rows")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Native package returned probabilities outside [0,1]")
    sums = pd.Series(probabilities).groupby(frame["race_id"].astype(str).to_numpy()).sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Native package probabilities must sum to one within every race")
    if _package_hashes(directory, root) != identity["artifact_sha256"]:
        raise ValueError("Pinned native package changed during inference")
    return [{"race_id": str(race_id), "horse_no": str(horse_no),
             "model_probability": repr(float(probability))}
            for race_id, horse_no, probability in zip(frame["race_id"], frame["horse_no"], probabilities)]


def _execution_receipt(package: Mapping[str, Any], feature_sha256: str,
                       predictions: list[dict[str, str]]) -> dict[str, Any]:
    return {"executor": "ima.research_model_package.load_research_package.predict_proba",
            "package_id": package["manifest"]["package_id"],
            "package_artifacts_sha256": package["artifact_sha256"],
            "feature_input_sha256": feature_sha256,
            "prediction_rows_sha256": _digest_bytes(_canonical_json(predictions)),
            "prediction_rows": len(predictions),
            "races": len({row["race_id"] for row in predictions})}


def seal_meeting_version(lock_path: str | Path, meeting_date: str, schedule_path: str | Path,
                         runners_path: str | Path, inputs_path: str | Path,
                         output_root: str | Path) -> Path:
    """Replay the two locked native packages on pinned features and seal the result."""
    lock_path, lock, _ = _json_file(lock_path)
    if lock.get("schema") != LOCK_SCHEMA: raise ValueError("Unsupported meeting-study lock")
    _verify_content_hash(lock, "study lock")
    _assert_no_new_overlap(lock)
    meeting = next((m for m in lock["horizon"]["meetings"] if m["date"] == meeting_date), None)
    if meeting is None: raise ValueError("Meeting is not in the locked horizon")
    card, races = _read_official_card(schedule_path, meeting_date)
    schedule_ids = {race["race_id"] for race in races}
    if len(schedule_ids) != len(races):
        raise ValueError("Official card race IDs must be unique")
    runners_file, runners, runners_hash = _json_file(runners_path)
    if (runners.get("schema") != "ima.actual_meeting_runners.v1"
            or runners.get("meeting_id") != card["meeting_id"] or runners.get("date") != meeting_date
            or not isinstance(runners.get("source_url"), str)
            or not runners["source_url"].startswith("https://")
            or runners.get("official_card_sha256") != card["sha256"]):
        raise ValueError("Actual runners source does not match the official card")
    inputs_file, inputs, inputs_hash = _json_file(inputs_path)
    if (inputs.get("schema") != "ima.meeting_forecast_inputs.v1"
            or inputs.get("meeting_id") != card["meeting_id"] or inputs.get("date") != meeting_date):
        raise ValueError("Feature input receipt does not match the official card")
    if _contains_label_key(runners) or _contains_label_key(inputs):
        raise ValueError("Runner and feature input receipts must be label-free")
    now = _now()
    runner_time = _instant(runners.get("captured_at"), "runners captured_at")
    input_time = _instant(inputs.get("captured_at"), "inputs captured_at")
    card_time = _instant(card["published_at"], "card published_at")
    if runner_time > now or input_time > now or card_time > now:
        raise ValueError("Official card, runner, or feature input timestamp is in the future")
    runner_rows = runners.get("runners")
    if not isinstance(runner_rows, list) or not runner_rows: raise ValueError("Actual runners are required")
    expected: set[tuple[str, str]] = set()
    for runner in runner_rows:
        if not isinstance(runner, Mapping): raise ValueError("Runner entries must be objects")
        key = (runner.get("race_id"), runner.get("horse_no"))
        if (key[0] not in schedule_ids or not isinstance(key[1], str) or not key[1]
                or key in expected):
            raise ValueError("Runner identity is outside locked card or duplicated")
        expected.add(key)
    if {key[0] for key in expected} != schedule_ids:
        raise ValueError("Actual runner source omits races from the official card")
    raw_feature_path = inputs.get("feature_input_path")
    if not isinstance(raw_feature_path, str) or not raw_feature_path:
        raise ValueError("Feature input receipt must identify its pinned Parquet input")
    feature_path = Path(raw_feature_path)
    feature_hash = _sha256(feature_path)
    if feature_hash != inputs.get("feature_input_sha256"):
        raise ValueError("Pinned feature input hash does not match its receipt")
    feature_path, observed_feature_hash, frame = _read_feature_input(feature_path, expected, meeting_date)
    if observed_feature_hash != feature_hash:
        raise ValueError("Pinned feature input changed between receipt and load")
    candidate_identity = _verify_package(lock["candidate"]["package_dir"],
                                         lock["candidate"]["trusted_root"], lock["candidate"])
    reference_identity = _verify_package(lock["reference"]["package_dir"],
                                         lock["reference"]["trusted_root"], lock["reference"])
    candidate_rows = _package_predictions(candidate_identity, frame)
    reference_rows = _package_predictions(reference_identity, frame)
    if (_sha256(runners_file) != runners_hash or _sha256(inputs_file) != inputs_hash
            or _sha256(feature_path) != feature_hash or _sha256(Path(card["path"])) != card["sha256"]):
        raise ValueError("Meeting inputs changed while sealing forecasts")
    eligible = [race["race_id"] for race in races
                if now <= _instant(race["deadline"], "deadline")
                and card_time <= _instant(race["deadline"], "deadline")
                and runner_time <= _instant(race["deadline"], "deadline")
                and input_time <= _instant(race["deadline"], "deadline")]
    late = [race["race_id"] for race in races if race["race_id"] not in eligible]
    parent = Path(output_root).resolve() / meeting_date / "versions"
    versions = sorted(parent.glob("v*.json"))
    version = len(versions) + 1
    if versions:
        _, previous, _ = _json_file(versions[-1])
        prior_card = previous.get("official_card", {})
        if (prior_card.get("meeting_id") != card["meeting_id"]
                or prior_card.get("races") != races):
            raise ValueError("Official meeting card race population and post times are frozen at first seal")
    payload = {"schema": VERSION_SCHEMA, "study_id": lock["study_id"],
               "meeting_date": meeting_date, "meeting_id": card["meeting_id"],
               "version": version, "sealed_at": now.isoformat(), "valid_predeadline": bool(eligible),
               "eligible_race_ids": eligible, "late_race_ids": late,
               "deadline_by_race": {r["race_id"]: r["deadline"] for r in races},
               "available_at": {"official_card": card["published_at"],
                                "runners": runner_time.isoformat(), "feature_input": input_time.isoformat()},
               "official_card": card,
               "runners": {"path": str(runners_file), "sha256": runners_hash, "rows": runner_rows},
               "inputs": {"path": str(inputs_file), "sha256": inputs_hash, "artifact": inputs,
                          "feature_frame_path": str(feature_path), "feature_frame_sha256": feature_hash},
               "candidate_package": candidate_identity, "reference_package": reference_identity,
               "candidate_forecast": {"rows": candidate_rows},
               "reference_forecast": {"rows": reference_rows},
               "execution_receipts": {
                   "candidate": _execution_receipt(candidate_identity, feature_hash, candidate_rows),
                   "reference": _execution_receipt(reference_identity, feature_hash, reference_rows),
               }}
    _add_content_hash(payload)
    return _write_immutable(parent / f"v{version:04d}.json", payload)


def _version_eligibility(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    races = payload["official_card"]["races"]
    available = payload["available_at"]
    selected = []
    for race in races:
        deadline = _instant(race["deadline"], "deadline")
        if (_instant(payload["sealed_at"], "sealed_at") <= deadline
                and _instant(available["official_card"], "card published_at") <= deadline
                and _instant(available["runners"], "runners captured_at") <= deadline
                and _instant(available["feature_input"], "feature input captured_at") <= deadline):
            selected.append(race["race_id"])
    return selected, [race["race_id"] for race in races if race["race_id"] not in selected]


def _versions(lock: Mapping[str, Any], root: str | Path):
    base = Path(root).resolve(); selected = {}; bound_cards = {}; late_count = 0
    race_owners: dict[str, str] = {}
    for meeting in lock["horizon"]["meetings"]:
        meeting_date = meeting["date"]
        files = sorted((base / meeting_date / "versions").glob("v*.json"))
        valid = []
        cards = []
        for path in files:
            _, payload, _ = _json_file(path)
            if payload.get("schema") != VERSION_SCHEMA or payload.get("study_id") != lock["study_id"]:
                raise ValueError(f"Unexpected forecast version: {path}")
            if payload.get("meeting_date") != meeting_date:
                raise ValueError(f"Forecast version belongs to a different meeting: {path}")
            if payload.get("version") != int(path.stem[1:]):
                raise ValueError(f"Forecast version number does not match its immutable filename: {path}")
            _verify_content_hash(payload, "forecast version")
            eligible, late = _version_eligibility(payload)
            if (payload.get("eligible_race_ids") != eligible or payload.get("late_race_ids") != late
                    or payload.get("valid_predeadline") != bool(eligible)):
                raise ValueError("Forecast version incorrectly records race-level deadline validity")
            late_count += bool(late)
            card = payload["official_card"]
            if _sha256(Path(card["path"])) != card["sha256"]:
                raise ValueError("Sealed official meeting card changed")
            if _read_official_card(card["path"], meeting_date)[0] != card:
                raise ValueError("Sealed official card content does not match its captured source")
            cards.append(card)
            if _sha256(Path(payload["runners"]["path"])) != payload["runners"]["sha256"]:
                raise ValueError("Sealed actual runners input changed")
            if _sha256(Path(payload["inputs"]["path"])) != payload["inputs"]["sha256"]:
                raise ValueError("Sealed model input artifact changed")
            _, source_runners, _ = _json_file(payload["runners"]["path"])
            _, source_inputs, _ = _json_file(payload["inputs"]["path"])
            source_runner_payload = dict(source_runners)
            runner_timestamp = _instant(source_runner_payload.pop("captured_at", None), "runners captured_at")
            if (runner_timestamp != _instant(payload["available_at"]["runners"], "runners captured_at")
                    or source_runner_payload != {
                        "schema": "ima.actual_meeting_runners.v1", "meeting_id": card["meeting_id"],
                        "date": meeting_date, "source_url": source_runners.get("source_url"),
                        "official_card_sha256": card["sha256"], "runners": payload["runners"]["rows"]}):
                raise ValueError("Sealed runner manifest does not match the source artifact")
            if source_inputs != payload["inputs"]["artifact"]:
                raise ValueError("Sealed feature input receipt does not match its source artifact")
            if _sha256(Path(payload["inputs"]["feature_frame_path"])) != payload["inputs"]["feature_frame_sha256"]:
                raise ValueError("Pinned feature input changed after package inference")
            for side in ("candidate_package", "reference_package"):
                _verify_package_identity(payload[side])
            if payload["candidate_package"] != lock["candidate"] or payload["reference_package"] != lock["reference"]:
                raise ValueError("Forecast version does not use the locked candidate/reference packages")
            if payload["execution_receipts"] != {
                    "candidate": _execution_receipt(payload["candidate_package"],
                                                      payload["inputs"]["feature_frame_sha256"],
                                                      payload["candidate_forecast"]["rows"]),
                    "reference": _execution_receipt(payload["reference_package"],
                                                      payload["inputs"]["feature_frame_sha256"],
                                                      payload["reference_forecast"]["rows"])}:
                raise ValueError("Sealed prediction execution receipt does not match packages and feature hash")
            if eligible:
                valid.append(payload)
        if cards:
            reference_card = cards[0]
            if any(card["meeting_id"] != reference_card["meeting_id"]
                   or card["races"] != reference_card["races"] for card in cards[1:]):
                raise ValueError("Versioned official card changed race population or scheduled post times")
            bound_cards[meeting_date] = reference_card
            for race in reference_card["races"]:
                previous = race_owners.setdefault(race["race_id"], meeting_date)
                if previous != meeting_date:
                    raise ValueError("Official race IDs must be unique across the locked horizon")
        if valid:
            card_races = valid[0]["official_card"]["races"]
        else:
            card_races = cards[0]["races"] if cards else []
        for race in card_races:
            eligible_versions = [item for item in valid if race["race_id"] in item["eligible_race_ids"]]
            if eligible_versions:
                selected[race["race_id"]] = max(eligible_versions, key=lambda x: x["version"])
    return selected, bound_cards, late_count


def collection_status(lock_path: str | Path, versions_root: str | Path) -> dict[str, Any]:
    """Label-free planner-safe coverage and operational misses only."""
    _, lock, lock_hash = _json_file(lock_path)
    if lock.get("schema") != LOCK_SCHEMA: raise ValueError("Unsupported meeting-study lock")
    _verify_content_hash(lock, "study lock")
    _assert_no_new_overlap(lock)
    selected, bound_cards, late = _versions(lock, versions_root)
    total = sum(len(bound_cards.get(m["date"], {}).get("races", [])) for m in lock["horizon"]["meetings"])
    forecasted = len(selected)
    missing_meetings = [m["date"] for m in lock["horizon"]["meetings"] if m["date"] not in bound_cards]
    race_coverage = forecasted / total if total else 0.
    meeting_coverage = (len(lock["horizon"]["meetings"]) - len(missing_meetings)) / len(lock["horizon"]["meetings"])
    return {"schema": "ima.research_meeting_collection_status.v1", "study_id": lock["study_id"],
            "lock_sha256": lock_hash, "scheduled_races": total, "covered_races": forecasted,
            "coverage": race_coverage, "calendar_meetings": len(lock["horizon"]["meetings"]),
            "bound_meetings": len(bound_cards), "meeting_coverage": meeting_coverage,
            "missing_meetings": missing_meetings,
            "operationally_qualified": (total > 0 and race_coverage >= MIN_COVERAGE
                                        and not missing_meetings),
            "missing_or_late_races": total - forecasted, "late_versions": late,
            "bound_meeting_dates": sorted(bound_cards),
            "planner_payload": "collection_status_only; no labels, scores, probabilities, or outcome-derived fields"}


def evaluate_fixed_horizon(lock_path: str | Path, versions_root: str | Path,
                           outcomes_path: str | Path, output: str | Path) -> Path:
    """Evaluate once after every locked race has a terminal, reasoned outcome."""
    _, lock, lock_hash = _json_file(lock_path)
    if lock.get("schema") != LOCK_SCHEMA: raise ValueError("Unsupported meeting-study lock")
    _verify_content_hash(lock, "study lock")
    _assert_no_new_overlap(lock)
    _, bound_cards, _ = _versions(lock, versions_root)
    calendar_end = _instant(lock["horizon"]["meetings"][-1]["date"] + "T23:59:59+08:00",
                            "fixed horizon calendar end")
    if len(bound_cards) != lock["horizon"]["meeting_count"] and _now() < calendar_end:
        raise ValueError("Fixed calendar horizon is not complete; outcomes remain unopened")
    if len(bound_cards) != lock["horizon"]["meeting_count"]:
        raise ValueError("Official meeting card missing for one or more locked calendar dates")
    scheduled = [r for card in bound_cards.values() for r in card["races"]]
    end = max([calendar_end, *(_instant(r["scheduled_post"], "scheduled_post") for r in scheduled)])
    if _now() < end: raise ValueError("Fixed horizon is not complete; outcomes remain unopened")
    status = collection_status(lock_path, versions_root)
    if not status["operationally_qualified"]:
        raise ValueError("Operational forecast coverage is below the frozen 95% threshold")
    outcome_file, outcomes, outcome_hash = _json_file(outcomes_path)
    if outcomes.get("schema") != "ima.meeting_outcomes.v1" or not isinstance(outcomes.get("races"), list):
        raise ValueError("Outcomes must use the separate terminal outcome schema")
    outcome_time = _instant(outcomes.get("published_at"), "outcome published_at")
    if outcome_time < end:
        raise ValueError("Outcome source predates completion of the fixed horizon")
    if outcome_time > _now():
        raise ValueError("Outcome source timestamp is in the future")
    by_race = {r.get("race_id"): r for r in outcomes["races"]}
    if len(by_race) != len(outcomes["races"]) or set(by_race) != {r["race_id"] for r in scheduled}:
        raise ValueError("Outcome file must contain exactly every locked race")
    selected, bound_cards, _ = _versions(lock, versions_root)
    blocks, block_races, exclusions = [], [], []
    for meeting in lock["horizon"]["meetings"]:
        diff_sum = 0.; used = 0
        card = bound_cards[meeting["date"]]
        for race in card["races"]:
            result = by_race[race["race_id"]]
            version = selected.get(race["race_id"])
            if version is None:
                exclusions.append({"race_id": race["race_id"], "status": "operational_miss",
                                   "reason": "no valid predeadline forecast version"})
                continue
            state = result.get("status")
            if state == "cancelled":
                if not isinstance(result.get("reason"), str) or not result["reason"].strip():
                    raise ValueError("Cancelled race exclusion needs a frozen reason")
                exclusions.append({"race_id": race["race_id"], "status": state, "reason": result["reason"]})
                continue
            if state != "completed": raise ValueError("Every race needs completed/cancelled terminal status")
            scratches = result.get("postdeadline_scratches", [])
            if not isinstance(scratches, list): raise ValueError("Scratch reason ledger must be a list")
            scratch_ids = set()
            for scratch in scratches:
                if not isinstance(scratch, Mapping) or not scratch.get("horse_no") or not scratch.get("reason"):
                    raise ValueError("Postdeadline scratch exclusions require runner and reason")
                scratch_ids.add((race["race_id"], str(scratch["horse_no"])))
                exclusions.append({"race_id": race["race_id"], "horse_no": str(scratch["horse_no"]),
                                   "status": "postdeadline_scratch", "reason": scratch["reason"]})
            winners = result.get("winners")
            if not isinstance(winners, list) or len(winners) != 1: raise ValueError("Win target requires exactly one winner")
            c = {(x["race_id"], x["horse_no"]): float(x["model_probability"])
                 for x in version["candidate_forecast"]["rows"]}
            b = {(x["race_id"], x["horse_no"]): float(x["model_probability"])
                 for x in version["reference_forecast"]["rows"]}
            runner_rows = version["runners"]["rows"]
            race_rows = [x for x in runner_rows if x["race_id"] == race["race_id"]]
            if len(scratch_ids) != len(scratches) or any(key[1] not in {str(x["horse_no"]) for x in race_rows}
                                                         for key in scratch_ids):
                raise ValueError("Scratch ledger contains duplicate or unknown locked runners")
            active = [str(x["horse_no"]) for x in race_rows if (race["race_id"], str(x["horse_no"])) not in scratch_ids]
            if str(winners[0]) not in active: raise ValueError("Winner is missing or scratched under locked rules")
            keys = [(race["race_id"], horse) for horse in active]
            if any(k not in c or k not in b for k in keys): raise ValueError("Missing sealed forecast runner")
            csum, bsum = sum(c[k] for k in keys), sum(b[k] for k in keys)
            if csum <= 0 or bsum <= 0: raise ValueError("Cannot renormalize scratched field probabilities")
            winner = (race["race_id"], str(winners[0]))
            diff_sum += -math.log(max(c[winner] / csum, 1e-15)) + math.log(max(b[winner] / bsum, 1e-15))
            used += 1
        if used:
            blocks.append(diff_sum); block_races.append(used)
    if _sha256(outcome_file) != outcome_hash: raise ValueError("Outcome source changed during evaluation")
    if len(blocks) < MIN_MEETINGS:
        payload = {"schema": RESULT_SCHEMA, "study_id": lock["study_id"], "lock_sha256": lock_hash,
                   "outcomes": {"path": str(outcome_file), "sha256": outcome_hash},
                   "created_at": _now().isoformat(), "fixed_horizon_complete": True,
                   "optional_stopping": False, "meeting_blocks": len(blocks),
                   "races": int(sum(block_races)), "delta_nats_per_race": None,
                   "bootstrap_95_percentile": None, "status": "operationally_unqualified",
                   "coverage": status["coverage"], "forecast_versions": {k: v["version"] for k, v in selected.items()},
                   "exclusions": exclusions,
                   "reason": "fewer than 30 evaluable meeting blocks remain after predeclared misses and exclusions"}
        return _write_immutable(Path(output), _add_content_hash(payload))
    values, race_counts = np.asarray(blocks), np.asarray(block_races)
    statistic = lambda data, races, axis: _weighted_ratio(data, races, axis)
    interval = bootstrap((values, race_counts), statistic, paired=True, vectorized=True,
                         axis=0, n_resamples=BOOTSTRAP_RESAMPLES, confidence_level=.95,
                         method="percentile", random_state=np.random.default_rng(BOOTSTRAP_SEED))
    delta = float(values.sum() / race_counts.sum())
    low, high = float(interval.confidence_interval.low), float(interval.confidence_interval.high)
    accepted = delta <= -.001 and high < 0
    payload = {"schema": RESULT_SCHEMA, "study_id": lock["study_id"], "lock_sha256": lock_hash,
               "outcomes": {"path": str(outcome_file), "sha256": outcome_hash},
               "created_at": _now().isoformat(), "fixed_horizon_complete": True,
               "optional_stopping": False, "meeting_blocks": len(blocks), "races": int(race_counts.sum()),
               "delta_nats_per_race": delta, "bootstrap_95_percentile": [low, high],
               "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
               "acceptance_rule": "delta <= -0.001 and upper 95% endpoint < 0",
               "status": ("confirmed" if accepted else "inconclusive" if high >= 0 else "not_confirmed"),
               "coverage": status["coverage"],
               "forecast_versions": {k: v["version"] for k, v in selected.items()},
               "exclusions": exclusions,
               "interpretation": "paired log-loss confirmation only; not profit evidence"}
    return _write_immutable(Path(output), _add_content_hash(payload))
