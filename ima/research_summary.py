"""Deterministic presentation of a supplied research snapshot, without I/O or LLMs.

Missing observations remain null. The caller owns snapshot consistency and any
ledger/resource collection; terminal results alone cannot describe active fits.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone


SUMMARY_SCHEMA_VERSION = 1
_FAILURES = {"failed", "error", "timeout", "timed_out", "interrupted",
             "blocked_tracking", "blocked_failures", "blocked_paper"}
_ACTIVE = {"reserved", "queued", "preparing", "running"}


def reported_cost(usage: dict) -> float | None:
    """Accept only an actual reported receipt, including a reported zero."""
    value = usage.get("total_cost_usd")
    if (usage.get("cost_status") != "reported" or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
        return None
    return value


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def _utc(value):
    parsed = _timestamp(value)
    return parsed.isoformat().replace("+00:00", "Z") if parsed else None


def _active_trials(snapshot, observed_at):
    rows = snapshot.get("active_trials", snapshot.get("attempts"))
    if rows is None:
        return None
    active = []
    for row in rows:
        status = row.get("status", "running" if "active_trials" in snapshot else "unknown")
        if status not in _ACTIVE:
            continue
        runtime = row.get("runtime") or {}
        progress = row.get("progress") or {}
        progress_fields = progress if isinstance(progress, dict) else {}
        recipe = row.get("recipe") or row.get("payload", {}).get("recipe", {})
        start = row.get("started_at", runtime.get("started_at"))
        elapsed = row.get("elapsed_seconds", runtime.get("elapsed_seconds"))
        if elapsed is None and _timestamp(start) and _timestamp(observed_at):
            elapsed = max(0, (_timestamp(observed_at) - _timestamp(start)).total_seconds())
        fold = next((source[name] for source in (row, progress_fields, runtime)
                     for name in ("fold", "fold_id") if source.get(name) is not None), None)
        active.append({
            "attempt_id": row.get("attempt_id"), "program_id": row.get("program_id", row.get("payload", {}).get("program_id")),
            "status": status, "model_family": row.get("model_family", recipe.get("model", {}).get("kind")),
            "target": row.get("target", recipe.get("target")), "fold": fold,
            "stage": row.get("stage", progress_fields.get("stage", runtime.get("stage"))), "progress": row.get("progress", runtime.get("progress")),
            "started_at": _utc(start), "elapsed_seconds": elapsed,
            "deadline_at": _utc(row.get("deadline_at", runtime.get("deadline_at"))),
            "max_wall_seconds": row.get("max_wall_seconds", runtime.get("max_wall_seconds")),
            "wait_reason": row.get("wait_reason", runtime.get("wait_reason")),
        })
    return sorted(active, key=lambda row: str(row["attempt_id"]))


def _champions(current, prior):
    rows = []
    for key, candidate in sorted(current.items()):
        value = candidate.get("objective_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        try:
            contract = json.loads(key)
        except (ValueError, TypeError):
            contract = {}
        if not isinstance(contract, dict):
            contract = {}
        previous = prior.get(key)
        if (candidate.get("comparison_key", key) != key
                or previous and previous.get("comparison_key", key) != key):
            previous = None
        previous_value = previous.get("objective_value") if previous else None
        if (isinstance(previous_value, bool) or not isinstance(previous_value, (int, float))
                or not math.isfinite(previous_value)):
            previous_value = None
        delta = value - previous_value if previous_value is not None else None
        rows.append({
            "comparison_key": key, "target": contract.get("target", candidate.get("recipe", {}).get("target")),
            "objective": candidate.get("objective_name", contract.get("objective")),
            "identity": {name: contract.get(name) for name in (
                "protocol_id", "protocol_hash", "dataset_hash", "evaluation_population_id",
                "availability_policy", "probability_basis", "target_unit", "metric_contract_version")},
            "model_family": candidate.get("recipe", {}).get("model", {}).get("kind"),
            "current_value": value, "prior_value": previous_value,
            "absolute_delta": delta,
            "relative_delta": delta / abs(previous_value) if delta is not None and previous_value else None,
            "current": candidate, "prior": previous,
        })
    return rows


def summary(snapshot: dict) -> dict:
    """Build the shared versioned object from one ledger/decision/resource snapshot.

    Rich callers can supply active_trials, invocation_history, resource caps,
    planner_spend and prior_champions_by_contract. Legacy trace payloads are also
    accepted, with unavailable observations explicitly left unknown.
    """
    snapshot = copy.deepcopy(snapshot)
    output = snapshot.get("output_snapshot") or {}
    evidence = snapshot.get("planner_evidence") or {}
    resources = snapshot.get("resources", output.get("resources", {})) or {}
    counts = snapshot.get("counts", snapshot.get("ledger", output.get("counts", {}))) or {}
    usage = snapshot.get("planner_usage") or {}
    role = snapshot.get("trace_role", "decision" if snapshot.get("decision_id") else "execution")
    observed_at = snapshot.get("updated_at", snapshot.get("created_at"))
    trials = _active_trials(snapshot, observed_at)
    active_counts = [counts[key] for key in sorted(_ACTIVE) if key in counts]
    active_count = sum(active_counts) if active_counts else len(trials) if trials is not None else None
    completion = snapshot.get("completion") or {}
    outcome = snapshot.get("operation_status", snapshot.get("planner_status") if role == "decision"
                           else completion.get("status", snapshot.get("status"))) or "unknown"
    error = snapshot.get("error") or completion.get("error")
    blockers = []
    for value in (snapshot.get("blocker"), error, snapshot.get("last_planner_error")):
        if value and value not in blockers:
            blockers.append(value)
    if snapshot.get("planner_spend_unknown"):
        blockers.append("Planner cost unresolved; further paid calls frozen")
    admission = resources.get("admission_blockers") or {}
    for attempt, reasons in sorted(admission.items()):
        blockers.append(f"{attempt}: {_text(reasons)}")
    runtime_history = snapshot.get("runtime_history") or {}
    history = snapshot.get("invocation_history", runtime_history.get("invocations", resources.get("invocation_history")))
    incidents = snapshot.get("oom_history", runtime_history.get("oom_facts", resources.get("oom_history")))
    historical_oom = bool(incidents) or any(
        row.get("oom_kill", row.get("oom_count", 0)) or row.get("oom") for row in history or [])
    failed = outcome in _FAILURES
    health = snapshot.get("health") or ("degraded" if failed or blockers else
                                        "incident_history" if historical_oom else "unknown")
    spend = snapshot.get("planner_spend") or {}
    receipt_cost = reported_cost(usage)
    unknown_receipts = spend.get("unknown_receipt_ids", spend.get("unresolved_decision_ids", snapshot.get("unknown_receipt_ids")))
    unresolved = (bool(unknown_receipts) or spend.get("spend_unknown") is True
                  or snapshot.get("planner_spend_unknown") is True
                  or role == "decision" and receipt_cost is None and usage.get("cost_status") not in {"fixture", "not_dispatched"})
    total = spend.get("total_cost_usd", snapshot.get("planner_spend_usd", receipt_cost))
    known = spend.get("known_spend_usd", snapshot.get("known_spend_usd", usage.get("known_cost_usd", receipt_cost)))
    current = snapshot.get("champions_by_contract", output.get("champions_by_contract", {})) or {}
    prior = snapshot.get("prior_champions_by_contract")
    if prior is None:
        prior = {}
        # Existing planner evidence carries historical family champions as a list.
        # Select within each comparison contract, never across targets/populations.
        for candidate in snapshot.get("references", evidence.get("references", [])) or []:
            key, value = candidate.get("comparison_key"), candidate.get("objective_value")
            if not key or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            previous = prior.get(key)
            if previous is None or (value, str(candidate.get("attempt_id"))) < (previous["objective_value"], str(previous.get("attempt_id"))):
                prior[key] = candidate
    programs = snapshot.get("programs")
    families = {row.get("model_family") for row in trials or []}
    families.update(row.get("recipe", {}).get("model", {}).get("kind") for row in programs or [])
    families.update(row.get("recipe", {}).get("model", {}).get("kind") for row in current.values())
    rationale_keys = ("hypothesis", "reason", "review_reason", "unallocated_reason")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "identity": {"campaign_id": snapshot.get("campaign_id"), "role": role,
                     "sequence": snapshot.get("sequence"), "observed_at_utc": _utc(observed_at),
                     "invocation_id": snapshot.get("invocation_id", runtime_history.get("current_invocation_id", resources.get("invocation_id"))),
                     "code_revision": snapshot.get("code_revision", snapshot.get("revision")),
                     "evidence_id": snapshot.get("evidence_id"), "output_evidence_id": snapshot.get("output_evidence_id"),
                     "input_terminal_watermark": snapshot.get("input_terminal_watermark", evidence.get("terminal_watermark")),
                     "output_terminal_watermark": snapshot.get("output_terminal_watermark", output.get("terminal_watermark", snapshot.get("terminal_watermark")))},
        "operation": {"status": outcome, "failed": failed, "error": error,
                      "blocker": blockers[0] if blockers else None, "blockers": blockers,
                      "delivery_status": snapshot.get("delivery_status", "unknown")},
        "health": health, "model_families": sorted(family for family in families if family),
        "decision": {"decision_id": snapshot.get("decision_id", snapshot.get("latest_decision_id")),
                     "planner_model": snapshot.get("planner_model"), "programs": programs,
                     "program_ids": snapshot.get("program_ids"), "extensions": snapshot.get("extensions"),
                     "budgets": {"chosen": snapshot.get("trial_budget"), "allocated": snapshot.get("allocated_trials"),
                                 "held": snapshot.get("unallocated_trials"),
                                 "continuing": snapshot.get("continuing_trials"),
                                 "remaining_program_capacity": snapshot.get("remaining_program_capacity", evidence.get("remaining_program_capacity"))},
                     "rejected_fields": snapshot.get("rejected_fields"), "next_action": snapshot.get("next_action"),
                     "memo_ack": snapshot.get("research_memo_sha256"),
                     "rationale": {key: snapshot[key] for key in rationale_keys if key in snapshot}},
        "execution": {"counts": counts, "active_trial_count": active_count, "active_trials": trials,
                      "active_details_complete": trials is not None and active_count == len(trials),
                      "preparing_programs": snapshot.get("preparing_programs"),
                      "caps": {"effective_fit_cap": snapshot.get("capacity_ramp", {}).get("cap", resources.get("max_fits", resources.get("max_jobs"))),
                               "fit_ceiling": snapshot.get("max_fits", resources.get("fit_ceiling")),
                               "cpu_budget": resources.get("cpu_budget"), "ram_budget_gib": resources.get("ram_budget_gib")},
                      "resources": resources, "host_resources": snapshot.get("host_resources"),
                      "invocation_history": history, "oom_history": incidents, "historical_oom": historical_oom,
                      "restart_count": snapshot.get("restart_count", runtime_history.get("restart_count", resources.get("restart_count"))),
                      "pending_trace_delivery": snapshot.get("pending_trace_delivery"),
                      "pending_tracking": counts.get("pending_tracking"), "pending_tells": counts.get("pending_tells")},
        "champions": _champions(current, prior or {}),
        "billing": {"known_subtotal_usd": known, "total_cost_usd": None if unresolved else total,
                    "receipt_cost_usd": receipt_cost, "cost_status": usage.get("cost_status", "unknown"),
                    "unknown_receipt_ids": unknown_receipts, "total_unresolved": unresolved,
                    "not_dispatched_decision_ids": spend.get("not_dispatched_decision_ids"),
                    "transport_phase": snapshot.get("transport_phase", usage.get("transport_phase")),
                    "transport_attempts": snapshot.get("transport_attempts"),
                    "generation_id": snapshot.get("generation_id", usage.get("generation_id")),
                    "generation_link": snapshot.get("generation_link", usage.get("generation_link"))},
    }


def _text(value):
    if value is None:
        return "unknown"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def summary_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def summary_preview(value: dict) -> str:
    """Order operational signals before scientific progress and billing."""
    operation, execution, billing = value["operation"], value["execution"], value["billing"]
    blocker = " ".join(_text(operation["blocker"]).split())
    if len(blocker) > 240:
        blocker = blocker[:237] + "..."
    parts = [f"health={value['health']}", f"operation={operation['status']}", f"blocker={blocker}"]
    if execution["historical_oom"]:
        parts.append("historical OOM retained")
    if billing["transport_phase"]:
        parts.append(f"transport={billing['transport_phase']}")
    champion_text = []
    for row in value["champions"][:3]:
        delta = f", delta={row['absolute_delta']:+.6f}" if row["absolute_delta"] is not None else ", no comparable prior"
        key_id = hashlib.sha256(row["comparison_key"].encode()).hexdigest()[:10]
        champion_text.append(f"{row['objective']}={row['current_value']:.6f}{delta} [contract {key_id}]")
    parts.append("; ".join(champion_text) or "no comparable completed result yet")
    parts.append(f"known subtotal USD {_text(billing['known_subtotal_usd'])}, total USD {_text(billing['total_cost_usd'])}")
    parts.append(", ".join(f"{key}={count}" for key, count in sorted(execution["counts"].items())) or "counts unavailable")
    budgets = value["decision"]["budgets"]
    if value["identity"]["role"] == "decision":
        parts.append(", ".join(f"{key}={_text(budgets[key])}" for key in ("chosen", "allocated", "held", "continuing")))
    if execution["active_trial_count"] is not None and not execution["active_details_complete"]:
        parts.append("active trial details unavailable or incomplete")
    for row in execution["active_trials"] or []:
        parts.append(f"{_text(row['model_family'])} {row['status']}, fold={_text(row['fold'])}, elapsed={_text(row['elapsed_seconds'])}s, deadline={_text(row['deadline_at'])}")
    caps = execution["caps"]
    parts.append(f"fit cap={_text(caps['effective_fit_cap'])}, ceiling={_text(caps['fit_ceiling'])}")
    return "; ".join(parts)


def summary_markdown(value: dict) -> str:
    """Render the complete shared object, retaining exact rationale and diagnostics."""
    lines = ["# Research Summary", "", summary_preview(value), ""]
    for section, fields in value.items():
        if isinstance(fields, dict):
            lines.extend([f"## {section.replace('_', ' ').title()}", "", "| Field | Value |", "| --- | --- |"])
            for key, item in fields.items():
                text = _text(item).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "&#124;").replace("\n", "<br>")
                lines.append(f"| {key} | {text} |")
            lines.append("")
    lines.extend(["## Snapshot", "", "```json", summary_json(value).rstrip(), "```", ""])
    return "\n".join(lines)
