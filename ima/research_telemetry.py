"""One comparison and evidence snapshot contract for V6 planning and traces."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .research_store import utc_now
from .research_summary import reported_cost, summary, summary_markdown, summary_preview


_TRACKING_OPERATION_LOCK = threading.RLock()
_TRACING_INITIALIZATION = None


@contextmanager
def tracking_operation():
    """Serialize trace emission with model logging's process-global provider swaps."""
    with _TRACKING_OPERATION_LOCK:
        yield


def initialize_required_tracing(config):
    """Initialize V6 tracing before workers; never override disabled tracing/sampling."""
    global _TRACING_INITIALIZATION
    if not config.mlflow_tracking_uri:
        return None
    from .mlflow_tracking import _mlflow
    from .research_controller import _tracking_names
    from mlflow.environment_variables import MLFLOW_TRACE_SAMPLING_RATIO
    from mlflow.tracing.context import _USER_TRACE_CONTEXT
    from mlflow.tracing.provider import _get_tracer, is_tracing_enabled
    context = _USER_TRACE_CONTEXT.get()
    if context is not None and context.enabled is False:
        raise RuntimeError("Required V6 tracing is explicitly disabled in this context")
    ratio = MLFLOW_TRACE_SAMPLING_RATIO.get()
    if ratio is not None and ratio != 1:
        raise RuntimeError("Required V6 traces cannot use partial or disabled sampling; sampling policy was not changed")
    with tracking_operation():
        # Check before changing destinations: MLflow can reset the provider there.
        if not is_tracing_enabled():
            raise RuntimeError("Required V6 tracing provider is disabled; policy was not changed")
        mlflow = _mlflow()
        key = (config.mlflow_tracking_uri, config.research_policy)
        if (_TRACING_INITIALIZATION is not None and _TRACING_INITIALIZATION[0] == key
                and mlflow.get_tracking_uri() == config.mlflow_tracking_uri):
            _get_tracer(__name__)
            return dict(_TRACING_INITIALIZATION[1])
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(_tracking_names(config.research_policy)[0])
        # MLflow 3.15/3.16 has no public span-free initializer. This pinned SDK
        # accessor uses its once lock without creating a synthetic trace.
        _get_tracer(__name__)
        if not is_tracing_enabled():
            raise RuntimeError("Required V6 tracing provider is disabled; policy was not changed")
        report = {"tracking_uri":config.mlflow_tracking_uri,"experiment_id":str(experiment.experiment_id),"tracing_enabled":True}
        _TRACING_INITIALIZATION = (key, report)
        return dict(report)


def comparison_key(result: dict, recipe: dict) -> str:
    lineage = result.get("lineage", {})
    contract = {
        "target": recipe.get("target"),
        "objective": result.get("objective_name"),
        "protocol_id": lineage.get("protocol_id", lineage.get("protocol_hash")),
        "protocol_hash": lineage.get("protocol_hash"),
        "evaluation_population_id": lineage.get("evaluation_population_id"),
        "dataset_hash": None if lineage.get("evaluation_population_id") else lineage.get("dataset_hash"),
        "availability_policy": lineage.get("availability_policy", "legacy_unspecified"),
        "probability_basis": lineage.get("probability_basis", "legacy_unspecified"),
        "target_unit": lineage.get("target_unit"),
        "metric_contract_version": result.get("metrics", {}).get("metric_contract_version"),
    }
    # Missing identity is never evidence that two unrelated runs are comparable.
    if not contract["protocol_id"] or not (lineage.get("dataset_hash") or contract["evaluation_population_id"]):
        contract["unknown_identity_attempt"] = result.get("attempt_id", "unknown")
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def champions(terminal: list[dict]) -> dict:
    by_contract, by_family = {}, {}
    for row in terminal:
        result = row.get("result", {})
        value = result.get("objective_value")
        if row.get("status") != "completed" or value is None or not math.isfinite(value):
            continue
        recipe = row.get("payload", {}).get("recipe", {})
        key = comparison_key(dict(result, attempt_id=row["attempt_id"]), recipe)
        candidate = {"attempt_id": row["attempt_id"], "objective_name": result["objective_name"],
                     "objective_value": value, "recipe": recipe,
                     "comparison_key": key, "artifacts": result.get("artifacts", {})}
        family = key + ":" + recipe.get("model", {}).get("kind", "unknown")
        for mapping, index in ((by_contract, key), (by_family, family)):
            if index not in mapping or (value, row["attempt_id"]) < (
                    mapping[index]["objective_value"], mapping[index]["attempt_id"]):
                mapping[index] = candidate
    return {"champions_by_contract": by_contract, "champions_by_family": by_family,
            "terminal_watermark": len(terminal), "scope": "development; lower objectives are better"}


def evidence_snapshot(terminal: list[dict], **context) -> dict:
    terminal_counts = {"terminal": len(terminal), "completed": 0, "failed": 0, "pruned": 0}
    for row in terminal:
        status = row.get("status", "unknown")
        terminal_counts[status] = terminal_counts.get(status, 0) + 1
    payload = {"counts": terminal_counts, **context, **champions(terminal), "created_at": utc_now()}
    payload["evidence_id"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "created_at"},
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:24]
    return payload


def _trace_payload(campaign, payload, role, planner_evidence, output_snapshot):
    payload = dict(payload)
    if role != "decision":
        return payload
    if planner_evidence is None:
        planner_evidence = payload.get("planner_evidence")
    if planner_evidence is None and payload.get("decision_id"):
        path = campaign / "evidence" / f"{payload['decision_id']}.json"
        if path.exists():
            planner_evidence = json.loads(path.read_text())
    if planner_evidence is not None:
        if planner_evidence.get("decision_id") != payload.get("decision_id"):
            raise ValueError("Planner evidence belongs to a different decision")
        original_id = planner_evidence["evidence_id"]
        if output_snapshot is None and payload.get("evidence_id") != original_id:
            # Recover older callers that merged post-decision evidence over the input ID.
            output_snapshot = {k: payload[k] for k in (
                "evidence_id", "terminal_watermark", "champions_by_contract",
                "champions_by_family", "counts", "scope", "created_at") if k in payload}
        payload.update(evidence_id=original_id, planner_evidence=planner_evidence)
        payload["input_terminal_watermark"] = planner_evidence.get("terminal_watermark")
    if output_snapshot is not None:
        payload.update(output_snapshot=output_snapshot,
                       output_evidence_id=output_snapshot.get("evidence_id"),
                       output_terminal_watermark=output_snapshot.get("terminal_watermark"))
        for key in ("champions_by_contract", "champions_by_family", "counts"):
            if key in output_snapshot:
                payload[key] = output_snapshot[key]
    return payload


def _outbox(campaign):
    campaign.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(campaign / "trace-outbox.sqlite", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("""CREATE TABLE IF NOT EXISTS traces (
        role TEXT NOT NULL, number INTEGER NOT NULL, payload TEXT NOT NULL,
        tracking_uri TEXT NOT NULL, receipt TEXT, delivered INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, trace_json TEXT,
        PRIMARY KEY(role, number))""")
    if "trace_json" not in {row[1] for row in conn.execute("PRAGMA table_info(traces)")}:
        conn.execute("ALTER TABLE traces ADD COLUMN trace_json TEXT")
    conn.commit()
    return conn


def _canonical(payload):
    if isinstance(payload, dict):
        return {k: _canonical(v) for k, v in payload.items() if k != "created_at"}
    if isinstance(payload, list):
        return [_canonical(v) for v in payload]
    return payload


def log_snapshot(campaign: Path, payload: dict, config, *, role: str, number: int,
                 planner_evidence: dict | None = None,
                 output_snapshot: dict | None = None) -> dict | None:
    """Durably enqueue before logging; failed attempts remain available to drain."""
    if not config.mlflow_tracking_uri:
        return None
    if role not in {"decision", "execution", "dataset", "betting"} or number < 0:
        raise ValueError("Invalid trace role or sequence number")
    campaign = Path(campaign)
    payload = _trace_payload(campaign, payload, role, planner_evidence, output_snapshot)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    conn = _outbox(campaign)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM traces WHERE role=? AND number=?", (role, number)).fetchone()
        if row is not None:
            if (_canonical(json.loads(row["payload"])) != _canonical(payload)
                    or row["tracking_uri"] != config.mlflow_tracking_uri):
                raise ValueError("Conflicting replay of trace summary")
        else:
            published = campaign / "traces" / f"{role}-{number:06d}.json"
            receipt = json.loads(published.read_text()) if published.exists() else None
            if receipt is not None:
                usage = payload.get("planner_usage", {})
                expected_cost = reported_cost(usage) if role == "decision" else None
                if (receipt.get("role") != role or receipt.get("number") != number
                        or receipt.get("total_cost_usd") != expected_cost):
                    raise ValueError("Conflicting replay of historical trace linkage")
            conn.execute("INSERT INTO traces(role,number,payload,tracking_uri,receipt,delivered) VALUES(?,?,?,?,?,?)",
                         (role, number, encoded, config.mlflow_tracking_uri,
                          json.dumps(receipt) if receipt is not None else None, int(receipt is not None)))
        conn.commit()
    finally:
        conn.close()
    return _deliver_trace(campaign, config, role, number)


def _deliver_trace(campaign, config, role, number):
    from .mlflow_tracking import _mlflow, _write_json_atomic
    conn = _outbox(campaign)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM traces WHERE role=? AND number=?", (role, number)).fetchone()
        if row is None:
            raise ValueError("Unknown trace outbox entry")
        if row["tracking_uri"] != config.mlflow_tracking_uri:
            raise ValueError("Trace destination changed; refusing cross-server replay")
        if row["delivered"]:
            return json.loads(row["receipt"])
        conn.execute("UPDATE traces SET attempts=attempts+1 WHERE role=? AND number=?", (role, number))
        if row["receipt"]:
            receipt = json.loads(row["receipt"])
            trace_json = row["trace_json"]
        else:
            def persist(receipt, trace_body):
                conn.execute("UPDATE traces SET receipt=?,trace_json=? WHERE role=? AND number=?",
                             (json.dumps(receipt, sort_keys=True), json.dumps(trace_body, sort_keys=True), role, number))
                conn.commit()
            receipt, trace_body = _emit_trace(
                campaign, json.loads(row["payload"]), config, role=role, number=number, persist=persist)
            trace_json = json.dumps(trace_body, sort_keys=True)
        # Identity is persisted before the cost-bearing span ends, not just before flush.
        conn.commit()
        mlflow = _mlflow()
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.flush_trace_async_logging()
        _verify_or_restore_trace(mlflow, config.mlflow_tracking_uri, receipt, trace_json)
        conn.execute("BEGIN IMMEDIATE")
        _write_json_atomic(campaign / "traces" / f"{role}-{number:06d}.json", receipt)
        conn.execute("UPDATE traces SET delivered=1,last_error=NULL WHERE role=? AND number=?", (role, number))
        conn.commit()
        return receipt
    except Exception as exc:
        conn.execute("UPDATE traces SET last_error=? WHERE role=? AND number=?", (str(exc), role, number))
        conn.commit()
        raise
    finally:
        conn.close()


def drain_trace_outbox(campaign: Path, config, *, limit: int = 20) -> dict:
    """Retry pending summaries independently; safe to call at startup and each checkpoint."""
    campaign = Path(campaign)
    if limit < 0:
        raise ValueError("Trace drain limit cannot be negative")
    if not config.mlflow_tracking_uri or not (campaign / "trace-outbox.sqlite").exists():
        return {"delivered": 0, "pending": 0, "errors": []}
    conn = _outbox(campaign)
    try:
        rows = conn.execute("SELECT role,number FROM traces WHERE delivered=0 ORDER BY role,number LIMIT ?",
                            (limit,)).fetchall()
    finally:
        conn.close()
    delivered, errors = 0, []
    for row in rows:
        try:
            _deliver_trace(campaign, config, row["role"], row["number"])
            delivered += 1
        except Exception as exc:
            errors.append({"role": row["role"], "number": row["number"], "error": str(exc)})
    conn = _outbox(campaign)
    try:
        pending = conn.execute("SELECT COUNT(*) FROM traces WHERE delivered=0").fetchone()[0]
    finally:
        conn.close()
    return {"delivered": delivered, "pending": pending, "errors": errors}


def next_trace_number(campaign: Path, role: str) -> int:
    """Single-controller sequence includes undelivered entries, not just published links."""
    if role not in {"decision", "execution", "dataset", "betting"}:
        raise ValueError("Invalid trace role")
    campaign = Path(campaign)
    numbers = [int(path.stem.rsplit("-", 1)[1]) for path in (campaign / "traces").glob(f"{role}-*.json")]
    if (campaign / "trace-outbox.sqlite").exists():
        conn = _outbox(campaign)
        try:
            number = conn.execute("SELECT MAX(number) FROM traces WHERE role=?", (role,)).fetchone()[0]
            if number is not None:
                numbers.append(number)
        finally:
            conn.close()
    return max(numbers, default=0) + 1


def _verify_or_restore_trace(mlflow, tracking_uri, receipt, trace_json):
    """MLflow exporters can swallow errors; require backend readback before delivery."""
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    try:
        trace = client.get_trace(receipt["trace_id"])
        if trace is not None:
            return
    except Exception:
        pass
    if not trace_json:
        raise RuntimeError("Trace is not readable and no serialized retry body is available")
    from mlflow.entities import Trace
    from mlflow.tracing.client import TracingClient
    trace = Trace.from_dict(json.loads(trace_json))
    transport = TracingClient(tracking_uri=tracking_uri)
    info = transport.start_trace(trace.info)
    if info.tags.get("mlflow.trace.spansLocation") == "TRACKING_STORE":
        transport.log_spans(receipt["experiment_id"], trace.data.spans)
    else:
        # Reuse MLflow's artifact transport, preserving the original trace/span IDs.
        transport._upload_trace_data(info, trace.data)
    if client.get_trace(receipt["trace_id"]) is None:
        raise RuntimeError("Trace upload completed without readable backend evidence")


def _require_recording_span(span):
    from mlflow.entities import NoOpSpan
    if isinstance(span,NoOpSpan):
        raise RuntimeError("MLflow returned NoOpSpan; no recording trace was created. Delivery remains pending; check tracing enablement, sampling and provider initialization.")


def _serialize_trace(span, experiment_id, tags, name, preview, campaign, usage):
    # NoOpSpan inherits a serializer that assumes a recording OpenTelemetry span.
    _require_recording_span(span)
    from mlflow.entities import Span, Trace, TraceData, TraceInfo, TraceLocation, TraceState
    metadata = {"mlflow.trace.session": str(campaign.resolve())}
    tokens = {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")
              if usage.get(k) is not None}
    if tokens:
        metadata["mlflow.trace.tokenUsage"] = json.dumps(tokens)
    cost = reported_cost(usage)
    if cost is not None:
        metadata["mlflow.trace.cost"] = json.dumps({"total_cost": float(cost)})
    end_time = span.end_time_ns or time.time_ns()
    serialized = span.to_dict()
    serialized["end_time_unix_nano"] = end_time
    failed = tags.get("ima.operation_status") in {"failed", "error", "timeout", "timed_out", "interrupted"}
    failed = failed or serialized.get("status", {}).get("code") == "STATUS_CODE_ERROR"
    if failed:
        serialized["status"] = {"code": "STATUS_CODE_ERROR",
                                "message": serialized.get("status", {}).get("message") or tags.get("ima.blocker", "")}
    elif serialized.get("status", {}).get("code") in {None, "STATUS_CODE_UNSET"}:
        serialized["status"] = {"code": "STATUS_CODE_OK", "message": ""}
    if tokens:
        serialized["attributes"]["mlflow.chat.tokenUsage"] = json.dumps(tokens)
    if cost is not None:
        serialized["attributes"]["mlflow.llm.cost"] = metadata["mlflow.trace.cost"]
    info = TraceInfo(trace_id=span.trace_id,
                     trace_location=TraceLocation.from_experiment_id(str(experiment_id)),
                     request_time=span.start_time_ns // 1_000_000,
                     execution_duration=(end_time - span.start_time_ns) // 1_000_000,
                     state=TraceState.ERROR if failed else TraceState.OK, tags=tags, trace_metadata=metadata,
                     request_preview=name, response_preview=preview)
    return Trace(info, TraceData(spans=[Span.from_dict(serialized)])).to_dict()


def _paper_preview(campaign: Path, payload: dict) -> str:
    completion = payload.get("completion") or {}
    preview = f"request {payload.get('request_id', 'unknown')}; status={completion.get('status', 'unknown')}"
    if completion.get("status") != "completed":
        return preview + f"; {completion.get('error') or 'report unavailable'}"
    action = payload.get("action_id", "")
    if len(action) != 24 or any(character not in "0123456789abcdef" for character in action):
        raise ValueError("Invalid paper trace action identity")
    path = campaign.resolve()/"paper-actions"/(action+".json")
    if (path.is_symlink() or path.resolve().parent != campaign.resolve()/"paper-actions"
            or Path(completion["report_path"]).resolve() != path):
        raise ValueError("Paper trace report must belong to the current campaign action")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != completion["report_sha256"]:
        raise ValueError("Paper trace report hash mismatch")
    body = json.loads(content)
    report = body["report"]
    if (body["action_id"] != action or report["request_id"] != payload.get("request_id")
            or report["evidence_id"] != payload.get("evidence_id")
            or report.get("paper_only") is not True or report.get("executable_evidence") is not False):
        raise ValueError("Paper trace report identity or offline-only contract mismatch")
    coverage = report.get("coverage", {})
    return (f"{preview}; races={coverage.get('evaluated_races', 'unknown')}/"
            f"{coverage.get('requested_races', 'unknown')} requested, "
            f"available={coverage.get('available_races', 'unknown')}; "
            f"models={len(report['model_ids'])}; basis={report['probability_basis']}; "
            f"quote_mode={report['request']['quote_mode']}; paper only")


def _emit_trace(campaign: Path, payload: dict, config, *, role: str, number: int, persist) -> tuple[dict, dict]:
    with tracking_operation():
        return _emit_trace_locked(campaign,payload,config,role=role,number=number,persist=persist)


def _emit_trace_locked(campaign: Path, payload: dict, config, *, role: str, number: int, persist) -> tuple[dict, dict]:
    from .mlflow_tracking import _mlflow
    from .research_controller import _tracking_names
    short = campaign.name
    label = {"decision":"decision D","execution":"execution S","dataset":"dataset B","betting":"paper P"}[role]
    name = f"{short} | {label}{number:06d}"
    usage = payload.get("planner_usage", {}) if role == "decision" else {}
    fixture = usage.get("cost_status") == "fixture"
    native_usage = {} if fixture else dict(usage)
    cost = reported_cost(native_usage)
    if cost is None:
        native_usage.pop("total_cost_usd", None)
    written = summary(dict(payload, campaign_id=short, trace_role=role, sequence=number,
                           delivery_status="pending"))
    markdown = summary_markdown(written)
    from .mlflow_tracking import _write_json_atomic
    artifact_dir = campaign / "traces" / f"{role}-{number:06d}"
    _write_json_atomic(artifact_dir / "summary.json", written)
    temporary = artifact_dir / "summary.md.tmp"
    temporary.write_text(markdown)
    temporary.replace(artifact_dir / "summary.md")
    preview = summary_preview(written)
    if role == "betting":
        preview = _paper_preview(campaign,payload)
    tags = {"ima.campaign_id": short, "ima.trace_role": role,
            "ima.evidence_id": payload.get("evidence_id", "unknown"),
            "ima.decision_id": str(payload.get("decision_id", "none")),
            "ima.planner_cost_status": str(usage.get("cost_status", "not_a_planner_call")),
            "ima.operation_status": written["operation"]["status"],
            "ima.blocker": str(written["operation"]["blocker"] or "none")[:1000],
            "ima.model_families": ",".join(written["model_families"]),
            "ima.active_trial_count": str(written["execution"]["active_trial_count"]
                                           if written["execution"]["active_trial_count"] is not None else "unknown"),
            "ima.completed_count": str(written["execution"]["counts"].get("completed", "unknown")),
            "ima.summary_schema_version": str(written["schema_version"]),
            "mlflow.traceName": name}
    tags["ima.summary_id"] = f"{campaign.resolve()}:{role}:{number}"
    if payload.get("output_evidence_id"):
        tags["ima.output_evidence_id"] = payload["output_evidence_id"]
    if cost is not None:
        tags["planner_cost_usd"] = str(cost)
    mlflow = _mlflow()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(_tracking_names(config.research_policy)[0])
    with mlflow.start_span(name=name, span_type="LLM" if role == "decision" and not fixture else "CHAIN") as span:
        _require_recording_span(span)
        trace_id = span.trace_id
        mlflow.update_current_trace(tags=tags, session_id=str(campaign.resolve()),
                                   request_preview=name, response_preview=preview)
        span.set_inputs(payload.get("planner_evidence") or {
            "evidence_id": payload.get("evidence_id"), "decision_id": payload.get("decision_id")})
        span.set_outputs(dict(payload, summary=written, summary_markdown=markdown))
        if written["operation"]["failed"]:
            from mlflow.entities import SpanStatus, SpanStatusCode
            span.set_status(SpanStatus(SpanStatusCode.ERROR, str(written["operation"]["error"] or
                                                               written["operation"]["blocker"] or written["operation"]["status"])))
        linkage = {"trace_id": trace_id, "experiment_id": str(experiment.experiment_id),
                   "role": role, "number": number, "total_cost_usd": cost,
                   "evidence_id": payload.get("evidence_id"),
                   "output_evidence_id": payload.get("output_evidence_id")}
        trace_body = _serialize_trace(span, experiment.experiment_id, tags, name, preview, campaign, native_usage)
        persist(linkage, trace_body)
        if cost is not None:
            span.set_attribute("mlflow.llm.cost", {"total_cost": float(cost)})
        tokens = {k: native_usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")
                  if native_usage.get(k) is not None}
        if tokens:
            span.set_attribute("mlflow.chat.tokenUsage", tokens)
    return linkage, trace_body
