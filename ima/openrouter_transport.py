"""Bounded HTTP delivery with early, credential-free transport receipts."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping
from typing import Any


def classify_transport_attempt(events: Iterable[Mapping[str, Any]] | None) -> dict[str, str]:
    """Classify dispatch evidence, independently of success or reported billing.

    Returns classification and a bounded, credential-free reason. Only a
    complete, matched TCP failure (including DNS errors) proves an attempt unsent. TLS,
    exception names alone, and truncated receipts cannot authorize a retry.
    Persisted classification events are ignored so replay uses original evidence.
    """
    unknown = {"classification": "unknown", "reason": "incomplete_or_ambiguous_evidence"}
    if events is None or isinstance(events, (str, bytes, Mapping)):
        return unknown
    try:
        iterator = iter(events)
    except TypeError:
        return unknown

    phases = []
    incomplete = False
    for event in iterator:
        if not isinstance(event, Mapping) or not isinstance(event.get("phase"), str):
            incomplete = True
            continue
        phase = event["phase"]
        # Even a failed send may have transmitted bytes; later failures cannot
        # revoke this evidence (including a DNS failure on a subsequent retry).
        protocol, separator, operation = phase.partition(".")
        operation, _, state = operation.rpartition(".")
        if phase in {"response_headers", "response_complete"} or (
            separator and protocol in {"http11", "http2"}
            and operation in {"send_request_headers", "send_request_body",
                              "receive_response_headers", "receive_response_body"}
            and state in {"started", "complete", "failed"}
        ):
            return {"classification": "possibly_dispatched",
                    "reason": "http_send_or_response_observed"}
        if phase != "transport_classification":
            phases.append(phase)

    if incomplete or not phases or phases[0] != "request_started":
        return unknown
    if phases[-1] != "request_failed" or phases.count("request_started") != 1:
        return unknown
    if phases.count("request_failed") != 1:
        return unknown

    pending = set()
    failed = False
    for phase in phases[1:-1]:
        operation, separator, state = phase.rpartition(".")
        if operation == "connection.close" and state in {"started", "complete", "failed"}:
            continue
        if not separator or operation not in {"connection.resolve_dns", "connection.connect_tcp"}:
            return unknown
        if state == "started" and operation not in pending:
            pending.add(operation)
            failed = False
        elif state == "failed" and operation in pending:
            pending.remove(operation)
            failed = operation == "connection.connect_tcp"
        elif state == "complete" and operation == "connection.resolve_dns" and operation in pending:
            pending.remove(operation)
        else:
            return unknown
    if failed and not pending:
        return {"classification": "not_dispatched", "reason": "complete_pre_send_connect_failure"}
    return unknown


def classify_transport_attempts(
    attempts: Iterable[Iterable[Mapping[str, Any]]] | None,
) -> dict[str, str]:
    """Combine physical-attempt event lists; every attempt must be proven unsent.

    A sent or ambiguous sibling keeps the aggregate unknown. Empty/missing
    attempts remain unknown, including a missing receipt among unsent attempts.
    """
    unknown = {"classification": "unknown", "reason": "not_all_attempts_proven_unsent"}
    if attempts is None or isinstance(attempts, (str, bytes, Mapping)):
        return unknown
    try:
        iterator = iter(attempts)
    except TypeError:
        return unknown
    classifications = [classify_transport_attempt(events)["classification"] for events in iterator]
    if classifications and all(value == "not_dispatched" for value in classifications):
        return {"classification": "not_dispatched", "reason": "all_attempts_proven_unsent"}
    return unknown


def post_json_bounded(url: str, payload: dict[str, Any], config) -> dict[str, Any]:
    import httpx

    started = time.monotonic()
    observer = getattr(config, "transport_observer", None)
    events = []

    def emit(phase, **metadata):
        event = {"schema_version": 1, "phase": phase,
                 "elapsed_seconds": time.monotonic() - started, **metadata}
        events.append(event)
        if observer is not None:
            observer(dict(event))

    def finish(phase, **metadata):
        # Include the terminal marker for classification, while keeping legacy
        # observers' response_complete/request_failed event last in the stream.
        classification = classify_transport_attempt([*events, {"phase": phase}])
        emit("transport_classification", **classification)
        emit(phase, **metadata)

    async def trace(event, info):
        if event.endswith((".started", ".complete", ".failed")):
            failure = info.get("exception")
            emit(event, **({"exception_type": type(failure).__name__}
                           if failure is not None else {}))

    async def send():
        async with asyncio.timeout(config.absolute_deadline_seconds):
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                emit("request_started")
                async with client.stream(
                    "POST", url, json=payload,
                    headers={"Authorization": f"Bearer {config.api_key}",
                             "Content-Type": "application/json"},
                    extensions={"trace": trace},
                ) as response:
                    # Save the generation identity before awaiting the full body.
                    emit("response_headers", status_code=response.status_code,
                         generation_id=response.headers.get("x-generation-id"),
                         request_id=response.headers.get("x-request-id"))
                    response.raise_for_status()
                    await response.aread()
                    result = response.json()
                    if not isinstance(result, dict):
                        raise ValueError("OpenRouter response must be a JSON object")
                    if config.response_observer:
                        config.response_observer(result)
                    return result

    try:
        result = asyncio.run(send())
    except Exception as exc:
        finish("request_failed", exception_type=type(exc).__name__)
        raise
    finish("response_complete")
    return result
