"""Bounded HTTP delivery with early, credential-free transport receipts."""
from __future__ import annotations

import asyncio
import time
from typing import Any


def post_json_bounded(url: str, payload: dict[str, Any], config) -> dict[str, Any]:
    import httpx

    started = time.monotonic()
    observer = getattr(config, "transport_observer", None)

    def emit(phase, **metadata):
        if observer is not None:
            observer({"schema_version": 1, "phase": phase,
                      "elapsed_seconds": time.monotonic() - started, **metadata})

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
                    emit("response_complete")
                    return result

    try:
        return asyncio.run(send())
    except Exception as exc:
        emit("request_failed", exception_type=type(exc).__name__)
        raise
