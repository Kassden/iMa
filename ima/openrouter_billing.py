"""Recover missing physical billing receipts without replaying inference."""

import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx


ATTEMPT = re.compile(r"D[0-9]{6}-[0-9]{2,}\.json\Z")
GENERATION = re.compile(r"gen-[A-Za-z0-9_-]+\Z")


def billing_receipt(payload, generation_id, transport, revision):
    if not isinstance(payload, dict):
        raise ValueError("Malformed provider payload")
    data = payload.get("data", {})
    if not isinstance(data, dict) or data.get("id") != generation_id:
        raise ValueError("Generation identity mismatch")
    cost = data.get("total_cost")
    try:
        valid_cost = not isinstance(cost, bool) and isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0
    except OverflowError:
        valid_cost = False
    if not valid_cost:
        raise ValueError("Authoritative charge is unavailable or invalid")
    finish_reason = data.get("finish_reason")
    if data.get("cancelled") is not True and not (isinstance(finish_reason, str) and finish_reason.strip()):
        raise ValueError("Provider generation is not finalized")
    usage = {"cost": cost}
    for name in ("prompt", "completion"):
        value = data.get("native_tokens_" + name, data.get("tokens_" + name))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[name + "_tokens"] = value
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return {
        "response": {"id": generation_id, "model": data.get("model"), "usage": usage},
        "evidence_id": transport.get("evidence_id"),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "billing_only": True,
        "billing_source": "openrouter_generation_api",
        "recovery_revision": revision,
        "provider_generation": payload,
    }


def _publish(path, receipt):
    descriptor, temporary = tempfile.mkstemp(prefix=".billing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(receipt, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        os.unlink(temporary)


def reconcile(campaign, api_key, *, revision, max_requests=3, timeout=15, client=None):
    """Bounded GET-only recovery; original responses are never replaced."""
    campaign = Path(campaign)
    if campaign.is_symlink() or campaign.resolve() != campaign.absolute() or not campaign.is_dir():
        raise ValueError("Campaign must be a canonical directory")
    if not 1 <= max_requests <= 3 or not 0 < timeout <= 15:
        raise ValueError("Recovery bounds exceeded")
    calls, transports = campaign / "planner-calls", campaign / "planner-transport"
    for directory in (calls, transports):
        if directory.is_symlink():
            raise ValueError("Receipt directories cannot be symlinks")
    calls.mkdir(exist_ok=True, mode=0o700)
    report = {"recovered": [], "skipped": [], "unresolved": [], "requests": 0}
    cursor = campaign / ".billing-recovery-cursor.json"
    if cursor.is_symlink():
        raise ValueError("Recovery cursor cannot be a symlink")
    paths = sorted(transports.glob("*.json"))
    # Reject duplicated physical identities even when one receipt already exists.
    owners = {}
    for path in paths:
        try:
            if path.is_symlink():
                continue
            item = json.loads(path.read_text())
            events = item.get("events", []) if isinstance(item, dict) else []
            if not isinstance(events, list):
                continue
            for event in events:
                identity = event.get("generation_id") if isinstance(event, dict) else None
                if isinstance(identity, str) and identity:
                    owners.setdefault(identity, set()).add(path.name)
        except (ValueError, OSError):
            pass
    for path in calls.glob("*.json"):
        try:
            if path.is_symlink():
                continue
            item = json.loads(path.read_text())
            response = item.get("response", {}) if isinstance(item, dict) else {}
            identity = response.get("id") if isinstance(response, dict) else None
            if isinstance(identity, str) and identity:
                owners.setdefault(identity, set()).add(path.name)
        except (ValueError, OSError):
            pass
    last = None
    if cursor.exists():
        state = json.loads(cursor.read_text())
        if not isinstance(state, dict) or not isinstance(state.get("last"), str):
            raise ValueError("Malformed recovery cursor")
        last = state["last"]
        paths = [p for p in paths if p.name > last] + [p for p in paths if p.name <= last]
    owned_client = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        for path in paths:
            try:
                if path.is_symlink() or not ATTEMPT.fullmatch(path.name) or int(path.stem.split("-")[1]) < 1:
                    raise ValueError("Unsafe physical-attempt path")
                destination = calls / path.name
                if destination.exists() or destination.is_symlink():
                    report["skipped"].append(path.name)
                    continue
                transport = json.loads(path.read_text())
                if not isinstance(transport, dict):
                    raise ValueError("Malformed transport record")
                events = transport.get("events", [])
                if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
                    raise ValueError("Malformed transport events")
                identities = {event["generation_id"] for event in events if event.get("generation_id")}
                if not identities:
                    continue
                if len(identities) != 1:
                    raise ValueError("Ambiguous generation identity")
                # In-flight generation charges are not final; let the response observer finish.
                if not events or events[-1].get("phase") != "request_failed":
                    continue
                generation_id = identities.pop()
                if not isinstance(generation_id, str) or not GENERATION.fullmatch(generation_id):
                    raise ValueError("Unsafe generation identity")
                if len(owners.get(generation_id, set())) != 1:
                    raise ValueError("Generation is attached to multiple physical attempts")
                if report["requests"] >= max_requests:
                    break
                report["requests"] += 1
                last = path.name
                response = client.get(
                    "https://openrouter.ai/api/v1/generation",
                    params={"id": generation_id},
                    headers={"Authorization": "Bearer " + api_key},
                    timeout=timeout,
                )
                response.raise_for_status()
                receipt = billing_receipt(response.json(), generation_id, transport, revision)
                if json.loads(path.read_text()) != transport:
                    raise ValueError("Transport changed during reconciliation")
                published = _publish(destination, receipt)
                report["recovered" if published else "skipped"].append(path.name)
            except (ValueError, OSError, httpx.HTTPError, TypeError, KeyError):
                # Do not log provider bodies, request headers or exception text containing credentials.
                report["unresolved"].append(path.name)
    finally:
        if owned_client:
            client.close()
        if last is not None:
            descriptor, temporary = tempfile.mkstemp(prefix=".billing-cursor-", dir=campaign)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump({"last": last}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, cursor)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return report
