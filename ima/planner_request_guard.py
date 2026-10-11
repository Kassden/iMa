"""Request-owned identities, bounded context and durable rejection protection.

The guard does not authorize science: the campaign's admission validators still
own recipes, datasets, parents and trial budgets. Raw provider records are never
rewritten when an omitted request envelope is supplied here.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile


VERSION = 1
CONTEXT_BYTES = 300_000
REQUEST_BYTES = 380_000


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def bind_response(response, evidence):
    result = deepcopy(response)
    decision = result["decision"]
    for key in ("decision_id", "evidence_id"):
        if decision.get(key) != evidence[key]:
            raise ValueError("Planner request identity mismatch: " + key)
    parents = {row["attempt_id"] for row in evidence.get("completed_trial_index", [])}
    bound = []
    for proposal in decision.get("programs", []):
        ids = proposal.get("evidence_ids", [])
        if ids == [] or ids == ():
            proposal["evidence_ids"] = [evidence["evidence_id"]]
            bound.append(proposal["proposal_id"])
        elif ids not in ([evidence["evidence_id"]], (evidence["evidence_id"],)):
            raise ValueError("Proposal has explicit stale evidence")
        if not set(proposal.get("parent_trial_ids", [])) <= parents:
            raise ValueError("Unknown parent trial")
    from .research_expansion import PlannerDecision
    result["decision"] = PlannerDecision.model_validate(decision).model_dump(mode="json")
    result["proposal_identity_binding"] = {
        "version": VERSION, "decision_id": evidence["decision_id"],
        "evidence_id": evidence["evidence_id"], "bound_proposal_ids": bound,
        "raw_response_sha256": hashlib.sha256(encoded(response.get("raw_response"))).hexdigest(),
        "rule": "Only absent/empty proposal evidence IDs inherit the verified current request",
    }
    return result


def _historical_summary(value, depth=0):
    """Bound historical numeric searches, with explicit counts instead of silent truncation."""
    if isinstance(value, list):
        if len(value) > 8:
            return {"count": len(value), "preview": [_historical_summary(v, depth+1) for v in value[:3]],
                    "omitted": len(value)-3, "full_values": "durable full evidence/report"}
        return [_historical_summary(v, depth+1) for v in value]
    if isinstance(value, dict):
        if depth > 5:
            return {"keys": list(value), "details": "durable full evidence/report"}
        return {k: _historical_summary(v, depth+1) for k, v in value.items()}
    if isinstance(value, str) and len(value) > 1200:
        return {"preview": value[:1200], "omitted_characters": len(value)-1200}
    return value


def bounded_context(projected, max_bytes=CONTEXT_BYTES):
    view = deepcopy(projected)
    receipt = view.setdefault("planner_decision_view", {})
    if receipt.get("request_guard_version") == VERSION:
        if len(encoded(view)) > max_bytes:
            raise ValueError("Mandatory planner context exceeds byte budget")
        return view
    before = len(encoded(view))
    evidence = view.get("evidence", view)
    paper = evidence.get("paper_research", {})
    for records in paper.get("by_comparison", {}).values():
        for row in records:
            # Eligibility, taint, coverage, assumptions and identities stay exact.
            for field in ("metrics", "pools", "model_weights", "summary"):
                if field in row:
                    row[field] = _historical_summary(row[field])
    capacities = evidence.get("remaining_program_capacity")
    if isinstance(capacities, dict):
        evidence["remaining_program_capacity"] = {k: v for k, v in capacities.items() if v != 0}
    discipline = evidence.get("experiment_discipline", {})
    for outcome in discipline.get("recent_outcomes", []):
        if isinstance(outcome, dict):
            outcome.pop("artifacts", None)
            outcome.pop("runtime", None)
    receipt.update(request_guard_version=VERSION, context_byte_limit=max_bytes,
                   bytes_before_request_guard=before,
                   request_guard_omissions={
                       "historical_paper_searches": "Counted previews only; full evidence/report remains durable",
                       "zero_capacity_programs": "Exhausted programs omitted, positive capacity retained",
                       "outcome_runtime_artifacts": "Execution paths omitted, science/metrics retained"})
    receipt["request_guard_projection_sha256"] = hashlib.sha256(encoded(view)).hexdigest()
    size = len(encoded(view))
    if size > max_bytes:
        raise ValueError(f"Mandatory planner context exceeds byte budget: {size}>{max_bytes}; no paid call")
    return view


class PaidCallCircuit:
    def __init__(self, campaign):
        self.campaign = Path(campaign)
        self.root = self.campaign / "planner-request-guard"

    def check_and_record(self, evidence):
        decision_id = evidence["decision_id"]
        if not re.fullmatch(r"D[0-9]+", decision_id):
            raise ValueError("Invalid planner decision filename")
        destination = self.root / (decision_id + ".json")
        if destination.exists():
            raise ValueError("Guarded decision was already dispatched; refuse duplicate paid request")
        watermark = evidence.get("terminal_watermark")
        if watermark is None:
            raise ValueError("Missing terminal watermark for paid-call protection")
        failures = []
        for path in sorted(self.root.glob("D*.json"), reverse=True):
            record = json.loads(path.read_text())
            if record["version"] != VERSION or record["terminal_watermark"] != watermark:
                break
            terminal = self.campaign / "decisions" / path.name
            if not terminal.exists():
                raise ValueError("Previous guarded planner request has no durable decision")
            result = json.loads(terminal.read_text())
            if result.get("planner_status") != "failed":
                break
            error = result.get("error", "")
            if not error or "Timeout" in error or "transport" in error.lower():
                break
            failures.append(error)
            if len(failures) == 2:
                if failures[0] == failures[1]:
                    raise ValueError("Paid-call circuit open: repeated rejection without new terminal evidence: " + error[:250])
                break
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", dir=self.root, delete=False) as stream:
            stream.write(encoded({"version": VERSION, "decision_id": decision_id,
                                  "evidence_id": evidence["evidence_id"],
                                  "terminal_watermark": watermark}))
            temporary = Path(stream.name)
        temporary.replace(destination)


def install_guard(campaign, context_bytes=CONTEXT_BYTES, request_bytes=REQUEST_BYTES):
    from . import openrouter_orchestrator as orchestrator
    from . import research_planner_context as context
    if getattr(orchestrator.choose_research_decision, "request_guard_version", None):
        raise ValueError("Planner request guard already installed")
    original_choose = orchestrator.choose_research_decision
    original_context = context.decision_planner_evidence
    original_post = orchestrator._post_json
    circuit = PaidCallCircuit(campaign)

    def project(bundle):
        return bounded_context(original_context(bundle), context_bytes)

    def post(url, payload, config):
        # Static schemas, dynamic context and schema-repair candidates all count.
        size = len(encoded(payload))
        if size > request_bytes:
            raise ValueError(f"Planner HTTP request exceeds byte budget: {size}>{request_bytes}; no paid call")
        return original_post(url, payload, config)

    def choose(evidence, limits, config):
        circuit.check_and_record(evidence)
        return bind_response(original_choose(evidence, limits, config), evidence)

    choose.request_guard_version = VERSION
    context.decision_planner_evidence = project
    orchestrator._post_json = post
    # The deployed policy compiles the planner with an isolated globals mapping.
    original_choose.__globals__["_post_json"] = post
    orchestrator.choose_research_decision = choose
    return {"request_guard_version": VERSION, "context_bytes": context_bytes,
            "request_bytes": request_bytes, "scientific_source_changed": False}
