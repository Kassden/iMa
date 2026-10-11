"""Request-owned identities, bounded context and durable rejection protection.

The guard does not authorize science: the campaign's admission validators still
own recipes, datasets, parents and trial budgets. Raw provider records are never
rewritten when an omitted request envelope is supplied here.
"""

from copy import deepcopy
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import tempfile
import textwrap
from types import CodeType


VERSION = 3
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
    raw = response.get("raw_response", {})
    if isinstance(raw, dict) and raw.get("choices"):
        from .openrouter_orchestrator import _json_message_from_response
        provider_decision = _json_message_from_response(raw)
        if isinstance(provider_decision, dict):
            for proposal in provider_decision.get("programs", []):
                if proposal.get("evidence_ids", []) == [] and proposal["proposal_id"] not in bound:
                    bound.append(proposal["proposal_id"])
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
    evidence["request_validation_contract"] = {
        "fixed_controls": "Copy every unchanged parent recipe field exactly, including transforms. "
                          "If changing transforms as well as a model/feature, declare transform among changed_axes "
                          "and describe a joint test, not a single-axis ablation.",
        "fold_local_graph_limit": "With recipe.transforms or feature_discovery, win-probability graph "
                                 "estimator nodes support logit, boosted, benter_conditional_logit and gaussian_probit; "
                                 "catboost_classifier and performance_distribution ancestors are unsupported on that path.",
    }
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


def effective_dataset_audit(original, decisions, evidence, default_dataset_id):
    """Compare effective data identities without changing executable recipes."""
    def normalize(value):
        if isinstance(value, list):
            return [normalize(row) for row in value]
        if isinstance(value, dict):
            result = {key: normalize(row) for key, row in value.items()}
            recipe = result.get("recipe")
            if isinstance(recipe, dict) and recipe.get("dataset_ref") is None:
                recipe["dataset_ref"] = default_dataset_id
            return result
        return value
    result = original(normalize(decisions), normalize(evidence))
    result["dataset_identity_normalization"] = {
        "default_dataset_id": default_dataset_id,
        "scope": "Comparison copies only; execution recipes and provider responses unchanged",
    }
    return result


def semantic_preflight(decision, evidence):
    from .research_expansion import PlannerDecision
    from .proposal_audit import audit_decisions
    bound = bind_response({"decision": decision.model_dump(mode="json")}, evidence)["decision"]
    candidate = PlannerDecision.model_validate(bound)
    fixed = {p.proposal_id for p in candidate.programs if p.fixed_parameters and p.parent_trial_ids}
    audit = audit_decisions([bound], evidence)
    errors = [finding["message"] for finding in audit["findings"]
              if finding["proposal_id"] in fixed and finding["code"] == "undeclared_recipe_changes"]
    for proposal in candidate.programs:
        recipe = proposal.recipe
        if recipe.pipeline_graph and (recipe.transforms or recipe.feature_discovery):
            from .pipeline_graph import PipelineGraph
            graph = PipelineGraph.from_dict(recipe.pipeline_graph).validate()
            for node in graph.nodes:
                if node.kind != "estimator":
                    continue
                kind = node.parameters.get("model_kind", recipe.model.kind if node.node_id == graph.primary_node_id
                                           else "benter_conditional_logit")
                if node.output.kind == "performance_distribution":
                    errors.append(f"{proposal.proposal_id}: fold-local learned-feature graph cannot use performance_distribution ancestors")
                elif node.output.kind == "win_probability" and kind not in {
                        "logit", "boosted", "benter_conditional_logit", "gaussian_probit"}:
                    errors.append(f"{proposal.proposal_id}: fold-local learned-feature win node does not implement {kind}")
    if errors:
        raise ValueError("Decision admission preflight: " + "; ".join(errors[:8]))
    return candidate


def _planner_with_preflight(original, source=None):
    strict_bytecode = source is not None
    source = textwrap.dedent(source if source is not None else inspect.getsource(original))
    tree = ast.parse(source)
    # The installed allocation policy compiles a prompt-only AST override and
    # leaves misleading inspect locations. Reconstruct it from verified source,
    # then prove opcode/constants parity before adding the validation hook.
    strings = [value for value in original.__code__.co_consts if isinstance(value, str)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 200:
            matches = [value for value in strings if value.startswith(node.value[:120])]
            if len(matches) == 1:
                node.value = matches[0]
    namespace = dict(original.__globals__)
    baseline = {}
    exec(compile(tree, "<request-guard-parity>", "exec"), namespace, baseline)

    def signature(code):
        # Full-module compilation can optimize imported module methods differently
        # from isolated functions. Sealed server policies use isolated compilation.
        return (code.co_code if strict_bytecode else None, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
                tuple(signature(value) if isinstance(value, CodeType) else value for value in code.co_consts))

    if signature(baseline[original.__name__].__code__) != signature(original.__code__):
        raise ValueError("Planner reconstruction differs from installed behavior")
    matches = 0

    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal matches
            if (any(isinstance(target, ast.Name) and target.id == "decision" for target in node.targets)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "model_validate"):
                matches += 1
                check = ast.parse("decision = _request_guard_semantic_preflight(decision, evidence_bundle)").body[0]
                return [node, ast.copy_location(check, node)]
            return node

    tree = Insert().visit(tree)
    if matches != 1:
        raise ValueError("Planner semantic repair integration point changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__) | {"_request_guard_semantic_preflight": semantic_preflight}
    compiled = {}
    exec(compile(tree, __file__+":semantic-preflight", "exec"), namespace, compiled)
    return compiled[original.__name__], hashlib.sha256(source.encode()).hexdigest()


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
            failures.append(error.partition(":")[0])
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


def install_guard(campaign, context_bytes=CONTEXT_BYTES, request_bytes=REQUEST_BYTES,
                  default_dataset_id=None, planner_source=None):
    from . import openrouter_orchestrator as orchestrator
    from . import research_planner_context as context
    if getattr(orchestrator.choose_research_decision, "request_guard_version", None):
        raise ValueError("Planner request guard already installed")
    original_choose, planner_source_sha256 = _planner_with_preflight(orchestrator.choose_research_decision, planner_source)
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
    if default_dataset_id is not None:
        from . import proposal_audit, experiment_discipline
        original_audit = proposal_audit.audit_decisions

        def audit(decisions, evidence):
            return effective_dataset_audit(original_audit, decisions, evidence, default_dataset_id)

        proposal_audit.audit_decisions = audit
        experiment_discipline.audit_decisions = audit
    return {"request_guard_version": VERSION, "context_bytes": context_bytes,
            "request_bytes": request_bytes, "default_dataset_id": default_dataset_id,
            "original_planner_source_sha256": planner_source_sha256,
            "scientific_source_changed": False}
