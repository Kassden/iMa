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
from contextvars import ContextVar
import uuid


VERSION = 6
CONTEXT_BYTES = None
REQUEST_BYTES = None
SOFT_CONTEXT_BYTES = 270_000


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
        if max_bytes is not None and len(encoded(view)) > max_bytes:
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
        "graph_primary_binding": "recipe.model.parameters overrides primary_node_id.model_parameters, "
                                 "including l2 and max_iter. These are executable settings, not metadata. "
                                 "For a pipeline_composition-only fixed test copy recipe.model exactly from the parent. "
                                 "To change its settings too declare hyperparameters and describe a joint test. "
                                 "fixed_parameters=true means no Optuna tuning, NOT unchanged parent settings.",
        "fold_local_graph_limit": "With recipe.transforms or feature_discovery, win-probability graph "
                                 "estimator nodes support logit, boosted, benter_conditional_logit and gaussian_probit; "
                                 "catboost_classifier and performance_distribution ancestors are unsupported on that path.",
        "adjusted_speed_graph_limit": "Graph feature_discovery.adjusted_speed_residuals=true requires an unimplemented nested residual fit contract. Use residuals=false or a supported non-graph recipe.",
    }
    receipt.update(request_guard_version=VERSION, context_byte_limit=max_bytes,
                   context_soft_target_bytes=SOFT_CONTEXT_BYTES if max_bytes is None else int(max_bytes * .90),
                   bytes_before_request_guard=before,
                   request_guard_omissions={
                       "historical_paper_searches": "Counted previews only; full evidence/report remains durable",
                       "zero_capacity_programs": "Exhausted programs omitted, positive capacity retained",
                       "outcome_runtime_artifacts": "Execution paths omitted, science/metrics retained"})
    receipt["request_guard_projection_sha256"] = hashlib.sha256(encoded(view)).hexdigest()
    size = len(encoded(view))
    omissions = receipt.setdefault("adaptive_omissions", {})
    target = SOFT_CONTEXT_BYTES if max_bytes is None else int(max_bytes * .90)
    for key, keep in (("hypothesis_memory", 1), ("recent_planner_failures", 1)):
        rows = evidence.get(key)
        if isinstance(rows, list) and len(rows) > keep and size > target:
            omissions[key] = len(rows)-keep
            evidence[key] = rows[:keep]
            size = len(encoded(view))
    # Full recipes remain in the contract champions; replace only exact duplicates.
    champions = evidence.get("champions_by_contract", {})
    champions = champions if isinstance(champions, dict) else {}
    families = evidence.get("champions_by_family", {})
    for family, row in list(families.items()) if isinstance(families, dict) else ():
        if size <= target:
            break
        if not isinstance(row, dict) or not row.get("recipe"):
            continue
        for key, champion in champions.items():
            if (isinstance(champion, dict) and champion.get("attempt_id") == row.get("attempt_id")
                    and champion.get("recipe") == row["recipe"]):
                replacement = deepcopy(row)
                replacement.pop("recipe")
                replacement["recipe_reference"] = {"champions_by_contract": key, "attempt_id": row["attempt_id"]}
                families[family] = replacement
                omissions["duplicate_family_recipes"] = omissions.get("duplicate_family_recipes", 0)+1
                break
        size = len(encoded(view))
    receipt["request_guard_projection_sha256"] = hashlib.sha256(encoded(view)).hexdigest()
    size = len(encoded(view))
    if max_bytes is not None and size > max_bytes:
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


class ProposalRepairError(ValueError):
    """Bounded machine-readable diagnostics; never edit the scientific proposal."""

    def __init__(self, errors):
        self.repair_errors = errors
        super().__init__("Decision admission preflight: " + "; ".join(e["message"] for e in errors))


def _control_repairs(audit, evidence, fixed):
    index = {}
    for key in ("completed_trial_index", "active_trials", "champions_by_family", "champions_by_contract"):
        rows = evidence.get(key, [])
        if not isinstance(rows, (dict, list)):
            continue
        for row in rows.values() if isinstance(rows, dict) else rows:
            if isinstance(row, dict) and isinstance(row.get("attempt_id"), str):
                index[row["attempt_id"]] = index.get(row["attempt_id"], {}) | row
    memory = evidence.get("hypothesis_memory", [])
    for item in memory if isinstance(memory, list) else ():
        if not isinstance(item, dict) or item.get("kind") not in {"evaluated", "preparation_rejected"}:
            continue
        payload = item.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("attempt_id"), str):
            aid = payload["attempt_id"]
            index[aid] = index.get(aid, {}) | payload
    errors = []
    for record in audit["proposals"]:
        if record["proposal_id"] not in fixed:
            continue
        for comparison in record["parent_comparisons"]:
            findings = [f for f in audit["findings"] if f["proposal_id"] == record["proposal_id"]
                        and f["code"] == "undeclared_recipe_changes"
                        and f["message"].startswith(f"Against {comparison['attempt_id']}, fields ")]
            if not findings:
                continue
            parent = index.get(comparison["attempt_id"], {}).get("recipe", {})
            proposed = record["declared_proposal"]["recipe"]
            differences = []

            def walk(old, new, path):
                if old == new or len(differences) >= 8:
                    return
                if isinstance(old, dict) and isinstance(new, dict) and len(path) < 8:
                    for key in sorted(set(old) | set(new)):
                        walk(old.get(key), new.get(key), path + [key])
                else:
                    def preview(value):
                        return value if len(encoded(value)) <= 240 else {"omitted": True, "source": "parent/candidate recipe in evidence"}
                    differences.append({"path": path, "parent": preview(old), "proposed": preview(new)})

            # Model precedence is the frequent failure; lead with exact values.
            fields = comparison.get("changed_recipe_fields", [])
            for field in sorted(fields, key=lambda value: (value != "model", value)):
                walk(parent.get(field), proposed.get(field), ["recipe", field])
            errors.append({"type": "undeclared_recipe_changes", "proposal_id": record["proposal_id"],
                "parent_attempt_id": comparison["attempt_id"], "declared_axes": record["declared_proposal"]["changed_axes"],
                "changed_recipe_fields": fields, "differences_preview": differences,
                "message": findings[0]["message"],
                "model_axis_hint": ("model_family" if parent.get("model", {}).get("kind") != proposed.get("model", {}).get("kind")
                                    else "hyperparameters") if "model" in fields else None,
                "repair_options": ["Restore undeclared fields exactly from parent for the claimed single-axis test.",
                    "Or explicitly declare all changed axes and describe a joint experiment; do not claim isolated causality."],
                "primary_binding": "Outer recipe.model.parameters overrides primary graph node parameters."})
            if len(errors) >= 8:
                return errors
    return errors


def _repair_errors(exc, fallback):
    if not isinstance(exc, ProposalRepairError):
        return fallback
    result = deepcopy(exc.repair_errors)
    # Feedback is optional detail, not a second copy of the full evidence bundle.
    while len(encoded(result)) > 12000:
        detailed = next((row for row in reversed(result) if row.get("differences_preview")), None)
        if detailed is not None:
            detailed["differences_preview"].pop()
            detailed["preview_truncated"] = True
        elif len(result) > 1:
            result.pop()
            result[0]["additional_errors_omitted"] = True
        else:
            return [{"type": "undeclared_recipe_changes", "message": str(exc)[:1500],
                     "detail_reference": "parent and proposed recipes in current evidence", "preview_truncated": True}]
    return result


def semantic_preflight(decision, evidence):
    from .research_expansion import PlannerDecision
    from .proposal_audit import audit_decisions
    bound = bind_response({"decision": decision.model_dump(mode="json")}, evidence)["decision"]
    candidate = PlannerDecision.model_validate(bound)
    fixed = {p.proposal_id for p in candidate.programs if p.fixed_parameters and p.parent_trial_ids}
    audit = audit_decisions([bound], evidence)
    repairs = _control_repairs(audit, evidence, fixed)
    errors = [finding["message"] for finding in audit["findings"]
              if finding["proposal_id"] in fixed and finding["code"] == "undeclared_recipe_changes"]
    from .experiment_discipline import proposal_signature
    pending = evidence.get("experiment_discipline", {}).get("pending_controls", [])
    signatures = {row["signature"]: row["program_id"] for row in pending
                  if row.get("pending_trials", 0) > 0 and row.get("signature") and row.get("program_id")}
    for proposal in candidate.programs:
        signature = proposal_signature(proposal)
        if signature in signatures:
            program_id = signatures[signature]
            errors.append(f"{proposal.proposal_id}: Pending experiment repeats {program_id}; "
                          "remove this new proposal and wait for outcomes, extend the existing program "
                          "through extensions={program_id: additional_trials} when more trials are justified, or propose a genuinely "
                          "different declared experiment. Do not clone its authorization.")
        else:
            signatures[signature] = proposal.proposal_id
    for proposal in candidate.programs:
        recipe = proposal.recipe
        if recipe.pipeline_graph and recipe.feature_discovery and recipe.feature_discovery.adjusted_speed_residuals:
            errors.append(f"{proposal.proposal_id}: Graph adjusted-speed histories require a nested residual fit contract; disable adjusted_speed_residuals or use a supported non-graph recipe")
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
        if repairs:
            repairs.extend({"message": error} for error in errors
                           if not any(r["message"] == error for r in repairs))
            raise ProposalRepairError(repairs[:8])
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
    repair_matches = 0

    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal matches, repair_matches
            if any(isinstance(target, ast.Name) and target.id == "errors" for target in node.targets):
                repair_matches += 1
                check = ast.parse("errors = _request_guard_repair_errors(exc, errors)").body[0]
                return [node, ast.copy_location(check, node)]
            if (any(isinstance(target, ast.Name) and target.id == "decision" for target in node.targets)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "model_validate"):
                matches += 1
                check = ast.parse("decision = _request_guard_semantic_preflight(decision, evidence_bundle)").body[0]
                return [node, ast.copy_location(check, node)]
            return node

    tree = Insert().visit(tree)
    if matches != 1 or repair_matches != 1:
        raise ValueError("Planner semantic repair integration point changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__) | {"_request_guard_semantic_preflight": semantic_preflight,
                                            "_request_guard_repair_errors": _repair_errors}
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
    dispatched = ContextVar("request_guard_dispatched", default=None)

    def project(bundle):
        return bounded_context(original_context(bundle), context_bytes)

    def post(url, payload, config):
        # Static schemas, dynamic context and schema-repair candidates all count.
        size = len(encoded(payload))
        if request_bytes is not None and size > request_bytes:
            raise ValueError(f"Planner HTTP request exceeds byte budget: {size}>{request_bytes}; no paid call")
        if dispatched.get() is not None:
            dispatched.set(dispatched.get()+1)
        return original_post(url, payload, config)

    def choose(evidence, limits, config):
        token = dispatched.set(0)
        try:
            circuit.check_and_record(evidence)
            return bind_response(original_choose(evidence, limits, config), evidence)
        except Exception:
            if dispatched.get() == 0 and config.transport_observer:
                identity = {"producer": "planner_request_guard", "preflight_version": 1,
                            "decision_id": evidence["decision_id"], "evidence_id": evidence["evidence_id"],
                            "preflight_nonce": uuid.uuid4().hex}
                config.transport_observer({**identity, "phase": "request_started"})
                config.transport_observer({**identity, "phase": "request.preflight_failed"})
            raise
        finally:
            dispatched.reset(token)

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
