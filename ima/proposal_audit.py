"""Deterministic, read-only proposal review; no planner or execution calls."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json

from .research_specs import PipelineRecipe, ResearchProposal


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _semantic_recipe(payload):
    result = deepcopy(payload)
    if result.get("pipeline_graph") is None:
        return result
    from .pipeline_graph import PipelineGraph
    graph = asdict(PipelineGraph.from_dict(result["pipeline_graph"]).validate())
    # Match fit_graph's outer tuned-estimator binding, including override precedence.
    # Legacy graphs use primary_node_id; newer graphs can tune a different estimator.
    tuned_id = graph.get("tune_node_id") or graph["primary_node_id"]
    for node in graph["nodes"]:
        if node["node_id"] == tuned_id:
            options = node["parameters"]
            options["model_kind"] = result["model"]["kind"]
            options["model_parameters"] = dict(options.get("model_parameters", {})) | result["model"]["parameters"]
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    mapping = {}

    # Ordered endpoint traversal preserves input order and shared-node identity.
    def visit(identifier):
        if identifier is None or identifier in mapping:
            return
        mapping[identifier] = f"n{len(mapping)}"
        for parent in nodes[identifier]["inputs"]:
            visit(parent)

    for key in ("output_node_id", "fundamental_node_id", "primary_node_id", "joint_node_id", "tune_node_id"):
        visit(graph[key])
    # Disconnected nodes stay conservatively ordered, never silently disappear.
    for node in graph["nodes"]:
        visit(node["node_id"])
    normalized = []
    for identifier in mapping:
        node = deepcopy(nodes[identifier])
        node["node_id"] = mapping[identifier]
        node["inputs"] = [mapping[parent] for parent in node["inputs"]]
        normalized.append(node)
    graph["nodes"] = normalized
    graph.pop("graph_id")
    for key in ("output_node_id", "fundamental_node_id", "primary_node_id", "joint_node_id", "tune_node_id"):
        graph[key] = mapping.get(graph[key])
    result["pipeline_graph"] = graph
    return result


@lru_cache(maxsize=512)
def _recipe(serialized):
    recipe = PipelineRecipe.model_validate(json.loads(serialized))
    exact = recipe.canonical_payload()
    return exact, _semantic_recipe(exact), recipe.recipe_hash()


@lru_cache(maxsize=512)
def _proposal(serialized):
    return ResearchProposal.model_validate(json.loads(serialized))


LIMITATIONS = [
    "Hypothesis prose cannot prove scientific relevance or gains; declarations are claims.",
    "Unfinished or unlinked trials cannot be evaluated. Accepted planning is not successful execution.",
    "Snapshot champions are not necessarily champions at earlier decision times; no stale-parent verdict follows.",
    "Recipe/search equality is a candidate repeat, not proof of redundant execution: dataset, protocol, code and environment must also match.",
    "Matched ablations, null results and exploration are legitimate tests. Long runtime is not inherently bad.",
    "Seed replicates explore variability but do not establish uncertainty bounds or independent holdout/profit gains.",
]


def audit_decisions(decisions, evidence):
    """Audit already-decoded JSON. Index evidence once and cache recipe validation.

    Returns JSON-friendly detail plus concise planner feedback. Does not mutate
    inputs, read files, fit models, or evaluate natural-language truth claims.
    """
    findings = []
    current_decision_id = None

    def finding(code, message, proposal_id=None, classification="warning"):
        findings.append(dict(code=code, classification=classification,
                             proposal_id=proposal_id, decision_id=current_decision_id, message=message))

    if not isinstance(evidence, dict):
        finding("invalid_evidence", "Evidence must be an object.", classification="factual_failure")
        evidence = {}
    if isinstance(decisions, dict):
        decisions = [decisions]
    if not isinstance(decisions, (list, tuple)):
        finding("invalid_decisions", "Decisions must be an object or sequence.", classification="factual_failure")
        decisions = []
    index, champions, outcomes = {}, {}, {}
    for key in ("completed_trial_index", "active_trials", "champions_by_family", "champions_by_contract"):
        rows = evidence.get(key, [])
        if not isinstance(rows, (dict, list)):
            finding("invalid_evidence_section", f"{key} must be a list or object.")
            continue
        for row in rows.values() if isinstance(rows, dict) else rows:
            if isinstance(row, dict) and isinstance(row.get("attempt_id"), str):
                aid = row["attempt_id"]
                index[aid] = index.get(aid, {}) | row
                if key == "champions_by_contract":
                    champions[aid] = row
                if key == "active_trials" or row.get("status"):
                    outcomes[aid] = row
    memory = evidence.get("hypothesis_memory", [])
    if isinstance(memory, list):
        for item in memory:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload", {})
            if isinstance(payload, dict) and item.get("kind") in {"evaluated", "preparation_rejected"}:
                payload = deepcopy(payload)
                payload.setdefault("program_id", item.get("hypothesis_id"))
                if item.get("kind") == "preparation_rejected":
                    payload.setdefault("status", "preparation_rejected")
                aid = payload.get("attempt_id")
                if isinstance(aid, str):
                    index[aid] = index.get(aid, {}) | payload
                outcomes[aid or f"memory:{len(outcomes)}"] = payload

    records, decision_records, seen, proposal_ids = [], [], {}, set()
    ordered = sorted(enumerate(decisions), key=lambda pair: (
        str(pair[1].get("decision_id", "")) if isinstance(pair[1], dict) else "", pair[0]))
    for _, decision in ordered:
        if not isinstance(decision, dict):
            finding("invalid_decision", "Decision must be an object.", classification="factual_failure")
            continue
        did = decision.get("decision_id")
        current_decision_id = did if isinstance(did, str) else None
        programs = decision.get("programs", decision.get("proposals", []))
        if not isinstance(programs, list):
            finding("invalid_programs", f"{did}: programs must be a list.", classification="factual_failure")
            continue
        budget = sum(p.get("max_trials", 1) for p in programs
                     if isinstance(p, dict) and type(p.get("max_trials", 1)) is int)
        dr = {key: deepcopy(decision.get(key)) for key in (
            "decision_id", "evidence_id", "planner_status", "planner_usage", "program_ids",
            "trial_budget", "allocated_trials", "unallocated_trials", "planning_wall_seconds")}
        dr.update(sha256=_hash(decision), declared_program_trials=budget)
        decision_records.append(dr)
        if type(decision.get("trial_budget")) is int and budget > decision["trial_budget"]:
            finding("program_budget_exceeds_decision", f"{did}: declared {budget} trials exceed decision budget {decision['trial_budget']}.")
        for position, raw in enumerate(programs):
            pid = raw.get("proposal_id") if isinstance(raw, dict) else None
            record = dict(decision_id=did, proposal_id=pid, source_sha256=_hash(raw),
                          valid=False, declared_proposal=deepcopy(raw))
            records.append(record)
            try:
                proposal = _proposal(_json(raw))
            except (ValueError, TypeError, KeyError) as exc:
                finding("invalid_proposal", str(exc), pid, "factual_failure")
                continue
            if pid in proposal_ids:
                finding("repeated_proposal_id", "Proposal identity appears more than once.", pid)
            proposal_ids.add(pid)
            exact, semantic, legacy_hash = _recipe(_json(proposal.recipe.canonical_payload()))
            # Match the executor's effective space: an empty exploratory
            # declaration uses model defaults, while a fixed proposal has no
            # search dimensions. Values overridden by those dimensions are not
            # part of the searched recipe, including on a graph's tuned node.
            from .research_search import _default_space
            space = ({} if proposal.fixed_parameters else
                     proposal.search_space or _default_space(proposal.recipe.model.kind))
            signature_recipe = deepcopy(semantic)
            for name in space:
                signature_recipe["model"]["parameters"].pop(name, None)
            graph = signature_recipe.get("pipeline_graph")
            if graph:
                tuned_id = graph.get("tune_node_id") or graph["primary_node_id"]
                tuned = next(node for node in graph["nodes"] if node["node_id"] == tuned_id)
                for name in space:
                    tuned["parameters"].get("model_parameters", {}).pop(name, None)
            search = {key: dimension.model_dump(mode="json") for key, dimension in space.items()}
            signature = _hash(dict(recipe=signature_recipe, search_space=search))
            record.update(valid=True, recipe_sha256=_hash(exact), recipe_hash=legacy_hash,
                          semantic_recipe_sha256=_hash(semantic), search_signature_sha256=signature,
                          seed=proposal.recipe.seed, max_trials=proposal.max_trials,
                          max_wall_seconds=proposal.max_wall_seconds,
                          declared_serial_wall_budget_seconds=proposal.max_trials * proposal.max_wall_seconds,
                          parent_trial_ids=list(proposal.parent_trial_ids),
                          champion_parent_ids=[p for p in proposal.parent_trial_ids if p in champions],
                          parent_comparisons=[], execution_evidence=[], gain_evidence="unresolved",
                          purpose="fixed_control_or_ablation" if proposal.fixed_parameters else "exploration")
            ids = decision.get("program_ids", [])
            program_id = ids[position] if isinstance(ids, list) and position < len(ids) else None
            record["program_id"] = program_id
            record["execution_evidence"] = [deepcopy(row) for row in outcomes.values()
                if (program_id is not None and row.get("program_id") == program_id)
                or row.get("proposal_id") == pid]
            for field in ("hypothesis", "expected_observation", "falsification_rule"):
                if not getattr(proposal, field).strip():
                    finding("missing_declaration", f"Empty {field}.", pid)
            if not proposal.changed_axes:
                finding("missing_axes", "No changed axes declared.", pid)
            if not proposal.parent_trial_ids:
                finding("unanchored_exploration", "No parent declared; exploration needs its own comparison baseline.", pid)
            for parent in proposal.parent_trial_ids:
                if parent not in index:
                    finding("unseen_parent", f"Parent {parent} is absent from this snapshot, not proven nonexistent.", pid)
                    continue
                row = index[parent]
                comparison = dict(attempt_id=parent, comparison_key=row.get("comparison_key"),
                                  status=row.get("status"), objective_value=row.get("objective_value"))
                record["parent_comparisons"].append(comparison)
                if not isinstance(row.get("recipe"), dict):
                    comparison["recipe_comparison"] = "unavailable"
                    continue
                try:
                    _, parent_semantic, _ = _recipe(_json(row["recipe"]))
                except (ValueError, TypeError, KeyError) as exc:
                    comparison["recipe_comparison"] = "invalid_parent_recipe"
                    finding("invalid_parent_recipe", f"{parent}: {exc}", pid)
                    continue
                changes = sorted(k for k in set(semantic) | set(parent_semantic)
                                 if semantic.get(k) != parent_semantic.get(k))
                comparison["changed_recipe_fields"] = changes
                field_axes = {
                    "model": {"hyperparameters", "model_family"}, "seed": {"hyperparameters"},
                    "feature_schema": {"feature_schema"}, "drop_feature_families": {"feature_family"},
                    "transforms": {"transform"}, "train_window": {"dataset_window"},
                    "target": {"target"}, "calibration": {"calibration"}, "blend": {"market_blend"},
                    "feature_definitions": {"feature_creation"},
                    "extra_numeric_features": {"feature_creation", "feature_family", "transform"},
                    "feature_discovery": {"feature_creation", "feature_family", "transform"},
                    "dataset_ref": {"dataset_build"},
                    "pipeline_graph": {"pipeline_composition", "model_family", "hyperparameters", "market_blend", "calibration", "performance_distribution"},
                    "performance_distribution": {"performance_distribution"},
                }
                unexplained = [k for k in changes if k in field_axes and not (
                    field_axes[k] & set(proposal.changed_axes))]
                if unexplained:
                    finding("undeclared_recipe_changes", f"Against {parent}, fields {unexplained} change outside declared axes; review the matched-control claim.", pid)
                if semantic["target"] != parent_semantic["target"]:
                    finding("cross_target_parent", "Different parent target; numeric gains require a compatible comparison contract.", pid)
                if not changes and proposal.fixed_parameters:
                    finding("same_seed_parent_control", f"Exact semantic recipe and seed match parent {parent}; a rerun is not a new seed replicate.", pid)
                if changes == ["seed"]:
                    comparison["test"] = "seed_replicate"
                elif changes:
                    comparison["test"] = "recipe_ablation_or_composition"
            contracts = {p["comparison_key"] for p in record["parent_comparisons"]
                         if isinstance(p.get("comparison_key"), str)}
            if len(contracts) > 1:
                finding("mixed_parent_contracts", "Parents have different comparison contracts; transfer/composition is a test, not a directly comparable gain.", pid)
            if signature in seen:
                prior = seen[signature]
                record["repeat_of"] = prior["proposal_id"]
                code = "repeated_fixed_same_seed_control" if proposal.fixed_parameters and prior["fixed"] else "repeated_recipe_search"
                finding(code, f"Same semantic recipe/search as {prior['proposal_id']}; check outcomes and execution contracts before allocating again.", pid)
            else:
                seen[signature] = dict(proposal_id=pid, fixed=proposal.fixed_parameters)
            for row in record["execution_evidence"]:
                if row.get("status") in ("failed", "rejected", "preparation_rejected") or row.get("error"):
                    finding("observed_execution_failure", f"Linked attempt {row.get('attempt_id')}: {row.get('error') or row.get('status')}", pid, "factual_failure")
                if row.get("status") == "completed":
                    record["gain_evidence"] = "completed_without_verified_matched_gain"
                    for comparison in record["parent_comparisons"]:
                        if (row.get("comparison_key") is not None
                            and row.get("comparison_key") == comparison.get("comparison_key")
                            and type(row.get("objective_value")) in (int, float)
                            and type(comparison.get("objective_value")) in (int, float)):
                            comparison["observed_objective_delta"] = row["objective_value"] - comparison["objective_value"]
                            record["gain_evidence"] = "matched_development_delta_only"

    counts = Counter(f["code"] for f in findings)
    champion_view = [{key: deepcopy(row.get(key)) for key in (
        "attempt_id", "comparison_key", "objective_name", "objective_value", "recipe")}
        for _, row in sorted(champions.items())]
    return dict(schema_version=1, evidence_sha256=_hash(evidence), evidence_id=evidence.get("evidence_id"),
                evidence_created_at=evidence.get("created_at"), decisions=decision_records,
                champions=champion_view, proposals=records, findings=findings,
                counts=dict(decisions=len(decision_records), proposals=len(records),
                            valid_proposals=sum(r["valid"] for r in records), findings_by_code=dict(sorted(counts.items()))),
                observed_outcomes=[deepcopy(row) for row in outcomes.values()],
                planner_feedback=[dict(code=f["code"], proposal_id=f["proposal_id"], message=f["message"])
                                  for f in findings], limitations=LIMITATIONS[:])


def planner_proposal_feedback(evidence):
    """Bounded advisory feedback from FULL raw memory, before any curation.

    Audit every available planned payload, then prioritize warnings by event
    time (decision ID / newest-first memory position when time is unavailable).
    This is a lossy advisory view, never a proposal rejection or gain verdict.
    """
    limits = dict(warnings=12, message_chars=800, decision_ids=12, recent_negatives=5)
    evidence = evidence if isinstance(evidence, dict) else {}
    memory = evidence.get("hypothesis_memory", [])
    decisions, extra, recency, proposal_decisions, negatives = [], [], {}, {}, []

    def warn(code, message, decision_id=None):
        extra.append(dict(code=code, message=message, decision_id=decision_id,
                          proposal_id=None, classification="warning"))

    if not isinstance(memory, list):
        warn("malformed_hypothesis_memory", "Raw hypothesis_memory must be a list; planned coverage is unavailable.")
        memory = []
    for position, row in enumerate(memory):
        if not isinstance(row, dict):
            warn("malformed_hypothesis_record", f"Raw memory entry {position} is not an object; it was not audited.")
            continue
        payload = row.get("payload")
        did = payload.get("decision_id") if isinstance(payload, dict) else None
        did = did if isinstance(did, str) else row.get("hypothesis_id")
        did = did if isinstance(did, str) else None
        try:
            timestamp = datetime.fromisoformat(row.get("created_at", "").replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            epoch = timestamp.timestamp()
        except (ValueError, TypeError, AttributeError, OverflowError):
            epoch = float("-inf")
        rank = (epoch, did or "", -position)
        if row.get("kind") != "planned":
            if isinstance(payload, dict) and (row.get("kind") == "preparation_rejected"
                    or payload.get("status") in ("failed", "rejected", "preparation_rejected")
                    or payload.get("error") or payload.get("blocked") is True):
                negative = {key: deepcopy(payload.get(key)) for key in (
                    "attempt_id", "program_id", "status", "error", "reason", "failures")}
                negative.update(kind=row.get("kind"), created_at=row.get("created_at"),
                                hypothesis_id=row.get("hypothesis_id"))
                for key in ("error", "reason"):
                    message = negative[key]
                    if isinstance(message, str) and len(message) > limits["message_chars"]:
                        negative[key] = message[:limits["message_chars"] - 12] + " [truncated]"
                        negative[key + "_truncated"] = True
                negatives.append((rank, negative))
            elif not isinstance(payload, dict):
                warn("malformed_hypothesis_payload", f"Raw {row.get('kind')} payload is missing or is not an object; outcome coverage is unavailable.")
            continue
        recency[did] = max(rank, recency.get(did, rank))
        if not isinstance(payload, dict):
            warn("malformed_planned_payload", "Planned payload is missing or is not an object; coverage is unavailable.", did)
            continue
        if "programs" not in payload and "proposals" not in payload:
            warn("missing_planned_programs", "Planned payload has no proposal list; this is not evidence of zero proposals or successful outcomes.", did)
            continue
        decision = dict(payload, decision_id=did)
        decisions.append(decision)
        programs = payload.get("programs", payload.get("proposals", []))
        if isinstance(programs, list):
            for program in programs:
                pid = program.get("proposal_id") if isinstance(program, dict) else None
                if isinstance(pid, str) and (pid not in proposal_decisions
                        or rank > recency.get(proposal_decisions[pid], (float("-inf"), "", 0))):
                    proposal_decisions[pid] = did
    audit = audit_decisions(decisions, evidence)
    warnings = extra + [dict(f, decision_id=f.get("decision_id") or (
                            proposal_decisions.get(f["proposal_id"]) if isinstance(f["proposal_id"], str) else None))
                        for f in audit["findings"]]
    warnings.sort(key=lambda f: recency.get(f["decision_id"], (float("-inf"), "", 0)), reverse=True)
    bounded = []
    for warning in warnings[:limits["warnings"]]:
        message = warning["message"]
        truncated = len(message) > limits["message_chars"]
        if truncated:
            suffix = " [truncated]"
            message = message[:limits["message_chars"] - len(suffix)] + suffix
        bounded.append(dict(code=warning["code"], decision_id=warning["decision_id"],
                            proposal_id=warning["proposal_id"], message=message,
                            message_truncated=truncated, classification="warning",
                            source_classification=warning["classification"]))
    ids = sorted({d["decision_id"] for d in audit["decisions"] if isinstance(d["decision_id"], str)},
                 key=lambda did: recency[did], reverse=True)
    gain_counts = Counter(row.get("gain_evidence", "unresolved") for row in audit["proposals"])
    return dict(schema_version=1, advisory_only=True,
                scope="All planned payloads available in full raw hypothesis_memory; not all campaign decisions.",
                evidence_sha256=audit["evidence_sha256"],
                audited_decision_ids=ids[:limits["decision_ids"]],
                audited_decision_ids_omitted=max(0, len(ids) - limits["decision_ids"]),
                proposal_count=audit["counts"]["proposals"],
                counts=dict(audit["counts"], warning_count=len(warnings),
                            warnings_by_code=dict(sorted(Counter(w["code"] for w in warnings).items())),
                            gain_evidence=dict(sorted(gain_counts.items())), negative_events=len(negatives)),
                recent_negatives=[n for _, n in sorted(negatives, key=lambda item: item[0], reverse=True)[:limits["recent_negatives"]]],
                negative_events_omitted=max(0, len(negatives) - limits["recent_negatives"]),
                warnings=bounded, warnings_omitted=max(0, len(warnings) - len(bounded)),
                limits=limits, limitations=LIMITATIONS[:] + [
                    "Warnings are advisory, not rejections. Missing outcomes remain unresolved.",
                    "Only available raw planned payloads are audited; omissions and truncation are explicit."])


def render_markdown(summary):
    """Human-readable detail retaining full identities and declarations."""
    lines = ["# Research Proposal Audit", "", "## Scope And Limits", ""]
    lines.extend(f"- {text}" for text in summary["limitations"])
    lines += ["", "## Snapshot And Counts", "", "```json", json.dumps({
        key: summary[key] for key in ("evidence_id", "evidence_sha256", "evidence_created_at", "counts")}, indent=2, sort_keys=True), "```",
        "", "## Champions And Comparison Contracts", "", "```json",
        json.dumps(summary["champions"], indent=2, sort_keys=True), "```",
        "", "## Decision Budgets And Reported Planner Costs", "", "```json",
        json.dumps(summary["decisions"], indent=2, sort_keys=True), "```", "", "## Findings", ""]
    lines += [f"- **{f['classification']} / {f['code']}** ({f['proposal_id']}): {f['message']}" for f in summary["findings"]]
    if not summary["findings"]:
        lines.append("No mechanical warnings found; this is not proof of scientific gains.")
    lines += ["", "## Observed Execution Evidence", "", "```json",
              json.dumps(summary["observed_outcomes"], indent=2, sort_keys=True), "```",
              "", "## Proposal Details", ""]
    for row in summary["proposals"]:
        lines += [f"### {row['proposal_id']}", "", "```json", json.dumps(row, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines) + "\n"
