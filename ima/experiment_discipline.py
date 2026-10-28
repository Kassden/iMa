"""Linked research lifecycle and deterministic admission, separate from scientific truth."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .proposal_audit import _semantic_recipe, audit_decisions
from .research_telemetry import comparison_key


POLICY = {
    "version": 1,
    "pending_controls": "Wait for structurally identical pending work; extend existing search programs instead of cloning them.",
    "controlled_tests": "Fixed parent controls must declare every changed axis; multi-axis changes are joint tests, not isolated causes.",
    "promotion": "Development gains are provisional. Do not permanently retire/promote features from one seed or a scalar threshold.",
    "confirmation": "Lock candidate/baseline and later population before a separate temporal comparison; never tune on protected final confirmation.",
    "confirmation_queue": "Improving matched development candidates require later frozen predictions. Queue entries are requests, never confirmed gains.",
}


def proposal_signature(proposal):
    from .research_search import _default_space
    # Match the executor's fallback space; an empty declaration is not a fixed fit.
    space = {} if proposal.fixed_parameters else (proposal.search_space or _default_space(proposal.recipe.model.kind))
    recipe = _semantic_recipe(proposal.recipe.canonical_payload())
    for name in space:
        recipe["model"]["parameters"].pop(name, None)
    graph = recipe.get("pipeline_graph")
    if graph:
        # Canonical recipes bind the outer search to tune_node_id (or the legacy
        # primary_node_id). The graph output may be a pool and is never an estimator.
        tuned_id = graph.get("tune_node_id") or graph["primary_node_id"]
        tuned = next(n for n in graph["nodes"] if n["node_id"] == tuned_id)
        for name in space:
            tuned["parameters"].get("model_parameters", {}).pop(name, None)
    payload = {"recipe": recipe,
               "search_space": {k: v.model_dump(mode="json") for k, v in space.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _decision_links(path):
    if not Path(path).is_file():
        return {}
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT d.id,d.payload,r.action,r.payload FROM decisions d JOIN receipts r ON d.id=r.decision_id WHERE r.action LIKE 'program:%' ORDER BY d.rowid").fetchall()
    links = {}
    for did, encoded, _, receipt in rows:
        decision, pid = json.loads(encoded), json.loads(receipt)["program_id"]
        links.setdefault(pid, []).append({"decision_id": did, "evidence_id": decision["evidence_id"]})
    return links


def attempt_rows(path):
    """Read authoritative history, never infer completion from a truncated memory."""
    if not Path(path).is_file():
        return []
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT attempt_id,status,payload_json,result_json FROM attempts ORDER BY rowid").fetchall()
    return [{"attempt_id": a, "status": s, "payload": json.loads(p), "result": json.loads(r or "{}")}
            for a, s, p, r in rows]


def reused_rows(path):
    if not Path(path).is_file():
        return []
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro",uri=True) as conn:
        rows=conn.execute("SELECT event_id,payload FROM events WHERE kind='reused_outcome' ORDER BY rowid").fetchall()
    return [json.loads(payload) | {"observation_id":event,"reused":True} for event,payload in rows]


def resolve_reused_attempt(ledger,search,hypotheses,record,suggestion):
    """Resolve a cached execution without changing its immutable original owner."""
    with sqlite3.connect(f"file:{ledger.path.resolve()}?mode=ro",uri=True) as conn:
        row=conn.execute("SELECT status,payload_json,result_json FROM attempts WHERE attempt_id=?",(record.attempt_id,)).fetchone()
    if row is None or row[0] not in {"completed","failed","pruned"}:
        raise ValueError("Reused attempt is not terminal")
    status,original,result=row[0],json.loads(row[1]),json.loads(row[2] or '{}')
    alias={"attempt_id":record.attempt_id,"status":status,
           "payload":original | suggestion.serializable(),"result":result}
    event=f"reused-outcome-v1:{suggestion.program_id}:{suggestion.trial_number}"
    hypotheses.record(event,suggestion.program_id,"reused_outcome",alias)
    _tell_reused(search,alias)
    return alias


def _tell_reused(search,alias):
    status,result=alias['status'],alias['result']
    pid,number=alias['payload']['program_id'],alias['payload']['trial_number']
    if search.trial_state(pid,number) != 'running':
        return
    if status=='completed' and type(result.get('objective_value')) in (int,float) and math.isfinite(result['objective_value']):
        search.tell(pid,number,result['objective_value'],result.get('metrics',{}))
    else:
        search.tell_failed(pid,number)


def reconcile_reused_tells(directory,search):
    for alias in reused_rows(Path(directory)/'hypotheses.sqlite'):
        _tell_reused(search,alias)


def linked_outcome(row, proposal=None, decision_links=()):
    payload, result = row.get("payload", {}), row.get("result", {})
    recipe = payload.get("recipe", {})
    value = result.get("objective_value")
    valid_value = type(value) in (int, float) and math.isfinite(value)
    return {"attempt_id": row["attempt_id"], "program_id": payload.get("program_id"),
            "proposal_id": proposal.proposal_id if proposal else payload.get("proposal_id"),
            "execution_proposal_id": payload.get("proposal_id"),
            "decision_links": list(decision_links), "parent_trial_ids": list(proposal.parent_trial_ids) if proposal else [],
            "status": row["status"], "objective_name": result.get("objective_name"),
            "objective_value": value if valid_value else None,
            "comparison_key": comparison_key(dict(result, attempt_id=row["attempt_id"]), recipe),
            "artifacts": result.get("artifacts", {}), "error": result.get("error"),
            "gain_evidence": "development_only_unconfirmed" if row["status"] == "completed" else "unfinished_or_failed"}


def lifecycle_snapshot(directory, search, retired=(), *, remaining_capacity=None, references=()):
    """Complete durable join; bounded planner view is made separately."""
    directory = Path(directory)
    rows = attempt_rows(directory / "ledger.sqlite")
    aliases = reused_rows(directory / "hypotheses.sqlite")
    links = _decision_links(directory / "decisions.sqlite")
    by_id = {r["attempt_id"]: r for r in rows}
    reference_index = {r["attempt_id"]: r for r in references}
    outcomes, programs = [], []
    for row in rows + aliases:
        pid = row["payload"].get("program_id")
        outcome = linked_outcome(row, search.programs.get(pid), links.get(pid, ()))
        if row.get("reused"):
            outcome.update(reused=True,observation_id=row["observation_id"])
        parents = []
        for parent in outcome["parent_trial_ids"]:
            parent_row = by_id.get(parent)
            baseline = linked_outcome(parent_row) if parent_row else reference_index.get(parent)
            if (row["status"] == "completed" and baseline and
                (parent_row is None or parent_row["status"] == "completed") and
                baseline.get("comparison_key") == outcome["comparison_key"]):
                old, new = baseline.get("objective_value"), outcome["objective_value"]
                if type(old) in (int, float) and math.isfinite(old) and new is not None:
                    parents.append({"attempt_id": parent, "objective_delta": new - old,
                                    "evidence": "matched_development_delta_only"})
        outcome["matched_parent_deltas"] = parents
        outcomes.append(outcome)
    for pid, proposal in search.programs.items():
        own = [o for o in outcomes if o["program_id"] == pid]
        counts = Counter(o["status"] for o in own)
        active = counts["reserved"] + counts["running"]
        if remaining_capacity is not None:
            remaining = max(0, remaining_capacity.get(pid, 0))
        else:
            trials = search.studies[pid].get_trials(deepcopy=False)
            remaining = max(0, proposal.max_trials - sum(t.state.name != "PRUNED" for t in trials))
        if pid in retired:
            remaining = 0
        pending = active + remaining
        programs.append({"program_id": pid, "proposal_id": proposal.proposal_id,
                         "evidence_ids": list(proposal.evidence_ids), "decision_links": links.get(pid, []),
                         "parent_trial_ids": list(proposal.parent_trial_ids), "signature": proposal_signature(proposal),
                         "hypothesis": proposal.hypothesis, "changed_axes": list(proposal.changed_axes),
                         "fixed_parameters": proposal.fixed_parameters, "recipe": proposal.recipe.canonical_payload(),
                         "search_space": {k: v.model_dump(mode="json") for k, v in proposal.search_space.items()},
                         "remaining_trials": remaining, "pending_trials": pending, "counts": dict(counts),
                         "status": "pending" if pending else "terminal_or_retired",
                         "gain_evidence": "development_only_unconfirmed"})
    candidates = {}
    for outcome in outcomes:
        improved = [p for p in outcome["matched_parent_deltas"] if p["objective_delta"] < 0]
        if not improved:
            continue
        key = outcome["comparison_key"]
        old = candidates.get(key)
        if old is None or (outcome["objective_value"],outcome["attempt_id"]) < (old["objective_value"],old["attempt_id"]):
            candidates[key] = {"attempt_id":outcome["attempt_id"],"program_id":outcome["program_id"],
                "proposal_id":outcome["proposal_id"],"objective_value":outcome["objective_value"],
                "objective_name":outcome["objective_name"],"comparison_key":key,
                "baseline_candidates":improved,"artifacts":outcome["artifacts"],
                "status":"awaiting_locked_later_predictions",
                "promotion_allowed":False,
                "reason":"Development improvement is a selection signal only; later unseen labels and frozen identities are required."}
    return {"schema_version": 1, "policy": POLICY, "programs": programs, "outcomes": outcomes,
            "confirmation_queue":list(candidates.values()),
            "counts": {"programs": len(programs), "attempts": len(rows),
                       "pending_programs": sum(p["pending_trials"] > 0 for p in programs),
                       "reused_observations":len(aliases),
                       "unlinked_attempts": sum(o["program_id"] not in search.programs for o in outcomes)},
            "scope": "Complete own ledgers; independent confirmation is a separately locked workflow."}


def planner_view(snapshot):
    pending = []
    for row in snapshot["programs"]:
        if row["pending_trials"]:
            pending.append({k: deepcopy(row[k]) for k in ("program_id", "proposal_id", "signature", "parent_trial_ids",
                "changed_axes", "pending_trials", "counts", "fixed_parameters")})
    recent = snapshot["outcomes"][-12:]
    return {"policy": POLICY, "counts": snapshot["counts"], "pending_controls": pending,
            "confirmation_queue":deepcopy(snapshot.get("confirmation_queue", [])),
            "recent_outcomes": deepcopy(recent), "outcomes_omitted": max(0, len(snapshot["outcomes"]) - len(recent)),
            "full_artifact": "experiment-discipline.json", "champion_status": "development_only_unconfirmed"}


def validate_decision(decision, evidence, snapshot):
    """Reject provable redundant work and false fixed-control declarations."""
    signatures = {}
    for row in snapshot["programs"]:
        if row["pending_trials"]:
            signatures.setdefault(row["signature"], []).append(row)
    for proposal in decision.programs:
        signature = proposal_signature(proposal)
        for earlier in signatures.get(signature, []):
            # Existing identical authorization replay must survive crash recovery.
            if earlier["proposal_id"] == proposal.proposal_id and earlier["evidence_ids"] == list(proposal.evidence_ids):
                continue
            raise ValueError(f"Pending experiment repeats {earlier['program_id']}; wait for outcomes or extend the existing program")
        signatures.setdefault(signature, []).append({"program_id": proposal.proposal_id,
            "proposal_id": proposal.proposal_id, "evidence_ids": list(proposal.evidence_ids)})
    audit = audit_decisions([decision.model_dump(mode="json")], evidence)
    fixed = {p.proposal_id for p in decision.programs if p.fixed_parameters and p.parent_trial_ids}
    bad = [f for f in audit["findings"] if f["proposal_id"] in fixed and f["code"] in {
        "undeclared_recipe_changes", "invalid_parent_recipe", "unseen_parent"}]
    for row in audit["proposals"]:
        if row["proposal_id"] in fixed and any(c.get("recipe_comparison") == "unavailable" for c in row["parent_comparisons"]):
            raise ValueError("Fixed parent control requires the exact parent recipe")
    if bad:
        raise ValueError("Invalid fixed control: " + "; ".join(f["message"] for f in bad))
    return {"policy_version": 1, "proposals": [{"proposal_id": row["proposal_id"],
        "comparison_class": "joint_or_exploratory_test" if not row["declared_proposal"].get("fixed_parameters") or
            len(row["declared_proposal"].get("changed_axes", [])) != 1 or
            any(len(c.get("changed_recipe_fields", [])) > 1 for c in row["parent_comparisons"])
            else "declared_control_not_confirmed",
        "parent_comparisons": row["parent_comparisons"], "gain_evidence": "not_yet_observed"} for row in audit["proposals"]],
        "warnings": audit["planner_feedback"]}


def admit_decision(decision, store, search, retired, config, evidence):
    """Current authoritative state, not the planner's possibly stale pending snapshot."""
    directory = Path(config.campaign_dir)
    current = lifecycle_snapshot(directory, search, retired)
    enriched = deepcopy(evidence)
    index = {r["attempt_id"]: r for r in enriched.get("completed_trial_index", [])}
    for row in attempt_rows(directory / "ledger.sqlite"):
        if row["status"] == "completed":
            outcome = linked_outcome(row)
            recipe = row['payload'].get('recipe')
            if recipe is not None:
                outcome['recipe'] = recipe
            else:
                # Legacy paper-analysis parents can have valid stored artifacts
                # without a trainable recipe. Keep the observed outcome, but
                # do not invent a recipe for fixed-control lineage checks.
                outcome['recipe_availability'] = 'missing_in_legacy_ledger'
            index[row["attempt_id"]] = outcome
    enriched["completed_trial_index"] = list(index.values())
    return validate_decision(decision, enriched, current)
