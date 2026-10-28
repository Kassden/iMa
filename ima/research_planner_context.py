"""Reversible, JSON-only compression of repeated planner context metadata."""
from collections import Counter
from copy import deepcopy
import json
import hashlib

from .proposal_audit import planner_proposal_feedback


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def _catalog(catalog):
    if not isinstance(catalog,dict) or not catalog or not all(isinstance(row,dict) for row in catalog.values()):
        return None
    rows = list(catalog.values())
    defaults,columns = {},{}
    for key in sorted(set().union(*(set(row) for row in rows))):
        entries = [(index,_json(row[key])) for index,row in enumerate(rows) if key in row]
        value,count = Counter(value for _,value in entries).most_common(1)[0]
        complete = len(entries)==len(rows)
        default = value if complete and count>1 else None
        if default is not None:
            defaults[key] = json.loads(default)
        groups = {}
        for index,value in entries:
            if value!=default:
                groups.setdefault(value,[]).append(index)
        if not groups:
            continue
        grouped = {"by_value":[[json.loads(value),indices] for value,indices in groups.items()]}
        if complete:
            dense = {"values":[row[key] for row in rows]}
            if len(_json(dense)) < len(_json(grouped)):
                columns[key] = dense
                defaults.pop(key,None)
                continue
        columns[key] = grouped
    encoded = {"encoding":"columnar-v1","names":list(catalog),"defaults":defaults,"columns":columns}
    return encoded if len(_json(encoded)) < len(_json(catalog)) else None


def _references(schema):
    if isinstance(schema,dict):
        if "$ref" in schema:
            yield schema["$ref"]
        for key,value in schema.items():
            if key != "$defs":
                yield from _references(value)
    elif isinstance(schema,list):
        for value in schema:
            yield from _references(value)


def _definition_name(reference):
    prefix = "#/$defs/"
    if not isinstance(reference,str) or not reference.startswith(prefix):
        return None
    return reference[len(prefix):].replace("~1","/").replace("~0","~")


def _schema_reference(candidate,decision):
    if not isinstance(candidate,dict) or not isinstance(decision,dict):
        return None
    definitions = decision.get("$defs",{})
    pending = list(_references(decision))
    reachable = set()
    while pending:
        name = _definition_name(pending.pop())
        if name is None or name not in definitions:
            return None
        if name not in reachable:
            reachable.add(name)
            pending.extend(_references(definitions[name]))
    root = {key:value for key,value in candidate.items() if key != "$defs"}
    source_defs = candidate.get("$defs",{})
    if any(name not in definitions or _json(value)!=_json(definitions[name])
           for name,value in source_defs.items()):
        return None
    # A copied root's references must remain valid in its own original schema.
    for schema in [root,*source_defs.values()]:
        if any(_definition_name(ref) not in source_defs for ref in _references(schema)):
            return None
    for name in sorted(reachable):
        if _json(root)==_json(definitions[name]):
            escaped = name.replace("~","~0").replace("/","~1")
            result = {"$ref":"#/decision_schema/$defs/"+escaped}
            if "$defs" in candidate:
                result["original_definition_names"] = list(source_defs)
            if len(_json(result)) < len(_json(candidate)):
                return result
    return None


def compact_planner_evidence(bundle):
    """Return a deep copy; never alter the durable full evidence or its identity.

    Accept either the complete prompt bundle (with decision_schema and evidence)
    or a bare evidence bundle. Schema deduplication requires decision_schema.
    """
    projected = deepcopy(bundle)
    if "planner_context_encoding" in projected:
        return projected
    evidence = projected.get("evidence",projected)
    capabilities = evidence.get("capabilities",{}) if isinstance(evidence,dict) else {}
    if not isinstance(capabilities,dict):
        return projected
    changed = False
    catalog = _catalog(capabilities.get("eligible_predictors"))
    if catalog is not None:
        capabilities["eligible_predictors"] = catalog
        changed = True
    decision = projected.get("decision_schema")
    locations = [(capabilities,"recipe_schema")]
    paper = capabilities.get("paper_research")
    if isinstance(paper,dict):
        locations.append((paper,"request_schema"))
    for parent,key in locations:
        reference = _schema_reference(parent.get(key),decision)
        if reference is not None:
            parent[key] = reference
            changed = True
    if changed:
        projected["planner_context_encoding"] = {
            "version":1,
            "instructions":"eligible_predictors columnar-v1: names defines zero-based row indices. Every row inherits defaults. Each columns field either has values aligned with names, or by_value pairs [value,[row indices]] overriding that field only at those rows. Fields absent from defaults and a row's columns remain absent; null is explicit. Safety, units and eligibility are unchanged. Schema $refs resolve against this bundle's decision_schema; original_definition_names records the standalone $defs subset. Do not return context encodings in a decision.",
        }
    return projected


def expand_planner_evidence(bundle):
    """Reconstruct the original JSON bundle for auditing and round-trip tests."""
    expanded = deepcopy(bundle)
    marker = expanded.get("planner_context_encoding")
    if not isinstance(marker,dict) or marker.get("version")!=1:
        return expanded
    evidence = expanded.get("evidence",expanded)
    capabilities = evidence.get("capabilities",{})
    catalog = capabilities.get("eligible_predictors")
    if isinstance(catalog,dict) and catalog.get("encoding")=="columnar-v1":
        rows = [deepcopy(catalog["defaults"]) for _ in catalog["names"]]
        for key,column in catalog["columns"].items():
            if "values" in column:
                for row,value in zip(rows,column["values"],strict=True):
                    row[key] = value
            else:
                for value,indices in column["by_value"]:
                    for index in indices:
                        rows[index][key] = deepcopy(value)
        capabilities["eligible_predictors"] = dict(zip(catalog["names"],rows,strict=True))
    locations = [(capabilities,"recipe_schema")]
    paper = capabilities.get("paper_research")
    if isinstance(paper,dict):
        locations.append((paper,"request_schema"))
    for parent,key in locations:
        schema = parent.get(key)
        if isinstance(schema,dict) and schema.get("$ref","").startswith("#/decision_schema/$defs/"):
            name = schema["$ref"].split("/$defs/",1)[1].replace("~1","/").replace("~0","~")
            definitions = expanded["decision_schema"]["$defs"]
            restored = deepcopy(definitions[name])
            if "original_definition_names" in schema:
                restored["$defs"] = {name:deepcopy(definitions[name]) for name in schema["original_definition_names"]}
            parent[key] = restored
    del expanded["planner_context_encoding"]
    return expanded


def decision_planner_evidence(bundle):
    """Curated decision view, not a reversible encoding of historical evidence.

    Full evidence remains durable. Keep scientific identities, champion recipes,
    source safety, memo and all valid parent IDs; omit verbose historical echoes.
    """
    if not isinstance(bundle, dict):
        raise ValueError("Planner bundle must be an object")
    projected = deepcopy(bundle)
    evidence = projected.get("evidence", projected)
    if not isinstance(evidence, dict):
        raise ValueError("Planner evidence must be an object")
    if "planner_decision_view" in projected:
        return projected
    original_bytes = len(_json(bundle).encode())
    evidence["proposal_audit_feedback"] = planner_proposal_feedback(evidence)
    from .experiment_discipline import POLICY
    evidence.setdefault("experiment_discipline", {"policy": POLICY,
        "champion_status": "development_only_unconfirmed",
        "scope": "Lifecycle unavailable in this legacy snapshot; do not infer pending work is absent."})
    omitted = {}
    memory = evidence.get("hypothesis_memory")
    if isinstance(memory, list):
        summaries = []
        for row in memory:
            if not isinstance(row, dict):
                omitted["malformed_hypothesis_records"] = omitted.get("malformed_hypothesis_records", 0) + 1
                continue
            payload = row.get("payload", {})
            summary = {k: deepcopy(v) for k, v in row.items() if k != "payload"}
            if not isinstance(payload, dict):
                payload = {}
                summary["payload_invalid"] = True
            summary["payload"] = {k: deepcopy(payload[k]) for k in (
                "planner_status", "error", "trial_budget", "allocated_trials", "unallocated_trials",
                "program_ids", "extensions", "retire_program_ids", "objective_name", "objective_value",
                "attempt_id", "status", "reason", "review_reason") if k in payload}
            programs = payload.get("programs", [])
            if isinstance(programs, list) and programs:
                summary["payload"]["programs"] = [{k: deepcopy(p[k]) for k in (
                    "proposal_id", "hypothesis", "changed_axes", "expected_observation",
                    "falsification_rule", "parent_trial_ids", "fixed_parameters", "max_trials",
                    "max_wall_seconds") if k in p} for p in programs if isinstance(p, dict)]
                invalid_count = sum(not isinstance(p, dict) for p in programs)
                if invalid_count:
                    summary["payload"]["programs_invalid_entries"] = invalid_count
            elif not isinstance(programs, list):
                summary["payload"]["programs_invalid"] = True
            for key in ("review_reason",):
                value = summary["payload"].get(key)
                if isinstance(value, str) and len(value) > 800:
                    summary["payload"][key] = value[:800]
                    summary["payload"][key + "_truncated"] = True
            for p in summary["payload"].get("programs", []):
                for key in ("hypothesis", "expected_observation", "falsification_rule"):
                    if isinstance(p.get(key), str) and len(p[key]) > 400:
                        p[key] = p[key][:400]
                        p[key + "_truncated"] = True
            summaries.append(summary)
        # Recent outcomes, not repeatedly embedding every previous complete decision.
        evidence["hypothesis_memory"] = summaries[:8]
        omitted["older_hypothesis_records"] = max(0, len(summaries) - 8)
        omitted["hypothesis_payloads"] = "Full recipes/decision echoes remain in durable evidence and ledgers"
    paper = evidence.get("paper_research")
    if isinstance(paper, dict):
        eligible = paper.get("eligible_attempts", [])
        if not isinstance(eligible, list) or any(not isinstance(row, dict) for row in eligible):
            raise ValueError("Paper eligible_attempts must be a list of objects")
        parents = evidence.get("completed_trial_index", [])
        if not isinstance(parents, list) or any(not isinstance(row, dict) for row in parents):
            raise ValueError("Completed parent index must be a list of objects")
        parent_ids = [row.get("attempt_id") for row in parents]
        index = {value: i for i, value in enumerate(parent_ids)}
        comparisons, keys, rows = [], {}, []
        for row in eligible:
            comparison = row.get("comparison", {})
            if not isinstance(comparison, dict):
                raise ValueError("Paper comparison must be an object")
            key = _json(comparison)
            if key not in keys:
                keys[key] = len(comparisons)
                comparisons.append(deepcopy(comparison))
            attempt = row.get("attempt_id")
            rows.append([index[attempt] if attempt in index else attempt, keys[key]])
        paper["eligible_attempts"] = {"encoding": "parent-comparison-table-v1",
            "columns": ["parent_index_or_attempt_id", "comparison_index"],
            "comparisons": comparisons, "rows": rows}
        histories = paper.get("by_comparison", {})
        if not isinstance(histories, dict) or any(
                not isinstance(records, list) or any(not isinstance(row, dict) for row in records)
                for records in histories.values()):
            raise ValueError("Paper by_comparison must map keys to lists of objects")
        compact_history = {}
        for key, records in histories.items():
            compact_history[key] = [{field: deepcopy(row[field]) for field in (
                "request_id", "status", "evaluation_key", "coverage", "probability_basis",
                "quote_mode", "ex_post_market_tainted", "executable_evidence", "assumptions",
                "comparison", "model_ids", "request", "unsupported_requirements", "paper_only",
                "summary", "metrics", "pools", "model_weights", "attempt_ids") if field in row}
                for row in records[-2:] if isinstance(row, dict)]
        paper["by_comparison"] = compact_history
        omitted["older_paper_records"] = sum(max(0, len(records)-2) for records in histories.values())
        omitted["paper_bulk_fields"] = "race_results_preview and report_path omitted; full reports remain in durable evidence"
    for key in ("champions_by_contract", "champions_by_family"):
        records = evidence.get(key)
        if isinstance(records, dict):
            for row in records.values():
                if isinstance(row, dict):
                    row.pop("artifacts", None)
    history = evidence.get("invocation_history")
    if isinstance(history, list):
        if any(not isinstance(row, dict) for row in history):
            raise ValueError("Invocation history must contain objects")
        evidence["invocation_history"] = [{k: deepcopy(row[k]) for k in (
            "invocation_id", "started_at", "last_observed_at", "cgroup_peak_bytes",
            "running_attempts") if k in row} for row in history[-2:]]
        omitted["older_invocations"] = max(0, len(history)-2)
    projected = compact_planner_evidence(projected)
    digest = hashlib.sha256(_json(projected).encode()).hexdigest()
    projected["planner_decision_view"] = {
        "version": 1, "full_evidence_id": evidence.get("evidence_id"),
        "projection_sha256": digest, "original_bytes": original_bytes,
        "projected_bytes_before_receipt": len(_json(projected).encode()), "omissions": omitted,
        "instructions": "Self-contained decision view, not lossless historical evidence. Latest champions and recipes, memo, availability and valid completed parent IDs are retained. Paper eligible_attempts table resolves an integer parent index against evidence.completed_trial_index[index].attempt_id; a string is a literal attempt ID. comparison_index resolves against the table comparisons. Only listed paper rows are eligible. Truncated historical explanations are labeled; never assume omitted outcomes succeeded. Full evidence remains stored under full_evidence_id. Copy complete literal attempt IDs, not indices, into proposals/requests.",
    }
    return projected
