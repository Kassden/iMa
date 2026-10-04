"""Reversible, JSON-only compression of repeated planner context metadata."""
from collections import Counter
from copy import deepcopy
import json


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
