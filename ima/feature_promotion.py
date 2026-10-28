"""Reference-specific feature evidence and fixed, complete-bundle transfers.

Screen merit is a scheduling hint within one comparison contract. It is never
incremental champion merit or evidence from an untouched confirmation sample.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from .dataset_registry import _digest, _write_json
from .feature_lifecycle import FeatureLifecycle, _outcome


def reference_identity(reference):
    required = {"attempt_id", "recipe", "parent_dataset_id", "comparison_contract",
                "code_revision", "environment_hash"}
    if not required <= reference.keys() or any(not reference[k] for k in required):
        raise ValueError("Reference requires pinned recipe, attempt, dataset, contract and runtime")
    _outcome({"attempt_id": reference["attempt_id"], "model_id": "reference",
              "dataset_id": reference["parent_dataset_id"], "definition_ids": [],
              "score": 0., "comparison_contract": reference["comparison_contract"]})
    return "feature-reference-" + _digest({k: reference[k] for k in sorted(required)})


def campaign_references(registry_root, directory, reference_directory=None):
    """Resolve actual completed compatible champions, never a historical scalar."""
    from .experiment_discipline import attempt_rows
    from .research_telemetry import champions
    lifecycle = FeatureLifecycle(registry_root)
    rows = attempt_rows(Path(directory) / "ledger.sqlite")
    if reference_directory:
        rows = attempt_rows(Path(reference_directory) / "ledger.sqlite") + rows
    by_id = {r["attempt_id"]: r for r in rows}
    references = []
    for champion in champions(rows)["champions_by_contract"].values():
        row = by_id[champion["attempt_id"]]
        result, recipe = row["result"], champion["recipe"]
        dataset_id = recipe.get("dataset_ref")
        if not dataset_id:
            continue
        manifest = lifecycle.registry.verify(dataset_id)
        lineage = result.get("lineage", {})
        needed = ("evaluation_population_id", "declared_evaluation_population_id", "protocol_hash",
                  "evaluation_population_hash", "probability_basis", "code_revision", "environment_hash")
        if any(not lineage.get(k) for k in needed):
            continue
        contract = {"comparison_contract_id": manifest["comparison_contract_id"],
                    "target_contract": recipe.get("target", {}).get("kind", "win_probability"),
                    "evaluation_population_id": lineage["evaluation_population_id"],
                    "declared_evaluation_population_id": lineage["declared_evaluation_population_id"],
                    "protocol_id": manifest["comparison_contract"]["protocol_id"],
                    "protocol_hash": lineage["protocol_hash"],
                    "score_population_sha256": lineage["evaluation_population_hash"],
                    "metric": result["objective_name"], "direction": "minimize",
                    "probability_basis": lineage["probability_basis"]}
        references.append({"attempt_id": row["attempt_id"], "recipe": recipe,
                           "parent_dataset_id": dataset_id, "comparison_contract": contract,
                           "code_revision": lineage["code_revision"], "environment_hash": lineage["environment_hash"]})
    return references


def _compatible(contract, reference):
    # Full evaluated identity, including protocol hash and probability basis.
    return contract == reference["comparison_contract"]


def promotion_view(registry_root, reference, *, limit=20):
    """Rank all persisted metadata before bounded hydration, without recency cuts."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Positive promotion limit required")
    reference_id = reference_identity(reference)
    lifecycle = FeatureLifecycle(registry_root)
    outcomes = {}
    for path in sorted((lifecycle.root / "outcomes").glob("*.json")):
        row = json.loads(path.read_text())
        outcomes.setdefault(row["build_id"], []).append(row)
    transfers = []
    for path in sorted((lifecycle.root / "transfers").glob("*.json")):
        row = json.loads(path.read_text())
        if "reference_id" not in row:
            continue
        if row["reference_id"] == reference_id:
            for suffix in ("attribution", "tracking"):
                extra = path.with_name(path.stem+"-"+suffix+".json")
                if extra.exists():
                    row[suffix] = json.loads(extra.read_text())
            transfers.append(row)
    by_build = {}
    for transfer in transfers:
        by_build.setdefault(transfer["build_id"], []).append(transfer)
    groups, pending, history = {}, [], []
    for path in sorted((lifecycle.root / "builds").glob("*.json")):
        receipt = json.loads(path.read_text())
        if receipt["spec"]["parent_dataset_id"] != reference["parent_dataset_id"]:
            continue
        build_id = receipt["build_id"]
        definitions = receipt["spec"]["definitions"]
        evidence = [r for r in outcomes.get(build_id, []) if r.get("paired")
                    and _compatible(r["outcome"]["comparison_contract"], reference)
                    and set(r["outcome"]["definition_ids"]) == {d["definition_id"] for d in definitions}]
        row = {"build_id": build_id, "dataset_id": receipt.get("dataset_id"),
               "definition_ids": sorted(d["definition_id"] for d in definitions),
               "status": receipt["status"], "reference_id": reference_id,
               "screen_evidence": evidence, "transfers": by_build.get(build_id, []),
               "next_action": "fixed_complete_bundle_with_matched_control"}
        if row["transfers"]:
            row["next_action"] = "await_pending_control_or_review_terminal_evidence"
            history.append(row)
            continue
        if receipt["status"] != "ready" or not evidence:
            row["next_action"] = "await_build_or_compatible_complete_bundle_screen"
            pending.append(row)
            continue
        # Conservative across repeats; never cherry-pick the best seed.
        row["screen_improvement"] = min(r["improvement"] for r in evidence)
        row["scope"] = "screen-specific development merit only"
        names = sorted(d["name"] for d in definitions)
        row["family"] = "|".join(sorted({name.split("__")[0].split("_")[0] for name in names}))
        groups.setdefault(row["family"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: (-r["screen_improvement"], r["build_id"]))
    ranked = []
    # Best overall first, then the best remaining member of each family.
    while groups:
        families = sorted(groups, key=lambda f: (-groups[f][0]["screen_improvement"], f))
        for family in families:
            ranked.append(groups[family].pop(0))
            if not groups[family]:
                del groups[family]
    result = {"reference_id": reference_id, "reference": deepcopy(reference),
              "candidates": ranked[:limit], "omitted_candidates": max(0, len(ranked)-limit),
              "pending": pending, "transfer_evidence": history,
              "scope": "adaptive development only; no automatic promotion"}
    result["evidence_id"] = _digest(result)
    return result


def validate_successor(parent, child, *, return_parent=False):
    """Validate real logical data, including all labels and unchanged parent values."""
    import pandas as pd
    if child.get("parent_dataset_id") != parent["dataset_id"]:
        raise ValueError("Transfer requires the exact feature parent")
    for key in ("comparison_contract_id", "comparison_contract", "evaluation_population_id"):
        if child.get(key) != parent.get(key) or not parent.get(key):
            raise ValueError("Transfer comparison/population mismatch: " + key)
    for name in ("protocol.json", "confirmation_keys.json"):
        if child["files"].get(name) != parent["files"].get(name) or not parent["files"].get(name):
            raise ValueError("Transfer protected protocol mismatch: " + name)
    from .dataset_delta import read_feature_table
    old, new = read_feature_table(parent["features_path"]), read_feature_table(child["features_path"])
    if not old.columns.is_unique or not new.columns.is_unique or not set(old.columns) <= set(new.columns):
        raise ValueError("Transfer parent schema mismatch")
    try:
        pd.testing.assert_frame_equal(old.reset_index(drop=True), new[old.columns].reset_index(drop=True), check_exact=True)
    except AssertionError as exc:
        raise ValueError("Transfer changes parent values, labels, row order or population") from exc
    if return_parent:
        return old


def transfer_plan(registry_root, build_id, reference, *, destination=None):
    from .research_specs import PipelineRecipe
    from .research_executor import RecipeExecutionRequest, _effective_frame_schema, _v6_feature_frame
    reference_id = reference_identity(reference)
    lifecycle = FeatureLifecycle(registry_root)
    receipt = lifecycle.receipt(build_id)
    if receipt["status"] != "ready":
        raise ValueError("Transfer needs a ready complete bundle")
    parent = lifecycle.registry.verify(reference["parent_dataset_id"])
    child = lifecycle.registry.verify(receipt["dataset_id"])
    parent_frame = validate_successor(parent, child, return_parent=True)
    control = PipelineRecipe.model_validate(reference["recipe"]).canonical_payload()
    if control.get("dataset_ref") != parent["dataset_id"]:
        raise ValueError("Reference recipe must bind the exact parent dataset")
    contract = reference["comparison_contract"]
    expected = {"comparison_contract_id": parent["comparison_contract_id"],
                "protocol_id": parent["comparison_contract"]["protocol_id"],
                "target_contract": PipelineRecipe.model_validate(control).target.kind}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Pinned reference differs from parent comparison contract")
    if control.get("feature_discovery") or control.get("feature_definitions"):
        raise ValueError("Adaptive/derived schemas require an explicitly frozen reference first")
    # Registered transforms are already an explicit frozen recipe. Both arms
    # retain the exact specifications and fit any state on the same past rows.
    # Rejecting them would exclude the actual rich champion from transfers.
    features = sorted(child["feature_build"]["feature_names"])
    if (not features or set(features) & set(parent_frame.columns)
            or set(features) & set(control.get("extra_numeric_features") or ())):
        raise ValueError("Complete bundle must add new feature columns")
    graph = control.get("pipeline_graph")
    if graph:
        from .pipeline_graph import PipelineGraph
        PipelineGraph.from_dict(graph).validate()
        nodes = {node["node_id"]: node for node in graph["nodes"]}
        if destination not in nodes or nodes[destination]["kind"] != "estimator":
            raise ValueError("Declare an estimator destination for graph transfers")
        control_recipe = PipelineRecipe.model_validate(control)
        schema_request = RecipeExecutionRequest("transfer-schema", "transfer-plan", 0, control_recipe,
            Path(parent["features_path"]), Path(parent["features_path"]).parent)
        parent_frame = _v6_feature_frame(parent_frame, schema_request)
        base = list(_effective_frame_schema(control_recipe, parent_frame).features)
        for node in list(nodes.values()):
            if node["kind"] == "estimator":
                # Feature routing belongs to graph feature_view nodes. An
                # estimator's parameters are model options, not feature lists.
                if not node.get("inputs"):
                    key = "transfer_features_" + node["node_id"]
                    while key in nodes:
                        key += "_"
                    view = {"node_id": key, "kind": "feature_view", "inputs": [],
                            "parameters": {"features": list(base)},
                            "output": {"kind": "feature_frame", "unit": "features"}}
                    graph["nodes"].append(view)
                    nodes[key] = view
                    node["inputs"] = [key]
    elif destination not in (None, "model"):
        raise ValueError("Non-graph destination must be model")
    candidate = deepcopy(control)
    candidate["dataset_ref"] = child["dataset_id"]
    candidate["extra_numeric_features"] = list(control.get("extra_numeric_features") or ()) + features
    if graph:
        candidate_graph = candidate["pipeline_graph"]
        nodes = {n["node_id"]: n for n in candidate_graph["nodes"]}
        node = nodes[destination]
        # Clone even an existing shared view: changing one estimator must not
        # add the bundle to another estimator through its shared parent.
        view = deepcopy(nodes[node["inputs"][0]])
        key = "transfer_bundle_" + destination
        while key in nodes:
            key += "_"
        view["node_id"] = key
        view["parameters"]["features"] = list(view["parameters"].get("features", base)) + features
        candidate_graph["nodes"].append(view)
        node["inputs"] = [key]
        PipelineGraph.from_dict(candidate_graph).validate()
    control = PipelineRecipe.model_validate(control).canonical_payload()
    candidate = PipelineRecipe.model_validate(candidate).canonical_payload()
    plan = {"reference_id": reference_id, "reference": deepcopy(reference), "build_id": build_id,
            "parent_dataset_id": parent["dataset_id"], "dataset_id": child["dataset_id"],
            "definition_ids": sorted(child["feature_definition_ids"]), "feature_ids": features,
            "destination": destination or "model", "control_recipe": control, "candidate_recipe": candidate,
            "changed_axes": ["complete_feature_bundle"], "fixed_parameters": True,
            "next_action": "run_complete_bundle_before_any_ablation_or_null_control"}
    plan["transfer_id"] = "feature-transfer-" + _digest(plan)
    return plan


def attribution_recipes(plan):
    """Full bundle must finish before these fixed child-dataset null/ablation fits."""
    null = deepcopy(plan["control_recipe"])
    null["dataset_ref"] = plan["dataset_id"]
    variants = {"null": null}
    for feature in plan["feature_ids"]:
        recipe = deepcopy(plan["candidate_recipe"])
        recipe["extra_numeric_features"].remove(feature)
        if recipe.get("pipeline_graph"):
            nodes = {n["node_id"]: n for n in recipe["pipeline_graph"]["nodes"]}
            node = nodes[plan["destination"]]
            nodes[node["inputs"][0]]["parameters"]["features"].remove(feature)
        variants["without-"+feature] = recipe
    return variants


def save_transfer(registry_root, plan, *, status="pending", result=None):
    lifecycle = FeatureLifecycle(registry_root)
    path = lifecycle.root / "transfers" / (plan["transfer_id"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lifecycle.registry._lock(plan["transfer_id"]):
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing and existing["plan"] != plan:
            raise ValueError("Transfer identity collision")
        if existing and status=='pending':
            return existing
        if existing and existing["status"] == "completed":
            if status == "completed" and existing["result"] != result:
                raise ValueError("Completed transfer evidence is immutable")
            return existing
        if status not in {"pending", "running", "failed", "completed"}:
            raise ValueError("Unknown transfer status")
        row = {"transfer_id": plan["transfer_id"], "reference_id": plan["reference_id"],
               "build_id": plan["build_id"], "plan": plan, "status": status, "result": result}
        _write_json(path, row)
        return row


def paired_transfer_report(candidate, control, target):
    """Reject changed scored labels before reusing race-level bootstrap reporting."""
    import pandas as pd
    from .feature_studies import paired_feature_report, race_objectives
    frames = [pd.read_csv(path) for path in (candidate, control)]
    keys = ["fold_id", "race_id", "horse_no"]
    labels = [c for c in ("date", "target_win", "label") if any(c in f for f in frames)]
    for frame in frames:
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError("Duplicate or missing scored runner identity")
    try:
        pd.testing.assert_frame_equal(*[f.sort_values(keys)[keys+labels].reset_index(drop=True) for f in frames], check_exact=True)
    except (AssertionError, KeyError) as exc:
        raise ValueError("Transfer scored labels/population mismatch") from exc
    report = paired_feature_report(candidate, control, target)
    if not math.isfinite(report["paired_delta"]):
        raise ValueError("Nonfinite transfer result")
    objectives = [race_objectives(f, target) for f in frames]
    paired = objectives[0].merge(objectives[1], on=["fold_id", "race_id"], suffixes=("_candidate", "_control"))
    paired["delta"] = paired.objective_candidate-paired.objective_control
    report["race_deltas"] = paired[["fold_id", "race_id", "delta"]].to_dict("records")
    report["fold_deltas"] = {str(k): float(v) for k, v in paired.groupby("fold_id").delta.mean().items()}
    return report
