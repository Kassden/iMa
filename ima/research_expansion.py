"""V6 controller: durable decisions, independent queues and a single ledger owner."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from . import research_controller as core
from .feature_discovery_specs import content_id
from .research_executor import RecipeExecutionRequest, execute_recipe, preparation_dependency_id, prepare_recipe_folds
from .research_hypotheses import HypothesisMemory
from .research_search import ProgramSearchController
from .research_specs import PipelineRecipe, ResearchProposal, V6_PORTFOLIO_VERSION
from .research_store import ResearchLedger, utc_now
from .research_telemetry import evidence_snapshot, log_snapshot, next_trace_number
from .research_betting import PaperResearchRequest
from .research_external_planner import read_external_decision as _external_decision
from .research_worker_runtime import BoundedFitExecutor, active_attempt_summary, cleanup_attempt_group, read_runtime
from .research_runtime_history import RuntimeHistory, read_own_unit_journal


class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decision_id: str
    evidence_id: str
    research_memo_sha256: str | None = None
    trial_budget: int = Field(ge=0, strict=True)
    programs: tuple[ResearchProposal, ...] = ()
    extensions: dict[str, StrictInt] = Field(default_factory=dict)
    retire_program_ids: tuple[str, ...] = ()
    dataset_requests: tuple[dict[str, Any], ...] = ()
    betting_requests: tuple[PaperResearchRequest, ...] = Field(default=(),max_length=4)
    unallocated_trials: int = Field(default=0, ge=0, strict=True)
    unallocated_reason: str | None = None
    review_reason: str | None = None

    @model_validator(mode="after")
    def budgets(self):
        if any(isinstance(v, bool) or v < 1 for v in self.extensions.values()):
            raise ValueError("Extensions must be positive integer trial allocations")
        allocated = sum(p.max_trials for p in self.programs) + sum(self.extensions.values())
        if allocated + self.unallocated_trials != self.trial_budget:
            raise ValueError("chosen budget must equal new + extensions + unallocated")
        if self.unallocated_trials and not (self.unallocated_reason or "").strip():
            raise ValueError("Unallocated trials require an explicit reason")
        if not (self.programs or self.extensions or self.retire_program_ids or self.dataset_requests or self.betting_requests) and not (self.review_reason or "").strip():
            raise ValueError("Review-only decisions require a reason")
        if len({p.proposal_id for p in self.programs}) != len(self.programs):
            raise ValueError("Duplicate proposal IDs")
        if set(self.extensions) & set(self.retire_program_ids):
            raise ValueError("Cannot extend and retire the same program")
        if len({request.request_id for request in self.betting_requests}) != len(self.betting_requests):
            raise ValueError("Duplicate paper request IDs")
        return self


class DecisionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, payload TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0)")
            conn.execute("CREATE TABLE IF NOT EXISTS receipts (decision_id TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(decision_id,action))")

    def record(self, decision: PlannerDecision, ceiling: int = 260) -> bool:
        if decision.trial_budget > ceiling:
            raise ValueError("Chosen trial budget exceeds the operator ceiling")
        payload = json.dumps(decision.model_dump(mode="json"), sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT payload FROM decisions WHERE id=?", (decision.decision_id,)).fetchone()
            if old:
                if old[0] != payload:
                    raise ValueError("Conflicting decision replay")
                return False
            conn.execute("INSERT INTO decisions(id,payload) VALUES (?,?)", (decision.decision_id, payload))
            return True

    def pending(self) -> list[PlannerDecision]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT payload FROM decisions WHERE applied=0 ORDER BY rowid").fetchall()
        return [PlannerDecision.model_validate_json(r[0]) for r in rows]

    def receipt(self, decision_id: str, action: str, payload: dict) -> dict:
        encoded = json.dumps(payload, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT id FROM decisions WHERE id=?", (decision_id,)).fetchone() is None:
                raise ValueError("Receipt references an unknown decision")
            old = conn.execute("SELECT payload FROM receipts WHERE decision_id=? AND action=?", (decision_id, action)).fetchone()
            if old:
                if old[0] != encoded:
                    raise ValueError("Conflicting action replay")
                return json.loads(old[0])
            conn.execute("INSERT INTO receipts VALUES (?,?,?)", (decision_id, action, encoded))
        return payload

    def action(self, decision_id: str, action: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT payload FROM receipts WHERE decision_id=? AND action=?", (decision_id, action)).fetchone()
        return json.loads(row[0]) if row else None

    def complete(self, decision_id: str):
        with sqlite3.connect(self.path) as conn:
            if conn.execute("UPDATE decisions SET applied=1 WHERE id=?", (decision_id,)).rowcount != 1:
                raise ValueError("Cannot complete an unknown decision")

    def authorized_program_ids(self):
        return set(self.authorized_budgets())

    def authorized_budgets(self):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT r.action,r.payload FROM receipts r JOIN decisions d ON d.id=r.decision_id WHERE d.applied=1 ORDER BY d.rowid,r.rowid").fetchall()
        budgets = {}
        for action,encoded in rows:
            payload = json.loads(encoded)
            if action.startswith("program:"):
                budgets.setdefault(payload["program_id"],payload["allocated"])
            elif action.startswith("extend:"):
                budgets[payload["program_id"]] = payload["budget"]
        return budgets

    def dataset_actions(self):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT r.payload FROM receipts r JOIN decisions d ON d.id=r.decision_id WHERE d.applied=1 AND r.action LIKE 'dataset:%' ORDER BY d.rowid,r.rowid").fetchall()
        return [json.loads(row[0])["request"] for row in rows]

    def betting_actions(self, *, pending_only=False):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT r.decision_id,r.action,r.payload FROM receipts r JOIN decisions d ON d.id=r.decision_id WHERE d.applied=1 AND r.action LIKE 'betting:%' ORDER BY d.rowid,r.rowid").fetchall()
        actions = []
        for decision_id,action,encoded in rows:
            payload = json.loads(encoded)
            completed = self.action(decision_id,"betting-complete:"+payload["action_id"])
            if not pending_only or completed is None:
                actions.append(dict(payload,decision_id=decision_id,completion=completed))
        return actions


def plan_decision(evidence: dict, config) -> dict:
    """One structured paid decision, independently of running fit capacity."""
    from .openrouter_orchestrator import choose_research_decision
    limits = {"trial_ceiling": config.proposal_batch_size,
              "max_new_programs": config.max_new_programs_per_decision,
              "max_pending_programs": config.max_pending_programs}
    if config.planner_mode == "external":
        return _external_decision(evidence, config)
    if config.planner_mode == "fixture":
        budget = min(config.proposal_batch_size, 5)
        recipes = core._v4_seed_proposals()[:min(2,budget,config.max_new_programs_per_decision)]
        programs = []
        for i, proposal in enumerate(recipes):
            recipe = proposal.recipe.model_copy(update={"schema_version": 3})
            programs.append(proposal.model_copy(update={
                "proposal_id": f"{evidence['decision_id']}-fixture-{i}", "recipe": recipe,
                "max_trials": budget-len(recipes)+1 if i == 0 else 1,
                "evidence_ids": (evidence["evidence_id"],),
            }).model_dump(mode="json"))
        return {"decision": {"decision_id": evidence["decision_id"], "evidence_id": evidence["evidence_id"],
                             "trial_budget": sum(p["max_trials"] for p in programs), "programs": programs},
                "usage": {"cost_status": "fixture"}}
    from .openrouter_orchestrator import OpenRouterConfig
    remote = OpenRouterConfig.from_env(model=config.model, max_output_tokens=config.max_output_tokens,
                                      timeout_seconds=config.planner_timeout_seconds,
                                      reasoning_effort=config.planner_reasoning_effort)
    calls = []
    transport_events = []
    transport_number = 0
    def observe_transport(event):
        nonlocal transport_number, transport_events
        if event["phase"] == "request_started":
            transport_number += 1
            transport_events = []
        transport_events.append(event)
        core._write_json_atomic(Path(config.campaign_dir)/"planner-transport"/f"{evidence['decision_id']}-{transport_number:02d}.json",
            {"evidence_id":evidence["evidence_id"],"events":transport_events,"received_at":utc_now()})
    def observe(response):
        calls.append(response)
        core._write_json_atomic(Path(config.campaign_dir)/"planner-calls"/f"{evidence['decision_id']}-{len(calls):02d}.json",
                                {"response": response, "evidence_id": evidence["evidence_id"], "received_at": utc_now()})
        from .openrouter_orchestrator import normalize_openrouter_usage
        cost = normalize_openrouter_usage(response)["total_cost_usd"]
        if cost is None or not math.isfinite(cost):
            raise ValueError("Planner call cost is unknown; further paid calls are frozen")
    remote = replace(remote, response_observer=observe, transport_observer=observe_transport,
                     absolute_deadline_seconds=config.planner_timeout_seconds)
    return choose_research_decision(evidence, limits, remote)


def apply_decision(decision, store, search, retired, config, evidence, profile):
    """Replay safe actions; no asks are admitted until all allocations are durable."""
    if decision.evidence_id != evidence["evidence_id"]:
        raise ValueError("Stale decision evidence")
    if len(decision.programs) > config.max_new_programs_per_decision:
        raise ValueError("Too many new programs in one decision")
    if not set(decision.extensions) | set(decision.retire_program_ids) <= set(search.programs):
        raise ValueError("Unknown extension or retirement program")
    if any(search.programs[pid].fixed_parameters for pid in decision.extensions):
        raise ValueError("Fixed controls cannot be extended")
    if decision.trial_budget > config.proposal_batch_size:
        raise ValueError("Chosen trial budget exceeds the operator ceiling")
    betting = _preflight_betting_requests(decision,store,config,evidence)
    registry = _dataset_registry(config)
    requested = {}
    if decision.dataset_requests:
        from .dataset_specs import DatasetRequest
        source,raw = _dataset_build_inputs(config)
        for value in decision.dataset_requests:
            request = DatasetRequest.model_validate(value)
            if request.evidence_watermark != decision.evidence_id:
                raise ValueError("Dataset request has stale evidence watermark")
            if request.request_id in requested:
                raise ValueError("Duplicate dataset request IDs")
            _validate_raw_manifest(request,raw)
            requested[request.request_id] = request
            state_path = registry.root/"requests"/(request.request_id+".json")
            if state_path.is_file() and registry.get(request.request_id)["request_fingerprint"] != request.fingerprint():
                raise ValueError("Conflicting dataset request replay")
    for proposal in decision.programs:
        if proposal.recipe.schema_version != 3:
            raise ValueError("V6 proposals require recipe schema_version=3")
        if proposal.evidence_ids != (evidence["evidence_id"],):
            raise ValueError("Proposal has stale evidence")
        known = {x["attempt_id"] for x in evidence.get("completed_trial_index", [])}
        if not set(proposal.parent_trial_ids) <= known:
            raise ValueError("Unknown parent trial")
        if getattr(config,"dataset_path",None):
            ref = proposal.recipe.dataset_ref
            awaiting_publication = ref in requested
            if ref and not awaiting_publication and not ref.startswith("dataset-"):
                state = registry.get(ref)
                awaiting_publication = state["status"] in {"requested","building","validating"}
            if not awaiting_publication:
                context = _program_context(proposal.recipe,config,registry)
                _preflight_recipe(proposal.recipe,context,config)
        if profile is not None and not proposal.recipe.dataset_ref:
            checked = proposal.recipe
            if checked.feature_discovery and checked.feature_discovery.schema_version == 2:
                standard = tuple(m for m in checked.feature_discovery.measurements if m in {"speed_mps","beaten_lengths","carried_weight"})
                checked = checked.model_copy(update={"feature_discovery":checked.feature_discovery.model_copy(update={"measurements":standard})})
            error = profile.admission_error(checked)
            if error:
                raise ValueError(error)
    registrations = _decision_feature_definitions(decision,config,registry)
    from .feature_definitions import FeatureRegistry
    feature_registry = FeatureRegistry(registry.feature_registry)
    for request in requested.values():
        for identifier in request.feature_definition_ids:
            if identifier not in registrations:
                try:
                    feature_registry.get(identifier)
                except (OSError,ValueError) as exc:
                    raise ValueError(f"Dataset request references an unknown feature definition: {identifier}") from exc
    store.record(decision, ceiling=config.proposal_batch_size)
    for definition,metadata in registrations.values():
        feature_registry.register(definition,metadata)
    accepted = []
    for proposal in decision.programs:
        pid = search.register(proposal)
        store.receipt(decision.decision_id, f"program:{pid}", {"program_id": pid, "allocated": proposal.max_trials})
        accepted.append(pid)
    for pid, extra in decision.extensions.items():
        receipt = store.action(decision.decision_id, f"extend:{pid}")
        if receipt is None:
            receipt = store.receipt(decision.decision_id, f"extend:{pid}",
                                    {"program_id": pid, "budget": search.programs[pid].max_trials+extra,
                                     "allocated": extra})
        search.set_budget(pid, receipt["budget"])
    retired.update(decision.retire_program_ids)
    core._write_json_atomic(Path(config.campaign_dir)/"retired-programs.json", sorted(retired))
    # Dataset actions are persisted before asynchronous preparation, never live-patched.
    for i, request in enumerate(decision.dataset_requests):
        store.receipt(decision.decision_id, f"dataset:{i}", {"request": request})
        registry.submit(request)
    for payload in betting:
        store.receipt(decision.decision_id,"betting:"+payload["action_id"],payload)
    store.complete(decision.decision_id)
    return accepted


def _current_paper_terminal(directory):
    path = Path(directory)/"ledger.sqlite"
    if not path.is_file():
        raise ValueError("Paper requests require the current completed-attempt ledger")
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro",uri=True) as conn:
        rows = conn.execute("SELECT attempt_id,status,payload_json,result_json FROM attempts WHERE status='completed' ORDER BY attempt_id").fetchall()
    return [{"attempt_id":attempt,"status":status,"payload":json.loads(payload),"result":json.loads(result)}
            for attempt,status,payload,result in rows]


def _preflight_betting_requests(decision,store,config,evidence):
    if not decision.betting_requests:
        return []
    from .research_betting import validate_paper_research
    known = {item["attempt_id"] for item in evidence.get("completed_trial_index",())}
    existing = store.betting_actions()
    pending = [item for item in existing if item["completion"] is None and item["decision_id"]!=decision.decision_id]
    if len(pending)+len(decision.betting_requests)>8:
        raise ValueError("Paper queue exceeds its independent eight-action ceiling")
    terminal = _current_paper_terminal(config.campaign_dir)
    validated = []
    for request in decision.betting_requests:
        if request.evidence_id!=decision.evidence_id or not set(request.attempt_ids)<=known:
            raise ValueError("Paper requests require exact evidence and current completed attempt IDs")
        body = request.model_dump(mode="json")
        action_id = content_id({"decision_id":decision.decision_id,"request":body})
        if any(item["request"]["request_id"]==request.request_id and item["action_id"]!=action_id for item in existing):
            raise ValueError("Paper request ID is already bound to another immutable action")
        previous = store.action(decision.decision_id,"betting:"+action_id)
        if previous is not None:
            validated.append(previous)
            continue
        preflight = validate_paper_research(request,campaign_dir=Path(config.campaign_dir),terminal_results=terminal)
        validated.append({"action_id":action_id,"request":body,"preflight":preflight,
                          "trace_number":len(existing)+len(validated)+1})
    return validated


def _paper_report_path(directory,action_id):
    import re
    if not re.fullmatch(r"[0-9a-f]{24}",action_id):
        raise ValueError("Invalid immutable paper action identity")
    return Path(directory)/"paper-actions"/(action_id+".json")


def _paper_workload(action,revision,environment):
    from .research_resources import JobWorkload
    request = PaperResearchRequest.model_validate(action["request"])
    return JobWorkload(stage="simulation",family="paper_plackett_luce",
        rows=max(1,int(action["preflight"].get("rows",1))),generated_features=0,
        selected_features=len(request.attempt_ids),native_threads=1,
        search_settings={"simulations":request.simulations,"max_races":request.max_races,
                         "models":len(request.attempt_ids),"pools":request.pools},
        implementation_revision=revision,dependency_versions={"environment_hash":environment})


def _paper_worker(request,campaign_dir,terminal):
    from .research_betting import evaluate_paper_research
    from .research_resources import JobMonitor
    from threadpoolctl import threadpool_limits
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with JobMonitor() as monitor,threadpool_limits(limits=1):
        try:
            result = {"success":True,"report":evaluate_paper_research(request,
                      campaign_dir=Path(campaign_dir),terminal_results=terminal)}
        except Exception as exc:
            result = {"success":False,"error":str(exc)}
    return result | {"worker_resources":monitor.report()}


def _paper_estimate(estimator,workload,action):
    cold = int(action["preflight"].get("cold_private_memory_bytes",512*1024**2))
    estimate = estimator.estimate(workload,cold_private_bytes=cold)
    return replace(estimate,private_peak_bytes=max(estimate.private_peak_bytes,math.ceil(cold*estimate.margin)),
                   disk_bytes=128*1024**2)


def _publish_paper_report(store,directory,action,report,resources=None):
    request = action["request"]
    if (report.get("request_id")!=request["request_id"] or report.get("evidence_id")!=request["evidence_id"]
            or report.get("paper_only") is not True or report.get("executable_evidence") is not False):
        raise ValueError("Paper report identity or offline-only contract mismatch")
    if report.get("coverage",{}).get("evaluated_races",0)<1:
        raise ValueError("No paper races were evaluated")
    for key in ("comparison","evaluation_key","model_ids"):
        if report[key]!=action["preflight"][key]:
            raise ValueError("Paper inputs changed after decision preflight")
    path = _paper_report_path(directory,action["action_id"])
    body = {"action_id":action["action_id"],"request":request,"report":report,
            "worker_resources":resources or {},"created_at":utc_now()}
    if path.is_file():
        previous = core._read_json(path)
        if previous["action_id"]!=action["action_id"] or previous["request"]!=request or previous["report"]!=report:
            raise ValueError("Conflicting immutable paper report replay")
    else:
        core._write_json_atomic(path,body)
    completion = {"status":"completed","report_path":str(path),"report_sha256":core._hash_file(path),
                  "evaluation_key":report["evaluation_key"]}
    return store.receipt(action["decision_id"],"betting-complete:"+action["action_id"],completion)


def _recover_paper_report(store,directory,action):
    path = _paper_report_path(directory,action["action_id"])
    if not path.is_file():
        return None
    body = core._read_json(path)
    if body["action_id"]!=action["action_id"] or body["request"]!=action["request"]:
        raise ValueError("Published paper artifact belongs to another action")
    return _publish_paper_report(store,directory,action,body["report"],body.get("worker_resources"))


def _paper_attempt_count(store,action):
    return sum(store.action(action["decision_id"],f"betting-attempt:{action['action_id']}:{number}") is not None
               for number in range(1,4))


def _paper_evidence(store,directory,terminal):
    current = [row["result"].get("lineage",{}) for row in terminal
               if row["status"]=="completed" and row["result"].get("target_kind")=="win_probability"]
    identity_fields = ("dataset_hash","protocol_id","protocol_hash","evaluation_population_id",
                       "evaluation_population_hash","availability_policy")
    eligible = [{"attempt_id":row["attempt_id"],"comparison":{key:row["result"]["lineage"][key] for key in identity_fields}}
                for row in terminal if row["status"]=="completed" and row["result"].get("target_kind")=="win_probability"
                and all(row["result"].get("lineage",{}).get(key) for key in identity_fields)
                and row["result"].get("lineage",{}).get("prediction_sha256")
                and all(row["result"].get("artifacts",{}).get(key) for key in ("predictions","protocol"))]
    grouped,failures = {},[]
    for action in store.betting_actions():
        completion = action["completion"]
        if completion is None:
            continue
        if completion["status"]!="completed":
            failures.append({"request_id":action["request"]["request_id"],"error":completion.get("error")})
            continue
        path = _paper_report_path(directory,action["action_id"])
        if core._hash_file(path)!=completion["report_sha256"]:
            raise ValueError("Immutable paper report hash changed")
        report = core._read_json(path)["report"]
        if not any(all(lineage.get(key)==value for key,value in report["comparison"].items()) for lineage in current):
            continue
        key = report["evaluation_key"]+":"+report["probability_basis"]+":"+report["request"]["quote_mode"]
        summary = {name:report[name] for name in ("request_id","evaluation_key","comparison","probability_basis",
                   "ex_post_market_tainted","coverage","assumptions","unsupported_requirements")}
        summary.update(report_path=str(path),model_ids=report["model_ids"],request=report["request"],quote_mode=report["request"]["quote_mode"],
                       paper_only=True,executable_evidence=False)
        summary["coverage"] = dict(report["coverage"],
            skipped_race_count=len(report["coverage"].get("skipped_races",[])),
            skipped_races=report["coverage"].get("skipped_races",[])[:8])
        summary["race_results_preview"] = report.get("races",[])[:2]
        grouped.setdefault(key,[]).append(summary)
    return {"by_comparison":{key:items[-8:] for key,items in grouped.items()},"failures":failures[-8:],
            "eligible_attempts":eligible,
            "pending":len(store.betting_actions(pending_only=True)),"scope":"Paper development research only; no independent strategy validation or wagers"}


def _trace_paper_action(store,directory,action,config):
    if store.action(action["decision_id"],"betting-traced:"+action["action_id"]) is not None:
        return
    completion = action["completion"]
    outbox = Path(directory)/"trace-outbox.sqlite"
    if outbox.is_file():
        with sqlite3.connect(f"file:{outbox.resolve()}?mode=ro",uri=True) as conn:
            exists = conn.execute("SELECT delivered FROM traces WHERE role='betting' AND number=?",(action["trace_number"],)).fetchone()
        if exists is not None:
            store.receipt(action["decision_id"],"betting-traced:"+action["action_id"],
                          {"status":"delivered" if exists[0] else "queued"})
            return
    payload = {"evidence_id":action["request"]["evidence_id"],"decision_id":action["decision_id"],
        "request_id":action["request"]["request_id"],"paper_only":True,"action_id":action["action_id"],
        "completion":completion,"native_planner_cost_usd":None,"additional_planner_calls":0}
    linkage = log_snapshot(Path(directory),payload,config,role="betting",number=action["trace_number"])
    store.receipt(action["decision_id"],"betting-traced:"+action["action_id"],{"trace":linkage})


def _decision_feature_definitions(decision,config,registry):
    """Validate the whole batch before publishing definitions used by dataset actions."""
    from .feature_definitions import FeatureDefinition
    from .research_executor import _load_dataset,_v6_feature_frame
    registrations = {}
    for proposal in decision.programs:
        recipe = proposal.recipe
        if not recipe.feature_definitions:
            continue
        try:
            context = _program_context(recipe,config,registry)
        except (OSError,ValueError):
            if not recipe.dataset_ref or recipe.dataset_ref.startswith("dataset-"):
                raise
            # Unpublished datasets cannot supply trusted catalog metadata yet.
            context = _program_context(recipe.model_copy(update={"dataset_ref":None}),config,registry)
        request = RecipeExecutionRequest("definition-validation",proposal.proposal_id,0,recipe,context["dataset"],Path(config.campaign_dir)/"definition-validation"/recipe.recipe_hash(),context["protocol"],context["digest"])
        frame = _v6_feature_frame(_load_dataset(context["dataset"]),request)
        metadata = frame.attrs.get("synthesis_metadata",frame.attrs["input_metadata"])
        for payload in recipe.feature_definitions:
            definition = FeatureDefinition.model_validate(payload)
            registrations.setdefault(definition.content_id(),(definition,metadata))
    return registrations


def _worker(request, workload, threads):
    from .research_resources import JobMonitor
    from threadpoolctl import threadpool_limits
    os.environ["IMA_RESEARCH_THREADS"] = str(threads)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    with JobMonitor(interval=.1) as monitor, threadpool_limits(limits=threads):
        result = execute_recipe(request)
    result.metrics["worker_resources"] = monitor.report() | {"native_threads": threads,
                                                            "workload_fingerprint": workload.fingerprint()}
    core._write_json_atomic(request.output_dir/"result.json", result.serializable())
    return result


def _prepare_program(dataset, recipe, digest, root):
    from .research_resources import JobMonitor
    with JobMonitor() as monitor:
        prepared = _prepare_program_frame(dataset,recipe,digest,root)
    return dict(prepared,worker_resources=monitor.report())


def _prepare_fold_dependency(request, workload):
    from .research_resources import JobMonitor
    with JobMonitor() as monitor:
        artifacts = prepare_recipe_folds(request)
    return {"artifacts":artifacts,"worker_resources":monitor.report()}


def _dependency_request(recipe,context,directory,revision,environment):
    identity = preparation_dependency_id(recipe,context["digest"],context["protocol"],
        code_revision=revision,environment_hash=environment,
        dataset_manifest_hash=core._hash_file(context["dataset"].parent/"manifest.json") if (context["dataset"].parent/"manifest.json").is_file() else None)
    request = RecipeExecutionRequest(identity or "graph", "fold-dependency",0,recipe,
        context["dataset"],directory/"trials"/("prepare-"+(identity or "graph")),
        context["protocol"],context["digest"],revision,environment,V6_PORTFOLIO_VERSION)
    return identity,request


def _pending_fold_request(payload, context, directory):
    bound = dict(context, dataset=Path(payload.get("dataset_path",context["dataset"])),
                 digest=payload["dataset_hash"], protocol=payload["protocol_parameters"])
    return _dependency_request(PipelineRecipe.model_validate(payload["recipe"]),bound,directory,
                               payload["code_revision"],payload["environment_hash"])


class _FoldDependencies:
    """Parent-owned preparation circuit and reader leases, shared by all followers."""

    def __init__(self, directory):
        import weakref
        self.path = Path(directory)/"fold-dependency-errors.json"
        stored = core._read_json(self.path) if self.path.is_file() else {"schema_version":1,"errors":{}}
        if stored.get("schema_version") != 1 or not isinstance(stored.get("errors"),dict):
            raise ValueError("Invalid fold dependency error state")
        self.errors = stored["errors"]
        for error in self.errors.values():
            if not isinstance(error,dict) or not isinstance(error.get("failures"),int) or error["failures"] < 1:
                raise ValueError("Invalid fold dependency failure count")
            error["blocked"] = error["failures"] >= 3
        self.ready, self._pins, self._stamps = {}, {}, {}
        # Finalization covers campaign returns without changing the executor lifetime.
        self._finalizer = weakref.finalize(self,type(self)._release_pins,self._pins)

    @staticmethod
    def _release_pins(pins):
        while pins:
            _, stack = pins.popitem()
            stack.close()

    def close(self):
        self._finalizer()
        self.ready.clear()
        self._stamps.clear()

    def _save(self):
        core._write_json_atomic(self.path,{"schema_version":1,"errors":self.errors})

    def blocked(self, dependency):
        return self.errors.get(dependency,{}).get("blocked",False)

    def can_prepare(self, dependency):
        error = self.errors.get(dependency,{})
        return not self.blocked(dependency) and time.time() >= error.get("retry_at_epoch",0)

    def failed(self, dependency, pid, exc):
        self.invalidate(dependency)
        failures = self.errors.get(dependency,{}).get("failures",0)+1
        diagnostic = getattr(exc,"diagnostic",None)
        self.errors[dependency] = {"error":f"{type(exc).__name__}: {exc}","program_id":pid,
            "failures":failures,"blocked":failures>=3,"last_failure_at_epoch":time.time(),
            "retry_at_epoch":time.time()+30,"diagnostic":diagnostic}
        self._save()

    def invalidate(self, dependency):
        self.ready.pop(dependency,None)
        self._stamps.pop(dependency,None)
        stack = self._pins.pop(dependency,None)
        if stack is not None:
            stack.close()

    @staticmethod
    def _stamp(artifact, names):
        result = []
        for name in names:
            stat = (artifact.path/name).stat()
            result.append((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
        return tuple(result)

    def is_ready(self, dependency):
        if dependency is None:
            return True
        if self.blocked(dependency) or dependency not in self.ready:
            return False
        try:
            # Checksums are verified when acquiring the lease. Inode/ctime/mtime checks
            # detect replacement or corruption without hashing whole memmaps each loop.
            for name, artifact in self.ready[dependency].items():
                files, before = self._stamps[dependency][name]
                if self._stamp(artifact,files) != before:
                    self.invalidate(dependency)
                    return False
        except OSError:
            self.invalidate(dependency)
            return False
        return True

    def publish(self, dependency, artifacts):
        from contextlib import ExitStack
        from .research_preparation import PreparationArtifact, PreparationCache
        if not isinstance(artifacts,dict) or not artifacts:
            raise ValueError("Fold dependency must publish nonempty artifact descriptors")
        stamps = {}
        with ExitStack() as pins:
            for name, artifact in artifacts.items():
                if not isinstance(artifact,PreparationArtifact):
                    raise TypeError("Invalid fold preparation descriptor")
                pins.enter_context(PreparationCache(artifact.root,lock_timeout=0).pin(artifact))
                manifest = artifact.manifest()
                artifact.load_arrays()
                files = ("manifest.json",*sorted(manifest["checksums"]))
                stamps[name] = (files,self._stamp(artifact,files))
            self.errors.pop(dependency,None)
            self._save()
            self.invalidate(dependency)
            self._pins[dependency] = pins.pop_all()
            self._stamps[dependency] = stamps
            self.ready[dependency] = dict(artifacts)


def _fail_fold_allocation(row, dependency, error, ledger, hypotheses, directory):
    """Terminalize only an unstarted durable allocation; preserve its attempt identity."""
    from .research_executor import RecipeExecutionResult
    attempt, payload = row["attempt_id"], row["payload"]
    if attempt not in {item["attempt_id"] for item in ledger.pending_attempts()}:
        return False
    recipe = PipelineRecipe.model_validate(payload["recipe"])
    diagnostic = dict(error,dependency_id=dependency,failure_kind="infrastructure_preparation_failed")
    result = RecipeExecutionResult(1,attempt,payload["proposal_id"],payload["trial_number"],
        recipe.recipe_hash(),recipe.target.kind,"failed","preparation_failed",None,
        {"infrastructure_failure":diagnostic},{},
        {name:payload[name] for name in ("dataset_hash","code_revision","environment_hash")},
        0.,f"FoldPreparationBlocked: {error['error']}").serializable()
    result.update(core._portfolio_identity(recipe,"expansion_v6"),
                  target_parameters=recipe.target.parameters,failure_class="preparation_failed")
    ledger.complete_attempt(attempt,result,status="failed")
    core._write_json_atomic(Path(directory)/"trials"/attempt/"result.json",result)
    core._append_jsonl(Path(directory)/"trials.jsonl",result)
    hypotheses.record("outcome:"+attempt,payload["program_id"],"evaluated",
        {"attempt_id":attempt,"objective_value":None,"status":"failed",
         "error":result["error"],"failure_class":"preparation_failed","dependency_id":dependency})
    return True


def _prepare_program_frame(dataset, recipe, digest, root):
    from .research_executor import _load_dataset, _v6_feature_frame
    if core._hash_file(dataset) != digest:
        raise ValueError("Preparation dataset digest changed")
    frame = _load_dataset(dataset)
    request = RecipeExecutionRequest("preparation", "preparation", 0, recipe,Path(dataset),Path(root)/"program-preparation"/recipe.recipe_hash(),{},digest,"preparation","preparation",V6_PORTFOLIO_VERSION)
    frame = _v6_feature_frame(frame,request)
    if recipe.feature_discovery:
        from .feature_program import materialize
        _, manifest = materialize(frame, recipe.feature_discovery, digest, root/"discovery-cache", shared=True,input_metadata=frame.attrs.get("synthesis_metadata",frame.attrs.get("input_metadata")))
        return dict(manifest,dataset_digest=digest)
    return {"matrix_id": digest,"dataset_digest":digest, "shared_bytes": 0, "status": "base_features_ready","rows":len(frame),"formula_features":list(frame.attrs.get("formula_features",()))}


def _workload(recipe, rows, revision, environment, *, stage="fit", shared=None,native_threads=1):
    from .research_resources import JobWorkload
    schema = core.FEATURE_SCHEMAS[recipe.feature_schema]
    spec = recipe.feature_discovery
    if stage == "preparation" and (spec or recipe.feature_definitions):
        stage = "feature_generation"
    selected = (getattr(spec, "max_selected", 16) or getattr(spec, "max_definitions", 500)) if spec else 0
    return JobWorkload(stage=stage, family=recipe.model.kind if not recipe.pipeline_graph else "graph:"+recipe.model.kind,
        rows=rows, generated_features=spec.max_definitions if spec else 0,
        selected_features=len(schema.numeric)+selected+len(recipe.extra_numeric_features or ()),
        categorical_cardinalities=(128,128) if schema.categorical else (), folds=3,
        native_threads=native_threads, native_thread_settings={"native":native_threads}, search_settings={"depth":recipe.model.parameters.get("depth"),
            "quadrature_order":recipe.model.parameters.get("quadrature_order"),
            "heteroscedastic":recipe.model.parameters.get("heteroscedastic"),
            "iteration_bucket":math.ceil(recipe.model.parameters.get("max_iter",recipe.model.parameters.get("iterations",200))/200),
            "graph_structure":_graph_cost_structure(recipe.pipeline_graph),
            "discovery_id":spec.discovery_id() if spec else None},
        shared_artifacts=shared or {}, implementation_revision=revision,
        dependency_versions={"environment":environment})


def _graph_cost_structure(graph):
    if not graph:
        return None
    cost = {"depth","max_depth","max_leaf_nodes","max_iter","iterations","n_estimators","folds","samples","quadrature_order","heteroscedastic"}
    return [{"id":node.get("node_id"),"kind":node.get("kind"),"inputs":node.get("inputs",()),"model":node.get("parameters",{}).get("model_kind"),"cost":{k:v for k,v in node.get("parameters",{}).get("model_parameters",{}).items() if k in cost}} for node in graph.get("nodes",())]


def _native_threads(recipe,rows,cpu_budget,*,pending_trials=None):
    kinds = {recipe.model.kind}
    if recipe.pipeline_graph:
        kinds.update(node.get("parameters",{}).get("model_kind","") for node in recipe.pipeline_graph["nodes"])
    parallel = {"catboost","catboost_regressor","catboost_ranker","lightgbm_ranker","boosted","hist_gradient_regressor","gaussian_probit","benter_conditional_logit"}
    threads = 4 if rows >= 50000 else 2 if rows >= 2000 else 1
    if pending_trials is not None:
        threads = min(threads, max(1, cpu_budget // max(1, pending_trials)))
    return min(cpu_budget,threads) if kinds & parallel else 1


def _dataset_registry(config):
    from .dataset_registry import DatasetRegistry
    return DatasetRegistry(getattr(config,"dataset_registry_path",None) or Path(config.campaign_dir)/"dataset-registry")


def _dataset_build_inputs(config):
    snapshot = getattr(config,"official_snapshot_path",None)
    if snapshot is None:
        raise ValueError("Dataset requests require an explicit official_snapshot_path")
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot/"manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Official snapshot manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    raw = manifest.get("raw_manifest_path")
    raw_path = (snapshot/str(raw)).resolve() if raw else (snapshot/"raw_manifest.json")
    if not raw_path.is_file():
        # Registry explicitly accepts the hash-bound official snapshot manifest itself.
        raw_path = manifest_path
    return snapshot,raw_path


def _raw_manifest_identity(path):
    manifest = json.loads(Path(path).read_text())
    digest = core._hash_file(path)
    identities = {digest}
    if isinstance(manifest,dict):
        identities.update(value for key in ("manifest_id","raw_corpus_manifest_id","snapshot_id")
                          if isinstance(value := manifest.get(key),str) and value)
    return {"raw_manifest_sha256":digest,"raw_corpus_manifest_id":digest,
            "accepted_raw_corpus_manifest_ids":sorted(identities)}


def _validate_raw_manifest(request,path):
    if request.raw_corpus_manifest_id not in _raw_manifest_identity(path)["accepted_raw_corpus_manifest_ids"]:
        raise ValueError("Dataset request raw manifest identity mismatch")


def _program_context(recipe,config,registry):
    dataset,protocol_path = Path(config.dataset_path),Path(config.protocol_path)
    manifest = {}
    if recipe.dataset_ref:
        ref = recipe.dataset_ref
        if ref.startswith("dataset-") and len(ref) == 72:
            manifest = registry.verify(ref)
        else:
            state = registry.get(ref)
            if state["status"] != "verified":
                raise ValueError(f"Dataset request is not ready: {ref} ({state['status']})")
            manifest = registry.verify(state["dataset_id"])
        dataset,protocol_path = Path(manifest["features_path"]),Path(manifest["protocol_path"])
        stored = json.loads(protocol_path.read_text())
        protocol = core.protocol_spec_parameters(stored)
    else:
        protocol = core._protocol_parameters(config)
    return {"dataset":dataset,"digest":core._hash_file(dataset),"protocol":protocol,"manifest":manifest,"rows":manifest.get("rows")}


def _preflight_recipe(recipe,context,config):
    from .research_executor import _load_dataset,_v6_feature_frame
    request = RecipeExecutionRequest("preflight","preflight",0,recipe,context["dataset"],Path(config.campaign_dir)/"preflight"/recipe.recipe_hash(),context["protocol"],context["digest"],"preflight","preflight",V6_PORTFOLIO_VERSION)
    frame = _v6_feature_frame(_load_dataset(context["dataset"]),request)
    if recipe.pipeline_graph:
        from .pipeline_graph import PipelineGraph
        PipelineGraph.from_dict(recipe.pipeline_graph).validate()
    from .research_targets import apply_target_contract,target_contract
    from .research_evaluation import build_expanding_folds
    contract = target_contract(recipe.target.kind,dict(recipe.target.parameters))
    labelled = apply_target_contract(frame,contract)
    build_expanding_folds(labelled,**context["protocol"])
    context["rows"] = len(frame)
    return context


def _build_dataset(registry_root,request_id,source_snapshot,raw_manifest,feature_registry=None):
    from .dataset_registry import DatasetRegistry
    from .research_resources import JobMonitor
    with JobMonitor() as monitor:
        manifest = DatasetRegistry(registry_root,feature_registry=feature_registry).build(request_id,source_snapshot=source_snapshot,raw_manifest=raw_manifest)
    return dict(manifest,worker_resources=monitor.report())


def _recover_orphan_asks(search,ledger,contexts,authorized,revision,environment):
    """Bridge Optuna's durable ask to the execution ledger after an interrupted reserve."""
    known = {(row["payload"]["program_id"],row["payload"]["trial_number"]) for row in ledger.reserved_attempts()}
    recovered = 0
    for pid in authorized:
        if pid not in contexts:
            continue
        proposal = search.programs[pid]
        context = contexts[pid]
        for trial in search.studies[pid].get_trials(deepcopy=False):
            if trial.state.name != "RUNNING" or (pid,trial.number) in known:
                continue
            value = trial.user_attrs.get("recipe")
            if value is None:
                model = proposal.recipe.model.model_copy(update={"parameters":dict(proposal.recipe.model.parameters)|trial.params})
                value = proposal.recipe.model_copy(update={"model":model}).model_dump(mode="json")
            recipe = PipelineRecipe.model_validate(value)
            from .research_search import RecipeSuggestion
            suggestion = RecipeSuggestion(f"{pid}-{trial.number:06d}",trial.number,recipe,proposal.hypothesis,proposal.changed_axes,pid,proposal.proposal_id)
            signature = core._execution_signature(recipe,context["digest"],context["protocol"],revision,environment)
            payload = suggestion.serializable() | {"signature":signature,"dataset_hash":context["digest"],"dataset_path":str(context["dataset"]),"protocol_parameters":context["protocol"],"code_revision":revision,"environment_hash":environment}
            ledger.reserve_attempt(signature,payload)
            recovered += 1
    return recovered


def _reference_context(path):
    if path is None or not (Path(path)/"ledger.sqlite").is_file():
        return []
    with sqlite3.connect(f"file:{Path(path).resolve()}/ledger.sqlite?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT attempt_id,status,payload_json,result_json FROM attempts WHERE status='completed'").fetchall()
    from .research_telemetry import champions
    terminal = [{"attempt_id":a,"status":s,"payload":json.loads(p),"result":json.loads(r)} for a,s,p,r in rows]
    current = list(champions(terminal)["champions_by_family"].values())
    inherited_path = Path(path)/"reference-champions.json"
    inherited = core._read_json(inherited_path) if inherited_path.is_file() else []
    by_family = {}
    for value in [*(inherited if isinstance(inherited,list) else []),*current]:
        key = (value["comparison_key"],value["recipe"]["model"]["kind"])
        if key not in by_family or value["objective_value"]<by_family[key]["objective_value"]:
            by_family[key] = value
    return list(by_family.values())


def _upload_result_with_tracing_lock(config, dataset, attempt):
    from .research_telemetry import tracking_operation
    from .research_v5 import upload_result
    with tracking_operation():
        return upload_result(config, dataset, attempt)


def run_expansion_campaign(config):
    """Continuous planning/preparation/fit/upload with deterministic resource dispatch."""
    from .research_resources import JobEstimator, observe_cgroup, observe_process_tree, ProgressiveCapacity
    from .research_scheduler import ResourceAdmission, fair_program_order, memory_budget_gib
    from .research_v5 import discovery_evidence
    config.validate()
    if not config.dataset_path or not config.protocol_path:
        raise ValueError("V6 requires immutable dataset and explicit protocol paths")
    directory = Path(config.campaign_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with core.campaign_lock(directory):
        campaign_started = time.monotonic()
        dataset = core._resolve_dataset(config, directory)
        digest, protocol = core._hash_file(dataset), core._protocol_parameters(config)
        revision, environment = core._code_revision(), core._environment_hash()
        core._validate_campaign_identity(directory, dataset_hash=digest, protocol_parameters=protocol,
            code_revision=revision, environment_hash=environment, research_policy=config.research_policy)
        ledger, search = ResearchLedger(directory/"ledger.sqlite"), ProgramSearchController(directory)
        decisions = DecisionStore(directory/"decisions.sqlite")
        hypotheses = HypothesisMemory(directory/"hypotheses.sqlite")
        estimator = JobEstimator(directory/"job-estimates.json")
        preparation_estimator = JobEstimator(directory/"preparation-estimates.json")
        paper_estimator = JobEstimator(directory/"paper-estimates.json")
        profile = core.DatasetFeatureProfile(dataset, protocol)
        registry = _dataset_registry(config)
        contexts = {}
        base_context = {"dataset":dataset,"digest":digest,"protocol":protocol,"rows":None,"manifest":{}}
        from .research_executor import _load_dataset
        inspection = _load_dataset(dataset)
        rows = len(inspection)
        base_context["rows"] = rows
        del inspection
        references = _reference_context(config.reference_campaign_dir)
        reference_bests = {}
        for value in references:
            key = value["comparison_key"]
            if key not in reference_bests or value["objective_value"]<reference_bests[key]["objective_value"]:
                reference_bests[key] = value
        core._write_json_atomic(directory/"reference-champions.json", references)
        retired_path = directory/"retired-programs.json"
        retired = set(core._read_json(retired_path) if retired_path.is_file() else [])
        for decision in decisions.pending():
            evidence_path = directory/"evidence"/f"{decision.decision_id}.json"
            if not evidence_path.is_file():
                raise ValueError("Recorded decision has no durable evidence")
            old_evidence = core._read_json(evidence_path)
            apply_decision(decision, decisions, search, retired, config, old_evidence, profile)
        history = RuntimeHistory(directory/"runtime-history.json")
        history_snapshot = history.startup(interrupted_attempts=[row["attempt_id"] for row in ledger.reserved_attempts() if row["status"]=="running"],
                                          journal_entries=read_own_unit_journal(history.unit))
        for row in ledger.reserved_attempts():
            workload_path = directory/"trials"/row["attempt_id"]/"workload.json"
            if workload_path.is_file():
                from .research_resources import JobWorkload
                estimator.record_censored_history(JobWorkload.model_validate(core._read_json(workload_path)),history_snapshot)
            if row["status"] in {"running","reserved"}:
                cleanup_attempt_group(directory/"trials"/row["attempt_id"]/"runtime.json")
        ledger.recover_running()
        startup_tell_error = None
        try:
            core._reconcile_tells(ledger,search)
        except Exception as exc:
            startup_tell_error = str(exc)
        pending = list(ledger.pending_attempts())
        inflight, preparing, ready, exhausted = {}, {}, {}, set()
        fold_state = _FoldDependencies(directory)
        fold_preparing,fold_ready,fold_errors = {},fold_state.ready,fold_state.errors
        def retire_fold_program(pid, dependency):
            if pid not in retired:
                retired.add(pid)
                core._write_json_atomic(retired_path,sorted(retired))
                hypotheses.record("fold-preparation:"+pid,pid,"preparation_rejected",fold_errors[dependency])
        building_datasets = {}
        dataset_errors = {}
        planning = uploading = None
        tracing = None
        paper_future = None
        paper_retry_after = {}
        paper_errors = {}
        paper_trace_failures = {}
        paper_drain_started = None
        upload_failures = {}
        trace_failures = 0
        tell_failures = 0
        next_upload_retry = next_trace_retry = 0.
        tracking_errors = [startup_tell_error] if startup_tell_error else []
        stop_mode = None
        last_error = None
        last_plan = last_snapshot = last_history = last_trace = last_journal = 0.
        last_summary_signature = None
        terminal_since_plan = 0
        decision_number = 1 + max([int(p.stem[1:]) for p in (directory/"evidence").glob("D*.json")]+[0])
        snapshot_number = next_trace_number(directory,"execution")
        jobs = config.max_concurrent_trials if isinstance(config.max_concurrent_trials, int) else config.cpu_thread_budget
        cpu = min(config.cpu_thread_budget, max(1, (psutil.cpu_count() or 1)-config.host_reserve_cpu_threads))
        budget_gib = min(memory_budget_gib(config.memory_budget_gb_decimal), config.ram_budget_gib)
        residency = observe_process_tree().get("private_bytes",0)/1024**3
        ramp = ProgressiveCapacity(tuple(sorted(set([min(jobs,2),*(n for n in (4,8,12,16) if n<jobs),jobs]))),emergency_gib=config.host_reserve_ram_gib,
                                   representative_baseline=sum(not sample.failed and not sample.censored for sample in estimator.samples))
        resources = ResourceAdmission(jobs+config.max_active_preparations+1, cpu, budget_gib, resident_gib=residency,
                                      emergency_gib=config.host_reserve_ram_gib,
                                      max_preparations=config.max_active_preparations,max_fits=ramp.cap,max_expensive_fits=1)
        core._write_json_atomic(directory/"campaign.json", config.serializable())
        from .research_telemetry import initialize_required_tracing
        initialize_required_tracing(config)
        with BoundedFitExecutor(max_workers=jobs,max_tasks_per_child=config.worker_max_tasks) as fits, \
             ThreadPoolExecutor(max_workers=1) as planner, \
             BoundedFitExecutor(max_workers=config.max_active_preparations,max_tasks_per_child=1) as builders, \
             ThreadPoolExecutor(max_workers=1) as uploader:
            while True:
                now = time.monotonic()
                drain_expired = bool(config.timeout_minutes and now-campaign_started>=config.timeout_minutes*60)
                if (directory/"STOP").exists():
                    stop_mode = "stopped"
                if config.timeout_minutes and now-campaign_started >= config.timeout_minutes*60:
                    stop_mode = "timeout"
                if config.max_trials is not None and core._success_count(ledger) >= config.max_trials:
                    stop_mode = "complete"
                if core._consecutive_failure_count(ledger) >= config.max_consecutive_failed_trials:
                    stop_mode = "blocked_failures"
                if stop_mode and paper_drain_started is None:
                    paper_drain_started = now
                paper_drain_expired = drain_expired or (paper_drain_started is not None and now-paper_drain_started>=30)
                host = psutil.virtual_memory()
                cgroup = observe_cgroup()
                active_runtime = active_attempt_summary(directory,ledger.reserved_attempts())
                if now-last_history>=10:
                    measurements = {}
                    for item in active_runtime:
                        if item["status"]=="running" and item["elapsed_seconds"] is not None and item["runtime"].get("private_peak_bytes",0)>0:
                            work = next((value[2] for value in inflight.values() if value[0]==item["attempt_id"]),None)
                            if work is not None:
                                measurements[item["attempt_id"]] = {"workload_fingerprint":work.fingerprint(),
                                    "private_peak_bytes":item["runtime"]["private_peak_bytes"],"wall_seconds":item["elapsed_seconds"],
                                    "failed":False,"censored":False,"memory_metric":"uss"}
                    journal = read_own_unit_journal(history.unit) if now-last_journal>=60 else []
                    if now-last_journal>=60:
                        last_journal = now
                    history_snapshot = history.observe(cgroup=cgroup,
                        running_attempts=[value[0] for value in inflight.values()],attempt_measurements=measurements,journal_entries=journal)
                    last_history = now
                observed = observe_process_tree().get("private_bytes",0)/1024**3
                active_private = sum(item["runtime"].get("private_current_bytes",0) for item in active_runtime if item["status"]=="running")/1024**3
                resources.resident_gib = max(residency,observed-active_private)
                limit = cgroup.get("memory.max")
                cgroup_headroom = (limit-cgroup.get("memory.current",0))/1024**3 if isinstance(limit,int) else None
                resources.set_headroom(memory_available_gib=max(0,host.available/1024**3-config.host_reserve_ram_gib),
                    cgroup_available_gib=cgroup_headroom,
                    disk_available_bytes=psutil.disk_usage(directory).free)
                representative = [sample for sample in estimator.samples if not sample.failed and not sample.censored]
                per_fit = max((s.private_peak_bytes/1024**3 for s in representative),default=2.)
                remaining = budget_gib-resources.snapshot()["reserved_ram_gib"]-resources.resident_gib
                ramp_state = ramp.observe(headroom_gib=min(remaining,host.available/1024**3,cgroup_headroom if cgroup_headroom is not None else remaining),private_per_fit_gib=per_fit,representative_folds=len(representative),now=now)
                resources.max_fits = ramp.cap
                resources.max_expensive_fits = max(1,ramp.cap-1)

                if uploading and uploading[0].done():
                    future, attempt = uploading
                    try:
                        link = future.result()
                        if not link:
                            raise ValueError("MLflow upload produced no run/model linkage")
                        ledger.mark_uploaded(attempt, json.dumps(link,sort_keys=True))
                    except Exception as exc:
                        tracking_errors.append(str(exc))
                        upload_failures[attempt] = upload_failures.get(attempt,0)+1
                        next_upload_retry = now+(1 if stop_mode else 30)
                    uploading = None
                if tracing and tracing.done():
                    try:
                        report = tracing.result()
                        tracking_errors.extend(str(error) for error in report.get("errors",[]))
                        trace_failures = trace_failures+1 if report.get("errors") else 0
                    except Exception as exc:
                        tracking_errors.append(str(exc))
                        trace_failures += 1
                    tracing = None
                    next_trace_retry = now+(1 if stop_mode else 30)
                if config.mlflow_tracking_uri and uploading is None and tracing is None and now>=next_upload_retry and not drain_expired:
                    outbox = ledger.pending_outbox()
                    feasible_outbox = [row for row in outbox if upload_failures.get(row["attempt_id"],0)<3]
                    if feasible_outbox:
                        uploading = (uploader.submit(_upload_result_with_tracing_lock,config,dataset,feasible_outbox[0]),feasible_outbox[0]["attempt_id"])
                if config.mlflow_tracking_uri and uploading is None and tracing is None and now>=next_trace_retry and trace_failures<3 and not drain_expired and _pending_trace_count(directory):
                    from .research_telemetry import drain_trace_outbox
                    tracing = uploader.submit(drain_trace_outbox,directory,config,limit=20)

                for future in list(building_datasets):
                    if not future.done():
                        continue
                    request_id,workload = building_datasets.pop(future)
                    cleanup_attempt_group(directory/"dataset-runtime"/request_id/"runtime.json")
                    resources.release("dataset:"+request_id)
                    try:
                        manifest = future.result()
                        actual = manifest.pop("worker_resources",None)
                        if actual and actual.get("supported"):
                            preparation_estimator.record(workload,actual)
                        dataset_errors.pop(request_id,None)
                        hypotheses.record("dataset:"+manifest["dataset_id"],request_id,"dataset_verified",{"dataset_id":manifest["dataset_id"],"validation":manifest["validation"]})
                    except Exception as exc:
                        dataset_errors[request_id] = str(exc)
                        hypotheses.record("dataset-rejected:"+request_id,request_id,"dataset_rejected",{"error":str(exc)})

                for future in list(preparing):
                    if not future.done():
                        continue
                    pid = preparing.pop(future)
                    cleanup_attempt_group(directory/"preparation-runtime"/pid/"runtime.json")
                    resources.release(f"prepare:{pid}")
                    try:
                        ready[pid] = future.result()
                        actual = ready[pid].get("worker_resources",{})
                        if actual.get("supported"):
                            context = contexts[pid]
                            preparation_estimator.record(_workload(search.programs[pid].recipe,context["rows"],revision,environment,stage="feature_generation"),actual)
                        core._write_json_atomic(directory/"prepared-programs"/f"{pid}.json", ready[pid])
                    except Exception as exc:
                        retired.add(pid)
                        core._write_json_atomic(retired_path,sorted(retired))
                        hypotheses.record(f"preparation:{pid}",pid,"preparation_rejected",{"error":str(exc)})

                for future in list(fold_preparing):
                    if not future.done():
                        continue
                    dependency,pid,workload,request = fold_preparing.pop(future)
                    cleanup_attempt_group(request.output_dir/"runtime.json")
                    resources.release("fold:"+dependency)
                    try:
                        outcome = future.result()
                        fold_state.publish(dependency,outcome["artifacts"])
                    except Exception as exc:
                        fold_state.failed(dependency,pid,exc)
                        if fold_state.blocked(dependency):
                            for candidate in contexts:
                                context = contexts[candidate]
                                candidate_dependency,_ = _dependency_request(search.programs[candidate].recipe,context,directory,revision,environment)
                                if candidate_dependency==dependency:
                                    retire_fold_program(candidate,dependency)
                    else:
                        actual = outcome.get("worker_resources",{})
                        if actual.get("supported"):
                            try:
                                preparation_estimator.record(workload,actual)
                            except Exception as exc:
                                tracking_errors.append(f"Fold preparation measurement {dependency}: {exc}")

                for future in list(inflight):
                    if not future.done():
                        continue
                    attempt,payload,workload = inflight.pop(future)
                    cleanup_attempt_group(directory/"trials"/attempt/"runtime.json")
                    resources.release(attempt)
                    try:
                        result = future.result().serializable()
                    except Exception as exc:
                        runtime_path = directory/"trials"/attempt/"runtime.json"
                        if isinstance(exc,TimeoutError):
                            cleanup_attempt_group(runtime_path)
                        runtime = read_runtime(runtime_path)
                        if runtime:
                            core._write_json_atomic(runtime_path,runtime | {"status":"timed_out" if isinstance(exc,TimeoutError) else "failed"})
                        result = {"schema_version":1,"attempt_id":attempt,"status":"failed",
                            "proposal_id":payload["proposal_id"],"trial_number":payload["trial_number"],
                            "recipe_hash":payload["recipe_hash"],"target_kind":payload["recipe"]["target"]["kind"],
                            "objective_name":"worker_failure","objective_value":None,"metrics":{},
                            "artifacts":{},"lineage":{"dataset_hash":payload["dataset_hash"]},
                            "duration_seconds":max(0,time.time()-datetime.fromisoformat(runtime["started_at"]).timestamp()) if runtime.get("started_at") else 0,
                            "error":f"{type(exc).__name__}: {exc}","failure_class":"execution_timeout" if isinstance(exc,TimeoutError) else "worker_failure"}
                        if runtime.get("private_peak_bytes"):
                            result["metrics"]["worker_resources"] = {"supported":True,"private_peak_bytes":runtime["private_peak_bytes"],
                                "wall_seconds":result["duration_seconds"],"failed":True,"censored":True,"memory_metric":"uss"}
                        last_error = str(exc)
                    result.update(core._portfolio_identity(PipelineRecipe.model_validate(payload["recipe"]),"expansion_v6"))
                    result["target_parameters"] = payload["recipe"]["target"]["parameters"]
                    if result["metrics"].get("worker_resources",{}).get("supported"):
                        actual = dict(result["metrics"]["worker_resources"],failed=result["status"]!="completed")
                        estimator.record(workload,actual)
                    ledger.complete_attempt(attempt,result,status=result["status"])
                    core._append_jsonl(directory/"trials.jsonl", result)
                    hypotheses.record(f"outcome:{attempt}",payload["program_id"],"evaluated",
                        {"attempt_id":attempt,"objective_value":result["objective_value"],"status":result["status"],"error":result.get("error")})
                    try:
                        core._reconcile_tells(ledger,search)
                    except Exception as exc:
                        tracking_errors.append(str(exc))
                    terminal_since_plan += 1

                if paper_future and paper_future[0].done():
                    future,action,workload = paper_future
                    cleanup_attempt_group(directory/"paper-runtime"/action["action_id"]/f"attempt-{_paper_attempt_count(decisions,action)}.json")
                    resources.release("paper:"+action["action_id"])
                    try:
                        outcome = future.result()
                        actual = outcome.get("worker_resources",{})
                        if actual.get("supported"):
                            paper_estimator.record(workload,dict(actual,failed=not outcome["success"]))
                        if not outcome["success"]:
                            raise ValueError(outcome["error"])
                        _publish_paper_report(decisions,directory,action,outcome["report"],actual)
                        paper_errors.pop(action["action_id"],None)
                    except Exception as exc:
                        paper_errors[action["action_id"]] = str(exc)
                        paper_retry_after[action["action_id"]] = now+(1 if stop_mode else 30)
                        if _paper_attempt_count(decisions,action)>=3:
                            decisions.receipt(action["decision_id"],"betting-complete:"+action["action_id"],
                                              {"status":"failed","error":str(exc)})
                    paper_future = None

                # Already authorized paper actions drain on STOP, independently of trial lanes.
                pending_paper = decisions.betting_actions(pending_only=True)
                for action in pending_paper:
                    if paper_future and action["action_id"]==paper_future[1]["action_id"]:
                        continue
                    try:
                        recovered = _recover_paper_report(decisions,directory,action)
                    except Exception as exc:
                        decisions.receipt(action["decision_id"],"betting-complete:"+action["action_id"],
                                          {"status":"failed","error":str(exc)})
                        recovered = True
                    if recovered or paper_drain_expired or paper_future or now<paper_retry_after.get(action["action_id"],0):
                        continue
                    attempts = _paper_attempt_count(decisions,action)
                    if attempts>=3:
                        decisions.receipt(action["decision_id"],"betting-complete:"+action["action_id"],
                                          {"status":"failed","error":"Paper action exhausted its three durable attempts"})
                        continue
                    workload = _paper_workload(action,revision,environment)
                    estimate = _paper_estimate(paper_estimator,workload,action)
                    if not resources.admits(estimate):
                        continue
                    resources.reserve("paper:"+action["action_id"],estimate)
                    decisions.receipt(action["decision_id"],f"betting-attempt:{action['action_id']}:{attempts+1}",
                                      {"attempt":attempts+1,"estimate":estimate.to_dict()})
                    try:
                        future = fits.submit(_paper_worker,action["request"],directory,ledger.terminal_results(),
                            runtime_path=directory/"paper-runtime"/action["action_id"]/f"attempt-{attempts+1}.json",timeout=1200)
                    except Exception as exc:
                        resources.release("paper:"+action["action_id"])
                        paper_errors[action["action_id"]] = str(exc)
                        paper_retry_after[action["action_id"]] = now+1
                        continue
                    paper_future = (future,action,workload)
                for action in decisions.betting_actions():
                    if action["completion"] is not None:
                        if paper_trace_failures.get(action["action_id"],0)>=3:
                            continue
                        try:
                            _trace_paper_action(decisions,directory,action,config)
                        except Exception as exc:
                            tracking_errors.append(str(exc))
                            identifier = action["action_id"]
                            paper_trace_failures[identifier] = paper_trace_failures.get(identifier,0)+1

                budgets = decisions.authorized_budgets()
                authorized = set(budgets)
                # Already reserved legacy/imported attempts are resumable, but confer no new budget.
                pending = list(ledger.pending_attempts())
                resumed_ids = {row["payload"]["program_id"] for row in pending}
                known = {(row["payload"]["program_id"],row["payload"]["trial_number"]) for row in ledger.reserved_attempts()}
                orphan_ids = {pid for pid in authorized for trial in search.studies[pid].get_trials(deepcopy=False) if trial.state.name == "RUNNING" and (pid,trial.number) not in known}
                capacity = {pid:max(0,min(p.max_trials,budgets[pid])-sum(t.state.name!="PRUNED" for t in search.studies[pid].get_trials(deepcopy=False)))
                            for pid,p in search.programs.items() if pid in authorized and pid not in retired|exhausted}
                preparation_capacity = dict(capacity)
                preparation_capacity.update({pid:max(1,capacity.get(pid,0)) for pid in resumed_ids|orphan_ids})
                programs_waiting = len({pid for pid,remaining in capacity.items() if remaining>0} | resumed_ids | orphan_ids)
                needs_plan = (terminal_since_plan >= config.replan_every_terminal_trials or
                    sum(capacity.values()) <= config.queue_low_watermark or
                    _needs_lane_refill(capacity,search.programs,len(ledger.reserved_attempts())) or
                    now-last_plan >= config.planning_checkpoint_seconds)
                if not stop_mode and planning is None and needs_plan and now-last_plan >= (60 if last_error else 5):
                    spend = _planner_spend(directory)
                    if config.planner_mode == "openrouter" and spend["spend_unknown"]:
                        last_error = "Planner call cost is unknown; further paid calls are frozen"
                        last_plan = now
                    elif config.planner_mode == "openrouter" and _unsent_failure_streak(directory)>=3:
                        last_error = "Planner pre-dispatch retry circuit open after three unsent failures"
                        last_plan = now
                    elif config.max_total_cost_usd is not None and spend["known_spend_usd"] >= config.max_total_cost_usd:
                        last_error = "Planner prior-spend admission threshold reached; not a provider-enforced billing cap"
                        last_plan = now
                    else:
                        terminal = ledger.terminal_results()
                        evidence = evidence_snapshot(terminal,decision_id=f"D{decision_number:06d}",
                            feature_profile=profile.summary, feature_evidence=discovery_evidence(terminal),
                            references=references, hypothesis_memory=hypotheses.retrieve(limit=30),
                            paper_research=_paper_evidence(decisions,directory,terminal),
                            remaining_program_capacity=capacity, resources=resources.snapshot(),
                            active_trials=active_runtime,invocation_history=history_snapshot["invocations"],
                            oom_history=history_snapshot["oom_facts"],
                            preparation_dependencies={"building":[item[0] for item in fold_preparing.values()],"errors":dict(fold_errors)},
                            pending_program_limit=config.max_pending_programs,
                            completed_trial_index=[{"attempt_id":r["attempt_id"]} for r in terminal if r["status"]=="completed"],
                            available_new_program_slots=max(0,config.max_pending_programs-programs_waiting),
                            next_required_lane=_lane(len(ledger.reserved_attempts())),
                            dataset_registry={"states":[registry.get(value["request_id"]) for value in decisions.dataset_actions()],"building":list(item[0] for item in building_datasets.values()),"errors":dict(dataset_errors)},
                            capabilities=_capabilities(config,dataset,dataset_digest=digest,registry=registry))
                        core._write_json_atomic(directory/"evidence"/f"{evidence['decision_id']}.json",evidence)
                        planning = (planner.submit(plan_decision,evidence,config),evidence)
                        decision_number += 1
                        last_plan = now
                        terminal_since_plan = 0
                if planning and planning[0].done():
                    future,evidence = planning
                    payload = {"decision_id":evidence["decision_id"],"evidence_id":evidence["evidence_id"],
                               "planner_model":config.model if config.planner_mode in {"openrouter", "external"} else None}
                    response = None
                    try:
                        response = future.result()
                        core._write_json_atomic(directory/"planner-responses"/f"{evidence['decision_id']}.json",{"response":response,"evidence_id":evidence["evidence_id"]})
                        decision = PlannerDecision.model_validate(response["decision"])
                        if programs_waiting+len(decision.programs)-len(decision.retire_program_ids) > config.max_pending_programs:
                            raise ValueError("Pending research-program backlog ceiling exceeded")
                        accepted = apply_decision(decision,decisions,search,retired,config,evidence,profile)
                        payload.update(decision.model_dump(mode="json"),program_ids=accepted,
                            allocated_trials=sum(p.max_trials for p in decision.programs)+sum(decision.extensions.values()),
                            planner_usage=response.get("usage",{}),planner_status="accepted",
                            planning_wall_seconds=response.get("planning_wall_seconds"))
                        hypotheses.record(f"decision:{decision.decision_id}",decision.decision_id,"planned",payload)
                        last_error = None
                    except Exception as exc:
                        last_error = str(exc)
                        from .openrouter_orchestrator import normalize_openrouter_usage, _sum_planner_usage
                        calls = [normalize_openrouter_usage(core._read_json(p)["response"])
                                 for p in (directory/"planner-calls").glob(f"{evidence['decision_id']}-*.json")]
                        payload.update(planner_status="failed",error=last_error,planner_usage=response.get("usage",{}) if isinstance(response,dict) and response.get("usage") else _sum_planner_usage(calls) if calls else {})
                    from .openrouter_orchestrator import normalize_openrouter_usage, _sum_planner_usage
                    physical_calls = [normalize_openrouter_usage(core._read_json(path)["response"])
                        for path in (directory/"planner-calls").glob(f"{evidence['decision_id']}-*.json")]
                    if physical_calls:
                        payload["planner_usage"] = _sum_planner_usage(physical_calls)
                    from .openrouter_transport import classify_transport_attempts
                    transport_attempts = [core._read_json(path) for path in sorted((directory/"planner-transport").glob(f"{evidence['decision_id']}-*.json"))]
                    dispatch = classify_transport_attempts([item.get("events") for item in transport_attempts])
                    payload.update(dispatch_status=dispatch,transport_attempts=transport_attempts,
                        transport_phase=transport_attempts[-1]["events"][-1]["phase"] if transport_attempts and transport_attempts[-1].get("events") else None)
                    if not physical_calls and dispatch["classification"]=="not_dispatched":
                        payload["planner_usage"] = {"cost_status":"not_dispatched"}
                    core._write_json_atomic(directory/"decisions"/f"{evidence['decision_id']}.json",payload)
                    try:
                        decision_snapshot = evidence_snapshot(ledger.terminal_results()) | payload
                        decision_snapshot.update(code_revision=revision,planner_spend=_planner_spend(directory),
                            campaign_id=directory.name,resources=resources.snapshot(),capacity_ramp=ramp_state,max_fits=jobs,
                            invocation_id=history_snapshot["current_invocation_id"],invocation_history=history_snapshot["invocations"],
                            oom_history=history_snapshot["oom_facts"],restart_count=history_snapshot["restart_count"],
                            prior_champions_by_contract=reference_bests,active_trials=active_attempt_summary(directory,ledger.reserved_attempts()))
                        log_snapshot(directory,decision_snapshot,config,
                            role="decision",number=int(evidence["decision_id"][1:]))
                    except Exception as exc:
                        tracking_errors.append(str(exc))
                    planning = None

                # Preparation capacity is independent of proposal and fit budgets.
                if not stop_mode:
                    budgets = decisions.authorized_budgets()
                    authorized = set(budgets)
                    for pid in authorized-retired-exhausted:
                        preparation_capacity[pid] = max(int(pid in resumed_ids|orphan_ids),min(search.programs[pid].max_trials,budgets[pid])-sum(t.state.name!="PRUNED" for t in search.studies[pid].get_trials(deepcopy=False)))
                    for value in decisions.dataset_actions():
                        request_id = value["request_id"]
                        state = registry.submit(value)
                        if state["status"] == "verified" or request_id in dataset_errors or request_id in {item[0] for item in building_datasets.values()}:
                            continue
                        source,raw = _dataset_build_inputs(config)
                        from .research_resources import JobWorkload
                        workload = JobWorkload(stage="feature_generation",family="official_dataset",rows=rows,generated_features=155,selected_features=155,implementation_revision=revision,dependency_versions={"environment":environment},search_settings={"request":state["request_fingerprint"]})
                        estimate = preparation_estimator.estimate(workload)
                        if _preparation_slot_available(config.max_active_preparations,search.programs,preparing.values(),len(building_datasets)) and resources.admits(estimate):
                            resources.reserve("dataset:"+request_id,estimate)
                            try:
                                future = builders.submit(_build_dataset,str(registry.root),request_id,source,raw,str(registry.feature_registry),
                                    runtime_path=directory/"dataset-runtime"/request_id/"runtime.json",timeout=3600)
                            except Exception:
                                resources.release("dataset:"+request_id)
                                raise
                            building_datasets[future] = (request_id,workload)
                    for pid in _preparation_program_order(preparation_capacity,search.programs,ready,preparing.values(),ledger.reserved_attempts()):
                        if pid in ready or pid in preparing.values() or not preparation_capacity[pid]:
                            continue
                        if pid not in contexts:
                            recipe = search.programs[pid].recipe
                            try:
                                context = _program_context(recipe,config,registry) if recipe.dataset_ref else dict(base_context)
                                contexts[pid] = _preflight_recipe(recipe,context,config)
                            except Exception as exc:
                                if recipe.dataset_ref:
                                    dataset_errors["program:"+pid] = str(exc)
                                    try:
                                        state = registry.get(recipe.dataset_ref) if not recipe.dataset_ref.startswith("dataset-") else {"status":"verified"}
                                        if state["status"] in {"verified","rejected"}:
                                            retired.add(pid)
                                            core._write_json_atomic(retired_path,sorted(retired))
                                    except (OSError,ValueError):
                                        pass
                                    continue
                                raise
                        context = contexts[pid]
                        prepared_path = directory/"prepared-programs"/f"{pid}.json"
                        persisted = core._read_json(prepared_path) if prepared_path.is_file() else None
                        if persisted and persisted.get("dataset_digest",context["digest"]) == context["digest"] and persisted.get("status") != "not_started":
                            try:
                                if persisted.get("artifact_descriptor"):
                                    from .feature_program import DiscoveryMatrixArtifact
                                    DiscoveryMatrixArtifact(**persisted["artifact_descriptor"]).manifest()
                                ready[pid] = persisted
                                continue
                            except (OSError, ValueError, TypeError):
                                # A durable allocation survives eviction of its preparation artifact.
                                pass
                        workload = _workload(search.programs[pid].recipe,context["rows"],revision,environment,stage="preparation")
                        estimate = preparation_estimator.estimate(workload)
                        if _preparation_slot_available(config.max_active_preparations,search.programs,preparing.values(),len(building_datasets),search.programs[pid].recipe) and resources.admits(estimate):
                            resources.reserve(f"prepare:{pid}",estimate)
                            try:
                                future = builders.submit(_prepare_program,context["dataset"],search.programs[pid].recipe,context["digest"],directory,
                                    runtime_path=directory/"preparation-runtime"/pid/"runtime.json",timeout=max(1200,search.programs[pid].max_wall_seconds))
                            except Exception:
                                resources.release(f"prepare:{pid}")
                                raise
                            preparing[future] = pid

                    fold_candidates = []
                    for row in pending:
                        pid = row["payload"]["program_id"]
                        if pid in ready:
                            dependency,request = _pending_fold_request(row["payload"],contexts[pid],directory)
                            fold_candidates.append((pid,dependency,request))
                    for pid in ready:
                        if pid not in authorized or pid in retired or preparation_capacity.get(pid,0)<=0:
                            continue
                        context = contexts[pid]
                        dependency,request = _dependency_request(search.programs[pid].recipe,context,directory,revision,environment)
                        fold_candidates.append((pid,dependency,request))
                    for pid,dependency,request in fold_candidates:
                        if dependency is None:
                            continue
                        if fold_state.blocked(dependency):
                            retire_fold_program(pid,dependency)
                            continue
                        if fold_state.is_ready(dependency) or dependency in {item[0] for item in fold_preparing.values()}:
                            continue
                        if not fold_state.can_prepare(dependency):
                            continue
                        context = contexts[pid]
                        workload = _workload(request.recipe,context["rows"],revision,environment,stage="preparation")
                        estimate = preparation_estimator.estimate(workload)
                        if resources.admits(estimate):
                            resources.reserve("fold:"+dependency,estimate)
                            try:
                                future = builders.submit(_prepare_fold_dependency,request,workload,
                                    runtime_path=request.output_dir/"runtime.json",
                                    timeout=max(1200,search.programs[pid].max_wall_seconds))
                            except Exception:
                                resources.release("fold:"+dependency)
                                raise
                            fold_preparing[future] = (dependency,pid,workload,request)

                    _recover_orphan_asks(search,ledger,contexts,authorized,revision,environment)
                    pending = list(ledger.pending_attempts())

                # Resume durable allocations before asking Optuna for replacement work.
                for row in pending:
                    if stop_mode or len(inflight)>=jobs:
                        break
                    payload = row["payload"]
                    pid = payload["program_id"]
                    dependency,_ = _pending_fold_request(payload,contexts.get(pid,base_context),directory)
                    if fold_state.blocked(dependency):
                        retire_fold_program(pid,dependency)
                        if _fail_fold_allocation(row,dependency,fold_errors[dependency],ledger,hypotheses,directory):
                            terminal_since_plan += 1
                            try:
                                core._reconcile_tells(ledger,search)
                            except Exception as exc:
                                tracking_errors.append(str(exc))
                        continue
                    if pid not in ready:
                        continue
                    if config.max_trials is not None and core._success_count(ledger)+len(inflight)>=config.max_trials:
                        break
                    recipe = PipelineRecipe.model_validate(payload["recipe"])
                    context = contexts[pid]
                    if not fold_state.is_ready(dependency):
                        continue
                    bound_dataset = Path(payload.get("dataset_path",context["dataset"]))
                    if core._hash_file(bound_dataset) != payload["dataset_hash"] or payload["code_revision"] != revision or payload["environment_hash"] != environment:
                        raise ValueError("Reserved attempt immutable execution identity changed")
                    manifest = ready[pid]
                    shared = {manifest["matrix_id"]:manifest["shared_bytes"]} if manifest.get("shared_bytes") else {}
                    pending_fits = len(inflight) + len(ledger.pending_attempts()) + sum(value for program,value in capacity.items() if program in ready)
                    threads = _native_threads(recipe,context["rows"],cpu,pending_trials=pending_fits)
                    workload = _workload(recipe,context["rows"],revision,environment,shared=shared,native_threads=threads)
                    estimate = estimator.estimate(workload)
                    if not resources.admits(estimate):
                        continue
                    attempt = row["attempt_id"]
                    resources.reserve(attempt,estimate)
                    request = RecipeExecutionRequest(attempt,payload["proposal_id"],payload["trial_number"],recipe,bound_dataset,directory/"trials"/attempt,payload["protocol_parameters"],payload["dataset_hash"],payload["code_revision"],payload["environment_hash"],V6_PORTFOLIO_VERSION)
                    core._write_json_atomic(request.output_dir/"workload.json",workload.model_dump(mode="json"))
                    try:
                        future = fits.submit(_worker,request,workload,estimate.cpu_threads,
                            runtime_path=request.output_dir/"runtime.json",timeout=search.programs[pid].max_wall_seconds)
                    except Exception:
                        resources.release(attempt)
                        raise
                    ledger.mark_running(attempt)
                    inflight[future] = (attempt,payload,workload)
                    core._append_jsonl(directory/"queue.jsonl",{"event":"resumed","attempt_id":attempt,
                        "program_id":pid,"time":utc_now(),"estimate":estimate.to_dict(),"resources":resources.snapshot()})

                # Each completion immediately releases resources and permits cross-program backfill.
                while not stop_mode and len(inflight)<jobs and ramp.cap:
                    if config.max_trials is not None and core._success_count(ledger)+len(inflight)+len(ledger.pending_attempts())>=config.max_trials:
                        break
                    lane = _lane(len(ledger.reserved_attempts()))
                    allowed = [pid for pid in capacity if capacity[pid]>0 and pid in ready and fold_state.is_ready(_dependency_request(search.programs[pid].recipe,contexts[pid],directory,revision,environment)[0]) and pid not in {row["payload"]["program_id"] for row in ledger.pending_attempts()} and
                               core._portfolio_identity(search.programs[pid].recipe,"expansion_v6")["lane"]==lane]
                    order = fair_program_order(allowed,ledger.reserved_attempts())
                    estimates = {}
                    workloads = {}
                    for pid in order:
                        manifest = ready[pid]
                        context = contexts[pid]
                        shared = {manifest["matrix_id"]:manifest.get("shared_bytes",0)} if manifest.get("shared_bytes") else {}
                        pending_fits = len(inflight) + len(ledger.pending_attempts()) + sum(value for program,value in capacity.items() if program in ready)
                        threads = _native_threads(search.programs[pid].recipe,context["rows"],cpu,pending_trials=pending_fits)
                        workload = _workload(search.programs[pid].recipe,context["rows"],revision,environment,shared=shared,native_threads=threads)
                        workloads[pid] = workload
                        estimates[pid] = estimator.estimate(workload)
                    candidate = resources.peek_feasible(estimates)
                    if candidate is None:
                        break
                    pid,estimate = candidate
                    suggestions = search.ask(1,allowed_program_ids=[pid])
                    if not suggestions:
                        exhausted.add(pid)
                        capacity[pid] = 0
                        continue
                    suggestion = suggestions[0]
                    context = contexts[pid]
                    signature = core._execution_signature(suggestion.recipe,context["digest"],context["protocol"],revision,environment)
                    payload = suggestion.serializable() | {"signature":signature,"dataset_hash":context["digest"],
                        "protocol_parameters":context["protocol"],"code_revision":revision,"environment_hash":environment,"dataset_path":str(context["dataset"])}
                    record = ledger.reserve_attempt(signature,payload)
                    if record.status in {"completed","failed","pruned"}:
                        capacity[pid] -= 1
                        continue
                    attempt = record.attempt_id
                    actual_workload = _workload(suggestion.recipe,context["rows"],revision,environment,shared=workloads[pid].shared_artifacts,native_threads=workloads[pid].native_threads)
                    estimate = estimator.estimate(actual_workload)
                    workloads[pid] = actual_workload
                    if not resources.admits(estimate):
                        capacity[pid] -= 1
                        continue
                    resources.reserve(attempt,estimate)
                    request = RecipeExecutionRequest(attempt,suggestion.proposal_id,suggestion.trial_number,suggestion.recipe,
                        context["dataset"],directory/"trials"/attempt,context["protocol"],context["digest"],revision,environment,V6_PORTFOLIO_VERSION)
                    core._write_json_atomic(request.output_dir/"workload.json",workloads[pid].model_dump(mode="json"))
                    try:
                        future = fits.submit(_worker,request,workloads[pid],estimate.cpu_threads,
                            runtime_path=request.output_dir/"runtime.json",timeout=search.programs[pid].max_wall_seconds)
                    except Exception:
                        resources.release(attempt)
                        raise
                    ledger.mark_running(attempt)
                    inflight[future] = (attempt,payload,workloads[pid])
                    core._append_jsonl(directory/"queue.jsonl",{"event":"dispatched","attempt_id":attempt,
                        "program_id":pid,"time":utc_now(),"estimate":estimate.to_dict(),"resources":resources.snapshot()})
                    capacity[pid] -= 1

                counts = ledger.snapshot()
                spend = _planner_spend(directory)
                status = {"campaign_id":directory.name,"status":"draining" if stop_mode else "training" if inflight else "preparing" if preparing or fold_preparing else "planning" if planning else "awaiting_decision",
                    "updated_at":utc_now(),"decision":decision_number-1,"ledger":counts,
                    "resources":resources.snapshot(),"preparing_programs":list(preparing.values()),
                    "capacity_ramp":ramp_state,
                    "active_trials":active_attempt_summary(directory,ledger.reserved_attempts()),
                    "invocation_id":history_snapshot["current_invocation_id"],"invocation_history":history_snapshot["invocations"],
                    "oom_history":history_snapshot["oom_facts"],"restart_count":history_snapshot["restart_count"],
                    "code_revision":revision,"max_fits":jobs,
                    "prior_champions_by_contract":reference_bests,
                    "preparation_dependencies":{"building":[item[0] for item in fold_preparing.values()],"ready":list(fold_ready),"errors":dict(fold_errors)},
                    "building_datasets":[item[0] for item in building_datasets.values()],"dataset_errors":dict(dataset_errors),
                    "pending_paper_actions":len(decisions.betting_actions(pending_only=True)),
                    "paper_action_inflight":paper_future is not None,"paper_errors":dict(paper_errors),
                    "ready_programs":list(ready),"pending_programs":programs_waiting,
                    "planner_inflight":planning is not None,"last_planner_error":last_error,
                    "planner_spend":spend,"planner_spend_usd":spend["total_cost_usd"],"planner_spend_unknown":spend["spend_unknown"],"tracking_errors":tracking_errors[-10:],
                    "planner_spend_limit_scope":"Prior reported spend admission threshold; not a provider-enforced billing cap",
                    "pending_trace_delivery":_pending_trace_count(directory),
                    "host_resources":{"cpu_busy_percent":psutil.cpu_percent(),
                        "load_one_minute_percent":100*psutil.getloadavg()[0]/(psutil.cpu_count() or 1),
                        "available_ram_gib":host.available/1024**3,"cgroup":cgroup}}
                core._write_json_atomic(directory/"status.json",status)
                if now-last_snapshot>=30:
                    snapshot = evidence_snapshot(ledger.terminal_results(),counts=counts,
                        resources=status["resources"],pending_programs=programs_waiting,
                        preparing_programs=list(preparing.values()),planner_inflight=planning is not None,
                        latest_decision_id=f"D{decision_number-1:06d}")
                    snapshot.update(status)
                    from .research_summary import summary,summary_markdown
                    current_summary = summary(snapshot)
                    core._write_json_atomic(directory/"summary.json",current_summary)
                    temporary = directory/"summary.md.tmp"
                    temporary.write_text(summary_markdown(current_summary))
                    temporary.replace(directory/"summary.md")
                    signature = content_id({"counts":counts,"error":last_error,"ready":sorted(ready),
                        "running":[(item["attempt_id"],item["status"],item["progress"].get("stage"),item["progress"].get("fold_id")) for item in status["active_trials"]],
                        "cap":ramp.cap,"decision":decision_number,"oom":len(history_snapshot["oom_facts"])})
                    if signature!=last_summary_signature or now-last_trace>=300:
                        try:
                            log_snapshot(directory,snapshot,config,role="execution",number=snapshot_number)
                        except Exception as exc:
                            tracking_errors.append(str(exc))
                        snapshot_number += 1
                        last_summary_signature = signature
                        last_trace = now
                    last_snapshot = now
                if stop_mode and not inflight and not preparing and not fold_preparing and not building_datasets and paper_future is None and planning is None and uploading is None and tracing is None:
                    pending_paper = decisions.betting_actions(pending_only=True)
                    if pending_paper and not paper_drain_expired:
                        time.sleep(1)
                        continue
                    pending_tracking = len(ledger.pending_outbox())
                    pending_tells = len(ledger.pending_tells())
                    if pending_tells and tell_failures<3 and not drain_expired:
                        try:
                            core._reconcile_tells(ledger,search)
                        except Exception as exc:
                            tracking_errors.append(str(exc))
                        pending_tells = len(ledger.pending_tells())
                        tell_failures = tell_failures+1 if pending_tells else 0
                    pending_traces = _pending_trace_count(directory)
                    untraced_paper = [action for action in decisions.betting_actions()
                        if action["completion"] is not None and
                        decisions.action(action["decision_id"],"betting-traced:"+action["action_id"]) is None]
                    tracking_blocked = (drain_expired or tell_failures>=3 or
                        any(paper_trace_failures.get(action["action_id"],0)>=3 for action in untraced_paper) or
                        (config.mlflow_tracking_uri and pending_tracking and all(upload_failures.get(row["attempt_id"],0)>=3 for row in ledger.pending_outbox())) or
                        (pending_traces and (trace_failures>=3 or not config.mlflow_tracking_uri)))
                    if (config.mlflow_tracking_uri and pending_tracking) or pending_tells or pending_traces or untraced_paper:
                        if not tracking_blocked:
                            time.sleep(1)
                            continue
                        finish_mode = "blocked_tracking"
                    else:
                        failed_paper = [item for item in decisions.betting_actions()
                                        if item["completion"] and item["completion"]["status"]=="failed"]
                        finish_mode = "blocked_paper" if pending_paper or failed_paper else stop_mode
                    finished = core._finish_payload(directory,ledger,search,decision_number-1,[],finish_mode,tracking_errors)
                    spend = _planner_spend(directory)
                    finished.update(planner_spend_usd=spend["total_cost_usd"],planner_spend_unknown=spend["spend_unknown"],
                        pending_tracking=pending_tracking,pending_tells=pending_tells,pending_trace_delivery=pending_traces,
                        pending_paper_trace_publication=len(untraced_paper),
                        pending_paper_actions=len(pending_paper),paper_research=_paper_evidence(decisions,directory,ledger.terminal_results()),
                        tracking_enabled=bool(config.mlflow_tracking_uri),requested_stop_mode=stop_mode)
                    core._write_json_atomic(directory/"status.json",status | finished | {"status":finish_mode})
                    from .research_summary import summary, summary_markdown
                    history_snapshot = history.observe(running_attempts=[],journal_entries=[])
                    final_snapshot = evidence_snapshot(ledger.terminal_results(),counts=ledger.snapshot()) | status | finished | {
                        "status":finish_mode,"campaign_id":directory.name,"active_trials":[],
                        "latest_decision_id":f"D{decision_number-1:06d}",
                        "invocation_history":history_snapshot["invocations"]}
                    try:
                        log_snapshot(directory,final_snapshot,config,role="execution",number=snapshot_number)
                        if config.mlflow_tracking_uri and trace_failures<3:
                            from .research_telemetry import drain_trace_outbox
                            delivered = drain_trace_outbox(directory,config,limit=1)
                            tracking_errors.extend(str(error) for error in delivered.get("errors",[]))
                    except Exception as exc:
                        tracking_errors.append(str(exc))
                    final_snapshot["pending_trace_delivery"] = _pending_trace_count(directory)
                    if config.mlflow_tracking_uri and final_snapshot["pending_trace_delivery"]:
                        finished.update(mode="blocked_tracking",pending_trace_delivery=final_snapshot["pending_trace_delivery"])
                        final_snapshot.update(finished,status="blocked_tracking")
                    core._write_json_atomic(directory/"status.json",final_snapshot)
                    final_summary = summary(final_snapshot)
                    core._write_json_atomic(directory/"summary.json",final_summary)
                    temporary = directory/"summary.md.tmp"
                    temporary.write_text(summary_markdown(final_summary))
                    temporary.replace(directory/"summary.md")
                    return finished
                if inflight:
                    wait(inflight,timeout=1,return_when=FIRST_COMPLETED)
                else:
                    time.sleep(1)


def _preparation_slot_available(max_preparations, programs, preparing, dataset_build_count, recipe=None):
    expensive = lambda value: bool(value.feature_discovery or value.feature_definitions)
    if recipe is not None and not expensive(recipe):
        return True
    # Dataset builds and generated-feature recipes share the expensive-work cap.
    active = dataset_build_count + sum(expensive(programs[pid].recipe) for pid in preparing)
    return active < max(1,max_preparations-1)


def _preparation_program_order(capacity, programs, ready, preparing, reservations):
    from .research_scheduler import fair_program_order

    covered_ids = set(ready) | set(preparing)
    lane = lambda pid: core._portfolio_identity(programs[pid].recipe,"expansion_v6")["lane"]
    covered = {lane(pid) for pid in covered_ids if capacity.get(pid,0)>0}
    required = _lane(len(reservations))
    eligible = [pid for pid,remaining in capacity.items() if remaining>0 and pid not in covered_ids]
    fair = fair_program_order(eligible,reservations)
    ranks = {pid:index for index,pid in enumerate(fair)}

    def priority(pid):
        uncovered = lane(pid) not in covered
        if not uncovered:
            return (True,False,False,0,ranks[pid])
        recipe = programs[pid].recipe
        spec = recipe.feature_discovery
        definitions = len(recipe.feature_definitions or ())
        complexity = definitions + (spec.max_definitions * spec.max_depth if spec else 0)
        return (not uncovered, uncovered and lane(pid)!=required,
                bool(spec or definitions), complexity, ranks[pid])

    ordered = []
    # Prospective coverage keeps the next slot from choosing the same cold lane.
    while eligible:
        pid = min(eligible,key=priority)
        ordered.append(pid)
        covered.add(lane(pid))
        eligible.remove(pid)
    return ordered


def _lane(index):
    return "benter" if round(4*(index+1)/5)>round(4*index/5) else "experimental"


def _needs_lane_refill(capacity,programs,attempt_count):
    required = _lane(attempt_count)
    return not any(remaining>0 and core._portfolio_identity(programs[pid].recipe,"expansion_v6")["lane"]==required
                   for pid,remaining in capacity.items())


def _pending_trace_count(directory):
    path = Path(directory)/"trace-outbox.sqlite"
    if not path.is_file():
        return 0
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as conn:
        return conn.execute("SELECT COUNT(*) FROM traces WHERE delivered=0").fetchone()[0]


def _planner_cost(directory):
    return _planner_spend(directory)["total_cost_usd"]


def _planner_spend(directory):
    """Keep reported subtotals, but require coverage before exempting failures."""
    from .openrouter_orchestrator import normalize_openrouter_usage
    from .openrouter_transport import classify_transport_attempt
    directory = Path(directory)
    receipts = {}
    for path in (directory/"planner-calls").glob("D*-*.json"):
        decision_id = path.stem.rsplit("-",1)[0]
        payload = core._read_json(path)
        cost = normalize_openrouter_usage(payload.get("response"))["total_cost_usd"]
        receipts.setdefault(decision_id,{})[path.stem] = cost if cost is not None and math.isfinite(cost) else None
    transport = {}
    for path in (directory/"planner-transport").glob("D*-*.json"):
        decision_id = path.stem.rsplit("-",1)[0]
        transport.setdefault(decision_id,{})[path.stem] = classify_transport_attempt(core._read_json(path).get("events"))["classification"]
    aggregates, failed, claimed_cost, expected_attempts = {},set(),set(),{}
    for folder,key in (("decisions","planner_usage"),("planner-responses","usage")):
        for path in (directory/folder).glob("D*.json"):
            payload = core._read_json(path)
            if folder == "decisions" and payload.get("planner_status") == "failed":
                failed.add(path.stem)
            if folder == "decisions" and isinstance(payload.get("transport_attempts"),list):
                expected_attempts[path.stem] = len(payload["transport_attempts"])
            if folder == "planner-responses":
                payload = payload.get("response",{})
            usage = payload.get(key,{}) if isinstance(payload,dict) else {}
            usage = usage if isinstance(usage,dict) else {}
            cost = usage.get("total_cost_usd")
            fixture = usage.get("cost_status") == "fixture"
            if not fixture and (cost is not None or usage.get("cost_status") == "reported"):
                claimed_cost.add(path.stem)
            if cost is not None and (isinstance(cost,bool) or not isinstance(cost,(float,int)) or not math.isfinite(cost) or cost < 0):
                cost = None
            aggregates.setdefault(path.stem,[]).append((cost,fixture,folder == "planner-responses"))

    def complete_coverage(identifiers):
        numbers = []
        for identifier in identifiers:
            suffix = identifier.rsplit("-",1)[-1]
            if not suffix.isascii() or not suffix.isdecimal():
                return False
            numbers.append(int(suffix))
        # Check gaps without allocating a range from an untrusted numeric suffix.
        return bool(numbers) and min(numbers)>0 and len(set(numbers))==len(numbers) and max(numbers)==len(numbers)

    costs, unsent = {},[]
    for decision_id in sorted(set(receipts)|set(transport)|set(aggregates)):
        calls = receipts.get(decision_id,{})
        attempts = transport.get(decision_id,{})
        records = aggregates.get(decision_id,[])
        if not calls and records and all(fixture for _,fixture,_ in records):
            costs[decision_id] = [0]
            continue
        reported = [cost for cost,fixture,_ in records if cost is not None and not fixture]
        has_response = any(response and not fixture for _,fixture,response in records)
        identifiers = set(calls)|set(attempts)
        covered = complete_coverage(identifiers) and len(identifiers)>=expected_attempts.get(decision_id,0)
        all_unsent = bool(attempts) and all(kind=="not_dispatched" for kind in attempts.values())
        if all_unsent and covered and not calls and not has_response and decision_id not in claimed_cost:
            costs[decision_id] = []
            unsent.append(decision_id)
            continue

        # Aggregates describe the same calls, so never add them to physical
        # receipts or each other. Retain the largest available aggregate subtotal.
        values = list(calls.values()) if calls else [max(reported)] if reported else []
        unknown = not values or any(value is None for value in values)
        if (identifiers or expected_attempts.get(decision_id,0)) and not covered:
            unknown = True
        if any(kind!="not_dispatched" and identifier not in calls for identifier,kind in attempts.items()):
            unknown = True
        if decision_id in failed and not attempts:
            unknown = True
        if all_unsent and (calls or has_response or decision_id in claimed_cost):
            unknown = True
        if any(kind=="not_dispatched" and identifier in calls for identifier,kind in attempts.items()):
            unknown = True
        if not calls and len(set(reported))>1:
            unknown = True
        if unknown and None not in values:
            values.append(None)
        costs[decision_id] = values
    unresolved = sorted(key for key,values in costs.items() if None in values)
    values = [value for row in costs.values() for value in row]
    unknown = any(value is None for value in values)
    known = sum(value for value in values if value is not None)
    return {"total_cost_usd":None if unknown else known,"known_spend_usd":known,"spend_unknown":unknown,
            "unresolved_decision_ids":unresolved,"not_dispatched_decision_ids":sorted(unsent)}


def _unsent_failure_streak(directory):
    unsent = set(_planner_spend(directory)["not_dispatched_decision_ids"])
    count = 0
    for path in reversed(sorted((Path(directory)/"decisions").glob("D*.json"))):
        payload = core._read_json(path)
        if payload.get("planner_status")!="failed" or path.stem not in unsent:
            break
        count += 1
    return count


def _capabilities(config,dataset,*,dataset_digest=None,registry=None):
    from .research_memo import research_memo_context
    manifest_path = Path(dataset).parent/"manifest.json"
    manifest = core._read_json(manifest_path) if manifest_path.is_file() else {}
    protocol_path = getattr(config,"protocol_path",None)
    stored_protocol = core._read_json(Path(protocol_path)) if protocol_path and Path(protocol_path).is_file() else {}
    protocol = core.protocol_spec_parameters(stored_protocol) if stored_protocol else None
    eligibility = {name:{key:row[key] for key in ("eligible_races","eligible_rows","contract","finish_time_basis") if key in row}
                   for name,row in manifest.get("target_eligibility",{}).items() if isinstance(row,dict)}
    current = {key:manifest.get(key) for key in ("dataset_id","raw_corpus_manifest_id","rows","races","date_min","date_max",
        "evaluation_population_id","comparison_contract_id","availability_policy","availability_tier","unsupported_requirements")}
    current.update(dataset_sha256=dataset_digest or core._hash_file(dataset),
        protocol_id=stored_protocol.get("protocol_id"),protocol_parameters=protocol,target_eligibility=eligibility,
        population_scope="published development only; confirmation identities and labels excluded")
    inputs = {"ready":False,"raw_manifest_sha256":None,"raw_corpus_manifest_id":None,"accepted_raw_corpus_manifest_ids":[],
        "supported_event_policy_ids":["strict","assumed_retrospective"],
        "event_policy_selection":"explicit; no availability policy is inferred from missing metadata",
        "retrospective_lags_required":True,"known_feature_definition_ids":[],
        "scope":"Identity inputs only; registry still validates source files, population and target eligibility"}
    try:
        _,raw = _dataset_build_inputs(config)
        inputs.update(_raw_manifest_identity(raw),ready=True,unavailable_reason=None)
    except (OSError,ValueError) as exc:
        inputs["unavailable_reason"] = str(exc)
    if registry is not None:
        feature_root = Path(registry.feature_registry)
    else:
        feature_root = Path(getattr(config,"dataset_registry_path",None) or Path(config.campaign_dir)/"dataset-registry")/"feature-definitions"
    if feature_root.is_dir():
        from .feature_definitions import FeatureRegistry
        features = FeatureRegistry(feature_root)
        for path in sorted(feature_root.glob("*.json")):
            try:
                inputs["known_feature_definition_ids"].append(features.get(path.stem).content_id())
            except (OSError,ValueError):
                continue
    return {"research_memo":research_memo_context(),
            "recipe_schema":PipelineRecipe.model_json_schema(),
            "parent_trial_id_contract":{
                "sole_valid_source":"completed_trial_index[].attempt_id",
                "references":"Historical context only. Reference champion attempt IDs are not valid parent_trial_ids unless also present in completed_trial_index.",
                "initial_campaign":"When completed_trial_index is empty, every proposal must use empty parent_trial_ids.",
            },
            "paper_research":{
                "request_schema":PaperResearchRequest.model_json_schema(),
                "parent_source":"paper_research.eligible_attempts from current completed win-probability trials only",
                "pending_ceiling":8,"active_ceiling":1,"requests_per_decision_ceiling":4,
                "budget_scope":"Independent of trial_budget and 80/20 fits; budget zero may request paper research",
                "quote_scope":"Fair prices without quotes; explicit hypothetical scenario payouts only; no wagering or extra planner calls",
            },
            "eligible_predictors":manifest.get("predictor_catalog",manifest.get("feature_catalog",manifest.get("eligible_predictors",[]))),
            "current_dataset":current,"dataset_request_inputs":inputs,
            "numeric_columns_by_schema":{k:list(v.numeric) for k,v in core.FEATURE_SCHEMAS.items()},
            "dataset_requests":"Immutable verified datasets only; publication tiers remain separate",
            "portfolio":"80% classical Benter; 20% planner-selected experimental ancestry",
            "trial_ceiling":config.proposal_batch_size}
