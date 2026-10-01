"""Continuous v5 controller: one ledger owner, planner checkpoints, shared evidence."""
from __future__ import annotations

import json
import time
import sys
import resource
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import psutil

from . import research_controller as core
from .feature_discovery_specs import DiscoverySpec, content_id
from .feature_studies import paired_feature_report
from .openrouter_orchestrator import OpenRouterConfig, choose_cycle_trial_budget, choose_research_proposals
from .research_executor import RecipeExecutionRequest, execute_recipe
from .research_hypotheses import HypothesisMemory, champion_snapshot
from .research_scheduler import ResourceAdmission, ResourceRequest
from .research_search import ProgramSearchController
from .research_specs import ResearchProposal, V5_PORTFOLIO_VERSION
from .research_store import ResearchLedger, utc_now


def _plan(evidence, config):
    if config.planner_mode == "fixture":
        proposals = []
        seeds=core._v4_seed_proposals()[:min(5,config.proposal_batch_size)]
        budget=config.proposal_batch_size
        for index, old in enumerate(seeds):
            recipe = old.recipe.model_copy(update={"feature_discovery": DiscoverySpec(windows_days=(90,),domain_history=index==0,adjusted_speed_residuals=index==1,sequence_windows=(3,) if index>=2 else ())})
            program_budget=budget-(len(seeds)-index-1) if index==0 else 1
            budget-=program_budget
            proposals.append(old.model_copy(update={"proposal_id":f"fixture-v5-{evidence['terminal_watermark']}-{index}","recipe":recipe,"max_trials":program_budget,"evidence_ids":(evidence["evidence_id"],)}).model_dump(mode="json"))
        return {"proposals":proposals,"trial_budget":config.proposal_batch_size,"usage":{},"source":"fixture"}
    remote = OpenRouterConfig.from_env(model=config.model, service_tier=config.service_tier, max_output_tokens=config.max_output_tokens, timeout_seconds=config.planner_timeout_seconds, provider_endpoint=config.provider_endpoint, reasoning_effort=config.planner_reasoning_effort)
    budget = choose_cycle_trial_budget(evidence, config.proposal_batch_size, remote)
    evidence=dict(evidence,requested_trial_budget=budget["trial_budget"])
    response = choose_research_proposals(evidence, min(5,budget["trial_budget"]), remote)
    response["trial_budget"] = budget["trial_budget"]
    response["budget_decision"] = budget
    response["source"] = "openrouter"
    parts=(budget.get("usage",{}),response.get("usage",{}))
    response["usage"] = {k: sum(float(part[k]) for part in parts) if all(part.get(k) is not None for part in parts) else None for k in ("input_tokens","output_tokens","total_tokens","total_cost_usd")}
    response["usage"]["cost_status"]="reported" if response["usage"]["total_cost_usd"] is not None else "unavailable"
    return response


def _worker(request):
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    before = psutil.Process().cpu_times()
    with threadpool_limits(limits=1):
        result = execute_recipe(request)
    after = psutil.Process().cpu_times()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.metrics["worker_resources"] = {"wall_seconds": time.monotonic()-started, "cpu_seconds": after.user+after.system-before.user-before.system, "peak_rss_gib": peak/(1024**3 if sys.platform=="darwin" else 1024**2), "thread_limit":1}
    return result


def run_v5_campaign(config):
    config.validate()
    if not config.protocol_path or not config.dataset_path:
        raise ValueError("v5 requires immutable dataset and protocol paths")
    directory = Path(config.campaign_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with core.campaign_lock(directory):
        dataset = core._resolve_dataset(config,directory)
        digest = core._hash_file(dataset)
        protocol = core._protocol_parameters(config)
        revision, environment = core._code_revision(),core._environment_hash()
        core._validate_campaign_identity(directory,dataset_hash=digest,protocol_parameters=protocol,code_revision=revision,environment_hash=environment,research_policy=config.research_policy)
        ledger = ResearchLedger(directory/"ledger.sqlite")
        search = ProgramSearchController(directory)
        memory = HypothesisMemory(directory/"hypotheses.sqlite")
        profile = core.DatasetFeatureProfile(dataset,protocol)
        ledger.recover_running()
        core._reconcile_tells(ledger,search)
        tracking_errors = core._reconcile_tracking(config,ledger,dataset)
        jobs = config.max_concurrent_trials if isinstance(config.max_concurrent_trials,int) else max(1,config.cpu_thread_budget)
        resources = ResourceAdmission(jobs,config.cpu_thread_budget,config.ram_budget_gib)
        requirement = ResourceRequest(1,4)
        completed_since_plan = 0
        last_plan = 0.0
        cycle = core._next_cycle_number(directory)
        decisions = {pid: int(path.stem.split("-")[-1]) for path in (directory/"decisions").glob("cycle-*.json") for pid in (core._read_json(path) or {}).get("program_ids",[])}
        attempt_cycles = {}
        cycle_results = {}
        for previous in ledger.terminal_results():
            number = decisions.get(previous["payload"].get("program_id"),0)
            cycle_results.setdefault(number,[]).append(previous["result"])
        terminal_new = []
        planning = None
        pending = list(ledger.pending_attempts())
        inflight = {}
        stop_mode = None
        last_error = None
        blocked_lane = None
        with ProcessPoolExecutor(max_workers=jobs) as pool, ThreadPoolExecutor(max_workers=1) as planner:
            while True:
                now = time.monotonic()
                if (directory/"STOP").exists():
                    stop_mode = "stopped"
                if config.max_trials is not None and core._success_count(ledger) >= config.max_trials:
                    stop_mode = "complete"
                if core._consecutive_failure_count(ledger) >= config.max_consecutive_failed_trials:
                    stop_mode = "blocked_failures"
                need_plan = blocked_lane is not None or not search.has_capacity() or completed_since_plan >= config.replan_every_terminal_trials or now-last_plan >= config.planning_checkpoint_seconds
                if not stop_mode and planning is None and need_plan and now-last_plan >= (60 if last_error else 5):
                    if config.planner_mode == "openrouter" and core._planner_spend(directory) >= config.max_total_cost_usd:
                        stop_mode = "paused_spend"
                    else:
                        evidence = core._build_evidence(ledger.terminal_results(),search,profile)
                        evidence.update(champion_snapshot(ledger.terminal_results()))
                        evidence["hypothesis_memory"] = memory.retrieve(limit=20)
                        evidence["feature_evidence"] = discovery_evidence(ledger.terminal_results())
                        evidence["next_required_lane"] = blocked_lane
                        evidence["discovery_capabilities"] = {"schema":DiscoverySpec.model_json_schema(),"engine":"Featuretools DFS","screening":"Feature-engine + sklearn; fitted on training only","policy":"80% Benter / 20% rotating E1..E4; propose programs for all required contracts","contracts":core.V4_CONTRACTS,"rules":"May create new definitions by composing allowed entities, measurements, aggregates and windows. Return budgets chosen from evidence, not a mandatory ceiling. No arbitrary code, current result features, final holdout, or live promotion."}
                        evidence["evidence_id"] = content_id(evidence)
                        core._write_json_atomic(directory/"evidence"/f"cycle-{cycle:04d}.json",evidence)
                        planning = (planner.submit(_plan,evidence,config),cycle,evidence)
                        cycle += 1
                        last_plan = now
                        completed_since_plan = 0
                if planning and planning[0].done():
                    future, number, evidence = planning
                    decision = {"cycle":number,"source":"discovery_v5","suggestions":[],"evidence_id":evidence["evidence_id"],"budget_ceiling":config.proposal_batch_size}
                    try:
                        response = future.result()
                        core._write_json_atomic(directory/"planner"/f"cycle-{number:04d}.json",{"response":response,"evidence_id":evidence["evidence_id"]})
                        budget = min(config.proposal_batch_size,int(response["trial_budget"]))
                        requested_budget = budget
                        if budget < 1: raise ValueError("Planner trial budget must be positive")
                        decision.update(trial_budget=budget,planner_model=config.model if config.planner_mode=="openrouter" else None,planner_usage=response.get("usage",{}),budget_decision=response.get("budget_decision"),planner_status=response["source"],rejected=[])
                        if sum(int(p["max_trials"]) for p in response["proposals"])>budget:
                            raise ValueError("Planner program budgets exceed its chosen decision budget")
                        known = {r["attempt_id"] for r in ledger.terminal_results() if r["status"]=="completed"}
                        accepted = []
                        for raw in response["proposals"]:
                            try:
                                proposal = ResearchProposal.model_validate(raw)
                                core._portfolio_identity(proposal.recipe,config.research_policy)
                                if proposal.evidence_ids != (evidence["evidence_id"],): raise ValueError("stale evidence")
                                if not set(proposal.parent_trial_ids) <= known: raise ValueError("unknown parent")
                                error = profile.admission_error(proposal.recipe)
                                if error: raise ValueError(error)
                                if budget <= 0: break
                                proposal = proposal.model_copy(update={"max_trials":min(proposal.max_trials,budget)})
                                pid = search.register(proposal)
                                if pid in decisions: continue
                                decisions[pid] = number
                                budget -= proposal.max_trials
                                accepted.append(pid)
                                memory.record(f"proposal:{pid}",pid,"proposed",{"proposal":proposal.model_dump(mode="json"),"evidence_id":evidence["evidence_id"],"cycle":number})
                            except Exception as exc:
                                decision["rejected"].append(str(exc))
                        decision["program_ids"] = accepted
                        decision["approved_trials"] = requested_budget-budget
                        last_error = None if accepted else "Planner returned no new executable programs"
                        blocked_lane = None
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        decision.update(planner_status="error",error=last_error)
                    core._persist_decision(directory,number,decision)
                    cycle_results[number]=[]
                    error = core._trace_planner_decision(config,directory,decision)
                    if error: tracking_errors.append(error)
                    planning = None
                for future in list(inflight):
                    if not future.done(): continue
                    attempt,payload = inflight.pop(future)
                    resources.release(attempt)
                    core._append_jsonl(directory/"queue.jsonl",{"event":"released","attempt_id":attempt,"time":utc_now(),"resources":resources.snapshot()})
                    try:
                        result = future.result()
                    except Exception as exc:
                        from .research_executor import RecipeExecutionResult
                        recipe = result_recipe(payload)
                        result = RecipeExecutionResult(1,attempt,payload.get("proposal_id",payload["trial_id"]),payload["trial_number"],recipe.recipe_hash(),recipe.target.kind,"failed","worker_failure",None,{}, {}, {"dataset_hash":digest,"code_revision":revision},0,f"{type(exc).__name__}: {exc}")
                        stop_mode = "worker_failure"
                    row = result.serializable() | {"trial_id":payload["trial_id"],"program_id":payload["program_id"],"target_parameters":payload["recipe"]["target"]["parameters"]}
                    row.update(core._portfolio_identity(result_recipe(payload),config.research_policy))
                    comparisons=[]
                    if row["status"]=="completed":
                        candidates=[r for r in ledger.terminal_results() if r["status"]=="completed" and r["result"]["target_kind"]==row["target_kind"] and r["result"]["objective_name"]==row["objective_name"] and r["payload"]["recipe"]["target"]["parameters"]==row["target_parameters"] and r["result"].get("lineage",{}).get("protocol_id")==row["lineage"].get("protocol_id")]
                        if candidates:
                            comparator=min(candidates,key=lambda r:r["result"]["objective_value"])
                            try:
                                report=paired_feature_report(row["artifacts"]["predictions"],comparator["result"]["artifacts"]["predictions"],row["target_kind"])
                                report["comparator_attempt_id"]=comparator["attempt_id"]
                                core._write_json_atomic(directory/"trials"/attempt/"paired-feature-report.json",report)
                                row["artifacts"]["paired_feature_report"]=str(directory/"trials"/attempt/"paired-feature-report.json")
                                comparisons.append(report)
                            except ValueError as exc:
                                comparisons.append({"verdict":"not_comparable","reason":str(exc)})
                    ledger.complete_attempt(attempt,row,status=row["status"])
                    core._append_jsonl(directory/"trials.jsonl",row)
                    terminal_new.append(row)
                    memory.record(f"outcome:{attempt}",payload["program_id"],"evaluated",{"attempt_id":attempt,"status":row["status"],"objective_name":row["objective_name"],"objective_value":row["objective_value"],"error":row.get("error"),"discovery_id":row["lineage"].get("discovery_id"),"comparisons":comparisons,"verdict":comparisons[0]["verdict"] if comparisons else "inconclusive"})
                    number = attempt_cycles.get(attempt,0)
                    cycle_results.setdefault(number,[]).append(row)
                    completed_since_plan += 1
                    core._reconcile_tells(ledger,search)
                    tracking_errors.extend(core._reconcile_tracking(config,ledger,dataset))
                    trace_decision = core._read_json(directory/"decisions"/f"cycle-{number:04d}.json") or {"cycle":number,"source":"resume"}
                    trace_decision["suggestions"] = [payload]
                    trace_decision["checkpoint"] = len(cycle_results[number])
                    trace_decision["planner_model"] = None
                    trace_decision["planner_usage"] = {"total_cost_usd":0,"cost_status":"not_a_planner_call"}
                    error=core._trace_completed_cycle(config,directory,trace_decision,cycle_results[number],ledger=ledger)
                    if error: tracking_errors.append(error)
                reserve_gib = min(8,psutil.virtual_memory().total/(1024**3)*.1)
                while not stop_mode and resources.admits(requirement) and psutil.virtual_memory().available/(1024**3)>reserve_gib:
                    if config.max_trials is not None and core._success_count(ledger)+len(inflight)>=config.max_trials: break
                    if pending:
                        original=pending.pop(0)
                        payload=original["payload"]
                        attempt=original["attempt_id"]
                    else:
                        lane = core._portfolio_slot(len(ledger.reserved_attempts()),config.research_policy)
                        target,model = core.V4_CONTRACTS[lane]
                        allowed=[pid for pid,p in search.programs.items() if (p.recipe.target.kind,p.recipe.model.kind)==(target,model)]
                        suggestions=search.ask(1,allowed_program_ids=allowed)
                        if not suggestions:
                            blocked_lane=lane
                            break
                        suggestion=suggestions[0]
                        signature=core._execution_signature(suggestion.recipe,digest,protocol,revision,environment)
                        payload=suggestion.serializable() | {"signature":signature,"dataset_hash":digest,"protocol_parameters":protocol,"code_revision":revision,"environment_hash":environment}
                        record=ledger.reserve_attempt(signature,payload)
                        if record.status=="completed": continue
                        attempt=record.attempt_id
                    ledger.mark_running(attempt)
                    recipe=result_recipe(payload)
                    number=decisions.get(payload["program_id"],0)
                    attempt_cycles[attempt]=number
                    resources.reserve(attempt,requirement)
                    core._append_jsonl(directory/"queue.jsonl",{"event":"dispatched","attempt_id":attempt,"program_id":payload["program_id"],"time":utc_now(),"resources":resources.snapshot()})
                    request=RecipeExecutionRequest(attempt,payload.get("proposal_id",payload["trial_id"]),payload["trial_number"],recipe,dataset,directory/"trials"/attempt,protocol,digest,revision,environment,V5_PORTFOLIO_VERSION)
                    inflight[pool.submit(_worker,request)]=(attempt,payload)
                host = psutil.virtual_memory()
                host_resources = {"logical_cpus":psutil.cpu_count(),"load_one_minute_percent":100*psutil.getloadavg()[0]/psutil.cpu_count(),"available_ram_gib":host.available/1024**3}
                core._write_json_atomic(directory/"status.json",{"status":"draining" if stop_mode else ("training" if inflight else "provider_planning" if planning else "paused_planning"),"updated_at":utc_now(),"cycle":cycle-1,"ledger":ledger.snapshot(),"resources":resources.snapshot(),"host_resources":host_resources,"planner_inflight":planning is not None,"blocked_lane":blocked_lane,"last_planner_error":last_error,"planner_spend_usd":core._planner_spend(directory),"tracking_errors":tracking_errors[-10:]})
                if stop_mode and not inflight and planning is None:
                    return core._finish_payload(directory,ledger,search,cycle-1,terminal_new,stop_mode,tracking_errors)
                if inflight:
                    wait(inflight,timeout=1,return_when=FIRST_COMPLETED)
                else:
                    time.sleep(1)


def result_recipe(payload):
    from .research_specs import PipelineRecipe
    return PipelineRecipe.model_validate(payload["recipe"])


def discovery_evidence(terminal):
    """Bounded training-only feature evidence shared across all model families."""
    reports = []
    seen = set()
    for item in reversed(terminal):
        result = item.get("result",{})
        discovery = result.get("lineage",{}).get("discovery_id")
        if not discovery or discovery in seen:
            continue
        path = result.get("artifacts",{}).get("discovery_manifest")
        if not path or not Path(path).is_file():
            continue
        manifest = core._read_json(Path(path))
        selections = [core._read_json(p) for p in sorted(Path(path).parent.glob("discovery-selection-*.json"))]
        reports.append({"discovery_id":discovery,"attempt_id":item["attempt_id"],"spec":manifest["spec"],"engine_version":manifest["engine_version"],"candidate_count":len(manifest["catalog"]),"deferred_count":manifest["deferred_count"],"catalog_preview":manifest["catalog"][:20],"training_fold_screening":selections[-1:],"artifact":path})
        seen.add(discovery)
        if len(reports)>=5:
            break
    return reports
