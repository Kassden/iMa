"""Continuous v5 controller: one ledger owner, planner checkpoints, shared evidence."""
from __future__ import annotations

import json
import time
import sys
import resource
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import multiprocessing

import psutil

from . import research_controller as core
from .feature_discovery_specs import DiscoverySpec, content_id
from .feature_studies import paired_feature_report
from .openrouter_orchestrator import OpenRouterConfig, choose_cycle_trial_budget, choose_research_proposals, normalize_openrouter_usage, _json_message_from_response
from .research_executor import RecipeExecutionRequest, execute_recipe
from .research_hypotheses import HypothesisMemory, champion_snapshot, reference_champions
from .research_scheduler import ResourceAdmission, ResourceRequest, fair_program_order
from .research_search import ProgramSearchController
from .research_specs import ResearchProposal, V5_PORTFOLIO_VERSION
from .research_store import ResearchLedger, utc_now


def _plan(evidence, config):
    started = time.monotonic()
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
    responses=[]
    def observe(response):
        responses.append(response)
        core._write_json_atomic(Path(config.campaign_dir)/"planner-calls"/f"{evidence['evidence_id']}-{len(responses):02d}.json",{"response":response,"evidence_id":evidence["evidence_id"],"received_at":utc_now()})
    remote=replace(remote,absolute_deadline_seconds=config.planner_timeout_seconds,response_observer=observe)
    budget = choose_cycle_trial_budget(evidence, config.proposal_batch_size, replace(remote,max_output_tokens=min(2000,remote.max_output_tokens),reasoning_effort="none"))
    evidence=dict(evidence,requested_trial_budget=budget["trial_budget"])
    response = choose_research_proposals(evidence, min(5,budget["trial_budget"],evidence.get("available_program_slots",5)), remote)
    response["trial_budget"] = budget["trial_budget"]
    response["budget_decision"] = budget
    response["source"] = "openrouter"
    response["planning_wall_seconds"] = time.monotonic()-started
    response["retire_program_ids"] = _json_message_from_response(response["raw_response"]).get("retire_program_ids",[])
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
    core._write_json_atomic(request.output_dir/"result.json",result.serializable())
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
        from importlib.metadata import version
        dependency_versions = {name:version(name) for name in ("featuretools","feature-engine","scikit-learn","pandas","optuna","mlflow","lightgbm","catboost","httpx","httpcore")}
        revision, environment = core._code_revision(),content_id({"runtime":core._environment_hash(),"dependencies":dependency_versions})
        core._write_json_atomic(directory/"environment.json",{"environment_hash":environment,"dependencies":dependency_versions})
        core._validate_campaign_identity(directory,dataset_hash=digest,protocol_parameters=protocol,code_revision=revision,environment_hash=environment,research_policy=config.research_policy)
        ledger = ResearchLedger(directory/"ledger.sqlite")
        search = ProgramSearchController(directory)
        memory = HypothesisMemory(directory/"hypotheses.sqlite")
        references_path = directory/"reference-champions.json"
        references = core._read_json(references_path) if references_path.is_file() else (reference_champions(config.reference_campaign_dir,digest,protocol) if config.reference_campaign_dir else [])
        core._write_json_atomic(references_path,references)
        profile = core.DatasetFeatureProfile(dataset,protocol)
        ledger.recover_running()
        core._reconcile_tells(ledger,search)
        tracking_errors = []
        tracking_upload = None
        next_tracking_retry = 0.0
        jobs = config.max_concurrent_trials if isinstance(config.max_concurrent_trials,int) else max(1,config.cpu_thread_budget)
        cpu_budget = min(config.cpu_thread_budget,max(1,psutil.cpu_count()-config.host_reserve_cpu_threads))
        ram_budget = min(config.ram_budget_gib,psutil.virtual_memory().total/1024**3-config.host_reserve_ram_gib)
        resources = ResourceAdmission(jobs,cpu_budget,ram_budget)
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
        exhausted_path = directory/"exhausted-programs.json"
        exhausted = set(core._read_json(exhausted_path) or [])
        with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context("spawn")) as pool, ThreadPoolExecutor(max_workers=1) as planner, ThreadPoolExecutor(max_workers=1) as uploader:
            while True:
                now = time.monotonic()
                if tracking_upload and tracking_upload[0].done():
                    upload,upload_attempt=tracking_upload
                    try:
                        linkage=upload.result()
                        if linkage is None:
                            raise ValueError("Tracking returned no run/model linkage")
                        ledger.mark_uploaded(upload_attempt,json.dumps(linkage,sort_keys=True))
                    except Exception as exc:
                        tracking_errors.append(f"{upload_attempt}: {type(exc).__name__}: {exc}")
                        next_tracking_retry=now+30
                    tracking_upload=None
                if config.mlflow_tracking_uri and tracking_upload is None and now>=next_tracking_retry:
                    outbox=ledger.pending_outbox()
                    if outbox:
                        tracking_upload=(uploader.submit(upload_result,config,dataset,outbox[0]),outbox[0]["attempt_id"])
                if (directory/"STOP").exists():
                    stop_mode = "stopped"
                if config.max_trials is not None and core._success_count(ledger) >= config.max_trials:
                    stop_mode = "complete"
                if core._consecutive_failure_count(ledger) >= config.max_consecutive_failed_trials:
                    stop_mode = "blocked_failures"
                capacity = {pid:0 if pid in exhausted else max(0,proposal.max_trials-sum(t.state.name!="PRUNED" for t in search.studies[pid].get_trials(deepcopy=False))) for pid,proposal in search.programs.items()}
                active_ids = set(pid for pid,v in capacity.items() if v>0) | {payload["program_id"] for _,payload in inflight.values()}
                available_program_slots = config.max_inflight_programs-len(active_ids)
                need_plan = blocked_lane is not None or sum(capacity.values()) <= config.queue_low_watermark or completed_since_plan >= config.replan_every_terminal_trials or now-last_plan >= config.planning_checkpoint_seconds
                quota_reserved = config.max_trials is not None and core._success_count(ledger)+len(inflight)>=config.max_trials
                if not stop_mode and not quota_reserved and planning is None and (available_program_slots>0 or blocked_lane is not None) and need_plan and now-last_plan >= (60 if last_error else 5):
                    if config.planner_mode == "openrouter" and core._planner_spend(directory) >= config.max_total_cost_usd:
                        stop_mode = "paused_spend"
                    else:
                        evidence = core._build_evidence(ledger.terminal_results(),search,profile)
                        evidence.update(champion_snapshot(references + ledger.terminal_results()))
                        evidence["reference_campaign_champions"] = [{"attempt_id":r["attempt_id"],"recipe":r["payload"]["recipe"],"objective_name":r["result"]["objective_name"],"objective_value":r["result"]["objective_value"],"source_campaign":str(config.reference_campaign_dir)} for r in references]
                        evidence["hypothesis_memory"] = memory.retrieve(limit=20)
                        evidence["feature_evidence"] = discovery_evidence(ledger.terminal_results())
                        evidence["numeric_columns_by_schema"] = {name:list(schema.numeric) for name,schema in core.FEATURE_SCHEMAS.items()}
                        recent_decisions = sorted((directory/"decisions").glob("cycle-*.json"))[-5:]
                        evidence["recent_planner_rejections"] = [{"cycle":d.get("cycle"),"rejected":d.get("rejected",[]),"provider_rejected":d.get("provider_rejected_proposals",[]),"error":d.get("error")} for d in (core._read_json(path) for path in recent_decisions) if d.get("rejected") or d.get("provider_rejected_proposals") or d.get("error")]
                        evidence["next_required_lane"] = blocked_lane
                        evidence["available_program_slots"] = max(1,available_program_slots)
                        evidence["retirement_required"] = available_program_slots <= 0
                        evidence["remaining_program_capacity"] = capacity
                        evidence["discovery_capabilities"] = {"schema":DiscoverySpec.model_json_schema(),"engine":"Featuretools DFS","screening":"Feature-engine + sklearn; fitted on training only","policy":"80% Benter / 20% rotating E1..E4; propose programs for all required contracts","contracts":core.V4_CONTRACTS,"rules":"May create new definitions by composing allowed entities, measurements, aggregates and windows. Return budgets chosen from evidence, not a mandatory ceiling. No arbitrary code, current result features, final holdout, or live promotion."}
                        evidence["evidence_id"] = content_id(evidence)
                        core._write_json_atomic(directory/"evidence"/f"cycle-{cycle:04d}.json",evidence)
                        planning = (planner.submit(_plan,evidence,config),cycle,evidence)
                        cycle += 1
                        last_plan = now
                        completed_since_plan = 0
                if planning and planning[0].done():
                    future, number, evidence = planning
                    decision = {"cycle":number,"source":"discovery_v5","suggestions":[],"evidence_id":evidence["evidence_id"],"budget_ceiling":config.proposal_batch_size,"planner_model":config.model if config.planner_mode=="openrouter" else None}
                    try:
                        response = future.result()
                        core._write_json_atomic(directory/"planner"/f"cycle-{number:04d}.json",{"response":response,"evidence_id":evidence["evidence_id"]})
                        budget = int(response["trial_budget"])
                        requested_budget = budget
                        if not 1<=budget<=config.proposal_batch_size: raise ValueError("Planner trial budget must be within the operator ceiling")
                        if not any(p.get("recipe",{}).get("feature_discovery") for p in response["proposals"]):
                            raise ValueError("v5 decision must include an executable feature-discovery program")
                        decision.update(trial_budget=budget,planner_model=config.model if config.planner_mode=="openrouter" else None,planner_usage=response.get("usage",{}),budget_decision=response.get("budget_decision"),planner_status=response["source"],rejected=[])
                        decision["planner_wall_seconds"] = response.get("planning_wall_seconds")
                        decision["provider_rejected_proposals"] = response.get("rejected_proposals",[])
                        retire = response.get("retire_program_ids",[])
                        if not isinstance(retire,list) or not set(retire)<=set(search.programs):
                            raise ValueError("Planner retirement references unknown programs")
                        exhausted.update(retire)
                        core._write_json_atomic(exhausted_path,sorted(exhausted))
                        decision["retired_program_ids"] = retire
                        available_program_slots = config.max_inflight_programs - len((set(pid for pid,v in capacity.items() if v>0)-exhausted)|{p["program_id"] for _,p in inflight.values()})
                        if sum(int(p["max_trials"]) for p in response["proposals"])>budget:
                            raise ValueError("Planner program budgets exceed its chosen decision budget")
                        known = {r["attempt_id"] for r in references+ledger.terminal_results() if r["status"]=="completed"}
                        accepted = []
                        for raw in response["proposals"]:
                            if len(accepted)>=available_program_slots:
                                decision["rejected"].append("Active program admission ceiling reached")
                                break
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
                                proposal_payload=proposal.model_dump(mode="json") | {"program_id":pid,"recipe_hash":proposal.recipe.recipe_hash(),"trial_id":proposal.proposal_id}
                                decision.setdefault("proposals",[]).append(proposal_payload)
                                decision["suggestions"].append(proposal_payload)
                                memory.record(f"proposal:{pid}",pid,"proposed",{"proposal":proposal.model_dump(mode="json"),"evidence_id":evidence["evidence_id"],"cycle":number})
                            except Exception as exc:
                                decision["rejected"].append(str(exc))
                        decision["program_ids"] = accepted
                        decision["approved_trials"] = requested_budget-budget
                        last_error = None if accepted else "Planner returned no new executable programs"
                        if accepted and not any(search.programs[pid].recipe.feature_discovery for pid in accepted):
                            last_error = "No feature-discovery program passed admission; inspect recorded rejection reasons"
                            decision["error"] = last_error
                        blocked_lane = None
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        decision.update(planner_status="error",error=last_error)
                        calls=[normalize_openrouter_usage(core._read_json(p)["response"]) for p in (directory/"planner-calls").glob(f"{evidence['evidence_id']}-*.json")]
                        if calls:
                            decision["planner_usage"]={k:sum(float(c[k]) for c in calls) if all(c.get(k) is not None for c in calls) else None for k in ("input_tokens","output_tokens","total_tokens","total_cost_usd")}
                            decision["planner_usage"]["cost_status"]="reported" if decision["planner_usage"]["total_cost_usd"] is not None else "unavailable"
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
                    trace_decision = core._read_json(directory/"decisions"/f"cycle-{number:04d}.json") or {"cycle":number,"source":"resume"}
                    cycle_attempts={r["attempt_id"] for r in cycle_results[number]}
                    trace_decision["suggestions"] = [r["payload"] for r in ledger.terminal_results() if r["attempt_id"] in cycle_attempts]
                    trace_decision["checkpoint"] = len(cycle_results[number])
                    trace_decision["planner_model"] = None
                    trace_decision["planner_usage"] = {"total_cost_usd":0,"cost_status":"not_a_planner_call"}
                    error=core._trace_completed_cycle(config,directory,trace_decision,cycle_results[number],ledger=ledger)
                    if error: tracking_errors.append(error)
                reserve_gib = min(config.host_reserve_ram_gib,psutil.virtual_memory().total/(1024**3)*.1)
                while not stop_mode and resources.admits(requirement) and psutil.virtual_memory().available/(1024**3)>reserve_gib:
                    if config.max_trials is not None and core._success_count(ledger)+len(inflight)>=config.max_trials: break
                    if pending:
                        original=pending.pop(0)
                        payload=original["payload"]
                        attempt=original["attempt_id"]
                    else:
                        reservations=ledger.reserved_attempts()
                        lane = core._portfolio_slot(len(reservations),config.research_policy)
                        target,model = core.V4_CONTRACTS[lane]
                        allowed=[pid for pid,p in search.programs.items() if pid not in exhausted and (p.recipe.target.kind,p.recipe.model.kind)==(target,model)]
                        suggestions=search.ask(1,allowed_program_ids=allowed,preferred_program_ids=fair_program_order(allowed,reservations))
                        if not suggestions:
                            exhausted.update(pid for pid in allowed if capacity.get(pid,0)>0)
                            core._write_json_atomic(exhausted_path,sorted(exhausted))
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
                core._write_json_atomic(directory/"status.json",{"status":"draining" if stop_mode else ("training" if inflight else "provider_planning" if planning else "paused_planning"),"updated_at":utc_now(),"cycle":cycle-1,"ledger":ledger.snapshot(),"resources":resources.snapshot(),"host_resources":host_resources,"planner_inflight":planning is not None,"planner_inflight_seconds":round(now-last_plan,1) if planning else 0,"blocked_lane":blocked_lane,"last_planner_error":last_error,"planner_spend_usd":core._planner_spend(directory),"tracking_errors":tracking_errors[-10:]})
                if stop_mode and not inflight and planning is None and tracking_upload is None and (not config.mlflow_tracking_uri or not ledger.pending_outbox()):
                    return core._finish_payload(directory,ledger,search,cycle-1,terminal_new,stop_mode,tracking_errors)
                if inflight:
                    wait(inflight,timeout=1,return_when=FIRST_COMPLETED)
                else:
                    time.sleep(1)


def result_recipe(payload):
    from .research_specs import PipelineRecipe
    return PipelineRecipe.model_validate(payload["recipe"])


def upload_result(config,dataset,item):
    """Network/artifact work only; the controller remains the sole ledger writer."""
    from .mlflow_tracking import MLflowConfig, log_research_package_version
    experiment,model=core._tracking_names(config.research_policy)
    tracking=MLflowConfig.from_values(tracking_uri=config.mlflow_tracking_uri,experiment_name=experiment,register_models=True,registered_model_name=model)
    return log_research_package_version(Path(item["result"]["artifacts"]["package"]),tracking,attempt_id=item["attempt_id"],result=item["result"],dataset_path=dataset)


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
        diagnostics = [core._read_json(p) for p in sorted(Path(path).parent.glob("discovery-diagnostics-*.json"))]
        selected = set(selections[-1].get("selected",[])) if selections else set()
        preview = sorted(manifest["catalog"],key=lambda row:(row["feature_id"] not in selected,row["feature_id"]))[:32]
        reports.append({"discovery_id":discovery,"attempt_id":item["attempt_id"],"spec":manifest["spec"],"engine_version":manifest["engine_version"],"candidate_count":len(manifest["catalog"]),"deferred_count":manifest["deferred_count"],"catalog_preview":preview,"training_fold_screening":selections[-1:],"diagnostics":diagnostics[-1:],"artifact":path})
        seen.add(discovery)
        if len(reports)>=5:
            break
    return reports
