import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ima import research_controller as core
from ima.optimizer import CampaignConfig
from ima.research_expansion import (
    DecisionStore, PlannerDecision, _prepare_program, _program_context, _build_dataset,
    _recover_orphan_asks, _native_threads, _workload, _planner_cost, _planner_spend, apply_decision,
    plan_decision, run_expansion_campaign, _capabilities, _validate_raw_manifest, _needs_lane_refill,
)
from ima.research_search import ProgramSearchController
from ima.research_store import ResearchLedger
from tests import test_research_controller as controller_fixtures
from tests.test_research_expansion import proposal


def executors(**kwargs):
    return ThreadPoolExecutor(max_workers=kwargs["max_workers"])


class ExpansionResourceIntegrationTests(unittest.TestCase):
    def test_native_threads_share_budget_with_deep_queue(self):
        recipe = proposal(trials=40).recipe
        self.assertEqual(4, _native_threads(recipe, 178000, 24))
        self.assertEqual(1, _native_threads(recipe, 178000, 24, pending_trials=40))
        self.assertEqual(2, _native_threads(recipe, 178000, 24, pending_trials=12))
        self.assertEqual(4, _native_threads(recipe, 178000, 24, pending_trials=3))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset,self.protocol = controller_fixtures.ResearchControllerTests().fixture(self.root)
        self.config = CampaignConfig(campaign_dir=self.root/"campaign",policy="agentic",
            research_policy="expansion_v6",planner_mode="fixture",max_trials=1,
            proposal_batch_size=1,max_concurrent_trials=1,max_active_preparations=1,
            host_reserve_cpu_threads=0,host_reserve_ram_gib=0,
            dataset_path=self.dataset,protocol_path=self.protocol)

    def test_partial_extension_does_not_authorize_its_budget(self):
        search = ProgramSearchController(self.config.campaign_dir)
        store = DecisionStore(self.config.campaign_dir/"decisions.sqlite")
        config = SimpleNamespace(campaign_dir=self.config.campaign_dir,proposal_batch_size=10,
            max_new_programs_per_decision=4,max_pending_programs=8)
        evidence = {"evidence_id":"evidence-1","completed_trial_index":[]}
        first = PlannerDecision(decision_id="first",evidence_id="evidence-1",trial_budget=2,programs=(proposal(trials=2),))
        pid = apply_decision(first,store,search,set(),config,evidence,None)[0]
        extension = PlannerDecision(decision_id="extend",evidence_id="evidence-1",trial_budget=3,extensions={pid:3})
        original = search.set_budget
        def interrupted(pid,budget):
            original(pid,budget)
            raise RuntimeError("after budget write")
        with patch.object(search,"set_budget",side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                apply_decision(extension,store,search,set(),config,evidence,None)
        self.assertEqual(5,search.programs[pid].max_trials)
        self.assertEqual(2,store.authorized_budgets()[pid])
        apply_decision(extension,store,search,set(),config,evidence,None)
        self.assertEqual(5,store.authorized_budgets()[pid])

    def test_ask_without_ledger_reservation_is_recovered_once(self):
        search = ProgramSearchController(self.config.campaign_dir)
        pid = search.register(proposal())
        suggested = search.ask(1,allowed_program_ids=[pid])[0]
        ledger = ResearchLedger(self.config.campaign_dir/"ledger.sqlite")
        context = {"dataset":self.dataset,"digest":core._hash_file(self.dataset),"protocol":core._protocol_parameters(self.config)}
        self.assertEqual(1,_recover_orphan_asks(search,ledger,{pid:context},{pid},"revision","environment"))
        self.assertEqual(0,_recover_orphan_asks(search,ledger,{pid:context},{pid},"revision","environment"))
        pending = ledger.pending_attempts()
        self.assertEqual(suggested.trial_number,pending[0]["payload"]["trial_number"])
        self.assertEqual(suggested.recipe.recipe_hash(),pending[0]["payload"]["recipe_hash"])

    def test_preparation_uses_formula_frame_and_same_shared_generator_identity(self):
        from ima.research_specs import PipelineRecipe
        from ima.research_executor import RecipeExecutionRequest,_load_dataset,_v6_feature_frame
        from ima.feature_program import materialize
        definition = {"name":"novel_rating_sq","expression_ast":{"op":"power","args":[{"op":"ref","ref":"horse_rating"},{"op":"constant","value":2}]},"input_refs":["horse_rating"],"unit":"rating^2"}
        recipe = PipelineRecipe(schema_version=3,model={"kind":"benter_conditional_logit"},feature_definitions=(definition,),feature_discovery={"schema_version":2,"measurements":["novel_rating_sq"],"windows_days":[90]})
        digest = core._hash_file(self.dataset)
        prepared = _prepare_program(self.dataset,recipe,digest,self.config.campaign_dir)
        request = RecipeExecutionRequest("trial","proposal",0,recipe,self.dataset,self.root/"trial")
        frame = _v6_feature_frame(_load_dataset(self.dataset),request)
        _,executed = materialize(frame,recipe.feature_discovery,digest,self.config.campaign_dir/"discovery-cache",shared=True,input_metadata=frame.attrs["synthesis_metadata"])
        self.assertEqual(prepared["matrix_id"],executed["matrix_id"])
        self.assertGreater(prepared["shared_bytes"],0)

    def test_paid_rejected_decision_keeps_cost_and_planner_evidence_id(self):
        from dataclasses import replace
        config = replace(self.config,max_trials=None,timeout_minutes=.03)
        traces = []
        def planned(evidence,config):
            item = proposal(trials=2).model_copy(update={"evidence_ids":(evidence["evidence_id"],)})
            return {"decision":{"decision_id":evidence["decision_id"],"evidence_id":evidence["evidence_id"],"trial_budget":2,"programs":[item.model_dump(mode="json")]},"usage":{"total_cost_usd":.125}}
        def trace(directory,payload,config,**kwargs):
            if kwargs["role"] == "decision":
                traces.append(payload)
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.plan_decision",side_effect=planned),patch("ima.research_expansion.log_snapshot",side_effect=trace):
            result = run_expansion_campaign(config)
        self.assertEqual("timeout",result["mode"])
        paid = json.loads((config.campaign_dir/"decisions"/"D000001.json").read_text())
        evidence = json.loads((config.campaign_dir/"evidence"/"D000001.json").read_text())
        self.assertEqual("failed",paid["planner_status"])
        self.assertEqual(.125,paid["planner_usage"]["total_cost_usd"])
        self.assertEqual(evidence["evidence_id"],traces[0]["evidence_id"])
        self.assertEqual([],ResearchLedger(config.campaign_dir/"ledger.sqlite").reserved_attempts())

    def test_structural_workload_excludes_regularization_floats_and_allocates_threads(self):
        recipe = proposal().recipe
        changed = recipe.model_copy(update={"model":recipe.model.model_copy(update={"parameters":{"l2":10.}})})
        self.assertEqual(_workload(recipe,10000,"r","e").fingerprint(),_workload(changed,10000,"r","e").fingerprint())
        self.assertEqual(4,_native_threads(recipe,100000,8))
        self.assertEqual(2,_native_threads(recipe,100000,2))
        self.assertEqual(1,_native_threads(recipe,100,8))

    def test_cost_survives_crash_before_receipt_without_double_charging(self):
        root = self.config.campaign_dir
        core._write_json_atomic(root/"planner-calls"/"D000001-01.json",{"response":{"usage":{"cost":.1}}})
        self.assertAlmostEqual(.1,_planner_cost(root))
        core._write_json_atomic(root/"planner-responses"/"D000001.json",{"response":{"usage":{"total_cost_usd":.2}}})
        self.assertAlmostEqual(.1,_planner_cost(root))
        core._write_json_atomic(root/"decisions"/"D000001.json",{"planner_usage":{"total_cost_usd":.2}})
        core._write_json_atomic(root/"planner-responses"/"D000002.json",{"response":{"usage":{"total_cost_usd":.3}}})
        self.assertAlmostEqual(.4,_planner_cost(root))
        core._write_json_atomic(root/"planner-calls"/"D000001-02.json",{"response":{"usage":{"cost":.15}}})
        self.assertAlmostEqual(.55,_planner_cost(root))
        core._write_json_atomic(root/"planner-calls"/"D000001-03.json",{"response":{"usage":{"prompt_tokens":10}}})
        self.assertIsNone(_planner_cost(root))
        self.assertTrue(_planner_spend(root)["spend_unknown"])
        self.assertAlmostEqual(.55,_planner_spend(root)["known_spend_usd"])

    def test_verified_dataset_reference_resolves_its_own_protocol_and_hash(self):
        registry = SimpleNamespace(verify=lambda identifier:{"features_path":str(self.dataset),"protocol_path":str(self.protocol),"rows":100})
        recipe = proposal().recipe.model_copy(update={"dataset_ref":"dataset-"+"a"*64})
        context = _program_context(recipe,self.config,registry)
        self.assertEqual(core._hash_file(self.dataset),context["digest"])
        self.assertEqual(core._protocol_parameters(self.config),context["protocol"])

    def test_registry_nested_protocol_uses_shared_parameters_contract(self):
        parameters = core._protocol_parameters(self.config)
        nested = self.root/"registry-protocol.json"
        core._write_json_atomic(nested,{"protocol_id":"registry-id","spec":parameters | {"final_confirmation_race_ids":[],"final_confirmation_start":None},"folds":[]})
        registry = SimpleNamespace(verify=lambda identifier:{"features_path":str(self.dataset),"protocol_path":str(nested),"rows":100})
        recipe = proposal().recipe.model_copy(update={"dataset_ref":"dataset-"+"b"*64})
        self.assertEqual(parameters,_program_context(recipe,self.config,registry)["protocol"])

    def test_unknown_call_is_receipted_and_stops_repair_call(self):
        from dataclasses import replace
        from ima.openrouter_orchestrator import OpenRouterConfig
        config = replace(self.config,planner_mode="openrouter")
        calls = []
        def choose(evidence,limits,remote):
            calls.append("first")
            remote.response_observer({"usage":{"prompt_tokens":10}})
            calls.append("repair")
        with patch("ima.openrouter_orchestrator.OpenRouterConfig.from_env",return_value=OpenRouterConfig("unused","model")),patch("ima.openrouter_orchestrator.choose_research_decision",side_effect=choose):
            with self.assertRaisesRegex(ValueError,"cost is unknown"):
                plan_decision({"decision_id":"D000001","evidence_id":"evidence"},config)
        self.assertEqual(["first"],calls)
        self.assertTrue((config.campaign_dir/"planner-calls"/"D000001-01.json").is_file())
        self.assertIsNone(_planner_cost(config.campaign_dir))

    def test_unknown_durable_spend_freezes_new_paid_planning_on_restart(self):
        from dataclasses import replace
        config = replace(self.config,planner_mode="openrouter",model="test/model",max_total_cost_usd=1.,max_trials=None,timeout_minutes=.03)
        core._write_json_atomic(config.campaign_dir/"planner-calls"/"D000001-01.json",{"response":{"usage":{}}})
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.plan_decision") as planner,patch("ima.research_expansion.log_snapshot"):
            result = run_expansion_campaign(config)
        planner.assert_not_called()
        self.assertEqual("timeout",result["mode"])
        status = core._read_json(config.campaign_dir/"status.json")
        self.assertTrue(status["planner_spend_unknown"])
        self.assertIsNone(status["planner_spend_usd"])

    def test_same_batch_definition_is_published_before_dataset_request(self):
        from ima.feature_definitions import FeatureDefinition,FeatureRegistry
        from ima.dataset_registry import DatasetRegistry
        definition = FeatureDefinition(name="rating_squared",expression_ast={"op":"power","args":[{"op":"ref","ref":"horse_rating"},{"op":"constant","value":2}]},input_refs=("horse_rating",),unit="rating^2")
        item = proposal().model_copy(update={"recipe":proposal().recipe.model_copy(update={"feature_definitions":(definition.model_dump(mode="json"),)})})
        registry = DatasetRegistry(self.root/"registry",feature_registry=self.root/"shared-definitions")
        raw = self.root/"raw.json"
        core._write_json_atomic(raw,{"manifest_id":"raw-1"})
        request = {"request_id":"dataset-request","raw_corpus_manifest_id":"raw-1","rationale":"test","evidence_watermark":"evidence-1","feature_definition_ids":[definition.content_id()]}
        decision = PlannerDecision(decision_id="definition-batch",evidence_id="evidence-1",trial_budget=1,programs=(item,),dataset_requests=(request,))
        search = ProgramSearchController(self.config.campaign_dir)
        store = DecisionStore(self.config.campaign_dir/"decisions.sqlite")
        original = registry.submit
        def submit(value):
            self.assertEqual(definition.content_id(),FeatureRegistry(registry.feature_registry).get(definition.content_id()).content_id())
            return original(value)
        with patch("ima.research_expansion._dataset_registry",return_value=registry),patch("ima.research_expansion._dataset_build_inputs",return_value=(self.root,raw)),patch.object(registry,"submit",side_effect=submit):
            apply_decision(decision,store,search,set(),self.config,{"evidence_id":"evidence-1","completed_trial_index":[]},None)
        self.assertEqual("requested",registry.get("dataset-request")["status"])
        known_request = dict(request,request_id="known-definition-request")
        known = PlannerDecision(decision_id="known-definition",evidence_id="evidence-1",trial_budget=1,programs=(proposal(),),dataset_requests=(known_request,))
        with patch("ima.research_expansion._dataset_registry",return_value=registry),patch("ima.research_expansion._dataset_build_inputs",return_value=(self.root,raw)):
            apply_decision(known,store,search,set(),self.config,{"evidence_id":"evidence-1","completed_trial_index":[]},None)
        self.assertEqual("requested",registry.get("known-definition-request")["status"])

    def test_unknown_dataset_definition_rejects_before_decision_actions(self):
        from ima.dataset_registry import DatasetRegistry
        registry = DatasetRegistry(self.root/"registry")
        raw = self.root/"raw.json"
        core._write_json_atomic(raw,{"manifest_id":"raw-1"})
        request = {"request_id":"unknown-request","raw_corpus_manifest_id":"raw-1","rationale":"test","evidence_watermark":"evidence-1","feature_definition_ids":["a"*24]}
        decision = PlannerDecision(decision_id="unknown-definition",evidence_id="evidence-1",trial_budget=1,programs=(proposal(),),dataset_requests=(request,))
        search = ProgramSearchController(self.config.campaign_dir)
        store = DecisionStore(self.config.campaign_dir/"decisions.sqlite")
        with patch("ima.research_expansion._dataset_registry",return_value=registry),patch("ima.research_expansion._dataset_build_inputs",return_value=(self.root,raw)):
            with self.assertRaisesRegex(ValueError,"unknown feature definition"):
                apply_decision(decision,store,search,set(),self.config,{"evidence_id":"evidence-1","completed_trial_index":[]},None)
        self.assertEqual({},search.programs)
        self.assertEqual([],store.dataset_actions())
        self.assertFalse((registry.root/"requests"/"unknown-request.json").exists())

    def test_dataset_worker_uses_same_shared_feature_registry(self):
        registry = SimpleNamespace(build=lambda *args,**kwargs:{"dataset_id":"built"})
        with patch("ima.dataset_registry.DatasetRegistry",return_value=registry) as constructor:
            result = _build_dataset("registry","request","snapshot","raw",feature_registry="shared-definitions")
        constructor.assert_called_once_with("registry",feature_registry="shared-definitions")
        self.assertEqual("built",result["dataset_id"])

    def test_initial_capabilities_advertise_exact_request_identity_and_safe_context(self):
        from dataclasses import replace
        from ima.dataset_specs import DatasetRequest
        from ima.dataset_registry import DatasetRegistry
        from ima.feature_definitions import FeatureDefinition,FeatureRegistry
        snapshot = self.root/"snapshot"
        snapshot.mkdir()
        raw = snapshot/"raw_manifest.json"
        core._write_json_atomic(snapshot/"manifest.json",{"raw_manifest_path":"raw_manifest.json"})
        core._write_json_atomic(raw,{"manifest_id":"raw-known","snapshot_id":"snapshot-known","raw_corpus_manifest_id":"corpus-known","labels":["protected-label"]})
        registry = DatasetRegistry(self.root/"registry",feature_registry=self.root/"shared-definitions")
        definition = FeatureDefinition(name="constant_feature",expression_ast={"op":"constant","value":1},input_refs=())
        identifier = FeatureRegistry(registry.feature_registry).register(definition,{})
        parameters = core._protocol_parameters(self.config)
        core._write_json_atomic(self.protocol,{"protocol_id":"protocol-known","spec":parameters | {"final_confirmation_races":20,"final_confirmation_race_ids":["protected-race"],"final_confirmation_start":"2099-01-01"},"folds":[{"score_race_ids":["protected-score"]}]})
        catalog = {"horse_age":{"unit":"1","temporal_scope":"pre_race","eligible":True}}
        core._write_json_atomic(self.dataset.parent/"manifest.json",{"dataset_id":"dataset-known","feature_catalog":catalog,"rows":100,"races":25,"date_min":"2000-01-01","date_max":"2020-01-01","availability_policy":"strict","target_eligibility":{"win":{"eligible_races":25,"eligible_rows":100,"contract":"win_probability","excluded_race_ids":["protected-excluded"],"labels":["protected-label"]}},"confirmation_keys":["protected-race"],"labels":["protected-label"]})
        config = replace(self.config,official_snapshot_path=snapshot)
        capabilities = _capabilities(config,self.dataset,registry=registry)
        current = capabilities["current_dataset"]
        inputs = capabilities["dataset_request_inputs"]
        self.assertTrue(inputs["ready"])
        self.assertEqual(core._hash_file(raw),inputs["raw_manifest_sha256"])
        self.assertEqual(core._hash_file(raw),inputs["raw_corpus_manifest_id"])
        self.assertEqual([identifier],inputs["known_feature_definition_ids"])
        self.assertEqual(catalog,capabilities["eligible_predictors"])
        self.assertEqual("dataset-known",current["dataset_id"])
        self.assertEqual(core._hash_file(self.dataset),current["dataset_sha256"])
        self.assertEqual(parameters,current["protocol_parameters"])
        self.assertEqual("protocol-known",current["protocol_id"])
        self.assertEqual("2000-01-01",current["date_min"])
        self.assertEqual(25,current["target_eligibility"]["win"]["eligible_races"])
        self.assertNotIn("protected-",json.dumps(capabilities))
        for identity in inputs["accepted_raw_corpus_manifest_ids"]:
            _validate_raw_manifest(DatasetRequest(request_id="request",raw_corpus_manifest_id=identity,rationale="test",evidence_watermark="evidence"),raw)
        with self.assertRaisesRegex(ValueError,"identity mismatch"):
            _validate_raw_manifest(DatasetRequest(request_id="request",raw_corpus_manifest_id="guessed",rationale="test",evidence_watermark="evidence"),raw)
        self.assertFalse((config.campaign_dir/"decisions.sqlite").exists())

    def test_capabilities_readiness_and_policy_are_not_inferred(self):
        from dataclasses import replace
        for config in (self.config,replace(self.config,official_snapshot_path=self.root/"absent")):
            with self.subTest(snapshot=config.official_snapshot_path):
                capabilities = _capabilities(config,self.dataset)
                self.assertFalse(capabilities["dataset_request_inputs"]["ready"])
                self.assertEqual([],capabilities["dataset_request_inputs"]["accepted_raw_corpus_manifest_ids"])
                self.assertIsNone(capabilities["current_dataset"]["availability_policy"])
                self.assertIsNone(capabilities["current_dataset"]["availability_tier"])
                self.assertNotIn("default_event_policy",capabilities["dataset_request_inputs"])
                self.assertIn("explicit",capabilities["dataset_request_inputs"]["event_policy_selection"])
                self.assertTrue(capabilities["dataset_request_inputs"]["retrospective_lags_required"])

    def test_final_completion_stop_waits_for_tracking_upload(self):
        from dataclasses import replace
        from ima.research_executor import RecipeExecutionResult
        config = replace(self.config,max_trials=None,mlflow_tracking_uri="http://unused",timeout_minutes=1)
        uploaded = []
        def worker(request,*args):
            core._write_json_atomic(config.campaign_dir/"STOP",{})
            return RecipeExecutionResult(1,request.attempt_id,request.proposal_id,request.trial_number,request.recipe.recipe_hash(),"win_probability","completed","fundamental_log_loss",2.,{},{},{},0.)
        def upload(config,dataset,row):
            uploaded.append(row["attempt_id"])
            return {"run_id":"verified-upload"}
        def planned(evidence,config):
            item = proposal().model_copy(update={"evidence_ids":(evidence["evidence_id"],)})
            return {"decision":{"decision_id":evidence["decision_id"],"evidence_id":evidence["evidence_id"],"trial_budget":1,"programs":[item.model_dump(mode="json")]},"usage":{"cost_status":"fixture"}}
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.plan_decision",side_effect=planned),patch("ima.research_expansion._worker",side_effect=worker),patch("ima.research_v5.upload_result",side_effect=upload),patch("ima.research_expansion.log_snapshot"):
            result = run_expansion_campaign(config)
        self.assertEqual("stopped",result["mode"],core._read_json(config.campaign_dir/"status.json"))
        self.assertEqual(1,len(uploaded))
        self.assertEqual(0,result["pending_tracking"])
        self.assertEqual(0,result["pending_tells"])

    def test_trace_outage_blocks_and_restart_drains_transient_error(self):
        from dataclasses import replace
        from ima.research_telemetry import _outbox
        config = replace(self.config,mlflow_tracking_uri="http://unused",timeout_minutes=.2)
        core._write_json_atomic(config.campaign_dir/"STOP",{})
        conn = _outbox(config.campaign_dir)
        conn.execute("INSERT INTO traces(role,number,payload,tracking_uri) VALUES(?,?,?,?)",("decision",1,"{}",config.mlflow_tracking_uri))
        conn.commit()
        conn.close()
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.log_snapshot"),patch("ima.research_telemetry._deliver_trace",side_effect=RuntimeError("offline")) as delivery:
            first = run_expansion_campaign(config)
        self.assertEqual("blocked_tracking",first["mode"])
        self.assertEqual(1,first["pending_trace_delivery"])
        self.assertEqual(3,delivery.call_count)
        calls = []
        def recover(campaign,config,role,number):
            calls.append(number)
            if len(calls)==1:
                raise RuntimeError("transient")
            conn = _outbox(campaign)
            conn.execute("UPDATE traces SET delivered=1 WHERE role=? AND number=?",(role,number))
            conn.commit()
            conn.close()
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.log_snapshot"),patch("ima.research_telemetry._deliver_trace",side_effect=recover):
            second = run_expansion_campaign(config)
        self.assertEqual("stopped",second["mode"])
        self.assertEqual(0,second["pending_trace_delivery"])
        self.assertEqual(2,len(calls))

    def test_required_lane_refill_is_independent_of_large_benter_budget(self):
        benter = proposal().recipe
        experiment = benter.model_copy(update={"model":benter.model.model_copy(update={"kind":"boosted","parameters":{"max_iter":8}})})
        programs = {"B":SimpleNamespace(recipe=benter),"E":SimpleNamespace(recipe=experiment)}
        self.assertFalse(_needs_lane_refill({"B":100},programs,0))
        self.assertTrue(_needs_lane_refill({"B":100},programs,2))
        self.assertFalse(_needs_lane_refill({"B":100,"E":1},programs,2))
        self.assertTrue(_needs_lane_refill({"B":100,"E":0},programs,2))

    def test_lane_starvation_refills_from_planner_and_preserves_four_to_one(self):
        from dataclasses import replace
        from ima.research_executor import RecipeExecutionResult
        config = replace(self.config,max_trials=None,proposal_batch_size=100,timeout_minutes=1)
        evidence_calls = []
        completed = []
        def planned(evidence,config):
            evidence_calls.append(evidence)
            item = proposal("benter",trials=100) if len(evidence_calls)==1 else proposal("experiment",trials=1)
            if len(evidence_calls)>1:
                recipe = item.recipe.model_copy(update={"model":item.recipe.model.model_copy(update={"kind":"boosted","parameters":{"max_iter":8}})})
                item = item.model_copy(update={"recipe":recipe})
            item = item.model_copy(update={"evidence_ids":(evidence["evidence_id"],)})
            return {"decision":{"decision_id":evidence["decision_id"],"evidence_id":evidence["evidence_id"],"trial_budget":item.max_trials,"programs":[item.model_dump(mode="json")]},"usage":{"cost_status":"fixture"}}
        def worker(request,*args):
            completed.append(request.attempt_id)
            if len(completed)==5:
                core._write_json_atomic(config.campaign_dir/"STOP",{})
            return RecipeExecutionResult(1,request.attempt_id,request.proposal_id,request.trial_number,request.recipe.recipe_hash(),"win_probability","completed","fundamental_log_loss",2.,{},{},{},0.)
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.plan_decision",side_effect=planned),patch("ima.research_expansion._worker",side_effect=worker),patch("ima.research_expansion.log_snapshot"):
            result = run_expansion_campaign(config)
        self.assertEqual("stopped",result["mode"])
        self.assertEqual("experimental",evidence_calls[1]["next_required_lane"])
        self.assertGreater(sum(evidence_calls[1]["remaining_program_capacity"].values()),config.queue_low_watermark)
        terminal = ResearchLedger(config.campaign_dir/"ledger.sqlite").terminal_results()
        self.assertEqual(4,sum(row["result"]["lane"]=="benter" for row in terminal))
        self.assertEqual(1,sum(row["result"]["lane"]=="experimental" for row in terminal))

    def test_pending_tells_block_finish_and_recover_on_restart(self):
        search = ProgramSearchController(self.config.campaign_dir)
        pid = search.register(proposal())
        suggestion = search.ask(1,allowed_program_ids=[pid])[0]
        ledger = ResearchLedger(self.config.campaign_dir/"ledger.sqlite")
        record = ledger.reserve_attempt("tell-retry",suggestion.serializable())
        ledger.complete_attempt(record.attempt_id,{"status":"completed","objective_name":"fundamental_log_loss","objective_value":2.,"metrics":{}},status="completed")
        core._write_json_atomic(self.config.campaign_dir/"STOP",{})
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.log_snapshot"),patch("ima.research_expansion.core._reconcile_tells",side_effect=RuntimeError("tell unavailable")):
            first = run_expansion_campaign(self.config)
        self.assertEqual("blocked_tracking",first["mode"])
        self.assertEqual(1,first["pending_tells"])
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.log_snapshot"):
            second = run_expansion_campaign(self.config)
        self.assertEqual("complete",second["mode"])
        self.assertEqual(0,second["pending_tells"])

    def test_trace_timeout_preserves_undelivered_evidence(self):
        from dataclasses import replace
        from ima.research_telemetry import _outbox
        config = replace(self.config,mlflow_tracking_uri="http://unused",timeout_minutes=.03)
        core._write_json_atomic(config.campaign_dir/"STOP",{})
        conn = _outbox(config.campaign_dir)
        conn.execute("INSERT INTO traces(role,number,payload,tracking_uri) VALUES(?,?,?,?)",("decision",1,"{}",config.mlflow_tracking_uri))
        conn.commit()
        conn.close()
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors),patch("ima.research_expansion.log_snapshot"),patch("ima.research_telemetry._deliver_trace",side_effect=RuntimeError("offline")) as delivery:
            result = run_expansion_campaign(config)
        self.assertEqual("blocked_tracking",result["mode"])
        self.assertEqual(1,result["pending_trace_delivery"])
        self.assertLessEqual(delivery.call_count,3)
