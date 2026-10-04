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
    plan_decision, run_expansion_campaign,
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
