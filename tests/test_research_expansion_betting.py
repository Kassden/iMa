import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ima.research_expansion import (DecisionStore, PlannerDecision, _paper_worker,
    _publish_paper_report, _recover_paper_report, _paper_report_path, _paper_evidence,
    _preflight_betting_requests, _paper_estimate, _paper_workload)
from ima.research_resources import JobWorkload, estimate_job
from ima.research_scheduler import ResourceAdmission, ResourceRequest


class PaperReceiptTests(unittest.TestCase):
    def test_zero_trial_paper_decision_and_independent_request_ceiling(self):
        request = {"request_id":"paper-1","evidence_id":"E1","attempt_ids":["a1"]}
        decision = PlannerDecision(decision_id="D1",evidence_id="E1",trial_budget=0,
                                   betting_requests=[request])
        self.assertEqual(0,decision.trial_budget)
        self.assertEqual("paper-1",decision.betting_requests[0].request_id)
        for requests in ([request,request], [dict(request,request_id=f"paper-{i}") for i in range(5)]):
            with self.assertRaises(ValueError):
                PlannerDecision(decision_id="D1",evidence_id="E1",trial_budget=0,betting_requests=requests)

    def test_only_fully_applied_actions_are_dispatchable_and_completion_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DecisionStore(Path(directory)/"decisions.sqlite")
            decision = PlannerDecision(decision_id="D1",evidence_id="E1",trial_budget=0,review_reason="paper review")
            store.record(decision)
            receipt = {"action_id":"paper-1","request":{"request_id":"request-1"}}
            store.receipt("D1","betting:paper-1",receipt)
            self.assertEqual([],store.betting_actions())
            store.complete("D1")
            self.assertEqual(1,len(store.betting_actions(pending_only=True)))
            completed = {"status":"completed","report_hash":"hash-1"}
            store.receipt("D1","betting-complete:paper-1",completed)
            store.receipt("D1","betting-complete:paper-1",completed)
            self.assertEqual([],store.betting_actions(pending_only=True))
            self.assertEqual(completed,store.betting_actions()[0]["completion"])
            with self.assertRaisesRegex(ValueError,"Conflicting action replay"):
                store.receipt("D1","betting-complete:paper-1",{"status":"failed"})

    def test_simulation_reserves_cpu_and_ram_without_consuming_fit_cap(self):
        workload = JobWorkload(stage="fit",family="benter",rows=100,generated_features=0,
            selected_features=3,implementation_revision="fixture",dependency_versions={"python":"fixture"})
        fit = estimate_job(workload)
        simulation = replace(fit,stage="simulation",family="paper")
        admission = ResourceAdmission(3,3,8,max_fits=1)
        admission.reserve("fit",fit)
        self.assertTrue(admission.admits(simulation))
        admission.reserve("paper",simulation)
        self.assertEqual(2,len(admission.active))
        self.assertIn("max_fits",admission.blockers(fit))
        self.assertIn("cpu_threads",admission.blockers(replace(simulation,cpu_threads=2)))

    def test_legacy_requests_and_graph_components_remain_fits(self):
        workload = JobWorkload(stage="graph_component",family="graph",rows=100,generated_features=0,
            selected_features=3,implementation_revision="fixture",dependency_versions={"python":"fixture"})
        admission = ResourceAdmission(3,3,16,max_fits=1)
        admission.reserve("legacy",ResourceRequest())
        self.assertIn("max_fits",admission.blockers(estimate_job(workload)))


class PaperPublicationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_research_betting import PaperResearchTests
        self.fixture = PaperResearchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.campaign
        self.store = DecisionStore(self.directory/"decisions.sqlite")
        self.decision = PlannerDecision(decision_id="D1",evidence_id="e1",trial_budget=0,
                                       betting_requests=[self.fixture.request])
        self.store.record(self.decision)
        self.outcome = _paper_worker(self.fixture.request,self.directory,self.fixture.rows)
        self.assertTrue(self.outcome["success"],self.outcome)
        report = self.outcome["report"]
        self.action = {"action_id":"a"*24,"decision_id":"D1","request":report["request"],
                       "preflight":{key:report[key] for key in ("comparison","evaluation_key","model_ids")}}
        self.store.receipt("D1","betting:"+self.action["action_id"],self.action)
        self.store.complete("D1")

    def publish(self):
        return _publish_paper_report(self.store,self.directory,self.action,
                                    self.outcome["report"],self.outcome["worker_resources"])

    def test_real_engine_report_is_published_once_with_process_resources(self):
        first = self.publish()
        original = _paper_report_path(self.directory,self.action["action_id"]).read_bytes()
        self.assertEqual(first,self.publish())
        self.assertEqual(original,_paper_report_path(self.directory,self.action["action_id"]).read_bytes())
        self.assertEqual([],self.store.betting_actions(pending_only=True))
        self.assertTrue(self.outcome["worker_resources"])

    def test_crash_after_atomic_report_before_receipt_recovers_without_simulation(self):
        with patch.object(self.store,"receipt",side_effect=OSError("receipt unavailable")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(1,len(self.store.betting_actions(pending_only=True)))
        restarted = DecisionStore(self.directory/"decisions.sqlite")
        with patch("ima.research_betting.evaluate_paper_research",side_effect=AssertionError("must not rerun")):
            completion = _recover_paper_report(restarted,self.directory,self.action)
        self.assertEqual("completed",completion["status"])
        self.assertEqual([],restarted.betting_actions(pending_only=True))

    def test_changed_inputs_or_executable_claim_cannot_publish_completion(self):
        import copy
        for field,value in (("paper_only",False),("executable_evidence",True),("model_ids",[]),
                            ("coverage",{"evaluated_races":0})):
            report = copy.deepcopy(self.outcome["report"])
            report[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                _publish_paper_report(self.store,self.directory,self.action,report)
        self.assertEqual(1,len(self.store.betting_actions(pending_only=True)))

    def test_feedback_only_current_compatible_population_and_no_fake_profit(self):
        self.publish()
        evidence = _paper_evidence(self.store,self.directory,self.fixture.rows)
        summary = next(iter(evidence["by_comparison"].values()))[0]
        self.assertEqual(self.outcome["report"]["request"],summary["request"])
        self.assertFalse(summary["executable_evidence"])
        self.assertIsNone(summary["race_results_preview"][0]["reports"][0]["expected_profit"])
        self.assertEqual({},_paper_evidence(self.store,self.directory,[])["by_comparison"])

    def test_paper_eligibility_excludes_legacy_predictions_without_completion_hash(self):
        import copy
        rows = copy.deepcopy(self.fixture.rows)
        rows[0]["result"]["lineage"].pop("prediction_sha256")
        eligible = _paper_evidence(self.store,self.directory,rows)["eligible_attempts"]
        self.assertEqual(["a2"],[item["attempt_id"] for item in eligible])

    def test_all_requests_preflight_before_any_receipt_and_historical_ids_rejected(self):
        from types import SimpleNamespace
        config = SimpleNamespace(campaign_dir=self.directory)
        request = dict(self.fixture.request,request_id="new-1")
        second = dict(request,request_id="new-2",attempt_ids=["historical-reference"])
        decision = PlannerDecision(decision_id="D2",evidence_id="e1",trial_budget=0,
                                   betting_requests=[request,second])
        evidence = {"completed_trial_index":[{"attempt_id":"a1"}]}
        with patch("ima.research_expansion._current_paper_terminal",return_value=self.fixture.rows), \
             patch("ima.research_betting.validate_paper_research",return_value=self.action["preflight"],create=True) as validate:
            with self.assertRaisesRegex(ValueError,"current completed"):
                _preflight_betting_requests(decision,self.store,config,evidence)
        self.assertEqual(1,validate.call_count)
        self.assertIsNone(self.store.action("D2","betting:"+"a"*24))
        self.assertEqual([],self.store.pending())

    def test_real_preflight_agrees_with_worker_without_running_simulation(self):
        from ima.research_betting import validate_paper_research
        with patch("ima.research_betting._race_report",side_effect=AssertionError("preflight must not simulate")):
            validated = validate_paper_research(self.fixture.request,campaign_dir=self.directory,
                                              terminal_results=self.fixture.rows)
        self.assertEqual(4,validated["rows"])
        for key in ("comparison","evaluation_key","model_ids"):
            self.assertEqual(self.outcome["report"][key],validated[key])

    def test_paper_estimate_uses_deep_input_memory_and_admission_rejects_insufficient_ram(self):
        from ima.research_resources import JobEstimator
        action = dict(self.action,preflight=dict(self.action["preflight"],rows=4,
                      input_memory_bytes=3*1024**3,cold_private_memory_bytes=7*1024**3))
        workload = _paper_workload(action,"fixture","environment")
        estimator = JobEstimator(self.directory/"estimates.json")
        with patch.object(estimator,"estimate",wraps=estimator.estimate) as estimate:
            request = _paper_estimate(estimator,workload,action)
        self.assertEqual(7*1024**3,estimate.call_args.kwargs["cold_private_bytes"])
        self.assertGreaterEqual(request.private_peak_bytes,7*1024**3)
        self.assertIn("configured_memory",ResourceAdmission(1,1,4).blockers(request))

    def test_older_receipt_fallback_preserves_explicit_512mib_default(self):
        from ima.research_resources import JobEstimator
        action = dict(self.action,preflight=dict(self.action["preflight"],rows=4))
        workload = _paper_workload(action,"fixture","environment")
        estimator = JobEstimator(self.directory/"estimates.json")
        with patch.object(estimator,"estimate",wraps=estimator.estimate) as estimate:
            request = _paper_estimate(estimator,workload,action)
        self.assertEqual(512*1024**2,estimate.call_args.kwargs["cold_private_bytes"])
        self.assertGreaterEqual(request.private_peak_bytes,512*1024**2)



class PaperControllerDrainTests(unittest.TestCase):
    def setUp(self):
        from tests.test_research_expansion_resources import ExpansionResourceIntegrationTests
        from tests.test_research_betting import PaperResearchTests
        from ima import research_controller as core
        from ima.research_store import ResearchLedger
        self.runtime = ExpansionResourceIntegrationTests()
        self.runtime.setUp()
        self.addCleanup(self.runtime.doCleanups)
        self.config = replace(self.runtime.config,max_trials=None,mlflow_tracking_uri=None,timeout_minutes=.2)
        fixture = PaperResearchTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.campaign = self.config.campaign_dir
        ledger = ResearchLedger(self.config.campaign_dir/"ledger.sqlite")
        record = ledger.reserve_attempt("paper-parent",{"program_id":"fixture-parent","trial_number":0})
        row = fixture.attempt(record.attempt_id)
        ledger.complete_attempt(record.attempt_id,row["result"])
        ledger.mark_told(record.attempt_id)
        ledger.mark_uploaded(record.attempt_id,"fixture")
        self.request = dict(fixture.request,attempt_ids=[record.attempt_id])
        report = _paper_worker(self.request,self.config.campaign_dir,[row])["report"]
        self.store = DecisionStore(self.config.campaign_dir/"decisions.sqlite")
        decision = PlannerDecision(decision_id="D1",evidence_id="e1",trial_budget=0,
                                   betting_requests=[self.request])
        self.store.record(decision)
        self.action = {"action_id":"b"*24,"request":report["request"],"trace_number":1,
                       "preflight":{key:report[key] for key in ("comparison","evaluation_key","model_ids")}}
        self.store.receipt("D1","betting:"+self.action["action_id"],self.action)
        self.store.complete("D1")
        core._write_json_atomic(self.config.campaign_dir/"STOP",{})

    def test_real_zero_trial_preflight_persists_before_dispatch_and_replay_is_stable(self):
        from ima.research_expansion import apply_decision
        from ima.research_search import ProgramSearchController
        search = ProgramSearchController(self.config.campaign_dir)
        request = dict(self.request,request_id="additional-paper")
        decision = PlannerDecision(decision_id="D2",evidence_id="e1",trial_budget=0,
                                   betting_requests=[request])
        evidence = {"evidence_id":"e1","completed_trial_index":[{"attempt_id":request["attempt_ids"][0]}]}
        with patch("ima.research_betting._race_report",side_effect=AssertionError("no simulation in apply")):
            self.assertEqual([],apply_decision(decision,self.store,search,set(),self.config,evidence,None))
            self.assertEqual([],apply_decision(decision,self.store,search,set(),self.config,evidence,None))
        self.assertEqual(2,len(self.store.betting_actions(pending_only=True)))
        self.assertEqual({},self.store.authorized_budgets())

    def test_invalid_batch_changes_no_search_or_decision_state(self):
        from ima.research_expansion import apply_decision
        from ima.research_search import ProgramSearchController
        search = ProgramSearchController(self.config.campaign_dir)
        first = dict(self.request,request_id="additional-paper")
        second = dict(first,request_id="bad-paper",attempt_ids=["historical-reference"])
        decision = PlannerDecision(decision_id="D2",evidence_id="e1",trial_budget=0,
                                   betting_requests=[first,second])
        evidence = {"evidence_id":"e1","completed_trial_index":[{"attempt_id":first["attempt_ids"][0]}]}
        with self.assertRaisesRegex(ValueError,"current completed"):
            apply_decision(decision,self.store,search,set(),self.config,evidence,None)
        self.assertEqual([],self.store.pending())
        self.assertEqual(1,len(self.store.betting_actions(pending_only=True)))
        self.assertEqual({},search.programs)

    def test_independent_queue_ceiling_and_request_id_binding(self):
        from ima.research_expansion import apply_decision
        from ima.research_search import ProgramSearchController
        search = ProgramSearchController(self.config.campaign_dir)
        evidence = {"evidence_id":"e1","completed_trial_index":[{"attempt_id":self.request["attempt_ids"][0]}]}
        requests = [dict(self.request,request_id=f"additional-{i}") for i in range(4)]
        decision = PlannerDecision(decision_id="D2",evidence_id="e1",trial_budget=0,betting_requests=requests)
        apply_decision(decision,self.store,search,set(),self.config,evidence,None)
        excessive = PlannerDecision(decision_id="D3",evidence_id="e1",trial_budget=0,
            betting_requests=[dict(self.request,request_id=f"overflow-{i}") for i in range(4)])
        with self.assertRaisesRegex(ValueError,"eight-action ceiling"):
            apply_decision(excessive,self.store,search,set(),self.config,evidence,None)
        rebound = PlannerDecision(decision_id="D4",evidence_id="e1",trial_budget=0,
                                  betting_requests=[requests[0]])
        with self.assertRaisesRegex(ValueError,"already bound"):
            apply_decision(rebound,self.store,search,set(),self.config,evidence,None)
        self.assertEqual(5,len(self.store.betting_actions(pending_only=True)))

    def run_controller(self,trace=None,worker=None):
        from ima.research_expansion import run_expansion_campaign
        from tests.test_research_expansion_resources import executors
        with patch("ima.research_expansion.ProcessPoolExecutor",side_effect=executors), \
             patch("ima.research_expansion.log_snapshot",side_effect=trace,return_value=None), \
             patch("ima.research_expansion.plan_decision",side_effect=AssertionError("no paid planner")), \
             patch("ima.research_expansion._worker",side_effect=AssertionError("no model fits")), \
             patch("ima.research_expansion._paper_worker",side_effect=worker or _paper_worker) as paper:
            return run_expansion_campaign(self.config),paper.call_count

    def test_stop_drains_authorized_paper_once_and_restart_does_not_rerun(self):
        result,calls = self.run_controller()
        self.assertEqual("stopped",result["mode"])
        self.assertEqual(0,result["pending_paper_actions"])
        self.assertEqual(1,calls)
        second,calls = self.run_controller()
        self.assertEqual("stopped",second["mode"])
        self.assertEqual(0,calls)

    def test_trace_enqueue_failure_is_bounded_blocked_and_restart_recovers(self):
        result,calls = self.run_controller(trace=RuntimeError("trace enqueue unavailable"))
        self.assertEqual("blocked_tracking",result["mode"])
        self.assertEqual(1,result["pending_paper_trace_publication"])
        self.assertEqual(1,calls)
        second,calls = self.run_controller()
        self.assertEqual("stopped",second["mode"])
        self.assertEqual(0,second["pending_paper_trace_publication"])
        self.assertEqual(0,calls)

    def test_transient_paper_failure_retries_without_new_trial_or_paid_call(self):
        attempts = []
        def worker(*args):
            attempts.append(1)
            return {"success":False,"error":"transient fixture"} if len(attempts)==1 else _paper_worker(*args)
        result,calls = self.run_controller(worker=worker)
        self.assertEqual("stopped",result["mode"])
        self.assertEqual(2,calls)
        self.assertIsNotNone(self.store.action("D1","betting-attempt:"+self.action["action_id"]+":2"))

    def test_permanent_failure_stops_after_three_durable_attempts(self):
        result,calls = self.run_controller(worker=lambda *args:{"success":False,"error":"unsupported fixture"})
        self.assertEqual("blocked_paper",result["mode"])
        self.assertEqual(3,calls)
        self.assertEqual(0,result["pending_paper_actions"])
        completion = self.store.action("D1","betting-complete:"+self.action["action_id"])
        self.assertEqual("failed",completion["status"])
        second,calls = self.run_controller()
        self.assertEqual("blocked_paper",second["mode"])
        self.assertEqual(0,calls)


class PlannerTransportIntegrationTests(unittest.TestCase):
    def test_bounded_delegate_preserves_header_identity_without_secrets_or_billing_guess(self):
        from ima.openrouter_orchestrator import OpenRouterConfig, _post_json
        from tests.test_openrouter_transport import endpoint
        events,responses = [],[]
        config = OpenRouterConfig(api_key="secret",model="fixture",absolute_deadline_seconds=5,
                                  transport_observer=events.append,response_observer=responses.append)
        with endpoint() as (url,requests):
            result = _post_json(url,{"model":"fixture"},config)
        self.assertEqual([result],responses)
        self.assertEqual(1,len(requests))
        self.assertEqual("gen-test-header",next(row for row in events if row["phase"]=="response_headers")["generation_id"])
        self.assertNotIn("secret",json.dumps(events))


if __name__ == "__main__":
    unittest.main()
