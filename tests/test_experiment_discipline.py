import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from ima.experiment_discipline import (admit_decision, lifecycle_snapshot, planner_view,
                                      proposal_signature, validate_decision,resolve_reused_attempt,reconcile_reused_tells)
from ima.research_expansion import DecisionStore, apply_decision
from ima.research_search import ProgramSearchController
from ima.research_store import ResearchLedger
from tests.test_research_expansion import proposal, decision


def snapshot(p, pending=1):
    return {"programs": [{"program_id": "existing", "proposal_id": p.proposal_id,
        "evidence_ids": list(p.evidence_ids), "signature": proposal_signature(p),
        "pending_trials": pending}]}


class DisciplineTests(unittest.TestCase):
    def test_cached_completion_links_new_hypothesis_and_recovers_tell(self):
        from ima.research_hypotheses import HypothesisMemory
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); search=ProgramSearchController(root); ledger=ResearchLedger(root/'ledger.sqlite')
            memory=HypothesisMemory(root/'hypotheses.sqlite'); p=proposal('new')
            pid=search.register(p); suggestion=search.ask(1,allowed_program_ids=[pid])[0]
            old=ledger.reserve_attempt('cached',suggestion.serializable()|{'program_id':'old'})
            result={'objective_name':'loss','objective_value':2.,'metrics':{},'lineage':{'protocol_id':'p','dataset_hash':'d'}}
            ledger.complete_attempt(old.attempt_id,result)
            record=ledger.reserve_attempt('cached',{})
            with patch.object(search,'tell',side_effect=RuntimeError('crash before tell')):
                with self.assertRaisesRegex(RuntimeError,'crash'):
                    resolve_reused_attempt(ledger,search,memory,record,suggestion)
            self.assertEqual(search.trial_state(pid,suggestion.trial_number),'running')
            reconcile_reused_tells(root,search)
            self.assertEqual(search.trial_state(pid,suggestion.trial_number),'complete')
            resolve_reused_attempt(ledger,search,memory,record,suggestion)
            full=lifecycle_snapshot(root,search)
            self.assertEqual(full['counts']['attempts'],1)
            self.assertEqual(full['counts']['reused_observations'],1)
            self.assertEqual(full['outcomes'][-1]['proposal_id'],'new')
            self.assertEqual(ledger.terminal_results()[0]['payload']['program_id'],'old')

    def test_effective_default_search_not_confused_with_fixed_control(self):
        exploratory=proposal()
        fixed=exploratory.model_copy(update={"fixed_parameters":True})
        self.assertNotEqual(proposal_signature(exploratory),proposal_signature(fixed))
        from ima.research_search import _default_space
        explicit=exploratory.model_copy(update={"search_space":_default_space(exploratory.recipe.model.kind)})
        self.assertEqual(proposal_signature(exploratory),proposal_signature(explicit))
        overridden=exploratory.model_copy(update={"recipe":exploratory.recipe.model_copy(update={
            "model":exploratory.recipe.model.model_copy(update={"parameters":{"l2":999.}})})})
        self.assertEqual(proposal_signature(exploratory),proposal_signature(overridden))

    def test_graph_search_canonicalizes_only_the_tuned_estimator(self):
        from tests.test_proposal_audit import graph_recipe
        from ima.research_specs import SearchDimension

        recipe = graph_recipe()
        recipe["pipeline_graph"]["tune_node_id"] = "b"
        recipe["model"] = {"kind": "boosted", "parameters": {"max_iter": 20}}
        recipe["pipeline_graph"]["nodes"][0]["parameters"]["model_parameters"] = {"l2": 1.0}
        recipe["pipeline_graph"]["nodes"][1]["parameters"]["model_parameters"] = {"max_iter": 5}
        dimensions = {"max_iter": SearchDimension(kind="int", low=10, high=30)}
        first = proposal().model_copy(update={"recipe": type(proposal().recipe).model_validate(recipe),
                                             "search_space": dimensions})
        tuned_initial_differs = copy.deepcopy(recipe)
        tuned_initial_differs["pipeline_graph"]["nodes"][1]["parameters"]["model_parameters"]["max_iter"] = 99
        second = first.model_copy(update={"recipe": type(first.recipe).model_validate(tuned_initial_differs)})
        self.assertEqual(proposal_signature(first), proposal_signature(second))
        untuned_differs = copy.deepcopy(recipe)
        untuned_differs["pipeline_graph"]["nodes"][0]["parameters"]["model_parameters"]["l2"] = 2.0
        third = first.model_copy(update={"recipe": type(first.recipe).model_validate(untuned_differs)})
        self.assertNotEqual(proposal_signature(first), proposal_signature(third))

    def test_legacy_paper_completion_keeps_outcome_without_recipe_or_control_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ResearchLedger(root / "ledger.sqlite")
            legacy = ledger.reserve_attempt("paper", {"paper_only": True})
            ledger.complete_attempt(legacy.attempt_id, {"objective_name": "paper_score", "objective_value": 1.0})
            search = ProgramSearchController(root)
            config = SimpleNamespace(campaign_dir=root)
            exploratory = proposal().model_copy(update={"fixed_parameters": False,
                "parent_trial_ids": (legacy.attempt_id,)})
            receipt = admit_decision(decision(programs=(exploratory,)), None, search, set(), config,
                                     {"completed_trial_index": []})
            comparison = receipt["proposals"][0]["parent_comparisons"][0]
            self.assertEqual(comparison["recipe_comparison"], "unavailable")
            self.assertEqual(comparison["status"], "completed")
            fixed = exploratory.model_copy(update={"fixed_parameters": True})
            with self.assertRaisesRegex(ValueError, "exact parent recipe"):
                admit_decision(decision(programs=(fixed,)), None, search, set(), config,
                               {"completed_trial_index": []})

    def test_pending_duplicate_rejected_new_seed_allowed_replay_allowed(self):
        first = proposal("old")
        second = proposal("new")
        with self.assertRaisesRegex(ValueError, "Pending experiment"):
            validate_decision(decision(programs=(second,)), {}, snapshot(first))
        validate_decision(decision(programs=(first,)), {}, snapshot(first))
        replicate = second.model_copy(update={"recipe": second.recipe.model_copy(update={"seed":43})})
        validate_decision(decision(programs=(replicate,)), {}, snapshot(first))
        validate_decision(decision(programs=(second,)), {}, snapshot(first, 0))

    def test_same_decision_clones_reject(self):
        with self.assertRaisesRegex(ValueError, "Pending experiment"):
            validate_decision(decision(programs=(proposal("a"),proposal("b")),trial_budget=2), {}, {"programs":[]})

    def test_failed_retry_not_blocked_when_terminal(self):
        first = proposal("failed")
        validate_decision(decision(), {}, snapshot(first, 0))

    def test_fixed_control_undeclared_change_rejected_exploration_allowed(self):
        parent = proposal().recipe
        test = proposal().model_copy(update={"fixed_parameters":True,
            "changed_axes":("transform",), "parent_trial_ids":("parent",)})
        baseline = parent.model_copy(update={"model":parent.model.model_copy(update={"parameters":{"l2":.9}})})
        evidence = {"completed_trial_index":[{"attempt_id":"parent", "recipe":baseline.canonical_payload()}]}
        with self.assertRaisesRegex(ValueError,"Invalid fixed control"):
            validate_decision(decision(programs=(test,)),evidence,{"programs":[]})
        exploration = test.model_copy(update={"fixed_parameters":False})
        receipt = validate_decision(decision(programs=(exploration,)),evidence,{"programs":[]})
        self.assertEqual(receipt["proposals"][0]["comparison_class"],"joint_or_exploratory_test")

    def test_declared_joint_fixed_test_allowed_no_isolated_claim(self):
        test = proposal().model_copy(update={"fixed_parameters":True,"changed_axes":("hyperparameters","transform"),"parent_trial_ids":("parent",)})
        baseline = test.recipe.model_copy(update={"model":test.recipe.model.model_copy(update={"parameters":{"l2":.9}})})
        receipt = validate_decision(decision(programs=(test,)),{"completed_trial_index":[{"attempt_id":"parent","recipe":baseline.canonical_payload()}]}, {"programs":[]})
        self.assertEqual(receipt["proposals"][0]["comparison_class"],"joint_or_exploratory_test")

    def test_fixed_missing_parent_recipe_fails(self):
        test=proposal().model_copy(update={"fixed_parameters":True,"parent_trial_ids":("parent",)})
        with self.assertRaisesRegex(ValueError,"exact parent recipe"):
            validate_decision(decision(programs=(test,)),{"completed_trial_index":[{"attempt_id":"parent"}]},{"programs":[]})

    def test_complete_durable_lineage_bounded_view_and_retired_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); search=ProgramSearchController(root)
            p=proposal(trials=20); pid=search.register(p)
            store=DecisionStore(root/"decisions.sqlite"); d=decision(programs=(p,),trial_budget=20)
            store.record(d); store.receipt(d.decision_id,"program:"+pid,{"program_id":pid,"allocated":20})
            ledger=ResearchLedger(root/"ledger.sqlite")
            for i in range(15):
                row=ledger.reserve_attempt(str(i),{"program_id":pid,"proposal_id":p.proposal_id,"recipe":p.recipe.canonical_payload()})
                ledger.complete_attempt(row.attempt_id,{"objective_value":2.,"objective_name":"loss","lineage":{"protocol_id":"p","dataset_hash":"d"}})
            active=ledger.reserve_attempt("active",{"program_id":pid,"recipe":p.recipe.canonical_payload()})
            ledger.mark_running(active.attempt_id)
            full=lifecycle_snapshot(root,search,{pid})
            self.assertEqual(full["counts"]["attempts"],16)
            self.assertEqual(full["programs"][0]["pending_trials"],1)
            self.assertEqual(full["outcomes"][0]["decision_links"][0]["decision_id"],d.decision_id)
            before=copy.deepcopy(full); view=planner_view(full)
            self.assertEqual(full,before)
            self.assertEqual(len(view["recent_outcomes"]),12)
            self.assertEqual(view["outcomes_omitted"],4)
            self.assertEqual(view["champion_status"],"development_only_unconfirmed")
            json.dumps(view,allow_nan=False)

    def test_current_admission_ignores_stale_empty_pending_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); search=ProgramSearchController(root)
            search.register(proposal("pending"))
            with self.assertRaisesRegex(ValueError,"Pending experiment"):
                admit_decision(decision(),None,search,set(),SimpleNamespace(campaign_dir=root),{"completed_trial_index":[]})

    def test_exact_contract_parent_delta_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); search=ProgramSearchController(root); ledger=ResearchLedger(root/"ledger.sqlite")
            baseline=ledger.reserve_attempt("parent",{"recipe":proposal().recipe.canonical_payload()})
            result={"objective_value":2.2,"objective_name":"loss","lineage":{"protocol_id":"p","dataset_hash":"d"}}
            ledger.complete_attempt(baseline.attempt_id,result)
            p=proposal().model_copy(update={"parent_trial_ids":(baseline.attempt_id,)})
            pid=search.register(p)
            child=ledger.reserve_attempt("child",{"program_id":pid,"recipe":p.recipe.canonical_payload()})
            ledger.complete_attempt(child.attempt_id,result|{"objective_value":2.1})
            full=lifecycle_snapshot(root,search)
            delta=full["outcomes"][-1]["matched_parent_deltas"][0]
            self.assertAlmostEqual(delta["objective_delta"],-.1)
            self.assertEqual(delta["evidence"],"matched_development_delta_only")
            self.assertEqual(full["confirmation_queue"][0]["attempt_id"],child.attempt_id)
            self.assertFalse(full["confirmation_queue"][0]["promotion_allowed"])

if __name__ == "__main__":
    unittest.main()
