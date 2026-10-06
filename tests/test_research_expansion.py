import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ima.research_controller import _portfolio_identity, _portfolio_slot
from ima.research_evaluation import ResearchEvaluationError, build_expanding_folds, make_protocol_manifest
from ima.research_expansion import DecisionStore, PlannerDecision, apply_decision
from ima.research_model_package import _validate_complete_races
from ima.research_search import ProgramSearchController
from ima.research_specs import PipelineRecipe, ResearchProposal, V5_PORTFOLIO_VERSION
from ima.research_targets import TargetContractError, apply_target_contract, target_contract


def proposal(identifier="probe", trials=1, *, schema_version=3):
    return ResearchProposal(
        proposal_id=identifier,
        evidence_ids=("evidence-1",),
        hypothesis="Measure a bounded change in regularization.",
        changed_axes=("hyperparameters",),
        recipe=PipelineRecipe(
            schema_version=schema_version,
            model={"kind": "benter_conditional_logit", "parameters": {"l2": 0.1}},
            blend={"kind": "none"},
        ),
        expected_observation="Lower development loss.",
        falsification_rule="Loss does not improve on matched races.",
        max_trials=trials,
    )


class BoundedFitsThreadPool(ThreadPoolExecutor):
    """Run mocked fits in threads without enforcing runtime deadlines."""

    def __init__(self, max_workers, *, max_tasks_per_child=1):
        super().__init__(max_workers=max_workers)

    def submit(self, function, *args, runtime_path, timeout, **kwargs):
        return super().submit(function, *args, **kwargs)


def decision(**changes):
    payload = {
        "decision_id": "decision-1",
        "evidence_id": "evidence-1",
        "trial_budget": 1,
        "programs": (proposal(),),
        "extensions": {},
        "retire_program_ids": (),
        "dataset_requests": (),
    }
    return PlannerDecision.model_validate(payload | changes)


class CampaignResourceContractTests(unittest.TestCase):
    def test_v6_resource_preflight_rejects_unsafe_boundaries_without_changing_legacy(self):
        from ima.optimizer import CampaignConfig

        defaults = {"campaign_dir": Path("unused-config-only"), "policy": "agentic",
                    "research_policy": "expansion_v6", "ram_budget_gib": 120,
                    "memory_budget_gb_decimal": 100, "cpu_thread_budget": 1,
                    "host_reserve_cpu_threads": 0, "host_reserve_ram_gib": 0}
        decimal_cap_gib = 100 * 1e9 / 1024**3
        CampaignConfig(**defaults).validate()
        CampaignConfig(**(defaults | {"host_reserve_ram_gib": np.nextafter(decimal_cap_gib, 0)})).validate()
        invalid = (
            {"host_reserve_ram_gib": decimal_cap_gib},
            {"host_reserve_ram_gib": decimal_cap_gib + 1},
            {"ram_budget_gib": 8, "host_reserve_ram_gib": 8},
            {"host_reserve_ram_gib": -1},
            {"host_reserve_ram_gib": float("nan")},
            {"memory_budget_gb_decimal": 101},
            {"memory_budget_gb_decimal": 0},
            {"cpu_thread_budget": 0},
            {"cpu_thread_budget": 0.5},
            {"host_reserve_cpu_threads": -1},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CampaignConfig(**(defaults | changes)).validate()
        CampaignConfig(**(defaults | {"policy": "local", "research_policy": "legacy",
                                      "ram_budget_gib": 8, "host_reserve_ram_gib": 8,
                                      "memory_budget_gb_decimal": 101})).validate()


class PlannerAPITests(unittest.TestCase):
    def test_one_valid_call_returns_all_independent_allocations_and_reported_usage(self):
        from ima.openrouter_orchestrator import OpenRouterConfig, choose_research_decision

        item = decision(trial_budget=37, programs=(proposal("first", 24), proposal("second", 9)),
                        extensions={"existing": 2}, unallocated_trials=2,
                        unallocated_reason="Preparation capacity reserved for review.")
        evidence = {"decision_id": item.decision_id, "evidence_id": item.evidence_id,
                    "workers_running": 8, "terminal_watermark": 19}
        limits = {"trial_ceiling": 260, "max_new_programs": 12, "max_pending_programs": 48}
        response = {"choices": [{"message": {"content": item.model_dump_json()}}],
                    "usage": {"prompt_tokens": 101, "completion_tokens": 23,
                              "total_tokens": 124, "cost": 0.012}}
        with patch("ima.openrouter_orchestrator._post_json", return_value=response) as post:
            actual = choose_research_decision(evidence, limits, OpenRouterConfig(api_key="test-only", model="test"))
        post.assert_called_once()
        submitted = json.loads(post.call_args.args[1]["messages"][1]["content"])
        self.assertEqual(submitted["evidence"], evidence)
        self.assertEqual(submitted["limits"], limits)
        self.assertEqual(PlannerDecision.model_validate(actual["decision"]), item)
        self.assertEqual(actual["repair_count"], 0)
        self.assertEqual(actual["usage"]["input_tokens"], 101)
        self.assertEqual(actual["usage"]["output_tokens"], 23)
        self.assertEqual(actual["usage"]["total_cost_usd"], 0.012)

    def test_validation_repair_reports_every_paid_call_not_only_accepted_response(self):
        from ima.openrouter_orchestrator import OpenRouterConfig, choose_research_decision

        item = decision()
        evidence = {"decision_id": item.decision_id, "evidence_id": item.evidence_id}
        invalid = item.model_dump(mode="json") | {"trial_budget": 2}
        responses = [{"choices": [{"message": {"content": json.dumps(payload)}}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                "total_tokens": 12, "cost": 0.003}}
                     for payload in (invalid, item.model_dump(mode="json"))]
        with patch("ima.openrouter_orchestrator._post_json", side_effect=responses) as post:
            actual = choose_research_decision(evidence, {"trial_ceiling": 260, "max_new_programs": 12},
                                             OpenRouterConfig(api_key="test-only", model="test"))
        self.assertEqual(post.call_count, 2)
        self.assertEqual(actual["repair_count"], 1)
        self.assertEqual(actual["usage"]["total_tokens"], 24)
        self.assertEqual(actual["usage"]["total_cost_usd"], 0.006)


class PlannerDecisionTests(unittest.TestCase):
    def test_budget_reconciles_new_extensions_and_explicit_remainder(self):
        item = decision(
            trial_budget=40,
            programs=(proposal("first", 24), proposal("second", 9)),
            extensions={"existing": 5},
            unallocated_trials=2,
            unallocated_reason="Preparation backlog is at its safety limit.",
        )
        self.assertEqual(
            item.trial_budget,
            sum(p.max_trials for p in item.programs)
            + sum(item.extensions.values())
            + item.unallocated_trials,
        )

    def test_underallocation_and_overallocation_are_rejected(self):
        for budget in (0, 2):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                decision(trial_budget=budget)

    def test_zero_budget_review_is_valid_without_new_programs(self):
        item = decision(
            trial_budget=0, programs=(), review_reason="Continue the occupied fits."
        )
        self.assertEqual(item.programs, ())
        self.assertEqual(item.trial_budget, 0)

    def test_no_op_requires_nonblank_review_reason(self):
        for reason in (None, "", "   "):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                decision(trial_budget=0, programs=(), review_reason=reason)

    def test_retirement_only_is_an_action_without_trial_allocation(self):
        item = decision(
            trial_budget=0, programs=(), retire_program_ids=("existing",)
        )
        self.assertEqual(item.retire_program_ids, ("existing",))

    def test_dataset_preparation_does_not_consume_outer_trial_budget(self):
        item = decision(
            trial_budget=0,
            programs=(),
            dataset_requests=({"request_id": "dataset-1", "parent_dataset_id": "base"},),
        )
        self.assertEqual(item.dataset_requests[0]["request_id"], "dataset-1")

    def test_single_program_can_receive_budget_unrelated_to_worker_count(self):
        item = decision(trial_budget=137, programs=(proposal(trials=137),))
        self.assertEqual(len(item.programs), 1)
        self.assertEqual(item.programs[0].max_trials, 137)

    def test_invalid_budget_types_and_negative_counts_are_rejected(self):
        for budget in (True, 1.5, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                decision(trial_budget=budget)
        for count in (True, 1.5, -1, 0):
            with self.subTest(extension=count), self.assertRaises(ValueError):
                decision(trial_budget=2, extensions={"existing": count})

    def test_negative_unallocated_count_cannot_cancel_overspending(self):
        with self.assertRaises(ValueError):
            decision(
                trial_budget=1,
                programs=(proposal(trials=2),),
                unallocated_trials=-1,
                unallocated_reason="Invalid negative remainder.",
            )

    def test_positive_remainder_requires_reason(self):
        for reason in (None, "", "   "):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                decision(trial_budget=2, unallocated_trials=1, unallocated_reason=reason)

    def test_unknown_decision_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            decision(unbounded_execution=True)

    def test_duplicate_proposals_and_conflicting_program_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            decision(trial_budget=2, programs=(proposal(), proposal()))
        with self.assertRaises(ValueError):
            decision(
                trial_budget=2,
                extensions={"existing": 1},
                retire_program_ids=("existing",),
            )


class DecisionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "decisions.sqlite"
        self.store = DecisionStore(self.path)

    def test_record_returns_new_then_duplicate_and_survives_reopen(self):
        item = decision()
        self.assertIs(self.store.record(item), True)
        self.assertIs(self.store.record(item), False)
        reopened = DecisionStore(self.path)
        self.assertIs(reopened.record(item), False)
        pending = reopened.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].model_dump(mode="json"), item.model_dump(mode="json"))

    def test_conflicting_replay_preserves_original_after_restart(self):
        original = decision()
        self.store.record(original)
        changed = decision(trial_budget=2, programs=(proposal(trials=2),))
        with self.assertRaises(ValueError):
            DecisionStore(self.path).record(changed)
        self.assertEqual(
            DecisionStore(self.path).pending()[0].model_dump(mode="json"),
            original.model_dump(mode="json"),
        )

    def test_conflicting_review_or_evidence_is_not_silently_overwritten(self):
        self.store.record(decision())
        for changes in (
            {"review_reason": "A different assessment."},
            {"evidence_id": "evidence-2"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.record(decision(**changes))

    def test_ceiling_rejection_does_not_enqueue_work(self):
        with self.assertRaises(ValueError):
            self.store.record(decision(trial_budget=261, programs=(proposal(trials=261),)))
        self.assertEqual(self.store.pending(), [])
        self.assertIs(
            self.store.record(decision(trial_budget=260, programs=(proposal(trials=260),))),
            True,
        )

    def test_operator_ceiling_is_independent_of_program_count(self):
        item = decision(trial_budget=7, programs=(proposal(trials=7),))
        with self.assertRaises(ValueError):
            self.store.record(item, ceiling=6)
        self.assertIs(self.store.record(item, ceiling=7), True)

    def test_applied_decisions_leave_pending_queue_durably(self):
        first = decision()
        second = decision(decision_id="decision-2")
        self.store.record(first)
        self.store.record(second)
        self.store.complete(first.decision_id)
        reopened = DecisionStore(self.path)
        self.assertEqual([item.decision_id for item in reopened.pending()], ["decision-2"])
        self.assertIs(reopened.record(first), False)
        self.assertEqual([item.decision_id for item in reopened.pending()], ["decision-2"])

    def test_action_receipts_are_durable_idempotent_and_conflict_checked(self):
        self.store.record(decision())
        payload = {"program_id": "existing", "budget": 11, "allocated": 3}
        self.assertIsNone(self.store.action("decision-1", "extend:existing"))
        self.assertEqual(self.store.receipt("decision-1", "extend:existing", payload), payload)
        reopened = DecisionStore(self.path)
        self.assertEqual(reopened.action("decision-1", "extend:existing"), payload)
        self.assertEqual(reopened.receipt("decision-1", "extend:existing", payload), payload)
        with self.assertRaises(ValueError):
            reopened.receipt("decision-1", "extend:existing", payload | {"budget": 14})
        self.assertEqual(DecisionStore(self.path).action("decision-1", "extend:existing"), payload)

    def test_partial_action_receipts_do_not_hide_unfinished_decision(self):
        item = decision()
        self.store.record(item)
        self.store.receipt(item.decision_id, "program:probe", {"program_id": "probe"})
        reopened = DecisionStore(self.path)
        self.assertEqual([item.decision_id for item in reopened.pending()], ["decision-1"])
        self.assertEqual(reopened.action(item.decision_id, "program:probe"), {"program_id": "probe"})

    def test_unknown_decision_cannot_receive_an_action_receipt(self):
        with self.assertRaises(ValueError):
            self.store.receipt("unknown", "extend:existing", {"budget": 3})
        self.assertIsNone(self.store.action("unknown", "extend:existing"))

    def test_unknown_decision_cannot_be_marked_complete(self):
        with self.assertRaises(ValueError):
            self.store.complete("unknown")


class PortfolioCompatibilityTests(unittest.TestCase):
    def graph_recipe(self, *, experimental=False):
        from ima.pipeline_graph import PipelineGraph

        nodes = [{
            "node_id": "benter", "kind": "estimator",
            "parameters": {"model_kind": "benter_conditional_logit"},
        }]
        if experimental:
            nodes.extend((
                {"node_id": "boosted", "kind": "estimator", "parameters": {"model_kind": "boosted"}},
                {"node_id": "pool", "kind": "weighted_probability_pool", "inputs": ("benter", "boosted")},
            ))
            output = "pool"
        else:
            nodes.append({"node_id": "calibration", "kind": "calibrate", "inputs": ("benter",)})
            output = "calibration"
        graph = {
            "graph_id": "classification-fixture",
            "nodes": nodes,
            "output_node_id": output,
            "primary_node_id": "benter",
        }
        PipelineGraph.from_dict(graph).validate()
        return PipelineRecipe(
            schema_version=3,
            model={"kind": "benter_conditional_logit"},
            pipeline_graph=graph,
            blend={"kind": "none"},
        )

    def test_legacy_benter_identity_is_unchanged(self):
        recipe = proposal(schema_version=2).recipe
        self.assertEqual(
            _portfolio_identity(recipe, "discovery_v5"),
            {"lane": "benter", "experiment_id": "B", "portfolio_version": V5_PORTFOLIO_VERSION},
        )

    def test_v5_rejects_speed_but_v6_admits_it_as_experimental(self):
        recipe = PipelineRecipe(
            target={"kind": "adjusted_finish_time_or_speed"},
            model={"kind": "ridge_regressor"},
            calibration={"kind": "none"},
            blend={"kind": "none"},
        )
        with self.assertRaises(ValueError):
            _portfolio_identity(recipe, "discovery_v5")
        self.assertEqual(_portfolio_identity(recipe, "expansion_v6")["lane"], "experimental")

    def test_legacy_lane_sequence_and_eighty_twenty_allocation_are_unchanged(self):
        slots = [_portfolio_slot(i, "discovery_v5") for i in range(20)]
        self.assertEqual(
            [slot for slot in slots if slot != "B"], ["E1", "E2", "E3", "E4"]
        )
        self.assertEqual(slots.count("B"), 16)

    def test_benter_only_graph_preserves_benter_lane(self):
        self.assertEqual(
            _portfolio_identity(self.graph_recipe(), "expansion_v6")["lane"], "benter"
        )

    def test_experimental_predictive_ancestor_classifies_graph_as_experimental(self):
        self.assertEqual(
            _portfolio_identity(self.graph_recipe(experimental=True), "expansion_v6")["lane"],
            "experimental",
        )

    def test_legacy_recipe_hash_matches_verified_git_baseline(self):
        # 6ef7481 canonical_payload excludes None, including feature_discovery.
        legacy = PipelineRecipe(schema_version=2)
        self.assertEqual(legacy.recipe_hash(), "3e56d8c782e0f515")
        self.assertNotIn("feature_discovery", legacy.canonical_payload())
        self.assertNotIn("pipeline_graph", legacy.canonical_payload())


class DecisionApplicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = DecisionStore(self.root / "decisions.sqlite")
        self.search = ProgramSearchController(self.root)
        self.retired = set()
        self.config = SimpleNamespace(
            campaign_dir=self.root,
            proposal_batch_size=260,
            max_new_programs_per_decision=8,
            max_pending_programs=12,
        )
        self.evidence = {"evidence_id": "evidence-1", "completed_trial_index": []}

    def apply(self, item):
        return apply_decision(
            item, self.store, self.search, self.retired, self.config, self.evidence, None
        )

    def v6_proposal(self, identifier="v6-probe", trials=1):
        item = proposal(identifier, trials)
        return item.model_copy(update={
            "recipe": item.recipe.model_copy(update={"schema_version": 3})
        })

    def test_new_program_application_and_replay_use_one_study(self):
        item = decision(programs=(self.v6_proposal(),))
        accepted = self.apply(item)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(self.apply(item), accepted)
        reopened = ProgramSearchController(self.root)
        self.assertEqual(list(reopened.programs), accepted)
        self.assertEqual(reopened.programs[accepted[0]].max_trials, 1)
        self.assertEqual(self.store.pending(), [])

    def test_extension_replay_after_crash_does_not_allocate_twice(self):
        pid = self.search.register(self.v6_proposal(trials=4))
        item = decision(trial_budget=3, programs=(), extensions={pid: 3})
        set_budget = self.search.set_budget

        def crash_after_budget_write(program_id, budget):
            set_budget(program_id, budget)
            raise RuntimeError("Injected crash after budget publication")

        with patch.object(self.search, "set_budget", side_effect=crash_after_budget_write):
            with self.assertRaisesRegex(RuntimeError, "Injected crash"):
                self.apply(item)
        self.assertEqual(self.search.programs[pid].max_trials, 7)
        self.assertEqual(len(self.store.pending()), 1)
        self.store = DecisionStore(self.root / "decisions.sqlite")
        self.search = ProgramSearchController(self.root)
        self.apply(self.store.pending()[0])
        self.apply(item)
        self.assertEqual(self.search.programs[pid].max_trials, 7)
        self.assertEqual(ProgramSearchController(self.root).programs[pid].max_trials, 7)
        self.assertEqual(self.store.pending(), [])

    def test_stale_evidence_and_unknown_program_reject_before_mutation(self):
        for item in (
            decision(evidence_id="old-evidence", programs=(self.v6_proposal(),)),
            decision(trial_budget=1, programs=(), extensions={"unknown-program": 1}),
        ):
            with self.subTest(decision=item.decision_id), self.assertRaises(ValueError):
                self.apply(item)
            self.assertEqual(self.store.pending(), [])
            self.assertEqual(self.search.programs, {})

    def test_retirement_durability_preserves_program_evidence(self):
        pid = self.search.register(self.v6_proposal())
        item = decision(trial_budget=0, programs=(), retire_program_ids=(pid,))
        self.apply(item)
        self.assertIn(pid, self.retired)
        self.assertIn(pid, ProgramSearchController(self.root).programs)
        self.assertEqual(json.loads((self.root / "retired-programs.json").read_text()), [pid])

    def test_invalid_fixed_control_extension_does_not_partially_apply_new_program(self):
        control = self.v6_proposal("control").model_copy(update={"fixed_parameters": True})
        pid = self.search.register(control)
        item = decision(
            trial_budget=2, programs=(self.v6_proposal("new-program"),), extensions={pid: 1}
        )
        with self.assertRaises(ValueError):
            self.apply(item)
        self.assertEqual(set(ProgramSearchController(self.root).programs), {pid})
        self.assertEqual(self.store.pending(), [])


class SpeedTargetContractTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "race_id": ["first", "first", "second", "second"],
            "horse_id": ["a", "b", "a", "b"],
            "distance": [1200.0, 1200.0, 1600.0, 1600.0],
            "finish_time": ["1.10.00", "1:11.50", "1.35.00", "1:36.25"],
        })

    def target(self, frame, min_coverage=1.0):
        return apply_target_contract(
            frame, target_contract("adjusted_finish_time_or_speed", {"min_coverage": min_coverage})
        )

    def test_formatted_raw_times_use_seconds_not_numeric_string_coercion(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        converted = self.target(frame)
        np.testing.assert_allclose(
            converted.target_speed.to_numpy(), [1200 / 70, 1200 / 71.5, 1600 / 95, 1600 / 96.25]
        )
        pd.testing.assert_frame_equal(frame, original)

    def test_canonical_seconds_take_precedence_over_conflicting_raw_times(self):
        frame = self.frame()
        frame["finish_seconds"] = [70, 71.5, 95, 96.25]
        frame["finish_time"] = "invalid raw evidence"
        converted = self.target(frame)
        np.testing.assert_allclose(
            converted.target_speed.to_numpy(), [1200 / 70, 1200 / 71.5, 1600 / 95, 1600 / 96.25]
        )

    def test_partial_timing_excludes_whole_race_without_mutating_source(self):
        frame = self.frame()
        frame["finish_seconds"] = [70, np.nan, 95, 96.25]
        frame.index = [10, 11, 20, 21]
        converted = self.target(frame, min_coverage=0.5)
        self.assertEqual(converted.race_id.tolist(), ["second", "second"])
        self.assertEqual(converted.horse_id.tolist(), ["a", "b"])
        self.assertEqual(len(frame), 4)

    def test_nonpositive_and_nonfinite_measurements_never_become_labels(self):
        for column in ("finish_seconds", "distance"):
            for invalid in (0, -1, np.inf, np.nan):
                frame = self.frame()
                frame["finish_seconds"] = [70.0, 71.5, 95.0, 96.25]
                frame.loc[0, column] = invalid
                with self.subTest(column=column, invalid=invalid):
                    converted = self.target(frame, min_coverage=0.5)
                    self.assertEqual(set(converted.race_id), {"second"})
                    self.assertTrue(np.isfinite(converted.target_speed).all())

    def test_adequate_row_coverage_cannot_claim_complete_race_coverage(self):
        frame = self.frame()
        frame["finish_seconds"] = [70, np.nan, 95, np.nan]
        with self.assertRaises(TargetContractError):
            self.target(frame, min_coverage=0.5)


def meetings(races_per_day=(3, 3, 3, 3, 3, 3)):
    rows = []
    for day, count in enumerate(races_per_day):
        for race_no in range(1, count + 1):
            for horse in range(3):
                rows.append({
                    "race_id": f"day-{day}-race-{race_no}",
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                    "race_no": race_no, "horse_no": horse + 1,
                    "result": horse + 1, "target_win": int(horse == 0),
                    "target_probability": float(horse == 0),
                })
    return pd.DataFrame(rows)


class MeetingBoundaryTests(unittest.TestCase):
    def test_explicit_meeting_boundaries_keep_all_fit_windows_on_separate_dates(self):
        frame = meetings()
        dates = frame.groupby("race_id").date.first()
        folds = build_expanding_folds(
            frame, min_train_races=4, calibration_races=1, score_races=2,
            whole_meeting_boundaries=True,
        )
        self.assertTrue(folds)
        for fold in folds:
            with self.subTest(fold=fold.fold_id):
                self.assertLess(dates.loc[list(fold.train_race_ids)].max(), dates.loc[list(fold.calibration_race_ids)].min())
                self.assertLess(dates.loc[list(fold.calibration_race_ids)].max(), dates.loc[list(fold.score_race_ids)].min())
                for race_ids in (fold.train_race_ids, fold.calibration_race_ids, fold.score_race_ids):
                    selected = frame[frame.race_id.isin(race_ids)]
                    for date in selected.date.unique():
                        self.assertEqual(
                            set(selected.loc[selected.date.eq(date), "race_id"]),
                            set(frame.loc[frame.date.eq(date), "race_id"]),
                        )

    def test_legacy_default_keeps_original_race_count_boundaries(self):
        frame = meetings()
        kwargs = dict(min_train_races=4, calibration_races=1, score_races=2)
        legacy = build_expanding_folds(frame, **kwargs)
        explicit = build_expanding_folds(frame, **kwargs, whole_meeting_boundaries=False)
        self.assertEqual(legacy, explicit)
        self.assertEqual(len(legacy[0].train_race_ids), 4)
        self.assertEqual(len(legacy[0].calibration_race_ids), 1)
        self.assertEqual(len(legacy[0].score_race_ids), 2)

    def test_insufficient_complete_meetings_reject_instead_of_splitting_a_day(self):
        with self.assertRaises(ResearchEvaluationError):
            build_expanding_folds(
                meetings((3, 3)), min_train_races=1, calibration_races=1,
                score_races=1, whole_meeting_boundaries=True,
            )

    def test_variable_meeting_sizes_do_not_score_the_same_race_twice(self):
        folds = build_expanding_folds(
            meetings((2, 2, 12, 2, 2, 2, 2)), min_train_races=2,
            calibration_races=4, score_races=1, whole_meeting_boundaries=True,
        )
        scored = [race for fold in folds for race in fold.score_race_ids]
        self.assertEqual(len(scored), len(set(scored)))

    def test_new_protocol_identity_includes_target_parameters_and_boundary_policy(self):
        frame = meetings()
        kwargs = dict(min_train_races=4, calibration_races=1, score_races=2)
        first = make_protocol_manifest(
            frame, target=target_contract("placing_top_k", {"top_k": 1}),
            **kwargs, whole_meeting_boundaries=True,
        )
        second = make_protocol_manifest(
            frame, target=target_contract("placing_top_k", {"top_k": 2}),
            **kwargs, whole_meeting_boundaries=True,
        )
        self.assertNotEqual(first.protocol_id, second.protocol_id)
        legacy = make_protocol_manifest(frame, **kwargs)
        explicit_legacy = make_protocol_manifest(frame, **kwargs, whole_meeting_boundaries=False)
        strict = make_protocol_manifest(frame, **kwargs, whole_meeting_boundaries=True)
        self.assertEqual(legacy.protocol_id, explicit_legacy.protocol_id)
        self.assertNotEqual(legacy.protocol_id, strict.protocol_id)


class PackageIndexAlignmentTests(unittest.TestCase):
    def test_valid_race_probabilities_do_not_depend_on_dataframe_index(self):
        frame = pd.DataFrame({"race_id": ["first", "first", "second", "second"]}, index=[3, 0, 2, 1])
        _validate_complete_races(frame, np.array([0.2, 0.8, 0.3, 0.7]))

    def test_invalid_probabilities_cannot_pass_vacuously_on_nonoverlapping_index(self):
        frame = pd.DataFrame({"race_id": ["first", "first"]}, index=[10, 11])
        with self.assertRaises(ValueError):
            _validate_complete_races(frame, np.array([0.2, 0.2]))


class RuntimeFixtureTests(unittest.TestCase):
    def setUp(self):
        from ima import research_controller as core

        revision = patch.dict("os.environ", {"IMA_CODE_REVISION": core._code_revision()})
        revision.start()
        self.addCleanup(revision.stop)
        fits = patch("ima.research_expansion.BoundedFitExecutor", BoundedFitsThreadPool)
        fits.start()
        self.addCleanup(fits.stop)

    def test_restart_resumes_reserved_attempt_before_asking_replacement_work(self):
        from ima import research_controller as core
        from ima.optimizer import CampaignConfig
        from ima.research_executor import RecipeExecutionResult
        from ima.research_expansion import run_expansion_campaign
        from ima.research_store import ResearchLedger
        from tests.test_research_controller import ResearchControllerTests

        def executor(**kwargs):
            return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

        def planned(evidence, config):
            item = proposal().model_copy(update={"evidence_ids": (evidence["evidence_id"],)})
            return {"decision": {
                "decision_id": evidence["decision_id"], "evidence_id": evidence["evidence_id"],
                "trial_budget": 1, "programs": [item.model_dump(mode="json")],
            }, "usage": {}}

        def worker(request, workload, threads):
            return RecipeExecutionResult(
                1, request.attempt_id, request.proposal_id, request.trial_number,
                request.recipe.recipe_hash(), request.recipe.target.kind,
                "completed", "fundamental_log_loss", 2.0, {}, {}, {}, 0.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = ResearchControllerTests().fixture(root)
            campaign = root / "campaign"
            config = CampaignConfig(
                campaign_dir=campaign, policy="agentic", research_policy="expansion_v6",
                planner_mode="fixture", max_trials=1, proposal_batch_size=1,
                max_concurrent_trials=1, max_active_preparations=1,
                host_reserve_cpu_threads=0, host_reserve_ram_gib=0,
                dataset_path=dataset, protocol_path=protocol,
            )
            search = ProgramSearchController(campaign)
            pid = search.register(proposal())
            suggestion = search.ask(1, allowed_program_ids=[pid])[0]
            digest = core._hash_file(dataset)
            parameters = core._protocol_parameters(config)
            revision, environment = core._code_revision(), core._environment_hash()
            signature = core._execution_signature(suggestion.recipe, digest, parameters, revision, environment)
            payload = suggestion.serializable() | {
                "signature": signature, "dataset_hash": digest, "protocol_parameters": parameters,
                "code_revision": revision, "environment_hash": environment,
            }
            ledger = ResearchLedger(campaign / "ledger.sqlite")
            original = ledger.reserve_attempt(signature, payload)
            ledger.mark_running(original.attempt_id)
            with patch("ima.research_expansion.ProcessPoolExecutor", side_effect=executor), \
                 patch("ima.research_expansion.plan_decision", side_effect=planned), \
                 patch("ima.research_expansion._worker", side_effect=worker):
                output = run_expansion_campaign(config)
            self.assertEqual(output["mode"], "complete")
            completed = ResearchLedger(campaign / "ledger.sqlite").terminal_results()
            self.assertIn(original.attempt_id, {row["attempt_id"] for row in completed})
            self.assertEqual(ResearchLedger(campaign / "ledger.sqlite").pending_attempts(), [])

    def test_missing_preparation_manifest_schedules_preparation_before_fit(self):
        from ima.optimizer import CampaignConfig
        from ima.research_executor import RecipeExecutionResult
        from ima.research_expansion import _prepare_fold_dependency, run_expansion_campaign
        from tests.test_research_controller import ResearchControllerTests

        stages = []

        def prewarm(request, workload):
            stages.append("prewarming")
            outcome = _prepare_fold_dependency(request, workload)
            self.assertTrue(outcome["artifacts"])
            stages.append("folds_ready")
            return outcome

        def executor(**kwargs):
            return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

        def planned(evidence, config):
            item = proposal(trials=2).model_copy(update={"evidence_ids": (evidence["evidence_id"],)})
            return {"decision": {
                "decision_id": evidence["decision_id"], "evidence_id": evidence["evidence_id"],
                "trial_budget": 2, "programs": [item.model_dump(mode="json")],
            }, "usage": {}}

        def worker(request, workload, threads):
            self.assertIn("folds_ready", stages)
            stages.append("fit")
            return RecipeExecutionResult(
                1, request.attempt_id, request.proposal_id, request.trial_number,
                request.recipe.recipe_hash(), request.recipe.target.kind,
                "completed", "fundamental_log_loss", 2.0, {}, {}, {}, 0.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = ResearchControllerTests().fixture(root)
            config = CampaignConfig(
                campaign_dir=root / "campaign", policy="agentic", research_policy="expansion_v6",
                planner_mode="fixture", max_trials=2, proposal_batch_size=2,
                max_concurrent_trials=1, max_active_preparations=1,
                host_reserve_cpu_threads=0, host_reserve_ram_gib=0,
                dataset_path=dataset, protocol_path=protocol,
            )
            with patch("ima.research_expansion.ProcessPoolExecutor", side_effect=executor), \
                 patch("ima.research_expansion.plan_decision", side_effect=planned), \
                 patch("ima.research_expansion._worker", side_effect=worker), \
                 patch("ima.research_expansion._prepare_fold_dependency", side_effect=prewarm) as fold_prepare, \
                 patch("ima.research_expansion._prepare_program", return_value={"matrix_id": "ready", "shared_bytes": 0}) as prepare:
                output = run_expansion_campaign(config)
            self.assertEqual(output["mode"], "complete")
            prepare.assert_called_once()
            fold_prepare.assert_called_once()
            self.assertEqual(["prewarming", "folds_ready", "fit", "fit"], stages)

    def test_fixture_planner_respects_one_trial_operator_ceiling(self):
        from ima.research_expansion import plan_decision

        config = SimpleNamespace(
            planner_mode="fixture", proposal_batch_size=1,
            max_new_programs_per_decision=1, max_pending_programs=4,
        )
        response = plan_decision({"decision_id": "D000001", "evidence_id": "e1"}, config)
        item = PlannerDecision.model_validate(response["decision"])
        self.assertLessEqual(item.trial_budget, 1)
        self.assertLessEqual(len(item.programs), 1)

    def test_tiny_chronological_campaign_completes_through_real_cli(self):
        from ima.research_store import ResearchLedger
        from tests.test_research_controller import ResearchControllerTests

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = ResearchControllerTests().fixture(root)
            from ima.feature_sets import FEATURE_SCHEMAS
            frame = pd.read_csv(dataset)
            for name in FEATURE_SCHEMAS["benter-rich-v1"].numeric:
                if name not in frame:
                    frame[name] = frame.horse_no.astype(float) + np.arange(len(frame)) * 0.01
            for name in FEATURE_SCHEMAS["benter-rich-v1"].categorical:
                if name not in frame:
                    frame[name] = "category-" + frame.horse_no.astype(str)
            frame.to_csv(dataset, index=False)
            campaign = root / "campaign"
            config = root / "config.json"
            config.write_text(json.dumps({
                "policy": "agentic", "research_policy": "expansion_v6",
                "planner_mode": "fixture", "max_trials": 5,
                "proposal_batch_size": 5, "max_concurrent_trials": 1,
                "dataset_path": str(dataset), "protocol_path": str(protocol),
                "max_active_preparations": 1, "host_reserve_cpu_threads": 0,
                "host_reserve_ram_gib": 0, "cpu_thread_budget": 1,
                "planning_checkpoint_seconds": 300, "timeout_minutes": 2,
            }))
            command = [sys.executable, "-m", "scripts.optimize", "run",
                       "--campaign", str(campaign), "--config", str(config)]
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[1],
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            # One-task spawn workers and serial admission took ~23s for 8 builders + 5 fits.
            # Keep a finite wall bound with startup headroom, not just a fit-time limit.
            timeout_seconds = 120
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                status = (campaign / "status.json").read_text() if (campaign / "status.json").exists() else "no status"
                trial_path = campaign / "trials.jsonl"
                errors = sorted({
                    row["error"] for row in (
                        json.loads(line) for line in trial_path.read_text().splitlines()
                    ) if row.get("error")
                }) if trial_path.exists() else []
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                self.fail(f"Fixture campaign did not complete within {timeout_seconds}s: {status}\nTrial errors: {errors}\n{stderr[-1000:]}")
            self.assertEqual(process.returncode, 0, stderr[-5000:])
            payload = json.loads((campaign / "status.json").read_text())
            trials = [json.loads(line) for line in (campaign / "trials.jsonl").read_text().splitlines()]
            errors = sorted({row["error"] for row in trials if row.get("error")})
            self.assertEqual(payload["mode"], "complete", f"Trial errors: {errors}")
            self.assertEqual(payload["ledger"]["completed"], 5)
            self.assertEqual(sum(row["lane"] == "benter" for row in trials), 4)
            self.assertEqual(sum(row["lane"] == "experimental" for row in trials), 1)
            final_summary = json.loads((campaign / "summary.json").read_text())
            self.assertEqual(final_summary["identity"]["campaign_id"], campaign.name)
            self.assertEqual(final_summary["operation"]["status"], "complete")
            execution = final_summary["execution"]
            self.assertEqual(execution["counts"]["completed"], 5)
            self.assertEqual(execution["active_trial_count"], 0)
            self.assertEqual(execution["active_trials"], [])
            self.assertTrue(execution["active_details_complete"])
            self.assertTrue(execution["invocation_history"])
            for invocation in execution["invocation_history"]:
                self.assertEqual(invocation["running_attempts"], [])
            terminal = {row["attempt_id"]: row["result"]
                        for row in ResearchLedger(campaign / "ledger.sqlite").terminal_results()}
            self.assertTrue(final_summary["champions"], "Completed real fits must appear in the final summary")
            for champion in final_summary["champions"]:
                result = terminal[champion["current"]["attempt_id"]]
                self.assertTrue(np.isfinite(champion["current_value"]))
                self.assertEqual(champion["objective"], result["objective_name"])
                self.assertAlmostEqual(champion["current_value"], result["objective_value"])
                self.assertAlmostEqual(champion["current_value"], result["metrics"]["objective"])
            self.assertAlmostEqual(min(row["current_value"] for row in final_summary["champions"]),
                                   min(result["metrics"]["objective"] for result in terminal.values()))
            self.assertNotIn("no comparable completed result yet", (campaign / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
