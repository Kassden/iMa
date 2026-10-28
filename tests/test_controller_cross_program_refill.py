"""Bounded integration proof for completion-driven cross-program refill."""
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import pandas as pd

from ima.optimizer import CampaignConfig
from ima.research_expansion import run_expansion_campaign
from ima.research_executor import RecipeExecutionResult
from tests.test_research_expansion import proposal


class ControlledFitPool:
    """Hold fit futures until the event-loop harness releases selected work."""

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submitted = []
        self.max_active = 0
        self.c_dispatched_while_a_active = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def submit(self, _function, request, _workload, _threads, *, runtime_path, timeout):
        future = Future()
        row = {"future": future, "request": request}
        self.submitted.append(row)
        active = [item for item in self.submitted if not item["future"].done()]
        self.max_active = max(self.max_active, len(active))
        assert len(active) <= self.max_workers
        if request.proposal_id == "program-C":
            by_program = {item["request"].proposal_id: item["future"] for item in self.submitted}
            assert "program-B" in by_program, by_program
            assert by_program["program-B"].done(), by_program
            assert not by_program["program-A"].done(), by_program
            self.c_dispatched_while_a_active = True
            assert len(active) == self.max_workers
            # Stop after the controlled three-fit episode; this sentinel lives only
            # in the temporary campaign created by the test.
            Path(runtime_path).parents[2].joinpath("STOP").write_text("controlled stop\n")
        return future

    @staticmethod
    def finish(row):
        request = row["request"]
        row["future"].set_result(RecipeExecutionResult(
            1, request.attempt_id, request.proposal_id, request.trial_number,
            request.recipe.recipe_hash(), request.recipe.target.kind,
            "completed", "fundamental_log_loss", 2.0, {}, {}, {}, 0.0,
        ))


def test_completion_refills_another_program_while_slow_program_remains_active():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        frame = pd.read_csv("tests/fixtures/research_races.csv")
        frame["horse_rating"] = 60 + frame["horse_no"]
        frame["horse_age"] = 4 + (frame["horse_no"] % 3)
        frame["field_size"] = frame.groupby("race_id")["race_id"].transform("size")
        frame["win_odds"] = 1.0 / frame["market_probability"]
        dataset = root / "dataset.csv"
        frame.to_csv(dataset, index=False)
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({
            "min_train_races": 2,
            "calibration_races": 1,
            "score_races": 1,
            "max_folds": 1,
        }))
        campaign = root / "campaign"
        config = CampaignConfig(
            campaign_dir=campaign,
            policy="agentic",
            research_policy="expansion_v6",
            planner_mode="fixture",
            dataset_path=dataset,
            protocol_path=protocol,
            max_trials=101,
            proposal_batch_size=101,
            max_concurrent_trials=2,
            max_active_preparations=2,
            max_new_programs_per_decision=3,
            max_pending_programs=3,
            cpu_thread_budget=2,
            ram_budget_gib=60,
            memory_budget_gb_decimal=100,
            host_reserve_cpu_threads=0,
            host_reserve_ram_gib=0,
            queue_low_watermark=0,
            replan_every_terminal_trials=150,
        )

        fit_pool = ControlledFitPool(max_workers=2)
        fit_pool_created = False
        plan_calls = []
        program_budgets = {}
        b_completed = False
        refill_waits = 0
        real_sleep = time.sleep

        def executor_factory(max_workers, *, max_tasks_per_child=1):
            nonlocal fit_pool_created
            if not fit_pool_created:
                fit_pool_created = True
                assert max_workers == 2
                return fit_pool
            return ThreadPoolExecutor(max_workers=max_workers)

        def planner(evidence, _config):
            plan_calls.append(evidence["decision_id"])
            programs = []
            for identifier, seed, budget in (("program-A", 101, 50),
                                             ("program-B", 102, 1),
                                             ("program-C", 103, 50)):
                candidate = proposal(identifier, trials=budget)
                program_budgets[identifier] = budget
                if identifier == "program-C":
                    model = candidate.recipe.model.model_copy(update={
                        "kind": "logit", "parameters": {"C": 1.0},
                    })
                    recipe = candidate.recipe.model_copy(update={"model": model, "seed": seed})
                else:
                    recipe = candidate.recipe.model_copy(update={"seed": seed})
                programs.append(candidate.model_copy(update={
                    "recipe": recipe,
                    "evidence_ids": (evidence["evidence_id"],),
                }))
            return {"decision": {
                "decision_id": evidence["decision_id"],
                "evidence_id": evidence["evidence_id"],
                "trial_budget": 101,
                "programs": [item.model_dump(mode="json") for item in programs],
            }, "usage": {}}

        def controlled_wait(futures, *, timeout=None, return_when=None):
            nonlocal b_completed, refill_waits
            rows = {item["future"]: item for item in fit_pool.submitted}
            pending = set(futures)
            if not b_completed and len(fit_pool.submitted) == 2:
                # B finishes first; A deliberately remains unresolved.
                b = next((item for item in fit_pool.submitted
                          if item["request"].proposal_id == "program-B"), None)
                assert b is not None, [item["request"].proposal_id for item in fit_pool.submitted]
                fit_pool.finish(b)
                b_completed = True
                return {b["future"]}, pending - {b["future"]}
            if not fit_pool.c_dispatched_while_a_active:
                refill_waits += 1
                assert refill_waits <= 100, (
                    "program-C was not dispatched while program-A remained active; "
                    f"submitted={[item['request'].proposal_id for item in fit_pool.submitted]}"
                )
                real_sleep(0.01)
                return set(), pending
            # Drain A and the refilled C only after the assertion point.
            for future in pending:
                ControlledFitPool.finish(rows[future])
            return pending, set()

        def immediate_program_prep(*_args, **_kwargs):
            return {"matrix_id": "fixture-ready", "shared_bytes": 0}

        def bounded_sleep(seconds):
            real_sleep(min(seconds, 0.01))

        with patch("ima.research_expansion.BoundedFitExecutor", side_effect=executor_factory), \
             patch("ima.research_expansion.plan_decision", side_effect=planner), \
             patch("ima.research_expansion.wait", side_effect=controlled_wait), \
             patch("ima.research_expansion._prepare_program", side_effect=immediate_program_prep), \
             patch("ima.research_expansion._dependency_request", return_value=(None, None)), \
             patch("ima.research_expansion.time.sleep", side_effect=bounded_sleep), \
             patch("ima.research_expansion.cleanup_attempt_group"):
            result = run_expansion_campaign(config)

        submitted_order = [item["request"].proposal_id for item in fit_pool.submitted]
        assert set(submitted_order[:2]) == {"program-A", "program-B"}
        assert submitted_order[2] == "program-C"
        assert fit_pool.c_dispatched_while_a_active
        assert fit_pool.max_active == config.max_concurrent_trials == 2
        assert program_budgets == {"program-A": 50, "program-B": 1, "program-C": 50}
        assert program_budgets["program-A"] > config.max_concurrent_trials
        assert program_budgets["program-C"] > config.max_concurrent_trials
        assert config.proposal_batch_size == 101 > config.max_concurrent_trials
        assert len(plan_calls) == 1  # fixture planner only; no paid/provider call path
        assert result["mode"] == "stopped"
        assert result["ledger"]["completed"] == 3
