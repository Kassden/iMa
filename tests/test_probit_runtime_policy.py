import json
import os
import time
from concurrent.futures import Future, TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ima.probit_runtime_policy import ProbitRuntimePolicy, resolve_probit_budget
from ima.research_resources import JobMeasurement, JobWorkload
from ima.research_worker_runtime import (
    BoundedFitExecutor, _update_runtime, extend_attempt_deadline, read_runtime,
    remaining_deadline,
)


def workload(**changes):
    return JobWorkload(stage="fit", family="gaussian_probit", rows=100,
        generated_features=2, selected_features=2, implementation_revision="test-v1",
        dependency_versions={"scipy": "test"}, search_settings={"gradient_mode": "analytic"}, **changes)


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def progress(now, *, iteration=1, objective=3., **changes):
    return dict(stage="optimizing", pid=os.getpid(), fold_id="fold-1",
        optimization_round=0, quadrature_order=48, updated_at_epoch=now,
        iteration=iteration, objective_evaluations=iteration*2, objective=objective, **changes)


def state_at(now, policy):
    return dict(status="running", pid=os.getpid(), deadline_at=iso(now+10),
        hard_deadline_at=iso(now+20), progress_not_before_epoch=now-1,
        progress_policy=policy.to_dict(), runtime_extensions=[], task_nonce="test")


def publish_and_extend(root, event, now):
    _update_runtime(root/"progress.json", event)
    return extend_attempt_deadline(root/"runtime.json", now=now)


def test_floor_exact_samples_and_failed_censored_bounds():
    job = workload()
    assert resolve_probit_budget(job, [], 1200)["initial_seconds"] == 14400
    foreign = JobMeasurement(workload(native_threads=2).fingerprint(), 1000, 1e7)
    assert resolve_probit_budget(job, [foreign], 1200)["initial_seconds"] == 14400
    samples = [JobMeasurement(job.fingerprint(), 1000, 1000) for _ in range(5)]
    for flags in ({"failed": True}, {"censored": True}):
        budget = resolve_probit_budget(job, samples+[JobMeasurement(job.fingerprint(), 1000, 20000, **flags)], 1200)
        assert budget["initial_seconds"] == 25000
        assert budget["hard_seconds"] == 32200
        assert budget["workload_fingerprint"] == job.fingerprint()
    assert resolve_probit_budget(job, samples, 30000)["initial_seconds"] == 30000


def test_cold_numerical_admission_extrapolation_cannot_inflate_initial_allowance():
    job = workload(folds=3).model_copy(update={"rows": 140000,
        "selected_features": 100, "search_settings": {"max_iter": 500,
        "race_count": 10000, "parameter_dimension": 200, "heteroscedastic": True,
        "gradient_mode": "numerical"}})
    assert resolve_probit_budget(job, [], 1200)["initial_seconds"] == 14400
    bound = JobMeasurement(job.fingerprint(), 1000, 14400, failed=True, censored=True)
    assert resolve_probit_budget(job, [bound], 1200)["initial_seconds"] > 14400
    success = JobMeasurement(job.fingerprint(), 1000, 20000)
    assert resolve_probit_budget(job, [success], 1200)["initial_seconds"] == 25000


def test_graph_probit_uses_actual_graph_identity_and_exact_samples():
    graph = workload().model_copy(update={"family": "graph:logit", "search_settings": {
        "graph_structure": [{"id": "distribution", "kind": "performance_distribution",
                             "model": "gaussian_probit", "cost": {"quadrature_order": 48}}]}})
    foreign = JobMeasurement(workload().fingerprint(), 1000, 50000)
    budget = resolve_probit_budget(graph, [foreign], 1200)
    assert budget["initial_seconds"] == 14400
    assert budget["workload_fingerprint"] == graph.fingerprint()
    actual = JobMeasurement(graph.fingerprint(), 1000, 20000)
    assert resolve_probit_budget(graph, [actual], 1200)["initial_seconds"] == 25000


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_requested_budget(value):
    with pytest.raises(ValueError):
        resolve_probit_budget(workload(), [], value)


def test_representative_success_runtime_can_raise_allowance():
    job = workload()
    samples = [JobMeasurement(job.fingerprint(), 1000, 30000) for _ in range(5)]
    assert resolve_probit_budget(job, samples, 1200)["initial_seconds"] == 37500
    with pytest.raises(ValueError):
        resolve_probit_budget(job.model_copy(update={"family": "logit"}), [], 1200)


def test_durable_extensions_require_new_productive_evidence_and_obey_cap(tmp_path):
    now = 1000.
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20,
        extension_seconds=6, extension_window_seconds=2, max_extensions=2)
    _update_runtime(tmp_path/"runtime.json", state_at(now, policy))
    first = publish_and_extend(tmp_path, progress(now), now)
    assert not first["runtime_extensions"]
    extended = publish_and_extend(tmp_path, progress(now+9, iteration=2, objective=2.), now+9)
    assert extended["deadline_at"] == iso(now+16)
    assert len(extended["runtime_extensions"]) == 1
    # The same evidence cannot buy another extension, even after the window opens.
    repeat = extend_attempt_deadline(tmp_path/"runtime.json", now=now+15)
    assert repeat["deadline_at"] == extended["deadline_at"]
    capped = publish_and_extend(tmp_path, progress(now+15.5, iteration=3, objective=1.), now+15.5)
    assert capped["deadline_at"] == iso(now+20)
    assert len(capped["runtime_extensions"]) == 2
    after = publish_and_extend(tmp_path, progress(now+19, iteration=4, objective=.5), now+19)
    assert after["deadline_at"] == iso(now+20)
    assert after["runtime_extensions"] == read_runtime(tmp_path/"runtime.json")["runtime_extensions"]


@pytest.mark.parametrize("bad", [
    {"objective": float("nan")}, {"objective": float("inf")},
    {"objective": 3.}, {"iteration": 1}, {"objective_evaluations": 2},
    {"updated_at_epoch": 900.}, {"updated_at_epoch": 1011.},
    {"stage": "training"}, {"stage": "verifying_quadrature"},
    {"pid": -1}, {"fold_id": "fold-2"}, {"optimization_round": 1},
    {"gradient_norm": .01, "objective": 3.},
    {"gradient_norm": float("nan")},
])
def test_nonproductive_or_untrusted_progress_cannot_extend(tmp_path, bad):
    now = 1000.
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20,
        extension_seconds=6, extension_window_seconds=2, progress_max_age_seconds=20)
    _update_runtime(tmp_path/"runtime.json", state_at(now, policy))
    publish_and_extend(tmp_path, progress(now), now)
    event = dict(progress(now+9, iteration=2, objective=2.), **bad)
    # Write NaN directly to exercise rejection of a corrupted external producer.
    (tmp_path/"progress.json").write_text(json.dumps(event))
    result = extend_attempt_deadline(tmp_path/"runtime.json", now=now+9)
    assert result["deadline_at"] == iso(now+10)
    assert not result["runtime_extensions"]


@pytest.mark.parametrize("raw", ["broken", "[]", "null", '{}'])
def test_malformed_progress_fails_closed(tmp_path, raw):
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20)
    _update_runtime(tmp_path/"runtime.json", state_at(1000, policy))
    (tmp_path/"progress.json").write_text(raw)
    assert not extend_attempt_deadline(tmp_path/"runtime.json", now=1009)["runtime_extensions"]


def test_verification_progress_and_completed_fold_can_extend_once(tmp_path):
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=30,
        extension_seconds=6, extension_window_seconds=2, max_extensions=3)
    original = state_at(1000, policy)
    original["hard_deadline_at"] = iso(1030)
    _update_runtime(tmp_path/"runtime.json", original)
    verify = dict(progress(1000), stage="verifying_quadrature",
                  verification_batches_completed=0, verification_batches_total=3)
    assert not publish_and_extend(tmp_path, verify, 1000)["runtime_extensions"]
    verify.update(updated_at_epoch=1009, verification_batches_completed=1)
    extended = publish_and_extend(tmp_path, verify, 1009)
    assert extended["deadline_at"] == iso(1016)
    assert extended["progress_health"] == "productive"
    completed = dict(progress(1015), stage="fit_completed", converged=True)
    extended = publish_and_extend(tmp_path, completed, 1015)
    assert extended["deadline_at"] == iso(1022)
    assert extended["completed_progress_folds"] == ["fold-1"]
    completed.update(updated_at_epoch=1021, stage="fold_completed")
    repeated = publish_and_extend(tmp_path, completed, 1021)
    assert repeated["deadline_at"] == iso(1022)
    assert len(repeated["runtime_extensions"]) == 2
    completed.update(updated_at_epoch=1021.5, fold_id="fold-2")
    next_fold = publish_and_extend(tmp_path, completed, 1021.5)
    assert next_fold["deadline_at"] == iso(1028)


def test_fresh_slow_progress_is_distinct_from_stale_stall(tmp_path):
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20,
        extension_seconds=6, extension_window_seconds=2, progress_max_age_seconds=5)
    _update_runtime(tmp_path/"runtime.json", state_at(1000, policy))
    publish_and_extend(tmp_path, progress(1000), 1000)
    slow = publish_and_extend(tmp_path, progress(1009, iteration=2, objective=3.), 1009)
    assert slow["progress_health"] == "slow_nonproductive"
    stale = extend_attempt_deadline(tmp_path/"runtime.json", now=1015)
    assert stale["progress_health"] == "stalled_stale"
    assert not stale["productive_progress"]


@pytest.mark.parametrize("stage", ["fit_completed", "fold_completed"])
def test_unconverged_completion_is_not_productive(tmp_path, stage):
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20,
        extension_window_seconds=2)
    _update_runtime(tmp_path/"runtime.json", state_at(1000, policy))
    completed = dict(progress(1009), stage=stage, converged=False)
    result = publish_and_extend(tmp_path, completed, 1009)
    assert not result["runtime_extensions"]
    assert result["progress_health"] == "invalid_progress"


def test_restart_preserves_policy_cap_receipts_but_discards_old_progress(tmp_path):
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20)
    old = state_at(time.time(), policy)
    old["runtime_extensions"] = [{"progress_updated_at_epoch": time.time()-1, "reason": "test"}]
    old["optimizer_observation"] = {"old": "observation"}
    path = tmp_path/"runtime.json"
    _update_runtime(path, old)
    with patch("pebble.ProcessPool") as pool:
        pool.return_value.schedule.return_value = Future()
        with BoundedFitExecutor(1) as executor:
            future = executor.submit(time.sleep, 0, runtime_path=path, timeout=10, progress_policy=policy)
            admitted = read_runtime(path)
            assert admitted["deadline_at"] == old["deadline_at"]
            assert admitted["hard_deadline_at"] == old["hard_deadline_at"]
            assert admitted["runtime_extensions"] == old["runtime_extensions"]
            assert admitted["optimizer_observation"] is None
            assert admitted["productive_progress"] is None
            assert pool.return_value.schedule.call_args.kwargs["timeout"] > 19
            future.cancel()
            assert read_runtime(path)["status"] == "cancelled"
            with pytest.raises(ValueError, match="cannot change"):
                executor.submit(time.sleep, 0, runtime_path=path, timeout=10)
    # Restarting an expired extended attempt cannot allocate a new initial budget.
    _update_runtime(path, {"deadline_at": iso(1)})
    with patch("pebble.ProcessPool") as pool:
        with BoundedFitExecutor(1) as executor:
            with pytest.raises(TimeoutError):
                executor.submit(time.sleep, 0, runtime_path=path, timeout=10, progress_policy=policy).result()
        pool.return_value.schedule.assert_not_called()


def test_old_task_cannot_extend_resubmitted_attempt_and_hard_cap_never_resets(tmp_path):
    policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20)
    _update_runtime(tmp_path/"runtime.json", state_at(1000, policy))
    _update_runtime(tmp_path/"progress.json", progress(1001))
    before = read_runtime(tmp_path/"runtime.json")
    assert extend_attempt_deadline(tmp_path/"runtime.json", now=1001, task_nonce="old-task") == before
    _update_runtime(tmp_path/"runtime.json", {"deadline_at": iso(time.time()+100), "hard_deadline_at": iso(1)})
    assert remaining_deadline(tmp_path/"runtime.json", 10) == 0
    with patch("pebble.ProcessPool") as pool:
        with BoundedFitExecutor(1) as executor:
            with pytest.raises(TimeoutError):
                executor.submit(time.sleep, 0, runtime_path=tmp_path/"runtime.json",
                                timeout=10, progress_policy=policy).result()
        pool.return_value.schedule.assert_not_called()


def test_recalculated_budget_on_restart_cannot_increase_admitted_limits(tmp_path):
    original_policy = ProbitRuntimePolicy(initial_seconds=10, hard_seconds=20,
                                         workload_fingerprint="same-workload")
    old = state_at(time.time(), original_policy)
    old["runtime_extensions"] = [{"progress_updated_at_epoch": time.time()-1, "reason": "test"}]
    path = tmp_path/"runtime.json"
    _update_runtime(path, old)
    new_policy = ProbitRuntimePolicy(initial_seconds=30000, hard_seconds=40000,
                                    workload_fingerprint="same-workload")
    with patch("pebble.ProcessPool") as pool:
        pool.return_value.schedule.return_value = Future()
        with BoundedFitExecutor(1) as executor:
            future = executor.submit(time.sleep, 0, runtime_path=path, timeout=30000, progress_policy=new_policy)
            admitted = read_runtime(path)
            assert admitted["progress_policy"] == original_policy.to_dict()
            assert admitted["deadline_at"] == old["deadline_at"]
            assert admitted["hard_deadline_at"] == old["hard_deadline_at"]
            assert admitted["runtime_extensions"] == old["runtime_extensions"]
            future.cancel()


def productive_fit(progress_path, duration):
    started = time.monotonic()
    iteration = 0
    while time.monotonic()-started < duration:
        iteration += 1
        _update_runtime(progress_path, progress(time.time(), iteration=iteration, objective=10./iteration))
        time.sleep(.08)
    return "converged"


def test_real_pebble_timeout_allows_productive_soft_extension(tmp_path):
    policy = ProbitRuntimePolicy(initial_seconds=4, hard_seconds=8,
        extension_seconds=3, max_extensions=1, extension_window_seconds=.5,
        progress_max_age_seconds=1)
    with BoundedFitExecutor(1) as executor:
        future = executor.submit(productive_fit, tmp_path/"progress.json", 4.2,
            runtime_path=tmp_path/"runtime.json", timeout=4, progress_policy=policy)
        assert future.result(15) == "converged"
    result = read_runtime(tmp_path/"runtime.json")
    assert result["status"] == "finished"
    assert len(result["runtime_extensions"]) == 1
    assert result["deadline_at"] == result["runtime_extensions"][-1]["deadline_at"]
    assert datetime.fromisoformat(result["deadline_at"]) <= datetime.fromisoformat(result["hard_deadline_at"])


def test_real_productive_worker_still_has_absolute_hard_cap(tmp_path):
    policy = ProbitRuntimePolicy(initial_seconds=4, hard_seconds=5,
        extension_seconds=3, max_extensions=4, extension_window_seconds=.5,
        progress_max_age_seconds=1)
    began = time.monotonic()
    with BoundedFitExecutor(1) as executor:
        future = executor.submit(productive_fit, tmp_path/"progress.json", 30,
            runtime_path=tmp_path/"runtime.json", timeout=4, progress_policy=policy)
        with pytest.raises(TimeoutError):
            future.result(15)
    assert time.monotonic()-began < 10
    result = read_runtime(tmp_path/"runtime.json")
    assert result["status"] == "timed_out"
    assert result["deadline_at"] == result["hard_deadline_at"]
