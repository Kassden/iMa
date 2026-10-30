"""Actual registry, feature builder, trainer and paired-outcome consumer."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from ima.feature_exploration import ExplorationConfig, FeatureExplorer
from ima.feature_lifecycle import catalog
import tests.test_dataset_registry as registry_fixtures


def explorer_fixture(tmp_path, **changes):
    registry, parent = registry_fixtures.DatasetRegistryTests()._build_row_hash_fixture(tmp_path)
    features = registry.dataset_path(parent["dataset_id"])/"features.parquet"
    config = ExplorationConfig(root=tmp_path/"exploration", parent_features=features, registry=registry.root,
                               disk_reserve_bytes=0, **changes)
    return FeatureExplorer(config)


def test_real_paired_features_cached_control_and_registry_feedback(tmp_path):
    explorer = explorer_fixture(tmp_path)
    explorer.refill()
    assert len(explorer.queue.rows()) > 8
    first = explorer.step()
    assert first["status"] == "screened", first
    controls = list((explorer.root/"trials").glob("feature-control-*/result.json"))
    assert len(controls) == 1
    cached = controls[0].read_bytes()
    results = []
    for _ in range(32):
        result = explorer.step()
        results.append(result)
        if result and result["status"] == "screened":
            break
    assert any(result and result["status"] == "screened" for result in results), results
    assert controls[0].read_bytes() == cached
    rows = explorer.queue.rows()
    screened = [row for row in rows if row["status"] == "screened"]
    assert len(screened) >= 2
    for row in screened:
        payload = row["result"]
        assert payload["result"]["lineage"]["evaluation_population_id"] == payload["baseline"]["lineage"]["evaluation_population_id"]
        assert payload["paired"]["races"] > 0
        assert payload["result"]["objective_name"] == payload["baseline"]["objective_name"]
    entries = catalog(explorer.config.registry)["builds"]
    assert len(entries) >= 2
    ready = [entry for entry in entries if entry["status"] == "ready"]
    assert all(entry["comparisons"][0]["model_outcomes"][0]["paired"] for entry in ready)
    assert explorer.status()["screened"] >= 2
    resumed = FeatureExplorer(explorer.config)
    assert resumed.status()["screened"] == len(screened)
    assert len(resumed.queue.rows()) == len(rows)
    with pytest.raises(ValueError, match="context changed"):
        FeatureExplorer(replace(explorer.config, code_revision="changed"))


def test_tracking_outage_does_not_turn_successful_fit_into_failure(tmp_path):
    uri = "sqlite:///"+str(tmp_path/"recovered.sqlite")
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("ima-feature-exploration", artifact_location=str(tmp_path/"screens"))
    client.create_experiment("ima-feature-builds", artifact_location=str(tmp_path/"builds"))
    explorer = explorer_fixture(tmp_path, tracking_uri=uri)
    # Test real scientific work, but inject the external transport failure.
    with patch("ima.feature_exploration.build_feature_candidate") as build, \
            patch("mlflow.tracking.MlflowClient", side_effect=OSError("tracking offline")):
        from ima.feature_lifecycle import build_feature_candidate
        build.side_effect = lambda *args, **kwargs: build_feature_candidate(*args, **dict(kwargs, tracking_uri=None))
        result = explorer.step()
    assert result["status"] == "screened"
    row = next(row for row in explorer.queue.rows() if row["status"] == "screened")
    assert row["result"]["tracking_status"] == "pending"
    assert "tracking offline" in row["result"]["tracking_error"]
    assert explorer.status()["pending_tracking"] == 1
    payload = row["result"]
    payload["tracking_retry_after"] = 0
    explorer.queue.update(row["id"], "screened", payload)
    explorer.retry_tracking()
    recovered = next(row for row in explorer.queue.rows() if row["status"] == "screened")["result"]
    assert recovered["tracking_status"] == "published", recovered
    assert client.get_run(recovered["build_run_id"]).info.status == "FINISHED"
    assert client.get_run(recovered["run_id"]).data.params["feature_build_run_id"] == recovered["build_run_id"]


def test_disk_pause_and_baseline_error_preserve_candidate(tmp_path):
    explorer = explorer_fixture(tmp_path)
    with patch("ima.feature_exploration.shutil.disk_usage") as disk:
        disk.return_value.free = -1
        assert explorer.step() is None
        assert json.loads((explorer.root/"status.json").read_text())["operation"] == "waiting_disk"
    with patch.object(explorer, "baseline", side_effect=RuntimeError("control failed")):
        with pytest.raises(RuntimeError, match="control failed"):
            explorer.step()
    assert not any(row["status"] == "failed" for row in explorer.queue.rows())
    assert any(row["error"] == "control failed" and row["status"] == "queued" for row in explorer.queue.rows())


def test_real_mlflow_stage_runs_and_baseline_linkage(tmp_path):
    from mlflow.tracking import MlflowClient
    from ima.feature_exploration import EXPERIMENT
    uri = "sqlite:///"+str(tmp_path/"tracking.sqlite")
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment(EXPERIMENT, artifact_location=str(tmp_path/"screen-artifacts"))
    client.create_experiment("ima-feature-builds", artifact_location=str(tmp_path/"build-artifacts"))
    explorer = explorer_fixture(tmp_path, tracking_uri=uri)
    result = explorer.step()
    assert result["status"] == "screened", result
    row = next(row for row in explorer.queue.rows() if row["status"] == "screened")
    payload = row["result"]
    assert payload["tracking_status"] == "published", payload.get("tracking_error")
    candidate = client.get_run(payload["run_id"])
    assert candidate.info.status == "FINISHED"
    assert candidate.data.params["baseline_run_id"] == payload["baseline_run_id"]
    assert candidate.data.metrics["paired_delta"] == pytest.approx(payload["paired_delta"])
    assert candidate.data.tags["ima.stage"] == "feature_screen"
    assert client.get_run(payload["baseline_run_id"]).data.tags["ima.stage"] == "control"
    assert client.get_run(payload["build_run_id"]).data.tags["ima.stage"] == "feature_build"
    explorer.retry_tracking()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert len(client.search_runs([experiment.experiment_id])) == 3


def test_cached_control_corruption_fails_closed(tmp_path):
    explorer = explorer_fixture(tmp_path)
    baseline = explorer.baseline()
    predictions = Path(baseline["artifacts"]["predictions"])
    predictions.write_text(predictions.read_text()+"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        explorer.baseline()


def test_candidate_failure_isolated_without_false_registry_merit(tmp_path):
    explorer = explorer_fixture(tmp_path)
    baseline = explorer.baseline()
    failed = dict(baseline, status="failed", error="synthetic candidate failure")
    with patch.object(explorer, "baseline", return_value=baseline), patch.object(explorer, "_execute", return_value=failed):
        result = explorer.step()
    assert result["status"] == "failed"
    assert explorer.status()["counts"]["failed"] == 1
    assert all(not build["comparisons"] for build in catalog(explorer.config.registry)["builds"])


def test_expired_control_execution_gets_fresh_runtime_path(tmp_path):
    from concurrent.futures import Future
    explorer = explorer_fixture(tmp_path)
    paths = []

    class ExpiringExecutor:
        def submit(self, *args, runtime_path, **kwargs):
            paths.append(runtime_path)
            runtime_path.write_text('{"status":"timed_out"}')
            future = Future()
            future.set_exception(TimeoutError("synthetic deadline"))
            return future

    for _ in range(2):
        with pytest.raises(TimeoutError):
            explorer.baseline(ExpiringExecutor())
    assert paths[0] != paths[1]
    assert paths[0].name == paths[1].name == "runtime.json"
    assert paths[0].parent != paths[1].parent
