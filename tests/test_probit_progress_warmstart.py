import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import minimize

from ima.feature_sets import FeatureSchema
from ima.performance_probit import GaussianRaceProbit, _array_digest, fit_probit
from tests.test_pipeline_graph import SCHEMA, race_frame


CONTEXT = {"fold_id": "fold-1", "dataset_digest": "dataset-v1",
           "train_spec": {"cutoff": "2020-01-01"}, "transform_spec": {"version": 1}}


def fit(frame, root, **options):
    return fit_probit(frame, SCHEMA, {"max_iter": 120, **options},
        cache_dir=root, training_context=CONTEXT)


def test_real_callbacks_have_iterations_objectives_evaluations_and_stage_timings():
    events = []
    model = fit_probit(race_frame(16), SCHEMA, {"max_iter": 120},
        progress_callback=events.append, training_context=CONTEXT)
    iterations = [event for event in events if event["stage"] == "optimizing"]
    assert len(iterations) == model.fit_diagnostics["iterations"] > 1
    assert [event["iteration"] for event in iterations] == list(range(1, len(iterations)+1))
    assert all(np.isfinite(event["objective"]) for event in iterations)
    assert iterations[-1]["objective_evaluations"] >= len(iterations)
    assert iterations[-1]["objective"] < iterations[0]["objective"]
    assert any(event["progress_kind"] == "objective_improvement" for event in iterations)
    assert all(np.isfinite(event["gradient_norm"]) for event in iterations)
    assert events[0]["stage"] == "prepared"
    assert events[-1]["stage"] == "fit_completed"
    assert any(event["stage"] == "verifying_quadrature" for event in events)
    for event in events:
        assert event["fold_id"] == "fold-1"
        assert event["updated_at_epoch"] > 0
        for key in ("preparation_seconds", "objective_seconds", "optimizer_seconds",
                    "quadrature_verification_seconds", "fit_seconds"):
            assert np.isfinite(event[key]) and event[key] >= 0
    assert events[-1]["objective_evaluations"] == model.fit_diagnostics["objective_evaluations"]


@pytest.mark.parametrize("heteroscedastic", [False, True])
def test_real_fold_local_reuse_and_regularization_change(tmp_path, heteroscedastic):
    frame = race_frame(16)
    cold = fit(frame, tmp_path, heteroscedastic=heteroscedastic)
    initial_values = []

    def capture(objective, initial, **kwargs):
        initial_values.append(initial.copy())
        return minimize(objective, initial, **kwargs)

    with patch("ima.performance_probit.minimize", side_effect=capture):
        warm = fit(frame, tmp_path, heteroscedastic=heteroscedastic)
    assert warm.fit_diagnostics["warmstart_used"]
    expected = np.r_[cold.coefficients, cold.scale_coefficients] if heteroscedastic else cold.coefficients
    np.testing.assert_array_equal(initial_values[0], expected)
    score = race_frame(3, 30)
    np.testing.assert_allclose(warm.predict(score), cold.predict(score), atol=1e-5)
    changed = fit(frame, tmp_path, heteroscedastic=heteroscedastic, l2=.2, scale_l2=.3)
    assert changed.fit_diagnostics["warmstart_used"]
    assert changed.fit_diagnostics["warmstart_key"] == cold.fit_diagnostics["warmstart_key"]
    assert changed.fit_diagnostics["converged"]
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert set(record) == {"schema_version", "key", "coefficients", "coefficients_hash"}
    assert len(record["coefficients"]) == len(expected)


@pytest.mark.parametrize("change", ["fold", "dataset", "transform", "train_spec", "features", "schema", "x", "y", "rows", "preprocessor", "link"])
def test_mismatched_training_context_or_values_cannot_reuse(tmp_path, change):
    frame = race_frame(10)
    cold = fit(frame, tmp_path)
    context, schema, parameters = dict(CONTEXT), SCHEMA, {"max_iter": 120}
    if change == "fold":
        context["fold_id"] = "fold-2"
    elif change == "dataset":
        context["dataset_digest"] = "dataset-v2"
    elif change == "transform":
        context["transform_spec"] = {"version": 2}
    elif change == "train_spec":
        context["train_spec"] = {"cutoff": "2021-01-01"}
    elif change == "features":
        schema = FeatureSchema(SCHEMA.name, tuple(reversed(SCHEMA.numeric)), ())
    elif change == "schema":
        schema = FeatureSchema("new-schema", SCHEMA.numeric, ())
    elif change == "x":
        frame.loc[frame.index[0], "ability"] += .1
    elif change == "y":
        frame.loc[frame.race_id == frame.race_id.iloc[0], "target_win"] = [1, 0, 0, 0]
        if np.array_equal(frame.target_win, race_frame(10).target_win):
            frame.loc[frame.race_id == frame.race_id.iloc[0], "target_win"] = [0, 1, 0, 0]
    elif change == "rows":
        frame = race_frame(10, 1)
    elif change == "preprocessor":
        # Scaling raw X preserves standardized X but changes fitted preprocessing.
        frame["ability"] *= 2
    elif change == "link":
        parameters["heteroscedastic"] = True
    model = fit_probit(frame, schema, parameters, cache_dir=tmp_path, training_context=context)
    assert not model.fit_diagnostics["warmstart_used"]
    assert model.fit_diagnostics["warmstart_key"] != cold.fit_diagnostics["warmstart_key"]


@pytest.mark.parametrize("corruption", ["hash", "nan", "shape", "bool", "key", "extra", "huge", "broken", "wrong_version"])
def test_numeric_cache_is_validated_and_corruption_falls_back_to_cold(tmp_path, corruption):
    frame = race_frame(10)
    cold = fit(frame, tmp_path)
    path = tmp_path/f'{cold.fit_diagnostics["warmstart_key"]}.json'
    record = json.loads(path.read_text())
    if corruption == "hash":
        record["coefficients_hash"] = "bad-hash"
    elif corruption == "nan":
        record["coefficients"][0] = float("nan")
    elif corruption == "shape":
        record["coefficients"].append(0.)
    elif corruption == "bool":
        record["coefficients"][0] = True
    elif corruption == "key":
        record["key"] = "wrong-key"
    elif corruption == "extra":
        record["target_win"] = [1, 0]
    elif corruption == "huge":
        record["coefficients"][0] = 1e10
        record["coefficients_hash"] = _array_digest(record["coefficients"])
    elif corruption == "wrong_version":
        record["schema_version"] = 2
    path.write_text("broken" if corruption == "broken" else json.dumps(record))
    model = fit(frame, tmp_path)
    assert not model.fit_diagnostics["warmstart_used"]
    assert json.loads(path.read_text())["coefficients_hash"] == _array_digest(model.coefficients)


def test_eval_labels_are_never_inputs_to_cache_and_predictions_do_not_modify_it(tmp_path):
    frame = race_frame(10)
    fit(frame, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    model = fit(frame, tmp_path)
    evaluation = race_frame(3, 30)
    expected = model.predict(evaluation)
    evaluation["target_win"] = 999
    np.testing.assert_array_equal(model.predict(evaluation), expected)
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == before


def test_parent_dataset_hash_context_is_supported(tmp_path):
    context = dict(CONTEXT)
    context["dataset_hash"] = context.pop("dataset_digest")
    model = fit_probit(race_frame(10), SCHEMA, {"max_iter": 120},
                       cache_dir=tmp_path, training_context=context)
    assert model.fit_diagnostics["warmstart_key"]


def test_missing_fold_identity_rejected_and_failed_fit_cannot_publish_cache(tmp_path):
    frame = race_frame(10)
    for context in (None, {}, {"dataset_digest": "x"}, {"fold_id": "fold-1"}):
        with pytest.raises(ValueError, match="Warm starts require"):
            fit_probit(frame, SCHEMA, cache_dir=tmp_path, training_context=context)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_probit(frame, SCHEMA, {"max_iter": 1}, cache_dir=tmp_path, training_context=CONTEXT)
    assert not list(tmp_path.glob("*.json"))
    model = GaussianRaceProbit(SCHEMA.numeric, max_iter=120).fit(frame)
    model.max_iter = 1
    with pytest.raises(RuntimeError):
        model.fit(frame)
    assert model.coefficients is None
