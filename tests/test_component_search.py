import copy
import json

import numpy as np
import pytest

from ima.pipeline_graph import PipelineGraph, fit_graph
from ima.research_search import ProgramSearchController, _default_space, component_controls, tune_component
from ima.research_specs import PipelineRecipe, ResearchProposal
from tests.test_pipeline_graph import SCHEMA, graph_spec, race_frame


def reference():
    spec = graph_spec()
    spec["nodes"][-1]["parameters"]["weights"] = [.35, .65]
    return PipelineRecipe(schema_version=3, pipeline_graph=spec,
                          model={"kind": "benter_conditional_logit", "parameters": {"l2": .2}})


def test_tune_binding_changes_only_intended_estimator_and_reports_actual_fit():
    recipe = tune_component(reference(), "boosted")
    assert recipe.model.kind == "boosted"
    assert recipe.pipeline_graph["primary_node_id"] == "benter"
    assert PipelineGraph.from_dict(recipe.pipeline_graph).tuned_estimator_id == "boosted"
    fitted = fit_graph(recipe.pipeline_graph, race_frame(10), None, SCHEMA,
                       model_spec={"kind": "boosted", "parameters": {"learning_rate": .07, "max_iter": 15}})
    benter, boosted = fitted.output.parents
    assert benter.state.conditional.l2 == .2
    assert boosted.state.parameters["learning_rate"] == .07
    assert boosted.state.parameters["max_iter"] == 15
    reports = {row["node_id"]: row for row in fitted.fit_report["component_fits"]}
    assert reports["boosted"]["tuned"] and not reports["benter"]["tuned"]
    assert reports["boosted"]["effective_parameters"]["model_parameters"]["max_iter"] == 15
    assert reports["boosted"]["features"] == list(SCHEMA.features)
    assert reports["boosted"]["wall_seconds"] > 0
    np.testing.assert_allclose(fitted.predict_proba(race_frame(2, 10)),
        .35*benter.predict(race_frame(2, 10))+.65*boosted.predict(race_frame(2, 10)))


def test_controls_execute_actual_pure_equal_and_reference_graphs():
    recipe = reference()
    original = copy.deepcopy(recipe.pipeline_graph)
    controls = component_controls(recipe)
    fitted = {name: fit_graph(control.pipeline_graph, race_frame(10), None, SCHEMA,
                              model_spec=control.model) for name, control in controls.items()}
    score = race_frame(2, 10)
    benter = fitted["pure_benter"].predict_proba(score)
    boosted = fitted["pure_boosted"].predict_proba(score)
    assert fitted["pure_benter"].fit_report["physical_fits"] == 1
    assert fitted["pure_boosted"].fit_report["physical_fits"] == 1
    np.testing.assert_allclose(fitted["fixed"].predict_proba(score), .35*benter+.65*boosted)
    np.testing.assert_allclose(fitted["fixed_champion"].predict_proba(score), .35*benter+.65*boosted)
    np.testing.assert_allclose(fitted["equal"].predict_proba(score), .5*benter+.5*boosted)
    assert recipe.pipeline_graph == original


def test_bad_tune_nodes_and_mismatched_families_fail():
    with pytest.raises(ValueError, match="estimator"):
        tune_component(reference(), "pool")
    payload = reference().model_dump(mode="json")
    payload["pipeline_graph"]["tune_node_id"] = "boosted"
    with pytest.raises(ValueError, match="family"):
        PipelineRecipe.model_validate(payload)
    payload["pipeline_graph"]["tune_node_id"] = "absent"
    with pytest.raises(ValueError, match="tune_node_id"):
        PipelineRecipe.model_validate(payload)


def test_boosted_study_budget_survives_restart_and_single_slot_refill(tmp_path):
    recipe = tune_component(reference(), "boosted")
    proposal = ResearchProposal(proposal_id="boosted-primary", hypothesis="Tune the dominant component.",
        changed_axes=("hyperparameters",), recipe=recipe, max_trials=4,
        expected_observation="Incremental paired change", falsification_rule="No development improvement")
    controller = ProgramSearchController(tmp_path)
    program_id = controller.register(proposal)
    first = controller.ask(1)[0]
    assert first.recipe.model.kind == "boosted"
    assert first.recipe.model.parameters == recipe.model.parameters
    study = controller.studies[program_id]
    assert study.trials[0].user_attrs["tune_node_id"] == "boosted"
    assert study.trials[0].user_attrs["original_graph_control"]
    controller.tell(program_id, first.trial_number, 2.1)
    # A scheduler's one-slot refill and temporary pause do not shrink the study allocation.
    reloaded = ProgramSearchController(tmp_path)
    assert reloaded.snapshot()["programs"][program_id]["budget"] == 4
    assert reloaded.ask(1, allowed_program_ids=[]) == []
    remaining = reloaded.ask(3)
    assert len(remaining) == 3
    assert reloaded.ask(1) == []
    for suggestion in remaining:
        assert set(suggestion.recipe.model.parameters) >= set(_default_space("boosted"))
        assert suggestion.recipe.model_dump(mode="json")["pipeline_graph"] == recipe.model_dump(mode="json")["pipeline_graph"]
        json.dumps(suggestion.serializable(), allow_nan=False)


def test_boosted_boundary_space_with_original_controls():
    recipe = tune_component(reference(), "boosted")
    for name, dimension in _default_space("boosted").items():
        for value in (dimension.low, dimension.high):
            payload = recipe.model_dump(mode="json")
            payload["model"]["parameters"][name] = int(value) if dimension.kind == "int" else value
            PipelineRecipe.model_validate(payload)
    assert component_controls(recipe)["fixed"].model.parameters["max_iter"] == 12
