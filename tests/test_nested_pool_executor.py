"""Actual recipe execution, immutable selector lineage and serialized inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ima.feature_sets import FEATURE_SCHEMAS
from ima.research_executor import RecipeExecutionRequest, execute_recipe
from ima.research_model_package import load_research_package
from ima.research_specs import FUNDAMENTAL_FIRST_PORTFOLIO_VERSION, PipelineRecipe
from tests.test_research_executor_v6 import dataset


def nested_graph():
    candidates=[]
    for name, feature in [('rating', 'horse_rating'), ('form', 'prior_win_rate')]:
        candidates.append({'candidate_id': name, 'graph': {
            'graph_id': name, 'primary_node_id': 'primary', 'output_node_id': 'primary',
            'horse_column': 'horse_no', 'nodes': [
                {'node_id': 'view', 'kind': 'feature_view', 'parameters': {'features': [feature]},
                 'output': {'kind': 'feature_frame', 'target': 'features', 'unit': 'features'}},
                {'node_id': 'primary', 'kind': 'estimator', 'inputs': ['view'],
                 'parameters': {'model_kind':'benter_conditional_logit','model_parameters':{'l2':.1,'max_iter':80}}},
                {'node_id': 'secondary', 'kind': 'estimator', 'inputs': ['view'],
                 'parameters': {'model_kind':'benter_conditional_logit','model_parameters':{'l2':1.,'max_iter':80}}},
            ]}})
    protocol={'schema_version':1,'selection_cutoff':'2020-01-12','validation_start':'2020-01-09',
        'min_prefix_train_meetings':8,'min_prefix_validation_meetings':4,
        'registered_features':list(FEATURE_SCHEMAS['baseline-v1'].features),'candidate_graphs':candidates,
        'selection_trial_budget':2,'seed':42,'objective':'mean_meeting_log_loss'}
    return {'graph_id':'nested-executor','primary_node_id':'primary','output_node_id':'pool',
        'horse_column':'horse_no','n_splits':2,'min_train_dates':2,'nodes':[
            {'node_id':'view','kind':'feature_view','parameters':{'features':['horse_rating']},
             'output':{'kind':'feature_frame','target':'features','unit':'features'}},
            {'node_id':'primary','kind':'estimator','inputs':['view'],'parameters':{'model_kind':'benter_conditional_logit'}},
            {'node_id':'secondary','kind':'estimator','inputs':['view'],'parameters':{'model_kind':'benter_conditional_logit'}},
            {'node_id':'pool','kind':'learned_probability_pool','inputs':['primary','secondary'],
             'parameters':{'selection_mode':'nested_selection','selection_cutoff':'2020-01-12',
                           'selection_protocol':protocol}},
        ]}


def test_nested_selection_runs_through_executor_and_saved_package(tmp_path):
    source, _=dataset(tmp_path, races=40)
    recipe=PipelineRecipe(schema_version=3,feature_schema='baseline-v1',
        model={'kind':'benter_conditional_logit','parameters':{}},pipeline_graph=nested_graph())
    request=RecipeExecutionRequest('nested','nested-executor',0,recipe,source,tmp_path/'trials/nested',
        {'min_train_races':25,'calibration_races':3,'score_races':3,'max_folds':1,'whole_meeting_boundaries':True},
        code_revision='actual-test-source',environment_hash='fixture',portfolio_version=FUNDAMENTAL_FIRST_PORTFOLIO_VERSION)
    result=execute_recipe(request)
    assert result.status=='completed', result.error
    assert np.isfinite(result.objective_value)
    fits=json.loads(Path(result.artifacts['graph_fits']).read_text())
    selection=fits['folds'][0]['selections'][0]
    artifact=json.loads(Path(selection['artifact_path']).read_text())
    assert artifact['source']['dataset_hash']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact['source']['dataset_manifest_hash']==hashlib.sha256((tmp_path/'manifest.json').read_bytes()).hexdigest()
    assert artifact['source']['source_revision']=='actual-test-source'
    assert artifact['source']['target_contract']['kind']=='win_probability'
    assert artifact['scope']['independent_confirmation'] is False
    assert fits['folds'][0]['learned_pools'][0]['selection_mode']=='nested_selection'
    package=load_research_package(result.artifacts['package'])
    manifest=json.loads((Path(result.artifacts['package'])/'manifest.json').read_text())
    assert manifest['selection_references'][0]['selection_sha256']==artifact['selection_sha256']
    # Use the exact dataset loading contract and compare serialized probabilities.
    import pandas as pd
    score=pd.read_csv(result.artifacts['predictions'])
    frame=pd.read_csv(source);frame['date']=pd.to_datetime(frame.date)
    frame=frame.loc[frame.race_id.isin(score.race_id)]
    probabilities=package.predict_proba(frame)
    np.testing.assert_allclose(probabilities,score.model_probability,rtol=1e-10,atol=1e-12)


def test_nested_candidates_cannot_silently_override_outer_tuning(tmp_path):
    source, _=dataset(tmp_path, races=40)
    recipe=PipelineRecipe(schema_version=3,feature_schema='baseline-v1',
        model={'kind':'benter_conditional_logit','parameters':{'max_iter':81}},pipeline_graph=nested_graph())
    request=RecipeExecutionRequest('conflict','nested-executor',0,recipe,source,tmp_path/'trials/conflict',
        {'min_train_races':25,'calibration_races':3,'score_races':3,'max_folds':1},code_revision='source')
    result=execute_recipe(request)
    assert result.status=='failed'
    assert 'Outer tuning conflicts' in result.error
    assert not list(request.output_dir.glob('nested-selection/*/selection-*.json'))
