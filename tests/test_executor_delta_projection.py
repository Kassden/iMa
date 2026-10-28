import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from ima.dataset_delta import digest, read_feature_table, write_feature_delta
from ima.research_executor import _load_dataset, _load_request_dataset
from ima.research_specs import PipelineRecipe


def fixture(tmp_path):
    registry=tmp_path/'registry'
    parent=registry/'datasets/parent';parent.mkdir(parents=True)
    frame=pd.DataFrame({'race_id':['a','a'],'horse_no':[1,2],'race_no':[1,1],
        'date':['2020-01-01']*2,'target_win':[1,0],'horse_rating':[40.,50.],
        'unregistered_historical_sidechannel':[.1,.2]})
    frame.to_parquet(parent/'features.parquet',index=False)
    manifest={'dataset_id':'parent','features_path':str(parent/'features.parquet'),
        'files':{'features.parquet':digest(parent/'features.parquet')},'predictor_catalog':{}}
    (parent/'manifest.json').write_text(json.dumps(manifest))
    enriched=frame.assign(useful=frame.horse_rating*2,unused=frame.horse_rating/2)
    metadata={'unit':'1','dtype':'float64','temporal_scope':'pre_race','target_tainted':False,
        'eligible':True,'source_family':'validated_formula','availability_proof':'verified fixture inputs'}
    child=registry/'datasets/child'
    write_feature_delta(parent/'manifest.json',child,enriched,registry_root=registry,
        columns=['useful','unused'],catalog={'useful':metadata,'unused':metadata})
    return child/'features.delta.json'


def test_recipe_projects_unused_delta_before_parquet_decode_preserving_parent(tmp_path):
    path=fixture(tmp_path)
    request=SimpleNamespace(dataset_path=path,recipe=PipelineRecipe(schema_version=3,extra_numeric_features=('useful',)))
    calls=[]
    from ima import dataset_delta
    original=dataset_delta._read_parquet_positions
    def read(source,columns,positions):
        calls.append((str(source),columns,positions))
        return original(source,columns,positions)
    with patch.object(dataset_delta,'_read_parquet_positions',side_effect=read):
        projected=_load_request_dataset(request)
    expected=_load_dataset(path).drop(columns=['unused'])
    pd.testing.assert_frame_equal(projected,expected)
    assert 'unregistered_historical_sidechannel' in projected
    delta_calls=[(columns,positions) for source,columns,positions in calls if source.endswith('feature-delta.parquet')]
    assert delta_calls==[(['race_id','horse_no'],None),(['race_id','horse_no','useful'],None)]


def test_fold_position_loader_keeps_transitive_binding_and_caller_order(tmp_path):
    path=fixture(tmp_path)
    metadata={'unit':'1','dtype':'float64','temporal_scope':'pre_race',
        'target_tainted':False,'eligible':True,'available_at_column':'unused'}
    manifest={'dataset_id':'fixture-child','files':{'features.delta.json':digest(path)},
        'predictor_catalog':{'useful':metadata}}
    (path.parent/'manifest.json').write_text(json.dumps(manifest))
    request=SimpleNamespace(dataset_path=path,
        recipe=PipelineRecipe(schema_version=3,extra_numeric_features=('useful',)))
    projected=_load_request_dataset(request,positions=[1,0,1],fold_columns=[
        'race_id','horse_no','race_no','date','target_win','target_probability','useful'])
    assert projected[['race_id','horse_no']].values.tolist()==[['a',2],['a',1],['a',2]]
    assert 'unused' in projected
    assert 'unregistered_historical_sidechannel' not in projected


def test_transform_recipe_keeps_full_dependency_population(tmp_path):
    path=fixture(tmp_path)
    recipe=PipelineRecipe(transforms=({'kind':'clip_numeric_quantiles','parameters':{'columns':['unused']}},))
    pd.testing.assert_frame_equal(_load_request_dataset(SimpleNamespace(dataset_path=path,recipe=recipe)),_load_dataset(path))


def test_projection_still_verifies_unused_delta_content_integrity(tmp_path):
    path=fixture(tmp_path)
    delta=path.parent/'feature-delta.parquet'
    frame=pd.read_parquet(delta);frame['unused']+=1;frame.to_parquet(delta,index=False)
    with pytest.raises(ValueError,match='Delta content changed'):
        _load_request_dataset(SimpleNamespace(dataset_path=path,recipe=PipelineRecipe(schema_version=3,extra_numeric_features=('useful',))))


def test_retained_predictor_keeps_delta_availability_binding(tmp_path):
    path=fixture(tmp_path)
    # Numeric availability clocks are supported by the registered vocabulary.
    # Bind useful to the otherwise unused numeric column, so its availability
    # mask cannot disappear merely because that column is not a model feature.
    metadata={'unit':'1','dtype':'float64','temporal_scope':'pre_race',
        'target_tainted':False,'eligible':True,'available_at_column':'unused'}
    manifest={'dataset_id':'fixture-child','files':{'features.delta.json':digest(path)},
        'predictor_catalog':{'useful':metadata}}
    (path.parent/'manifest.json').write_text(json.dumps(manifest))
    request=SimpleNamespace(dataset_path=path,recipe=PipelineRecipe(schema_version=3,extra_numeric_features=('useful',)))
    projected=_load_request_dataset(request)
    assert 'unused' in projected
    from ima.research_executor import _v6_feature_frame
    selected=_v6_feature_frame(projected,request)
    assert selected['useful'].notna().all()


def test_nested_delta_retains_declared_parent_replacements(tmp_path):
    path=fixture(tmp_path)
    metadata={'unit':'1','dtype':'float64','temporal_scope':'pre_race','target_tainted':False,
        'eligible':True,'source_family':'validated_formula','availability_proof':'verified fixture inputs'}
    (path.parent/'manifest.json').write_text(json.dumps({'dataset_id':'child',
        'features_path':str(path),'files':{'features.delta.json':digest(path)},
        'predictor_catalog':{'unused':metadata}}))
    frame=read_feature_table(path)
    frame['unused']+=10;frame['spare']=1.
    grandchild=path.parent.parent/'grandchild'
    write_feature_delta(path.parent/'manifest.json',grandchild,frame,registry_root=path.parent.parent.parent,
        columns=['unused','spare'],replacements=['unused'],catalog={'unused':metadata,'spare':metadata})
    source=grandchild/'features.delta.json'
    request=SimpleNamespace(dataset_path=source,recipe=PipelineRecipe())
    pd.testing.assert_frame_equal(_load_request_dataset(request),_load_dataset(source).drop(columns=['spare']))
