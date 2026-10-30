from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import pandas as pd
import pytest

import ima.research_storage as storage


def authority_at(tmp_path,available):
    return storage.StorageAuthority(tmp_path/'control',headroom_bytes=0,headroom_inodes=0,
        statvfs=lambda path:SimpleNamespace(f_frsize=1,f_blocks=available,f_bavail=available,
            f_bfree=available,f_favail=100000))


def test_forward_oof_exports_are_admitted_and_replay_without_pickle(tmp_path,monkeypatch):
    import numpy as np
    from ima.prediction_store import PredictionArtifact,PredictionStore,prediction_key
    authority=authority_at(tmp_path,10*1024**2)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    metadata={'fit_scope':'forward_oof','training_cutoff':'2020-01-01','prediction_start':'2020-01-02'}
    artifact=PredictionArtifact(np.array([.4,.6]),(('r','1'),('r','2')),metadata)
    store=PredictionStore(tmp_path/'oof')
    key=prediction_key(metadata)
    with patch.object(authority,'reserve',wraps=authority.reserve) as reserve:
        store.put(key,artifact)
        assert reserve.call_args.kwargs['stage']=='forward_oof'
    np.testing.assert_array_equal(store.get(key,artifact.row_keys).values,artifact.values)
    assert authority.status()['reserved_bytes']==0


def test_denied_forward_oof_admission_creates_no_prediction_payload(tmp_path,monkeypatch):
    import numpy as np
    from ima.prediction_store import PredictionArtifact,PredictionStore,prediction_key
    authority=authority_at(tmp_path,10000)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    metadata={'fit_scope':'forward_oof','training_cutoff':'2020-01-01','prediction_start':'2020-01-02'}
    artifact=PredictionArtifact(np.array([.4,.6]),(('r','1'),('r','2')),metadata)
    root=tmp_path/'oof';root.mkdir()
    store=PredictionStore(root)
    with pytest.raises(storage.StorageBlocked):
        store.put(prediction_key(metadata),artifact)
    assert list(store.root.iterdir())==[]


def test_growth_is_denied_before_the_next_export_write(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,5000)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    destination=tmp_path/'exports';destination.mkdir()
    @storage.storage_writer(stage='fixture',output=lambda a:destination,estimate=lambda a:16)
    def export():
        with storage.reserved_output_file(destination/'partial.bin') as stream:
            stream.write(b'a'*16)
            stream.flush()
            assert authority.status()['reserved_bytes']==0
            stream.write(b'b'*17)
    with pytest.raises(storage.StorageBlocked):export()
    assert (destination/'partial.bin').read_bytes()==b'a'*16
    assert authority.status()['reserved_bytes']==0


def test_pandas_csv_and_parquet_roundtrip_through_admitted_streams(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,10*1024**2)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    destination=tmp_path/'exports';destination.mkdir()
    frame=pd.DataFrame({'horse':[1,2],'probability':[.2,.8]})
    @storage.storage_writer(stage='fixture',output=lambda a:destination,estimate=lambda a:1)
    def export():
        with storage.reserved_output_file(destination/'predictions.csv','w') as stream:
            frame.to_csv(stream,index=False)
        with storage.reserved_output_file(destination/'features.parquet') as stream:
            frame.to_parquet(stream,index=False)
    with patch.object(authority,'extend',wraps=authority.extend) as extend:
        export()
        assert extend.call_count>=1
    pd.testing.assert_frame_equal(frame,pd.read_csv(destination/'predictions.csv'))
    pd.testing.assert_frame_equal(frame,pd.read_parquet(destination/'features.parquet'))
    assert authority.status()['reserved_bytes']==0


def test_export_cannot_escape_the_admitted_root(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,10000)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    root=tmp_path/'exports';root.mkdir()
    @storage.storage_writer(stage='fixture',output=lambda a:root,estimate=lambda a:64)
    def export():
        with storage.reserved_output_file(tmp_path/'outside.bin') as stream:stream.write(b'no')
    with pytest.raises(ValueError,match='escaped'):export()
    assert not (tmp_path/'outside.bin').exists()


def test_real_model_package_replays_after_admitted_joblib_export(tmp_path,monkeypatch):
    import numpy as np
    from ima.research_executor import RecipeExecutionRequest,execute_recipe
    from ima.research_model_package import load_research_package
    from ima.research_specs import PipelineRecipe
    from tests.test_research_executor_v6 import dataset
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    source,frame=dataset(tmp_path)
    recipe=PipelineRecipe(schema_version=3,model={'kind':'benter_conditional_logit'},blend={'kind':'none'})
    request=RecipeExecutionRequest('fit','fixture',0,recipe,source,tmp_path/'trials'/'fit',
        {'min_train_races':12,'calibration_races':3,'score_races':3,'max_folds':1})
    result=execute_recipe(request)
    assert result.status=='completed',result.error
    package=load_research_package(Path(result.artifacts['package']))
    predictions=pd.read_csv(result.artifacts['predictions'])
    score=frame.loc[frame.race_id.isin(predictions.race_id)].reset_index(drop=True)
    expected=package.predict(score)
    authority=authority_at(tmp_path,128*1024**2)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    with patch.object(authority,'consume',wraps=authority.consume) as consume:
        saved=package.save(tmp_path/'admitted-package')
        assert consume.call_count>=1
    np.testing.assert_array_equal(load_research_package(saved).predict(score),expected)
    assert authority.status()['reserved_bytes']==0


def test_hardlinked_export_is_rejected_before_truncating_protected_data(tmp_path,monkeypatch):
    import os
    authority=authority_at(tmp_path,10000)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    root=tmp_path/'exports';root.mkdir()
    protected=tmp_path/'protected.bin';protected.write_bytes(b'keep this model')
    os.link(protected,root/'alias.bin')
    @storage.storage_writer(stage='fixture',output=lambda a:root,estimate=lambda a:64)
    def export():
        with storage.reserved_output_file(root/'alias.bin') as stream:stream.write(b'bad')
    with pytest.raises(ValueError,match='Unsafe'):export()
    assert protected.read_bytes()==b'keep this model'


@pytest.mark.parametrize('kind',['csv','parquet','joblib'])
def test_actual_serializers_propagate_denied_growth_as_storage_pressure(tmp_path,monkeypatch,kind):
    import joblib
    authority=authority_at(tmp_path,5000)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    root=tmp_path/'exports';root.mkdir()
    frame=pd.DataFrame({'horse':range(512),'probability':[.01]*512})
    path=root/('partial.'+kind)
    @storage.storage_writer(stage='fixture',output=lambda a:root,estimate=lambda a:64)
    def export():
        with storage.reserved_output_file(path,'w' if kind=='csv' else 'wb') as stream:
            if kind=='csv':frame.to_csv(stream,index=False)
            elif kind=='parquet':frame.to_parquet(stream,index=False)
            else:joblib.dump(frame,stream)
    with pytest.raises(storage.StorageBlocked):export()
    assert path.stat().st_size<=64
    assert authority.status()['reserved_bytes']==0


def test_full_recipe_exports_and_package_replay_with_storage_admission_enabled(tmp_path,monkeypatch):
    import numpy as np
    from ima.research_executor import RecipeExecutionRequest,execute_recipe
    from ima.research_model_package import load_research_package
    from ima.research_specs import PipelineRecipe
    from tests.test_research_executor_v6 import dataset
    source,frame=dataset(tmp_path)
    authority=authority_at(tmp_path,512*1024**2)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    recipe=PipelineRecipe(schema_version=3,model={'kind':'benter_conditional_logit'},blend={'kind':'none'})
    request=RecipeExecutionRequest('admitted','fixture',0,recipe,source,tmp_path/'trials'/'admitted',
        {'min_train_races':12,'calibration_races':3,'score_races':3,'max_folds':1})
    result=execute_recipe(request)
    assert result.status=='completed',result.error
    predictions=pd.read_csv(result.artifacts['predictions'])
    score=frame.loc[frame.race_id.isin(predictions.race_id)].reset_index(drop=True)
    package=load_research_package(Path(result.artifacts['package']))
    np.testing.assert_allclose(package.predict(score),predictions.model_probability,rtol=1e-10,atol=1e-12)
    assert authority.status()['reserved_bytes']==0


def dataset_request_fixture(root):
    from ima.dataset_registry import DatasetRegistry
    from tests.test_dataset_registry_adversarial import source_rows,event,digest
    snapshot=root/'snapshot';snapshot.mkdir()
    rows=source_rows()
    rows.to_parquet(snapshot/'runners.parquet',index=False)
    (snapshot/'events.jsonl').write_text(json.dumps(event('guarded-workout'))+'\n')
    raw=snapshot/'manifest.json'
    raw.write_text(json.dumps({'source_policy':'HKJC-only; storage fixture',
        'rows':len(rows),'races':int(rows.race_id.nunique()),
        'files':{name:digest(snapshot/name) for name in ('runners.parquet','events.jsonl')}}))
    registry=DatasetRegistry(root/'registry')
    registry.submit({'request_id':'guarded-build','raw_corpus_manifest_id':digest(raw),
        'rationale':'Verify output admission','evidence_watermark':'fixture',
        'protocol':{'min_train_races':2,'calibration_races':1,'score_races':1,
                    'max_folds':2,'final_confirmation_races':2}})
    return registry,snapshot,raw


def test_dataset_admission_denial_precedes_materialization(tmp_path,monkeypatch):
    registry,snapshot,raw=dataset_request_fixture(tmp_path)
    authority=authority_at(tmp_path,8*1024**2)
    initial=registry.get('guarded-build')['status']
    before=hashlib.sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    with patch.object(pd,'read_parquet',side_effect=AssertionError('decoded before admission')):
        with pytest.raises(storage.StorageBlocked):
            registry.build('guarded-build',source_snapshot=snapshot,raw_manifest=raw)
    assert registry.get('guarded-build')['status']==initial
    assert list((registry.root/'datasets').iterdir())==[]
    assert list((registry.root/'staging').iterdir())==[]
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==before


def test_dataset_build_roundtrip_with_admitted_parquet_exports(tmp_path,monkeypatch):
    registry,snapshot,raw=dataset_request_fixture(tmp_path)
    authority=authority_at(tmp_path,512*1024**2)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    manifest=registry.build('guarded-build',source_snapshot=snapshot,raw_manifest=raw)
    assert registry.verify(manifest['dataset_id'])['status']=='verified'
    assert authority.status()['reserved_bytes']==0


def test_object_snapshot_estimation_uses_declared_parquet_metadata(tmp_path):
    from ima.research_objects import ObjectStore,snapshot_storage_estimate
    store=ObjectStore(tmp_path/'objects')
    frame=pd.DataFrame({'value':[1.]*1000})
    table=store.publish_table(frame)
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    (snapshot/'manifest.json').write_text(json.dumps({'storage_format':'ima.snapshot.objects.v2',
        'tables':{'runners.parquet':table}}))
    (snapshot/'events.jsonl').write_text('')
    with patch.object(pd,'read_parquet',side_effect=AssertionError('decoded for estimate')):
        estimate= snapshot_storage_estimate(snapshot)
    assert estimate>=1000*16


@pytest.mark.parametrize('representation',['legacy','delta-v2'])
def test_feature_successors_use_the_registry_admission_root(tmp_path,monkeypatch,representation):
    from ima.feature_definitions import FeatureDefinition
    from ima.feature_lifecycle import build_feature_candidate
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    registry,snapshot,raw=dataset_request_fixture(tmp_path)
    parent=registry.build('guarded-build',source_snapshot=snapshot,raw_manifest=raw)
    parent_dir=registry.dataset_path(parent['dataset_id'])
    authority=authority_at(tmp_path,4*1024**3)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setenv('IMA_DATASET_STORAGE_FORMAT',representation)
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    definition=FeatureDefinition(name='guarded_carried_fraction',unit='1',
        input_refs=('actual_weight','declared_weight'),expression_ast={'op':'divide','args':[
            {'op':'ref','ref':'actual_weight'},{'op':'ref','ref':'declared_weight'}]})
    successor=build_feature_candidate(parent_dir/'features.parquet',parent_dir/'manifest.json',registry.root,
                                      definitions=[definition])
    verified=registry.verify(successor['dataset_id'])
    assert verified['ordered_row_keys_sha256']==parent['ordered_row_keys_sha256']
    assert definition.feature_id in verified['predictor_catalog']
    assert authority.status()['reserved_bytes']==0
