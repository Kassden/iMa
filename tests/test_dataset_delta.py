import json
import errno
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import pandas as pd
import pytest
import ima.dataset_delta as delta_module
from ima.dataset_delta import digest,read_feature_table,write_feature_delta,dataset_storage_estimate


def parent_fixture(tmp_path):
    registry=tmp_path/'registry'; parent=registry/'datasets/dataset-parent'; parent.mkdir(parents=True)
    frame=pd.DataFrame({'race_id':['a','a'],'horse_no':[1,2],'horse_id':['h1','h2'],
                        'target_win':[1,0],'x':[2.,3.]})
    frame.to_parquet(parent/'features.parquet',index=False)
    manifest={'dataset_id':'dataset-parent','features_path':str(parent/'features.parquet'),
        'files':{'features.parquet':digest(parent/'features.parquet')},
        'predictor_catalog':{'x':{'unit':'1','temporal_scope':'pre_race'}}}
    (parent/'manifest.json').write_text(json.dumps(manifest))
    return registry,parent,frame


def predictor(unit='1', **overrides):
    return {'unit':unit,'dtype':'float64','temporal_scope':'pre_race',
            'target_tainted':False,'eligible':True,'source_family':'validated_formula',
            'availability_proof':'typed formula over verified pre-race inputs'} | overrides


def test_estimate_counts_decoded_dictionary_encoded_numeric_columns(tmp_path):
    import numpy as np
    frame=pd.DataFrame({f'x{i}':np.arange(10000,dtype=float)%2 for i in range(64)})
    path=tmp_path/'encoded.parquet'; frame.to_parquet(path,index=False)
    assert dataset_storage_estimate(path)>=frame.to_numpy().nbytes


def test_delta_projection_and_corruption_checks(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    child=registry/'datasets/dataset-child'
    enriched=frame.assign(y=frame.x*2)
    write_feature_delta(parent/'manifest.json',child,enriched,registry_root=registry,
        columns=['y'],catalog={'y':predictor()})
    assert not (child/'features.parquet').exists()
    assert dataset_storage_estimate(child/'features.delta.json')>=dataset_storage_estimate(parent/'features.parquet')
    pd.testing.assert_frame_equal(read_feature_table(child/'features.delta.json'),enriched)
    pd.testing.assert_frame_equal(read_feature_table(child/'features.delta.json',columns=['y'],positions=[1]),enriched[['y']].iloc[[1]].reset_index(drop=True))
    (child/'feature-delta.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError): read_feature_table(child/'features.delta.json')


def test_row_positions_preserve_order_duplicates_and_reject_invalid_values(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    child=registry/'datasets/position-child'
    write_feature_delta(parent/'manifest.json',child,frame.assign(y=frame.x*2),
        registry_root=registry,columns=['y'],catalog={'y':predictor()})
    path=child/'features.delta.json'
    expected=frame.assign(y=frame.x*2)[['race_id','horse_no','y']].iloc[[1,0,1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(read_feature_table(path,columns=['race_id','horse_no','y'],positions=[1,0,1]),expected)
    for invalid in ([-1], [2], [True], [0.0]):
        with pytest.raises(ValueError,match='position|integers'):
            read_feature_table(path,columns=['y'],positions=invalid)


def test_projected_read_checks_declared_parent_manifest_key_hash(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    manifest_path=parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['ordered_row_keys_sha256']='0'*64
    manifest_path.write_text(json.dumps(manifest))
    child=registry/'datasets/key-hash-child'
    with pytest.raises(ValueError,match='manifest ordered runner identity'):
        write_feature_delta(manifest_path,child,frame.assign(y=frame.x+1),registry_root=registry,
            columns=['y'],catalog={'y':predictor()})
    assert not child.exists()


def test_projected_delta_reads_rows_before_pandas_and_keeps_full_key_validation(tmp_path,monkeypatch):
    import pyarrow.parquet as pq
    registry=tmp_path/'registry'; parent=registry/'datasets/root'; parent.mkdir(parents=True)
    count=40
    frame=pd.DataFrame({'race_id':[f'r{i:02}' for i in range(count)],'horse_no':[1]*count,
        'horse_id':[f'h{i}' for i in range(count)],'target_win':[i%2 for i in range(count)],
        'x':range(count)})
    pq.write_table(__import__('pyarrow').Table.from_pandas(frame,preserve_index=False),
                   parent/'features.parquet',row_group_size=5)
    (parent/'manifest.json').write_text(json.dumps({'dataset_id':'root',
        'features_path':str(parent/'features.parquet'),
        'files':{'features.parquet':digest(parent/'features.parquet')},'predictor_catalog':{}}))
    metadata=predictor()
    child=registry/'datasets/child'
    first=frame.assign(y=frame.x*2)
    write_feature_delta(parent/'manifest.json',child,first,registry_root=registry,
        columns=['y'],catalog={'y':metadata})
    child_manifest={'dataset_id':'child','features_path':str(child/'features.delta.json'),
        'files':{'features.delta.json':digest(child/'features.delta.json')},'predictor_catalog':{'y':metadata}}
    (child/'manifest.json').write_text(json.dumps(child_manifest))
    grandchild=registry/'datasets/grandchild'
    full=first.assign(z=first.y+1)
    write_feature_delta(child/'manifest.json',grandchild,full,registry_root=registry,
        columns=['z'],catalog={'z':metadata})
    source=grandchild/'features.delta.json'
    positions=[31,2,31,18]

    real_parquet_file=pq.ParquetFile
    row_group_reads=[]
    pandas_lengths=[]
    class SpyParquetFile:
        def __init__(self,*args,**kwargs): self.inner=real_parquet_file(*args,**kwargs)
        def __getattr__(self,name): return getattr(self.inner,name)
        def read_row_groups(self,groups,**kwargs):
            result=self.inner.read_row_groups(groups,**kwargs)
            row_group_reads.append((tuple(groups),tuple(kwargs.get('columns',())),result.num_rows))
            class SpyTable:
                def __init__(self,table): self.table=table
                def __getattr__(self,name): return getattr(self.table,name)
                def take(self,indices): return SpyTable(self.table.take(indices))
                def to_pandas(self,*args,**kw):
                    value=self.table.to_pandas(*args,**kw); pandas_lengths.append(len(value)); return value
            return SpyTable(result)
    monkeypatch.setattr(delta_module.pq,'ParquetFile',SpyParquetFile)
    actual=read_feature_table(source,columns=['race_id','horse_no','x','y','z'],positions=positions)
    expected=full[['race_id','horse_no','x','y','z']].iloc[positions].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected)
    assert any(groups and len(groups)<8 and set(cols)=={'race_id','horse_no','x'}
               for groups,cols,_ in row_group_reads)
    assert any(length==4 for length in pandas_lengths)
    # Even when only the first key is requested, the full key chain is checked.
    corrupt=pd.read_parquet(grandchild/'feature-delta.parquet')
    corrupt.loc[39,'horse_no']=999
    corrupt.to_parquet(grandchild/'feature-delta.parquet',index=False)
    descriptor=json.loads(source.read_text())
    descriptor['delta_sha256']=digest(grandchild/'feature-delta.parquet')
    source.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError,match='Delta runner order/population|ordered'):
        read_feature_table(source,columns=['z'],positions=[0])


def test_multilayer_delta_preserves_parents_and_projects(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    child=registry/'datasets/dataset-child'
    enriched=frame.assign(y=frame.x*2)
    metadata={'y':predictor()}
    write_feature_delta(parent/'manifest.json',child,enriched,registry_root=registry,
                        columns=['y'],catalog=metadata)
    child_manifest={'dataset_id':'dataset-child','features_path':str(child/'features.delta.json'),
        'files':{'features.delta.json':digest(child/'features.delta.json')},'predictor_catalog':metadata}
    (child/'manifest.json').write_text(json.dumps(child_manifest))
    grandchild=registry/'datasets/dataset-grandchild'
    full=enriched.assign(z=enriched.y+1)
    before=digest(parent/'features.parquet')
    write_feature_delta(child/'manifest.json',grandchild,full,registry_root=registry,
                        columns=['z'],catalog={'z':predictor()})
    pd.testing.assert_frame_equal(read_feature_table(grandchild/'features.delta.json'),full)
    pd.testing.assert_frame_equal(read_feature_table(grandchild/'features.delta.json',columns=['z','x']),full[['z','x']])
    assert digest(parent/'features.parquet')==before
    assert dataset_storage_estimate(grandchild/'features.delta.json')>dataset_storage_estimate(child/'features.delta.json')
    (parent/'features.parquet').write_bytes(b'changed ancestor')
    with pytest.raises(ValueError,match='Parent table changed'):
        read_feature_table(grandchild/'features.delta.json')


def test_delta_writer_rejects_duplicate_key_collision(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    colliding=frame.copy()
    colliding.loc[1,['race_id','horse_no']]=colliding.loc[0,['race_id','horse_no']].to_numpy()
    with pytest.raises(ValueError,match='Changed delta population'):
        write_feature_delta(parent/'manifest.json',registry/'datasets/key-collision',
            colliding.assign(y=[4.,5.]),registry_root=registry,columns=['y'],catalog={'y':predictor()})


def test_delta_reader_rejects_cyclic_ancestor_chain(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    child=registry/'datasets/dataset-cycle'
    child.mkdir()
    pointer=child/'features.delta.json'
    delta=child/'feature-delta.parquet'
    pd.DataFrame({'race_id':frame.race_id,'horse_no':frame.horse_no,'y':[4.,5.]}).to_parquet(delta,index=False)
    descriptor={'storage_format':delta_module.FORMAT,'registry_root':str(registry),
        'parent_manifest_path':str(child/'manifest.json'),'parent_manifest_sha256':'bound',
        'parent_dataset_id':'dataset-cycle','delta_sha256':'bound','delta_columns':['y'],
        'replacements':[],'rows':len(frame),'columns':['race_id','horse_no','y'],
        'ordered_keys_sha256':delta_module.key_digest(frame)}
    pointer.write_text(json.dumps(descriptor))
    (child/'manifest.json').write_text(json.dumps({'dataset_id':'dataset-cycle',
        'features_path':str(pointer),'files':{'features.delta.json':'bound'}}))
    with patch.object(delta_module,'digest',return_value='bound'):
        with pytest.raises(ValueError,match='Dataset parent cycle'):
            read_feature_table(pointer)


def test_delta_reader_rejects_missing_ancestor_table(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    child=registry/'datasets/dataset-child'
    write_feature_delta(parent/'manifest.json',child,frame.assign(y=frame.x*2),
        registry_root=registry,columns=['y'],catalog={'y':predictor()})
    (parent/'features.parquet').unlink()
    with pytest.raises(FileNotFoundError):
        read_feature_table(child/'features.delta.json')


def test_replacement_requires_matching_units_and_protects_labels(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    with pytest.raises(ValueError,match='units'):
        write_feature_delta(parent/'manifest.json',registry/'datasets/bad',frame.assign(x=frame.x*2),
            registry_root=registry,columns=['x'],replacements=['x'],
            catalog={'x':predictor(unit='seconds')})
    with pytest.raises(ValueError,match='labels'):
        write_feature_delta(parent/'manifest.json',registry/'datasets/bad-label',frame,
            registry_root=registry,columns=['target_win'],replacements=['target_win'])


@pytest.mark.parametrize('fault',['labels','order','missing','replacement'])
def test_delta_rejects_silent_parent_changes(tmp_path,fault):
    registry,parent,frame=parent_fixture(tmp_path); enriched=frame.assign(y=frame.x*2)
    if fault=='labels': enriched.loc[0,'target_win']=0
    if fault=='order': enriched=enriched.iloc[::-1].reset_index(drop=True)
    if fault=='missing': enriched=enriched.iloc[:1]
    if fault=='replacement': enriched.loc[0,'x']=99
    with pytest.raises(ValueError):
        write_feature_delta(parent/'manifest.json',registry/'datasets/dataset-child',enriched,
            registry_root=registry,columns=['y'],catalog={'y':predictor()})


def test_partial_descriptor_failure_publishes_no_artifacts(tmp_path,monkeypatch):
    registry,parent,frame=parent_fixture(tmp_path)
    output=registry/'datasets/partial-child'
    enriched=frame.assign(y=frame.x*2)
    original_dump=json.dump

    def partial_dump(value,stream,**kwargs):
        stream.write('{"partial":')
        raise RuntimeError('descriptor write fault')

    monkeypatch.setattr(delta_module.json,'dump',partial_dump)
    with pytest.raises(RuntimeError,match='descriptor write fault'):
        write_feature_delta(parent/'manifest.json',output,enriched,registry_root=registry,
            columns=['y'],catalog={'y':predictor()})
    assert not (output/'features.delta.json').exists()
    assert not (output/'feature-delta.parquet').exists()
    assert not output.exists()
    monkeypatch.setattr(delta_module.json,'dump',original_dump)


def test_storage_pressure_denies_before_output_mkdir(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from ima.research_storage import StorageBlocked
    registry,parent,frame=parent_fixture(tmp_path)
    output=registry/'datasets/denied-child'

    @contextmanager
    def deny(*args,**kwargs):
        raise StorageBlocked('capacity denied')
        yield

    monkeypatch.setattr(delta_module,'admitted_output',deny)
    with pytest.raises(StorageBlocked,match='capacity denied'):
        write_feature_delta(parent/'manifest.json',output,frame.assign(y=frame.x*2),
            registry_root=registry,columns=['y'],catalog={'y':predictor()})
    assert not output.exists()


def test_existing_delta_outputs_are_never_replaced_or_removed(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    output=registry/'datasets/existing-child'
    output.mkdir()
    delta=output/'feature-delta.parquet'
    pointer=output/'features.delta.json'
    delta.write_bytes(b'keep delta')
    pointer.write_bytes(b'keep descriptor')
    with pytest.raises(ValueError,match='already exists'):
        write_feature_delta(parent/'manifest.json',output,frame.assign(y=frame.x*2),
            registry_root=registry,columns=['y'],catalog={'y':predictor()})
    assert delta.read_bytes()==b'keep delta'
    assert pointer.read_bytes()==b'keep descriptor'


def test_post_pointer_fsync_failure_keeps_pair_readable_to_concurrent_reader(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from ima.research_storage import StorageBlocked
    registry,parent,frame=parent_fixture(tmp_path)
    output=registry/'datasets/late-fsync-child'
    enriched=frame.assign(y=frame.x*2)
    admission={'entered':False,'released':False}

    @contextmanager
    def tracked_admission(*args,**kwargs):
        admission['entered']=True
        try:
            yield
        finally:
            admission['released']=True

    pointer_published=threading.Event()
    allow_fsync_failure=threading.Event()

    def fail_after_pointer(_path):
        pointer_published.set()
        assert allow_fsync_failure.wait(10), 'writer did not receive fsync-failure signal'
        raise OSError(errno.ENOSPC,'injected late directory fsync failure')

    monkeypatch.setenv('IMA_STORAGE_ROOT',str(tmp_path/'storage-control'))
    monkeypatch.setattr(delta_module,'admitted_output',tracked_admission)
    monkeypatch.setattr(delta_module,'_sync_directory',fail_after_pointer)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(write_feature_delta,parent/'manifest.json',output,enriched,
            registry_root=registry,columns=['y'],catalog={'y':predictor()})
        assert pointer_published.wait(10), 'writer did not publish its descriptor'
        pointer=output/'features.delta.json'
        delta=output/'feature-delta.parquet'
        assert pointer.is_file() and delta.is_file()
        pd.testing.assert_frame_equal(read_feature_table(pointer),enriched,check_exact=True)
        allow_fsync_failure.set()
        with pytest.raises(StorageBlocked):
            future.result(timeout=10)

    assert admission=={'entered':True,'released':True}
    assert pointer.is_file() and delta.is_file()
    pd.testing.assert_frame_equal(read_feature_table(pointer),enriched,check_exact=True)


@pytest.mark.parametrize('name',['target_rank_score','future_market_probability','win_odds','market_raw',
                                  'market_odds','place_odds','final_place_odds','current_decimal_odds'])
def test_added_protected_label_and_market_names_are_rejected(tmp_path,name):
    registry,parent,frame=parent_fixture(tmp_path)
    with pytest.raises(ValueError,match='Unsafe|Protected|forbids'):
        write_feature_delta(parent/'manifest.json',registry/'datasets'/('protected-'+name),
            frame.assign(**{name:[0.2,0.8]}),registry_root=registry,columns=[name],
            catalog={name:predictor()})


@pytest.mark.parametrize('field',['market_only_columns','target_only_columns'])
def test_parent_declared_exclusions_cannot_be_added_as_eligible_predictors(tmp_path,field):
    registry,parent,frame=parent_fixture(tmp_path)
    path=parent/'manifest.json'
    manifest=json.loads(path.read_text());manifest[field]=['source_specific_excluded']
    path.write_text(json.dumps(manifest))
    output=registry/'datasets'/'excluded-alias'
    with pytest.raises(ValueError,match='Protected'):
        write_feature_delta(path,output,frame.assign(source_specific_excluded=[.2,.8]),
            registry_root=registry,columns=['source_specific_excluded'],
            catalog={'source_specific_excluded':predictor()})
    assert not output.exists()


def test_invalid_unit_and_untrusted_availability_metadata_are_rejected(tmp_path):
    registry,parent,frame=parent_fixture(tmp_path)
    with pytest.raises(ValueError,match='Unsupported predictor unit'):
        write_feature_delta(parent/'manifest.json',registry/'datasets/bad-unit',frame.assign(y=frame.x*2),
            registry_root=registry,columns=['y'],catalog={'y':predictor(unit='furlongs per week')})
    with pytest.raises(ValueError,match='pre-race availability'):
        write_feature_delta(parent/'manifest.json',registry/'datasets/tainted',frame.assign(y=frame.x*2),
            registry_root=registry,columns=['y'],catalog={'y':predictor(target_tainted=True)})


def test_real_registry_delta_reaches_executor(tmp_path,monkeypatch):
    from ima.feature_definitions import FeatureDefinition
    from ima.feature_lifecycle import build_feature_candidate
    from ima.research_controller import protocol_spec_parameters
    from ima.research_executor import RecipeExecutionRequest,execute_recipe
    from ima.research_specs import PipelineRecipe
    from tests.test_dataset_registry import DatasetRegistryTests
    registry,parent=DatasetRegistryTests()._build_row_hash_fixture(tmp_path)
    parent_path=registry.dataset_path(parent['dataset_id'])
    definition=FeatureDefinition(name='carried_body_fraction',unit='1',input_refs=('actual_weight','declared_weight'),
        expression_ast={'op':'divide','args':[{'op':'ref','ref':'actual_weight'},{'op':'ref','ref':'declared_weight'}]})
    monkeypatch.setenv('IMA_DATASET_STORAGE_FORMAT','delta-v2')
    child=build_feature_candidate(parent_path/'features.parquet',parent_path/'manifest.json',registry.root,definitions=[definition])
    pointer=Path(child['features_path'])
    assert pointer.name=='features.delta.json' and not pointer.with_name('features.parquet').exists()
    assert registry.verify(child['dataset_id'])['ordered_row_keys_sha256']==parent['ordered_row_keys_sha256']
    from ima.research_controller import DatasetFeatureProfile
    DatasetFeatureProfile(pointer,protocol_spec_parameters(json.loads((parent_path/'protocol.json').read_text())),
                          extra_numeric_features=(definition.feature_id,))
    recipe=PipelineRecipe.model_validate({'schema_version':3,'feature_schema':'baseline-v1','dataset_ref':child['dataset_id'],
        'extra_numeric_features':[definition.feature_id],'model':{'kind':'benter_conditional_logit','parameters':{'max_iter':1000}},'blend':{'kind':'none'}})
    result=execute_recipe(RecipeExecutionRequest('delta-consumer','delta-canary',0,recipe,pointer,tmp_path/'trial',
        protocol_spec_parameters(json.loads((parent_path/'protocol.json').read_text()))))
    assert result.status=='completed',result.error
