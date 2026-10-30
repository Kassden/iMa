import errno
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ima import mlflow_tracking as tracking
from ima import research_storage as storage
from ima.research_model_package import ResearchModelPackage
from ima.research_specs import PipelineRecipe
from tests.test_mlflow_tracking import DummyProbabilityModel


@pytest.fixture
def upload_context(tmp_path,monkeypatch):
    root=tmp_path.resolve()
    package=ResearchModelPackage(DummyProbabilityModel(),PipelineRecipe(),
        protocol_id='upload-protocol',code_revision='fixture')
    package_dir=package.save(root/'package')
    authority=storage.StorageAuthority(root/'control',headroom_bytes=0,headroom_inodes=0,
        statvfs=lambda path:SimpleNamespace(f_frsize=1,f_blocks=512*1024**2,
            f_bavail=512*1024**2,f_bfree=512*1024**2,f_favail=100000))
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setenv('MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR','false')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    config=tracking.MLflowConfig(tracking_uri=f'sqlite:///{root / "mlflow.db"}',
        experiment_name='admitted-model-upload',enabled=True,
        registered_model_name='admitted-candidates')
    result={'target_kind':'win_probability','recipe_hash':package.recipe.recipe_hash(),
        'objective_name':'development_race_log_loss','objective_value':1.,
        'lineage':{'dataset_hash':'a'*64,'protocol_id':'upload-protocol'}}
    return root,package_dir,authority,config,result


def test_denied_model_upload_never_initializes_sdk_or_creates_tracking_run(upload_context):
    root,package,authority,config,result=upload_context
    before={p.name for p in root.iterdir()}
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('fixture full')):
        with patch.object(tracking,'_mlflow') as sdk:
            with pytest.raises(storage.StorageBlocked):
                tracking.log_research_package_version(package,config,attempt_id='denied',result=result)
            sdk.assert_not_called()
    assert not (root/'mlflow.db').exists()
    assert before=={p.name for p in root.iterdir()}


def test_tracking_json_metadata_preserves_existing_receipt_on_denied_admission(upload_context):
    root,package,authority,config,result=upload_context
    path=root/'linkage.json'
    tracking._write_json_atomic(path,{'run_id':'kept','tokens':120})
    original=path.read_bytes()
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('fixture metadata full')):
        with pytest.raises(storage.StorageBlocked):tracking._write_json_atomic(path,{'run_id':'rejected'})
    assert path.read_bytes()==original
    assert not list(root.glob('*.tmp'))


def test_real_sdk_upload_holds_reservation_and_registered_model_reloads(upload_context):
    import mlflow
    root,package,authority,config,result=upload_context
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.create_experiment(config.experiment_name,artifact_location=(root/'mlartifacts').as_uri())
    actual=tracking._log_research_package_version
    observations=[]
    def observe(*args,**kwargs):
        snapshot=authority.status()
        assert snapshot['reserved_bytes']>=64*1024**2
        observations.append(snapshot['reserved_bytes'])
        return actual(*args,**kwargs)
    with patch.object(tracking,'_log_research_package_version',side_effect=observe):
        first=tracking.log_research_package_version(package,config,attempt_id='admitted',result=result)
        second=tracking.log_research_package_version(package,config,attempt_id='admitted',result=result)
    assert first==second and len(observations)==2
    assert authority.status()['reserved_bytes']==0
    run=mlflow.get_run(first['run_id'])
    assert run.info.status=='FINISHED'
    loaded=mlflow.pyfunc.load_model(first['registered_model_uri'])
    np.testing.assert_allclose(loaded.predict(pd.DataFrame({'race_id':['r','r']})),[.5,.5])


def test_model_upload_native_enospc_is_retryable_and_releases_reservation(upload_context):
    root,package,authority,config,result=upload_context
    with patch.object(tracking,'_log_research_package_version',side_effect=OSError(errno.ENOSPC,'fixture full')):
        with pytest.raises(storage.StorageBlocked) as raised:
            tracking.log_research_package_version(package,config,attempt_id='full',result=result)
    assert raised.value.diagnostic['stage']=='mlflow_model_upload'
    assert raised.value.diagnostic['retryable'] is True
    assert authority.status()['reserved_bytes']==0


def test_code_copy_retains_native_enospc_before_copytree_can_aggregate_it(upload_context):
    root,package,authority,config,result=upload_context
    source=root/'code';source.mkdir();(source/'module.py').write_text('value = 1\n')
    with patch.object(shutil,'copyfile',side_effect=OSError(errno.ENOSPC,'fixture code copy full')):
        with pytest.raises(storage.StorageBlocked) as raised:
            with tracking._writable_code_copy(source):pytest.fail('Copy must not complete')
    assert raised.value.diagnostic['errno']==errno.ENOSPC
    assert raised.value.diagnostic['stage']=='tracking_code_copy'
    assert (source/'module.py').read_text()=='value = 1\n'


def test_code_copy_retains_native_enospc_from_nested_directory_creation(upload_context):
    root,package,authority,config,result=upload_context
    source=root/'code';(source/'nested').mkdir(parents=True)
    (source/'nested'/'module.py').write_text('value = 1\n')
    mkdir=Path.mkdir
    def full(path,*args,**kwargs):
        if path.name=='nested' and 'ima-mlflow-code-' in str(path):
            raise OSError(errno.ENOSPC,'fixture directory full')
        return mkdir(path,*args,**kwargs)
    with patch.object(Path,'mkdir',full):
        with pytest.raises(storage.StorageBlocked) as raised:
            with tracking._writable_code_copy(source):pytest.fail('Copy must not complete')
    assert raised.value.diagnostic['errno']==errno.ENOSPC
    assert raised.value.diagnostic['stage']=='tracking_code_copy'
    assert (source/'nested'/'module.py').read_text()=='value = 1\n'


def test_wrapped_database_full_keeps_native_retryable_storage_identity(tmp_path,monkeypatch):
    import sqlite3
    from sqlalchemy.exc import OperationalError
    from mlflow.exceptions import MlflowException
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(tmp_path/'control'))
    with sqlite3.connect(tmp_path/'bounded.sqlite') as conn:
        conn.execute('PRAGMA page_size=512')
        conn.execute('PRAGMA max_page_count=2')
        conn.execute('CREATE TABLE payload(value BLOB)');conn.commit()
        try:conn.execute('INSERT INTO payload VALUES(?)',(b'x'*100000,))
        except sqlite3.Error as exc:native=exc
        else:pytest.fail('Fixture must produce real SQLITE_FULL')
    assert native.sqlite_errorcode==sqlite3.SQLITE_FULL
    sql=OperationalError('INSERT INTO payload VALUES(?)',{},native)
    outer=MlflowException('Session failed');outer.__cause__=sql
    result=storage.as_storage_blocked(outer,stage='model_upload',output=tmp_path)
    assert isinstance(result,storage.StorageBlocked)
    assert result.diagnostic['sqlite_errorcode']==sqlite3.SQLITE_FULL
    assert result.diagnostic['wrapper_exception_type']=='MlflowException'
    assert result.diagnostic['cause_exception_type']==type(native).__name__


def test_upload_budget_includes_reports_and_ignores_full_training_dataset(upload_context):
    root,package,authority,config,result=upload_context
    dataset=root/'features.csv';dataset.write_bytes(b'training data is referenced, not uploaded')
    initial=tracking._research_upload_budget(package,result,dataset,config)
    report=root/'graph-fits.json';report.write_bytes(b'x'*100000)
    changed=tracking._research_upload_budget(package,result,dataset,config)
    assert changed[0]-initial[0]==400000
    assert changed[1]>initial[1]
    dataset.write_bytes(b'y'*1000000)
    assert tracking._research_upload_budget(package,result,dataset,config)==changed


def test_package_bytecode_is_budgeted_even_though_code_copy_excludes_it(upload_context):
    root,package,authority,config,result=upload_context
    before=tracking._research_upload_budget(package,result,None,config)
    (package/'extra.pyc').write_bytes(b'x'*1000000)
    after=tracking._research_upload_budget(package,result,None,config)
    assert after[0]-before[0]==4000000 and after[1]>before[1]


def test_actual_experiment_artifact_filesystem_is_checked_before_starting_run(upload_context):
    import mlflow
    root,package,authority,config,result=upload_context
    artifact=root/'foreign-artifacts';artifact.mkdir()
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment=mlflow.create_experiment(config.experiment_name,artifact_location=artifact.as_uri())
    real_stat=Path.stat
    def stat(path,*args,**kwargs):
        value=real_stat(path,*args,**kwargs)
        if path==artifact:return SimpleNamespace(st_dev=value.st_dev+1,st_uid=value.st_uid,st_mode=value.st_mode)
        return value
    with patch.object(Path,'stat',stat),patch.object(mlflow,'start_run') as start:
        with pytest.raises(storage.StorageBlocked):
            tracking.log_research_package_version(package,config,attempt_id='foreign-artifacts',result=result)
        start.assert_not_called()
    assert mlflow.tracking.MlflowClient().search_runs([experiment])==[]
    assert list(artifact.iterdir())==[]


def test_separate_registry_filesystem_is_checked_before_experiment_creation(upload_context):
    import mlflow
    root,package,authority,config,result=upload_context
    foreign=root/'foreign-registry';foreign.mkdir()
    real_stat=Path.stat
    def stat(path,*args,**kwargs):
        value=real_stat(path,*args,**kwargs)
        if path==foreign:return SimpleNamespace(st_dev=value.st_dev+1,st_uid=value.st_uid,st_mode=value.st_mode)
        return value
    with patch.object(Path,'stat',stat),patch.object(mlflow,'get_registry_uri',return_value=f'sqlite:///{foreign / "registry.db"}'),patch.object(mlflow,'set_experiment') as experiment:
        with pytest.raises(storage.StorageBlocked):
            tracking.log_research_package_version(package,config,attempt_id='foreign-registry',result=result)
        experiment.assert_not_called()
    assert not (root/'mlflow.db').exists() and not (foreign/'registry.db').exists()


def test_unknown_proxy_artifact_store_is_not_assumed_local(upload_context):
    import mlflow
    root,package,authority,config,result=upload_context
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.create_experiment(config.experiment_name,artifact_location='mlflow-artifacts:/1')
    with patch.object(mlflow,'start_run') as start:
        with pytest.raises(storage.StorageBlocked,match='Unqualified'):
            tracking.log_research_package_version(package,config,attempt_id='proxy-store',result=result)
        start.assert_not_called()


def test_unprovisioned_experiment_is_not_implicitly_created_by_guarded_upload(upload_context):
    import mlflow
    root,package,authority,config,result=upload_context
    with patch.object(mlflow,'set_experiment') as experiment,patch.object(mlflow,'start_run') as start:
        with pytest.raises(storage.StorageBlocked,match='bootstrapped experiment'):
            tracking.log_research_package_version(package,config,attempt_id='unprovisioned',result=result)
        experiment.assert_not_called();start.assert_not_called()
    assert mlflow.tracking.MlflowClient().get_experiment_by_name(config.experiment_name) is None


def test_cross_filesystem_scratch_is_rejected_before_sdk(upload_context,tmp_path):
    root,package,authority,config,result=upload_context
    scratch=root/'other-filesystem';scratch.mkdir()
    real_stat=Path.stat
    def stat(path,*args,**kwargs):
        value=real_stat(path,*args,**kwargs)
        if path==scratch:return SimpleNamespace(st_dev=value.st_dev+1,st_mode=value.st_mode)
        return value
    with patch.object(tracking.tempfile,'gettempdir',return_value=str(scratch)),patch.object(Path,'stat',stat),patch.object(tracking,'_mlflow') as sdk:
        with pytest.raises(storage.StorageBlocked):
            tracking.log_research_package_version(package,config,attempt_id='foreign-scratch',result=result)
        sdk.assert_not_called()


def test_disabled_model_upload_remains_a_side_effect_free_noop(upload_context):
    root,package,authority,config,result=upload_context
    from dataclasses import replace
    with patch.object(tracking,'_research_upload_budget') as budget,patch.object(tracking,'_mlflow') as sdk:
        assert tracking.log_research_package_version(package,replace(config,enabled=False),attempt_id='disabled',result=result) is None
        budget.assert_not_called();sdk.assert_not_called()


def test_model_upload_reserves_inodes_before_sdk(upload_context):
    root,package,authority,config,result=upload_context
    real_statvfs=authority._statvfs
    def constrained(path):
        value=real_statvfs(path)
        return SimpleNamespace(f_frsize=value.f_frsize,f_blocks=value.f_blocks,
            f_bavail=value.f_bavail,f_bfree=value.f_bfree,f_favail=128)
    with patch.object(authority,'_statvfs',side_effect=constrained),patch.object(tracking,'_mlflow') as sdk:
        with pytest.raises(storage.StorageBlocked):
            tracking.log_research_package_version(package,config,attempt_id='inodes',result=result)
        sdk.assert_not_called()


def test_model_upload_rejects_source_symlinks_and_non_json_result_before_sdk(upload_context):
    root,package,authority,config,result=upload_context
    with patch.object(tracking,'_mlflow') as sdk:
        (package/'unexpected-link').symlink_to(package/'manifest.json')
        with pytest.raises(ValueError,match='symlink'):
            tracking.log_research_package_version(package,config,attempt_id='symlink',result=result)
        (package/'unexpected-link').unlink()
        with pytest.raises(TypeError):
            tracking.log_research_package_version(package,config,attempt_id='nonjson',result=result|{'unsupported':object()})
        sdk.assert_not_called()


def test_real_sdk_sampled_copy_peak_fits_held_reservation(upload_context,record_property):
    import mlflow
    root,package,authority,config,result=upload_context
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.create_experiment(config.experiment_name,artifact_location=(root/'mlartifacts').as_uri())
    report=root/'paired-report.json';report.write_text(json.dumps({'padding':'x'*1000000}))
    result=result|{'artifacts':{'paired_feature_report':str(report)}}
    estimate,inode_budget=tracking._research_upload_budget(package,result,None,config)
    roots={root}
    def usage():
        seen=set();size=0
        for tree in tuple(roots):
            if not tree.exists():continue
            for path in [tree,*tree.rglob('*')]:
                try:info=path.lstat()
                except FileNotFoundError:continue
                key=(info.st_dev,info.st_ino)
                if key not in seen:seen.add(key);size+=info.st_blocks*512
        return size,len(seen)
    before,before_inodes=usage()
    samples=[]
    def sample():
        allocated,count=usage();samples.append((allocated-before,count-before_inodes))
    mkdtemp=tempfile.mkdtemp;copyfile=shutil.copyfile;rmtree=shutil.rmtree
    def temporary(*args,**kwargs):
        path=mkdtemp(*args,**kwargs);roots.add(Path(path).resolve());sample();return path
    def copy(*args,**kwargs):
        value=copyfile(*args,**kwargs);sample();return value
    def remove(*args,**kwargs):
        sample();return rmtree(*args,**kwargs)
    with patch.object(tempfile,'mkdtemp',side_effect=temporary),patch.object(shutil,'copyfile',side_effect=copy),patch.object(shutil,'rmtree',side_effect=remove):
        linkage=tracking.log_research_package_version(package,config,attempt_id='measured',result=result)
    sample()
    assert len(samples)>10
    assert 0<max(size for size,count in samples)<estimate
    assert 0<max(count for size,count in samples)<inode_budget
    record_property('sampled_peak_allocated_bytes',max(size for size,count in samples))
    record_property('sampled_peak_new_inodes',max(count for size,count in samples))
    record_property('reserved_bytes',estimate)
    record_property('reserved_inodes',inode_budget)
    record_property('sampling_scope','copy and cleanup boundaries; not a hard peak bound')
    assert authority.status()['reserved_bytes']==0
    run=mlflow.get_run(linkage['run_id'])
    downloaded=Path(mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path='analysis/paired-report.json'))
    assert downloaded.read_bytes()==report.read_bytes()


def test_special_cloud_registry_name_is_not_misclassified_as_a_local_path(upload_context):
    import mlflow
    root,package,authority,config,result=upload_context
    with patch.object(mlflow,'get_registry_uri',return_value='databricks'),patch.object(mlflow,'set_experiment') as experiment:
        with pytest.raises(storage.StorageBlocked,match='absolute'):
            tracking.log_research_package_version(package,config,attempt_id='cloud-registry',result=result)
        experiment.assert_not_called()


def test_dangling_destination_symlinks_are_not_validated_using_only_parent_device(upload_context):
    root,package,authority,config,result=upload_context
    link=root/'linked.db';link.symlink_to(root/'absent.db')
    with pytest.raises(storage.StorageBlocked,match='Noncanonical'):
        tracking._upload_destination(f'sqlite:///{link}',authority.root.stat().st_dev)


def test_async_publication_drains_while_reservation_is_held_without_phantom_runs(upload_context,monkeypatch):
    import mlflow
    root,package,authority,config,result=upload_context
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment=mlflow.create_experiment(config.experiment_name,artifact_location=(root/'mlartifacts').as_uri())
    monkeypatch.setenv('MLFLOW_ENABLE_ASYNC_LOGGING','true')
    calls=[]
    metadata=mlflow.flush_async_logging;artifacts=mlflow.flush_artifact_async_logging
    def flush_metadata():
        assert authority.status()['reserved_bytes']>=64*1024**2
        calls.append(('metadata',mlflow.active_run() is not None))
        return metadata()
    def flush_artifacts():
        assert authority.status()['reserved_bytes']>=64*1024**2
        assert mlflow.active_run() is not None
        calls.append(('artifacts',True))
        return artifacts()
    with patch.object(mlflow,'flush_async_logging',side_effect=flush_metadata),patch.object(mlflow,'flush_artifact_async_logging',side_effect=flush_artifacts):
        linkage=tracking.log_research_package_version(package,config,attempt_id='async',result=result)
    assert ('artifacts',True) in calls and calls[-1]==('metadata',False)
    run=mlflow.get_run(linkage['run_id'])
    assert run.info.status=='FINISHED' and run.data.metrics['objective']==1.
    assert len(mlflow.tracking.MlflowClient().search_runs([experiment]))==1
    assert authority.status()['reserved_bytes']==0


def test_admitted_upload_does_not_implicitly_resume_an_environment_run(upload_context,monkeypatch):
    import mlflow
    root,package,authority,config,result=upload_context
    monkeypatch.setenv('MLFLOW_RUN_ID','unrelated-protected-run')
    with patch.object(mlflow,'set_experiment') as experiment:
        with pytest.raises(storage.StorageBlocked,match='implicitly resume'):
            tracking.log_research_package_version(package,config,attempt_id='resume',result=result)
        experiment.assert_not_called()
    assert not (root/'mlflow.db').exists()
