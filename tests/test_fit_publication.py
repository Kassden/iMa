import json
import time
import threading
from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ima.research_executor import RecipeExecutionRequest,RecipeExecutionResult
from ima.research_publication import stage_arguments,publish_accepted,read_accepted,pending_feedback,deliver_feedback
from ima.research_specs import PipelineRecipe
from ima.research_worker_runtime import BoundedFitExecutor,read_runtime


def request_at(root):
    recipe=PipelineRecipe.model_validate({'schema_version':3,'model':{'kind':'benter_conditional_logit'},
                                         'blend':{'kind':'none'}})
    return RecipeExecutionRequest('attempt-one','proposal',0,recipe,root/'dataset.csv',
        root/'trials'/'attempt-one',{},'dataset-hash','code','environment',supervised_fit=True)


def row_for(request):
    return {'attempt_id':request.attempt_id,'status':'running','payload':{
        'recipe':request.recipe.model_dump(mode='json'),'recipe_hash':request.recipe.recipe_hash(),
        'dataset_path':str(request.dataset_path),'protocol_parameters':request.protocol_parameters,
        **{name:getattr(request,name) for name in ('proposal_id','trial_number','dataset_hash','code_revision','environment_hash')}}}


def write_private_result(request,delay=0):
    request.output_dir.mkdir(parents=True,exist_ok=True)
    artifact=request.output_dir/'model.txt';artifact.write_text('fixture model')
    result=RecipeExecutionResult(1,request.attempt_id,request.proposal_id,request.trial_number,
        request.recipe.recipe_hash(),'win_probability','completed','development_fundamental_race_log_loss',
        2.,{}, {'model':str(artifact)}, {},0.)
    (request.output_dir/'result.json').write_text(json.dumps(result.serializable()))
    time.sleep(delay)
    return result


def test_actual_supervisor_publishes_and_recovers_the_same_result(tmp_path):
    request=request_at(tmp_path)
    with BoundedFitExecutor(1) as pool:
        result=pool.submit(write_private_result,request,runtime_path=request.output_dir/'runtime.json',timeout=30).result(40)
    runtime=read_runtime(request.output_dir/'runtime.json')
    assert Path(runtime['episode_path']).parent==tmp_path/'.fit-episodes'
    assert Path(result.artifacts['model']).read_text()=='fixture model'
    assert Path(result.artifacts['model']).parent.parent.parent==tmp_path
    assert read_accepted(tmp_path,row_for(request))==result.serializable()
    canonical=json.loads((request.output_dir/'result.json').read_text())
    assert canonical['_execution_acceptance']==runtime['execution_acceptance']


def test_timeout_keeps_worker_outputs_private(tmp_path):
    request=request_at(tmp_path)
    with BoundedFitExecutor(1) as pool:
        future=pool.submit(write_private_result,request,10,runtime_path=request.output_dir/'runtime.json',timeout=3)
        with pytest.raises(TimeoutError):future.result(15)
    assert not (request.output_dir/'result.json').exists()
    assert read_accepted(tmp_path,row_for(request)) is None


def accepted_fixture(root):
    request=request_at(root)
    request.output_dir.mkdir(parents=True)
    nonce='a'*32
    args,managed=stage_arguments((request,),request.output_dir/'runtime.json',nonce)
    assert args[0].output_dir.parent.parent==request.output_dir.parent.parent
    result=write_private_result(args[0])
    now=time.time()
    state={'status':'finished','task_nonce':nonce,
           'deadline_at':datetime.fromtimestamp(now+30,timezone.utc).isoformat()}
    receipt=publish_accepted(managed,result,state,now)
    (request.output_dir/'runtime.json').write_text(json.dumps(state|{'execution_acceptance':receipt}))
    return request,args[0],result,state


def test_recovery_rejects_changed_payload_or_execution_inputs(tmp_path):
    request,staged,result,state=accepted_fixture(tmp_path)
    row=row_for(request)
    assert read_accepted(tmp_path,row)==result.serializable()
    row['payload']['code_revision']='other-code'
    with pytest.raises(ValueError,match='input changed'):read_accepted(tmp_path,row)
    row=row_for(request)
    path=request.output_dir/'result.json'
    changed=json.loads(path.read_text());changed['objective_value']=1.
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='identity'):read_accepted(tmp_path,row)


def test_late_or_stale_publication_creates_no_canonical_result(tmp_path):
    request=request_at(tmp_path);request.output_dir.mkdir(parents=True)
    args,managed=stage_arguments((request,),request.output_dir/'runtime.json','a'*32)
    result=write_private_result(args[0])
    state={'task_nonce':'b'*32,'status':'finished','deadline_at':'2000-01-01T00:00:00+00:00'}
    with pytest.raises(RuntimeError):publish_accepted(managed,result,state,time.time())
    state['task_nonce']='a'*32
    with pytest.raises(TimeoutError):publish_accepted(managed,result,state,time.time())
    assert not (request.output_dir/'result.json').exists()


def test_feedback_is_delivered_only_from_an_accepted_episode(tmp_path):
    request,staged,result,state=accepted_fixture(tmp_path)
    row=row_for(request);row['status']='completed'
    intent={'attempt_id':request.attempt_id,'task_nonce':staged.managed_episode_nonce}
    (staged.output_dir/'feature-feedback-intent.json').write_text(json.dumps(intent))
    assert request.attempt_id in pending_feedback(tmp_path,[row])
    with patch('ima.research_lifecycle.link_feature_outcome',return_value={'registered':True}) as link:
        deliver_feedback(tmp_path,row)
        link.assert_called_once()
    assert pending_feedback(tmp_path,[row])=={}
    canonical=request.output_dir/'result.json'
    canonical.unlink()
    with patch('ima.research_lifecycle.link_feature_outcome') as link:
        with pytest.raises(ValueError,match='no supervisor'):deliver_feedback(tmp_path,row)
        link.assert_not_called()


def test_managed_lifecycle_stages_feedback_without_shared_mutation(tmp_path):
    from ima import research_expansion as expansion
    from ima.research_lifecycle import lifecycle_worker
    request=request_at(tmp_path)
    args,_=stage_arguments((request,),request.output_dir/'runtime.json','a'*32)
    staged=args[0]
    result=write_private_result(staged)
    with patch.object(expansion,'_lifecycle_original_worker',return_value=result,create=True), \
         patch('ima.research_lifecycle.link_feature_outcome') as link:
        assert lifecycle_worker(staged,None,1).status=='completed'
        link.assert_not_called()
    assert (staged.output_dir/'feature-feedback-intent.json').exists()
    assert not (request.output_dir/'result.json').exists()


def test_real_fit_package_replays_from_the_accepted_episode(tmp_path):
    import hashlib
    import numpy as np
    import pandas as pd
    from ima.research_executor import execute_recipe
    from ima.research_model_package import load_research_package
    from tests.test_research_executor_v6 import dataset
    source,frame=dataset(tmp_path)
    request=replace(request_at(tmp_path),dataset_path=source,dataset_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        protocol_parameters={'min_train_races':12,'calibration_races':3,'score_races':3,'max_folds':1})
    with BoundedFitExecutor(1) as pool:
        result=pool.submit(execute_recipe,request,runtime_path=request.output_dir/'runtime.json',timeout=60).result(75)
    assert result.status=='completed',result.error
    package=load_research_package(Path(result.artifacts['package']))
    predictions=pd.read_csv(result.artifacts['predictions'])
    score=frame.loc[frame.race_id.isin(predictions.race_id)].reset_index(drop=True)
    np.testing.assert_allclose(package.predict(score),predictions.model_probability,rtol=1e-10,atol=1e-12)
    assert read_accepted(tmp_path,row_for(request))==result.serializable()
    assert (tmp_path/'fold-preparation-cache').is_dir()
    assert not (request.output_dir/'fold-preparation-cache').exists()


def test_fold_preparation_request_is_not_treated_as_a_model_fit(tmp_path):
    request=replace(request_at(tmp_path),attempt_id='dependency',
                    output_dir=tmp_path/'trials'/'prepare-dependency',supervised_fit=False)
    args,managed=stage_arguments((request,),request.output_dir/'runtime.json','a'*32)
    assert args==(request,) and managed is None
    with BoundedFitExecutor(1) as pool:
        result=pool.submit(preparation_dictionary,request,
            runtime_path=request.output_dir/'runtime.json',timeout=30).result(40)
    assert result=={'prepared':True}
    assert not (request.output_dir/'result.json').exists()


def preparation_dictionary(request):
    assert request.managed_episode_nonce is None
    return {'prepared':True}


def test_cancellation_and_publication_have_exactly_one_winner(tmp_path):
    import ima.research_publication as publication
    request=request_at(tmp_path)
    raw=Future()
    entered,release,cancel_started=threading.Event(),threading.Event(),threading.Event()
    original=publication.publish_accepted
    def blocked_publication(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    cancellation=[]
    with patch('pebble.ProcessPool') as process_pool, \
         patch.object(publication,'publish_accepted',side_effect=blocked_publication):
        process_pool.return_value.schedule.return_value=raw
        with BoundedFitExecutor(1) as pool:
            future=pool.submit(write_private_result,request,runtime_path=request.output_dir/'runtime.json',timeout=60)
            staged=process_pool.return_value.schedule.call_args.kwargs['args'][1][0]
            result=write_private_result(staged)
            runtime=read_runtime(request.output_dir/'runtime.json')|{'status':'finished'}
            (request.output_dir/'runtime.json').write_text(json.dumps(runtime))
            completion=threading.Thread(target=lambda:raw.set_result(result))
            def cancel():
                cancel_started.set()
                cancellation.append(future.cancel())
            cancelling=threading.Thread(target=cancel)
            try:
                completion.start()
                assert entered.wait(5)
                cancelling.start()
                assert cancel_started.wait(5)
            finally:
                release.set()
                completion.join(5)
                if cancelling.ident is not None:cancelling.join(5)
            assert not completion.is_alive() and not cancelling.is_alive()
            assert cancellation==[False]
            assert future.result()==result
            assert read_accepted(tmp_path,row_for(request))==result.serializable()


def test_cancellation_before_acceptance_does_not_publish(tmp_path):
    request=request_at(tmp_path)
    raw=Future()
    with patch('pebble.ProcessPool') as process_pool:
        process_pool.return_value.schedule.return_value=raw
        with BoundedFitExecutor(1) as pool:
            future=pool.submit(write_private_result,request,runtime_path=request.output_dir/'runtime.json',timeout=60)
            assert future.cancel()
            assert read_runtime(request.output_dir/'runtime.json')['status']=='cancelled'
    assert not (request.output_dir/'result.json').exists()


def test_timeout_recording_failure_does_not_strand_the_future(tmp_path):
    import errno
    import ima.research_worker_runtime as runtime
    request=request_at(tmp_path)
    raw=Future()
    original=runtime._update_runtime
    def update(path,changes):
        if callable(changes) and changes.__name__=='<lambda>':
            raise OSError(errno.ENOSPC,'control filesystem full')
        return original(path,changes)
    with patch('pebble.ProcessPool') as process_pool:
        process_pool.return_value.schedule.return_value=raw
        with BoundedFitExecutor(1) as pool:
            future=pool.submit(write_private_result,request,runtime_path=request.output_dir/'runtime.json',timeout=60)
            staged=process_pool.return_value.schedule.call_args.kwargs['args'][1][0]
            result=write_private_result(staged)
            state=read_runtime(request.output_dir/'runtime.json')|{
                'status':'finished','deadline_at':'2000-01-01T00:00:00+00:00'}
            (request.output_dir/'runtime.json').write_text(json.dumps(state))
            with patch.object(runtime,'_update_runtime',side_effect=update):
                raw.set_result(result)
                with pytest.raises(OSError,match='control filesystem full'):future.result(2)
    assert not (request.output_dir/'result.json').exists()
