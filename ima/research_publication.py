"""Private physical fit episodes and durable supervisor acceptance receipts."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .research_executor import RecipeExecutionRequest


def _write_publication_json(path, payload):
    """Use the qualified bounded controller publisher for supervisor metadata."""
    from .research_controller import _write_json_atomic
    _write_json_atomic(path, payload)


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,
                                    separators=(',',':')).encode()).hexdigest()


def stage_arguments(args, runtime_path, nonce):
    if not args or not isinstance(args[0],RecipeExecutionRequest) or not args[0].supervised_fit:
        return args,None
    request=args[0]
    canonical=Path(request.output_dir).absolute()
    if canonical.resolve()!=canonical or Path(runtime_path).absolute()!=canonical/'runtime.json':
        raise ValueError('Managed fit requires its canonical runtime directory')
    if canonical.parent.name!='trials' or canonical.name!=request.attempt_id:
        raise ValueError('Managed fit requires campaign/trials/attempt layout')
    if not re.fullmatch(r'[a-f0-9]{32}',nonce):
        raise ValueError('Invalid physical episode nonce')
    episode=canonical.parent.parent/'.fit-episodes'/(canonical.name+'.'+nonce)
    if episode.resolve()!=episode:
        raise ValueError('Physical episode path is not canonical')
    staged=replace(request,output_dir=episode,managed_episode_nonce=nonce)
    return (staged,*args[1:]),(request,staged)


def publish_accepted(managed,result,state,now):
    """Caller holds completion and runtime locks and has checked the deadline."""
    request,staged=managed
    nonce=staged.managed_episode_nonce
    if state.get('task_nonce')!=nonce or state.get('status')!='finished':
        raise RuntimeError('Cannot publish an unaccepted physical episode')
    payload=result.serializable()
    if (payload['attempt_id']!=request.attempt_id
            or payload['recipe_hash']!=request.recipe.recipe_hash()
            or payload['trial_number']!=request.trial_number
            or payload['proposal_id']!=request.proposal_id):
        raise ValueError('Fit result identity differs from admitted request')
    deadline=min(datetime.fromisoformat(state[key]).timestamp()
                 for key in ('deadline_at','hard_deadline_at') if state.get(key))
    if now>=deadline:
        raise TimeoutError('Fit publication missed the accepted deadline')
    receipt={'schema_version':1,'task_nonce':nonce,'attempt_id':request.attempt_id,
        'episode_path':str(staged.output_dir),
        'accepted_at':datetime.fromtimestamp(now,timezone.utc).isoformat(),
        'deadline_at':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        'result_sha256':payload_digest(payload),
        'dataset_hash':request.dataset_hash,'code_revision':request.code_revision,
        'environment_hash':request.environment_hash}
    path=request.output_dir/'result.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('_execution_acceptance')!=receipt:
            raise RuntimeError('An existing canonical result must not be overwritten')
    else:
        _write_publication_json(path,{**payload,'_execution_acceptance':receipt})
    return receipt


def read_accepted(directory,row):
    """Recover the publication-to-ledger crash window; never trust legacy files."""
    path=Path(directory)/'trials'/row['attempt_id']/'result.json'
    if not path.exists():return None
    payload=json.loads(path.read_text())
    receipt=payload.pop('_execution_acceptance',None)
    if receipt is None:return None
    request=row['payload']
    nonce=receipt.get('task_nonce','')
    expected=Path(directory)/'.fit-episodes'/(row['attempt_id']+'.'+nonce)
    if (receipt.get('schema_version')!=1 or not re.fullmatch(r'[a-f0-9]{32}',nonce)
            or receipt.get('attempt_id')!=row['attempt_id']
            or payload.get('attempt_id')!=row['attempt_id']
            or receipt.get('episode_path')!=str(expected) or expected.resolve()!=expected
            or receipt.get('result_sha256')!=payload_digest(payload)
            or payload.get('recipe_hash')!=request['recipe_hash']
            or payload.get('proposal_id')!=request['proposal_id']
            or payload.get('trial_number')!=request['trial_number']):
        raise ValueError('Invalid supervisor acceptance identity')
    for key in ('dataset_hash','code_revision','environment_hash'):
        if receipt.get(key)!=request.get(key):
            raise ValueError('Accepted fit execution input changed: '+key)
    if datetime.fromisoformat(receipt['accepted_at'])>=datetime.fromisoformat(receipt['deadline_at']):
        raise ValueError('Accepted fit missed its deadline')
    runtime=json.loads((path.parent/'runtime.json').read_text())
    if runtime.get('task_nonce')!=nonce or runtime.get('status')!='finished':
        raise ValueError('Accepted fit runtime was replaced or rejected')
    if runtime.get('execution_acceptance') not in (None,receipt):
        raise ValueError('Accepted fit receipt differs from supervisor runtime')
    return payload


def pending_feedback(directory,rows):
    pending={}
    for row in rows:
        if row['status']!='completed':continue
        canonical=Path(directory)/'trials'/row['attempt_id']/'result.json'
        if not canonical.exists():continue
        receipt=json.loads(canonical.read_text()).get('_execution_acceptance')
        if not receipt:continue
        episode=Path(receipt['episode_path'])
        expected=Path(directory)/'.fit-episodes'/(row['attempt_id']+'.'+receipt['task_nonce'])
        if episode!=expected or episode.resolve()!=episode:
            raise ValueError('Feedback episode escaped admitted campaign')
        if (episode/'feature-feedback-intent.json').exists() and not (episode/'feature-feedback-delivered.json').exists():
            pending[row['attempt_id']]=row
    return pending


def deliver_feedback(directory,row):
    """Deliver only a supervisor-accepted result; preserve the scientific payload."""
    from dataclasses import fields
    from .research_executor import RecipeExecutionResult
    from .research_lifecycle import link_feature_outcome
    from .research_specs import PipelineRecipe
    payload=read_accepted(directory,row)
    if payload is None:raise ValueError('Feature feedback has no supervisor acceptance')
    canonical=Path(directory)/'trials'/row['attempt_id']/'result.json'
    receipt=json.loads(canonical.read_text())['_execution_acceptance']
    episode=Path(receipt['episode_path'])
    intent=json.loads((episode/'feature-feedback-intent.json').read_text())
    if intent!={'attempt_id':row['attempt_id'],'task_nonce':receipt['task_nonce']}:
        raise ValueError('Feature feedback intent identity changed')
    source=row['payload']
    request=RecipeExecutionRequest(row['attempt_id'],source['proposal_id'],source['trial_number'],
        PipelineRecipe.model_validate(source['recipe']),Path(source['dataset_path']),episode,
        source['protocol_parameters'],source['dataset_hash'],source['code_revision'],
        source['environment_hash'],managed_episode_nonce=receipt['task_nonce'])
    result=RecipeExecutionResult(**{f.name:payload[f.name] for f in fields(RecipeExecutionResult)})
    report=link_feature_outcome(request,result)
    _write_publication_json(episode/'feature-feedback-delivered.json',
        {'attempt_id':row['attempt_id'],'task_nonce':receipt['task_nonce'],
         'result_sha256':receipt['result_sha256'],'delivered':True,'report':report})
