"""Versioned keyed feature deltas against checksum-bound immutable parents."""
from __future__ import annotations
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
import pandas as pd
from .feature_expressions import InputMetadata
from .research_storage import (
    _OUTPUT_BUDGET,
    admitted_output,
    as_storage_blocked,
    reserved_output_file,
)
from .research_dataset_capacity import dataset_storage_estimate


FORMAT='ima.dataset.delta.v2'
KEYS=['race_id','horse_no']
PROTECTED={'date','race_id','horse_id','horse_no','result','finish_seconds','finish_time',
           'target_win','target_probability','target_rank_score','target_speed',
           'target_future_market_probability','target_log_final_win_odds',
           'market_probability','future_market_probability','market_raw',
           'odds_snapshot_at','forecast_at','final_win_odds','win_odds','market_odds',
           'place_odds','final_place_odds','current_decimal_odds'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def key_digest(frame):
    return hashlib.sha256(json.dumps(frame[KEYS].astype(str).values.tolist(),separators=(',',':')).encode()).hexdigest()


@contextmanager
def _temporary_directory_preserving_error(**kwargs):
    temporary = tempfile.TemporaryDirectory(**kwargs)
    try:
        yield temporary.name
    except BaseException as primary:
        try:
            temporary.cleanup()
        except BaseException as cleanup_error:
            if hasattr(primary, 'add_note'):
                primary.add_note(f'Dataset delta staging cleanup also failed: {cleanup_error!r}')
        raise
    else:
        temporary.cleanup()


def _validate_predictor(name, metadata, *, declared_forbidden=()):
    # Reuse the lifecycle's vocabulary and target-aware naming checks so this
    # low-level API cannot publish columns the registered path would reject.
    from .feature_lifecycle import _known_unit, _safe_name
    if (name in PROTECTED or name in declared_forbidden or isinstance(name,str)
            and name.startswith(('target_','future_','postrace_','post_race_'))):
        raise ValueError(f'Protected labels, outcomes or market column: {name}')
    _safe_name(name)
    if not isinstance(metadata, dict):
        raise ValueError(f'Missing validated predictor metadata: {name}')
    required={'unit','dtype','temporal_scope','target_tainted','eligible',
              'availability_proof','source_family'}
    if not required<=metadata.keys():
        raise ValueError(f'Predictor metadata lacks semantic availability fields: {name}')
    try:
        typed = InputMetadata.model_validate({key: metadata[key] for key in InputMetadata.model_fields if key in metadata})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid predictor metadata: {name}') from exc
    if not _known_unit(typed.unit):
        raise ValueError(f'Unsupported predictor unit: {name}: {typed.unit}')
    if (typed.temporal_scope != 'pre_race' or typed.target_tainted
            or metadata.get('eligible') is not True
            or not isinstance(metadata.get('availability_proof'), str)
            or not metadata['availability_proof'].strip()
            or not isinstance(typed.source_family, str) or not typed.source_family.strip()):
        raise ValueError(f'Predictor requires eligible, untainted, semantically declared pre-race availability: {name}')
    return typed


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _admit_output(output, estimated_bytes):
    # FeatureLifecycle.build already holds a registry-wide admission covering
    # its staging directory. Standalone calls acquire their own before mkdir.
    if _OUTPUT_BUDGET.get() is not None:
        yield
    else:
        with admitted_output(output,estimated_bytes=estimated_bytes,
                             stage='dataset_feature_delta',inodes=8):
            yield


def _unlink_if_same(path, identity):
    try:
        observed = Path(path).lstat()
        if (observed.st_dev, observed.st_ino) == identity:
            Path(path).unlink()
    except FileNotFoundError:
        pass


def read_feature_table(path,*,columns=None,positions=None,_seen=None):
    path=Path(path).absolute()
    if path.resolve()!=path: raise ValueError('Noncanonical dataset input')
    if path.suffix=='.parquet':
        frame=pd.read_parquet(path,columns=columns)
        return frame if positions is None else frame.iloc[list(positions)].reset_index(drop=True)
    if path.suffix!='.json': raise ValueError('Unsupported dataset representation')
    seen=set() if _seen is None else set(_seen)
    if path in seen: raise ValueError('Dataset parent cycle')
    seen.add(path)
    descriptor=json.loads(path.read_text())
    if descriptor.get('storage_format')!=FORMAT: raise ValueError('Unknown feature descriptor format')
    parent_path=Path(descriptor['parent_manifest_path']).absolute()
    registry=Path(descriptor['registry_root']).absolute()
    if (registry.resolve()!=registry or parent_path.resolve()!=parent_path or
        parent_path.parent.parent!=registry/'datasets' or parent_path.name!='manifest.json'):
        raise ValueError('Parent outside registered dataset namespace')
    if digest(parent_path)!=descriptor['parent_manifest_sha256']: raise ValueError('Parent manifest changed')
    parent=json.loads(parent_path.read_text())
    if parent['dataset_id']!=descriptor['parent_dataset_id']: raise ValueError('Parent identity mismatch')
    filename=Path(parent['features_path']).name
    if filename not in parent['files']: raise ValueError('Unbound parent table')
    parent_features=parent_path.parent/filename
    if digest(parent_features)!=parent['files'][filename]: raise ValueError('Parent table changed')
    delta=path.parent/'feature-delta.parquet'
    if delta.resolve()!=delta or digest(delta)!=descriptor['delta_sha256']:
        raise ValueError('Delta content changed')
    requested=list(descriptor['columns'] if columns is None else columns)
    if len(requested)!=len(set(requested)) or set(requested)-set(descriptor['columns']):
        raise ValueError('Invalid projected predictor columns')
    delta_columns=set(descriptor['delta_columns'])
    wanted_parent=list(dict.fromkeys(KEYS+[c for c in requested if c not in delta_columns]))
    base=read_feature_table(parent_features,columns=wanted_parent,_seen=seen)
    added=pd.read_parquet(delta,columns=list(dict.fromkeys(KEYS+[c for c in requested if c in delta_columns])))
    if base.duplicated(KEYS).any() or added.duplicated(KEYS).any() or len(base)!=descriptor['rows']:
        raise ValueError('Duplicate or changed delta population')
    try: pd.testing.assert_frame_equal(base[KEYS].reset_index(drop=True),added[KEYS].reset_index(drop=True),check_exact=True)
    except AssertionError as exc: raise ValueError('Delta runner order/population changed') from exc
    if key_digest(base)!=descriptor['ordered_keys_sha256']: raise ValueError('Parent runner identity changed')
    result=base.copy(deep=False)
    for name in requested:
        if name in delta_columns: result[name]=added[name].to_numpy()
    result=result[requested]
    return result if positions is None else result.iloc[list(positions)].reset_index(drop=True)


def write_feature_delta(parent_manifest_path,output,frame,*,registry_root,columns,replacements=(),catalog=None):
    parent_manifest_path=Path(parent_manifest_path).absolute(); output=Path(output).absolute()
    if output.resolve()!=output:
        raise ValueError('Noncanonical delta output path')
    parent=json.loads(parent_manifest_path.read_text())
    original=read_feature_table(parent_manifest_path.parent/Path(parent['features_path']).name)
    columns=list(columns); replacements=set(replacements)
    declared_forbidden=set()
    for field in ('market_only_columns','target_only_columns'):
        values=parent.get(field,())
        if not isinstance(values,(list,tuple)) or any(not isinstance(value,str) for value in values):
            raise ValueError('Invalid parent predictor exclusion contract')
        declared_forbidden.update(values)
    if not columns or len(columns)!=len(set(columns)) or not replacements<=set(columns):
        raise ValueError('Explicit unique delta columns/replacements required')
    for name in columns:
        _validate_predictor(name,(catalog or {}).get(name),declared_forbidden=declared_forbidden)
    if replacements & PROTECTED or any(c.startswith('target_') for c in replacements):
        raise ValueError('Cannot replace labels, identities or current outcome/market columns')
    if set(columns)&set(original)-replacements: raise ValueError('Undeclared parent column replacement')
    if not set(KEYS+columns)<=set(frame.columns):
        raise ValueError('Delta frame is missing declared predictor columns')
    if len(frame)!=len(original) or frame.duplicated(KEYS).any(): raise ValueError('Changed delta population')
    unchanged=[c for c in original if c not in replacements]
    try: pd.testing.assert_frame_equal(original[unchanged],frame[unchanged],check_exact=True)
    except (AssertionError,KeyError) as exc: raise ValueError('Undeclared parent values/order changed') from exc
    for name in columns:
        metadata=(catalog or {}).get(name,{})
        if name in replacements:
            old=parent['predictor_catalog'][name]
            if metadata['unit']!=old['unit'] or metadata['temporal_scope']!=old.get('temporal_scope','pre_race'):
                raise ValueError('Replacement changes units or temporal contract')
    delta=output/'feature-delta.parquet'
    pointer=output/'features.delta.json'
    if delta.exists() or pointer.exists():
        raise ValueError('Immutable delta output already exists')

    payload_bytes=int(frame[KEYS+columns].memory_usage(index=True,deep=True).sum())
    estimate=max(64*1024**2,2*payload_bytes+256*1024)
    output_created=False
    publication_committed=False
    published=[]
    identities={}
    try:
        with _admit_output(output,estimate):
            if output.exists():
                if not output.is_dir():
                    raise ValueError('Delta output path is not a directory')
            else:
                try:
                    output.mkdir(parents=True,exist_ok=False)
                    output_created=True
                except FileExistsError:
                    if not output.is_dir():
                        raise
            if delta.exists() or pointer.exists():
                raise ValueError('Immutable delta output already exists')
            with _temporary_directory_preserving_error(dir=output,prefix='.feature-delta-') as staging_name:
                staging=Path(staging_name)
                staged_delta=staging/'feature-delta.parquet'
                with reserved_output_file(staged_delta) as stream:
                    frame[KEYS+columns].to_parquet(stream,index=False,compression='zstd')
                descriptor={'storage_format':FORMAT,'registry_root':str(Path(registry_root).absolute()),
                    'parent_manifest_path':str(parent_manifest_path),'parent_manifest_sha256':digest(parent_manifest_path),
                    'parent_dataset_id':parent['dataset_id'],'delta_sha256':digest(staged_delta),
                    'delta_columns':columns,'replacements':sorted(replacements),'rows':len(frame),
                    'columns':list(frame.columns),'ordered_keys_sha256':key_digest(original)}
                staged_pointer=staging/'features.delta.json'
                with reserved_output_file(staged_pointer,'x') as stream:
                    json.dump(descriptor,stream,sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                pd.testing.assert_frame_equal(read_feature_table(staged_pointer),frame,check_exact=True)

                for staged,final in ((staged_delta,delta),(staged_pointer,pointer)):
                    info=staged.lstat()
                    identity=(info.st_dev,info.st_ino)
                    # Hard-link publication is atomic and fails if an output
                    # appeared after the immutable-name preflight.
                    os.link(staged,final)
                    if final==pointer:
                        # The complete descriptor makes the pair consumable.
                        # Never roll it back because a later durability step
                        # failed; readers may already have accepted it.
                        publication_committed=True
                    identities[final]=identity
                    published.append(final)
                    staged.unlink()
                    if final==pointer:
                        _sync_directory(output)
        return descriptor
    except BaseException as primary:
        if not publication_committed:
            for final in reversed(published):
                try:
                    _unlink_if_same(final,identities[final])
                except BaseException as cleanup_error:
                    if hasattr(primary,'add_note'):
                        primary.add_note(f'Dataset delta publication cleanup also failed: {cleanup_error!r}')
            if output_created:
                try:
                    output.rmdir()
                except (FileNotFoundError, OSError):
                    pass
        converted=as_storage_blocked(primary,stage='dataset_feature_delta',output=output)
        if converted is not None and converted is not primary:
            raise converted from primary
        raise
