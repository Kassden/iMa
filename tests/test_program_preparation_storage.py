import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ima import research_storage as storage
from ima.research_expansion import _prepare_program
from ima.research_specs import PipelineRecipe
from tests.test_research_executor_v6 import dataset


def authority_at(root,monkeypatch,*,available=1024**3):
    authority=storage.StorageAuthority(root/'authority',headroom_bytes=0,headroom_inodes=0,
        statvfs=lambda _:SimpleNamespace(f_frsize=1,f_blocks=2*1024**3,f_bavail=available,f_favail=100_000))
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda _:authority)
    return authority


def recipe():
    return PipelineRecipe.model_validate({'schema_version':3,'model':{'kind':'benter_conditional_logit'},
                                          'blend':{'kind':'none'}})


def test_program_preparation_denial_precedes_dataset_loading(tmp_path,monkeypatch):
    source,_=dataset(tmp_path)
    authority=authority_at(tmp_path,monkeypatch,available=0)
    output=tmp_path/'programs'
    with patch('ima.research_executor._load_dataset') as load:
        with pytest.raises(storage.StorageBlocked):
            _prepare_program(source,recipe(),hashlib.sha256(source.read_bytes()).hexdigest(),output)
        load.assert_not_called()
    assert not output.exists()
    assert authority.status()['reserved_bytes']==0


def test_program_preparation_actual_frame_and_lease_release(tmp_path,monkeypatch):
    source,frame=dataset(tmp_path)
    authority=authority_at(tmp_path,monkeypatch)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    result=_prepare_program(source,recipe(),digest,tmp_path/'programs')
    assert result['dataset_digest']==digest
    assert result['rows']==len(frame)
    assert result['status']=='base_features_ready'
    assert authority.status()['reserved_bytes']==0
