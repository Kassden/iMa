from types import SimpleNamespace
from unittest.mock import Mock
import errno

import pandas as pd
import pytest

import ima.research_storage as storage
from ima.research_controller import _resolve_dataset


def configure(tmp_path, monkeypatch, available):
    root=tmp_path.resolve()
    authority=storage.StorageAuthority(root/'control',headroom_bytes=0,headroom_inodes=0,
        statvfs=lambda path:SimpleNamespace(f_frsize=1,f_blocks=available,
            f_bavail=available,f_bfree=available,f_favail=10000))
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    campaign=root/'campaign';campaign.mkdir()
    return authority,campaign


def test_fallback_dataset_denied_before_loading_or_creating_inputs(tmp_path,monkeypatch):
    authority,campaign=configure(tmp_path,monkeypatch,1024)
    loader=Mock(side_effect=AssertionError('Must admit before loading data'))
    monkeypatch.setattr('ima.rich_features.load_full_rich_history',loader)
    with pytest.raises(storage.StorageBlocked):
        _resolve_dataset(SimpleNamespace(dataset_path=None),campaign)
    loader.assert_not_called()
    assert not (campaign/'inputs').exists()
    assert authority.status()['reserved_bytes']==0


def test_fallback_gzip_roundtrip_and_cache_hit_without_new_admission(tmp_path,monkeypatch):
    authority,campaign=configure(tmp_path,monkeypatch,128*1024**2)
    frame=pd.DataFrame({'race_id':['r1','r2'],'horse_id':['H1','H2'],'rating':[50,60]})
    loader=Mock(return_value=frame)
    monkeypatch.setattr('ima.rich_features.load_full_rich_history',loader)
    path=_resolve_dataset(SimpleNamespace(dataset_path=None),campaign)
    pd.testing.assert_frame_equal(pd.read_csv(path),frame)
    assert list(path.parent.iterdir())==[path]
    assert authority.status()['reserved_bytes']==0
    monkeypatch.setattr(authority,'reserve',Mock(side_effect=AssertionError('Cache hit must not reserve')))
    assert _resolve_dataset(SimpleNamespace(dataset_path=None),campaign)==path
    loader.assert_called_once()


def test_fallback_native_full_leaves_no_published_or_partial_dataset(tmp_path,monkeypatch):
    authority,campaign=configure(tmp_path,monkeypatch,128*1024**2)
    frame=pd.DataFrame({'race_id':['r1']})
    monkeypatch.setattr('ima.rich_features.load_full_rich_history',lambda *args:frame)
    monkeypatch.setattr(frame,'to_csv',Mock(side_effect=OSError(errno.ENOSPC,'fixture full')))
    with pytest.raises(storage.StorageBlocked) as error:
        _resolve_dataset(SimpleNamespace(dataset_path=None),campaign)
    assert error.value.diagnostic['stage']=='controller_fallback_dataset'
    assert list((campaign/'inputs').iterdir())==[]
    assert authority.status()['reserved_bytes']==0


def test_explicit_dataset_is_read_only_and_does_not_reserve(tmp_path,monkeypatch):
    authority,campaign=configure(tmp_path,monkeypatch,1024)
    source=tmp_path/'source.csv';source.write_text('race_id\nr1\n')
    monkeypatch.setattr(authority,'reserve',Mock(side_effect=AssertionError('Explicit input is read-only')))
    assert _resolve_dataset(SimpleNamespace(dataset_path=source),campaign)==source


def test_fallback_reuses_single_orphan_staging_slot_after_interruption(tmp_path,monkeypatch):
    authority,campaign=configure(tmp_path,monkeypatch,128*1024**2)
    inputs=campaign/'inputs';inputs.mkdir()
    orphan=inputs/'rich-history.csv.gz.tmp';orphan.write_bytes(b'interrupted gzip')
    frame=pd.DataFrame({'race_id':['r1']})
    monkeypatch.setattr('ima.rich_features.load_full_rich_history',lambda *args:frame)
    path=_resolve_dataset(SimpleNamespace(dataset_path=None),campaign)
    pd.testing.assert_frame_equal(pd.read_csv(path),frame)
    assert list(inputs.iterdir())==[path]
    assert authority.status()['reserved_bytes']==0


def test_fallback_refuses_staging_symlink_without_deleting_foreign_target(tmp_path,monkeypatch):
    authority,campaign=configure(tmp_path,monkeypatch,128*1024**2)
    inputs=campaign/'inputs';inputs.mkdir()
    protected=tmp_path/'protected';protected.write_bytes(b'preserve')
    staging=inputs/'rich-history.csv.gz.tmp';staging.symlink_to(protected)
    monkeypatch.setattr('ima.rich_features.load_full_rich_history',lambda *args:pd.DataFrame({'race_id':['r1']}))
    with pytest.raises(ValueError):
        _resolve_dataset(SimpleNamespace(dataset_path=None),campaign)
    assert staging.is_symlink()
    assert protected.read_bytes()==b'preserve'
    assert authority.status()['reserved_bytes']==0
