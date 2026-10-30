"""Preparation metadata and stop requests use admitted publication paths."""
import errno
import threading
from types import SimpleNamespace

import pytest

from ima import research_controller as core
from ima import research_storage as storage
from ima.research_preparation import PreparationCache


@pytest.fixture
def authority(tmp_path, monkeypatch):
    a = storage.StorageAuthority(tmp_path / 'authority', headroom_bytes=0,
                                 headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT', str(a.root))
    monkeypatch.setattr(storage, 'StorageAuthority', lambda root: a)
    return a


def test_cache_directory_denial_precedes_mkdir(authority, tmp_path, monkeypatch):
    monkeypatch.setattr(authority, '_statvfs', lambda _: SimpleNamespace(
        f_bavail=0, f_frsize=1, f_favail=1000))
    root = tmp_path / 'missing' / 'folds'
    with pytest.raises(storage.StorageBlocked):
        PreparationCache(root)
    assert not root.parent.exists()


def test_stop_denial_preserves_old_marker(authority, tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    (campaign / 'STOP').write_text('accepted old stop\n')
    monkeypatch.setattr(authority, '_statvfs', lambda _: SimpleNamespace(
        f_bavail=0, f_frsize=1, f_favail=1000))
    with pytest.raises(storage.StorageBlocked):
        core.request_campaign_stop(campaign)
    assert (campaign / 'STOP').read_text() == 'accepted old stop\n'
    assert not (campaign / 'STOP.tmp').exists()


def test_owner_metadata_and_stop_readback_release_leases(authority, tmp_path):
    cache = PreparationCache(tmp_path / 'folds')
    identity = 'a' * 64
    with cache._producer_status(identity, {'attempt_id': 'fixture'}) as update:
        update('writing')
        assert cache.owner_status(identity)['stage'] == 'writing'
    assert cache.owner_status(identity)['stage'] == 'released'
    marker = core.request_campaign_stop(cache.root)
    assert marker.read_text().endswith('\n')
    assert authority.status()['reserved_bytes'] == 0


def test_heartbeat_pressure_surfaces_in_owner_thread(authority, tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path / 'folds')
    failed = threading.Event()
    real_write = core._write_json_atomic
    primary = storage.StorageBlocked('heartbeat full')

    def publish(path, payload):
        if threading.current_thread() is not threading.main_thread():
            failed.set()
            raise primary
        return real_write(path, payload)

    monkeypatch.setattr(core, '_write_json_atomic', publish)
    with pytest.raises(storage.StorageBlocked) as caught:
        with cache._producer_status('a' * 64, {}):
            assert failed.wait(5), 'heartbeat did not run'
    assert caught.value is primary
    assert authority.status()['reserved_bytes'] == 0


def test_owner_finalization_does_not_replace_builder_failure(authority, tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path / 'folds')
    real_write = core._write_json_atomic
    primary = ValueError('builder failed')

    def publish(path, payload):
        if payload['stage'] == 'released':
            raise OSError(errno.ENOSPC, 'secondary status failure')
        return real_write(path, payload)

    monkeypatch.setattr(core, '_write_json_atomic', publish)
    with pytest.raises(ValueError) as caught:
        with cache._producer_status('a' * 64, {}):
            raise primary
    assert caught.value is primary
    assert any('finalization also failed' in note for note in primary.__notes__)


def test_preparation_cleanup_failure_retains_native_storage_cause(authority, tmp_path, monkeypatch):
    from tests.test_research_preparation import key
    import ima.research_preparation as preparation

    cache = PreparationCache(tmp_path / 'folds')
    native = OSError(errno.ENOSPC, 'primary build storage failure')

    def fail_builder():
        raise native

    def fail_cleanup(path):
        raise OSError(errno.EACCES, 'secondary cleanup failure')

    monkeypatch.setattr(preparation.shutil, 'rmtree', fail_cleanup)
    with pytest.raises(storage.StorageBlocked) as caught:
        cache.prepare_fold(key(), fail_builder)
    assert caught.value.__cause__ is native
    assert any('staging cleanup also failed' in note for note in native.__notes__)
    assert not (cache.root / key().cache_id()).exists()
    assert authority.status()['reserved_bytes'] == 0
