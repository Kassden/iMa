"""Storage pressure must not publish an incomplete forward-prediction object."""
import errno
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ima import research_storage as storage
from ima.prediction_store import PredictionArtifact, PredictionStore, prediction_key


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    authority = storage.StorageAuthority(tmp_path / 'authority', headroom_bytes=0,
                                         headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT', str(authority.root))
    monkeypatch.setattr(storage, 'StorageAuthority', lambda root: authority)
    return authority


def artifact():
    metadata = {'fit_scope': 'forward_oof', 'training_cutoff': '2020-01-01',
                'prediction_start': '2020-01-02'}
    return PredictionArtifact(np.array([.4, .6]), (('r', '1'), ('r', '2')), metadata)


def test_directory_admission_denial_precedes_every_mkdir(admitted, tmp_path, monkeypatch):
    monkeypatch.setattr(admitted, '_statvfs', lambda _: SimpleNamespace(
        f_bavail=0, f_frsize=1, f_favail=1000))
    root = tmp_path / 'missing' / 'oof'
    with pytest.raises(storage.StorageBlocked):
        PredictionStore(root)
    assert not root.parent.exists()
    assert admitted.status()['reserved_bytes'] == 0


def test_complete_npz_replays_and_releases_leases(admitted, tmp_path):
    store = PredictionStore(tmp_path / 'oof')
    value = artifact()
    key = prediction_key(value.metadata)
    store.put(key, value)
    store.put(key, value)
    np.testing.assert_array_equal(store.get(key, value.row_keys).values, value.values)
    assert admitted.status()['reserved_bytes'] == 0
    assert [path.name for path in store.root.iterdir()] == [key + '.npz']


def test_partial_native_enospc_never_publishes(admitted, tmp_path, monkeypatch):
    store = PredictionStore(tmp_path / 'oof')
    value = artifact()

    def partial(stream, **kwargs):
        stream.write(b'partial-npz')
        raise OSError(errno.ENOSPC, 'injected native full filesystem')

    monkeypatch.setattr(np, 'savez', partial)
    with pytest.raises(storage.StorageBlocked) as caught:
        store.put(prediction_key(value.metadata), value)
    assert caught.value.diagnostic['errno'] == errno.ENOSPC
    assert list(store.root.iterdir()) == []
    assert admitted.status()['reserved_bytes'] == 0


def test_cleanup_failure_keeps_native_pressure_cause(admitted, tmp_path, monkeypatch):
    store = PredictionStore(tmp_path / 'oof')
    value = artifact()
    real_unlink = __import__('os').unlink

    def partial(stream, **kwargs):
        stream.write(b'partial-npz')
        raise OSError(errno.ENOSPC, 'primary failure')

    def cleanup(path, *args, **kwargs):
        if Path(path).parent == store.root:
            raise OSError(errno.EACCES, 'secondary cleanup failure')
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(np, 'savez', partial)
    monkeypatch.setattr('ima.prediction_store.os.unlink', cleanup)
    with pytest.raises(storage.StorageBlocked) as caught:
        store.put(prediction_key(value.metadata), value)
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert any('cleanup also failed' in note for note in caught.value.__cause__.__notes__)
    assert store.get(prediction_key(value.metadata)) is None
    assert admitted.status()['reserved_bytes'] == 0
