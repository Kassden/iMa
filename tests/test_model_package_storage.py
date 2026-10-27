"""Admitted packages acquire their acceptance manifest only after all payloads."""
import errno
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from ima import research_model_package as packages
from ima import research_storage as storage
from ima.research_specs import PipelineRecipe
from tests.test_research_model_package import DummyProbabilityModel


@pytest.fixture
def authority(tmp_path, monkeypatch):
    a = storage.StorageAuthority(tmp_path / 'authority', headroom_bytes=0,
                                 headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT', str(a.root))
    monkeypatch.setattr(storage, 'StorageAuthority', lambda root: a)
    return a


class DistinctProbabilityModel:
    def __init__(self, label, probability):
        self.label = label
        self.probability = probability

    def predict_proba(self, frame):
        result = np.zeros(len(frame), dtype=float)
        for positions in frame.groupby('race_id', sort=False).indices.values():
            if len(positions) == 1:
                result[positions[0]] = 1.0
            elif len(positions) > 1:
                result[positions] = (1 - self.probability) / max(1, len(positions) - 1)
                result[positions[0]] = self.probability
        return result


def package(model=None):
    return packages.ResearchModelPackage(model or DummyProbabilityModel(), PipelineRecipe(),
                                         protocol_id='fixture', code_revision='test')


def test_package_admission_denied_before_directory(authority, tmp_path, monkeypatch):
    monkeypatch.setattr(authority, '_statvfs', lambda _: SimpleNamespace(
        f_bavail=0, f_frsize=1, f_favail=1000))
    path = tmp_path / 'package'
    with pytest.raises(storage.StorageBlocked):
        package().save(path)
    assert not path.exists()


def test_package_readback_and_accepted_identity_are_preserved(authority, tmp_path):
    import pandas as pd
    path = package().save(tmp_path / 'package')
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    loaded = packages.load_research_package(path)
    frame = pd.DataFrame({'race_id': ['r', 'r'], 'field_size': [2, 2]})
    np.testing.assert_array_equal(loaded.predict_proba(frame), [.5, .5])
    with pytest.raises(FileExistsError):
        package().save(path)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert authority.status()['reserved_bytes'] == 0


@pytest.mark.parametrize('boundary', ['model', 'recipe', 'manifest'])
def test_native_write_failure_never_creates_acceptance_marker(authority, tmp_path, monkeypatch, boundary):
    path = tmp_path / 'package'
    real_dump = packages.joblib.dump
    real_write = storage.ReservedFile.write

    def model_dump(value, stream, **kwargs):
        if boundary == 'model':
            stream.write(b'partial')
            raise OSError(errno.ENOSPC, 'native model full')
        return real_dump(value, stream, **kwargs)

    def metadata_write(stream, value):
        target = path / ('recipe.json' if boundary == 'recipe' else 'manifest.json.tmp')
        if boundary != 'model' and target.exists() and os.path.samestat(
                os.fstat(stream.fileno()), target.stat()):
            real_write(stream, value[:1])
            raise OSError(errno.ENOSPC, 'native metadata full')
        return real_write(stream, value)

    monkeypatch.setattr(packages.joblib, 'dump', model_dump)
    monkeypatch.setattr(storage.ReservedFile, 'write', metadata_write)
    with pytest.raises(storage.StorageBlocked):
        package().save(path)
    assert not (path / 'manifest.json').exists()
    with pytest.raises(FileNotFoundError):
        packages.load_research_package(path)
    assert authority.status()['reserved_bytes'] == 0


def test_private_retry_replaces_marker_after_later_artifact_failure(authority, tmp_path):
    import pandas as pd

    path = package(DistinctProbabilityModel('first', .25)).save(tmp_path / 'package')
    lock_path = path / '.save.lock'
    lock_inode = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    # Model the supervised caller failing after save() has returned. The next
    # private attempt may replace this package, while an ordinary save may not.
    with pytest.raises(OSError, match='later artifact'):
        with storage.reserved_output_file(path / 'feature-weights.csv', 'w') as stream:
            stream.write('partial diagnostic')
            raise OSError(errno.ENOSPC, 'later artifact write failed')
    with pytest.raises(FileExistsError):
        package(DistinctProbabilityModel('untrusted', .5)).save(path)

    package(DistinctProbabilityModel('retry', .75)).save(path, private_retry=True)
    loaded = packages.load_research_package(path)
    frame = pd.DataFrame({'race_id': ['r', 'r'], 'field_size': [2, 2]})
    assert loaded.model.label == 'retry'
    np.testing.assert_array_equal(loaded.predict_proba(frame), [.75, .25])
    assert not (path / 'feature-weights.csv').exists()
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == lock_inode


def test_private_retry_write_failure_keeps_primary_storage_error(authority, tmp_path, monkeypatch):
    path = package(DistinctProbabilityModel('accepted', .25)).save(tmp_path / 'package')

    def fail_after_partial(value, stream, **kwargs):
        stream.write(b'partial retry model')
        raise OSError(errno.ENOSPC, 'native retry model full')

    monkeypatch.setattr(packages.joblib, 'dump', fail_after_partial)
    with pytest.raises(storage.StorageBlocked, match='native retry model full') as caught:
        package(DistinctProbabilityModel('retry', .75)).save(path, private_retry=True)
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert not (path / 'manifest.json').exists()
    assert (path / '.save.lock').is_file()
    assert authority.status()['reserved_bytes'] == 0


def test_concurrent_saves_serialize_and_cannot_mix_models(authority, tmp_path, monkeypatch):
    import pandas as pd

    path = tmp_path / 'package'
    first_in_dump = threading.Event()
    second_in_dump = threading.Event()
    release_first = threading.Event()
    real_dump = packages.joblib.dump

    def controlled_dump(value, stream, **kwargs):
        if value.model.label == 'first':
            first_in_dump.set()
            if not release_first.wait(5):
                raise TimeoutError('first writer was not released')
        elif value.model.label == 'second':
            second_in_dump.set()
        return real_dump(value, stream, **kwargs)

    monkeypatch.setattr(packages.joblib, 'dump', controlled_dump)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(package(DistinctProbabilityModel('first', .2)).save, path)
        assert first_in_dump.wait(5)
        second_started = threading.Event()

        def save_second():
            second_started.set()
            return package(DistinctProbabilityModel('second', .8)).save(path)

        second = workers.submit(save_second)
        assert second_started.wait(5)
        try:
            second_entered_while_first_wrote = second_in_dump.wait(.25)
        finally:
            release_first.set()
        assert first.result(timeout=5) == path
        with pytest.raises(FileExistsError):
            second.result(timeout=5)

    assert not second_entered_while_first_wrote
    loaded = packages.load_research_package(path)
    frame = pd.DataFrame({'race_id': ['r', 'r'], 'field_size': [2, 2]})
    assert loaded.model.label == 'first'
    np.testing.assert_array_equal(loaded.predict_proba(frame), [.2, .8])
    assert authority.status()['reserved_bytes'] == 0


def test_legacy_package_load_does_not_create_lock(tmp_path, monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT', raising=False)
    path = tmp_path / 'legacy'
    path.mkdir()
    legacy = package()
    packages.joblib.dump(legacy, path / 'model.joblib')
    (path / 'manifest.json').write_text(
        json.dumps(asdict(legacy.manifest())),
        encoding='utf-8')

    assert packages.load_research_package(path).manifest().package_id == legacy.manifest().package_id
    assert not (path / '.save.lock').exists()


def test_disabled_legacy_save_keeps_overwrite_behavior_without_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT', raising=False)
    path = package(DistinctProbabilityModel('first', .2)).save(tmp_path / 'legacy')
    package(DistinctProbabilityModel('second', .8)).save(path)

    assert packages.load_research_package(path).model.label == 'second'
    assert not (path / '.save.lock').exists()


@pytest.mark.parametrize('kind', ['unknown', 'hardlink', 'symlink'])
def test_private_retry_preserves_all_payloads_when_any_entry_is_unsafe(authority, tmp_path, kind):
    path = package().save(tmp_path/'package')
    protected = tmp_path/'protected.txt'
    protected.write_bytes(b'keep this external payload')
    if kind == 'unknown':
        (path/'operator-notes.txt').write_text('keep these notes')
    elif kind == 'hardlink':
        os.link(protected, path/'feature-weights.csv')
    else:
        (path/'feature-weights.csv').symlink_to(protected)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError, match='Unsafe entry'):
        package().save(path, private_retry=True)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert protected.read_bytes() == b'keep this external payload'


@pytest.mark.parametrize('boundary', ['lock', 'directory'])
def test_package_lock_sync_failure_closes_every_duplicated_descriptor(authority, tmp_path, monkeypatch, boundary):
    path = tmp_path/'package'
    duplicated = []
    real_dup, real_sync = os.dup, os.fsync

    def duplicate(fd):
        retained = real_dup(fd)
        duplicated.append(retained)
        return retained

    def sync(fd):
        target = path/'.save.lock' if boundary == 'lock' else path
        if target.exists() and os.path.samestat(os.fstat(fd), target.stat()):
            raise OSError(errno.ENOSPC, 'injected lock setup sync failure')
        return real_sync(fd)

    monkeypatch.setattr(packages.os, 'dup', duplicate)
    monkeypatch.setattr(packages.os, 'fsync', sync)
    with pytest.raises(storage.StorageBlocked):
        package().save(path)
    assert duplicated
    for fd in duplicated:
        with pytest.raises(OSError) as caught:
            os.fstat(fd)
        assert caught.value.errno == errno.EBADF
    assert authority.status()['reserved_bytes'] == 0


def test_loader_waits_for_private_retry_under_shared_lock(authority, tmp_path, monkeypatch):
    path = package(DistinctProbabilityModel('accepted', .2)).save(tmp_path / 'package')
    retry_in_dump = threading.Event()
    release_retry = threading.Event()
    loader_entered_joblib = threading.Event()
    real_dump = packages.joblib.dump
    real_load = packages.joblib.load

    def paused_dump(value, stream, **kwargs):
        if value.model.label == 'retry':
            retry_in_dump.set()
            if not release_retry.wait(5):
                raise TimeoutError('retry writer was not released')
        return real_dump(value, stream, **kwargs)

    def tracked_load(*args, **kwargs):
        loader_entered_joblib.set()
        return real_load(*args, **kwargs)

    monkeypatch.setattr(packages.joblib, 'dump', paused_dump)
    monkeypatch.setattr(packages.joblib, 'load', tracked_load)
    with ThreadPoolExecutor(max_workers=2) as workers:
        retry = workers.submit(
            package(DistinctProbabilityModel('retry', .9)).save,
            path, private_retry=True)
        assert retry_in_dump.wait(5)
        loader = workers.submit(packages.load_research_package, path)
        try:
            loader_read_during_retry = loader_entered_joblib.wait(.25)
        finally:
            release_retry.set()
        assert retry.result(timeout=5) == path
        loaded = loader.result(timeout=5)

    assert not loader_read_during_retry
    assert loaded.model.label == 'retry'
