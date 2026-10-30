import errno
import json
import os
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ima import research_storage as storage
from ima.feature_definitions import FeatureDefinition, FeatureRegistry


def _authority(tmp_path, monkeypatch, free_bytes=8 * 1024**2, free_inodes=100_000):
    authority = storage.StorageAuthority(
        tmp_path / 'authority', headroom_bytes=0, headroom_inodes=0,
        statvfs=lambda _: SimpleNamespace(f_bavail=free_bytes, f_bfree=free_bytes,
                                          f_frsize=1, f_favail=free_inodes),
    )
    monkeypatch.setenv('IMA_STORAGE_ROOT', str(authority.root))
    monkeypatch.setattr(storage, 'StorageAuthority', lambda root: authority)
    return authority


def _definition(hypothesis='first narrative'):
    return FeatureDefinition.model_validate({
        'name': 'prior_speed',
        'expression_ast': {'op': 'divide', 'args': [
            {'op': 'ref', 'ref': 'distance'}, {'op': 'ref', 'ref': 'seconds'}]},
        'input_refs': ['distance', 'seconds'], 'unit': 'm/s', 'hypothesis': hypothesis,
    })


METADATA = {'distance': {'unit': 'm'}, 'seconds': {'unit': 's'}}


def test_feature_registry_root_denial_precedes_mkdir(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch, free_bytes=64 * 1024 - 1)
    root = tmp_path / 'denied' / 'feature-definitions'
    with pytest.raises(storage.StorageBlocked):
        FeatureRegistry(root)
    assert not root.parent.exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_register_is_readable_and_immutable(tmp_path, monkeypatch, enabled):
    monkeypatch.delenv('IMA_STORAGE_ROOT', raising=False)
    root = tmp_path / 'feature-definitions'
    registry = FeatureRegistry(root)
    authority = _authority(tmp_path, monkeypatch) if enabled else None
    definition = _definition()
    identifier = registry.register(definition, METADATA)
    path = root / f'{identifier}.json'
    published = path.read_bytes()
    assert registry.get(identifier).content_id() == identifier
    assert json.loads(published)['definition_id'] == identifier

    assert registry.register(_definition('updated narrative'), METADATA) == identifier
    assert path.read_bytes() == published
    assert list(root.glob('*.tmp')) == []
    if authority is not None:
        assert authority.status()['reserved_bytes'] == 0


def test_disabled_register_keeps_existing_write_once_contract(tmp_path, monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT', raising=False)
    root = tmp_path / 'feature-definitions'
    registry = FeatureRegistry(root)
    definition = _definition()
    identifier = registry.register(definition, METADATA)
    path = root / f'{identifier}.json'
    original = path.read_bytes()

    assert registry.register(_definition('updated narrative'), METADATA) == identifier
    assert path.read_bytes() == original
    assert registry.get(identifier).content_id() == identifier


def test_concurrent_duplicate_register_keeps_one_immutable_publication(tmp_path, monkeypatch):
    root = tmp_path / 'feature-definitions'
    registry = FeatureRegistry(root)
    authority = _authority(tmp_path, monkeypatch)
    barrier = Barrier(2)
    original_link = os.link
    links = []

    def synchronized_link(source, destination):
        barrier.wait(timeout=5)
        original_link(source, destination)
        links.append(Path(destination).read_bytes())

    with patch('ima.feature_definitions.os.link', side_effect=synchronized_link):
        with ThreadPoolExecutor(max_workers=2) as workers:
            jobs = [workers.submit(registry.register, _definition(text), METADATA)
                    for text in ('first narrative', 'second narrative')]
            identifiers = [job.result(timeout=10) for job in jobs]

    assert identifiers[0] == identifiers[1]
    path = root / f'{identifiers[0]}.json'
    assert path.is_file()
    assert len(links) == 1
    assert path.read_bytes() == links[0]
    assert registry.get(identifiers[0]).content_id() == identifiers[0]
    assert list(root.glob('*.tmp')) == []
    assert authority.status()['reserved_bytes'] == 0


def test_partial_enospc_cleans_staging_and_preserves_storage_classification(tmp_path, monkeypatch):
    root = tmp_path / 'feature-definitions'
    registry = FeatureRegistry(root)
    authority = _authority(tmp_path, monkeypatch)
    real_reserved = __import__('ima.feature_definitions', fromlist=['reserved_output_file']).reserved_output_file

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream

        def write(self, value):
            self.stream.write(value[:12])
            raise OSError(errno.ENOSPC, 'injected definition write full')

        def __getattr__(self, name):
            return getattr(self.stream, name)

    @contextmanager
    def partial_writer(path, mode='wb', **kwargs):
        with real_reserved(path, mode, **kwargs) as stream:
            yield PartialWriter(stream)

    monkeypatch.setattr('ima.feature_definitions.reserved_output_file', partial_writer)
    with pytest.raises(storage.StorageBlocked) as caught:
        registry.register(_definition(), METADATA)

    assert caught.value.diagnostic['failure_kind'] == 'infrastructure_storage'
    assert not list(root.glob('*.json'))
    assert not list(root.glob('*.tmp'))
    assert authority.status()['reserved_bytes'] == 0


def test_cleanup_failure_does_not_replace_primary_enospc(tmp_path, monkeypatch):
    root = tmp_path / 'feature-definitions'
    registry = FeatureRegistry(root)
    _authority(tmp_path, monkeypatch)
    module = __import__('ima.feature_definitions', fromlist=['reserved_output_file'])
    real_reserved = module.reserved_output_file
    real_unlink = Path.unlink

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream

        def write(self, value):
            self.stream.write(value[:12])
            raise OSError(errno.ENOSPC, 'primary definition full')

        def __getattr__(self, name):
            return getattr(self.stream, name)

    @contextmanager
    def partial_writer(path, mode='wb', **kwargs):
        with real_reserved(path, mode, **kwargs) as stream:
            yield PartialWriter(stream)

    def fail_temp_cleanup(path, *args, **kwargs):
        if path.parent == root and path.name.endswith('.tmp'):
            raise OSError(errno.EIO, 'injected cleanup failure')
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(module, 'reserved_output_file', partial_writer)
    monkeypatch.setattr(Path, 'unlink', fail_temp_cleanup)
    with pytest.raises(storage.StorageBlocked) as caught:
        registry.register(_definition(), METADATA)

    cause = caught.value.__cause__
    assert isinstance(cause, OSError) and cause.errno == errno.ENOSPC
    assert any('cleanup also failed' in note for note in getattr(cause, '__notes__', []))
    monkeypatch.setattr(Path, 'unlink', real_unlink)
    for temporary in root.glob('*.tmp'):
        temporary.unlink()
