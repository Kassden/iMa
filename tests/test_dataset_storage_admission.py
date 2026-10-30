import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from ima import research_storage as storage
from ima.dataset_registry import DatasetRegistry
from ima.feature_definitions import FeatureDefinition
from ima.feature_lifecycle import FeatureLifecycle
from ima.research_dataset_capacity import snapshot_storage_estimate
from tests.test_dataset_registry_adversarial import event, source_rows


def _authority(tmp_path, monkeypatch, free_bytes=4 * 1024**3, free_inodes=100_000):
    authority = storage.StorageAuthority(
        tmp_path / 'authority', headroom_bytes=0, headroom_inodes=0,
        statvfs=lambda _: SimpleNamespace(f_bavail=free_bytes, f_bfree=free_bytes,
                                          f_frsize=1, f_favail=free_inodes),
    )
    monkeypatch.setenv('IMA_STORAGE_ROOT', str(authority.root))
    monkeypatch.setattr(storage, 'StorageAuthority', lambda root: authority)
    return authority


def _dataset_fixture(tmp_path):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    frame = source_rows()
    frame.to_parquet(snapshot / 'runners.parquet', index=False)
    (snapshot / 'events.jsonl').write_text(json.dumps(event('admission-fixture')) + '\n')
    raw = snapshot / 'raw.json'
    raw.write_text(json.dumps({'manifest_id': 'admission-fixture'}))
    manifest = {
        'source_policy': 'HKJC-only; native storage admission fixture',
        'rows': len(frame), 'races': int(frame.race_id.nunique()),
        'files': {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
                  for name in ('runners.parquet', 'events.jsonl')},
    }
    (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    registry = DatasetRegistry(tmp_path / 'registry')
    registry.submit({
        'request_id': 'storage-admission',
        'raw_corpus_manifest_id': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'rationale': 'Exercise bounded original-format dataset admission.',
        'evidence_watermark': 'fixture',
        'protocol': {'min_train_races': 2, 'calibration_races': 1, 'score_races': 1,
                     'max_folds': 2, 'final_confirmation_races': 2},
    })
    return registry, snapshot, raw


def test_registry_root_denial_precedes_directory_creation(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch, free_bytes=0, free_inodes=0)
    root = tmp_path / 'denied-registry'
    with pytest.raises(storage.StorageBlocked):
        DatasetRegistry(root)
    assert not root.exists()


def test_lifecycle_constructor_denial_precedes_directory_creation(tmp_path, monkeypatch):
    registry = DatasetRegistry(tmp_path / 'registry')
    _authority(tmp_path, monkeypatch, free_bytes=0, free_inodes=100_000)
    with pytest.raises(storage.StorageBlocked):
        FeatureLifecycle(registry)
    assert not (registry.root / 'feature-lifecycle').exists()


def test_bytes_only_denial_precedes_directory_and_lock_creation(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch, free_bytes=0, free_inodes=100_000)
    denied_root = tmp_path / 'bytes-denied-registry'
    with pytest.raises(storage.StorageBlocked):
        DatasetRegistry(denied_root)
    assert not denied_root.exists()

    monkeypatch.delenv('IMA_STORAGE_ROOT')
    registry = DatasetRegistry(tmp_path / 'lock-registry')
    monkeypatch.setenv('IMA_STORAGE_ROOT', str(authority.root))
    lock = registry.root / 'locks' / 'bytes-pressure.lock'
    with pytest.raises(storage.StorageBlocked):
        with registry._lock('bytes-pressure'):
            pytest.fail('lock unexpectedly admitted')
    assert not lock.exists()
    assert authority.status()['reserved_bytes'] == 0


def test_missing_snapshot_estimate_does_not_create_build_paths(tmp_path, monkeypatch):
    registry = DatasetRegistry(tmp_path / 'registry')
    authority = _authority(tmp_path, monkeypatch)
    raw = tmp_path / 'raw.json'
    raw.write_text('{}')
    with pytest.raises(FileNotFoundError):
        registry.build('missing-source', source_snapshot=tmp_path / 'absent', raw_manifest=raw)
    assert not (registry.root / 'locks' / 'missing-source.lock').exists()
    assert list((registry.root / 'datasets').iterdir()) == []
    assert list((registry.root / 'staging').iterdir()) == []
    assert authority.status()['reserved_bytes'] == 0


def test_missing_raw_manifest_does_not_create_build_lock(tmp_path, monkeypatch):
    registry, snapshot, _ = _dataset_fixture(tmp_path)
    authority = _authority(tmp_path, monkeypatch)
    before = registry.get('storage-admission')
    with pytest.raises(FileNotFoundError):
        registry.build('storage-admission', source_snapshot=snapshot,
                       raw_manifest=tmp_path / 'absent-raw.json')
    assert registry.get('storage-admission') == before
    assert list((registry.root / 'datasets').iterdir()) == []
    assert list((registry.root / 'staging').iterdir()) == []
    assert authority.status()['reserved_bytes'] == 0


def test_standalone_lock_creation_is_denied_before_lock_or_metadata(tmp_path, monkeypatch):
    registry = DatasetRegistry(tmp_path / 'registry')
    lock = registry.root / 'locks' / 'pressure.lock'
    authority = _authority(tmp_path, monkeypatch, free_bytes=0, free_inodes=0)
    with pytest.raises(storage.StorageBlocked):
        with registry._lock('pressure'):
            pytest.fail('lock unexpectedly admitted')
    assert not lock.exists()
    assert authority.status()['reserved_bytes'] == 0


def test_snapshot_v2_estimate_verifies_existing_objects_without_writing(tmp_path):
    objects = tmp_path / 'objects'
    objects.mkdir()
    temporary = objects / 'table.parquet'
    pd.DataFrame({'value': [1.0, 2.0, 3.0]}).to_parquet(temporary, index=False)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    shard = objects / digest[:2]
    shard.mkdir()
    object_path = shard / digest
    temporary.rename(object_path)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'events.jsonl').write_text('event metadata\n')
    manifest = {'storage_format': 'ima.snapshot.objects.v2', 'tables': {'runners.parquet': {
        'object_root': str(objects), 'partitions': [{'sha256': digest, 'size': object_path.stat().st_size}],
    }}}
    (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    before = {str(path.relative_to(tmp_path)): path.stat().st_size for path in tmp_path.rglob('*') if path.is_file()}

    estimate = snapshot_storage_estimate(snapshot)

    after = {str(path.relative_to(tmp_path)): path.stat().st_size for path in tmp_path.rglob('*') if path.is_file()}
    assert estimate > object_path.stat().st_size
    assert after == before
    manifest['tables']['runners.parquet']['partitions'][0]['size'] += 1
    (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum or size'):
        snapshot_storage_estimate(snapshot)


def test_native_dataset_and_feature_parquet_roundtrip_under_admission(tmp_path, monkeypatch):
    registry, snapshot, raw = _dataset_fixture(tmp_path)
    authority = _authority(tmp_path, monkeypatch)
    parent = registry.build('storage-admission', source_snapshot=snapshot, raw_manifest=raw)
    verified = registry.verify(parent['dataset_id'])
    pd.testing.assert_frame_equal(pd.read_parquet(verified['features_path']),
                                  pd.read_parquet(registry.dataset_path(parent['dataset_id']) / 'features.parquet'))

    parent_dir = registry.dataset_path(parent['dataset_id'])
    definition = FeatureDefinition(
        name='admitted_weight_ratio', unit='1', input_refs=('actual_weight', 'declared_weight'),
        expression_ast={'op': 'divide', 'args': [
            {'op': 'ref', 'ref': 'actual_weight'}, {'op': 'ref', 'ref': 'declared_weight'}]},
    )
    lifecycle = FeatureLifecycle(registry)
    receipt = lifecycle.submit(parent_dir / 'features.parquet', parent_dir / 'manifest.json',
                               definitions=[definition])
    successor = lifecycle.build(receipt['build_id'])
    successor = registry.verify(successor['dataset_id'])
    features = pd.read_parquet(successor['features_path'])
    assert definition.feature_id in features
    assert features[definition.feature_id].notna().all()
    assert Path(successor['features_path']).name == 'features.parquet'
    assert (registry.dataset_path(successor['dataset_id']) / 'features.parquet').is_file()
    assert authority.status()['reserved_bytes'] == 0


def test_partial_native_parquet_full_is_blocked_and_never_published(tmp_path, monkeypatch):
    registry, snapshot, raw = _dataset_fixture(tmp_path)
    authority = _authority(tmp_path, monkeypatch)
    native = pd.DataFrame.to_parquet

    def partial_full(frame, path=None, *args, **kwargs):
        if hasattr(path, 'write'):
            path.write(b'partial-parquet')
            raise OSError(errno.ENOSPC, 'injected parquet full')
        return native(frame, path, *args, **kwargs)

    with patch.object(pd.DataFrame, 'to_parquet', partial_full):
        with pytest.raises(storage.StorageBlocked):
            registry.build('storage-admission', source_snapshot=snapshot, raw_manifest=raw)
    state = registry.get('storage-admission')
    assert state['status'] == 'blocked_storage'
    assert list((registry.root / 'datasets').iterdir()) == []
    assert list((registry.root / 'staging').iterdir()) == []
    assert authority.status()['reserved_bytes'] == 0
