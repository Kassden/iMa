"""A cache below quota can still block the physical filesystem headroom."""
import fcntl
import errno
import os
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from ima.research_cache_capacity import trim_for_capacity
from ima.research_storage import StorageAuthority, StorageBlocked


def fixture(tmp_path, *, available=1000, inodes=1000):
    root = tmp_path / 'fold-cache'
    root.mkdir(mode=0o700)
    paths, allocated = [], {}
    for letter in ('a', 'b'):
        path = root / (letter * 64)
        path.mkdir()
        for name in ('arrays.joblib', 'fitted.joblib', 'manifest.json'):
            (path / name).write_bytes(b'fixture-derived-payload' * 1000)
        (root / (path.name + '.lock')).touch()
        (root / (path.name + '.pin.lock')).touch()
        paths.append(path)
        allocated[path] = sum(p.stat().st_blocks * 512 for p in path.iterdir())
    original_count = len(list(root.rglob('*')))

    def space(_):
        reclaimed = sum(size for path, size in allocated.items() if not path.exists())
        freed_inodes = original_count - len(list(root.rglob('*')))
        return SimpleNamespace(f_bavail=available + reclaimed, f_frsize=1,
                               f_favail=inodes + freed_inodes)

    authority = StorageAuthority(tmp_path / 'authority', headroom_bytes=2000,
                                 headroom_inodes=4, cache_limit_bytes=1024**3,
                                 statvfs=space)
    for path in paths:
        authority.record_cache(root, path.name, allocated[path])
    return authority, root, paths


def test_physical_pressure_evicts_oldest_even_far_below_quota(tmp_path):
    authority, root, paths = fixture(tmp_path)
    before = {p.name: p.stat().st_ino for p in root.glob('*.lock')}
    result = trim_for_capacity(authority, required_bytes=5000)
    assert result['evicted'] == [str(paths[0])]
    assert not paths[0].exists() and paths[1].exists()
    assert {p.name: p.stat().st_ino for p in root.glob('*.lock')} == before
    assert result['physical_after']['available_bytes'] >= 7000
    with authority.admitted(bytes_required=5000, inodes_required=8,
                            stage='consumer', output=root):
        assert authority.status()['reserved_bytes'] == 5000


def test_active_pin_wins_and_inactive_candidate_can_be_reclaimed(tmp_path):
    authority, root, paths = fixture(tmp_path)
    with (root / (paths[0].name + '.lock')).open('rb') as active:
        fcntl.flock(active, fcntl.LOCK_SH)
        result = trim_for_capacity(authority, required_bytes=5000)
    assert result['evicted'] == [str(paths[1])]
    assert paths[0].exists() and not paths[1].exists()


def test_all_pinned_returns_retryable_storage_block_without_deletion(tmp_path):
    authority, root, paths = fixture(tmp_path)
    first = (root / (paths[0].name + '.lock')).open('rb')
    second = (root / (paths[1].name + '.lock')).open('rb')
    try:
        fcntl.flock(first, fcntl.LOCK_SH)
        fcntl.flock(second, fcntl.LOCK_SH)
        with pytest.raises(StorageBlocked) as caught:
            trim_for_capacity(authority, required_bytes=5000)
        assert caught.value.diagnostic['deficit']['bytes'] == 6000
        assert caught.value.diagnostic['retryable']
        assert all(path.exists() for path in paths)
    finally:
        first.close()
        second.close()


def test_inode_pressure_also_reclaims_and_healthy_hit_does_not_evict(tmp_path):
    authority, root, paths = fixture(tmp_path, available=10**7, inodes=4)
    result = trim_for_capacity(authority, required_bytes=0, required_inodes=4,
                               cache_growth_bytes=0)
    assert result['evicted'] == [str(paths[0])]
    assert paths[1].exists()
    result = trim_for_capacity(authority, required_bytes=1, required_inodes=1,
                               cache_growth_bytes=0)
    assert result['evicted'] == []


def test_producer_winning_after_trim_is_still_denied_by_final_reservation(tmp_path):
    authority, root, paths = fixture(tmp_path, available=10_000_000)
    result = trim_for_capacity(authority, required_bytes=1024**2, cache_growth_bytes=0)
    assert result['evicted'] == []
    with authority.admitted(bytes_required=9_000_000, inodes_required=1,
                            stage='competing_producer', output=root):
        with pytest.raises(StorageBlocked):
            authority.reserve(bytes_required=1024**2, inodes_required=8,
                              stage='fold_preparation', output=root)
        assert all(path.exists() for path in paths)
    assert authority.status()['reserved_bytes'] == 0


@pytest.mark.parametrize('boundary', ['open', 'dup'])
def test_failed_atomic_pin_setup_keeps_no_extra_descriptor(tmp_path, monkeypatch, boundary):
    import builtins
    from ima.research_preparation import PreparationCache
    from tests.test_research_preparation import key, build

    monkeypatch.delenv('IMA_STORAGE_ROOT', raising=False)
    cache = PreparationCache(tmp_path/'cache')
    cache.prepare(key(), build)
    real_open, real_dup = builtins.open, os.dup
    pins, duplicate_calls = [], []

    def opened(path, *args, **kwargs):
        if str(path).endswith('.pin.lock'):
            if boundary == 'open':
                raise OSError(errno.EACCES, 'injected pin-open failure')
            stream = real_open(path, *args, **kwargs)
            pins.append(stream)
            return stream
        return real_open(path, *args, **kwargs)

    def duplicate(fd):
        duplicate_calls.append(fd)
        if boundary == 'dup':
            raise OSError(errno.EMFILE, 'injected dup failure')
        return real_dup(fd)

    monkeypatch.setattr(builtins, 'open', opened)
    monkeypatch.setattr(os, 'dup', duplicate)
    with ExitStack() as owned_pins:
        with pytest.raises(OSError):
            cache.prepare(key(), build, pins=owned_pins)
    if boundary == 'open':
        assert duplicate_calls == []
    else:
        assert pins and all(pin.closed for pin in pins)
