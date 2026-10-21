from dataclasses import FrozenInstanceError, replace
import errno
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
from types import SimpleNamespace

import pytest

from ima.research_control import ControlIntentStore, ControlPolicy
from ima.research_storage import StorageAuthority, StorageBlocked


H = 16 * 1024**2


def policy():
    return ControlPolicy(max_envelope_bytes=8192, page_size=4096, max_db_pages=32,
                         max_journal_bytes=3 * 1024**2, metadata_bytes=65536,
                         allowance_bytes=4 * 1024**2)


def authority(root, free=64 * 1024**2, inodes=100):
    return StorageAuthority(root, headroom_bytes=H, headroom_inodes=4,
                            statvfs=lambda _: SimpleNamespace(
                                f_bavail=free, f_frsize=1, f_favail=inodes))


@pytest.fixture
def store(tmp_path):
    a = authority(tmp_path / 'authority')
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    return ControlIntentStore.bootstrap(a, policy()), campaign


def enqueue(campaign, role, number, payload, destination='tracking'):
    conn = sqlite3.connect(campaign / 'trace-outbox.sqlite')
    conn.execute('CREATE TABLE IF NOT EXISTS traces(role TEXT, number INTEGER, payload TEXT, tracking_uri TEXT, PRIMARY KEY(role,number))')
    conn.execute('INSERT OR REPLACE INTO traces VALUES(?,?,?,?)',
                 (role, number, json.dumps(payload), destination))
    conn.commit()
    conn.close()


def test_explicit_policy_bootstrap_and_sqlite_readback(store):
    s, campaign = store
    assert s.policy == policy()
    with pytest.raises(FrozenInstanceError):
        s.policy.allowance_bytes = 1
    with s.claim(campaign, 'tracking'):
        reservation = s.reserve_decision(1, {'evidence': 'sha256:abc'})
        assert json.loads(reservation)['state'] == 'reserved'
        raw = s.stage('decision', 1, {'nested': [1, {'created_at': 'exact'}]},
                      evidence_identity={'evidence': 'sha256:abc'})
        assert type(raw) is bytes
        assert s.pending() == (raw,)
        with sqlite3.connect(s.db) as conn:
            assert conn.execute('SELECT envelope FROM slots').fetchone() == (raw,)
            assert conn.execute('PRAGMA page_size').fetchone()[0] == policy().page_size
        with s._connect() as conn:
            assert conn.execute('PRAGMA journal_mode').fetchone() == ('persist',)
            assert conn.execute('PRAGMA synchronous').fetchone() == (2,)
            assert conn.execute('PRAGMA temp_store').fetchone() == (2,)
            assert conn.execute('PRAGMA max_page_count').fetchone() == (32,)
            assert conn.execute('PRAGMA journal_size_limit').fetchone() == (policy().max_journal_bytes,)
    reopened = ControlIntentStore(s.authority)
    assert reopened.pending() == (raw,)
    assert reopened.allocated_bytes() < policy().allowance_bytes


@pytest.mark.parametrize('role', ['execution', 'dataset', 'betting'])
def test_global_two_slot_limit_and_all_operational_roles(store, role):
    s, campaign = store
    with s.claim(campaign, 'tracking'):
        s.reserve_decision(1, {'revision': 'r1'})
        raw = s.stage(role, 3, {'a': 1}, signature='execution-v1')
        assert len(s.pending()) == 2
        assert s.stage(role, 3, {'a': 1}, signature='execution-v1') == raw
        for other in {'execution', 'dataset', 'betting'}:
            with pytest.raises(StorageBlocked):
                s.stage(other, 4, {'a': 1})
        with pytest.raises(StorageBlocked):
            s.reserve_decision(2, {'revision': 'r2'})
        with pytest.raises(StorageBlocked):
            s.stage(role, 3, {'a': 2}, signature='execution-v1')
        with pytest.raises(StorageBlocked):
            s.stage(role, 3, {'a': 1}, signature='changed')


def test_reserved_decision_survives_and_requires_exact_evidence(store):
    s, campaign = store
    with s.claim(campaign, 'tracking'):
        with pytest.raises(ValueError, match='Reserve decision'):
            s.stage('decision', 1, {}, evidence_identity={'revision': 'frozen'})
        raw = s.reserve_decision(1, {'revision': 'frozen'})
    s = ControlIntentStore(s.authority)
    with s.claim(campaign, 'tracking'):
        assert s.reserve_decision(1, {'revision': 'frozen'}) == raw
        with pytest.raises(StorageBlocked):
            s.reserve_decision(2, {'revision': 'frozen'})
        with pytest.raises(StorageBlocked):
            s.stage('decision', 1, {}, evidence_identity={'revision': 'different'})
        with pytest.raises(ValueError, match='complete'):
            s.acknowledge('decision', 1)
        assert s.pending() == (raw,)
        complete = s.stage('decision', 1, {'response': 'received'}, evidence_identity={'revision': 'frozen'})
        assert s.stage('decision', 1, {'response': 'received'}, evidence_identity={'revision': 'frozen'}) == complete


def test_complete_envelope_limit_freezes_payload_and_no_truncation(store):
    s, campaign = store
    payload = {'evidence': ['initial'], 'unicode': '\u4e2d'}
    with s.claim(campaign, 'tracking'):
        raw = s.stage('execution', 1, payload)
        payload['evidence'].append('mutated')
        assert json.loads(s.pending()[0])['payload']['evidence'] == ['initial']
        assert s.pending() == (raw,)
    other = campaign.parent / 'other'
    other.mkdir()
    with pytest.raises(StorageBlocked):
        with s.claim(other, 'tracking'):
            pass
    with s.claim(campaign, 'tracking'):
        enqueue(campaign, 'execution', 1, json.loads(raw)['payload'])
        s.acknowledge('execution', 1)
        with pytest.raises(StorageBlocked, match='envelope'):
            s.stage('execution', 2, {'x': 'a' * s.policy.max_envelope_bytes})
        with pytest.raises(ValueError):
            s.stage('execution', 2, {'x': float('nan')})
        assert s.pending() == ()


def test_ack_requires_committed_full_payload_and_destination(store):
    s, campaign = store
    payload = {'created_at': 'must-match', 'evidence': {'a': [1, 2]}}
    with s.claim(campaign, 'tracking'):
        raw = s.stage('execution', 1, payload, signature={'revision': 2})
        with pytest.raises(FileNotFoundError):
            s.acknowledge('execution', 1)
        enqueue(campaign, 'execution', 2, payload)
        with pytest.raises(ValueError, match='does not match'):
            s.acknowledge('execution', 1)
        enqueue(campaign, 'execution', 1, payload, 'wrong')
        with pytest.raises(ValueError):
            s.acknowledge('execution', 1)
        enqueue(campaign, 'execution', 1, {**payload, 'created_at': 'changed'})
        with pytest.raises(ValueError):
            s.acknowledge('execution', 1)
        assert s.pending() == (raw,)
        enqueue(campaign, 'execution', 1, payload)
        conn = sqlite3.connect(campaign / 'trace-outbox.sqlite')
        conn.execute("UPDATE traces SET payload='{}' WHERE number=1")
        s.acknowledge('execution', 1)  # An uncommitted conflicting update is invisible.
        conn.rollback()
        conn.close()
        assert s.pending() == ()


def test_owner_lock_is_distinct_and_blocks_instances_and_campaigns(store):
    s, campaign = store
    second = ControlIntentStore(s.authority)
    inode = (s.root / 'owner.lock').stat().st_ino
    with s.claim(campaign, 'tracking'):
        assert s.authority.status()['headroom_bytes'] == H
        with pytest.raises(StorageBlocked, match='controller'):
            with second.claim(campaign, 'tracking'):
                pass
        s.stage('betting', 1, {})
    with pytest.raises(StorageBlocked, match='destination'):
        with second.claim(campaign, 'different'):
            pass
    with second.claim(campaign, 'tracking'):
        assert second.pending()
    assert (s.root / 'owner.lock').stat().st_ino == inode


def _process_owner(root, campaign, ready, stop, reserve=False):
    s = ControlIntentStore(authority(Path(root)))
    try:
        with s.claim(Path(campaign), 'tracking'):
            if reserve:
                s.reserve_decision(9, {'revision': 'before-dispatch'})
            else:
                s.stage('execution', 9, {'stable': [1, 2]}, signature='sig')
            ready.put(s.pending())
            stop.wait(20)
    except StorageBlocked:
        ready.put('blocked')


@pytest.mark.parametrize('reserve', [False, True])
def test_process_ownership_and_kill_reopen_persistence(store, reserve):
    s, campaign = store
    ctx = multiprocessing.get_context('spawn')
    ready, stop = ctx.Queue(), ctx.Event()
    first = ctx.Process(target=_process_owner, args=(s.authority.root, campaign, ready, stop, reserve))
    first.start()
    try:
        raw = ready.get(timeout=15)
        second = ctx.Process(target=_process_owner, args=(s.authority.root, campaign, ready, stop))
        second.start()
        assert ready.get(timeout=15) == 'blocked'
        second.join(15)
        assert second.exitcode == 0
        os.kill(first.pid, signal.SIGKILL)
        first.join(15)
        assert first.exitcode == -signal.SIGKILL
        reopened = ControlIntentStore(s.authority)
        with reopened.claim(campaign, 'tracking'):
            assert reopened.pending() == raw
            if reserve:
                with pytest.raises(StorageBlocked):
                    reopened.reserve_decision(10, {'revision': 'next'})
    finally:
        if first.is_alive():
            first.kill()
            first.join(15)


def test_control_allowance_and_normal_reservations_do_not_reduce_H(store):
    s, campaign = store
    usage = s.allocated_bytes()
    # Available bytes include unused C inside H, not an additional normal lease.
    threshold = H - usage
    s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=threshold, f_frsize=1, f_favail=100)
    with s.claim(campaign, 'tracking'):
        s.stage('execution', 1, {})
        assert s.authority.status()['headroom_bytes'] == H
        assert s.authority.status()['reservations'] == {}
        assert not (s.authority.root / 'state.json').exists()
        with pytest.raises(StorageBlocked):
            s.authority.reserve(bytes_required=1, stage='normal', output=campaign)
        s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=0, f_frsize=1, f_favail=100)
        with pytest.raises(StorageBlocked):
            s.stage('execution', 1, {})


def test_outstanding_normal_reservations_and_inode_pressure_block_control(store):
    s, campaign = store
    token = s.authority.reserve(bytes_required=1024**2, stage='normal', output=campaign)
    ledger = (s.authority.root / 'state.json').read_bytes()
    usage = s.allocated_bytes()
    s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=H - usage, f_frsize=1, f_favail=100)
    with pytest.raises(StorageBlocked):
        s.pending()
    assert (s.authority.root / 'state.json').read_bytes() == ledger
    s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=64 * 1024**2, f_frsize=1, f_favail=3)
    with pytest.raises(StorageBlocked):
        s.pending()
    s.authority.release(token)


def test_dead_reservations_reconcile_without_writes_but_unknown_owners_block(store,monkeypatch):
    import ima.research_storage as storage
    s,campaign=store
    token=s.authority.reserve(bytes_required=1024**2,stage='normal',output=campaign)
    pid=os.fork()
    if pid==0:os._exit(0)
    os.waitpid(pid,0)
    assert storage._process_identity(pid) is None
    path=s.authority.root/'state.json'
    state=json.loads(path.read_text())
    state['reservations'][token].update(pid=pid,process_identity='dead-fixture')
    path.write_text(json.dumps(state))
    original=path.read_bytes()
    usage=s.allocated_bytes()
    s.authority._statvfs=lambda _:SimpleNamespace(f_bavail=H-usage,f_frsize=1,f_favail=100)
    actual_identity=storage._process_identity
    with monkeypatch.context() as patched:
        patched.setattr(storage,'_process_identity',lambda value:'unknown' if value==pid else actual_identity(value))
        with pytest.raises(StorageBlocked):s.pending()
    assert s.pending()==()
    assert ControlIntentStore(s.authority).pending()==()
    assert path.read_bytes()==original


def test_bootstrap_requires_healthy_complete_allowance_and_preserves_partial(tmp_path):
    a = authority(tmp_path / 'a', free=H + policy().allowance_bytes - 1)
    with pytest.raises(StorageBlocked):
        ControlIntentStore.bootstrap(a, policy())
    assert not (a.root / 'control-intents').exists()
    a = authority(tmp_path / 'b', inodes=8)
    with pytest.raises(StorageBlocked):
        ControlIntentStore.bootstrap(a, policy())
    a = authority(tmp_path / 'c')
    (a.root / 'control-intents').mkdir()
    evidence = a.root / 'control-intents' / 'failure-evidence'
    evidence.write_bytes(b'preserve')
    with pytest.raises(FileNotFoundError):
        ControlIntentStore.bootstrap(a, policy())
    assert evidence.read_bytes() == b'preserve'


def test_missing_corrupt_changed_policy_and_identity_are_never_empty(store):
    s, campaign = store
    with pytest.raises(ValueError, match='changed'):
        ControlIntentStore.bootstrap(s.authority, replace(policy(), max_envelope_bytes=4096))
    db_bytes = s.db.read_bytes()
    s.db.write_bytes(b'corrupt')
    with pytest.raises(sqlite3.DatabaseError):
        ControlIntentStore(s.authority)
    assert s.db.read_bytes() == b'corrupt'
    s.db.write_bytes(db_bytes)
    lock = s.root / 'owner.lock'
    lock.rename(s.root / 'old-owner')
    lock.touch(mode=0o600)
    with pytest.raises(ValueError):
        ControlIntentStore(s.authority)


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'extra', 'missing'])
def test_reject_unsafe_or_missing_control_files(store, kind):
    s, _ = store
    journal = s.root / 'intents.sqlite-journal'
    if kind == 'extra':
        (s.root / 'retry-1').touch()
    elif kind == 'missing':
        (s.root / 'policy.json').unlink()
    elif kind == 'hardlink':
        os.link(journal, s.root.parent / 'hardlink')
    else:
        journal.unlink()
        journal.symlink_to(s.db)
    with pytest.raises((ValueError, FileNotFoundError)):
        ControlIntentStore(s.authority)


def test_retries_and_reuse_keep_fixed_files_and_bounded_allocations(store):
    s, campaign = store
    names = {p.name for p in s.root.iterdir()}
    owner_inode = (s.root / 'owner.lock').stat().st_ino
    peaks = []
    with s.claim(campaign, 'tracking'):
        for number in range(25):
            payload = {'data': 'x' * 6000, 'number': number}
            raw = s.stage('dataset', number, payload)
            for _ in range(3):
                assert s.stage('dataset', number, payload) == raw
                with pytest.raises(StorageBlocked):
                    s.stage('betting', number, {})
            enqueue(campaign, 'dataset', number, payload)
            s.acknowledge('dataset', number)
            peaks.append(s.allocated_bytes())
            assert {p.name for p in s.root.iterdir()} == names
    assert max(peaks) <= s.policy.allowance_bytes
    assert max(peaks[5:]) == min(peaks[5:])
    assert (s.root / 'owner.lock').stat().st_ino == owner_inode
    assert s.authority.status()['reservations']=={}
    assert json.loads((s.authority.root/'state.json').read_text())['reservations']=={}


def test_full_and_os_pressure_typed_other_errors_preserved(store, monkeypatch):
    s, campaign = store
    original = sqlite3.connect
    def full(*args, **kwargs):
        exc = sqlite3.OperationalError('database or disk is full')
        exc.sqlite_errorcode = sqlite3.SQLITE_FULL
        raise exc
    with s.claim(campaign, 'tracking'):
        monkeypatch.setattr(sqlite3, 'connect', full)
        with pytest.raises(StorageBlocked):
            s.stage('execution', 1, {})
        monkeypatch.setattr(sqlite3, 'connect', original)
        assert s.pending() == ()
    for error in (OSError(errno.ENOSPC, 'full'), OSError(errno.EDQUOT, 'quota')):
        with pytest.raises(StorageBlocked):
            with s._pressure():
                raise error
    error = sqlite3.OperationalError('unrelated failure')
    with pytest.raises(sqlite3.OperationalError) as caught:
        with s._pressure():
            raise error
    assert caught.value is error


def test_invalid_policy_limits(tmp_path):
    with pytest.raises(ValueError):
        replace(policy(), max_journal_bytes=100)
    with pytest.raises(ValueError):
        replace(policy(), allowance_bytes=1)
    with pytest.raises(ValueError):
        replace(policy(), page_size=True)
    a = authority(tmp_path / 'a')
    with pytest.raises(ValueError, match='strictly below'):
        ControlIntentStore.bootstrap(a, replace(policy(), allowance_bytes=H))


def _kill_before_commit(root, campaign):
    s = ControlIntentStore(authority(Path(root)))
    original = sqlite3.connect
    class KillCommit(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.execute('PRAGMA cache_size=1')
            self.execute('PRAGMA cache_spill=1')
        def commit(self):
            os.kill(os.getpid(), signal.SIGKILL)
    with s.claim(Path(campaign), 'tracking'):
        sqlite3.connect = lambda *args, **kwargs: original(*args, **kwargs, factory=KillCommit)
        s.stage('decision', 7, {'response': 'x' * 30000}, evidence_identity={'revision': 'sealed'})


def test_kill_before_commit_recovers_prior_reserved_state(tmp_path):
    a = authority(tmp_path / 'authority')
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    s = ControlIntentStore.bootstrap(a, replace(policy(), max_envelope_bytes=50000))
    with s.claim(campaign, 'tracking'):
        reserved = s.reserve_decision(7, {'revision': 'sealed'})
    ctx = multiprocessing.get_context('spawn')
    child = ctx.Process(target=_kill_before_commit, args=(s.authority.root, campaign))
    child.start()
    child.join(15)
    try:
        assert child.exitcode == -signal.SIGKILL
        assert (s.root / 'intents.sqlite-journal').read_bytes()[:8] == bytes.fromhex('d9d505f920a163d7')
        reopened = ControlIntentStore(s.authority)
        with reopened.claim(campaign, 'tracking'):
            assert reopened.pending() == (reserved,)
            with pytest.raises(StorageBlocked):
                reopened.reserve_decision(8, {'revision': 'new'})
    finally:
        if child.is_alive():
            child.kill()
            child.join(15)


def test_actual_sqlite_full_rolls_back_and_preserves_reservation(tmp_path):
    a = authority(tmp_path / 'authority')
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    small = replace(policy(), max_db_pages=5, max_envelope_bytes=50000)
    s = ControlIntentStore.bootstrap(a, small)
    with s.claim(campaign, 'tracking'):
        reserved = s.reserve_decision(1, {'revision': 'sealed'})
        with pytest.raises(StorageBlocked) as caught:
            s.stage('decision', 1, {'large': 'x' * 30000}, evidence_identity={'revision': 'sealed'})
        assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
        assert caught.value.__cause__.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert s.pending() == (reserved,)
    assert ControlIntentStore(a).pending() == (reserved,)


def test_reopen_pressure_does_not_attempt_sqlite_recovery(store, monkeypatch):
    s, _ = store
    s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=0, f_frsize=1, f_favail=100)
    def forbidden(*args, **kwargs):
        pytest.fail('SQLite must not open before capacity admission')
    monkeypatch.setattr(sqlite3, 'connect', forbidden)
    with pytest.raises(StorageBlocked):
        ControlIntentStore(s.authority)
    with pytest.raises(StorageBlocked):
        ControlIntentStore.bootstrap(s.authority, policy())


def test_wal_outbox_committed_readback(store):
    s, campaign = store
    conn = sqlite3.connect(campaign / 'trace-outbox.sqlite')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE traces(role TEXT, number INTEGER, payload TEXT, tracking_uri TEXT)')
    conn.commit()
    try:
        with s.claim(campaign, 'tracking'):
            raw = s.stage('execution', 1, {'evidence': 'WAL'})
            conn.execute('INSERT INTO traces VALUES(?,?,?,?)', ('execution', 1, '{"evidence":"WAL"}', 'tracking'))
            with pytest.raises(ValueError):
                s.acknowledge('execution', 1)
            assert s.pending() == (raw,)
            conn.commit()
            s.acknowledge('execution', 1)
            assert s.pending() == ()
    finally:
        conn.close()


def test_wal_acknowledgement_admits_missing_sidecars_and_releases_normal_lease(store):
    from unittest.mock import patch
    s,campaign=store
    outbox=campaign/'trace-outbox.sqlite'
    with sqlite3.connect(outbox) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE traces(role TEXT,number INTEGER,payload TEXT,tracking_uri TEXT)')
        conn.execute('INSERT INTO traces VALUES(?,?,?,?)',('execution',1,'{"exact":true}','tracking'))
    conn.close()
    assert not Path(str(outbox)+'-wal').exists()
    assert not Path(str(outbox)+'-shm').exists()
    with s.claim(campaign,'tracking'):
        original=s.stage('execution',1,{'exact':True})
        with patch.object(s.authority,'reserve',side_effect=StorageBlocked('blocked normal readback')):
            with pytest.raises(StorageBlocked):s.acknowledge('execution',1)
        assert s.pending()==(original,)
        assert not Path(str(outbox)+'-shm').exists()
        with patch.object(s.authority,'reserve',wraps=s.authority.reserve) as reserve:
            s.acknowledge('execution',1)
            assert reserve.call_args.kwargs['stage']=='control_outbox_readback'
        assert s.pending()==()
    assert s.authority.status()['reserved_bytes']==0


def test_missing_journal_scratch_requires_inode_capacity_and_keeps_state(store):
    s, campaign = store
    with s.claim(campaign, 'tracking'):
        raw = s.stage('execution', 1, {'retained': True})
    (s.root / 'intents.sqlite-journal').unlink()
    s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=64 * 1024**2, f_frsize=1, f_favail=4)
    with pytest.raises(StorageBlocked):
        ControlIntentStore(s.authority)
    assert not (s.root / 'intents.sqlite-journal').exists()
    s.authority._statvfs = lambda _: SimpleNamespace(f_bavail=64 * 1024**2, f_frsize=1, f_favail=5)
    reopened = ControlIntentStore(s.authority)
    assert reopened.pending() == (raw,)
    assert {p.name for p in reopened.root.iterdir()} == reopened._names


def test_actual_peak_pages_and_journal_measured_inside_transaction(store, monkeypatch):
    s, campaign = store
    measurements = []
    original = s._health
    def measured():
        usage = original()
        measurements.append({p.name: p.stat().st_blocks * 512 for p in s.root.iterdir()})
        return usage
    monkeypatch.setattr(s, '_health', measured)
    with s.claim(campaign, 'tracking'):
        s.reserve_decision(1, {'revision': 'r'})
        s.stage('decision', 1, {'a': 'd' * 6000}, evidence_identity={'revision': 'r'})
        s.stage('execution', 1, {'a': 'e' * 6000})
    assert max(m['intents.sqlite'] for m in measurements) <= s.policy.page_size * s.policy.max_db_pages
    assert 0 < max(m.get('intents.sqlite-journal', 0) for m in measurements) <= s.policy.max_journal_bytes
    assert all(sum(m.values()) + s.root.stat().st_blocks * 512 <= s.policy.allowance_bytes for m in measurements)


def test_payload_hash_conflict_is_corruption_not_empty(store):
    s, campaign = store
    with s.claim(campaign, 'tracking'):
        raw = s.stage('execution', 1, {'immutable': True})
    body = json.loads(raw)
    body['payload_hash'] = '0' * 64
    corrupted = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    with sqlite3.connect(s.db) as conn:
        conn.execute('UPDATE slots SET envelope=?', (corrupted,))
    with pytest.raises(ValueError, match='envelope'):
        ControlIntentStore(s.authority)
    with sqlite3.connect(s.db) as conn:
        assert conn.execute('SELECT envelope FROM slots').fetchone() == (corrupted,)


def test_database_file_ceiling_checked_before_sqlite_access(store):
    s, _ = store
    with s.db.open('r+b') as stream:
        stream.truncate(s.policy.page_size * (s.policy.max_db_pages + 1))
    with pytest.raises(StorageBlocked, match='database ceiling'):
        ControlIntentStore(s.authority)
