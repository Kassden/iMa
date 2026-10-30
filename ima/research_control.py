"""Bounded, explicit-bootstrap intent storage, independent of normal writers.

Envelopes are canonical UTF-8 JSON bytes (immutable, including on readback).
No provider execution or delivery is performed here. Bounds are cooperating
writer admission, not an OS quota; target-filesystem peak qualification is
still required, including SQLite recovery and filesystem allocation overhead.
"""
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat

from .research_storage import StorageBlocked


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class ControlPolicy:
    max_envelope_bytes: int
    page_size: int
    max_db_pages: int
    max_journal_bytes: int
    metadata_bytes: int
    allowance_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('Explicit positive integer control limits required')
        if self.page_size not in {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}:
            raise ValueError('Invalid SQLite page size')
        # Conservatively allow a maximum-size sector header per page record,
        # plus a final header. journal_size_limit alone does not bound peak.
        journal = self.max_db_pages * (self.page_size + 8 + 65536) + 65536
        if self.max_journal_bytes < journal:
            raise ValueError('Journal allowance below worst-case page transaction')
        if self.allowance_bytes < (self.page_size * self.max_db_pages
                                   + self.max_journal_bytes + self.metadata_bytes):
            raise ValueError('Complete control allowance is insufficient')


class ControlIntentStore:
    """One authority-wide two-slot store; claim before staging or acknowledging.

    reserve_decision(number, evidence_identity) precedes paid dispatch. Filling
    it requires stage(..., evidence_identity=the_same_metadata). pending()
    includes reserved envelopes, which must be reconciled rather than replayed
    as provider calls. All envelope-returning APIs return immutable JSON bytes.
    """
    _names = {'policy.json', 'owner.lock', 'intents.sqlite', 'intents.sqlite-journal'}

    def __init__(self, authority):
        self.authority = authority
        self.root = authority.root / 'control-intents'
        self.db = self.root / 'intents.sqlite'
        self._owner = None
        self._campaign = None
        self._destination = None
        with self._pressure(), authority._locked() as state:
            self._load()
            self._capacity(state, usage=self._health())
            with self._connect() as conn:
                self._validate(conn)

    @classmethod
    def bootstrap(cls, authority, policy):
        """Provision once in healthy storage; preserve every failed partial store."""
        if not isinstance(policy, ControlPolicy):
            raise TypeError('ControlPolicy required')
        obj = cls.__new__(cls)
        obj.authority = authority
        obj.root = authority.root / 'control-intents'
        obj.db = obj.root / 'intents.sqlite'
        obj.policy = policy
        obj._owner = obj._campaign = obj._destination = None
        obj._check_policy()
        with obj._pressure(), authority._locked() as state:
            if obj.root.exists() or obj.root.is_symlink():
                obj._load()
                if obj.policy != policy:
                    raise ValueError('Control policy changed')
                obj._capacity(state, usage=obj._health(), bootstrap=True)
                with obj._connect() as conn:
                    obj._validate(conn)
                return obj
            obj._capacity(state, usage=0, bootstrap=True)
            obj.root.mkdir(mode=0o700)
            for name in ('owner.lock', 'intents.sqlite', 'intents.sqlite-journal'):
                fd = os.open(obj.root / name, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            identity = obj._identity()
            metadata = {'schema': 1, 'policy': asdict(policy), 'identity': identity}
            encoded = _json(metadata)
            if len(encoded) > policy.metadata_bytes:
                raise ValueError('Policy metadata exceeds allowance')
            obj._metadata = metadata
            with obj._connect(bootstrap=True) as conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), body BLOB NOT NULL)')
                conn.execute('CREATE TABLE slots (slot TEXT PRIMARY KEY CHECK(slot IN (\'decision\',\'operational\')), envelope BLOB NOT NULL)')
                conn.execute('INSERT INTO metadata VALUES(1, ?)', (encoded,))
                conn.commit()
            with (obj.root / 'policy.json').open('xb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            for directory in (obj.root, authority.root):
                fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            obj._health()
        return obj

    @contextmanager
    def _pressure(self):
        try:
            yield
        except (OSError, sqlite3.Error) as exc:
            if ((isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT})
                    or (getattr(exc, 'sqlite_errorcode', 0) & 255) == sqlite3.SQLITE_FULL):
                raise StorageBlocked('Control storage pressure', {'cause': str(exc)}) from exc
            raise

    def _check_policy(self):
        if self.policy.allowance_bytes >= self.authority.headroom_bytes:
            raise ValueError('Control allowance must be strictly below normal headroom')

    def _safe(self, path, *, directory=False):
        if path.resolve() != path or '..' in path.parts:
            raise ValueError('Noncanonical control path')
        info = path.lstat()
        if (info.st_uid != os.getuid() or info.st_dev != self.authority.root.stat().st_dev
                or stat.S_ISLNK(info.st_mode) or info.st_mode & 0o022
                or (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1))
                or (directory and not stat.S_ISDIR(info.st_mode))):
            raise ValueError('Unsafe control owner, device, mode or link')
        return info

    def _identity(self):
        result = {'root': str(self.authority.root), 'uid': os.getuid(),
                  'headroom_bytes': self.authority.headroom_bytes,
                  'headroom_inodes': self.authority.headroom_inodes}
        for key, path in (('authority', self.authority.root), ('directory', self.root),
                          ('database', self.db), ('owner', self.root / 'owner.lock')):
            info = self._safe(path, directory=key in {'authority', 'directory'})
            result[key] = [info.st_dev, info.st_ino]
        return result

    def _load(self):
        self._safe(self.root, directory=True)
        path = self.root / 'policy.json'
        info = self._safe(path)
        if info.st_size > 65536:
            raise ValueError('Invalid policy metadata size')
        raw = path.read_bytes()
        metadata = json.loads(raw)
        if set(metadata) != {'schema', 'policy', 'identity'} or metadata['schema'] != 1:
            raise ValueError('Unknown control schema')
        policy = ControlPolicy(**metadata['policy'])
        if hasattr(self, '_metadata') and self._metadata != metadata:
            raise ValueError('Control policy or identity changed after opening')
        self.policy = policy
        self._check_policy()
        if raw != _json(metadata) or metadata['identity'] != self._identity():
            raise ValueError('Control identity or policy metadata changed')
        self._metadata = metadata
        self._health()

    def _health(self):
        self._safe(self.root, directory=True)
        names = {p.name for p in self.root.iterdir()}
        # SQLite can delete a hot rollback journal while recovering, before
        # journal_mode=PERSIST is applied. Only this scratch file is optional.
        if not self._names - {'intents.sqlite-journal'} <= names <= self._names:
            raise ValueError('Missing or unexpected control files')
        usage = self.root.stat().st_blocks * 512
        metadata_usage = usage
        for name in names:
            path = self.root / name
            info = self._safe(path)
            allocated = info.st_blocks * 512
            usage += allocated
            if name == 'intents.sqlite':
                if max(info.st_size, allocated) > self.policy.page_size * self.policy.max_db_pages:
                    raise StorageBlocked('Control database ceiling exceeded')
            elif name == 'intents.sqlite-journal':
                if max(info.st_size, allocated) > self.policy.max_journal_bytes:
                    raise StorageBlocked('Control journal ceiling exceeded')
            else:
                metadata_usage += max(info.st_size, allocated)
        if metadata_usage > self.policy.metadata_bytes or usage > self.policy.allowance_bytes:
            raise StorageBlocked('Control allocated allowance exceeded')
        return usage

    def allocated_bytes(self):
        """Actual allocated DB, journal, metadata, lock and directory blocks."""
        with self.authority._locked():
            self._load()
            return self._health()

    def _capacity(self, state, *, usage, bootstrap=False):
        # Reconcile this locked snapshot only; do not recursively write JSON
        # leases from the emergency control path.
        self.authority._reconcile(state)
        snap = self.authority._snapshot(state)
        growth = self.policy.allowance_bytes - usage
        floor = self.authority.headroom_bytes
        if not bootstrap:
            floor -= self.policy.allowance_bytes
        # All normal reservations remain outstanding; control creates no JSON lease.
        scratch_inode = int(not (self.root / 'intents.sqlite-journal').exists())
        if (snap['available_bytes'] - snap['reserved_bytes'] - growth < floor
                or snap['available_inodes'] - snap['reserved_inodes'] <
                self.authority.headroom_inodes + (5 if bootstrap else scratch_inode)):
            raise StorageBlocked('Insufficient complete control capacity', snap)

    @contextmanager
    def _connect(self, *, bootstrap=False):
        with self._pressure(), closing(sqlite3.connect(
                self.db.as_uri() + '?mode=rw', uri=True, timeout=0, isolation_level=None)) as conn:
            if bootstrap:
                conn.execute(f'PRAGMA page_size={self.policy.page_size}')
            checks = [('journal_mode=PERSIST', 'persist'), ('synchronous=FULL', None),
                      ('synchronous', 2), ('temp_store=MEMORY', None), ('temp_store', 2),
                      ('page_size', self.policy.page_size),
                      (f'max_page_count={self.policy.max_db_pages}', self.policy.max_db_pages),
                      (f'journal_size_limit={self.policy.max_journal_bytes}', self.policy.max_journal_bytes)]
            for pragma, expected in checks:
                row = conn.execute('PRAGMA ' + pragma).fetchone()
                if expected is not None and (row is None or row[0] != expected):
                    raise ValueError('SQLite refused control PRAGMA: ' + pragma)
            if not bootstrap and not (self.root / 'intents.sqlite-journal').exists():
                # Recreate only empty scratch after validating recovered state.
                # Policy/database/owner identity are never initialized here.
                self._validate(conn)
                fd = os.open(self.root / 'intents.sqlite-journal',
                             os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            try:
                yield conn
            finally:
                if conn.in_transaction:
                    conn.rollback()

    def _validate(self, conn):
        if conn.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('Corrupt control database')
        if conn.execute('SELECT id,body FROM metadata').fetchall() != [(1, _json(self._metadata))]:
            raise ValueError('Control database metadata changed')
        rows = conn.execute('SELECT slot,envelope FROM slots ORDER BY slot').fetchall()
        if len(rows) > 2:
            raise ValueError('Too many control slots')
        for slot, raw in rows:
            body = json.loads(raw)
            fields = {'schema', 'policy', 'campaign', 'destination', 'role', 'sequence',
                      'state', 'payload', 'payload_hash', 'signature', 'evidence_identity'}
            if (set(body) != fields or type(raw) is not bytes or raw != _json(body)
                    or len(raw) > self.policy.max_envelope_bytes
                    or slot != self._slot(body['role']) or body['schema'] != 1
                    or body['policy'] != hashlib.sha256(_json(self._metadata)).hexdigest()
                    or body['state'] not in {'reserved', 'complete'}
                    or type(body['sequence']) is not int or body['sequence'] < 0
                    or body['payload_hash'] != hashlib.sha256(_json(body['payload'])).hexdigest()
                    or not isinstance(body['campaign'], str) or not Path(body['campaign']).is_absolute()
                    or not isinstance(body['destination'], str) or not body['destination']
                    or (body['state'] == 'complete' and not isinstance(body['payload'], dict))
                    or (body['role'] == 'decision' and
                        (not isinstance(body['evidence_identity'], dict) or not body['evidence_identity']))
                    or (body['state'] == 'reserved' and
                        (body['role'] != 'decision' or body['payload'] is not None or body['signature'] is not None))):
                raise ValueError('Invalid stored control envelope')
        return tuple(raw for _, raw in rows)

    @staticmethod
    def _slot(role):
        if role not in {'decision', 'execution', 'dataset', 'betting'}:
            raise ValueError('Unknown control role')
        return 'decision' if role == 'decision' else 'operational'

    @contextmanager
    def claim(self, campaign: Path, tracking_uri: str):
        """Nonblocking ownership; contention raises StorageBlocked for caller wait."""
        if self._owner is not None:
            raise ValueError('Already claimed')
        campaign = Path(campaign).absolute()
        self._safe(campaign, directory=True)
        if not isinstance(tracking_uri, str) or not tracking_uri:
            raise ValueError('Explicit destination required')
        self._load()
        fd = os.open(self.root / 'owner.lock', os.O_RDWR | os.O_NOFOLLOW)
        acquired = False
        try:
            if [os.fstat(fd).st_dev, os.fstat(fd).st_ino] != self._metadata['identity']['owner']:
                raise ValueError('Owner lock replaced')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StorageBlocked('Another controller owns control intents') from exc
            acquired = True
            self._owner = (fd, os.getpid())
            self._campaign, self._destination = str(campaign), tracking_uri
            for raw in self.pending():
                body = json.loads(raw)
                if body['campaign'] != self._campaign or body['destination'] != tracking_uri:
                    raise StorageBlocked('Pending intents belong to another campaign or destination')
            yield self
        finally:
            if acquired:
                self._owner = self._campaign = self._destination = None
            os.close(fd)

    def _claimed(self):
        if self._owner is None or self._owner[1] != os.getpid():
            raise ValueError('Own-process controller claim required')

    def pending(self):
        """Read exact immutable envelopes, including unresolved reservations."""
        with self._pressure(), self.authority._locked() as state:
            self._load()
            self._capacity(state, usage=self._health())
            with self._connect() as conn:
                return self._validate(conn)

    def _envelope(self, role, number, payload, signature, evidence_identity, state):
        self._claimed()
        self._slot(role)
        if type(number) is not int or not 0 <= number <= 2**63 - 1:
            raise ValueError('Invalid control sequence')
        # Freeze the supplied JSON once; never revisit mutable evidence sources.
        frozen = json.loads(_json({'payload': payload, 'signature': signature,
                                  'evidence_identity': evidence_identity}))
        body = {'schema': 1, 'policy': hashlib.sha256(_json(self._metadata)).hexdigest(),
                'campaign': self._campaign, 'destination': self._destination,
                'role': role, 'sequence': number, 'state': state, **frozen,
                'payload_hash': hashlib.sha256(_json(frozen['payload'])).hexdigest()}
        raw = _json(body)
        if len(raw) > self.policy.max_envelope_bytes:
            raise StorageBlocked('Complete control envelope exceeds policy')
        return raw

    def reserve_decision(self, number, evidence_identity):
        if not isinstance(evidence_identity, dict) or not evidence_identity:
            raise ValueError('Explicit evidence identity metadata required')
        raw = self._envelope('decision', number, None, None, evidence_identity, 'reserved')
        return self._stage(raw)

    def stage(self, role, number, payload, signature=None, *, evidence_identity=None):
        if not isinstance(payload, dict):
            raise ValueError('Normalized payload must be a JSON object')
        if role == 'decision' and (not isinstance(evidence_identity, dict) or not evidence_identity):
            raise ValueError('Matching decision evidence identity required')
        raw = self._envelope(role, number, payload, signature, evidence_identity, 'complete')
        return self._stage(raw)

    def _stage(self, raw):
        body = json.loads(raw)
        slot = self._slot(body['role'])
        with self._pressure(), self.authority._locked() as state:
            self._load()
            self._capacity(state, usage=self._health())
            with self._connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                self._validate(conn)
                row = conn.execute('SELECT envelope FROM slots WHERE slot=?', (slot,)).fetchone()
                if row and row[0] == raw:
                    return raw
                if row:
                    prior = json.loads(row[0])
                    identity = ('campaign', 'destination', 'role', 'sequence', 'evidence_identity', 'policy')
                    if not (prior['state'] == 'reserved' and body['state'] == 'complete'
                            and all(prior[k] == body[k] for k in identity)):
                        raise StorageBlocked('Occupied control slot conflicts with intent')
                elif body['role'] == 'decision' and body['state'] != 'reserved':
                    raise ValueError('Reserve decision before dispatch and staging')
                conn.execute('INSERT OR REPLACE INTO slots VALUES(?,?)', (slot, raw))
                self._health()
                conn.commit()
            self._health()
        return raw

    def acknowledge(self, role, number):
        """Clear only after exact committed normal-outbox readback, never delivery.

        Normal readback admission is outside the authority lock. This covers
        WAL sidecar creation even if another connection closes concurrently;
        emergency control transactions never allocate these normal leases.
        """
        self._claimed()
        slot = self._slot(role)
        if type(number) is not int or not 0 <= number <= 2**63 - 1:
            raise ValueError('Invalid control sequence')
        expected = next((raw for raw in self.pending()
                         if json.loads(raw)['role'] == role and json.loads(raw)['sequence'] == number), None)
        if expected is None or json.loads(expected)['state'] != 'complete':
            raise ValueError('No matching complete intent')
        body = json.loads(expected)
        outbox = Path(self._campaign) / 'trace-outbox.sqlite'
        self._safe(outbox)
        wal_bytes=0
        for suffix in ('-wal','-shm','-journal'):
            sidecar=Path(str(outbox)+suffix)
            if sidecar.exists() or sidecar.is_symlink():
                info=self._safe(sidecar)
                if suffix=='-wal':wal_bytes=info.st_size
        with self._pressure(), self.authority.admitted(
                bytes_required=1024**2+wal_bytes,inodes_required=2,
                stage='control_outbox_readback',output=outbox.parent):
            with closing(sqlite3.connect(outbox.as_uri() + '?mode=ro', uri=True, timeout=0)) as conn:
                row = conn.execute('SELECT tracking_uri,payload FROM traces WHERE role=? AND number=?',
                                   (role, number)).fetchone()
                if row is None or row[0] != body['destination'] or _json(json.loads(row[1])) != _json(body['payload']):
                    raise ValueError('Committed outbox destination or full payload does not match')
        with self._pressure(), self.authority._locked() as state:
            self._load()
            self._capacity(state, usage=self._health())
            with self._connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                self._validate(conn)
                row = conn.execute('SELECT envelope FROM slots WHERE slot=?', (slot,)).fetchone()
                if row != (expected,):
                    raise ValueError('Control intent changed during outbox verification')
                conn.execute('DELETE FROM slots WHERE slot=?', (slot,))
                self._health()
                conn.commit()
            self._health()
