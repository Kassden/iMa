import sqlite3
from pathlib import Path
import pytest
from ima.research_storage import capture_sqlite


def test_online_capture_includes_committed_wal_and_keeps_source(tmp_path):
    source=tmp_path/'source.sqlite'
    with sqlite3.connect(source) as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('CREATE TABLE attempts(value)')
        c.execute('INSERT INTO attempts VALUES (42)'); c.commit()
        assert Path(str(source)+'-wal').exists()
        receipt=capture_sqlite(source,tmp_path/'healthy/copy.sqlite')
        assert receipt['method']=='sqlite_online_backup'
        assert receipt['application_counts']['attempts']==1
        with sqlite3.connect(receipt['destination']) as copy:
            assert copy.execute('SELECT value FROM attempts').fetchone()==(42,)
        assert c.execute('SELECT value FROM attempts').fetchone()==(42,)


def test_capture_never_overwrites_existing_destination(tmp_path):
    source=tmp_path/'source.sqlite'
    with sqlite3.connect(source) as c: c.execute('CREATE TABLE keep(value)')
    destination=tmp_path/'existing'; destination.write_bytes(b'protected')
    with pytest.raises(ValueError): capture_sqlite(source,destination)
    assert destination.read_bytes()==b'protected'


@pytest.mark.parametrize('kind',['dangling','ancestor','sidecar'])
def test_capture_rejects_destination_symlink_escapes(tmp_path,kind):
    source=tmp_path/'source.sqlite'
    with sqlite3.connect(source) as c: c.execute('CREATE TABLE keep(value)')
    outside=tmp_path/'outside'; outside.mkdir()
    healthy=tmp_path/'healthy'; healthy.mkdir()
    destination=healthy/'copy.sqlite'
    if kind=='dangling': destination.symlink_to(outside/'unexpected.sqlite')
    elif kind=='ancestor':
        (healthy/'escape').symlink_to(outside,target_is_directory=True)
        destination=healthy/'escape/copy.sqlite'
    else: Path(str(destination)+'-wal').symlink_to(outside/'unexpected-wal')
    with pytest.raises(ValueError): capture_sqlite(source,destination)
    assert not list(outside.iterdir())


def test_truncated_shm_preserved_before_any_source_sqlite_access(tmp_path):
    source=tmp_path/'source.sqlite'
    with sqlite3.connect(source) as c:
        c.execute('CREATE TABLE decisions(value)')
        c.execute('INSERT INTO decisions VALUES (42)'); c.commit()
    shm=Path(str(source)+'-shm'); shm.write_bytes(b'bad')
    wal=Path(str(source)+'-wal'); wal.write_bytes(b'')
    before={p:p.read_bytes() for p in (source,shm,wal)}
    receipt=capture_sqlite(source,tmp_path/'healthy/copy.sqlite',allow_quiescent_capture=True)
    assert receipt['application_counts']['decisions']==1
    assert 'quiescent' in receipt['method']
    assert all(p.read_bytes()==data for p,data in before.items())


def test_failed_capture_does_not_publish_empty_backup_and_can_retry(tmp_path):
    source=tmp_path/'source.sqlite'
    with sqlite3.connect(source) as c: c.execute('CREATE TABLE decisions(value)')
    Path(str(source)+'-shm').write_bytes(b'bad')
    destination=tmp_path/'healthy/copy.sqlite'
    with pytest.raises(sqlite3.OperationalError): capture_sqlite(source,destination)
    assert not destination.exists()
    assert capture_sqlite(source,destination,allow_quiescent_capture=True)['integrity']==['ok']
