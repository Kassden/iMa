import argparse
import errno
import gzip
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
from filelock import FileLock,Timeout

import pytest
from scrapy import Request
from scrapy.http import HtmlResponse

import ima.research_storage as storage
from scrapper import official_corpus
from scrapper.official_corpus import CorpusSpider,ROOT
from scripts import collect_official_corpus as collector
from ima.research_storage import StorageBlocked


def authority_at(root,monkeypatch):
    authority=storage.StorageAuthority(root/'control',headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    return authority


def response():
    url=ROOT+'horse?HorseId=HK_2020_D001'
    body=('<html><a href="'+ROOT+'horse?HorseId=HK_2020_D002">Horse</a></html>').encode()
    return HtmlResponse(url,body=body,encoding='utf-8',request=Request(url),
                        headers={'Content-Type':'text/html'})


def test_capture_reservations_end_before_the_callback_yields_requests(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch)
    spider=CorpusSpider(seeds=[],output=tmp_path/'corpus')
    inventory_lease=authority.status()['reserved_bytes']
    requests=spider.parse(response())
    first=next(requests)
    assert isinstance(first,Request)
    assert authority.status()['reserved_bytes']>=inventory_lease
    assert storage._OUTPUT_BUDGET.get() is None
    requests.close()
    spider.closed('fixture')
    assert authority.status()['reserved_bytes']==0
    report=json.loads((spider.output/'closed.json').read_text())
    assert report['reason']=='fixture' and report['pages']==1


def test_denied_capture_closes_as_storage_pressure_without_raw_publication(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch)
    spider=CorpusSpider(seeds=[],output=tmp_path/'corpus')
    inventory_lease=authority.status()['reserved_bytes']
    spider.crawler=SimpleNamespace(engine=Mock())
    existing=authority._statvfs(authority.root)
    authority._statvfs=lambda root:SimpleNamespace(f_frsize=existing.f_frsize,f_blocks=existing.f_blocks,
        f_bavail=0,f_favail=existing.f_favail)
    assert list(spider.parse(response()))==[]
    assert spider.storage_error['failure_kind']=='infrastructure_storage'
    spider.crawler.engine.close_spider.assert_called_once_with(spider,reason='blocked_storage')
    assert not (spider.output/'raw').exists()
    assert authority.status()['reserved_bytes']==inventory_lease
    with pytest.raises(StorageBlocked):spider.closed('blocked_storage')
    assert authority.status()['reserved_bytes']==0


def test_inventory_capacity_denial_precedes_output_and_database_creation(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch)
    existing=authority._statvfs(authority.root)
    authority._statvfs=lambda root:SimpleNamespace(f_frsize=existing.f_frsize,f_blocks=existing.f_blocks,
        f_bavail=0,f_favail=existing.f_favail)
    output=tmp_path/'not-created'/'corpus'
    with pytest.raises(StorageBlocked):CorpusSpider(seeds=[],output=output)
    assert not output.exists()
    assert not (output/'inventory.sqlite').exists()
    assert authority.status()['reserved_bytes']==0


def test_schema_failure_closes_native_sqlite_lease(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch)
    native=official_corpus.native_sqlite_connection

    class BrokenSchemaConnection:
        def __init__(self,connection):self.connection=connection
        def __enter__(self):return self
        def __exit__(self,kind,value,traceback):
            if kind is not None and self.connection.in_transaction:self.connection.rollback()
            return False
        def execute(self,*args,**kwargs):raise sqlite3.OperationalError('schema fixture failure')
        def close(self):self.connection.close()

    monkeypatch.setattr(official_corpus,'native_sqlite_connection',
        lambda *args,**kwargs:BrokenSchemaConnection(native(*args,**kwargs)))
    with pytest.raises(sqlite3.OperationalError,match='schema fixture'):
        CorpusSpider(seeds=[],output=tmp_path/'corpus')
    assert authority.status()['reserved_bytes']==0


def test_capture_reopens_with_index_and_complete_raw_blob(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch)
    spider=CorpusSpider(seeds=[],output=tmp_path/'corpus')
    page=response()
    list(spider.parse(page))
    spider.db.close()
    with sqlite3.connect(spider.output/'inventory.sqlite') as db:
        row=db.execute('SELECT status,body_hash,metadata_json FROM pages WHERE url=?',(page.url,)).fetchone()
        assert row and row[0]=='fetched_unparsed'
        metadata=json.loads(row[2])
        raw=spider.output/metadata['raw_path'].removeprefix(str(spider.output)+'/')
        assert gzip.decompress(raw.read_bytes())==page.body
        assert db.execute('SELECT count(*) FROM captures WHERE source_url=?',(page.url,)).fetchone()[0]==1
    reopened=CorpusSpider(seeds=[],output=spider.output)
    assert reopened.db.execute('SELECT count(*) FROM pages WHERE url=?',(page.url,)).fetchone()[0]==1
    reopened.closed('fixture')


def test_partial_raw_enospc_is_never_published_and_cleanup_error_is_not_primary(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch)
    spider=CorpusSpider(seeds=[],output=tmp_path/'corpus')
    page=response()
    digest=__import__('hashlib').sha256(page.body).hexdigest()
    blob=spider.output/'raw'/f'{digest}.html.gz'
    temporary=blob.with_suffix('.tmp')

    class FailingStream:
        def __init__(self,stream):self.stream=stream
        def write(self,data):
            self.stream.write(bytes(data[:4]))
            raise OSError(errno.ENOSPC,'fixture disk full')
        def __getattr__(self,name):return getattr(self.stream,name)

    @contextmanager
    def partial_writer(path,mode='wb'):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:
            yield FailingStream(stream)

    monkeypatch.setattr('scrapper.official_corpus.reserved_output_file',partial_writer)
    original_unlink=__import__('pathlib').Path.unlink
    def fail_temp_cleanup(path,*args,**kwargs):
        if path==temporary:raise OSError(errno.EACCES,'fixture cleanup denied')
        return original_unlink(path,*args,**kwargs)
    monkeypatch.setattr('pathlib.Path.unlink',fail_temp_cleanup)
    with pytest.raises(StorageBlocked) as failure:spider._capture_response(page)
    assert not blob.exists()
    assert temporary.exists() and temporary.stat().st_size>0
    cause=failure.value.__cause__
    notes=' '.join(getattr(cause,'__notes__',[]))
    assert cause.errno == errno.ENOSPC
    assert 'cleanup also failed' in notes
    assert cause.errno==errno.ENOSPC
    spider.db.close()


def test_sqlite_full_rolls_back_inventory_without_publishing_status(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch)
    spider=CorpusSpider(seeds=[],output=tmp_path/'corpus')
    spider.db._lease_growth_bytes=0
    record={'source_url':ROOT+'horse?HorseId=HK_2020_D001','status':'captured',
            'family':'horse','fetched_at':'2026-10-27T00:00:00+00:00','padding':'x'*100_000}
    with pytest.raises(StorageBlocked) as failure:spider.store(record)
    assert failure.value.diagnostic['sqlite_errorcode']&255==sqlite3.SQLITE_FULL
    assert spider.db.execute('SELECT count(*) FROM pages').fetchone()[0]==0
    assert spider.db.execute('SELECT count(*) FROM captures').fetchone()[0]==0
    assert not (spider.output/'status.json').exists()
    spider.db.close()


def test_existing_wal_database_is_rejected_without_repair(tmp_path,monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    output=tmp_path/'corpus';output.mkdir()
    path=output/'inventory.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE protected(value TEXT)')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute("INSERT INTO protected VALUES ('keep')")
    before=path.read_bytes()
    with pytest.raises(RuntimeError,match='WAL'):
        CorpusSpider(seeds=[],output=output)
    assert path.read_bytes()==before
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT value FROM protected').fetchone()==('keep',)


def test_symlink_output_alias_cannot_touch_protected_inventory(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch)
    protected=tmp_path/'protected';protected.mkdir()
    marker=protected/'keep.txt';marker.write_text('preserve')
    alias=tmp_path/'alias';alias.symlink_to(protected,target_is_directory=True)
    with pytest.raises(ValueError):CorpusSpider(seeds=[],output=alias)
    assert marker.read_text()=='preserve'
    assert not (protected/'inventory.sqlite').exists()


def test_blocked_segment_cannot_reuse_an_older_success_receipt(tmp_path,monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    output=tmp_path/'corpus';output.mkdir()
    previous={'reason':'closespider_pagecount','pages':123}
    (output/'closed.json').write_text(json.dumps(previous))
    seed=tmp_path/'seeds.json';seed.write_text(json.dumps([ROOT+'horse?HorseId=HK_2020_D001']))
    args=SimpleNamespace(output=output,seed_file=seed,horse_seed_file=None,additional_seed_file=[],
        limit=1,seconds=1,retry_scope_errors=False)
    process=Mock()
    process.create_crawler.return_value=SimpleNamespace(spider=SimpleNamespace(storage_error={'retryable':True}))
    monkeypatch.setattr(collector,'CrawlerProcess',lambda settings:process)
    with pytest.raises(SystemExit) as failure:
        collector._collect_segment(args,json.loads(seed.read_text()),argparse.ArgumentParser())
    assert failure.value.code==2
    assert not (output/'closed.json').exists()
    archives=list(output.glob('closed-*.previous.json'))
    assert len(archives)==1 and json.loads(archives[0].read_text())==previous


def test_duplicate_segment_cannot_mutate_another_collectors_receipt(tmp_path,monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    output=tmp_path/'corpus';output.mkdir()
    closed=output/'closed.json';closed.write_text('{"reason":"previous"}')
    args=SimpleNamespace(output=output)
    process=Mock();monkeypatch.setattr(collector,'CrawlerProcess',lambda settings:process)
    with FileLock(output/'collector.lock',timeout=0):
        with pytest.raises(Timeout):collector._collect_segment(args,[],argparse.ArgumentParser())
    assert closed.read_text()=='{"reason":"previous"}'
    process.start.assert_not_called()


def test_frozen_collector_inputs_are_reused_but_never_repaired_in_place(tmp_path):
    target=tmp_path/'source.py'
    collector._publish_release_file(target,b'original')
    before=target.stat()
    collector._publish_release_file(target,b'original')
    assert target.stat().st_ino==before.st_ino and target.stat().st_mtime_ns==before.st_mtime_ns
    with pytest.raises(ValueError,match='changed'):
        collector._publish_release_file(target,b'replacement')
    assert target.read_bytes()==b'original'
    assert not list(tmp_path.glob('.collector-*.tmp'))
