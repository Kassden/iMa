import json
import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import scrapy
import queuelib.queue as native_queue
from scrapy import Request, Spider
from scrapy.squeues import PickleFifoDiskQueue

import ima.research_storage as storage
import scrapper.research_storage_io as native_io
from ima.research_storage import StorageAuthority, admitted_output
from scripts import collect_official_corpus as collector
from scrapper.research_storage_io import (
    AdmittedFifoDiskQueue,
    AdmittedPickleFifoDiskQueue,
    AdmittedRFPDupeFilter,
    AdmittedScheduler,
    AdmittedSpiderState,
    BoundedRotatingFileHandler,
    atomic_native_bytes,
    prepare_jobdir,
)


def authority_at(tmp_path, monkeypatch):
    authority = StorageAuthority(
        tmp_path / "control", headroom_bytes=0, headroom_inodes=0
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    return authority


def crawler():
    return SimpleNamespace(spider=Spider("storage-io-test"))


@pytest.mark.parametrize('release', [False, True])
def test_exclusive_temp_collision_retains_foreign_inode(tmp_path, monkeypatch, release):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / 'corpus'
    output.mkdir()
    module = collector if release else native_io
    monkeypatch.setattr(module, 'uuid4', lambda: SimpleNamespace(hex='fixed-collision'))
    target = output / 'state.json'
    temporary = output / ('.collector-fixed-collision.tmp' if release else
                          f'.state.json.{os.getpid()}.fixed-collision.tmp')
    temporary.write_bytes(b'pre-existing file must survive')
    before = temporary.stat().st_ino
    with admitted_output(output, estimated_bytes=1024**2, stage='collision_fixture', inodes=8):
        with pytest.raises(FileExistsError):
            (collector._publish_release_file if release else atomic_native_bytes)(target, b'new')
    assert temporary.read_bytes() == b'pre-existing file must survive'
    assert temporary.stat().st_ino == before
    assert not target.exists()


@pytest.mark.parametrize('release', [False, True])
def test_replaced_temp_is_neither_published_nor_deleted(tmp_path, monkeypatch, release):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / 'corpus'
    output.mkdir()
    target = output / 'state.json'
    write = storage.ReservedFile.write
    replacement = []

    def replace_after_write(stream, body):
        result = write(stream, body)
        for candidate in output.glob('.*.tmp'):
            if os.path.samestat(os.fstat(stream.fileno()), candidate.stat()):
                candidate.unlink()
                candidate.write_bytes(b'replacement must survive')
                replacement.append(candidate)
                break
        return result

    with admitted_output(output, estimated_bytes=1024**2, stage='replacement_fixture', inodes=8):
        monkeypatch.setattr(storage.ReservedFile, 'write', replace_after_write)
        with pytest.raises(RuntimeError, match='staging identity changed'):
            (collector._publish_release_file if release else atomic_native_bytes)(target, b'new')
    assert not target.exists()
    assert len(replacement) == 1
    assert replacement[0].read_bytes() == b'replacement must survive'


def test_jobdir_byte_denial_precedes_native_directory_creation(tmp_path, monkeypatch):
    authority = authority_at(tmp_path, monkeypatch)
    output = tmp_path / 'corpus'
    output.mkdir()
    with admitted_output(output, estimated_bytes=0, stage='directory_fixture', inodes=0):
        authority._statvfs = lambda root: SimpleNamespace(
            f_frsize=4096, f_blocks=1000, f_bavail=0, f_favail=1000)
        with pytest.raises(storage.StorageBlocked):
            prepare_jobdir(output / 'job', output)
    assert not (output / 'job').exists()


def test_log_native_full_notifies_owner_and_rolls_back_partial_record(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / 'corpus'
    output.mkdir()
    target = output / 'crawl.log'
    target.write_bytes(b'accepted prior record\n')
    notified = []
    native_write = native_io._native_write

    def partial_full(descriptor, body):
        native_write(descriptor, body[:3])
        raise OSError(28, 'injected native log full')

    handler = BoundedRotatingFileHandler(target, on_storage_block=notified.append)
    with admitted_output(output, estimated_bytes=1024**2, stage='log_fixture', inodes=8):
        monkeypatch.setattr(native_io, '_native_write', partial_full)
        with pytest.raises(storage.StorageBlocked) as failure:
            handler.emit(logging.LogRecord('fixture', logging.INFO, '', 0, 'new record', (), None))
    handler.close()
    assert notified == [failure.value]
    assert target.read_bytes() == b'accepted prior record\n'


def test_fifo_queue_retains_native_pickle_format_and_reopens_with_scrapy(tmp_path, monkeypatch):
    authority = authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    prepare_jobdir(output / "job", output)
    queue_path = output / "job" / "requests.queue" / "-17"
    request = Request("https://racing.hkjc.com/en-us/local/information/horse?HorseId=HK_2020_D001")
    with admitted_output(output, estimated_bytes=1, stage="queue_fixture", inodes=1):
        queue = AdmittedPickleFifoDiskQueue(crawler(), str(queue_path))
        queue.push(request)
        queue.close()
    native = PickleFifoDiskQueue(crawler(), str(queue_path))
    restored = native.pop()
    assert restored.url == request.url
    assert native.pop() is None
    native.close()
    assert authority.status()["reserved_bytes"] == 0


def test_fifo_push_extends_before_native_write_and_rolls_back_partial_append(tmp_path, monkeypatch):
    authority = authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    prepare_jobdir(output / "job", output)
    queue_path = output / "job" / "requests.queue" / "0"
    extensions = []
    extend = authority.extend

    def record_extend(token, **kwargs):
        extensions.append(kwargs.copy())
        return extend(token, **kwargs)

    monkeypatch.setattr(authority, "extend", record_extend)
    with admitted_output(output, estimated_bytes=0, stage="queue_fixture", inodes=0):
        queue = AdmittedFifoDiskQueue(queue_path)
        data_path = queue_path / "q00000"
        original_write = os.write

        def forbidden_write(fd, data):
            raise AssertionError("native queue write ran before denied admission")

        current_statvfs = authority._statvfs
        authority._statvfs = lambda root: SimpleNamespace(
            f_frsize=4096, f_blocks=1000, f_bavail=0, f_favail=1000
        )
        monkeypatch.setattr(native_queue.os, "write", forbidden_write)
        with pytest.raises(Exception) as denial:
            queue.push(b"must not reach os.write")
        assert getattr(denial.value, "diagnostic", {}).get("failure_kind") == "infrastructure_storage"
        assert data_path.read_bytes() == b""
        authority._statvfs = current_statvfs

        def partial_then_full(fd, data):
            original_write(fd, data[:7])
            raise OSError(28, "fixture ENOSPC")

        monkeypatch.setattr(native_queue.os, "write", partial_then_full)
        with pytest.raises(Exception) as failure:
            queue.push(b"serialized request bytes")
        assert getattr(failure.value, "diagnostic", {}).get("failure_kind") == "infrastructure_storage"
        assert data_path.read_bytes() == b""
        assert queue.info["size"] == 0
        assert queue.info["head"] == [0, 0]
        assert extensions and any(item["extra_bytes"] > 0 for item in extensions)
        monkeypatch.setattr(native_queue.os, "write", original_write)
        queue.close()


def test_fifo_chunk_rollover_keeps_queuelib_file_names_and_head_state(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    prepare_jobdir(output / "job", output)
    queue_path = output / "job" / "requests.queue" / "0"
    request = Request("https://racing.hkjc.com/en-us/local/information/horse?HorseId=HK_2020_D001")
    with admitted_output(output, estimated_bytes=1, stage="queue_fixture", inodes=1):
        queue = AdmittedFifoDiskQueue(queue_path, chunksize=1)
        queue.push(pickle.dumps(request.to_dict(spider=crawler().spider), protocol=4))
        assert (queue_path / "q00000").exists()
        assert (queue_path / "q00001").exists()
        assert json.loads((queue_path / "info.json").read_text())["head"] == [1, 0]
        queue.close()
    native = PickleFifoDiskQueue(crawler(), str(queue_path))
    assert native.pop().url == request.url
    assert native.pop() is None
    native.close()


def test_dupefilter_appends_flushes_and_native_filter_reads_existing_format(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    prepare_jobdir(output / "job", output)
    request = Request("https://racing.hkjc.com/en-us/local/information/horse?HorseId=HK_2020_D001")
    with admitted_output(output, estimated_bytes=1, stage="dupe_fixture", inodes=1):
        filter_ = AdmittedRFPDupeFilter(str(output / "job"))
        assert filter_.request_seen(request) is False
        assert filter_.file.tell() == 22
        filter_.close("fixture")
    native = scrapy.dupefilters.RFPDupeFilter(str(output / "job"))
    assert native.request_seen(request) is True
    native.close("fixture")


def test_dupefilter_capacity_denial_happens_before_append(tmp_path, monkeypatch):
    authority = authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    prepare_jobdir(output / "job", output)
    request = Request("https://racing.hkjc.com/en-us/local/information/horse?HorseId=HK_2020_D001")
    with admitted_output(output, estimated_bytes=0, stage="dupe_fixture", inodes=1):
        filter_ = AdmittedRFPDupeFilter(str(output / "job"))
        before = filter_._dupefilter_path.stat().st_size
        prior = authority._statvfs
        authority._statvfs = lambda root: SimpleNamespace(
            f_frsize=4096, f_blocks=1000, f_bavail=0, f_favail=1000
        )
        with pytest.raises(Exception) as failure:
            filter_.request_seen(request)
        assert getattr(failure.value, "diagnostic", {}).get("failure_kind") == "infrastructure_storage"
        assert filter_._dupefilter_path.stat().st_size == before
        assert not filter_._fingerprints
        authority._statvfs = prior
        filter_.close("fixture")


def test_scheduler_rolls_back_seen_fingerprint_if_queue_admission_fails(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    prepare_jobdir(output / "job", output)
    request = Request("https://racing.hkjc.com/en-us/local/information/horse?HorseId=HK_2020_D001")
    with admitted_output(output, estimated_bytes=1024, stage="dupe_fixture", inodes=4):
        dupefilter = AdmittedRFPDupeFilter(str(output / "job"))
        scheduler = AdmittedScheduler.__new__(AdmittedScheduler)
        scheduler.df = dupefilter
        scheduler.spider = Spider("storage-io-test")
        scheduler.dqdir = None
        scheduler.dqs = None
        scheduler._dqpush = lambda request: (_ for _ in ()).throw(
            RuntimeError("queue admission rejected")
        )
        with pytest.raises(RuntimeError, match="queue admission rejected"):
            scheduler.enqueue_request(request)
        assert dupefilter.file.seek(0, os.SEEK_END) == 0
        assert not dupefilter._fingerprints
        dupefilter.close("fixture")


def test_scheduler_and_spider_metadata_publish_atomically_in_native_formats(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    jobdir = output / "job"
    prepare_jobdir(jobdir, output)
    state = [-30, 0, 8]
    scheduler = AdmittedScheduler.__new__(AdmittedScheduler)
    with admitted_output(output, estimated_bytes=0, stage="metadata_fixture", inodes=0):
        scheduler._write_dqs_state(jobdir / "requests.queue", state)
        AdmittedSpiderState(str(jobdir)).spider_closed(SimpleNamespace(state={"cursor": 9}))
    assert json.loads((jobdir / "requests.queue" / "active.json").read_text()) == state
    assert pickle.loads((jobdir / "spider.state").read_bytes()) == {"cursor": 9}
    assert not list(jobdir.rglob("*.tmp"))


def test_rotating_log_caps_records_and_retained_files(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    log_path = output / "crawl.log"
    handler = BoundedRotatingFileHandler(log_path, max_bytes=256, backup_count=2)
    handler.setFormatter(logging.Formatter("%(message)s"))
    with admitted_output(output, estimated_bytes=0, stage="log_fixture", inodes=0):
        for index in range(20):
            record = logging.LogRecord("scrapy.test", logging.INFO, "", 0, f"entry-{index}-" + "x" * 40, (), None)
            handler.emit(record)
        oversized = logging.LogRecord("scrapy.test", logging.INFO, "", 0, "y" * 1000, (), None)
        handler.emit(oversized)
    files = [log_path, *sorted(output.glob("crawl.log.[0-9]*"))]
    assert len(files) <= 3
    assert all(path.stat().st_size <= 256 for path in files)
    assert files and b"truncated" in log_path.read_bytes()
    handler.close()


def test_atomic_metadata_never_replaces_target_with_partial_bytes(tmp_path, monkeypatch):
    authority_at(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    output.mkdir()
    target = output / "active.json"
    target.write_bytes(b'{"old":true}')
    with admitted_output(output, estimated_bytes=0, stage="metadata_fixture", inodes=0):
        atomic_native_bytes(target, b'{"new":true}')
    assert json.loads(target.read_text()) == {"new": True}
    assert not list(output.glob("*.tmp"))


def test_collector_binds_all_native_writers_to_adapters_without_http(tmp_path, monkeypatch):
    output = tmp_path / "corpus"
    output.mkdir()
    seed = tmp_path / "seeds.json"
    seed.write_text("[]")
    args = SimpleNamespace(
        output=output, seed_file=seed, additional_seed_file=[], horse_seed_file=None,
        limit=1, seconds=1, retry_scope_errors=False,
    )
    captured = {}

    class FakeProcess:
        def __init__(self, settings):
            captured["settings"] = dict(settings)
        def create_crawler(self, spider_class):
            captured["spider_class"] = spider_class
            return SimpleNamespace(spider=SimpleNamespace(storage_error=None))
        def crawl(self, *args, **kwargs):
            captured["crawl"] = (args, kwargs)
        def start(self):
            (output / "closed.json").write_text('{"reason":"fixture"}')

    monkeypatch.setattr(collector, "CrawlerProcess", FakeProcess)
    collector._collect_locked(args, [], __import__("argparse").ArgumentParser())
    settings = captured["settings"]
    assert json.loads((output / "closed.json").read_text())["reason"] == "fixture"
    assert settings["SCHEDULER"].endswith("AdmittedScheduler")
    assert settings["SCHEDULER_DISK_QUEUE"].endswith("AdmittedPickleFifoDiskQueue")
    assert settings["DUPEFILTER_CLASS"].endswith("AdmittedRFPDupeFilter")
    assert settings["EXTENSIONS"]["scrapy.extensions.spiderstate.SpiderState"] is None
    assert settings["EXTENSIONS"]["scrapper.research_storage_io.AdmittedSpiderState"] == 0
    assert settings["LOG_FILE"] is None and settings["LOG_ENABLED"] is False
    assert captured["spider_class"].__name__ == "CorpusSpider"
    manifests = list((output / "collector-releases").rglob("manifest.json"))
    assert len(manifests) == 1
    identity = json.loads(manifests[0].read_text())
    required_sources = {
        "scrapper/research_storage_io.py",
        "ima/research_storage.py",
        "ima/research_sqlite.py",
    }
    assert required_sources <= set(identity["code_hashes"])
    release_code = manifests[0].parent / "code"
    for relative in required_sources:
        assert (release_code / relative).read_bytes() == Path(relative).read_bytes()
