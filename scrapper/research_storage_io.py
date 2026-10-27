"""Bounded adapters for the installed Scrapy 2.19 JOBDIR writer seams.

The disk queue keeps Scrapy's request serializer and queuelib's FIFO layout.
Only writes and close-time state publication are intercepted; existing JOBDIRs
remain readable by Scrapy's native PickleFifoDiskQueue and RFPDupeFilter.
Callers serialize cooperating writers to each JOBDIR. Path identity checks do
not isolate these files from arbitrary hostile code running as the same UID.
"""
from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path
import pickle
import stat
import struct
from contextlib import contextmanager
from importlib.metadata import version
from uuid import uuid4

import scrapy
from scrapy import squeues
from queuelib.queue import FifoDiskQueue
from scrapy.core.scheduler import Scheduler
from scrapy.dupefilters import RFPDupeFilter, _read_fingerprints
from scrapy.extensions.spiderstate import SpiderState
from scrapy.pqueues import DownloaderAwarePriorityQueue
from scrapy.utils.job import job_dir

from ima.research_storage import (
    _OUTPUT_BUDGET,
    StorageBlocked,
    as_storage_blocked,
    reserved_output_file,
)


_SCRAPY_VERSION = "2.19.0"
_QUEUELIB_VERSION = "1.10.0"
if scrapy.__version__ != _SCRAPY_VERSION or version("queuelib") != _QUEUELIB_VERSION:
    raise RuntimeError(
        "Storage adapters are qualified only for Scrapy 2.19.0 and queuelib 1.10.0"
    )


@contextmanager
def admitted_native_growth(path, *, bytes_required: int, inodes_required: int = 0,
                          hold_inodes: int = 0):
    """Extend the active output lease before a native writer grows a file."""
    if (isinstance(bytes_required, bool) or not isinstance(bytes_required, int)
            or bytes_required < 0 or isinstance(inodes_required, bool)
            or not isinstance(inodes_required, int) or inodes_required < 0
            or isinstance(hold_inodes, bool) or not isinstance(hold_inodes, int)
            or hold_inodes < 0):
        raise ValueError("Native growth must be nonnegative integer bytes and inodes")
    budget = _OUTPUT_BUDGET.get()
    if budget is None:
        yield
        return
    if not budget.get("consume", True):
        raise RuntimeError("Native JOBDIR writers require a consuming output lease")
    # New directories and empty files also allocate filesystem metadata blocks.
    bytes_required += 64 * 1024 * inodes_required
    target = Path(path).absolute()
    root = Path(budget["root"])
    if target.resolve(strict=False) != target or not target.is_relative_to(root):
        raise ValueError("Native writer escaped its admitted output root")

    target_bytes = budget["written"] + bytes_required
    extra_bytes = max(0, target_bytes - budget["limit"])
    extra_inodes = max(0, inodes_required + hold_inodes - budget["inode_remaining"])
    if extra_bytes or extra_inodes:
        budget["authority"].extend(
            budget["token"], extra_bytes=extra_bytes, extra_inodes=extra_inodes
        )
        budget["limit"] += extra_bytes
        budget["inode_remaining"] += extra_inodes
    prior_written = budget["written"]
    budget["written"] = target_bytes
    try:
        yield
    except BaseException as exc:
        budget["written"] = prior_written
        converted = as_storage_blocked(
            exc, stage="scrapy_native_io", output=budget["root"]
        )
        if converted is not None and converted is not exc:
            raise converted from exc
        raise
    else:
        budget["authority"].consume(
            budget["token"], written_bytes=bytes_required,
            created_inodes=inodes_required,
        )
        budget["inode_remaining"] -= inodes_required


def atomic_native_bytes(path, body: bytes) -> None:
    """Atomically replace small native metadata without changing its format."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        _validate_owned_file(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    identity = None
    try:
        with admitted_native_growth(target, bytes_required=64 * 1024):
            with reserved_output_file(temporary, "xb") as stream:
                opened = os.fstat(stream.fileno())
                identity = (opened.st_dev, opened.st_ino)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            validate_owned_temporary(temporary, identity)
            if target.exists() or target.is_symlink():
                _validate_owned_file(target)
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException as primary:
        cleanup_owned_temporary(temporary, identity, primary)
        raise


def validate_owned_temporary(path, identity):
    _validate_owned_file(path)
    current = Path(path).lstat()
    if (current.st_dev, current.st_ino) != identity:
        raise RuntimeError("Native publication staging identity changed")


def cleanup_owned_temporary(path, identity, primary=None):
    """Only remove the staging inode created by this operation."""
    if identity is None:
        return
    try:
        try:
            current = Path(path).lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != identity:
            raise RuntimeError("Changed native staging identity retained")
        Path(path).unlink()
    except BaseException as cleanup:
        if primary is None:
            raise
        primary.add_note(f"Native metadata cleanup also failed for {path}: {cleanup!r}")


def _missing_directory_count(path: Path, root: Path) -> int:
    missing = 0
    current = path
    while current != root and not current.exists():
        if current.is_symlink():
            raise ValueError("Symlink in native JOBDIR path")
        missing += 1
        current = current.parent
    if current != root and not current.is_relative_to(root):
        raise ValueError("Native JOBDIR path escaped output root")
    return missing


def prepare_jobdir(jobdir, output_root) -> None:
    """Admit JOBDIR and requests.queue directories before Scrapy initializes."""
    root = Path(output_root).absolute()
    job = Path(jobdir).absolute()
    queue = job / "requests.queue"
    if job.resolve(strict=False) != job or queue.resolve(strict=False) != queue:
        raise ValueError("Symlinked Scrapy JOBDIR path")
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid():
        raise ValueError("Scrapy output root must be an own-user directory")
    budget = _OUTPUT_BUDGET.get()
    inodes = 0 if budget is None else _missing_directory_count(queue, root)
    with admitted_native_growth(queue, bytes_required=0, inodes_required=inodes):
        job.mkdir(parents=True, exist_ok=True)
        queue.mkdir(exist_ok=True)
    if not job.is_dir() or not queue.is_dir():
        raise ValueError("Scrapy JOBDIR components must be directories")
    for directory in (job, queue):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError(f"Unsafe Scrapy JOBDIR directory: {directory}")


class AdmittedFifoDiskQueue(FifoDiskQueue):
    """Queuelib FIFO with the native record format and pre-write admission."""

    def __init__(self, path: str | os.PathLike[str], chunksize: int = 100000) -> None:
        directory = Path(path).absolute()
        if directory.exists():
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError(f"Unsafe native FIFO directory: {directory}")
        budget = _OUTPUT_BUDGET.get()
        inodes = 0
        if budget is not None:
            root = Path(budget["root"])
            inodes += _missing_directory_count(directory, root)
        info_path = directory / "info.json"
        head_number = 0
        if info_path.exists():
            _validate_owned_file(info_path)
            try:
                head_number = int(json.loads(info_path.read_text())["head"][0])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                head_number = 0
        head_path = directory / f"q{head_number:05d}"
        if head_path.exists():
            _validate_owned_file(head_path)
        if not head_path.exists():
            inodes += 1
        with admitted_native_growth(directory, bytes_required=0, inodes_required=inodes):
            super().__init__(directory, chunksize=chunksize)

    def push(self, string: bytes) -> None:
        if not isinstance(string, bytes):
            raise TypeError(f"Unsupported type: {type(string).__name__}")
        head_number, head_position = self.info["head"]
        record_size = struct.calcsize(self.szhdr_format) + len(string)
        source_path = Path(self.path, f"q{head_number:05d}")
        next_number = head_number + 1
        next_path = Path(self.path, f"q{next_number:05d}")
        rolls = head_position + 1 == self.chunksize
        _validate_owned_file(source_path)
        opened = os.fstat(self.headf.fileno())
        current = source_path.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Native FIFO head file changed while open")
        if next_path.exists():
            _validate_owned_file(next_path)
        old_size = source_path.stat().st_size
        next_existed = next_path.exists()
        old_info = {"size": self.info["size"], "head": list(self.info["head"])}
        committed = False
        try:
            with admitted_native_growth(
                source_path, bytes_required=record_size,
                inodes_required=int(rolls and not next_existed), hold_inodes=1,
            ):
                super().push(string)
                if source_path.stat().st_size - old_size != record_size:
                    raise OSError(errno.EIO, "short native FIFO append")
                self._saveinfo(self.info)
                committed = True
        except BaseException as primary:
            info_path = self._infopath()
            if not committed and info_path.exists() and self.info != old_info:
                try:
                    committed = info_path.read_bytes() == json.dumps(self.info).encode("utf-8")
                except BaseException as cleanup:
                    primary.add_note(f"Native FIFO commit inspection failed: {cleanup!r}")
            if committed:
                raise
            self.info["size"] = old_info["size"]
            self.info["head"] = old_info["head"]
            try:
                if self.headf.closed or Path(self.headf.name) != source_path:
                    if not self.headf.closed:
                        self.headf.close()
                    self.headf = self._openchunk(head_number, "ab+")
                with source_path.open("r+b") as rollback:
                    rollback.truncate(old_size)
                    rollback.flush()
                    os.fsync(rollback.fileno())
                os.lseek(self.headf.fileno(), 0, os.SEEK_END)
                if not next_existed and next_path.exists():
                    next_path.unlink()
            except BaseException as cleanup:
                primary.add_note(f"Native FIFO rollback also failed: {cleanup!r}")
            raise

    def _saveinfo(self, info) -> None:
        atomic_native_bytes(self._infopath(), json.dumps(info).encode("utf-8"))

    def clear(self) -> None:
        with admitted_native_growth(self.path, bytes_required=0, inodes_required=1):
            super().clear()


def _with_admitted_parent(queue_class):
    class ParentCreatedFIFO(queue_class):
        def __init__(self, path, *args, **kwargs):
            parent = Path(path).absolute().parent
            budget = _OUTPUT_BUDGET.get()
            inodes = 0 if budget is None else _missing_directory_count(
                parent, Path(budget["root"])
            )
            with admitted_native_growth(parent, bytes_required=0, inodes_required=inodes):
                parent.mkdir(parents=True, exist_ok=True)
            super().__init__(path, *args, **kwargs)

    ParentCreatedFIFO.__module__ = __name__
    return ParentCreatedFIFO


_SerializableFIFO = squeues._serializable_queue(
    _with_admitted_parent(AdmittedFifoDiskQueue),
    squeues._pickle_serialize,
    pickle.loads,
)
AdmittedPickleFifoDiskQueue = squeues._scrapy_serialization_queue(_SerializableFIFO)
AdmittedPickleFifoDiskQueue.__module__ = __name__
AdmittedPickleFifoDiskQueue.__name__ = "AdmittedPickleFifoDiskQueue"


class AdmittedRFPDupeFilter(RFPDupeFilter):
    """Native RFPDupeFilter with guarded, flushed requests.seen appends."""

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            job_dir(crawler.settings), crawler.settings.getbool("DUPEFILTER_DEBUG"),
            fingerprinter=crawler.request_fingerprinter,
        )

    def __init__(self, path=None, debug=False, *, fingerprinter=None):
        super().__init__(None, debug, fingerprinter=fingerprinter)
        self._last_append = None
        if not path:
            return
        target = Path(path, "requests.seen")
        self._dupefilter_path = target
        if target.exists():
            _validate_owned_file(target)
        create_inode = int(not target.exists())
        with admitted_native_growth(target, bytes_required=0, inodes_required=create_inode):
            fd = os.open(
                target,
                os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                _validate_open_file(target, fd)
                self.file = os.fdopen(fd, "a+b", buffering=0)
            except BaseException:
                os.close(fd)
                raise
            self.file.seek(0)
            self._fingerprints.update(_read_fingerprints(self.file.read()))
            self.file.seek(0, os.SEEK_END)

    def request_seen(self, request):
        fingerprint = self._fingerprint(request)
        if fingerprint in self._fingerprints:
            return True
        if self.file:
            record = len(fingerprint).to_bytes(2, "big") + fingerprint
            self.file.flush()
            _validate_owned_file(self._dupefilter_path)
            opened = os.fstat(self.file.fileno())
            current = self._dupefilter_path.lstat()
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("Dupefilter file changed while open")
            self.file.seek(0, os.SEEK_END)
            old_size = self.file.tell()
            attempted = False
            try:
                with admitted_native_growth(
                    self._dupefilter_path, bytes_required=len(record)
                ):
                    attempted = True
                    written = self.file.write(record)
                    if written != len(record):
                        raise OSError(errno.EIO, "short requests.seen append")
                    self.file.flush()
                    os.fsync(self.file.fileno())
                self._last_append = (fingerprint, old_size, old_size + len(record))
            except BaseException as primary:
                if attempted:
                    try:
                        self.file.flush()
                        os.ftruncate(self.file.fileno(), old_size)
                        self.file.seek(0, os.SEEK_END)
                        self.file.flush()
                    except BaseException as cleanup:
                        primary.add_note(f"Dupefilter append rollback also failed: {cleanup!r}")
                raise
        self._fingerprints.add(fingerprint)
        return False

    def unsee(self, request):
        """Undo the immediately preceding persisted fingerprint after queue denial."""
        fingerprint = self._fingerprint(request)
        if fingerprint not in self._fingerprints:
            return
        if self.file:
            last = self._last_append
            if last is None or last[0] != fingerprint:
                raise RuntimeError("Cannot safely roll back a non-tail dupefilter append")
            _, start, end = last
            self.file.flush()
            if os.fstat(self.file.fileno()).st_size != end:
                raise RuntimeError("Dupefilter tail changed before scheduler rollback")
            os.ftruncate(self.file.fileno(), start)
            self.file.seek(0, os.SEEK_END)
            self.file.flush()
            os.fsync(self.file.fileno())
            self._last_append = None
        self._fingerprints.remove(fingerprint)


class AdmittedScheduler(Scheduler):
    def _write_dqs_state(self, dqdir, state):
        atomic_native_bytes(Path(dqdir, "active.json"), json.dumps(state).encode("utf-8"))

    def close(self, reason):
        try:
            return super().close(reason)
        except StorageBlocked as blocked:
            self.spider.storage_error = {**blocked.diagnostic, 'error': str(blocked)}
            raise

    def enqueue_request(self, request):
        try:
            return self._enqueue_request(request)
        except StorageBlocked as blocked:
            self.spider._storage_hold(blocked)
            raise

    def _enqueue_request(self, request):
        fingerprint = None
        if not request.dont_filter:
            fingerprint = self.df._fingerprint(request)
            if fingerprint in self.df._fingerprints:
                self.df.log(request, self.spider)
                return False
        if self.dqdir and self.dqs is not None:
            if isinstance(self.dqs, DownloaderAwarePriorityQueue):
                active = {slot: sorted(set(queue.queues) | set(queue._start_queues))
                          for slot, queue in self.dqs.pqueues.items()}
                slot = self.dqs._downloader_interface.get_slot_key(request)
                active[slot] = sorted(set(active.get(slot, ())) | {-request.priority})
            else:
                active = sorted(set(self.dqs.queues) | set(self.dqs._start_queues) | {-request.priority})
            self._write_dqs_state(self.dqdir, active)
        disk_queued = self._dqpush(request)
        assert self.stats is not None
        if disk_queued:
            self.stats.inc_value("scheduler/enqueued/disk")
        else:
            self._mqpush(request)
            self.stats.inc_value("scheduler/enqueued/memory")
        if not request.dont_filter:
            try:
                self.df.request_seen(request)
            except BaseException as primary:
                primary.add_note("Request is already queued; resume from JOBDIR after storage recovery")
                raise
        self.stats.inc_value("scheduler/enqueued")
        return True


class AdmittedSpiderState(SpiderState):
    def spider_closed(self, spider):
        if self.jobdir:
            assert hasattr(spider, "state")
            try:
                atomic_native_bytes(self.statefn, pickle.dumps(spider.state, protocol=4))
            except StorageBlocked as blocked:
                spider.storage_error = {**blocked.diagnostic, 'error': str(blocked)}
                raise


class BoundedRotatingFileHandler(logging.Handler):
    """Per-record admitted rotating log with bounded retained file size."""

    def __init__(self, path, *, max_bytes=1024**2, backup_count=2, encoding="utf-8", on_storage_block=None):
        super().__init__()
        self.baseFilename = str(Path(path).absolute())
        self.maxBytes = max_bytes
        self.backupCount = backup_count
        self.encoding = encoding
        self.on_storage_block = on_storage_block

    def _backup(self, index):
        return Path(f"{self.baseFilename}.{index}")

    def _rotate(self):
        oldest = self._backup(self.backupCount)
        if oldest.exists():
            _validate_owned_file(oldest)
            oldest.unlink()
        for index in range(self.backupCount - 1, 0, -1):
            source, destination = self._backup(index), self._backup(index + 1)
            if source.exists():
                _validate_owned_file(source)
                if destination.exists():
                    _validate_owned_file(destination)
                os.replace(source, destination)
        base = Path(self.baseFilename)
        if base.exists():
            _validate_owned_file(base)
            destination = self._backup(1)
            if destination.exists():
                _validate_owned_file(destination)
            os.replace(base, self._backup(1))

    def emit(self, record):
        text = self.format(record) + "\n"
        encoded = text.encode(self.encoding, errors="replace")
        if len(encoded) > self.maxBytes:
            marker = b"...[log record truncated]\n"
            encoded = encoded[: self.maxBytes - len(marker)]
            encoded = encoded.decode(self.encoding, errors="ignore").encode(self.encoding) + marker
        base = Path(self.baseFilename)
        if base.exists():
            _validate_owned_file(base)
        current_size = base.stat().st_size if base.exists() else 0
        rollover = current_size > 0 and current_size + len(encoded) > self.maxBytes
        create_inode = int(rollover or not base.exists())
        self.acquire()
        try:
            with admitted_native_growth(
                base, bytes_required=len(encoded), inodes_required=create_inode
            ):
                if rollover:
                    self._rotate()
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(base, flags, 0o600)
                try:
                    _validate_open_file(base, fd)
                    old_size = os.fstat(fd).st_size
                    try:
                        offset = 0
                        while offset < len(encoded):
                            written = _native_write(fd, encoded[offset:])
                            if written <= 0:
                                raise OSError(errno.EIO, "short bounded log append")
                            offset += written
                        os.fsync(fd)
                    except BaseException as primary:
                        try:
                            os.ftruncate(fd, old_size)
                            os.fsync(fd)
                        except BaseException as cleanup:
                            primary.add_note(f"Log append rollback also failed: {cleanup!r}")
                        raise
                finally:
                    os.close(fd)
        except StorageBlocked as blocked:
            if self.on_storage_block is not None:
                try:self.on_storage_block(blocked)
                except BaseException as callback_error:
                    blocked.add_note(f"Storage hold callback also failed: {callback_error!r}")
            raise
        finally:
            self.release()

    def flush(self):
        pass


_native_write = os.write


def _validate_owned_file(path):
    info = Path(path).lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid()):
        raise ValueError(f"Unsafe native Scrapy output: {path}")


def _validate_open_file(path, fd):
    opened = os.fstat(fd)
    current = Path(path).lstat()
    if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
        raise ValueError(f"Native Scrapy output changed while opening: {path}")
