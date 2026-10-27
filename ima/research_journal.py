"""Storage-admitted adapter for Optuna's native journal file backend.

The protocol implementation remains Optuna's JournalFileBackend and lock. The
small subclass below was canaried against Optuna 4.5.0 and 4.9.0 and follows the
append contract: compact JSON records, one newline per batch, and
``get_lock_file`` around the append. It relies on private ``_file_path``,
``_lock``, and ``get_lock_file`` details; recheck compatibility when upgrading.
"""
from __future__ import annotations

import fcntl
import json
import os
import stat
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal._file import get_lock_file


_PROGRAM_ASK_ADMISSION: ContextVar[Path | None] = ContextVar(
    "ima_program_ask_admission", default=None
)


@contextmanager
def program_ask_lock(path: Path):
    """Serialize program ask envelopes with recovery across processes."""
    path = Path(path).absolute()
    if path.resolve() != path or not path.parent.is_dir():
        raise ValueError("Program ask lock requires an existing canonical directory")
    if not path.exists():
        from .research_storage import admitted_output, as_storage_blocked

        try:
            with admitted_output(
                path.parent, estimated_bytes=65536,
                stage="optuna_program_ask_lock", inodes=1,
            ):
                try:
                    fd = os.open(
                        path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                    )
                except FileExistsError:
                    pass
                else:
                    os.close(fd)
        except Exception as exc:
            converted = as_storage_blocked(
                exc, stage="optuna_program_ask_lock", output=path
            )
            if converted is None or converted is exc:
                raise
            raise converted from exc
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened_info = os.fstat(fd)
        if (not stat.S_ISREG(opened_info.st_mode) or opened_info.st_nlink != 1
                or opened_info.st_uid != os.getuid()):
            raise ValueError("Program ask lock must be an own-user single-link file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            locked_info = os.fstat(fd)
            path_info = path.lstat()
            if (not stat.S_ISREG(path_info.st_mode) or path_info.st_nlink != 1
                    or path_info.st_uid != os.getuid()
                    or not os.path.samestat(opened_info, locked_info)
                    or not os.path.samestat(locked_info, path_info)):
                raise ValueError("Program ask lock identity changed while acquiring lock")
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def admitted_program_ask(path: Path, *, estimated_bytes: int):
    """Hold one storage admission across every native append in a program ask."""
    path = Path(path).absolute()
    marker = _PROGRAM_ASK_ADMISSION.set(path)
    try:
        from .research_storage import admitted_output

        with admitted_output(
            path.parent,
            estimated_bytes=estimated_bytes,
            stage="optuna_program_ask",
            inodes=2,
        ):
            yield
    finally:
        _PROGRAM_ASK_ADMISSION.reset(marker)


def _inode_budget(directory: Path) -> int:
    required = 2  # Journal data file plus Optuna's transient symlink lock.
    current = directory
    while not current.exists():
        required += 1
        if current == current.parent:
            raise ValueError("Journal directory has no existing filesystem ancestor")
        current = current.parent
    return required


def _validate_data_file(path: Path, *, may_be_missing: bool = False) -> None:
    if path.absolute() != path or path.resolve() != path:
        raise ValueError("Journal path must be canonical and contain no symlink aliases")
    if not path.exists() and not path.is_symlink() and may_be_missing:
        return
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid()):
        raise ValueError("Journal data must be an own-user regular single-link file")


def _encoded_logs(logs: list[dict[str, Any]]) -> bytes:
    # Matches the native compact-JSON, one-record-per-line journal format.
    text = "\n".join(json.dumps(log, separators=(",", ":")) for log in logs) + "\n"
    return text.encode("utf-8")


class AdmittedJournalFileBackend(JournalFileBackend):
    """Optuna journal backend that admits bytes before lock and append writes."""

    def read_logs(self, log_number_from):
        # Native readers tolerate a partial final line, but could otherwise
        # observe complete lines from an append that is subsequently rolled
        # back. A shared lock on the stable data inode hides that interval
        # without creating any new lock files during read-only status access.
        if not os.environ.get('IMA_STORAGE_ROOT'):
            yield from super().read_logs(log_number_from)
            return
        path = Path(self._file_path)
        _validate_data_file(path)
        with path.open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            opened = os.fstat(stream.fileno())
            current = path.lstat()
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or opened.st_uid != os.getuid() or current.st_nlink != 1
                    or not os.path.samestat(opened, current)):
                raise ValueError('Journal data identity changed before replay')
            yield from super().read_logs(log_number_from)

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        if not os.environ.get("IMA_STORAGE_ROOT"):
            # Preserve the installed backend's exact legacy behavior when disabled.
            return super().append_logs(logs)

        from .research_storage import admitted_output, reserved_output_file, as_storage_blocked

        encoded = _encoded_logs(logs)
        primary: BaseException | None = None
        original_length: int | None = None
        lock_fd: int | None = None
        path = Path(self._file_path)
        try:
            if _PROGRAM_ASK_ADMISSION.get() == path.absolute():
                # reserved_output_file charges writes to the enclosing ask lease.
                admission = nullcontext()
            else:
                admission = admitted_output(
                    path.parent,
                    estimated_bytes=len(encoded)+65536,
                    stage="optuna_journal_append",
                    inodes=1,
                )
            with admission:
                _validate_data_file(path)
                # Optuna owns locking, including symlink creation and rename on release.
                with get_lock_file(self._lock):
                    try:
                        with reserved_output_file(path, "ab") as handle:
                            stream_info = os.fstat(handle.fileno())
                            if (not stat.S_ISREG(stream_info.st_mode)
                                    or stream_info.st_nlink != 1
                                    or stream_info.st_uid != os.getuid()
                                    or not os.path.samestat(stream_info, os.stat(path, follow_symlinks=False))):
                                raise ValueError("Journal data identity changed before append")
                            lock_fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                            locked_info = os.fstat(lock_fd)
                            if not os.path.samestat(stream_info, locked_info):
                                raise ValueError("Journal data identity changed while opening")
                            fcntl.flock(lock_fd, fcntl.LOCK_EX)
                            locked_info = os.fstat(lock_fd)
                            _validate_data_file(path)
                            if (locked_info.st_nlink != 1
                                    or not os.path.samestat(locked_info, path.lstat())):
                                raise ValueError('Journal data identity changed while acquiring write lock')
                            original_length = locked_info.st_size
                            try:
                                handle.write(encoded)
                                handle.flush()
                                os.fsync(handle.fileno())
                            except BaseException as exc:
                                primary = exc
                    except BaseException as exc:
                        if primary is None:
                            primary = exc
                    if primary is not None and original_length is not None and lock_fd is not None:
                        try:
                            locked_info = os.fstat(lock_fd)
                            if (not stat.S_ISREG(locked_info.st_mode)
                                    or locked_info.st_nlink != 1
                                    or locked_info.st_uid != os.getuid()
                                    or not os.path.samestat(locked_info, os.stat(path, follow_symlinks=False))):
                                raise ValueError("Journal data identity changed before rollback")
                            os.ftruncate(lock_fd, original_length)
                            os.fsync(lock_fd)
                        except BaseException as cleanup:
                            primary.add_note(
                                "Optuna journal rollback failed: "
                                f"{type(cleanup).__name__}, errno={getattr(cleanup, 'errno', None)}; "
                                f"original file length was {original_length}"
                            )
                    if primary is not None:
                        raise primary
        except Exception as exc:
            converted = as_storage_blocked(exc, stage="optuna_journal_append", output=path)
            if converted is None or converted is exc:
                raise
            raise converted from exc
        finally:
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Optuna journal descriptor close failed: "
                        f"{type(cleanup).__name__}, errno={cleanup.errno}"
                    )


def admitted_journal_backend(path: Path) -> JournalFileBackend:
    """Create native journal storage only after enabled admission."""
    path = Path(path).absolute()
    if not os.environ.get("IMA_STORAGE_ROOT"):
        path.parent.mkdir(parents=True, exist_ok=True)
        return JournalFileBackend(str(path))

    from .research_storage import admitted_output, reserved_output_file

    try:
        with admitted_output(
            path.parent,
            estimated_bytes=65536*_inode_budget(path.parent),
            stage="optuna_journal_create",
            inodes=_inode_budget(path.parent),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            _validate_data_file(path, may_be_missing=True)
            # The admitted bounded opener creates and checks the data inode first;
            # Optuna's constructor then observes the existing empty file.
            with reserved_output_file(path, "ab"):
                pass
            _validate_data_file(path)
            return AdmittedJournalFileBackend(str(path))
    except Exception as exc:
        from .research_storage import as_storage_blocked

        converted = as_storage_blocked(exc, stage="optuna_journal_create", output=path)
        if converted is None or converted is exc:
            raise
        raise converted from exc
