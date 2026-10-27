"""Bounded transactions for trusted, single-file research SQLite stores.

This is a cooperative internal-use API, not an OS quota or a general-purpose
SQLite sandbox. Callers may use only the supplied main database for ordinary
SQL; ATTACH/DETACH, temporary objects, explicit transaction control,
``executescript``, extension loading, and external side effects are forbidden.
The SQLite authorizer enforces the database/transaction restrictions it can
observe. The caller must still keep SQL trusted and must size growth honestly.

Only DELETE, TRUNCATE, and PERSIST rollback journals are admitted. WAL and
hot-journal recovery are refused without changing or repairing the source.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from urllib.parse import quote

from .research_storage import admitted_output, as_storage_blocked


_METADATA_BYTES = 128 * 1024
_ALLOWED_JOURNAL_MODES = {"delete", "truncate", "persist"}


def _existing_bytes(path: Path) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        raise ValueError(f"Unsafe SQLite file: {path}")
    return info.st_size


def _check_rollback_journal(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        raise ValueError("Unsafe SQLite rollback journal")
    with path.open("rb") as stream:
        header = stream.read(8)
    # TRUNCATE and PERSIST may leave an empty or zeroed journal. A nonzero
    # header could be a hot journal; never let SQLite repair it implicitly.
    if header and any(header):
        raise RuntimeError("Unexpected rollback journal; preserving database without recovery")


def _missing_parent_count(parent: Path) -> int:
    count = 0
    current = parent
    while not current.exists():
        if current.is_symlink():
            raise ValueError("Symlink parent rejected")
        count += 1
        current = current.parent
    if current.is_symlink() or current.stat().st_uid != os.getuid():
        raise ValueError("SQLite parent must be own-user and non-symlink")
    return count


def _journal_peak_bytes(main_size: int) -> int:
    # SQLite's rollback journal stores a page number beside each original
    # page. 512 bytes is the smallest SQLite page, so this conservatively
    # bounds those per-page headers without opening the database first.
    pages_upper_bound = math.ceil(main_size / 512)
    return main_size + pages_upper_bound * 8 + 4096


def _uri(path: Path, mode: str) -> str:
    return f"file:{quote(str(path), safe='/')}?mode={mode}"


def _authorizer(action, arg1, arg2, database, trigger):
    forbidden = {
        sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_PRAGMA, sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX, sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TEMP_INDEX, sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER, sqlite3.SQLITE_DROP_TEMP_VIEW,
    }
    if action == sqlite3.SQLITE_PRAGMA:
        # Schema introspection is the sole caller-visible PRAGMA. In
        # particular, do not permit callers to alter journal/page settings.
        if (arg1 or "").lower() == "table_info" and database in (None, "main"):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action in forbidden:
        return sqlite3.SQLITE_DENY
    if database not in (None, "main"):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _legacy_authorizer(action, arg1, arg2, database, trigger):
    # The default sqlite3 transaction mode emits BEGIN for the first DML.
    # User-authored transaction statements are rejected by _MainDatabase.
    if action == sqlite3.SQLITE_TRANSACTION:
        return sqlite3.SQLITE_OK
    return _authorizer(action, arg1, arg2, database, trigger)


def _leading_keyword(sql):
    statement = sql.lstrip()
    while True:
        if statement.startswith("--"):
            newline = statement.find("\n")
            statement = "" if newline < 0 else statement[newline + 1:].lstrip()
        elif statement.startswith("/*"):
            end = statement.find("*/", 2)
            if end < 0:
                break
            statement = statement[end + 2:].lstrip()
        else:
            break
    match = re.match(r"[A-Za-z]+", statement)
    return match.group(0).lower() if match else ""


class _MainDatabase:
    """Small SQL surface: no connection/cursor lifecycle or script escape."""
    __slots__ = ("__connection",)

    def __init__(self, connection):
        self.__connection = connection

    def execute(self, sql, parameters=()):
        if not isinstance(sql, str) or _leading_keyword(sql) in {"begin", "commit", "end", "rollback", "savepoint", "release"}:
            raise ValueError("Transaction control is owned by admitted_sqlite")
        return _MainCursor(self.__connection.execute(sql, parameters))

    def executemany(self, sql, parameters):
        if not isinstance(sql, str) or _leading_keyword(sql) in {"begin", "commit", "end", "rollback", "savepoint", "release"}:
            raise ValueError("Transaction control is owned by admitted_sqlite")
        return _MainCursor(self.__connection.executemany(sql, parameters))


class _MainCursor:
    __slots__ = ("__cursor",)

    def __init__(self, cursor):
        self.__cursor = cursor

    @property
    def rowcount(self):
        return self.__cursor.rowcount

    @property
    def lastrowid(self):
        return self.__cursor.lastrowid

    def fetchone(self):
        return self.__cursor.fetchone()

    def fetchall(self):
        return self.__cursor.fetchall()


@contextmanager
def _managed_transaction(conn):
    """Commit/rollback/close without allowing cleanup to mask the primary."""
    primary = None
    try:
        conn.set_authorizer(_legacy_authorizer if conn.isolation_level is not None else _authorizer)
        try:
            yield _MainDatabase(conn)
            conn.set_authorizer(None)
            conn.commit()
        except BaseException as exc:
            primary = exc
            try:
                conn.set_authorizer(None)
            except BaseException as cleanup:
                exc.add_note(f"SQLite authorizer cleanup also failed: {cleanup!r}")
            try:
                if conn.in_transaction:
                    conn.rollback()
            except BaseException as cleanup:
                exc.add_note(f"SQLite rollback cleanup also failed: {cleanup!r}")
            raise
    except BaseException as exc:
        if primary is None:
            primary = exc
        raise
    finally:
        try:
            conn.set_authorizer(None)
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                primary.add_note(f"SQLite authorizer cleanup also failed: {cleanup!r}")
        try:
            conn.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"SQLite close cleanup also failed: {cleanup!r}")


@contextmanager
def admitted_sqlite(path, *, estimated_growth_bytes: int, stage="research_sqlite", row_factory=None):
    """Yield a restricted main-database SQL surface inside an admitted transaction.

    With ``IMA_STORAGE_ROOT`` enabled, admission precedes creation of missing
    parent directories, the lock inode, and the database. The reservation
    covers the existing main-file rollback-journal peak, admitted database
    growth, metadata allowance, and missing/transient inodes. ``max_page_count``
    caps the main database at its current page count plus admitted growth, and
    SQLite's reported setting is checked explicitly. With storage admission
    disabled, this uses the legacy parent-creation/default-transaction path
    without the lock inode or page cap. Callers must not run untrusted SQL.
    The connection and enabled reservation remain live through cleanup.
    """
    if isinstance(estimated_growth_bytes, bool) or not isinstance(estimated_growth_bytes, int) or estimated_growth_bytes < 0:
        raise ValueError("estimated_growth_bytes must be a nonnegative integer")
    db = Path(path).absolute()
    parent = db.parent
    if not os.environ.get("IMA_STORAGE_ROOT"):
        # Preserve the historical ungoverned path: parent creation and a
        # default SQLite transaction, with no lease, page cap, or lock inode.
        parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db, timeout=30)
        connection.row_factory = row_factory
        with _managed_transaction(connection) as conn:
            yield conn
        return

    if db.resolve() != db or db.name in {"", ".", ".."}:
        raise ValueError("SQLite path must be canonical")

    missing_dirs = _missing_parent_count(parent)
    main_size = _existing_bytes(db)
    reserve_bytes = _journal_peak_bytes(main_size) + estimated_growth_bytes + _METADATA_BYTES
    reserve_inodes = missing_dirs + (3 if main_size == 0 else 2)  # lock, main, and rollback journal

    try:
        with admitted_output(db, estimated_bytes=reserve_bytes, stage=stage, inodes=reserve_inodes) as admission:
            if missing_dirs:
                parent.mkdir(parents=True, exist_ok=True)
            if parent.resolve() != parent or parent.is_symlink():
                raise ValueError("SQLite parent changed or became a symlink")

            lock_path = Path(str(db) + ".admission-lock")
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            lock_fd = os.open(lock_path, flags, 0o600)
            try:
                lock_info = os.fstat(lock_fd)
                if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1
                        or lock_info.st_uid != os.getuid()):
                    raise ValueError("Unsafe SQLite admission lock")
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    # Reconcile reservation after serialization: a preceding
                    # admitted writer may have grown the database while this
                    # writer waited. Extend before opening/mutating SQLite.
                    fresh_size = _existing_bytes(db)
                    fresh_required = _journal_peak_bytes(fresh_size) + estimated_growth_bytes + _METADATA_BYTES
                    if admission is not None:
                        authority, token = admission
                        if fresh_required > reserve_bytes:
                            authority.extend(token, extra_bytes=fresh_required - reserve_bytes)
                            reserve_bytes = fresh_required

                    _check_rollback_journal(Path(str(db) + "-journal"))
                    if db.exists():
                        # Read mode using a read-only connection before any
                        # writable connection can migrate or recover the DB.
                        with closing(sqlite3.connect(_uri(db, "ro"), uri=True, isolation_level=None)) as probe:
                            mode = probe.execute("PRAGMA journal_mode").fetchone()[0].lower()
                        if mode == "wal":
                            raise RuntimeError("SQLite WAL mode is unsupported by admitted_sqlite")
                        if mode not in _ALLOWED_JOURNAL_MODES:
                            raise RuntimeError(f"Unsupported SQLite journal mode: {mode}")

                    conn = sqlite3.connect(db, timeout=30, isolation_level=None)
                    conn.row_factory = row_factory
                    managed = False
                    try:
                        # Do not change journal mode. A new database defaults
                        # to DELETE; an existing one was checked read-only.
                        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
                        if mode not in _ALLOWED_JOURNAL_MODES:
                            raise RuntimeError(f"Unsupported SQLite journal mode: {mode}")
                        conn.execute("PRAGMA temp_store=MEMORY")
                        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
                        conn.execute("BEGIN IMMEDIATE")
                        current_pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
                        old_cap = int(conn.execute("PRAGMA max_page_count").fetchone()[0])
                        if old_cap < current_pages:
                            raise RuntimeError("SQLite max_page_count is below current page_count")
                        target_cap = current_pages + math.ceil(estimated_growth_bytes / page_size)
                        # Never request a cap below the current database. An
                        # empty new DB still needs one page for its first write.
                        requested_cap = max(1, current_pages, target_cap)
                        returned_cap = int(conn.execute(f"PRAGMA max_page_count={requested_cap}").fetchone()[0])
                        readback_cap = int(conn.execute("PRAGMA max_page_count").fetchone()[0])
                        if returned_cap != requested_cap or readback_cap != requested_cap:
                            raise RuntimeError("SQLite max_page_count did not accept the requested bound")
                        managed = True
                        with _managed_transaction(conn) as restricted:
                            yield restricted
                    except BaseException:
                        # _managed_transaction closes after it has attempted
                        # rollback; setup failures before entering it still
                        # need a close, without replacing the setup exception.
                        if not managed:
                            import sys
                            primary = sys.exception()
                            try:
                                conn.close()
                            except BaseException as cleanup:
                                if primary is not None:
                                    primary.add_note(f"SQLite close cleanup also failed: {cleanup!r}")
                                else:
                                    raise
                        raise
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
    except Exception as exc:
        blocked = as_storage_blocked(exc, stage=stage, output=db)
        if blocked is None or blocked is exc:
            raise
        raise blocked from exc
