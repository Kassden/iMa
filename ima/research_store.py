"""SQLite durability ledger for research optimizer attempts."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .research_sqlite import admitted_sqlite


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    signature: str
    status: str
    payload: dict[str, Any]


class ResearchLedger:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only:
            self._validate_read_only_schema()
        else:
            self._init()

    def reserve_attempt(self, signature: str, payload: dict[str, Any]) -> AttemptRecord:
        attempt_id = f"attempt-{signature}"
        payload_json = _json(payload)
        updated_at = utc_now()
        with self._connect(estimated_growth_bytes=_growth_estimate(
            attempt_id, signature, "reserved", payload_json, updated_at,
        )) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO attempts
                (attempt_id, signature, status, payload_json, updated_at)
                VALUES (?, ?, 'reserved', ?, ?)
                """,
                (attempt_id, signature, payload_json, updated_at),
            )
            row = conn.execute(
                "SELECT attempt_id, signature, status, payload_json FROM attempts WHERE signature = ?",
                (signature,),
            ).fetchone()
        return _record(row)

    def mark_running(self, attempt_id: str, *, now: float | None = None) -> AttemptRecord:
        now = time.time() if now is None else now
        if isinstance(now,bool) or not isinstance(now,(int,float)) or not math.isfinite(now):
            raise ValueError("Finite retry clock required")
        with self._connect(estimated_growth_bytes=_growth_estimate(attempt_id, "running", utc_now())) as conn:
            retry = conn.execute('SELECT retry_at_epoch FROM infrastructure_retries '
                'WHERE attempt_id=? AND active=1',(attempt_id,)).fetchone()
            if retry is not None and retry['retry_at_epoch']>now:
                from .research_storage import StorageBlocked
                raise StorageBlocked('Research attempt is waiting for infrastructure retry',
                    {'stage':'research_ledger_backoff','retry_at_epoch':retry['retry_at_epoch']})
            conn.execute(
                """UPDATE attempts SET status = 'running', updated_at = ?
                   WHERE attempt_id = ? AND status = 'reserved'""",
                (utc_now(), attempt_id),
            )
            conn.execute('UPDATE infrastructure_retries SET active=0 WHERE attempt_id=?',(attempt_id,))
            row = conn.execute(
                "SELECT attempt_id, signature, status, payload_json FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown attempt_id: {attempt_id}")
        return _record(row)

    def defer_infrastructure(self,attempt_id,diagnostic,*,delay_seconds=30,now=None,runtime_snapshot=None):
        """Preserve one logical allocation without emitting a scientific outcome."""
        now,encoded,runtime=_retry_values(diagnostic,delay_seconds,now,runtime_snapshot)
        with self._connect(estimated_growth_bytes=_growth_estimate(
            attempt_id, encoded, runtime, utc_now(), now + delay_seconds,
        )) as conn:
            row=conn.execute('SELECT status FROM attempts WHERE attempt_id=?',(attempt_id,)).fetchone()
            if row is None:raise KeyError(attempt_id)
            if row['status'] not in {'reserved','running'}:
                raise ValueError('Cannot defer a terminal scientific result')
            conn.execute("UPDATE attempts SET status='reserved',updated_at=? WHERE attempt_id=?",(utc_now(),attempt_id))
            retry=_record_retry(conn,attempt_id,now+delay_seconds,encoded,runtime)
        return {'attempt_id':attempt_id,'retry_count':retry,'retry_at_epoch':now+delay_seconds,'diagnostic':diagnostic}

    def defer_operation(self,operation_id,diagnostic,*,delay_seconds=30,now=None,runtime_snapshot=None):
        """Back off a non-trial artifact worker without inventing model attempts."""
        if (not isinstance(operation_id,str)
                or not operation_id.startswith(('dataset:','program:','fold:'))
                or not operation_id.partition(':')[2].strip()):
            raise ValueError('Namespaced non-trial operation required')
        now,encoded,runtime=_retry_values(diagnostic,delay_seconds,now,runtime_snapshot)
        with self._connect(estimated_growth_bytes=_growth_estimate(
            operation_id, encoded, runtime, now + delay_seconds,
        )) as conn:
            retry=_record_retry(conn,operation_id,now+delay_seconds,encoded,runtime)
        return {'operation_id':operation_id,'retry_count':retry,'retry_at_epoch':now+delay_seconds}

    def complete_operation(self,operation_id):
        with self._connect(estimated_growth_bytes=_growth_estimate(operation_id)) as conn:
            conn.execute('UPDATE infrastructure_retries SET active=0 WHERE attempt_id=?',(operation_id,))

    def operation_waits(self):
        with self._connect(read_only=True) as conn:
            rows=conn.execute('SELECT attempt_id AS operation_id,retry_at_epoch,retry_count,diagnostic_json '
                              'FROM infrastructure_retries WHERE active=1 ORDER BY attempt_id').fetchall()
        return [{**dict(row),'diagnostic':json.loads(row['diagnostic_json'])} for row in rows]

    def infrastructure_waits(self):
        with self._connect(read_only=True) as conn:
            rows=conn.execute('SELECT r.attempt_id,r.retry_at_epoch,r.retry_count,r.diagnostic_json '
                "FROM infrastructure_retries r JOIN attempts a ON a.attempt_id=r.attempt_id "
                "WHERE r.active=1 AND a.status='reserved' ORDER BY r.attempt_id").fetchall()
        return [{**dict(row),'diagnostic':json.loads(row['diagnostic_json'])} for row in rows]

    def infrastructure_history(self,attempt_id):
        with self._connect(read_only=True) as conn:
            rows=conn.execute('SELECT retry_count,diagnostic_json,runtime_json,recorded_at '
                'FROM infrastructure_events WHERE attempt_id=? ORDER BY retry_count',(attempt_id,)).fetchall()
        return [{'retry_count':r['retry_count'],'diagnostic':json.loads(r['diagnostic_json']),
                 'runtime':json.loads(r['runtime_json']),'recorded_at':r['recorded_at']} for r in rows]

    def latest_infrastructure_runtime(self,attempt_id):
        with self._connect(read_only=True) as conn:
            row=conn.execute('SELECT runtime_json FROM infrastructure_events WHERE attempt_id=? '
                'ORDER BY retry_count DESC LIMIT 1',(attempt_id,)).fetchone()
        if row is None:raise KeyError('No preserved infrastructure episode for '+attempt_id)
        return json.loads(row['runtime_json'])

    def complete_attempt(
        self,
        attempt_id: str,
        result: dict[str, Any],
        *,
        status: str = "completed",
    ) -> AttemptRecord:
        if status not in {"completed", "failed", "pruned"}:
            raise ValueError(f"Unsupported terminal status: {status}")
        result_json = _json(result)
        existing_db_bytes = self.path.stat().st_size if self.path.exists() else 0
        updated_at = utc_now()
        with self._connect(estimated_growth_bytes=_growth_estimate(
            attempt_id, status, result_json, updated_at,
            overhead_bytes=64 * 1024 + existing_db_bytes,
        )) as conn:
            before = conn.execute(
                "SELECT status, result_json FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if before is None:
                raise KeyError(f"Unknown attempt_id: {attempt_id}")
            if before["status"] in {"completed", "failed", "pruned"}:
                stored = json.loads(before["result_json"] or "{}")
                if stored != result:
                    raise ValueError(f"Attempt {attempt_id} already completed with different result")
            else:
                conn.execute(
                    """
                    UPDATE attempts
                    SET status = ?, result_json = ?, updated_at = ?
                    WHERE attempt_id = ?
                    """,
                    (status, result_json, updated_at, attempt_id),
                )
            conn.execute('UPDATE infrastructure_retries SET active=0 WHERE attempt_id=?',(attempt_id,))
            row = conn.execute(
                "SELECT attempt_id, signature, status, payload_json FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        return _record(row)

    def pending_outbox(self) -> list[dict[str, Any]]:
        with self._connect(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT attempt_id, result_json FROM attempts
                WHERE status = 'completed' AND uploaded_at IS NULL
                ORDER BY attempt_id
                """
            ).fetchall()
        return [
            {"attempt_id": row["attempt_id"], "result": json.loads(row["result_json"] or "{}")}
            for row in rows
        ]

    def mark_uploaded(self, attempt_id: str, remote_id: str) -> None:
        with self._connect(estimated_growth_bytes=_growth_estimate(attempt_id, remote_id, utc_now())) as conn:
            conn.execute(
                """
                UPDATE attempts
                SET uploaded_at = ?, remote_id = ?, updated_at = ?
                WHERE attempt_id = ?
                """,
                (utc_now(), remote_id, utc_now(), attempt_id),
            )

    def pending_tells(self) -> list[dict[str, Any]]:
        with self._connect(read_only=True) as conn:
            rows = conn.execute(
                """SELECT attempt_id, payload_json, result_json FROM attempts
                   WHERE status IN ('completed', 'failed', 'pruned') AND told_at IS NULL
                   ORDER BY updated_at, attempt_id"""
            ).fetchall()
        return [{
            "attempt_id": row["attempt_id"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"] or "{}"),
        } for row in rows]

    def mark_told(self, attempt_id: str) -> None:
        with self._connect(estimated_growth_bytes=_growth_estimate(attempt_id, utc_now())) as conn:
            conn.execute(
                "UPDATE attempts SET told_at = ?, updated_at = ? WHERE attempt_id = ?",
                (utc_now(), utc_now(), attempt_id),
            )

    def terminal_results(self) -> list[dict[str, Any]]:
        with self._connect(read_only=True) as conn:
            rows = conn.execute(
                """SELECT attempt_id, signature, status, payload_json, result_json,
                          remote_id, uploaded_at
                   FROM attempts WHERE status IN ('completed', 'failed', 'pruned')
                   ORDER BY updated_at, attempt_id"""
            ).fetchall()
        return [{
            "attempt_id": row["attempt_id"],
            "signature": row["signature"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"] or "{}"),
            "remote_id": row["remote_id"],
            "uploaded_at": row["uploaded_at"],
        } for row in rows]

    def reserved_attempts(self) -> list[dict[str, Any]]:
        """All durable allocations in reservation order, including failures."""
        with self._connect(read_only=True) as conn:
            rows = conn.execute(
                "SELECT attempt_id, status, payload_json FROM attempts ORDER BY rowid"
            ).fetchall()
        return [{
            "attempt_id": row["attempt_id"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
        } for row in rows]

    def pending_attempts(self) -> list[dict[str, Any]]:
        return [row for row in self.reserved_attempts() if row["status"] == "reserved"]

    def retry_ready(self,attempt_id,*,now=None):
        now=time.time() if now is None else now
        if isinstance(now,bool) or not isinstance(now,(int,float)) or not math.isfinite(now):
            raise ValueError('Finite retry clock required')
        with self._connect(read_only=True) as conn:
            row=conn.execute('SELECT active,retry_at_epoch FROM infrastructure_retries WHERE attempt_id=?',
                             (attempt_id,)).fetchone()
        return row is None or not row['active'] or row['retry_at_epoch']<=now

    def recover_running(self) -> int:
        """Return unsupervised running attempts to reserved on controller startup."""
        with self._connect(estimated_growth_bytes=_growth_estimate("reserved", utc_now())) as conn:
            cursor = conn.execute(
                """UPDATE attempts SET status = 'reserved', updated_at = ?
                   WHERE status = 'running'""",
                (utc_now(),),
            )
        return int(cursor.rowcount)

    def snapshot(self) -> dict[str, int]:
        with self._connect(read_only=True) as conn:
            rows = conn.execute("SELECT status, COUNT(*) count FROM attempts GROUP BY status").fetchall()
            pending_tracking = conn.execute(
                """SELECT COUNT(*) count FROM attempts
                   WHERE status = 'completed' AND uploaded_at IS NULL"""
            ).fetchone()["count"]
            pending_tells = conn.execute(
                """SELECT COUNT(*) count FROM attempts
                   WHERE status IN ('completed', 'failed', 'pruned') AND told_at IS NULL"""
            ).fetchone()["count"]
        snapshot = {row["status"]: int(row["count"]) for row in rows}
        snapshot["pending_tracking"] = int(pending_tracking)
        snapshot["pending_tells"] = int(pending_tells)
        return snapshot

    def _init(self) -> None:
        with self._connect(estimated_growth_bytes=_growth_estimate(
            "infrastructure_retries", "infrastructure_events", "attempts", "told_at",
        )) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS infrastructure_retries ('
                'attempt_id TEXT PRIMARY KEY,retry_at_epoch REAL NOT NULL,retry_count INTEGER NOT NULL,'
                'diagnostic_json TEXT NOT NULL,active INTEGER NOT NULL DEFAULT 1)')
            conn.execute('CREATE TABLE IF NOT EXISTS infrastructure_events ('
                'attempt_id TEXT NOT NULL,retry_count INTEGER NOT NULL,diagnostic_json TEXT NOT NULL,'
                'runtime_json TEXT NOT NULL,recorded_at TEXT NOT NULL,PRIMARY KEY(attempt_id,retry_count))')
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    remote_id TEXT,
                    uploaded_at TEXT,
                    told_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(attempts)").fetchall()
            }
            if "told_at" not in columns:
                conn.execute("ALTER TABLE attempts ADD COLUMN told_at TEXT")

    def _validate_read_only_schema(self) -> None:
        if not self.path.is_file():
            raise RuntimeError("Read-only research ledger does not exist")
        self._validate_read_only_mode()
        uri = self.path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(attempts)").fetchall()}
        required = {"status", "uploaded_at", "told_at"}
        if not required.issubset(columns):
            raise RuntimeError("Read-only research ledger cannot read snapshot from attempts schema")

    def _validate_read_only_mode(self) -> None:
        import os
        if not os.environ.get('IMA_STORAGE_ROOT'):
            return
        # SQLite header offsets 18/19 are the write/read format versions.
        # Reject WAL before SQLite can create or extend unadmitted sidecars.
        with self.path.open('rb') as stream:
            header = stream.read(20)
        if len(header)==20 and (header[18]==2 or header[19]==2):
            raise RuntimeError('WAL ledger readback is not qualified under storage admission')

    @contextmanager
    def _connect(self, *, read_only: bool = False, estimated_growth_bytes: int = 0):
        if read_only:
            if not self.path.is_file():
                raise RuntimeError("Read-only research ledger does not exist")
            self._validate_read_only_mode()
            uri = self.path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            with closing(conn):
                yield conn
            return
        if self.read_only:
            raise PermissionError("ResearchLedger is read-only")
        with admitted_sqlite(
            self.path,
            estimated_growth_bytes=estimated_growth_bytes,
            stage="research_ledger",
            row_factory=sqlite3.Row,
        ) as conn:
            yield conn


def _record(row: sqlite3.Row) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=str(row["attempt_id"]),
        signature=str(row["signature"]),
        status=str(row["status"]),
        payload=json.loads(row["payload_json"]),
    )


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _growth_estimate(*fields, overhead_bytes: int = 64 * 1024) -> int:
    """Bound transient SQLite growth from serialized payloads and bound text."""
    field_bytes = 0
    for value in fields:
        if value is None:
            continue
        if isinstance(value, bytes):
            field_bytes += len(value)
        elif isinstance(value, str):
            field_bytes += len(value.encode("utf-8"))
        else:
            field_bytes += len(str(value).encode("utf-8"))
    return max(1024 * 1024, 3 * field_bytes + overhead_bytes)


def _retry_values(diagnostic,delay_seconds,now,runtime):
    now=time.time() if now is None else now
    if (not isinstance(diagnostic,dict) or diagnostic.get('retryable') is not True or
        (runtime is not None and not isinstance(runtime,dict)) or
        any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)
            for x in (delay_seconds,now)) or not 0<delay_seconds<=3600):
        raise ValueError('Retryable diagnostic, runtime object and bounded finite backoff required')
    return now,json.dumps(diagnostic,sort_keys=True,allow_nan=False),json.dumps(runtime or {},sort_keys=True,allow_nan=False)


def _record_retry(conn,identity,retry_at,diagnostic,runtime):
    # This metadata namespace also holds non-trial operations; it never creates
    # an attempts row or changes a scientific recipe.
    conn.execute('INSERT INTO infrastructure_retries '
        '(attempt_id,retry_at_epoch,retry_count,diagnostic_json,active) VALUES (?,?,1,?,1) '
        'ON CONFLICT(attempt_id) DO UPDATE SET retry_at_epoch=excluded.retry_at_epoch, '
        'retry_count=infrastructure_retries.retry_count+1,diagnostic_json=excluded.diagnostic_json,active=1',
        (identity,retry_at,diagnostic))
    retry=conn.execute('SELECT retry_count FROM infrastructure_retries WHERE attempt_id=?',(identity,)).fetchone()[0]
    conn.execute('INSERT INTO infrastructure_events VALUES (?,?,?,?,?)',
                 (identity,retry,diagnostic,runtime,utc_now()))
    return retry
