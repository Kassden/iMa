"""SQLite durability ledger for research optimizer attempts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    signature: str
    status: str
    payload: dict[str, Any]


class ResearchLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def reserve_attempt(self, signature: str, payload: dict[str, Any]) -> AttemptRecord:
        attempt_id = f"attempt-{signature}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO attempts
                (attempt_id, signature, status, payload_json, updated_at)
                VALUES (?, ?, 'reserved', ?, ?)
                """,
                (attempt_id, signature, _json(payload), utc_now()),
            )
            row = conn.execute(
                "SELECT attempt_id, signature, status, payload_json FROM attempts WHERE signature = ?",
                (signature,),
            ).fetchone()
        return _record(row)

    def mark_running(self, attempt_id: str) -> AttemptRecord:
        with self._connect() as conn:
            conn.execute(
                """UPDATE attempts SET status = 'running', updated_at = ?
                   WHERE attempt_id = ? AND status = 'reserved'""",
                (utc_now(), attempt_id),
            )
            row = conn.execute(
                "SELECT attempt_id, signature, status, payload_json FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown attempt_id: {attempt_id}")
        return _record(row)

    def complete_attempt(
        self,
        attempt_id: str,
        result: dict[str, Any],
        *,
        status: str = "completed",
    ) -> AttemptRecord:
        if status not in {"completed", "failed", "pruned"}:
            raise ValueError(f"Unsupported terminal status: {status}")
        with self._connect() as conn:
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
                    (status, _json(result), utc_now(), attempt_id),
                )
            row = conn.execute(
                "SELECT attempt_id, signature, status, payload_json FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        return _record(row)

    def pending_outbox(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
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
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE attempts
                SET uploaded_at = ?, remote_id = ?, updated_at = ?
                WHERE attempt_id = ?
                """,
                (utc_now(), remote_id, utc_now(), attempt_id),
            )

    def pending_tells(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
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
        with self._connect() as conn:
            conn.execute(
                "UPDATE attempts SET told_at = ?, updated_at = ? WHERE attempt_id = ?",
                (utc_now(), utc_now(), attempt_id),
            )

    def terminal_results(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT attempt_id, signature, status, payload_json, result_json
                   FROM attempts WHERE status IN ('completed', 'failed', 'pruned')
                   ORDER BY updated_at, attempt_id"""
            ).fetchall()
        return [{
            "attempt_id": row["attempt_id"],
            "signature": row["signature"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"] or "{}"),
        } for row in rows]

    def recover_running(self) -> int:
        """Return unsupervised running attempts to reserved on controller startup."""
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE attempts SET status = 'reserved', updated_at = ?
                   WHERE status = 'running'""",
                (utc_now(),),
            )
        return int(cursor.rowcount)

    def snapshot(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) count FROM attempts GROUP BY status").fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def _init(self) -> None:
        with self._connect() as conn:
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


def _record(row: sqlite3.Row) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=str(row["attempt_id"]),
        signature=str(row["signature"]),
        status=str(row["status"]),
        payload=json.loads(row["payload_json"]),
    )


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
