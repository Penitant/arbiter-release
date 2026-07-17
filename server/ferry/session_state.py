"""Durable `session_model_state` store — Ferry's switch-detection state.

Single source of truth for "which model handled a session's most recent turn." Shares
arbiter.db with the correction log but in its own table (transient state, not an audit log).
One row per session; the 24h TTL sweep runs once at Server startup, not as a background job.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Own connection to arbiter.db; `_lock` serializes concurrent request-driven upserts.
_conn: sqlite3.Connection | None = None
_db_path: Path | None = None
_lock = threading.Lock()


def init(db_path: str | Path) -> None:
    """Open (creating if absent) the session-state db and apply the schema. Idempotent."""
    global _conn, _db_path
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    with _lock:
        # Close any prior handle (e.g. re-init under test) before swapping it in.
        if _conn is not None:
            _conn.close()
        _conn = conn
        _db_path = path


def close() -> None:
    """Close the connection (shutdown / test teardown)."""
    global _conn, _db_path
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _db_path = None


def get_last_model(session_id: str) -> str | None:
    """Return the model that handled this session's most recent turn, or None if unseen."""
    if _conn is None:
        raise RuntimeError("ferry.session_state.init() must run before get_last_model()")
    with _lock:
        row = _conn.execute(
            "SELECT last_model FROM session_model_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row[0] if row else None


def set_last_model(
    session_id: str, last_model: str, parent_session: str | None = None
) -> None:
    """Upsert this session's most-recent-model row, stamping updated_at (ISO 8601, UTC)."""
    if _conn is None:
        raise RuntimeError("ferry.session_state.init() must run before set_last_model()")
    updated_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        _conn.execute(
            """
            INSERT INTO session_model_state (session_id, last_model, parent_session, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_model     = excluded.last_model,
                parent_session = excluded.parent_session,
                updated_at     = excluded.updated_at
            """,
            (session_id, last_model, parent_session, updated_at),
        )
        _conn.commit()


def sweep_expired(ttl_hours: int = 24) -> int:
    """Delete rows older than the TTL (default 24h), returning how many were removed.

    Run once at Server startup, not as a background job. `datetime(updated_at)` normalizes the
    stored ISO 8601 strings to SQLite's UTC format so the comparison against `datetime('now')`
    isn't a raw lexical compare (the stored 'T' would otherwise sort after a space-separated now).
    """
    if _conn is None:
        raise RuntimeError("ferry.session_state.init() must run before sweep_expired()")
    with _lock:
        cur = _conn.execute(
            "DELETE FROM session_model_state "
            "WHERE datetime(updated_at) < datetime('now', ?)",
            (f"-{ttl_hours} hours",),
        )
        _conn.commit()
        return cur.rowcount


def count() -> int:
    """Total active session rows — a cheap validation/observability hook."""
    if _conn is None:
        return 0
    with _lock:
        return int(_conn.execute("SELECT COUNT(*) FROM session_model_state").fetchone()[0])
