"""SQLite correction-log store: the write side of the correction loop."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# One shared connection; `_lock` serializes writes from concurrent /corrections calls.
_conn: sqlite3.Connection | None = None
_db_path: Path | None = None
_lock = threading.Lock()

# Tier vocabulary, validated on write so a malformed payload can't poison the training set.
_VALID_TIERS = {"local-simple", "cloud-mid", "cloud-heavy"}


def init(db_path: str | Path) -> None:
    """Open (creating if absent) the correction db and apply the schema. Idempotent."""
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


def record(
    *,
    task_text: str,
    predicted_tier: str,
    predicted_model: str,
    confidence: float,
    session_id: str,
    corrected_tier: str | None = None,
    corrected_model: str | None = None,
) -> int:
    """Insert one correction row (corrected_* NULL = accepted as-is), returning its id."""
    if _conn is None:
        raise RuntimeError("corrections.store.init() must run before record()")
    if predicted_tier not in _VALID_TIERS:
        raise ValueError(f"unknown predicted_tier: {predicted_tier!r}")
    if corrected_tier is not None and corrected_tier not in _VALID_TIERS:
        raise ValueError(f"unknown corrected_tier: {corrected_tier!r}")

    created_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        cur = _conn.execute(
            """
            INSERT INTO corrections (
                task_text, predicted_tier, predicted_model, confidence,
                corrected_tier, corrected_model, session_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_text,
                predicted_tier,
                predicted_model,
                confidence,
                corrected_tier,
                corrected_model,
                session_id,
                created_at,
            ),
        )
        _conn.commit()
        return int(cur.lastrowid)


def count() -> int:
    """Total rows logged — a cheap validation/observability hook."""
    if _conn is None:
        return 0
    with _lock:
        return int(_conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0])
