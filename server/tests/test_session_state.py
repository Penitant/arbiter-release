"""Ferry session_model_state store + switch detection over durable SQLite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ferry import session_state, switch_detector


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "arbiter.db"
    session_state.init(db)
    try:
        yield session_state
    finally:
        session_state.close()


def test_schema_creates_table(store):
    # Table exists and is empty on a fresh db.
    assert store.count() == 0
    assert store.get_last_model("ses_x") is None


def test_upsert_roundtrip(store):
    store.set_last_model("ses_a", "openai/gpt-5", parent_session=None)
    assert store.get_last_model("ses_a") == "openai/gpt-5"
    # Upsert on the same primary key replaces, never duplicates.
    store.set_last_model("ses_a", "ollama/qwen2.5-coder:7b", parent_session="ses_root")
    assert store.get_last_model("ses_a") == "ollama/qwen2.5-coder:7b"
    assert store.count() == 1


def test_sweep_deletes_only_expired(store):
    # Insert a stale row directly with an updated_at well past the TTL.
    stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    with session_state._lock:
        session_state._conn.execute(
            "INSERT INTO session_model_state (session_id, last_model, parent_session, updated_at)"
            " VALUES (?, ?, ?, ?)",
            ("ses_old", "m", None, stale),
        )
        session_state._conn.commit()
    store.set_last_model("ses_new", "m", None)  # fresh, within TTL

    removed = store.sweep_expired()

    assert removed == 1
    assert store.get_last_model("ses_old") is None
    assert store.get_last_model("ses_new") == "m"


def test_check_switch_new_session_is_switch(store):
    assert switch_detector.check_switch("ses_1", "openai/gpt-5") is True
    # Persisted durably after the first call.
    assert store.get_last_model("ses_1") == "openai/gpt-5"


def test_check_switch_same_model_no_switch(store):
    switch_detector.check_switch("ses_2", "openai/gpt-5")
    assert switch_detector.check_switch("ses_2", "openai/gpt-5") is False


def test_check_switch_model_change_is_switch(store):
    switch_detector.check_switch("ses_3", "openai/gpt-5")
    assert switch_detector.check_switch("ses_3", "ollama/gemma3:12b") is True
    assert store.get_last_model("ses_3") == "ollama/gemma3:12b"


def test_state_survives_reopen(tmp_path):
    db = tmp_path / "arbiter.db"
    session_state.init(db)
    session_state.set_last_model("ses_p", "openai/gpt-5", None)
    session_state.close()

    # Re-open the same file: durable, not in-memory.
    session_state.init(db)
    try:
        assert session_state.get_last_model("ses_p") == "openai/gpt-5"
    finally:
        session_state.close()
