"""Tests for POST /classify and POST /corrections (temp db per test, fixed candidate pool)."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

import main
from corrections import store
from registry.client import Candidate
from triage import resolver

# Same fixed pool as test_intercept: easy->local qwen3, medium->flash, hard->pro.
_POOL = [
    Candidate("ollama", "qwen3:8b", "local", reasoning=False, input_cost=0.0,
              context_window=None, parameter_size="8B", supports_effort=False),
    Candidate("google", "gemini-2.5-flash", "cloud", reasoning=True, input_cost=0.3,
              context_window=1_000_000, parameter_size=None, supports_effort=True),
    Candidate("google", "gemini-2.5-pro", "cloud", reasoning=True, input_cost=2.5,
              context_window=1_000_000, parameter_size=None, supports_effort=True),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "all_candidates", lambda: list(_POOL))
    db = tmp_path / "arbiter.db"
    store.init(db)
    yield TestClient(main.app), db
    store.close()


# --- POST /classify ------------------------------------------------------------

def test_classify_returns_contract_shape(client):
    c, _ = client
    resp = c.post("/classify", json={
        "task_text": "fix a typo in the readme",
        "session_id": "ses_1",
        "context_tokens_estimate": 100,
    })
    assert resp.status_code == 200
    body = resp.json()
    # Exactly the correction-log fields.
    assert set(body) == {"difficulty", "confidence", "tier", "model",
                         "reasoning_effort", "reason"}
    assert body["difficulty"] == "easy"
    assert body["tier"] == "local-simple"
    assert body["model"] == "ollama/qwen3:8b"


def test_classify_hard_task_reaches_for_capable_model(client):
    c, _ = client
    resp = c.post("/classify", json={
        "task_text": "refactor the architecture to fix the race condition and "
                     "optimize performance",
        "session_id": "ses_1",
    })
    body = resp.json()
    assert body["tier"] == "cloud-heavy"
    assert body["model"] == "google/gemini-2.5-pro"
    assert body["reasoning_effort"] == "high"


def test_classify_503_when_nothing_routable(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(resolver, "all_candidates", lambda: [])
    resp = c.post("/classify", json={"task_text": "anything", "session_id": "s"})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"


# --- POST /corrections ---------------------------------------------------------

def test_correction_accepted_as_is_is_logged(client):
    c, db = client
    resp = c.post("/corrections", json={
        "task_text": "implement JWT refresh logic",
        "predicted_tier": "cloud-mid",
        "predicted_model": "google/gemini-2.5-flash",
        "confidence": 0.82,
        "session_id": "ses_abc123",
    })
    assert resp.status_code == 204
    assert resp.content == b""
    row = sqlite3.connect(db).execute(
        "SELECT corrected_tier, corrected_model FROM corrections"
    ).fetchone()
    assert row == (None, None)  # accept-as-is stored with NULL corrections


def test_override_is_logged_with_corrected_fields(client):
    c, db = client
    c.post("/corrections", json={
        "task_text": "implement JWT refresh logic",
        "predicted_tier": "cloud-mid",
        "predicted_model": "google/gemini-2.5-flash",
        "confidence": 0.82,
        "corrected_tier": "cloud-heavy",
        "corrected_model": "google/gemini-2.5-pro",
        "session_id": "ses_abc123",
    })
    row = sqlite3.connect(db).execute(
        "SELECT predicted_model, corrected_tier, corrected_model FROM corrections"
    ).fetchone()
    assert row == ("google/gemini-2.5-flash", "cloud-heavy", "google/gemini-2.5-pro")


def test_bad_tier_is_rejected(client):
    c, _ = client
    resp = c.post("/corrections", json={
        "task_text": "x",
        "predicted_tier": "ultra-mega",  # not a real tier
        "predicted_model": "m",
        "confidence": 0.5,
        "session_id": "s",
    })
    assert resp.status_code == 400


def test_finetune_cursor_row_exists_after_init(client):
    _, db = client
    row = sqlite3.connect(db).execute(
        "SELECT id, last_processed_id FROM finetune_state"
    ).fetchone()
    assert row == (1, 0)
