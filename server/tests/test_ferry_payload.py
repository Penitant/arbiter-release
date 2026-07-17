"""Ferry trigger + payload tests: switch detection and tier-scaled context shaping."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from ferry import payload, session_state
from ferry.switch_detector import check_switch


# --- switch detection (trigger 1 + brand-new session) --------------------------

@pytest.fixture
def state(tmp_path):
    """A freshly-initialized, isolated session_model_state store for one test."""
    session_state.init(tmp_path / "ferry.db")
    yield
    session_state.close()


def test_switch_fires_when_model_changes_on_same_session(state):
    sid = "ses_abc"
    check_switch(sid, "ollama/qwen3:8b")          # establish the session's current model
    assert check_switch(sid, "google/gemini-2.5-pro") is True


def test_no_switch_when_model_is_unchanged(state):
    sid = "ses_abc"
    check_switch(sid, "ollama/qwen3:8b")
    assert check_switch(sid, "ollama/qwen3:8b") is False


def test_brand_new_session_fires(state):
    # No prior row: first request on a never-seen session counts as a switch by design.
    assert check_switch("ses_never_seen", "ollama/qwen3:8b") is True


# --- tier-scaled payload shaping -----------------------------------------------

_CONVO = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "add a login endpoint"},
    {"role": "assistant", "content": "done; wrote handlers.py"},
    {"role": "user", "content": "now add logout too"},
]


def test_local_simple_strips_to_last_user_message():
    out = asyncio.run(payload.build_payload(_CONVO, "local-simple", is_switch=True))
    assert out == [{"role": "user", "content": "now add logout too"}]


def test_cloud_heavy_is_byte_identical_to_input():
    out = asyncio.run(payload.build_payload(_CONVO, "cloud-heavy", is_switch=True))
    assert out is _CONVO  # same object → forwarded bytes are unchanged
    assert json.dumps(out) == json.dumps(_CONVO)


def test_cloud_mid_is_a_single_system_message(monkeypatch):
    monkeypatch.setattr(payload, "summarize", AsyncMock(return_value="STRUCTURED SUMMARY"))
    out = asyncio.run(payload.build_payload(_CONVO, "cloud-mid", is_switch=True))
    assert out == [{"role": "system", "content": "STRUCTURED SUMMARY"}]


def test_cloud_mid_single_turn_has_nothing_to_hand_off(monkeypatch):
    # A session that is only the current turn has no prior history: pass it through untouched
    # rather than invoking the summary writer on a single message.
    summarize = AsyncMock(return_value="x")
    monkeypatch.setattr(payload, "summarize", summarize)
    single = [{"role": "user", "content": "just this"}]
    out = asyncio.run(payload.build_payload(single, "cloud-mid", is_switch=True))
    assert out is single
    summarize.assert_not_awaited()


@pytest.mark.parametrize("tier", ["local-simple", "cloud-mid", "cloud-heavy"])
def test_not_a_switch_returns_messages_untouched(tier, monkeypatch):
    # is_switch=False short-circuits before any tier logic — even the summarizer seam.
    summarize = AsyncMock(side_effect=AssertionError("must not summarize"))
    monkeypatch.setattr(payload, "summarize", summarize)
    out = asyncio.run(payload.build_payload(_CONVO, tier, is_switch=False))
    assert out is _CONVO
    summarize.assert_not_awaited()
