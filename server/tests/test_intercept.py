"""Tests for intercept auto-routing: observability, local/cloud routes, exact passthrough."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ferry import session_state
from proxy import forward
from proxy.intercept import intercept_log
from registry.client import Candidate
from triage import resolver

# Fixed pool: local-simple -> qwen3 (free local), cloud-mid -> flash, cloud-heavy -> pro.
_POOL = [
    Candidate("ollama", "qwen3:8b", "local", reasoning=False, input_cost=0.0,
              context_window=None, parameter_size="8B", supports_effort=False),
    Candidate("google", "gemini-2.5-flash", "cloud", reasoning=True, input_cost=0.3,
              context_window=1_000_000, parameter_size=None, supports_effort=True),
    Candidate("google", "gemini-2.5-pro", "cloud", reasoning=True, input_cost=2.5,
              context_window=1_000_000, parameter_size=None, supports_effort=True),
]

# Text fixtures chosen to land deterministically in each classifier bucket.
EASY_BODY = b'{"model":"placeholder","messages":[{"role":"user","content":"fix a typo"}]}'
HARD_BODY = (
    b'{"model":"placeholder","messages":[{"role":"user","content":'
    b'"refactor the architecture to fix the race condition and optimize performance"}]}'
)
MEDIUM_TEXT = (
    "Please add a new REST endpoint that returns the current user's profile, "
    "including their display name and email, reading the values from the existing "
    "users table without changing any of the surrounding handlers."
)
MEDIUM_BODY = (
    b'{"model":"placeholder","messages":[{"role":"user","content":"' + MEDIUM_TEXT.encode() + b'"}]}'
)


@pytest.fixture
def client():
    return TestClient(main.app)


def _streamed(status, body: bytes, headers=None) -> httpx.Response:
    async def gen():
        yield body

    return httpx.Response(status, content=gen(), headers=headers or {})


def _wire(
    monkeypatch, *, google_key="g-key", captured=None, handler=None,
    ollama_base_url="http://ollama.test",
    google_base="https://google.test/v1beta/openai",
):
    """Point forward.py at a mock upstream + fake config, with a fixed resolver pool."""
    monkeypatch.setattr(resolver, "all_candidates", lambda: list(_POOL))
    monkeypatch.setattr(
        forward,
        "config",
        SimpleNamespace(
            ollama_base_url=ollama_base_url,
            google_openai_base_url=google_base,
            provider_api_keys=({"google": google_key} if google_key else {}),
            provider_base_urls={},
        ),
    )

    def default_handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return _streamed(200, b'{"ok": true}', {"content-type": "application/json"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler or default_handler))
    monkeypatch.setattr(forward, "_client", mock)


# --- interception observability ------------------------------------------------

def test_health_ping_reports_liveness(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_request_is_recorded_with_routing_decision(client, monkeypatch):
    _wire(monkeypatch)
    before = len(intercept_log)
    client.post("/v1/chat/completions", content=EASY_BODY,
                headers={"content-type": "application/json"})
    assert len(intercept_log) == before + 1
    entry = intercept_log[-1]
    assert entry["difficulty"] == "easy"
    assert entry["routed_to"] == "ollama/qwen3:8b"
    assert entry["requested_model"] == "placeholder"


# --- automatic routing by difficulty (the core of 5c) --------------------------

def test_easy_task_routes_to_ollama(client, monkeypatch):
    captured = []
    _wire(monkeypatch, captured=captured)
    client.post("/v1/chat/completions", content=EASY_BODY,
                headers={"content-type": "application/json", "authorization": "Bearer x"})
    sent = captured[0]
    # Local: ollama base + full path (no /v1 stripping), no auth, model rewritten.
    assert str(sent.url) == "http://ollama.test/v1/chat/completions"
    assert "authorization" not in sent.headers
    assert "x-api-key" not in sent.headers
    import json
    assert json.loads(sent.content)["model"] == "qwen3:8b"


def test_hard_task_routes_to_google(client, monkeypatch):
    captured = []
    _wire(monkeypatch, google_key="g-REAL", captured=captured)
    client.post("/v1/chat/completions", content=HARD_BODY,
                headers={"content-type": "application/json"})
    sent = captured[0]
    # Cloud: /v1 stripped, google base, bearer key injected, model rewritten.
    assert str(sent.url) == "https://google.test/v1beta/openai/chat/completions"
    assert sent.headers["authorization"] == "Bearer g-REAL"
    import json
    assert json.loads(sent.content)["model"] == "gemini-2.5-pro"


def test_medium_task_routes_to_google_flash(client, monkeypatch):
    captured = []
    _wire(monkeypatch, captured=captured)
    client.post("/v1/chat/completions", content=MEDIUM_BODY,
                headers={"content-type": "application/json"})
    sent = captured[0]
    import json
    assert json.loads(sent.content)["model"] == "gemini-2.5-flash"


# --- faithful response forwarding ----------------------------------------------

def test_streaming_response_passes_through_exactly(client, monkeypatch):
    sse = (
        b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _streamed(
            200, sse,
            {"content-type": "text/event-stream", "x-upstream-marker": "kept"},
        )

    _wire(monkeypatch, handler=handler)
    resp = client.post("/v1/chat/completions", content=EASY_BODY)
    assert resp.status_code == 200
    assert resp.content == sse
    assert resp.headers["content-type"] == "text/event-stream"
    assert resp.headers["x-upstream-marker"] == "kept"


def test_upstream_status_is_preserved(client, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _streamed(429, b'{"type": "error"}', {"content-type": "application/json"})

    _wire(monkeypatch, handler=handler)
    resp = client.post("/v1/chat/completions", content=EASY_BODY)
    assert resp.status_code == 429


def test_missing_cloud_key_fails_loud(client, monkeypatch):
    # Hard task -> google, but no google key configured.
    _wire(monkeypatch, google_key=None)
    resp = client.post("/v1/chat/completions", content=HARD_BODY)
    assert resp.status_code == 503
    assert "google" in resp.json()["error"]


# --- Ferry integration: switch detection reshapes the outgoing context ----------

# A multi-turn "easy" conversation: routes to the local model, and is long enough that a
# local-simple strip is observable (it collapses to just the final user turn).
MULTI_EASY_MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "fix a typo"},
    {"role": "assistant", "content": "which file?"},
    {"role": "user", "content": "fix a typo in the readme"},
]
MULTI_EASY_BODY = json.dumps(
    {"model": "placeholder", "messages": MULTI_EASY_MESSAGES}
).encode()


@pytest.fixture
def ferry_state(tmp_path):
    """Initialize Ferry's session-state store (the lifespan doesn't run under TestClient)."""
    session_state.init(tmp_path / "ferry.db")
    yield
    session_state.close()


def _last_sent_messages(captured):
    return json.loads(captured[-1].content)["messages"]


def test_switch_reshapes_context_to_the_new_tier(client, monkeypatch, ferry_state):
    captured = []
    _wire(monkeypatch, google_key="g-REAL", captured=captured)
    sid = {"x-session-id": "ses_switch", "content-type": "application/json"}

    # Turn 1: a hard task -> cloud-heavy (google/pro). Establishes the session's model.
    client.post("/v1/chat/completions", content=HARD_BODY, headers=sid)
    assert json.loads(captured[-1].content)["model"] == "gemini-2.5-pro"

    # Turn 2: an easy multi-turn task on the SAME session -> local model (a genuine switch).
    # Ferry fires and strips the payload to the local-simple minimal form: last user turn only.
    client.post("/v1/chat/completions", content=MULTI_EASY_BODY, headers=sid)
    assert json.loads(captured[-1].content)["model"] == "qwen3:8b"
    assert _last_sent_messages(captured) == [
        {"role": "user", "content": "fix a typo in the readme"}
    ]


def test_no_switch_leaves_context_untouched(client, monkeypatch, ferry_state):
    captured = []
    _wire(monkeypatch, captured=captured)
    sid = {"x-session-id": "ses_stable", "content-type": "application/json"}

    # Turn 1: new session -> treated as a switch, so the payload is stripped.
    client.post("/v1/chat/completions", content=MULTI_EASY_BODY, headers=sid)
    assert _last_sent_messages(captured) == [
        {"role": "user", "content": "fix a typo in the readme"}
    ]

    # Turn 2: same session, same resolved model -> not a switch -> full history forwarded intact.
    client.post("/v1/chat/completions", content=MULTI_EASY_BODY, headers=sid)
    assert _last_sent_messages(captured) == MULTI_EASY_MESSAGES
