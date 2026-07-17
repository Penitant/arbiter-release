"""Tests for provider validation, Ollama polling, and the merged pool (mocked, no network)."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from registry import discovery
from registry.discovery import (
    AvailableModel,
    LocalDiscovery,
    LocalModel,
    ProviderState,
    _Snapshot,
)


def _wire(monkeypatch, handler, *, api_keys=None, base_urls=None,
          ollama_base_url="http://ollama.test"):
    """Point discovery at a mock transport + a fake config."""
    monkeypatch.setattr(
        discovery,
        "config",
        SimpleNamespace(
            provider_api_keys=api_keys or {},
            provider_base_urls=base_urls or {},
            provider_key_sources={p: "env" for p in (api_keys or {})},
            ollama_base_url=ollama_base_url,
            discovery_interval=0.01,
            discovery_timeout=5,
        ),
    )
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(discovery, "_client", mock)
    discovery._capability_cache.clear()  # capabilities are cached across calls


# --- 4a: provider validation is a real auth call, not a non-empty check --------

def test_valid_key_authenticates_and_is_eligible(monkeypatch):
    def handler(request):
        assert request.headers["x-api-key"] == "good-key"  # key actually attached
        return httpx.Response(200, json={"data": []})

    _wire(monkeypatch, handler)
    status = asyncio.run(discovery._validate_provider("anthropic", "good-key"))
    assert status.state is ProviderState.VALIDATED
    assert status.eligible is True


def test_rejected_key_is_invalid_not_eligible(monkeypatch):
    _wire(monkeypatch, lambda r: httpx.Response(401, json={"error": "nope"}))
    status = asyncio.run(discovery._validate_provider("openai", "bad-key"))
    assert status.state is ProviderState.INVALID_KEY
    assert status.eligible is False


def test_server_error_is_unreachable(monkeypatch):
    _wire(monkeypatch, lambda r: httpx.Response(500))
    status = asyncio.run(discovery._validate_provider("groq", "k"))
    assert status.state is ProviderState.UNREACHABLE


def test_connection_failure_is_unreachable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    _wire(monkeypatch, handler)
    status = asyncio.run(discovery._validate_provider("mistral", "k"))
    assert status.state is ProviderState.UNREACHABLE


def test_unknown_provider_has_no_validator(monkeypatch):
    _wire(monkeypatch, lambda r: httpx.Response(200))
    status = asyncio.run(discovery._validate_provider("acme", "k"))
    assert status.state is ProviderState.NO_VALIDATOR
    assert status.eligible is False


def test_base_url_override_is_used(monkeypatch):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200)

    _wire(monkeypatch, handler, base_urls={"openai": "http://stand-in.test"})
    asyncio.run(discovery._validate_provider("openai", "k"))
    assert seen[0].startswith("http://stand-in.test/")


# --- 4a (summary): missing key vs invalid key, with google/gemini collapsed ----

def test_validate_reports_missing_and_invalid_distinctly(monkeypatch):
    def handler(request):
        if "api.anthropic.com" in str(request.url):
            return httpx.Response(200)
        return httpx.Response(401)  # openai key present but bad

    _wire(monkeypatch, handler, api_keys={"anthropic": "ok", "openai": "bad"})
    by_provider = {s.provider: s for s in asyncio.run(discovery.validate_cloud_providers())}

    assert by_provider["anthropic"].state is ProviderState.VALIDATED
    assert by_provider["openai"].state is ProviderState.INVALID_KEY
    assert by_provider["groq"].state is ProviderState.MISSING_KEY
    # Alias group collapsed: exactly one of google/gemini is reported as missing.
    present_aliases = {p for p in by_provider if p in {"google", "gemini"}}
    assert len(present_aliases) == 1


def test_configured_alias_marks_the_whole_group(monkeypatch):
    _wire(monkeypatch, lambda r: httpx.Response(200), api_keys={"gemini": "k"})
    by_provider = {s.provider: s for s in asyncio.run(discovery.validate_cloud_providers())}
    assert by_provider["gemini"].state is ProviderState.VALIDATED
    # google must NOT also be reported as missing once its alias is configured.
    assert "google" not in by_provider


# --- 4b: Ollama /api/tags discovery --------------------------------------------

def _ollama_handler(tags, capabilities):
    """Mock handler answering /api/tags and per-model /api/show capabilities."""
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": tags})
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            return httpx.Response(200, json={"capabilities": capabilities.get(name, [])})
        return httpx.Response(404)
    return handler


def test_local_models_are_parsed_from_tags(monkeypatch):
    tags = [
        {"name": "qwen2.5-coder:7b", "details": {"parameter_size": "7B", "family": "qwen2"}},
        {"name": "llama3.2:3b", "details": {"parameter_size": "3B"}},
        {"details": {}},  # nameless entry is skipped
    ]
    _wire(monkeypatch, _ollama_handler(tags, {}))
    result = asyncio.run(discovery.discover_local_models())
    assert result.reachable is True
    assert [m.name for m in result.models] == ["qwen2.5-coder:7b", "llama3.2:3b"]
    assert result.models[0].parameter_size == "7B"


def test_unreachable_ollama_is_not_an_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no ollama")

    _wire(monkeypatch, handler)
    result = asyncio.run(discovery.discover_local_models())
    assert result.reachable is False
    assert result.models == ()


# --- reasoning-effort capability detection -------------------------------------

def test_local_effort_support_comes_from_thinking_capability(monkeypatch):
    tags = [
        {"name": "qwen3:8b", "details": {"parameter_size": "8.2B"}},
        {"name": "tinyllama:latest", "details": {"parameter_size": "1B"}},
    ]
    caps = {"qwen3:8b": ["completion", "tools", "thinking"],
            "tinyllama:latest": ["completion"]}
    _wire(monkeypatch, _ollama_handler(tags, caps))
    by_name = {m.name: m for m in asyncio.run(discovery.discover_local_models()).models}
    assert by_name["qwen3:8b"].supports_effort is True       # has "thinking"
    assert by_name["tinyllama:latest"].supports_effort is False


def test_supports_reasoning_effort_helper(monkeypatch):
    snap = _Snapshot(
        providers=(discovery.ProviderStatus("anthropic", ProviderState.VALIDATED, "ok"),
                   discovery.ProviderStatus("groq", ProviderState.VALIDATED, "ok")),
        local=LocalDiscovery(True, (LocalModel("qwen3:8b", "8.2B", "qwen3", True),
                                    LocalModel("gemma:2b", "2B", "gemma", False)), "2"),
    )
    monkeypatch.setattr(discovery, "_snapshot", snap)
    # Cloud: provider-level table.
    assert discovery.supports_reasoning_effort("anthropic") is True
    assert discovery.supports_reasoning_effort("groq") is False
    # Local: real per-model signal.
    assert discovery.supports_reasoning_effort("ollama", "qwen3:8b") is True
    assert discovery.supports_reasoning_effort("ollama", "gemma:2b") is False
    assert discovery.supports_reasoning_effort("ollama", "unknown") is False


# --- 4c: the single merged accessor --------------------------------------------

def test_get_available_models_merges_cloud_and_local(monkeypatch):
    snap = _Snapshot(
        providers=(
            discovery.ProviderStatus("anthropic", ProviderState.VALIDATED, "ok"),
            discovery.ProviderStatus("openai", ProviderState.INVALID_KEY, "bad"),
        ),
        local=LocalDiscovery(True, (LocalModel("qwen2.5-coder:7b", "7B", "qwen2"),), "1"),
    )
    monkeypatch.setattr(discovery, "_snapshot", snap)

    models = discovery.get_available_models()
    # Only the eligible cloud provider + the local model survive; anthropic has the effort dial.
    assert AvailableModel("anthropic", "cloud", supports_effort=True) in models
    assert all(m.provider != "openai" for m in models)
    local = [m for m in models if m.source == "local"]
    assert local == [AvailableModel("ollama", "local", "qwen2.5-coder:7b", "7B")]


# --- refresh end-to-end (4a+4b+4c+4d) ------------------------------------------

def test_refresh_populates_snapshot_and_logs(monkeypatch, capsys):
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "llama3.2:3b", "details": {"parameter_size": "3B"}}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(200)  # anthropic validates

    _wire(monkeypatch, handler, api_keys={"anthropic": "ok"})
    asyncio.run(discovery.refresh())

    models = discovery.get_available_models()
    assert AvailableModel("anthropic", "cloud", supports_effort=True) in models
    assert any(m.source == "local" and m.id == "llama3.2:3b" for m in models)

    out = capsys.readouterr().out
    assert "Cloud provider validation" in out
    assert "Routing pool" in out
