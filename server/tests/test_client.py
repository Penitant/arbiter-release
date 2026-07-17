"""Tests for the models.dev catalog fetch/cache and the merged candidate pool (mocked I/O)."""

import asyncio
from types import SimpleNamespace

import httpx

from registry import client, discovery
from registry.client import CatalogModel, Candidate
from registry.discovery import AvailableModel

# A trimmed models.dev /api.json: providers -> models -> {reasoning, cost, limit}.
_API_JSON = {
    "google": {"models": {
        "gemini-2.5-flash": {"id": "gemini-2.5-flash", "reasoning": True,
                             "cost": {"input": 0.3}, "limit": {"context": 1000000}},
        "gemini-2.5-pro": {"id": "gemini-2.5-pro", "reasoning": True,
                           "cost": {"input": 2.5}, "limit": {"context": 1000000}},
    }},
    "openai": {"models": {
        "gpt-mini": {"reasoning": False, "cost": {"input": 0.15}, "limit": {"context": 128000}},
    }},
    "bad-provider": {"no_models_key": True},  # skipped, doesn't blow up the parse
}


def _wire(monkeypatch, tmp_path, handler, *, url="https://models.dev/api.json"):
    monkeypatch.setattr(
        client, "config",
        SimpleNamespace(
            models_dev_url=url,
            registry_cache_dir=str(tmp_path),
            discovery_timeout=5,
            discovery_enabled=True,
        ),
    )
    monkeypatch.setattr(
        client, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


# --- _parse_catalog: defensive shape handling ----------------------------------

def test_parse_extracts_cost_context_reasoning():
    by_provider = client._parse_catalog(_API_JSON)
    assert set(by_provider) == {"google", "openai"}  # bad-provider skipped
    flash = next(m for m in by_provider["google"] if m.id == "gemini-2.5-flash")
    assert flash == CatalogModel("google", "gemini-2.5-flash", 0.3, 1000000, True)
    assert by_provider["openai"][0].reasoning is False


def test_parse_tolerates_missing_cost_and_reasoning():
    out = client._parse_catalog({"p": {"models": {"m": {}}}})
    assert out["p"][0] == CatalogModel("p", "m", None, None, False)


# --- fetch_catalog: success writes cache ---------------------------------------

def test_fetch_success_populates_and_caches(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, lambda r: httpx.Response(200, json=_API_JSON))
    snap = asyncio.run(client.fetch_catalog())
    assert snap.source == "fetch"
    assert "google" in snap.by_provider
    # last-good cache was written for a future offline restart.
    assert (tmp_path / "models_dev.json").exists()


def test_fetch_failure_falls_back_to_cache(monkeypatch, tmp_path):
    (tmp_path / "models_dev.json").write_text(httpx.Response(200, json=_API_JSON).text)
    _wire(monkeypatch, tmp_path, lambda r: httpx.Response(500))
    snap = asyncio.run(client.fetch_catalog())
    assert snap.source == "cache"
    assert "google" in snap.by_provider


def test_fetch_failure_without_cache_is_empty(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, lambda r: httpx.Response(500))
    snap = asyncio.run(client.fetch_catalog())
    assert snap.source == "empty"
    assert snap.by_provider == {}


# --- all_candidates: merge discovery availability with the catalog --------------

def test_all_candidates_expands_cloud_and_adds_local(monkeypatch):
    monkeypatch.setattr(discovery, "get_available_models", lambda: [
        AvailableModel(provider="google", source="cloud", supports_effort=True),
        AvailableModel(provider="anthropic", source="cloud"),  # eligible but NOT forwardable
        AvailableModel(provider="ollama", source="local", id="qwen3:8b", parameter_size="8B"),
    ])
    monkeypatch.setattr(client, "_snapshot", client._Snapshot(
        client._parse_catalog(_API_JSON), "fetch", "x"))

    cands = client.all_candidates()
    by_model = {c.model: c for c in cands}
    # Google expands to its catalog models; anthropic dropped; local model is free/non-reasoning.
    assert "google/gemini-2.5-pro" in by_model
    assert "google/gemini-2.5-flash" in by_model
    assert not any(c.provider == "anthropic" for c in cands)
    local = by_model["ollama/qwen3:8b"]
    assert local.source == "local" and local.input_cost == 0.0 and local.reasoning is False


def test_all_candidates_drops_forwardable_provider_absent_from_catalog(monkeypatch):
    # openai is forwardable & eligible, but the catalog has no entry for it here.
    monkeypatch.setattr(discovery, "get_available_models", lambda: [
        AvailableModel(provider="openai", source="cloud"),
    ])
    monkeypatch.setattr(client, "_snapshot", client._Snapshot({}, "empty", "x"))
    assert client.all_candidates() == []
