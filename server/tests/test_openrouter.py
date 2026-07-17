"""Tests for the OpenRouter benchmark fetch/cache: only coding_index survives the parse."""

import asyncio
from types import SimpleNamespace

import httpx

from registry import openrouter

# A trimmed /api/v1/models: one AA-benchmarked, one design_arena-only, one unbenchmarked.
_API = {"data": [
    {"id": "z-ai/glm-5.2", "hugging_face_id": "zai-org/GLM-5.2",
     "benchmarks": {"artificial_analysis": {"intelligence_index": 51.1, "coding_index": 68.8,
                                            "agentic_index": 43.1}}},
    {"id": "foo/design-only", "hugging_face_id": None,
     "benchmarks": {"design_arena": [{"arena": "agents", "category": "x", "elo": 1300}]}},
    {"id": "bar/plain", "hugging_face_id": None},  # no benchmarks at all
]}


def _wire(monkeypatch, tmp_path, handler, *, url="https://openrouter.ai/api/v1/models"):
    monkeypatch.setattr(
        openrouter, "config",
        SimpleNamespace(openrouter_models_url=url, registry_cache_dir=str(tmp_path),
                        discovery_timeout=5, discovery_enabled=True),
    )
    monkeypatch.setattr(
        openrouter, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def test_parse_keeps_only_aa_coding_index():
    entries = openrouter._parse_benchmarks(_API)
    assert len(entries) == 1                       # design-only + plain dropped
    e = entries[0]
    assert e.id == "z-ai/glm-5.2"
    assert e.hugging_face_id == "zai-org/GLM-5.2"
    assert e.coding_index == 68.8


def test_parse_tolerates_garbage():
    assert openrouter._parse_benchmarks({"data": "nonsense"}) == ()
    assert openrouter._parse_benchmarks({"data": [{"id": 5}, "x", {}]}) == ()


def test_fetch_success_populates_and_caches(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, lambda r: httpx.Response(200, json=_API))
    snap = asyncio.run(openrouter.fetch_benchmarks())
    assert snap.source == "fetch"
    assert [e.id for e in snap.entries] == ["z-ai/glm-5.2"]
    assert (tmp_path / "openrouter.json").exists()


def test_fetch_failure_falls_back_to_cache(monkeypatch, tmp_path):
    (tmp_path / "openrouter.json").write_text(httpx.Response(200, json=_API).text)
    _wire(monkeypatch, tmp_path, lambda r: httpx.Response(503))
    snap = asyncio.run(openrouter.fetch_benchmarks())
    assert snap.source == "cache"
    assert snap.entries[0].coding_index == 68.8


def test_fetch_failure_without_cache_is_empty(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, lambda r: httpx.Response(503))
    snap = asyncio.run(openrouter.fetch_benchmarks())
    assert snap.source == "empty" and snap.entries == ()
