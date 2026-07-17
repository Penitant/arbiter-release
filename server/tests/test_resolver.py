"""Tests for resolve() over a monkeypatched pool: floors, ranking, effort, fallbacks."""

import pytest

from registry.client import Candidate
from triage import resolver


def _pool(monkeypatch, candidates):
    monkeypatch.setattr(resolver, "all_candidates", lambda: candidates)


def _effort(monkeypatch, supported: bool):
    monkeypatch.setattr(
        resolver.discovery, "supports_reasoning_effort", lambda *a, **k: supported
    )


def _local(name, **kw):
    kw.setdefault("input_cost", 0.0)
    return Candidate(
        provider="ollama", id=name, source="local", reasoning=kw.pop("reasoning", False),
        input_cost=kw.pop("input_cost"), context_window=None,
        parameter_size=kw.pop("parameter_size", "7B"), supports_effort=kw.pop("supports_effort", False),
    )


def _cloud(provider, name, cost, reasoning=True, supports_effort=True):
    return Candidate(
        provider=provider, id=name, source="cloud", reasoning=reasoning,
        input_cost=cost, context_window=200_000, parameter_size=None,
        supports_effort=supports_effort,
    )


# --- per-tier capability floor + ranking direction -----------------------------

def test_easy_picks_cheapest_and_prefers_free_local(monkeypatch):
    # local-simple: no reasoning required, cheapest. Local cost 0.0 beats any cloud cost.
    _pool(monkeypatch, [_cloud("openai", "gpt-x", 1.0), _local("qwen3:8b")])
    _effort(monkeypatch, False)
    d = resolver.resolve("easy")
    assert d.tier == "local-simple"
    assert d.model == "ollama/qwen3:8b"


def test_medium_picks_cheapest_reasoning_model(monkeypatch):
    # cloud-mid: cheapest among reasoning-capable; non-reasoning and local are filtered out.
    _pool(monkeypatch, [
        _cloud("google", "pro", 2.5),
        _cloud("google", "flash", 0.3),
        _cloud("openai", "cheap-noreason", 0.01, reasoning=False),
        _local("qwen3:8b"),
    ])
    _effort(monkeypatch, True)
    d = resolver.resolve("medium")
    assert d.tier == "cloud-mid"
    assert d.model == "google/flash"
    assert d.reasoning_effort == "medium"


def test_hard_picks_most_expensive_reasoning_model(monkeypatch):
    # cloud-heavy: reasoning required, most_capable == most expensive in the pool.
    _pool(monkeypatch, [
        _cloud("google", "flash", 0.3),
        _cloud("openai", "frontier", 9.0),
        _cloud("google", "pro", 2.5),
    ])
    _effort(monkeypatch, True)
    d = resolver.resolve("hard")
    assert d.tier == "cloud-heavy"
    assert d.model == "openai/frontier"
    assert d.reasoning_effort == "high"


# --- reasoning-effort is dropped when the chosen model doesn't expose the dial --

def test_effort_dropped_when_model_does_not_support_it(monkeypatch):
    _pool(monkeypatch, [_cloud("groq", "r1", 0.5)])
    _effort(monkeypatch, False)
    d = resolver.resolve("medium")
    assert d.reasoning_effort is None  # silently dropped, never errors


# --- fallbacks: relax capability floor, then 503 -------------------------------

def test_relaxes_capability_when_no_reasoning_model_available(monkeypatch):
    # No reasoning-capable model exists -> relax to the cheapest non-reasoning one.
    _pool(monkeypatch, [_cloud("openai", "a", 1.0, reasoning=False),
                        _cloud("openai", "b", 0.2, reasoning=False)])
    _effort(monkeypatch, True)
    d = resolver.resolve("medium")
    assert d.model == "openai/b"           # cheapest, since relaxed
    assert "relaxed" in d.reason


def test_no_candidates_raises(monkeypatch):
    _pool(monkeypatch, [])
    with pytest.raises(resolver.NoCandidateError):
        resolver.resolve("hard")


# --- 6d: benchmark drives cloud-heavy end-to-end -------------------------------

def test_hard_picks_highest_benchmark_over_highest_cost(monkeypatch):
    # cloud-heavy: best-benchmarked wins even when not the most expensive; reason names the score.
    _pool(monkeypatch, [_cloud("openai", "pricey", 9.0), _cloud("google", "smart", 2.0)])
    _effort(monkeypatch, True)
    monkeypatch.setattr(resolver.matching, "get_benchmark_score",
                        lambda c: {"smart": 82.0, "pricey": 55.0}.get(c.id))
    d = resolver.resolve("hard")
    assert d.model == "google/smart"
    assert "82" in d.reason and "coding_index" in d.reason


def test_local_simple_falls_back_to_cloud_when_no_local(monkeypatch):
    # No local model installed: local-simple (no reasoning req) takes the cheapest cloud.
    _pool(monkeypatch, [_cloud("openai", "mini", 0.15, reasoning=False),
                        _cloud("openai", "big", 3.0)])
    _effort(monkeypatch, False)
    d = resolver.resolve("easy")
    assert d.model == "openai/mini"
