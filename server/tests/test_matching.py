"""Tests for cross-catalog matching: names collapse to one key, non-exact returns None."""

import pytest

from registry import matching
from registry.client import Candidate
from registry.openrouter import BenchmarkEntry


def _bench(monkeypatch, entries):
    """Point the matcher at a fixed benchmark snapshot and reset its memoized index."""
    monkeypatch.setattr(matching.openrouter, "entries", lambda: tuple(entries))
    monkeypatch.setattr(matching, "_cache_key", None)
    monkeypatch.setattr(matching.aliases, "ALIASES", {})


def _local(id_):
    return Candidate("ollama", id_, "local", False, 0.0, None, "7B", False)


def _cloud(provider, id_):
    return Candidate(provider, id_, "cloud", True, 1.0, 200000, None, True)


# --- normalize: the three naming schemes collapse to one key -------------------

@pytest.mark.parametrize("name", [
    "qwen2.5-coder:7b",                  # Ollama tag
    "qwen/qwen2.5-coder-7b-instruct",    # OpenRouter slug
    "Qwen/Qwen2.5-Coder-7B-Instruct",    # Hugging Face id
])
def test_normalize_collapses_naming_schemes(name):
    assert matching.normalize(name) == "qwen25coder7b"


def test_normalize_keeps_size_tag_but_drops_noise_tags():
    assert matching.normalize("gemma:latest") == "gemma"
    assert matching.normalize("google/gemma-4-26b-a4b-it:free") == "gemma426ba4b"
    assert matching.normalize("llama3.2:3b") == "llama323b"


# --- get_benchmark_score: exact match across catalogs --------------------------

def test_ollama_tag_matches_openrouter_slug(monkeypatch):
    _bench(monkeypatch, [BenchmarkEntry("qwen/qwen2.5-coder-7b-instruct", None, 61.0)])
    assert matching.get_benchmark_score(_local("qwen2.5-coder:7b")) == 61.0


def test_matches_via_hugging_face_id(monkeypatch):
    _bench(monkeypatch, [BenchmarkEntry("zai/glm", "zai-org/GLM-5.2", 68.8)])
    # cloud candidate whose models.dev id aligns with the HF id, not the OR slug.
    assert matching.get_benchmark_score(_cloud("zai", "glm-5.2")) == 68.8


def test_no_match_returns_none(monkeypatch):
    _bench(monkeypatch, [BenchmarkEntry("qwen/qwen2.5-coder-7b-instruct", None, 61.0)])
    # A different model must NOT be scored (conservative: no fuzzy/substring match).
    assert matching.get_benchmark_score(_local("llama3.2:3b")) is None


def test_alias_table_takes_precedence(monkeypatch):
    _bench(monkeypatch, [BenchmarkEntry("vendor/canonical-slug", None, 55.5)])
    monkeypatch.setattr(matching.aliases, "ALIASES", {
        "x": matching.aliases.CanonicalAlias(
            ollama="weird-local:tag", openrouter="vendor/canonical-slug"),
    })
    monkeypatch.setattr(matching, "_cache_key", None)
    assert matching.get_benchmark_score(_local("weird-local:tag")) == 55.5


def test_unmatched_lists_only_unscored(monkeypatch):
    _bench(monkeypatch, [BenchmarkEntry("qwen/qwen2.5-coder-7b-instruct", None, 61.0)])
    pool = [_local("qwen2.5-coder:7b"), _local("llama3.2:3b")]
    assert [c.id for c in matching.unmatched(pool)] == ["llama3.2:3b"]
