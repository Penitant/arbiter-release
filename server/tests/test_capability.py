"""Tests for the capability floor + ranking (pure functions, no I/O)."""

from config import TIER_CAPABILITY_REQUIREMENTS
from registry import capability
from registry.client import Candidate


def _c(name, cost, reasoning=True):
    return Candidate(
        provider="p", id=name, source="cloud", reasoning=reasoning,
        input_cost=cost, context_window=None, parameter_size=None, supports_effort=False,
    )


def _local(name, param):
    return Candidate(
        provider="ollama", id=name, source="local", reasoning=False,
        input_cost=0.0, context_window=None, parameter_size=param, supports_effort=False,
    )


def _scores(mapping):
    """A score_of stub: candidate id -> benchmark score (or None)."""
    return lambda c: mapping.get(c.id)


# --- meets_capability_floor: the reasoning gate --------------------------------

def test_reasoning_required_filters_non_reasoning():
    req = TIER_CAPABILITY_REQUIREMENTS["cloud-mid"]
    assert capability.meets_capability_floor(_c("a", 1.0, reasoning=True), req) is True
    assert capability.meets_capability_floor(_c("b", 1.0, reasoning=False), req) is False


def test_local_simple_admits_non_reasoning():
    req = TIER_CAPABILITY_REQUIREMENTS["local-simple"]
    assert capability.meets_capability_floor(_c("b", 1.0, reasoning=False), req) is True


# --- rank_candidates: direction per tier ---------------------------------------

def test_cheapest_first_for_cloud_mid():
    ranked = capability.rank_candidates([_c("pricey", 5.0), _c("cheap", 0.5)], "cloud-mid")
    assert [c.id for c in ranked] == ["cheap", "pricey"]


def test_most_expensive_first_for_cloud_heavy():
    ranked = capability.rank_candidates([_c("cheap", 0.5), _c("pricey", 5.0)], "cloud-heavy")
    assert [c.id for c in ranked] == ["pricey", "cheap"]


def test_unknown_cost_sorts_last_in_both_directions():
    pool = [_c("known", 1.0), _c("unknown", None)]
    assert [c.id for c in capability.rank_candidates(pool, "cloud-mid")][-1] == "unknown"
    assert [c.id for c in capability.rank_candidates(pool, "cloud-heavy")][-1] == "unknown"


def test_tie_break_is_stable_input_order():
    # Equal cost, no benchmarks -> original order preserved (deterministic).
    pool = [_c("first", 1.0), _c("second", 1.0)]
    none = lambda c: None
    assert [c.id for c in capability.rank_candidates(pool, "cloud-mid", score_of=none)] == ["first", "second"]
    assert [c.id for c in capability.rank_candidates(pool, "cloud-heavy", score_of=none)] == ["first", "second"]


# --- 6d: benchmark-first ranking (cloud-heavy) ---------------------------------

def test_cloud_heavy_ranks_by_benchmark_not_cost():
    # Cheaper but higher coding_index wins cloud-heavy: real score beats cost proxy.
    pool = [_c("pricey", 9.0), _c("cheap-but-smart", 0.5)]
    ranked = capability.rank_candidates(
        pool, "cloud-heavy", score_of=_scores({"cheap-but-smart": 80.0, "pricey": 60.0}))
    assert [c.id for c in ranked] == ["cheap-but-smart", "pricey"]


def test_cloud_heavy_prefers_benchmarked_over_unbenchmarked():
    # An unbenchmarked model never outranks a benchmarked one, even if pricier.
    pool = [_c("expensive-unscored", 50.0), _c("scored", 1.0)]
    ranked = capability.rank_candidates(
        pool, "cloud-heavy", score_of=_scores({"scored": 40.0}))
    assert [c.id for c in ranked] == ["scored", "expensive-unscored"]


# --- cloud-mid stays cost-first, benchmark only breaks ties --------------------

def test_cloud_mid_is_cost_first_despite_benchmark():
    # Cloud-mid stays cost-first even when the pricier model is better-benchmarked.
    pool = [_c("cheap", 0.3), _c("pricey-smart", 2.5)]
    ranked = capability.rank_candidates(
        pool, "cloud-mid", score_of=_scores({"pricey-smart": 90.0, "cheap": 50.0}))
    assert ranked[0].id == "cheap"


def test_cloud_mid_benchmark_breaks_cost_ties():
    pool = [_c("a", 1.0), _c("b", 1.0)]
    ranked = capability.rank_candidates(pool, "cloud-mid", score_of=_scores({"b": 70.0}))
    assert ranked[0].id == "b"  # equal cost, b is benchmarked


# --- 6e: parameter size differentiates free local models -----------------------

def test_param_size_parsing():
    assert capability.parse_param_size("8.2B") == 8.2
    assert capability.parse_param_size("560M") == 0.56
    assert capability.parse_param_size(None) == 0.0
    assert capability.parse_param_size("garbage") == 0.0


def test_local_simple_uses_param_size_when_unbenchmarked():
    # Two free local models, neither benchmarked -> the larger one is preferred (6e).
    pool = [_local("small:3b", "3B"), _local("big:14b", "14B")]
    ranked = capability.rank_candidates(pool, "local-simple", score_of=lambda c: None)
    assert [c.id for c in ranked] == ["big:14b", "small:3b"]


def test_local_simple_benchmark_beats_size():
    # A benchmarked small model beats a larger unbenchmarked one (score > size).
    pool = [_local("small:3b", "3B"), _local("big:14b", "14B")]
    ranked = capability.rank_candidates(
        pool, "local-simple", score_of=_scores({"small:3b": 55.0}))
    assert ranked[0].id == "small:3b"
