"""Capability-floor filter + per-tier ranking (benchmark, then cost, then parameter size)."""

from __future__ import annotations

from config import TIER_CAPABILITY_REQUIREMENTS
from registry import matching
from registry.client import Candidate


def meets_capability_floor(candidate: Candidate, requirements: dict[str, object]) -> bool:
    """Apply the tier's capability gate (currently only the reasoning requirement)."""
    if requirements.get("reasoning_required") and not candidate.reasoning:
        return False
    return True


def is_tool_safe(model: Candidate) -> bool:
    """False for Google reasoning models (unreliable tool calls), True otherwise."""
    return not (model.provider == "google" and model.reasoning)


def parse_param_size(value: str | None) -> float:
    """Ollama `parameter_size` -> billions ("8.2B" -> 8.2, "560M" -> 0.56, unparseable -> 0.0)."""
    if not value:
        return 0.0
    text = value.strip().upper()
    unit = 1.0
    if text.endswith("B"):
        text = text[:-1]
    elif text.endswith("M"):
        text, unit = text[:-1], 0.001
    try:
        return float(text) * unit
    except ValueError:
        return 0.0


# A missing cost is least preferred: +inf ascending, -inf descending.
def _cost_asc(c: Candidate) -> float:
    return c.input_cost if c.input_cost is not None else float("inf")


def _cost_desc(c: Candidate) -> float:
    return c.input_cost if c.input_cost is not None else float("-inf")


def rank_candidates(
    candidates: list[Candidate], tier: str, *, score_of=None
) -> list[Candidate]:
    """Order candidates best-first for `tier`; `score_of` is injectable for tests."""
    if score_of is None:
        score_of = matching.get_benchmark_score
    rank_by = TIER_CAPABILITY_REQUIREMENTS[tier]["rank_by"]
    scored = [(c, score_of(c)) for c in candidates]

    if rank_by == "most_capable":
        # cloud-heavy: benchmarked (score desc) strictly above unbenchmarked (cost/size desc).
        benchmarked = sorted(
            (c for c, s in scored if s is not None),
            key=lambda c: (score_of(c), parse_param_size(c.parameter_size)),
            reverse=True,
        )
        unbenchmarked = sorted(
            (c for c, s in scored if s is None),
            key=lambda c: (_cost_desc(c), parse_param_size(c.parameter_size)),
            reverse=True,
        )
        return benchmarked + unbenchmarked

    # cloud-mid / local-simple: cost-first; benchmark then parameter size break ties.
    def key(c: Candidate):
        s = score_of(c)
        return (
            _cost_asc(c),
            -(s if s is not None else float("-inf")),
            -parse_param_size(c.parameter_size),
        )

    return sorted(candidates, key=key)
