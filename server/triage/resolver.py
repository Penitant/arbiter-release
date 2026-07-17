"""Deterministic difficulty -> tier -> concrete model resolution over the live Registry."""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    CONFIDENCE_THRESHOLD,
    DIFFICULTY_TO_TIER,
    TIER_CAPABILITY_REQUIREMENTS,
    TIER_DEFAULT_EFFORT,
)
from registry import capability, discovery, matching
from registry.client import Candidate, all_candidates


@dataclass(frozen=True)
class Decision:
    difficulty: str
    tier: str
    provider: str
    model_id: str            # concrete id stamped onto the upstream request
    reasoning_effort: str | None
    reason: str

    @property
    def model(self) -> str:
        """`provider/id`, the form used in logs and decision headers."""
        return f"{self.provider}/{self.model_id}"


class NoCandidateError(RuntimeError):
    """Nothing routable for a tier even after relaxing the capability floor."""


def apply_confidence_escalation(difficulty: str, confidence: float) -> str:
    """Bump a low-confidence `easy` call up to `medium`; never escalate `medium`->`hard`."""
    if difficulty == "easy" and confidence < CONFIDENCE_THRESHOLD:
        return "medium"
    return difficulty


def resolve(
    difficulty: str, tools_present: bool = False, confidence: float | None = None
) -> Decision:
    """Map a classifier difficulty to a concrete routing decision."""
    if confidence is not None:
        difficulty = apply_confidence_escalation(difficulty, confidence)
    tier = DIFFICULTY_TO_TIER[difficulty]

    requirements = TIER_CAPABILITY_REQUIREMENTS[tier]
    pool = all_candidates()

    survivors = [c for c in pool if capability.meets_capability_floor(c, requirements)]
    if tools_present:
        survivors = [c for c in survivors if capability.is_tool_safe(c)]
    relaxed = False
    if not survivors and requirements.get("reasoning_required"):
        # Relax the capability filter rather than fail; log loudly.
        relaxed = True
        survivors = list(pool)
        print(
            f"[routing] WARNING: no reasoning-capable model available for {tier}; "
            f"falling back to a non-reasoning model (quality bar not met)."
        )

    if not survivors:
        raise NoCandidateError(
            f"no model available for tier '{tier}': no provider authenticated and no "
            f"local model installed (check the discovery summary)."
        )

    chosen = capability.rank_candidates(survivors, tier)[0]
    effort = _effort_for(tier, chosen)
    return Decision(
        difficulty=difficulty,
        tier=tier,
        provider=chosen.provider,
        model_id=chosen.id,
        reasoning_effort=effort,
        reason=_reason(tier, requirements, chosen, relaxed),
    )


def _effort_for(tier: str, chosen: Candidate) -> str | None:
    """Tier's default reasoning effort, or None when the model doesn't expose the dial."""
    effort = TIER_DEFAULT_EFFORT.get(tier)
    if effort and not discovery.supports_reasoning_effort(chosen.provider, chosen.id):
        return None
    return effort


def _reason(tier: str, requirements: dict[str, object], chosen: Candidate, relaxed: bool) -> str:
    floor = "" if relaxed else ("reasoning-capable " if requirements.get("reasoning_required") else "")
    suffix = " (capability filter relaxed — quality bar not met)" if relaxed else ""
    score = matching.get_benchmark_score(chosen)
    if requirements.get("rank_by") == "most_capable":
        # Select on the benchmark when present, else the cost proxy.
        basis = (
            f"highest-benchmarked (coding_index {score:g})"
            if score is not None
            else "most capable (cost proxy, unbenchmarked)"
        )
    else:
        # Cost-first tiers: benchmark noted only as informational.
        basis = "cheapest" + (f" (coding_index {score:g})" if score is not None else "")
    return f"{basis} {floor}{tier} model{suffix}"
