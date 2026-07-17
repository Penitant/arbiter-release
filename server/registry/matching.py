"""Cross-catalog model matching: alias table first, then exact normalized match, else None."""

from __future__ import annotations

from registry import aliases, openrouter
from registry.client import Candidate

# Noise stripped during normalization; size-bearing tags like `:7b` are kept.
_NOISE_SUFFIXES = ("-instruct", "-it", "-chat", "-base", "-bf16", "-fp16", "-fp8")
_NOISE_TAGS = frozenset({"latest", "free"})
_SEPARATORS = "-_. "


def normalize(name: str) -> str:
    """Reduce a catalog identifier to a comparable family+size key (e.g. `qwen25coder7b`)."""
    s = name.lower().strip()
    if "/" in s:  # drop a leading provider segment (qwen/..., Qwen/..., google/...)
        s = s.split("/", 1)[1]
    # `:` variants: keep size-bearing tags (`:7b`), drop pure noise (`:latest`, `:free`).
    parts = s.split(":")
    s = "-".join([parts[0]] + [p for p in parts[1:] if p not in _NOISE_TAGS])
    changed = True
    while changed:  # strip stacked noise suffixes (e.g. "...-instruct-fp16")
        changed = False
        for suf in _NOISE_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    for ch in _SEPARATORS:
        s = s.replace(ch, "")
    return s


# Lookup memoized against the entries tuple identity; a refresh invalidates it transparently.
_cache_key: object | None = None
_lookup: dict[str, float] = {}


def _index() -> dict[str, float]:
    global _cache_key, _lookup
    entries = openrouter.entries()
    if entries is _cache_key:
        return _lookup
    lookup: dict[str, float] = {}
    for e in entries:
        for raw in (e.id, e.hugging_face_id):
            if not raw:
                continue
            key = normalize(raw)
            if key:  # keep the best score on a normalized collision
                lookup[key] = max(lookup.get(key, e.coding_index), e.coding_index)
    _cache_key, _lookup = entries, lookup
    return lookup


def get_benchmark_score(candidate: Candidate) -> float | None:
    """Candidate's coding_index via alias table then normalized match; None if unmatched."""
    lookup = _index()
    alias_or = aliases.openrouter_id_for(candidate.provider, candidate.id)
    if alias_or is not None:
        score = lookup.get(normalize(alias_or))
        if score is not None:
            return score
    return lookup.get(normalize(candidate.id))


def unmatched(candidates: list[Candidate]) -> list[Candidate]:
    """Candidates with no benchmark match (surfaced by the startup diagnostic)."""
    return [c for c in candidates if get_benchmark_score(c) is None]
