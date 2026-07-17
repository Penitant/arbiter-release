"""Hand-maintained cross-catalog alias table for names normalization can't bridge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalAlias:
    """Known identifiers for one set of weights across the three catalogs."""

    ollama: str | None = None
    models_dev: str | None = None
    openrouter: str | None = None


# Keep small; fill only from real gaps surfaced by the startup diagnostic.
ALIASES: dict[str, CanonicalAlias] = {}


def openrouter_id_for(provider: str, model_id: str) -> str | None:
    """OpenRouter id for an aliased candidate, else None (ollama slot local, models_dev cloud)."""
    want_ollama = provider == "ollama"
    for alias in ALIASES.values():
        catalog_id = alias.ollama if want_ollama else alias.models_dev
        if catalog_id is not None and catalog_id == model_id and alias.openrouter:
            return alias.openrouter
    return None
