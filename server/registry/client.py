"""models.dev catalog client + the merged candidate pool the resolver ranks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from config import config
from proxy.forward import FORWARDABLE_PROVIDERS
from registry import discovery


# --- result types --------------------------------------------------------------

@dataclass(frozen=True)
class CatalogModel:
    """One models.dev model entry, reduced to the fields Half 2 ranks on."""

    provider: str
    id: str
    input_cost: float | None     # $ per million input tokens
    context_window: int | None
    reasoning: bool              # absent/false -> False


@dataclass(frozen=True)
class Candidate:
    """One routable option in the merged pool the resolver ranks."""

    provider: str
    id: str
    source: str                  # "cloud" | "local"
    reasoning: bool
    input_cost: float | None     # local models are free -> 0.0
    context_window: int | None
    parameter_size: str | None   # local only; last-resort ranking signal
    supports_effort: bool

    @property
    def model(self) -> str:
        return f"{self.provider}/{self.id}"


# --- mutable snapshot (atomic ref swap; reads need no lock in CPython) ----------

@dataclass(frozen=True)
class _Snapshot:
    by_provider: dict[str, tuple[CatalogModel, ...]]  # provider -> its catalog models
    source: str                                       # "fetch" | "cache" | "empty"
    detail: str


_snapshot = _Snapshot({}, "empty", "not yet fetched")

# Own client; models.dev 403s without a User-Agent. Monkeypatchable in tests.
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(config.discovery_timeout),
    follow_redirects=True,
    headers={"User-Agent": "arbiter-registry/0.1"},
)


# --- 6a: fetch + parse + cache -------------------------------------------------

def _parse_catalog(data: dict) -> dict[str, tuple[CatalogModel, ...]]:
    """models.dev shape -> {provider: (CatalogModel, ...)}; malformed entries are skipped."""
    out: dict[str, tuple[CatalogModel, ...]] = {}
    if not isinstance(data, dict):
        return out
    for provider, pdata in data.items():
        if not isinstance(pdata, dict):
            continue
        models = pdata.get("models")
        if not isinstance(models, dict):
            continue
        parsed: list[CatalogModel] = []
        for model_id, m in models.items():
            if not isinstance(m, dict):
                continue
            cost = m.get("cost") if isinstance(m.get("cost"), dict) else {}
            limit = m.get("limit") if isinstance(m.get("limit"), dict) else {}
            parsed.append(
                CatalogModel(
                    provider=provider,
                    id=str(m.get("id") or model_id),
                    input_cost=_as_float(cost.get("input")),
                    context_window=_as_int(limit.get("context")),
                    reasoning=bool(m.get("reasoning")),
                )
            )
        if parsed:
            out[provider] = tuple(parsed)
    return out


def _as_float(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _as_int(v: object) -> int | None:
    return int(v) if isinstance(v, (int, float)) else None


def _cache_path() -> Path:
    return Path(config.registry_cache_dir) / "models_dev.json"


def _write_cache(raw: str) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
    except OSError as exc:
        print(f"[registry] could not write catalog cache: {type(exc).__name__}: {exc}")


def _load_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return None


async def fetch_catalog() -> _Snapshot:
    """Fetch models.dev; on any failure fall back to the last-good on-disk cache."""
    try:
        resp = await _client.get(config.models_dev_url)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        cached = _load_cache()
        if cached is not None:
            by_provider = _parse_catalog(cached)
            return _Snapshot(
                by_provider, "cache",
                f"fetch failed ({type(exc).__name__}); served cached catalog",
            )
        return _Snapshot(
            {}, "empty",
            f"fetch failed ({type(exc).__name__}) and no cache available",
        )

    _write_cache(resp.text)
    by_provider = _parse_catalog(data)
    n = sum(len(v) for v in by_provider.values())
    return _Snapshot(by_provider, "fetch", f"{n} models across {len(by_provider)} providers")


# --- 6c: the merged candidate pool the resolver ranks --------------------------

def all_candidates() -> list[Candidate]:
    """Eligible providers x their catalog models, plus local models; forwardable-only."""
    catalog = _snapshot.by_provider
    out: list[Candidate] = []
    for av in discovery.get_available_models():
        if av.source == "local":
            out.append(
                Candidate(
                    provider="ollama", id=av.id or "", source="local",
                    reasoning=False, input_cost=0.0, context_window=None,
                    parameter_size=av.parameter_size, supports_effort=av.supports_effort,
                )
            )
        elif av.source == "cloud" and av.provider in FORWARDABLE_PROVIDERS:
            for cm in catalog.get(av.provider, ()):
                out.append(
                    Candidate(
                        provider=av.provider, id=cm.id, source="cloud",
                        reasoning=cm.reasoning, input_cost=cm.input_cost,
                        context_window=cm.context_window,
                        parameter_size=None, supports_effort=av.supports_effort,
                    )
                )
    return out


def catalog_size() -> int:
    return sum(len(v) for v in _snapshot.by_provider.values())


# --- refresh + lifecycle -------------------------------------------------------

async def refresh() -> None:
    global _snapshot
    _snapshot = await fetch_catalog()


async def start() -> None:
    """Fetch the catalog and log a summary, flagging eligible providers absent from it."""
    if not config.discovery_enabled:
        return
    await refresh()
    n = catalog_size()
    print(f"[registry] models.dev catalog: {_snapshot.detail} (source={_snapshot.source})")

    eligible_cloud = {
        av.provider
        for av in discovery.get_available_models()
        if av.source == "cloud" and av.provider in FORWARDABLE_PROVIDERS
    }
    unmatched = sorted(eligible_cloud - set(_snapshot.by_provider))
    if unmatched:
        print(
            f"[registry] WARNING: eligible provider(s) with no catalog entry "
            f"(no routable cloud models from them): {', '.join(unmatched)}"
        )
    if n == 0:
        print(
            "[registry] WARNING: catalog is empty — cloud resolution will fall back to "
            "local models only. Check network access to models.dev."
        )


async def stop() -> None:
    await _client.aclose()
