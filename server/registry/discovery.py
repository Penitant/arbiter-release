"""Provider & local-model discovery: what can Arbiter actually route to right now?"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

import httpx

from config import config


# --- provider validation specs -------------------------------------------------

@dataclass(frozen=True)
class _Validator:
    """How to make one cheap authenticated GET that proves a key works."""

    path: str            # appended to the provider base url
    auth_style: str      # "x-api-key" | "bearer" | "query"
    base_url: str        # default; overridable via {PROVIDER}_BASE_URL
    extra_headers: dict[str, str] = field(default_factory=dict)


# Cheap auth probes: 2xx = good key; 400/401/403 = bad key; anything else = unreachable.
_VALIDATORS: dict[str, _Validator] = {
    "anthropic": _Validator(
        "/v1/models", "x-api-key", "https://api.anthropic.com",
        {"anthropic-version": "2023-06-01"},
    ),
    "openai": _Validator("/v1/models", "bearer", "https://api.openai.com"),
    "openrouter": _Validator("/v1/key", "bearer", "https://openrouter.ai/api"),
    "groq": _Validator("/v1/models", "bearer", "https://api.groq.com/openai"),
    "mistral": _Validator("/v1/models", "bearer", "https://api.mistral.ai"),
    "deepseek": _Validator("/models", "bearer", "https://api.deepseek.com"),
    "xai": _Validator("/v1/models", "bearer", "https://api.x.ai"),
    "google": _Validator(
        "/v1beta/models", "query", "https://generativelanguage.googleapis.com"
    ),
    "gemini": _Validator(
        "/v1beta/models", "query", "https://generativelanguage.googleapis.com"
    ),
}

# google/gemini hit the same backend; report them as one in the missing-key line.
_ALIAS_GROUPS: list[set[str]] = [{"google", "gemini"}]

# Cloud providers exposing a request-time reasoning_effort dial (provider-level for now).
_EFFORT_PROVIDERS = {"openai", "anthropic", "google", "gemini", "xai"}

# /api/show capabilities are immutable per tag, so cache them; "thinking" = effort dial.
_capability_cache: dict[str, frozenset[str]] = {}


# --- result types --------------------------------------------------------------

class ProviderState(str, Enum):
    """Why a provider is / isn't eligible (str-valued so it serializes to JSON)."""

    VALIDATED = "validated"        # authenticated OK -> eligible
    INVALID_KEY = "invalid key"    # key present but provider rejected it
    UNREACHABLE = "unreachable"    # network/HTTP error; couldn't confirm
    NO_VALIDATOR = "no validator"  # key present but we don't know how to probe it
    MISSING_KEY = "missing key"    # known provider, no key configured


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    state: ProviderState
    detail: str

    @property
    def eligible(self) -> bool:
        return self.state is ProviderState.VALIDATED


@dataclass(frozen=True)
class LocalModel:
    name: str                      # Ollama tag, e.g. "qwen2.5-coder:7b"
    parameter_size: str | None     # e.g. "7B" — last-resort ranking signal
    family: str | None
    supports_effort: bool = False  # /api/show capabilities contains "thinking"


@dataclass(frozen=True)
class LocalDiscovery:
    reachable: bool
    models: tuple[LocalModel, ...]
    detail: str


@dataclass(frozen=True)
class AvailableModel:
    """One routable option in the merged pool returned by get_available_models()."""

    provider: str                  # "anthropic", "ollama", ...
    source: str                    # "cloud" | "local"
    id: str | None = None          # local: the model tag; cloud: None until models.dev
    parameter_size: str | None = None  # local only
    supports_effort: bool = False  # accepts a reasoning_effort/think field


# --- mutable snapshot (atomic ref swap; reads need no lock in CPython) ----------

@dataclass(frozen=True)
class _Snapshot:
    providers: tuple[ProviderStatus, ...] = ()
    local: LocalDiscovery = LocalDiscovery(False, (), "not yet polled")


_snapshot = _Snapshot()

# Own short-timeout client so a dead provider can't hang the server; monkeypatchable in tests.
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(config.discovery_timeout), follow_redirects=True
)
_poll_task: asyncio.Task | None = None


# --- 4a: cloud provider validation ---------------------------------------------

def _auth(validator: _Validator, key: str) -> tuple[dict[str, str], dict[str, str]]:
    headers = dict(validator.extra_headers)
    params: dict[str, str] = {}
    if validator.auth_style == "x-api-key":
        headers["x-api-key"] = key
    elif validator.auth_style == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif validator.auth_style == "query":
        params["key"] = key
    return headers, params


async def _validate_provider(provider: str, key: str) -> ProviderStatus:
    """One real auth call; eligible only on a 2xx."""
    validator = _VALIDATORS.get(provider)
    if validator is None:
        return ProviderStatus(
            provider, ProviderState.NO_VALIDATOR,
            f"key is set but no validation endpoint is known for '{provider}'",
        )

    base = config.provider_base_urls.get(provider, validator.base_url).rstrip("/")
    url = base + validator.path
    headers, params = _auth(validator, key)

    try:
        resp = await _client.get(url, headers=headers, params=params)
    except httpx.HTTPError as exc:
        return ProviderStatus(
            provider, ProviderState.UNREACHABLE,
            f"could not reach {url}: {type(exc).__name__}",
        )

    if resp.status_code in (400, 401, 403):
        return ProviderStatus(
            provider, ProviderState.INVALID_KEY,
            f"{provider} rejected the key (HTTP {resp.status_code})",
        )
    if resp.status_code >= 400:
        return ProviderStatus(
            provider, ProviderState.UNREACHABLE,
            f"unexpected HTTP {resp.status_code} from {url}",
        )
    return ProviderStatus(provider, ProviderState.VALIDATED, "authenticated OK")


async def validate_cloud_providers() -> list[ProviderStatus]:
    """Validate every configured key concurrently; never raises."""
    present = sorted(config.provider_api_keys.items())
    results = list(
        await asyncio.gather(*(_validate_provider(p, k) for p, k in present))
    )

    # Add keyless known providers as "missing key"; collapse alias groups to one entry.
    configured = set(config.provider_api_keys)
    suppressed = set()
    for group in _ALIAS_GROUPS:
        if configured & group:
            configured |= group
        else:
            suppressed |= set(sorted(group)[1:])  # keep one representative
    for provider in sorted(_VALIDATORS):
        if provider in configured or provider in suppressed:
            continue
        results.append(
            ProviderStatus(
                provider, ProviderState.MISSING_KEY,
                f"{provider.upper()}_API_KEY not set",
            )
        )
    return results


# --- 4b: local (Ollama) discovery ----------------------------------------------

async def _model_capabilities(name: str) -> frozenset[str]:
    """Cached /api/show capabilities for one tag; failure -> empty set (no effort dial)."""
    cached = _capability_cache.get(name)
    if cached is not None:
        return cached
    url = config.ollama_base_url.rstrip("/") + "/api/show"
    try:
        resp = await _client.post(url, json={"model": name})
        resp.raise_for_status()
        caps = frozenset(resp.json().get("capabilities") or [])
    except (httpx.HTTPError, ValueError):
        caps = frozenset()
    _capability_cache[name] = caps
    return caps


async def discover_local_models() -> LocalDiscovery:
    """Poll Ollama /api/tags + per-model capabilities; unreachable is reported, not raised."""
    url = config.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        resp = await _client.get(url)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return LocalDiscovery(False, (), f"unreachable: {type(exc).__name__}")

    entries = [
        (e["name"], e.get("details") or {})
        for e in data.get("models", [])
        if e.get("name")
    ]
    caps = await asyncio.gather(*(_model_capabilities(name) for name, _ in entries))
    models = [
        LocalModel(
            name=name,
            parameter_size=details.get("parameter_size"),
            family=details.get("family"),
            supports_effort="thinking" in cap,
        )
        for (name, details), cap in zip(entries, caps)
    ]
    return LocalDiscovery(True, tuple(models), f"{len(models)} model(s)")


# --- 4c: the single merged accessor every resolver reads -----------------------

def get_available_models() -> list[AvailableModel]:
    """Merged routable pool: provider-level cloud entries + one entry per local model."""
    snap = _snapshot
    models: list[AvailableModel] = [
        AvailableModel(
            provider=s.provider, source="cloud",
            supports_effort=s.provider in _EFFORT_PROVIDERS,
        )
        for s in snap.providers
        if s.eligible
    ]
    models.extend(
        AvailableModel(
            provider="ollama", source="local",
            id=m.name, parameter_size=m.parameter_size,
            supports_effort=m.supports_effort,
        )
        for m in snap.local.models
    )
    return models


def supports_reasoning_effort(provider: str, model_id: str | None = None) -> bool:
    """Whether the resolved model accepts a request-time reasoning_effort/think field."""
    if provider == "ollama":
        return any(
            m.name == model_id and m.supports_effort
            for m in _snapshot.local.models
        )
    return provider in _EFFORT_PROVIDERS


def provider_statuses() -> list[ProviderStatus]:
    """Current cloud-provider validation results (for observability / the summary)."""
    return list(_snapshot.providers)


def local_discovery() -> LocalDiscovery:
    return _snapshot.local


# --- refresh + lifecycle -------------------------------------------------------

async def refresh() -> None:
    """Full discovery: validate cloud keys, poll Ollama, swap the snapshot, log the summary."""
    global _snapshot
    providers, local = await asyncio.gather(
        validate_cloud_providers(), discover_local_models()
    )
    _snapshot = _Snapshot(tuple(providers), local)
    _log_summary(providers, local)


async def refresh_local() -> None:
    """Re-poll only Ollama; log a line when the set changes."""
    global _snapshot
    prev = _snapshot.local
    local = await discover_local_models()
    _snapshot = _Snapshot(_snapshot.providers, local)
    if {m.name for m in prev.models} != {m.name for m in local.models} or (
        prev.reachable != local.reachable
    ):
        names = ", ".join(m.name for m in local.models) or "(none)"
        print(f"[discovery] Ollama models changed -> {len(local.models)}: {names}")


async def _local_poll_loop() -> None:
    while True:
        await asyncio.sleep(config.discovery_interval)
        try:
            await refresh_local()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a transient poll error must not kill the loop
            print(f"[discovery] local poll failed: {type(exc).__name__}: {exc}")


async def start() -> None:
    """Run initial discovery and start the recurring local poller (from lifespan)."""
    if not config.discovery_enabled:
        return
    await refresh()
    global _poll_task
    _poll_task = asyncio.create_task(_local_poll_loop())


async def stop() -> None:
    """Cancel the poller and close the discovery client (from lifespan)."""
    global _poll_task
    if _poll_task is not None:
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass
        _poll_task = None
    await _client.aclose()


# --- 4d: human-readable startup summary ----------------------------------------

_STATE_TAG = {
    ProviderState.VALIDATED: "OK  ",
    ProviderState.INVALID_KEY: "FAIL",
    ProviderState.UNREACHABLE: "WARN",
    ProviderState.NO_VALIDATOR: "WARN",
    ProviderState.MISSING_KEY: "--  ",
}


def _log_summary(providers: list[ProviderStatus], local: LocalDiscovery) -> None:
    print("[discovery] Cloud provider validation:")
    checked = [p for p in providers if p.state is not ProviderState.MISSING_KEY]
    if checked:
        width = max(len(p.provider) for p in checked)
        for p in sorted(checked, key=lambda s: (not s.eligible, s.provider)):
            src = config.provider_key_sources.get(p.provider)
            note = "  (via OpenCode auth.json)" if src == "opencode" else ""
            print(
                f"[discovery]   {_STATE_TAG[p.state]}  {p.provider:<{width}}  "
                f"{p.detail}{note}"
            )
    else:
        print("[discovery]   (no {PROVIDER}_API_KEY found in config)")

    missing = sorted(p.provider for p in providers if p.state is ProviderState.MISSING_KEY)
    if missing:
        print(f"[discovery]   no key for: {', '.join(missing)}")

    if local.reachable:
        def _fmt(m: LocalModel) -> str:
            extra = [x for x in (m.parameter_size, "effort" if m.supports_effort else None) if x]
            return m.name + (f" ({', '.join(extra)})" if extra else "")

        listing = ", ".join(_fmt(m) for m in local.models) or "(none installed)"
        print(
            f"[discovery] Local models (Ollama @ {config.ollama_base_url}): "
            f"{len(local.models)} — {listing}"
        )
    else:
        print(
            f"[discovery] Local models (Ollama @ {config.ollama_base_url}): "
            f"{local.detail} — is `ollama serve` running?"
        )

    eligible = sum(1 for p in providers if p.eligible)
    n_local = len(local.models)
    print(
        f"[discovery] Routing pool: {eligible} cloud provider(s) eligible, "
        f"{n_local} local model(s) available."
    )
    if eligible == 0 and n_local == 0:
        print(
            "[discovery] WARNING: nothing is routable — no cloud provider "
            "authenticated and no local models found. Every resolution will 503. "
            "Check server/.env (see server/.env.example)."
        )
