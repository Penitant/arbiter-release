"""Server configuration: listen host/port, provider credentials, and tier lookup tables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


# The classifier's difficulty label maps 1:1 to a routing tier.
DIFFICULTY_TO_TIER: dict[str, str] = {
    "easy": "local-simple",
    "medium": "cloud-mid",
    "hard": "cloud-heavy",
}

# Per-tier capability floor (`reasoning_required`) + ranking direction (`rank_by`).
TIER_CAPABILITY_REQUIREMENTS: dict[str, dict[str, object]] = {
    "local-simple": {"reasoning_required": False, "rank_by": "cheapest"},
    "cloud-mid": {"reasoning_required": True, "rank_by": "cheapest"},
    "cloud-heavy": {"reasoning_required": True, "rank_by": "most_capable"},
}

# Below this classifier confidence, an `easy` label is escalated to `medium` (never medium->hard).
CONFIDENCE_THRESHOLD: float = float(os.environ.get("ARBITER_CONFIDENCE_THRESHOLD", "0.65"))

# Per-tier reasoning-effort default; recorded on the decision, not injected into bodies.
TIER_DEFAULT_EFFORT: dict[str, str | None] = {
    "local-simple": None,
    "cloud-mid": "medium",
    "cloud-heavy": "high",
}


def _load_env_file(path: Path) -> None:
    """Minimal .env loader (no dependency)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# SDK-specific env names normalized onto the canonical provider name.
_PROVIDER_NAME_ALIASES = {
    "google_generative_ai": "google",
    "googleai": "google",
    "google_ai": "google",
    "azure_openai": "openai",
}

# auth.json entries that are not borrowable cloud credentials.
_OPENCODE_SKIP = {"ollama", "arbiter"}


def _default_opencode_auth_path() -> str:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(base) / "opencode" / "auth.json")


def _load_opencode_auth(path: Path) -> dict[str, str]:
    """Borrow `type: "api"` provider keys from OpenCode's auth.json (OAuth entries skipped)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for provider, entry in data.items():
        name = _PROVIDER_NAME_ALIASES.get(provider, provider)
        if name in _OPENCODE_SKIP or not isinstance(entry, dict):
            continue
        if entry.get("type") == "api" and entry.get("key"):
            out[name] = entry["key"]
    return out


def _collect_env_suffixed(suffix: str) -> dict[str, str]:
    """All env vars `{PROVIDER}{suffix}` collapsed to {provider_lower: value}."""
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.endswith(suffix) and value.strip():
            provider = key[: -len(suffix)].lower()
            provider = _PROVIDER_NAME_ALIASES.get(provider, provider)
            if provider:
                out[provider] = value.strip()
    return out


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    anthropic_base_url: str
    anthropic_api_key: str | None
    anthropic_version: str
    # Forward the client's own auth verbatim to `upstream_url` instead of injecting a key.
    passthrough: bool
    upstream_url: str
    # Manual local-routing toggle, kept for validation only.
    force_local: bool
    ollama_base_url: str
    # Google's OpenAI-compatible endpoint; same wire format as Ollama.
    google_openai_base_url: str
    # Every `{PROVIDER}_API_KEY` / `{PROVIDER}_BASE_URL` in the env, keyed by provider.
    provider_api_keys: dict[str, str]
    provider_base_urls: dict[str, str]
    # Key origin per provider: "env" or "opencode".
    provider_key_sources: dict[str, str]
    # Startup/recurring discovery controls; disabled under test.
    discovery_enabled: bool
    discovery_interval: float
    discovery_timeout: float
    # models.dev catalog URL + on-disk cache dir for the last successful fetch.
    models_dev_url: str
    # OpenRouter /api/v1/models — the benchmark source (public, no key).
    openrouter_models_url: str
    registry_cache_dir: str
    # Correction-log SQLite file; overridable for tests via ARBITER_DB_PATH.
    corrections_db_path: str
    # Incremental fine-tune trigger; off in tests so posting corrections is side-effect-free.
    finetune_enabled: bool

    @classmethod
    def load(cls) -> "Config":
        _load_env_file(Path(__file__).parent / ".env")
        # Explicit env keys win over keys borrowed from OpenCode's auth.json.
        env_keys = _collect_env_suffixed("_API_KEY")
        read_auth = os.environ.get("ARBITER_READ_OPENCODE_AUTH", "1").lower() in (
            "1", "true", "yes"
        )
        auth_keys = (
            _load_opencode_auth(
                Path(os.environ.get("OPENCODE_AUTH_PATH", _default_opencode_auth_path()))
            )
            if read_auth
            else {}
        )
        provider_api_keys = {**auth_keys, **env_keys}
        provider_key_sources = {
            p: ("env" if p in env_keys else "opencode") for p in provider_api_keys
        }
        return cls(
            host=os.environ.get("ARBITER_HOST", "127.0.0.1"),
            port=int(os.environ.get("ARBITER_PORT", "4317")),
            anthropic_base_url=os.environ.get(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            ).rstrip("/"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            anthropic_version=os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            passthrough=os.environ.get("ARBITER_PASSTHROUGH", "").lower()
            in ("1", "true", "yes"),
            upstream_url=os.environ.get(
                "ARBITER_UPSTREAM_URL", ""
            ).rstrip("/"),
            force_local=os.environ.get("FORCE_LOCAL", "").lower() in ("1", "true", "yes"),
            # No /v1 suffix: the client path already includes it.
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            google_openai_base_url=os.environ.get(
                "GOOGLE_OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ).rstrip("/"),
            provider_api_keys=provider_api_keys,
            provider_base_urls=_collect_env_suffixed("_BASE_URL"),
            provider_key_sources=provider_key_sources,
            discovery_enabled=os.environ.get("ARBITER_DISCOVERY", "1").lower()
            in ("1", "true", "yes"),
            discovery_interval=float(os.environ.get("OLLAMA_POLL_INTERVAL", "30")),
            discovery_timeout=float(os.environ.get("DISCOVERY_TIMEOUT", "5")),
            models_dev_url=os.environ.get("MODELS_DEV_URL", "https://models.dev/api.json"),
            openrouter_models_url=os.environ.get(
                "OPENROUTER_MODELS_URL", "https://openrouter.ai/api/v1/models"
            ),
            registry_cache_dir=os.environ.get(
                "REGISTRY_CACHE_DIR", str(Path(__file__).parent / "cache")
            ),
            corrections_db_path=os.environ.get(
                "ARBITER_DB_PATH", str(Path(__file__).parent / "db" / "arbiter.db")
            ),
            finetune_enabled=os.environ.get("ARBITER_FINETUNE", "1").lower()
            in ("1", "true", "yes"),
        )


config = Config.load()
