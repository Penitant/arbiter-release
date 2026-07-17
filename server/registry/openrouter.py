"""OpenRouter benchmark source: fetch + cache artificial_analysis.coding_index per model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from config import config

# The single benchmark metric used for ranking; deliberately not a config knob.
_AA_SOURCE = "artificial_analysis"
_AA_METRIC = "coding_index"


@dataclass(frozen=True)
class BenchmarkEntry:
    """One OpenRouter model carrying a usable coding_index, reduced to matching keys."""

    id: str                       # OpenRouter slug, e.g. "qwen/qwen2.5-coder-7b-instruct"
    hugging_face_id: str | None   # e.g. "Qwen/Qwen2.5-Coder-7B-Instruct" (often null)
    coding_index: float           # artificial_analysis.coding_index, 0–100


@dataclass(frozen=True)
class _Snapshot:
    entries: tuple[BenchmarkEntry, ...]
    source: str                   # "fetch" | "cache" | "empty"
    detail: str


_snapshot = _Snapshot((), "empty", "not yet fetched")

# Own client; OpenRouter 403s without a User-Agent. Monkeypatchable in tests.
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(config.discovery_timeout),
    follow_redirects=True,
    headers={"User-Agent": "arbiter-registry/0.1"},
)


# --- parse ---------------------------------------------------------------------

def _parse_benchmarks(data: dict) -> tuple[BenchmarkEntry, ...]:
    """Keep only models with a numeric coding_index; malformed entries are skipped."""
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return ()
    out: list[BenchmarkEntry] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        benchmarks = m.get("benchmarks")
        aa = benchmarks.get(_AA_SOURCE) if isinstance(benchmarks, dict) else None
        score = aa.get(_AA_METRIC) if isinstance(aa, dict) else None
        if not isinstance(score, (int, float)):
            continue
        model_id = m.get("id")
        if not isinstance(model_id, str):
            continue
        hf = m.get("hugging_face_id")
        out.append(
            BenchmarkEntry(
                id=model_id,
                hugging_face_id=hf if isinstance(hf, str) and hf else None,
                coding_index=float(score),
            )
        )
    return tuple(out)


# --- cache ---------------------------------------------------------------------

def _cache_path() -> Path:
    return Path(config.registry_cache_dir) / "openrouter.json"


def _write_cache(raw: str) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
    except OSError as exc:
        print(f"[registry] could not write openrouter cache: {type(exc).__name__}: {exc}")


def _load_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return None


# --- fetch ---------------------------------------------------------------------

async def fetch_benchmarks() -> _Snapshot:
    """Fetch OpenRouter; on any failure fall back to the last-good on-disk cache."""
    try:
        resp = await _client.get(config.openrouter_models_url)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        cached = _load_cache()
        if cached is not None:
            entries = _parse_benchmarks(cached)
            return _Snapshot(
                entries, "cache",
                f"fetch failed ({type(exc).__name__}); served cached benchmarks",
            )
        return _Snapshot(
            (), "empty",
            f"fetch failed ({type(exc).__name__}) and no cache available",
        )

    _write_cache(resp.text)
    entries = _parse_benchmarks(data)
    return _Snapshot(entries, "fetch", f"{len(entries)} models with coding_index")


# --- accessors + lifecycle -----------------------------------------------------

def entries() -> tuple[BenchmarkEntry, ...]:
    """Current benchmark entries; identity changes on refresh (memoization key)."""
    return _snapshot.entries


async def refresh() -> None:
    global _snapshot
    _snapshot = await fetch_benchmarks()


async def start() -> None:
    if not config.discovery_enabled:
        return
    await refresh()
    print(
        f"[registry] OpenRouter benchmarks: {_snapshot.detail} (source={_snapshot.source})"
    )
    if not _snapshot.entries:
        print(
            "[registry] WARNING: no OpenRouter benchmark data — ranking falls back to "
            "cost/parameter-size only. Check network access to openrouter.ai."
        )


async def stop() -> None:
    await _client.aclose()
