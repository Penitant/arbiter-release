"""Forward an intercepted request to the resolved model, rewriting only the `model` field."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from config import config
from proxy import anthropic_wire

# Hop-by-hop headers dropped in both directions; content-encoding stays (raw byte passthrough).
_HOP_BY_HOP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
}

# Shared client for connection reuse; no timeout because streamed LLM responses run long.
_client = httpx.AsyncClient(timeout=httpx.Timeout(None), follow_redirects=False)


async def aclose() -> None:
    await _client.aclose()


@dataclass(frozen=True)
class Route:
    """A resolved destination: target base, auth, and the model id to stamp onto the body."""

    provider: str
    base_url: str
    model_id: str
    auth: str              # "none" | "bearer" | "x-api-key"
    api_key: str | None
    strip_v1: bool         # strip a leading "/v1" from the client path before appending


# Bearer-auth OpenAI-wire bases (base + "/v1/chat/completions"); overridable via {PROVIDER}_BASE_URL.
_OPENAI_WIRE_BASES: dict[str, str] = {
    "openai": "https://api.openai.com",
    "groq": "https://api.groq.com/openai",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai",
    "xai": "https://api.x.ai",
    "openrouter": "https://openrouter.ai/api",
}

# Every provider this transport can reach (anthropic via wire translation, not byte passthrough).
FORWARDABLE_PROVIDERS: frozenset[str] = frozenset(
    {"ollama", "google", "gemini", "anthropic"} | set(_OPENAI_WIRE_BASES)
)


def build_route(provider: str, model_id: str) -> Route:
    """Turn a resolver Decision's (provider, model_id) into a transport Route."""
    if provider == "ollama":
        return Route(
            provider="ollama",
            base_url=config.ollama_base_url,
            model_id=model_id,
            auth="none",
            api_key=None,
            strip_v1=False,
        )
    if provider == "anthropic":
        return Route(
            provider="anthropic",
            base_url=config.anthropic_base_url,
            model_id=model_id,
            auth="x-api-key",
            api_key=config.anthropic_api_key,
            strip_v1=False,
        )
    if provider in ("google", "gemini"):
        key = config.provider_api_keys.get("google") or config.provider_api_keys.get(
            "gemini"
        )
        return Route(
            provider=provider,
            base_url=config.google_openai_base_url,
            model_id=model_id,
            auth="bearer",
            api_key=key,
            strip_v1=True,
        )
    if provider in _OPENAI_WIRE_BASES:
        base = config.provider_base_urls.get(provider, _OPENAI_WIRE_BASES[provider])
        return Route(
            provider=provider,
            base_url=base.rstrip("/"),
            model_id=model_id,
            auth="bearer",
            api_key=config.provider_api_keys.get(provider),
            strip_v1=False,
        )
    raise ValueError(f"no route known for provider '{provider}'")


def _rewrite_model(body: bytes, model_id: str) -> bytes:
    """Stamp the resolved model id onto a JSON chat body; non-JSON/model-less bodies pass untouched."""
    if not body:
        return body
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body
    if not isinstance(parsed, dict) or "model" not in parsed:
        return body
    parsed["model"] = model_id
    return json.dumps(parsed).encode()


def _upstream_headers(request: Request, route: Route) -> dict[str, str]:
    """Client headers, minus hop-by-hop, with auth set for the chosen route."""
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    # Auth is the one thing we own — strip whatever the client sent and set per route.
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)
    if route.auth == "bearer" and route.api_key:
        headers["Authorization"] = f"Bearer {route.api_key}"
    return headers


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}


def _target_url(request: Request, route: Route) -> httpx.URL:
    """base + (optionally /v1-stripped) request path (+ query); string-concat keeps prefix-path bases intact."""
    path = request.url.path
    if route.strip_v1 and path.startswith("/v1"):
        path = path[len("/v1"):]
    target_str = route.base_url + path
    if request.url.query:
        target_str += "?" + request.url.query
    return httpx.URL(target_str)


def _anthropic_headers(request: Request, route: Route) -> dict[str, str]:
    """Client headers, minus hop-by-hop, with Anthropic's x-api-key + version auth."""
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)
    headers["x-api-key"] = route.api_key or ""
    headers["anthropic-version"] = config.anthropic_version
    headers["content-type"] = "application/json"
    return headers


async def _forward_anthropic(
    request: Request, body: bytes, route: Route
) -> StreamingResponse | JSONResponse:
    """Translate an OpenAI-wire request to Anthropic's Messages API and translate the reply back."""
    if not route.api_key:
        return JSONResponse(
            status_code=503,
            content={
                "error": "arbiter: no API key for provider 'anthropic'",
                "hint": "set ANTHROPIC_API_KEY in server/.env or connect Anthropic in OpenCode "
                        "(see server/.env.example)",
            },
        )

    stream = anthropic_wire.wants_stream(body)
    translated = anthropic_wire.translate_request(body, route.model_id)
    upstream_req = _client.build_request(
        "POST",
        httpx.URL(route.base_url + "/v1/messages"),
        headers=_anthropic_headers(request, route),
        content=translated,
    )
    upstream = await _client.send(upstream_req, stream=True)

    if upstream.status_code != 200:
        # Surface the upstream error body verbatim; don't pretend it succeeded.
        raw = await upstream.aread()
        await upstream.aclose()
        try:
            content = json.loads(raw)
        except (ValueError, TypeError):
            content = {"error": raw.decode(errors="replace")}
        return JSONResponse(status_code=upstream.status_code, content=content)

    if stream:
        return StreamingResponse(
            anthropic_wire.translate_stream(upstream, route.model_id),
            status_code=200,
            headers={"content-type": "text/event-stream"},
            background=BackgroundTask(upstream.aclose),
        )

    raw = await upstream.aread()
    await upstream.aclose()
    return JSONResponse(
        status_code=200,
        content=anthropic_wire.translate_response_json(raw, route.model_id),
    )


async def forward(
    request: Request, body: bytes, route: Route
) -> StreamingResponse | JSONResponse:
    """Forward `body` (model field rewritten) to `route` and stream the response."""
    if route.provider == "anthropic":
        return await _forward_anthropic(request, body, route)
    if route.auth == "bearer" and not route.api_key:
        # Fail loud rather than forwarding a request that will 401 upstream.
        return JSONResponse(
            status_code=503,
            content={
                "error": f"arbiter: no API key for provider '{route.provider}'",
                "hint": "set it in server/.env or connect the provider in OpenCode "
                        "(see server/.env.example)",
            },
        )

    body = _rewrite_model(body, route.model_id)
    target = _target_url(request, route)
    upstream_req = _client.build_request(
        request.method,
        target,
        headers=_upstream_headers(request, route),
        content=body,
    )
    upstream = await _client.send(upstream_req, stream=True)

    async def body_iter() -> AsyncIterator[bytes]:
        # aiter_raw() yields undecoded bytes — exact passthrough, no recompression.
        async for chunk in upstream.aiter_raw():
            yield chunk

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
        background=BackgroundTask(upstream.aclose),
    )
