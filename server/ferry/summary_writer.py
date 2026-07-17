"""Resolve and run the summary-writer model for Ferry's `cloud-mid` handoff.

The writer is independent of the destination: cheapest available, local-first, via the same
cost ranking Triage uses. Local Ollama models are free, so cost-first ranking picks them ahead
of cloud whenever one is installed — local-first falls out of the cost rule.

Reuses the OpenAI-wire route the forwarder builds. Anthropic is skipped as a writer for v1
(it needs wire translation rather than a plain chat-completions POST).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from proxy import forward
from registry import capability
from registry.client import Candidate, all_candidates

Message = dict[str, Any]

# Writers rank cost-first (cheapest, local-first) exactly like a cloud-mid resolution.
_WRITER_RANK_TIER = "cloud-mid"

# Own short-timeout async client: a handoff summary should be quick, and a slow writer must not
# stall the request path indefinitely. Separate from the async forwarder's unbounded client.
_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0), follow_redirects=False)


class NoWriterError(RuntimeError):
    """No model available to write a handoff summary."""


def select_writer() -> Candidate:
    """Cheapest available writer, local-first, skipping Anthropic (unsupported wire for v1)."""
    pool = [c for c in all_candidates() if c.provider != "anthropic"]
    if not pool:
        raise NoWriterError("no model available to write a handoff summary")
    return capability.rank_candidates(pool, _WRITER_RANK_TIER)[0]


def _endpoint(route: forward.Route) -> str:
    """The chat-completions URL for a writer route (mirrors forward.py's per-provider bases)."""
    if route.provider == "ollama":
        return f"{route.base_url}/v1/chat/completions"
    if route.provider in ("google", "gemini"):
        # google_openai_base_url already ends at the OpenAI-compatible root (…/openai).
        return f"{route.base_url}/chat/completions"
    # openai/groq/deepseek/mistral/xai/openrouter: base + standard /v1 chat path.
    return f"{route.base_url}/v1/chat/completions"


def _headers(route: forward.Route) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if route.auth == "bearer" and route.api_key:
        headers["Authorization"] = f"Bearer {route.api_key}"
    return headers


def _prompt(messages: list[Message], sections: tuple[str, ...]) -> list[Message]:
    """A chat request instructing the writer to fill the fixed handoff sections."""
    section_list = "\n".join(f"## {s}" for s in sections)
    system = (
        "You are a context-handoff summarizer. Read the conversation and produce a structured "
        "handoff for a different model that will continue the work. Use exactly these markdown "
        f"sections, in this order, and no others:\n{section_list}\n"
        "Be faithful and concise. Preserve exact identifiers, file paths, error strings, and "
        "hard constraints verbatim — do not paraphrase them. Omit nothing that a fresh model "
        "would need to continue correctly."
    )
    history = json.dumps(messages, ensure_ascii=False, default=str)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Conversation to hand off:\n{history}"},
    ]


def _extract_content(payload: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-wire chat completion response."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise NoWriterError("summary writer returned no usable content")


async def write_summary(messages: list[Message], sections: tuple[str, ...]) -> str:
    """Resolve a writer and return its structured handoff summary for `messages`.

    Raises on any failure (no writer, transport error, empty/malformed response) so the request
    path can fall back to forwarding the raw history rather than a silently-truncated payload.
    """
    writer = select_writer()
    route = forward.build_route(writer.provider, writer.id)
    if route.auth == "bearer" and not route.api_key:
        raise NoWriterError(f"no API key for summary writer provider '{route.provider}'")

    request_body = {
        "model": route.model_id,
        "messages": _prompt(messages, sections),
        "stream": False,
    }
    resp = await _client.post(_endpoint(route), headers=_headers(route), json=request_body)
    resp.raise_for_status()
    return _extract_content(resp.json())
