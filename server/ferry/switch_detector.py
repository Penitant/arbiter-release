"""Derive a stable session key and flag mid-session model switches.

OpenCode carries the session id in an HTTP header (not the body), stable across a session's
requests. session_id is derived, in priority order:

  1. `x-session-id` header (what OpenCode sets).
  2. `x-session-affinity` header (mirror fallback).
  3. Otherwise a content-derived key: SHA-256 of the first message, prefixed `ses_local_`.

State lives in the durable `session_model_state` store (ferry/session_state.py); this module
owns the request-facing logic: extracting the key and the switch comparison.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ferry import session_state


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup that tolerates plain dicts and Starlette headers alike."""
    getter = getattr(headers, "get", None)
    if getter is not None:
        # Starlette Headers.get is already case-insensitive; dict.get is not, so try both cases.
        value = headers.get(name) or headers.get(name.lower()) or headers.get(name.title())
        if value:
            return value.strip() or None
    return None


def _fallback_session_id(body: dict[str, Any] | None) -> str:
    """Content-derived stable key: SHA-256 of the first message, used only when no header exists."""
    first = ""
    if isinstance(body, dict):
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            # Serialize the whole first message (role + content, blocks included) deterministically.
            first = json.dumps(messages[0], sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(first.encode("utf-8")).hexdigest()
    return f"ses_local_{digest[:24]}"


def extract_session_id(headers: Mapping[str, str], body: dict[str, Any] | None) -> str:
    """Derive a stable session key for a request. See module docstring for the derivation order."""
    sid = _header(headers, "x-session-id") or _header(headers, "x-session-affinity")
    if sid:
        return sid
    return _fallback_session_id(body)


def check_switch(
    session_id: str,
    resolved_model: str,
    parent_session: str | None = None,
) -> bool:
    """Return True when this request should trigger Ferry's context payload.

    A brand-new session (no prior row) counts as a switch by design — its first request has no
    prior history to dump anyway, so the payload step runs uniformly. Reads/writes go through
    session_state, which must be init()-ed at Server startup before this is called.
    """
    last = session_state.get_last_model(session_id)
    session_state.set_last_model(session_id, resolved_model, parent_session)
    if last is None:
        return True
    return last != resolved_model
