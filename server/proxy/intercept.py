"""baseUrl-facing interception: classify each request, resolve a model, forward to it."""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Deque, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ferry.payload import build_payload
from ferry.switch_detector import check_switch, extract_session_id
from proxy import forward
from triage import resolver
from triage.classifier import classify

router = APIRouter()

# Bounded in-memory record of recent intercepts, for observability only.
_MAX_INTERCEPTS = 100
intercept_log: Deque[Dict[str, Any]] = deque(maxlen=_MAX_INTERCEPTS)


def _parse(body: bytes) -> dict | None:
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _record(request: Request, body: bytes, entry_extra: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "ts": time.time(),
        "method": request.method,
        "path": request.url.path,
        "bytes": len(body),
        **entry_extra,
    }
    intercept_log.append(entry)
    print(
        f"[intercept] {entry['method']} {entry['path']} "
        f"requested={entry.get('requested_model')} -> "
        f"{entry.get('difficulty')}/{entry.get('routed_to')} ({entry['bytes']}B)"
    )
    return entry


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness ping for validating the server is up and reachable."""
    return {"status": "ok", "service": "arbiter-server", "intercepts_seen": len(intercept_log)}


@router.get("/__arbiter/intercepts")
async def intercepts() -> Dict[str, Any]:
    """Validation hook: the requests this proxy has intercepted so far."""
    return {"count": len(intercept_log), "entries": list(intercept_log)}


@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def intercept_and_forward(request: Request, full_path: str):
    """Classify the request, resolve a model, then forward it to that model."""
    body = await request.body()
    parsed = _parse(body)

    classification = classify(parsed)
    tools_present = bool(parsed.get("tools")) if parsed else False
    try:
        decision = resolver.resolve(
            classification.difficulty, tools_present, classification.confidence
        )
    except resolver.NoCandidateError as exc:
        # Nothing routable for this tier: fail loud rather than guess a model.
        return JSONResponse(
            status_code=503,
            content={
                "error": f"arbiter: {exc}",
                "hint": "configure a provider in server/.env or connect one in OpenCode, "
                        "and/or install a local model in Ollama (see server/.env.example)",
            },
        )
    route = forward.build_route(decision.provider, decision.model_id)

    # Ferry (reactive): on a genuine mid-session switch, reshape the outgoing context to the
    # resolved tier. Best-effort by design — a Ferry failure (uninitialized state store, a
    # summary-writer round-trip error) must never fail the request, so fall back to forwarding
    # the original body untouched.
    if parsed is not None and isinstance(parsed.get("messages"), list):
        try:
            session_id = extract_session_id(request.headers, parsed)
            if check_switch(session_id, decision.model):
                parsed["messages"] = await build_payload(
                    parsed["messages"], decision.tier, is_switch=True
                )
                body = json.dumps(parsed).encode()
        except Exception as exc:  # noqa: BLE001 - Ferry is non-fatal to forwarding.
            print(f"[ferry] context reshaping skipped, forwarding original: {exc}")

    _record(
        request,
        body,
        {
            "requested_model": parsed.get("model") if parsed else None,
            "difficulty": classification.difficulty,
            "confidence": classification.confidence,
            "tier": decision.tier,
            "routed_to": decision.model,
            "signals": list(classification.signals),
        },
    )
    return await forward.forward(request, body, route)
