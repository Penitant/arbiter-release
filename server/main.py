"""FastAPI app entrypoint. Run: `uv run uvicorn main:app --host $ARBITER_HOST --port $ARBITER_PORT`."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import DIFFICULTY_TO_TIER, config
from corrections import finetune as corrections_finetune
from corrections import store as corrections_store
from ferry import session_state
from proxy import forward, intercept
from registry import client as registry_client
from registry import discovery, matching, openrouter
from triage import resolver
from triage.classifier import classify


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup order: correction db, Ferry session state (+ one-shot TTL sweep), provider
    # discovery, catalog, benchmarks, then diagnostics.
    corrections_store.init(config.corrections_db_path)
    session_state.init(config.corrections_db_path)
    swept = session_state.sweep_expired()
    if swept:
        print(f"[ferry] TTL sweep: cleared {swept} session-state row(s) older than 24h.")
    await discovery.start()
    await registry_client.start()
    await openrouter.start()
    if config.discovery_enabled:
        _log_benchmark_match()
    # Log what each tier resolves to right now, warning if a tier is unsatisfiable.
    if config.discovery_enabled:
        print("[routing] live tier resolution against the current Registry:")
        for difficulty, tier in DIFFICULTY_TO_TIER.items():
            try:
                d = resolver.resolve(difficulty)
                print(f"[routing]   {tier:<13} -> {d.model}  ({d.reason})")
            except resolver.NoCandidateError as exc:
                print(f"[routing]   {tier:<13} -> UNRESOLVABLE: {exc}")
    yield
    await discovery.stop()
    await registry_client.stop()
    await openrouter.stop()
    await forward.aclose()
    corrections_store.close()
    session_state.close()


def _log_benchmark_match() -> None:
    """Report benchmark coverage over the routable candidate pool."""
    pool = registry_client.all_candidates()
    if not pool:
        return
    missing = matching.unmatched(pool)
    matched = len(pool) - len(missing)
    print(f"[registry] benchmark match: {matched}/{len(pool)} routable candidate(s) scored.")
    if missing:
        names = ", ".join(sorted(c.model for c in missing))
        print(f"[registry]   unmatched (cost/size fallback; add to aliases.py if wrong): {names}")


app = FastAPI(title="Arbiter Server", version="0.0.1", lifespan=lifespan)


@app.get("/__arbiter/discovery")
async def discovery_state() -> dict:
    """Validation hook: current provider eligibility + the merged routing pool."""
    return {
        "providers": [
            {"provider": s.provider, "state": s.state.value,
             "eligible": s.eligible, "detail": s.detail}
            for s in discovery.provider_statuses()
        ],
        "available_models": [vars(m) for m in discovery.get_available_models()],
        "catalog_size": registry_client.catalog_size(),
    }


@app.post("/__arbiter/classify")
async def classify_preview(payload: dict) -> dict:
    """Validation hook: classify + resolve a body without forwarding it."""
    body = payload
    if "messages" not in payload and "text" in payload:
        body = {"messages": [{"role": "user", "content": payload["text"]}]}
    classification = classify(body)
    decision = resolver.resolve(classification.difficulty)
    return {
        "difficulty": classification.difficulty,
        "confidence": classification.confidence,
        "score": classification.score,
        "signals": list(classification.signals),
        "tier": decision.tier,
        "routed_to": decision.model,
        "reasoning_effort": decision.reasoning_effort,
    }


# Notary endpoints: declared before include_router so they win over the catch-all proxy route.


class ClassifyRequest(BaseModel):
    task_text: str
    session_id: str | None = None
    # Reserved for a context-window-fit filter; accepted now so the request shape is stable.
    context_tokens_estimate: int | None = None


class CorrectionRequest(BaseModel):
    task_text: str
    predicted_tier: str
    predicted_model: str
    confidence: float
    session_id: str
    # Omitted/null when the user accepted the prediction as-is.
    corrected_tier: str | None = None
    corrected_model: str | None = None


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest):
    """Resolve a task's difficulty -> tier -> concrete model for Notary's proposal menu."""
    body = {"messages": [{"role": "user", "content": req.task_text}]}
    classification = classify(body)
    try:
        decision = resolver.resolve(
            classification.difficulty, confidence=classification.confidence
        )
    except resolver.NoCandidateError as exc:
        # 503 + Retry-After: Notary falls back to the user's last manual model selection.
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "30"},
            content={"error": f"arbiter: {exc}"},
        )
    return {
        "difficulty": decision.difficulty,
        "confidence": classification.confidence,
        "tier": decision.tier,
        "model": decision.model,
        "reasoning_effort": decision.reasoning_effort,
        "reason": decision.reason,
    }


@app.post("/corrections", status_code=204)
async def corrections_endpoint(req: CorrectionRequest):
    """Log one resolution (accepted or overridden), then enqueue the incremental fine-tune."""
    try:
        corrections_store.record(
            task_text=req.task_text,
            predicted_tier=req.predicted_tier,
            predicted_model=req.predicted_model,
            confidence=req.confidence,
            session_id=req.session_id,
            corrected_tier=req.corrected_tier,
            corrected_model=req.corrected_model,
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": f"arbiter: {exc}"})
    # Fire-and-forget: never awaited, so the response carries no gradient-step latency.
    if config.finetune_enabled:
        corrections_finetune.enqueue(config.corrections_db_path)
    return Response(status_code=204)


# Registered last so specific routes win over intercept's catch-all /{full_path}.
app.include_router(intercept.router)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
