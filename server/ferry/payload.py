"""Ferry: shape the context that accompanies a switched task. Reactive — never picks a model.

Context scales with Triage's tier:
  * local-simple → minimal: just the task (last user message).
  * cloud-mid    → a structured handoff summary in one system message, written by the
                   cheapest available model (local-first), independent of the destination.
  * cloud-heavy  → full raw history, byte-identical to input.

Non-switch requests pass through untouched regardless of tier.
"""

from __future__ import annotations

from typing import Any

# Fixed handoff sections. Order is load-bearing: the template the writer fills and the shape
# a destination model reads back.
SUMMARY_SECTIONS: tuple[str, ...] = (
    "Goal",
    "Decisions",
    "Work done",
    "Remaining tasks",
    "Hard constraints",
)

Message = dict[str, Any]


async def build_payload(
    messages: list[Message], tier: str, *, is_switch: bool = True
) -> list[Message]:
    """Return the messages to forward, reshaped to `tier` when `is_switch` is True.

    A no-op (returns `messages` unchanged) when `is_switch` is False, when `tier` is
    unrecognized, or when there is no prior history to hand off. `cloud-heavy` is always a
    verbatim passthrough — the returned list is the same object, so the forwarded bytes stay
    identical to the input.
    """
    if not is_switch:
        return messages
    if tier == "cloud-heavy":
        # Full fidelity: hand back the input untouched (same object → byte-identical forward).
        return messages
    if tier == "local-simple":
        return _minimal(messages)
    if tier == "cloud-mid":
        return await _structured_summary(messages)
    # Unknown tier: don't guess — forward as-is rather than silently drop context.
    return messages


def _minimal(messages: list[Message]) -> list[Message]:
    """Strip to the task itself: the last user message only."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return [msg]
    # No user turn at all (unusual): keep the last message rather than send nothing.
    return messages[-1:] if messages else messages


async def _structured_summary(messages: list[Message]) -> list[Message]:
    """Collapse the history into a single structured system message.

    A single-turn session has no prior history to compress, so it passes through unchanged;
    otherwise a summary-writer model produces the handoff.
    """
    if len(messages) < 2:
        return messages
    return [{"role": "system", "content": await summarize(messages)}]


async def summarize(messages: list[Message]) -> str:
    """Produce the structured handoff summary text for `messages`.

    The writer is resolved independently of the destination — cheapest available, local-first.
    Performs a real model round-trip, so request-path callers must treat a failure here as
    non-fatal and fall back to forwarding the raw history.
    """
    # Lazy import to stay import-cheap and avoid a registry/proxy cycle.
    from ferry import summary_writer

    return await summary_writer.write_summary(messages, SUMMARY_SECTIONS)
