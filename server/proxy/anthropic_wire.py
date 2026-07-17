"""Translate between OpenAI chat-completions wire format and Anthropic's Messages API.

Anthropic is the one forwardable provider that doesn't speak the OpenAI wire format, so its
requests/responses are rewritten here rather than passed through byte-for-byte like every other
provider in `forward.py`.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx

# Anthropic requires an explicit output cap; OpenAI clients often omit it.
DEFAULT_MAX_TOKENS = 8096

# OpenAI top-level params Anthropic also accepts (built into a fresh body so unknown OpenAI-only
# fields like presence_penalty/n/seed never reach Anthropic, which 400s on unknown params).
_PASSTHROUGH_PARAMS = ("temperature", "top_p", "top_k")

# Anthropic stop_reason -> OpenAI finish_reason.
_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}


def strip_prefix(model: str) -> str:
    """Drop a leading `anthropic/` so the id sent upstream is the bare model name."""
    return model[len("anthropic/"):] if model.startswith("anthropic/") else model


def wants_stream(body: bytes) -> bool:
    parsed = _load(body)
    return bool(parsed.get("stream")) if parsed else False


def _load(body: bytes) -> dict | None:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# --- request: OpenAI -> Anthropic ----------------------------------------------


def translate_request(body: bytes, model_id: str) -> bytes:
    """Rewrite an OpenAI chat-completions body into an Anthropic Messages body."""
    parsed = _load(body) or {}
    messages, system = _split_messages(parsed.get("messages") or [])

    out: dict[str, Any] = {
        "model": strip_prefix(model_id),
        "max_tokens": parsed.get("max_tokens") or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system:
        out["system"] = system
    if parsed.get("stream"):
        out["stream"] = True

    tools = parsed.get("tools")
    if tools:
        out["tools"] = _convert_tools(tools)
    tool_choice = _convert_tool_choice(parsed.get("tool_choice"))
    if tool_choice is not None:
        out["tool_choice"] = tool_choice

    stop = parsed.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    for param in _PASSTHROUGH_PARAMS:
        if parsed.get(param) is not None:
            out[param] = parsed[param]

    return json.dumps(out).encode()


def _split_messages(messages: list) -> tuple[list[dict], str | None]:
    """Hoist `role: system` messages into the top-level system string; convert the rest."""
    system_parts: list[str] = []
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            text = _text_of(msg.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            # OpenAI groups parallel results as separate `tool` turns; Anthropic wants them
            # collected into one user turn of tool_result blocks.
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": _text_of(msg.get("content")),
            })
            continue

        flush_tool_results()
        if role == "assistant" and msg.get("tool_calls"):
            out.append({"role": "assistant", "content": _assistant_blocks(msg)})
        else:
            out.append({"role": role, "content": msg.get("content") or ""})

    flush_tool_results()
    system = "\n\n".join(p for p in system_parts if p)
    return out, (system or None)


def _assistant_blocks(msg: dict) -> list[dict]:
    """Turn an OpenAI assistant turn with tool_calls into Anthropic text + tool_use blocks."""
    blocks: list[dict] = []
    text = _text_of(msg.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except (ValueError, TypeError):
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id"),
            "name": fn.get("name"),
            "input": args,
        })
    return blocks


def _text_of(content: Any) -> str:
    """Flatten OpenAI content (string, or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _convert_tools(tools: list) -> list[dict]:
    """OpenAI `{"type":"function","function":{...}}` -> Anthropic `{name,description,input_schema}`."""
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            out.append({
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        else:
            # Already Anthropic-shaped (or unknown) — pass through untouched.
            out.append(tool)
    return out


def _convert_tool_choice(choice: Any) -> dict | None:
    if choice in (None, "auto"):
        return None
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


# --- response: Anthropic -> OpenAI ---------------------------------------------


def _chatcmpl_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def translate_response_json(body: bytes, model: str) -> dict:
    """Translate a non-streaming Anthropic Messages response into an OpenAI chat.completion."""
    try:
        resp = json.loads(body)
    except (ValueError, TypeError):
        resp = {}

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in resp.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp.get("usage") or {}
    prompt = usage.get("input_tokens", 0) or 0
    completion = usage.get("output_tokens", 0) or 0
    return {
        "id": _chatcmpl_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": strip_prefix(model),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _FINISH_REASON.get(resp.get("stop_reason"), "stop"),
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


async def translate_stream(upstream: httpx.Response, model: str) -> AsyncIterator[bytes]:
    """Map Anthropic's SSE event stream onto OpenAI chat.completion.chunk SSE lines."""
    cmpl_id = _chatcmpl_id()
    created = int(time.time())
    model = strip_prefix(model)

    def chunk(delta: dict, finish_reason: str | None) -> bytes:
        payload = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    # Opening chunk establishes the assistant role, matching OpenAI's stream shape.
    yield chunk({"role": "assistant", "content": ""}, None)

    tool_index_of_block: dict[int, int] = {}
    next_tool_index = 0
    finish_reason = "stop"

    async for line in upstream.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data:
            continue
        try:
            evt = json.loads(data)
        except (ValueError, TypeError):
            continue

        etype = evt.get("type")
        if etype == "content_block_start":
            block = evt.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = next_tool_index
                next_tool_index += 1
                tool_index_of_block[evt.get("index")] = idx
                yield chunk(
                    {"tool_calls": [{
                        "index": idx,
                        "id": block.get("id"),
                        "type": "function",
                        "function": {"name": block.get("name"), "arguments": ""},
                    }]},
                    None,
                )
        elif etype == "content_block_delta":
            delta = evt.get("delta") or {}
            if delta.get("type") == "text_delta":
                yield chunk({"content": delta.get("text", "")}, None)
            elif delta.get("type") == "input_json_delta":
                idx = tool_index_of_block.get(evt.get("index"), 0)
                yield chunk(
                    {"tool_calls": [{
                        "index": idx,
                        "function": {"arguments": delta.get("partial_json", "")},
                    }]},
                    None,
                )
        elif etype == "message_delta":
            stop_reason = (evt.get("delta") or {}).get("stop_reason")
            if stop_reason:
                finish_reason = _FINISH_REASON.get(stop_reason, "stop")
        elif etype == "message_stop":
            break
        elif etype == "error":
            finish_reason = "stop"
            break

    yield chunk({}, finish_reason)
    yield b"data: [DONE]\n\n"
