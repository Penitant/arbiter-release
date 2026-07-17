"""Tests for Anthropic wire translation (OpenAI <-> Messages API) and the forward path."""

import asyncio
import json
from types import SimpleNamespace

import httpx
from fastapi import Request

from proxy import anthropic_wire as aw
from proxy import forward


# --- request translation: OpenAI -> Anthropic ----------------------------------

def test_system_message_is_hoisted_out_of_messages():
    body = json.dumps({
        "model": "x",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
    }).encode()
    out = json.loads(aw.translate_request(body, "anthropic/claude-sonnet-4-5"))
    assert out["system"] == "be terse"
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_model_prefix_is_stripped():
    out = json.loads(aw.translate_request(b'{"messages":[]}', "anthropic/claude-opus-4-8"))
    assert out["model"] == "claude-opus-4-8"


def test_max_tokens_defaults_when_absent():
    out = json.loads(aw.translate_request(b'{"messages":[]}', "claude-x"))
    assert out["max_tokens"] == aw.DEFAULT_MAX_TOKENS


def test_max_tokens_preserved_when_present():
    out = json.loads(aw.translate_request(b'{"messages":[],"max_tokens":42}', "claude-x"))
    assert out["max_tokens"] == 42


def test_openai_tool_schema_becomes_anthropic_schema():
    body = json.dumps({
        "messages": [],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "current weather",
                "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}},
            },
        }],
    }).encode()
    out = json.loads(aw.translate_request(body, "claude-x"))
    assert out["tools"] == [{
        "name": "get_weather",
        "description": "current weather",
        "input_schema": {"type": "object", "properties": {"loc": {"type": "string"}}},
    }]


def test_stream_flag_carried_through():
    out = json.loads(aw.translate_request(b'{"messages":[],"stream":true}', "c"))
    assert out["stream"] is True


def test_openai_only_params_are_dropped():
    body = json.dumps({
        "messages": [],
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "n": 2,
        "temperature": 0.5,
    }).encode()
    out = json.loads(aw.translate_request(body, "c"))
    assert "presence_penalty" not in out and "frequency_penalty" not in out and "n" not in out
    assert out["temperature"] == 0.5


def test_assistant_tool_calls_and_tool_results_round_trip():
    body = json.dumps({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"loc":"Paris"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
        ],
    }).encode()
    out = json.loads(aw.translate_request(body, "c"))
    assert out["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather",
                     "input": {"loc": "Paris"}}],
    }
    assert out["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "72F"}],
    }


# --- non-streaming response translation: Anthropic -> OpenAI --------------------

def test_response_text_translation():
    resp = json.dumps({
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }).encode()
    out = aw.translate_response_json(resp, "anthropic/claude-x")
    assert out["object"] == "chat.completion"
    assert out["model"] == "claude-x"
    assert out["choices"][0]["message"]["content"] == "hello"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_response_tool_use_becomes_tool_calls():
    resp = json.dumps({
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}}],
        "stop_reason": "tool_use",
    }).encode()
    out = aw.translate_response_json(resp, "claude-x")
    tc = out["choices"][0]["message"]["tool_calls"][0]
    assert tc == {"id": "toolu_1", "type": "function",
                  "function": {"name": "f", "arguments": '{"a": 1}'}}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


# --- streaming response translation via the forward path ------------------------

def _fake_request(headers: dict) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


def _anthropic_config(key="a-key"):
    return SimpleNamespace(
        anthropic_base_url="https://anthropic.test",
        anthropic_api_key=key,
        anthropic_version="2023-06-01",
    )


def test_forward_translates_request_and_streams_openai_chunks(monkeypatch):
    monkeypatch.setattr(forward, "config", _anthropic_config())

    sse = (
        b'event: message_start\ndata: {"type":"message_start"}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        forward, "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    route = forward.build_route("anthropic", "anthropic/claude-sonnet-4-5")
    body = json.dumps({
        "model": "placeholder",
        "stream": True,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    }).encode()
    resp = asyncio.run(
        forward.forward(_fake_request({"authorization": "Bearer client"}), body, route)
    )

    # Upstream request: /v1/messages, x-api-key auth, translated body.
    sent = captured[0]
    assert str(sent.url) == "https://anthropic.test/v1/messages"
    assert sent.headers["x-api-key"] == "a-key"
    assert "authorization" not in sent.headers
    assert sent.headers["anthropic-version"] == "2023-06-01"
    sent_body = json.loads(sent.content)
    assert sent_body["model"] == "claude-sonnet-4-5"
    assert sent_body["system"] == "sys"

    # Downstream response: OpenAI SSE chunks ending in [DONE].
    async def drain():
        return b"".join([part async for part in resp.body_iterator])

    text = asyncio.run(drain()).decode()
    assert "chat.completion.chunk" in text
    assert '"content": "Hi"' in text
    assert '"finish_reason": "stop"' in text
    assert text.rstrip().endswith("data: [DONE]")


def test_forward_missing_key_fails_loud(monkeypatch):
    monkeypatch.setattr(forward, "config", _anthropic_config(key=None))
    route = forward.build_route("anthropic", "anthropic/claude-x")
    resp = asyncio.run(forward.forward(_fake_request({}), b'{"messages":[]}', route))
    assert resp.status_code == 503
    assert "anthropic" in json.loads(bytes(resp.body))["error"]


def test_forward_translates_non_streaming_response(monkeypatch):
    monkeypatch.setattr(forward, "config", _anthropic_config())
    anthropic_resp = json.dumps({
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=anthropic_resp,
                              headers={"content-type": "application/json"})

    monkeypatch.setattr(
        forward, "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    route = forward.build_route("anthropic", "claude-x")
    resp = asyncio.run(
        forward.forward(_fake_request({}), b'{"messages":[{"role":"user","content":"hi"}]}', route)
    )
    out = json.loads(bytes(resp.body))
    assert out["choices"][0]["message"]["content"] == "done"
