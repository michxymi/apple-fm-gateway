from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.openai.adapter as adapter
from app.core.errors import GatewayError
from app.core.token_estimation import estimate_tokens
from app.main import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def patch_generation_success(monkeypatch):
    captured = {
        "request": None,
    }

    async def fake_generate_response_text(request):
        captured["request"] = request
        if request.json_schema is not None:
            return json.dumps({"ok": True, "mode": "json_schema"})
        return "stub completion"

    async def fake_stream_response_deltas(request):
        captured["request"] = request
        for delta in ("s", "t", "ub"):
            yield delta

    monkeypatch.setattr(adapter, "generate_response_text", fake_generate_response_text)
    monkeypatch.setattr(adapter, "stream_response_deltas", fake_stream_response_deltas)

    return captured


def test_models_endpoint_returns_canonical_model(client: TestClient):
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "apple.fm.system"


def test_chat_completion_non_stream_success(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "apple.fm.system"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "stub completion"
    expected_prompt_tokens = estimate_tokens("Hello")
    expected_completion_tokens = estimate_tokens("stub completion")
    assert body["usage"] == {
        "prompt_tokens": expected_prompt_tokens,
        "completion_tokens": expected_completion_tokens,
        "total_tokens": expected_prompt_tokens + expected_completion_tokens,
    }


def test_chat_completion_flattens_messages_into_prompt(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "Second message"},
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    request = patch_generation_success["request"]
    assert request is not None
    assert request.instructions is not None
    assert "You are concise." in request.instructions
    assert "User: First message" in request.conversation_prompt
    assert "Assistant: Prior answer" in request.conversation_prompt
    assert "User: Second message" in request.conversation_prompt
    assert request.conversation_prompt.endswith("Assistant:")


def test_chat_completion_streaming_sse(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "stream": True,
        "messages": [{"role": "user", "content": "Stream please"}],
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    chunks: list[dict[str, object]] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        raw_payload = line.removeprefix("data: ")
        if raw_payload == "[DONE]":
            continue
        chunks.append(json.loads(raw_payload))

    assert chunks
    assert all(chunk.get("object") == "chat.completion.chunk" for chunk in chunks)
    assert all("usage" not in chunk for chunk in chunks)
    assert any(
        choice.get("finish_reason") == "stop"
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if isinstance(choice, dict)
    )
    assert "data: [DONE]" in body


def test_chat_completion_streaming_includes_usage_when_requested(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Stream please"}],
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    chunks: list[dict[str, object]] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        raw_payload = line.removeprefix("data: ")
        if raw_payload == "[DONE]":
            continue
        chunks.append(json.loads(raw_payload))

    usage_chunks = [chunk for chunk in chunks if "usage" in chunk]
    assert len(usage_chunks) == 1
    usage_chunk = usage_chunks[0]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": estimate_tokens("Stream please"),
        "completion_tokens": estimate_tokens("stub"),
        "total_tokens": estimate_tokens("Stream please") + estimate_tokens("stub"),
    }
    assert "data: [DONE]" in body


def test_chat_completion_usage_counts_text_only_prompt_content(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "messages": [
            {"role": "system", "content": "Rule"},
            {"role": "developer", "content": "Policy"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/img"}},
                ],
            },
            {"role": "assistant", "content": "Ack"},
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    expected_prompt_tokens = estimate_tokens("Rule\nPolicy\nHi\nAck")
    expected_completion_tokens = estimate_tokens("stub completion")
    assert body["usage"] == {
        "prompt_tokens": expected_prompt_tokens,
        "completion_tokens": expected_completion_tokens,
        "total_tokens": expected_prompt_tokens + expected_completion_tokens,
    }

    warnings = response.headers.get("x-openai-compat-warnings")
    assert warnings is not None
    assert "Ignored non-text content parts in messages[2]." in warnings


def test_chat_completion_json_schema_mode(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "Return object"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Demo",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    request = patch_generation_success["request"]
    assert request is not None
    assert request.json_schema is not None

    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["ok"] is True


def test_chat_completion_json_schema_stream_returns_400(client: TestClient):
    payload = {
        "model": "apple.fm.system",
        "stream": True,
        "messages": [{"role": "user", "content": "Return object"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Demo",
                "schema": {"type": "object", "properties": {}},
            },
        },
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"


def test_chat_completion_invalid_model_returns_400(client: TestClient):
    payload = {
        "model": "not-supported",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "model_not_found"


def test_chat_completion_model_unavailable_returns_503(client: TestClient, monkeypatch):
    async def failing_generate(_request):
        raise GatewayError(
            status_code=503,
            message="Foundation model unavailable",
            code="model_unavailable",
        )

    monkeypatch.setattr(adapter, "generate_response_text", failing_generate)

    payload = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "model_unavailable"


def test_chat_completion_ignores_tools_and_unsupported_fields(
    client: TestClient,
    patch_generation_success,
):
    payload = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.2,
        "unknown_field": "ignored",
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    warnings = response.headers.get("x-openai-compat-warnings")
    assert warnings is not None
    assert "tool" in warnings.lower()
    assert "temperature" in warnings
    assert "unknown_field" in warnings
