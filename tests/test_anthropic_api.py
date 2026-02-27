from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.anthropic.adapter as anthropic_adapter
from app.core.errors import GatewayError
from app.main import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def patch_anthropic_generation(monkeypatch):
    captured = {
        "request": None,
    }

    async def fake_generate_response_text(request):
        captured["request"] = request
        return "anthropic stub completion"

    async def fake_stream_response_deltas(request):
        captured["request"] = request
        for delta in ("Hello", " ", "world"):
            yield delta

    monkeypatch.setattr(
        anthropic_adapter,
        "generate_response_text",
        fake_generate_response_text,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "stream_response_deltas",
        fake_stream_response_deltas,
    )

    return captured


def test_messages_non_stream_success(client: TestClient, patch_anthropic_generation):
    payload = {
        "model": "sonnet",
        "system": "You are concise.",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hello."}],
            }
        ],
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "sonnet"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "anthropic stub completion"
    assert "usage" in body

    request = patch_anthropic_generation["request"]
    assert request is not None
    assert request.resolved_model == "apple.fm.system"
    assert request.instructions == "You are concise."
    assert "User: Say hello." in request.conversation_prompt

    warnings = response.headers.get("x-anthropic-compat-warnings")
    assert warnings is not None
    assert "Mapped model 'sonnet'" in warnings


def test_messages_streaming_events(client: TestClient, patch_anthropic_generation):
    payload = {
        "model": "haiku",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Stream please."}],
            }
        ],
    }

    with client.stream("POST", "/v1/messages", json=payload) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: message_start" in body
    assert "event: content_block_start" in body
    assert "event: content_block_delta" in body
    assert "event: content_block_stop" in body
    assert "event: message_delta" in body
    assert "event: message_stop" in body


def test_count_tokens_returns_deterministic_value(client: TestClient):
    payload = {
        "model": "sonnet",
        "system": "A",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "abcd"}],
            },
            {
                "role": "assistant",
                "content": "efgh",
            },
        ],
    }

    response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 200
    body = response.json()
    # text length is deterministic and should be > 0 for this payload
    assert body["input_tokens"] > 0


def test_model_alias_and_claude_prefix_mapping(
    client: TestClient, patch_anthropic_generation
):
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    request = patch_anthropic_generation["request"]
    assert request is not None
    assert request.resolved_model == "apple.fm.system"


def test_unknown_model_returns_400(client: TestClient):
    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_unsupported_content_block_returns_400(client: TestClient):
    payload = {
        "model": "sonnet",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "source": "abc"}],
            }
        ],
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert "Only 'text' blocks" in body["error"]["message"]


def test_system_block_array_is_supported(
    client: TestClient, patch_anthropic_generation
):
    payload = {
        "model": "opus",
        "system": [
            {"type": "text", "text": "System A. "},
            {"type": "text", "text": "System B."},
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    request = patch_anthropic_generation["request"]
    assert request is not None
    assert request.instructions == "System A. System B."


def test_backend_unavailable_maps_to_503(client: TestClient, monkeypatch):
    async def failing_generate(_request):
        raise GatewayError(
            status_code=503,
            message="Foundation model unavailable",
            code="model_unavailable",
        )

    monkeypatch.setattr(
        anthropic_adapter,
        "generate_response_text",
        failing_generate,
    )

    payload = {
        "model": "sonnet",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
