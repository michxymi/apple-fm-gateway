from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from app.core.generation import generate_response_text, stream_response_deltas
from app.core.token_estimation import estimate_tokens
from app.core.types import CANONICAL_MODEL_ID, NormalizedGenerationRequest

from .errors import AnthropicCompatError, map_anthropic_error
from .schemas import CountTokensRequest, MessageContentBlock, MessagesMessage, MessagesRequest

_MODEL_ALIASES = {
    "sonnet": CANONICAL_MODEL_ID,
    "opus": CANONICAL_MODEL_ID,
    "haiku": CANONICAL_MODEL_ID,
    CANONICAL_MODEL_ID: CANONICAL_MODEL_ID,
}


def warning_headers(warnings: list[str]) -> dict[str, str]:
    if not warnings:
        return {}

    value = " | ".join(_dedupe_preserve_order(warnings))
    if len(value) > 2048:
        value = value[:2045] + "..."
    return {"X-Anthropic-Compat-Warnings": value}


async def create_messages_response(
    request: MessagesRequest,
) -> tuple[dict[str, Any], list[str]]:
    resolved_model, warnings = _resolve_model(request.model)
    instructions = _extract_system_text(request.system)
    prompt = _flatten_messages(request.messages)

    normalized = NormalizedGenerationRequest(
        resolved_model=resolved_model,
        instructions=instructions,
        conversation_prompt=prompt,
        stream=False,
    )

    try:
        content = await generate_response_text(normalized)
    except Exception as exc:
        raise map_anthropic_error(exc) from exc

    input_tokens = estimate_input_tokens_from_messages(request)
    output_tokens = estimate_tokens(str(content))

    payload = {
        "id": _new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": [{"type": "text", "text": str(content)}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }

    return payload, warnings


async def create_messages_stream(
    request: MessagesRequest,
) -> tuple[AsyncIterator[bytes], list[str]]:
    resolved_model, warnings = _resolve_model(request.model)
    instructions = _extract_system_text(request.system)
    prompt = _flatten_messages(request.messages)

    normalized = NormalizedGenerationRequest(
        resolved_model=resolved_model,
        instructions=instructions,
        conversation_prompt=prompt,
        stream=True,
    )

    message_id = _new_message_id()
    created_at = int(time.time())
    input_tokens = estimate_input_tokens_from_messages(request)

    async def _iterator() -> AsyncIterator[bytes]:
        output_tokens = 0

        try:
            start_event = {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": request.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                    },
                    "created_at": created_at,
                },
            }
            yield _anthropic_sse_event("message_start", start_event)

            block_start_event = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": "",
                },
            }
            yield _anthropic_sse_event("content_block_start", block_start_event)

            async for delta in stream_response_deltas(normalized):
                output_tokens += estimate_tokens(delta)
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": delta,
                    },
                }
                yield _anthropic_sse_event("content_block_delta", delta_event)

            block_stop_event = {
                "type": "content_block_stop",
                "index": 0,
            }
            yield _anthropic_sse_event("content_block_stop", block_stop_event)

            message_delta_event = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": output_tokens,
                },
            }
            yield _anthropic_sse_event("message_delta", message_delta_event)

            yield _anthropic_sse_event("message_stop", {"type": "message_stop"})

        except Exception as exc:
            mapped = map_anthropic_error(exc)
            error_event = {
                "type": "error",
                "error": {
                    "type": mapped.error_type,
                    "message": mapped.message,
                },
            }
            yield _anthropic_sse_event("error", error_event)

    return _iterator(), warnings


def count_tokens(request: CountTokensRequest) -> dict[str, int]:
    _resolve_model(request.model)
    input_tokens = estimate_input_tokens_from_count_request(request)
    return {"input_tokens": input_tokens}


def estimate_input_tokens_from_messages(request: MessagesRequest) -> int:
    text_parts: list[str] = []

    system_text = _extract_system_text(request.system)
    if system_text:
        text_parts.append(system_text)

    for message in request.messages:
        text_parts.append(_extract_content_text(message.content))

    return estimate_tokens("\n".join(text_parts))


def estimate_input_tokens_from_count_request(request: CountTokensRequest) -> int:
    text_parts: list[str] = []

    system_text = _extract_system_text(request.system)
    if system_text:
        text_parts.append(system_text)

    for message in request.messages:
        text_parts.append(_extract_content_text(message.content))

    return estimate_tokens("\n".join(text_parts))


def _resolve_model(model: str) -> tuple[str, list[str]]:
    if model in _MODEL_ALIASES:
        resolved = _MODEL_ALIASES[model]
        warnings: list[str] = []
        if model != resolved:
            warnings.append(
                f"Mapped model '{model}' to backend model '{resolved}'."
            )
        return resolved, warnings

    if model.startswith("claude-"):
        resolved = CANONICAL_MODEL_ID
        return resolved, [f"Mapped model '{model}' to backend model '{resolved}'."]

    raise AnthropicCompatError(
        status_code=400,
        message=(
            f"Unknown model '{model}'. Supported aliases: "
            "sonnet, opus, haiku, claude-*, apple.fm.system."
        ),
        error_type="invalid_request_error",
    )


def _extract_system_text(system: str | list[MessageContentBlock] | None) -> str | None:
    if system is None:
        return None

    if isinstance(system, str):
        return system.strip() or None

    blocks = [_parse_block(block, field_name="system") for block in system]
    merged = "".join(blocks).strip()
    return merged or None


def _flatten_messages(messages: list[MessagesMessage]) -> str:
    if not messages:
        raise AnthropicCompatError(
            status_code=400,
            message="messages must contain at least one item.",
            error_type="invalid_request_error",
        )

    dialogue_lines: list[str] = []

    for index, message in enumerate(messages):
        text = _extract_content_text(message.content, field_name=f"messages[{index}].content")
        label = "User" if message.role == "user" else "Assistant"
        dialogue_lines.append(f"{label}: {text}" if text else f"{label}:")

    prompt = (
        "Use the following conversation history to produce the next assistant message.\n\n"
        "Conversation:\n"
        f"{'\n'.join(dialogue_lines)}\n\n"
        "Assistant:"
    )
    return prompt


def _extract_content_text(
    content: str | list[MessageContentBlock],
    *,
    field_name: str = "messages.content",
) -> str:
    if isinstance(content, str):
        return content.strip()

    blocks = [_parse_block(block, field_name=field_name) for block in content]
    return "".join(blocks).strip()


def _parse_block(block: MessageContentBlock, *, field_name: str) -> str:
    if block.type != "text":
        raise AnthropicCompatError(
            status_code=400,
            message=(
                f"Unsupported content block type '{block.type}' in {field_name}. "
                "Only 'text' blocks are supported in v1."
            ),
            error_type="invalid_request_error",
        )

    if block.text is None:
        raise AnthropicCompatError(
            status_code=400,
            message=f"Text block in {field_name} must include 'text'.",
            error_type="invalid_request_error",
        )

    return block.text


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _anthropic_sse_event(event: str, payload: dict[str, Any]) -> bytes:
    serialized = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {serialized}\n\n".encode("utf-8")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []

    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    return deduped
