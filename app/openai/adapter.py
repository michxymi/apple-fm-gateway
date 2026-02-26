from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.core.generation import generate_response_text, stream_response_deltas
from app.core.token_estimation import estimate_tokens
from app.core.types import CANONICAL_MODEL_ID, NormalizedGenerationRequest

from .errors import OpenAICompatError, map_apple_fm_error
from .schemas import ChatCompletionMessage, ChatCompletionRequest


@dataclass
class PreparedChatRequest:
    model: str
    instructions: str | None
    prompt: str
    prompt_tokens: int
    json_schema: dict[str, Any] | None
    include_stream_usage: bool
    warnings: list[str]


def canonical_model_card() -> dict[str, Any]:
    return {
        "id": CANONICAL_MODEL_ID,
        "object": "model",
        "created": 0,
        "owned_by": "apple",
    }


def warning_headers(warnings: list[str]) -> dict[str, str]:
    if not warnings:
        return {}

    value = " | ".join(_dedupe_preserve_order(warnings))
    if len(value) > 2048:
        value = value[:2045] + "..."
    return {"X-OpenAI-Compat-Warnings": value}


def prepare_chat_request(request: ChatCompletionRequest) -> PreparedChatRequest:
    _validate_model(request.model)
    instructions, prompt, prompt_tokens, message_warnings = _flatten_messages(request.messages)
    json_schema = _extract_json_schema(request)

    if request.stream and json_schema is not None:
        raise OpenAICompatError(
            status_code=400,
            message="stream=true is not supported when response_format.type is json_schema.",
            error_type="invalid_request_error",
            code="unsupported_combination",
            param="stream",
        )

    warnings = _collect_warnings(request)
    warnings.extend(message_warnings)

    return PreparedChatRequest(
        model=request.model,
        instructions=instructions,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        json_schema=json_schema,
        include_stream_usage=bool(
            request.stream_options is not None and request.stream_options.include_usage
        ),
        warnings=_dedupe_preserve_order(warnings),
    )


async def create_chat_completion(
    request: ChatCompletionRequest,
) -> tuple[dict[str, Any], list[str]]:
    prepared = prepare_chat_request(request)
    completion_id = _new_chat_completion_id()
    created_at = int(time.time())

    normalized_request = NormalizedGenerationRequest(
        resolved_model=prepared.model,
        instructions=prepared.instructions,
        conversation_prompt=prepared.prompt,
        stream=False,
        json_schema=prepared.json_schema,
    )

    try:
        content = await generate_response_text(normalized_request)
    except Exception as exc:
        raise map_apple_fm_error(exc) from exc

    completion_text = str(content)
    completion_tokens = estimate_tokens(completion_text)

    payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_at,
        "model": prepared.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_payload(prepared.prompt_tokens, completion_tokens),
    }

    return payload, prepared.warnings


async def create_chat_completion_stream(
    request: ChatCompletionRequest,
) -> tuple[AsyncIterator[bytes], list[str]]:
    prepared = prepare_chat_request(request)
    completion_id = _new_chat_completion_id()
    created_at = int(time.time())

    normalized_request = NormalizedGenerationRequest(
        resolved_model=prepared.model,
        instructions=prepared.instructions,
        conversation_prompt=prepared.prompt,
        stream=True,
        json_schema=prepared.json_schema,
    )

    async def _iterator() -> AsyncIterator[bytes]:
        completion_parts: list[str] = []
        try:
            initial = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": prepared.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield _sse_data(initial)

            async for delta in stream_response_deltas(normalized_request):
                completion_parts.append(delta)
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": prepared.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield _sse_data(chunk)

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": prepared.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield _sse_data(final_chunk)

            if prepared.include_stream_usage:
                completion_tokens = estimate_tokens("".join(completion_parts))
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": prepared.model,
                    "choices": [],
                    "usage": _usage_payload(prepared.prompt_tokens, completion_tokens),
                }
                yield _sse_data(usage_chunk)

        except Exception as exc:
            mapped = map_apple_fm_error(exc)
            yield _sse_data({"error": mapped.to_error()})

        finally:
            yield b"data: [DONE]\n\n"

    return _iterator(), prepared.warnings


def _validate_model(model_id: str) -> None:
    if model_id != CANONICAL_MODEL_ID:
        raise OpenAICompatError(
            status_code=400,
            message=(
                f"Unknown model '{model_id}'. "
                f"This server currently supports only '{CANONICAL_MODEL_ID}'."
            ),
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )


def _extract_json_schema(request: ChatCompletionRequest) -> dict[str, Any] | None:
    if request.response_format is None:
        return None

    response_format_type = request.response_format.type

    if response_format_type == "text":
        return None

    if response_format_type != "json_schema":
        raise OpenAICompatError(
            status_code=400,
            message=(
                "Unsupported response_format.type. "
                "Only 'json_schema' is supported in v1."
            ),
            error_type="invalid_request_error",
            code="unsupported_response_format",
            param="response_format",
        )

    if request.response_format.json_schema is None:
        raise OpenAICompatError(
            status_code=400,
            message="response_format.json_schema must be provided when type is json_schema.",
            error_type="invalid_request_error",
            code="missing_json_schema",
            param="response_format",
        )

    return request.response_format.json_schema.json_schema_definition


def _flatten_messages(
    messages: list[ChatCompletionMessage],
) -> tuple[str | None, str, int, list[str]]:
    if not messages:
        raise OpenAICompatError(
            status_code=400,
            message="messages must contain at least one item.",
            error_type="invalid_request_error",
            code="empty_messages",
            param="messages",
        )

    warnings: list[str] = []
    instructions: list[str] = []
    dialogue_lines: list[str] = []
    prompt_token_parts: list[str] = []

    for idx, message in enumerate(messages):
        role = message.role.lower()
        text, content_warnings = _extract_text_content(message, idx)
        warnings.extend(content_warnings)
        if text:
            prompt_token_parts.append(text)

        if role in {"system", "developer"}:
            if text:
                instructions.append(text)
            continue

        label = _role_label(role, message)
        if text:
            dialogue_lines.append(f"{label}: {text}")
        else:
            dialogue_lines.append(f"{label}:")

    if not dialogue_lines:
        raise OpenAICompatError(
            status_code=400,
            message="messages must include at least one non-system message.",
            error_type="invalid_request_error",
            code="missing_conversation",
            param="messages",
        )

    prompt = (
        "Use the following conversation history to produce the next assistant message.\n\n"
        "Conversation:\n"
        f"{'\n'.join(dialogue_lines)}\n\n"
        "Assistant:"
    )

    merged_instructions = "\n\n".join(instructions) if instructions else None
    return merged_instructions, prompt, estimate_tokens("\n".join(prompt_token_parts)), warnings


def _extract_text_content(
    message: ChatCompletionMessage,
    message_index: int,
) -> tuple[str, list[str]]:
    content = message.content
    warnings: list[str] = []

    if content is None:
        return "", warnings

    if isinstance(content, str):
        return content.strip(), warnings

    parts: list[str] = []
    ignored_non_text = False

    for part in content:
        if not isinstance(part, dict):
            ignored_non_text = True
            continue

        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
        else:
            ignored_non_text = True

    if ignored_non_text:
        warnings.append(
            f"Ignored non-text content parts in messages[{message_index}]."
        )

    return "".join(parts).strip(), warnings


def _role_label(role: str, message: ChatCompletionMessage) -> str:
    if role == "user":
        return "User"
    if role == "assistant":
        return "Assistant"
    if role in {"tool", "function"}:
        if message.name:
            return f"Tool({message.name})"
        if message.tool_call_id:
            return f"Tool[{message.tool_call_id}]"
        return "Tool"

    return role.capitalize()


def _collect_warnings(request: ChatCompletionRequest) -> list[str]:
    warnings: list[str] = []

    if request.tools is not None or request.tool_choice is not None:
        warnings.append(
            "Received tools/tool_choice, but tool calling is ignored in v1."
        )

    ignored_fields: list[str] = []
    for field_name in (
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "n",
        "stop",
        "seed",
        "user",
        "parallel_tool_calls",
        "metadata",
    ):
        value = getattr(request, field_name)
        if value is not None:
            ignored_fields.append(field_name)

    if request.model_extra:
        ignored_fields.extend(sorted(request.model_extra.keys()))

    if ignored_fields:
        warnings.append(
            "Ignored unsupported request fields: "
            + ", ".join(_dedupe_preserve_order(ignored_fields))
        )

    return warnings


def _usage_payload(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _new_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _sse_data(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []

    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    return deduped
