from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionResponseFormatJSONSchema(BaseModel):
    name: str | None = None
    json_schema_definition: dict[str, Any] = Field(alias="schema")
    description: str | None = None
    strict: bool | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ChatCompletionResponseFormat(BaseModel):
    type: str
    json_schema: ChatCompletionResponseFormatJSONSchema | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool = False

    model_config = ConfigDict(extra="allow")


class ChatCompletionMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    stream: bool = False
    stream_options: ChatCompletionStreamOptions | None = None
    response_format: ChatCompletionResponseFormat | None = None

    # Accepted but not implemented in v1 (ignored with warning)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logprobs: bool | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None
    parallel_tool_calls: bool | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")
