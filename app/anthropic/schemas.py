from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class MessageContentBlock(BaseModel):
    type: str
    text: str | None = None

    model_config = ConfigDict(extra="allow")


class MessagesMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[MessageContentBlock]

    model_config = ConfigDict(extra="allow")


class MessagesRequest(BaseModel):
    model: str
    messages: list[MessagesMessage]
    system: str | list[MessageContentBlock] | None = None
    max_tokens: int | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    model_config = ConfigDict(extra="allow")


class CountTokensRequest(BaseModel):
    model: str
    messages: list[MessagesMessage]
    system: str | list[MessageContentBlock] | None = None

    model_config = ConfigDict(extra="allow")


class CountTokensResponse(BaseModel):
    input_tokens: int


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[dict[str, str]]
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: dict[str, int]
