from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.openai.adapter import (
    create_chat_completion,
    create_chat_completion_stream,
    warning_headers,
)
from app.openai.schemas import ChatCompletionRequest

router = APIRouter(prefix="/v1", tags=["openai"])


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    if payload.stream:
        iterator, warnings = await create_chat_completion_stream(payload)
        headers = warning_headers(warnings)
        headers["Cache-Control"] = "no-cache"

        return StreamingResponse(
            iterator,
            media_type="text/event-stream",
            headers=headers,
        )

    response_payload, warnings = await create_chat_completion(payload)
    return JSONResponse(content=response_payload, headers=warning_headers(warnings))
