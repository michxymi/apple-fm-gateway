from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.anthropic.adapter import (
    count_tokens,
    create_messages_response,
    create_messages_stream,
    warning_headers,
)
from app.anthropic.schemas import CountTokensRequest, MessagesRequest

router = APIRouter(prefix="/v1", tags=["anthropic"])


@router.post("/messages")
async def messages(payload: MessagesRequest):
    if payload.stream:
        iterator, warnings = await create_messages_stream(payload)
        headers = warning_headers(warnings)
        headers["Cache-Control"] = "no-cache"

        return StreamingResponse(
            iterator,
            media_type="text/event-stream",
            headers=headers,
        )

    response_payload, warnings = await create_messages_response(payload)
    return JSONResponse(content=response_payload, headers=warning_headers(warnings))


@router.post("/messages/count_tokens")
async def messages_count_tokens(payload: CountTokensRequest):
    return count_tokens(payload)
