from __future__ import annotations

from fastapi import APIRouter

from app.openai.adapter import canonical_model_card

router = APIRouter(prefix="/v1", tags=["openai"])


@router.get("/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [canonical_model_card()],
    }
