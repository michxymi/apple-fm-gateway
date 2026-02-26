from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/internal", tags=["internal"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
