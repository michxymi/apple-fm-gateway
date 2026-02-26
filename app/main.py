from __future__ import annotations

from fastapi import FastAPI

from app.dependencies import register_exception_handlers
from app.internal import admin
from app.routers import anthropic, chat, models


def create_app() -> FastAPI:
    app = FastAPI(
        title="apple-fm-openai",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    register_exception_handlers(app)

    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(anthropic.router)
    app.include_router(admin.router)

    return app


app = create_app()
