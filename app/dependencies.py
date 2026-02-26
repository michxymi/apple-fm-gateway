from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.anthropic.errors import AnthropicCompatError
from app.openai.errors import OpenAICompatError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(OpenAICompatError)
    async def handle_openai_error(
        _request: Request,
        exc: OpenAICompatError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.to_error()},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        first_error = exc.errors()[0]["msg"] if exc.errors() else "Invalid request"

        if request.url.path.startswith("/v1/messages"):
            compat_error = AnthropicCompatError(
                status_code=400,
                message=first_error,
                error_type="invalid_request_error",
            )
            return JSONResponse(
                status_code=compat_error.status_code,
                content=compat_error.to_error(),
            )

        compat_error = OpenAICompatError(
            status_code=400,
            message=first_error,
            error_type="invalid_request_error",
            code="invalid_request",
        )
        return JSONResponse(
            status_code=compat_error.status_code,
            content={"error": compat_error.to_error()},
        )

    @app.exception_handler(AnthropicCompatError)
    async def handle_anthropic_error(
        _request: Request,
        exc: AnthropicCompatError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_error(),
        )
