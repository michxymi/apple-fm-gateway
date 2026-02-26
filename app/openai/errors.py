from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import apple_fm_sdk as fm

from app.core.errors import GatewayError


@dataclass
class OpenAICompatError(Exception):
    """OpenAI-style error wrapper with HTTP metadata."""

    status_code: int
    message: str
    error_type: str = "invalid_request_error"
    code: str | None = None
    param: str | None = None

    def to_error(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "type": self.error_type,
            "param": self.param,
            "code": self.code,
        }


def map_apple_fm_error(exc: Exception) -> OpenAICompatError:
    """Map Foundation Models exceptions to OpenAI-style API errors."""

    if isinstance(exc, OpenAICompatError):
        return exc

    if isinstance(exc, GatewayError):
        if exc.status_code == 429:
            error_type = "rate_limit_error"
        elif exc.status_code >= 500:
            error_type = "server_error"
        else:
            error_type = "invalid_request_error"

        return OpenAICompatError(
            status_code=exc.status_code,
            message=exc.message,
            error_type=error_type,
            code=exc.code,
            param=exc.param,
        )

    if isinstance(exc, fm.ExceededContextWindowSizeError):
        return OpenAICompatError(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="context_length_exceeded",
        )

    if isinstance(exc, fm.InvalidGenerationSchemaError):
        return OpenAICompatError(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_json_schema",
            param="response_format",
        )

    if isinstance(exc, (fm.UnsupportedGuideError, fm.UnsupportedLanguageOrLocaleError)):
        return OpenAICompatError(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="unsupported_parameter",
        )

    if isinstance(exc, (fm.GuardrailViolationError, fm.RefusalError)):
        return OpenAICompatError(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="content_policy_violation",
        )

    if isinstance(exc, (fm.RateLimitedError, fm.ConcurrentRequestsError)):
        return OpenAICompatError(
            status_code=429,
            message=str(exc),
            error_type="rate_limit_error",
            code="rate_limited",
        )

    if isinstance(exc, fm.AssetsUnavailableError):
        return OpenAICompatError(
            status_code=503,
            message=str(exc),
            error_type="server_error",
            code="assets_unavailable",
        )

    if isinstance(exc, fm.DecodingFailureError):
        return OpenAICompatError(
            status_code=500,
            message=str(exc),
            error_type="server_error",
            code="decoding_failure",
        )

    if isinstance(exc, fm.GenerationError):
        return OpenAICompatError(
            status_code=500,
            message=str(exc),
            error_type="server_error",
            code="generation_error",
        )

    if isinstance(exc, fm.FoundationModelsError):
        return OpenAICompatError(
            status_code=500,
            message=str(exc),
            error_type="server_error",
            code="foundation_models_error",
        )

    return OpenAICompatError(
        status_code=500,
        message=f"Unexpected server error: {exc}",
        error_type="server_error",
        code="internal_error",
    )
