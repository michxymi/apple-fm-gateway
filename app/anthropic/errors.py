from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

from app.core.errors import GatewayError

fm: Any = None
if importlib.util.find_spec("apple_fm_sdk") is not None:
    fm = importlib.import_module("apple_fm_sdk")
HAS_APPLE_FM_SDK = fm is not None


@dataclass
class AnthropicCompatError(Exception):
    status_code: int
    message: str
    error_type: str = "invalid_request_error"

    def to_error(self) -> dict[str, Any]:
        return {
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        }


def map_anthropic_error(exc: Exception) -> AnthropicCompatError:
    if isinstance(exc, AnthropicCompatError):
        return exc

    if isinstance(exc, GatewayError):
        if exc.status_code == 429:
            error_type = "rate_limit_error"
        elif exc.status_code >= 500:
            error_type = "api_error"
        else:
            error_type = "invalid_request_error"

        return AnthropicCompatError(
            status_code=exc.status_code,
            message=exc.message,
            error_type=error_type,
        )

    if HAS_APPLE_FM_SDK and isinstance(exc, fm.ExceededContextWindowSizeError):
        return AnthropicCompatError(400, str(exc), "invalid_request_error")

    if HAS_APPLE_FM_SDK and isinstance(
        exc, (fm.InvalidGenerationSchemaError, fm.UnsupportedGuideError)
    ):
        return AnthropicCompatError(400, str(exc), "invalid_request_error")

    if HAS_APPLE_FM_SDK and isinstance(
        exc, (fm.GuardrailViolationError, fm.RefusalError)
    ):
        return AnthropicCompatError(400, str(exc), "invalid_request_error")

    if HAS_APPLE_FM_SDK and isinstance(
        exc, (fm.RateLimitedError, fm.ConcurrentRequestsError)
    ):
        return AnthropicCompatError(429, str(exc), "rate_limit_error")

    if HAS_APPLE_FM_SDK and isinstance(exc, fm.AssetsUnavailableError):
        return AnthropicCompatError(503, str(exc), "api_error")

    if HAS_APPLE_FM_SDK and isinstance(
        exc, (fm.GenerationError, fm.FoundationModelsError)
    ):
        return AnthropicCompatError(500, str(exc), "api_error")

    return AnthropicCompatError(500, f"Unexpected server error: {exc}", "api_error")
