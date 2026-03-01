from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any, AsyncIterator

from .errors import GatewayError
from .types import CANONICAL_MODEL_ID, NormalizedGenerationRequest

if TYPE_CHECKING:
    import apple_fm_sdk as fm_types

fm: Any = None
if importlib.util.find_spec("apple_fm_sdk") is not None:
    fm = importlib.import_module("apple_fm_sdk")
HAS_APPLE_FM_SDK = fm is not None


async def generate_response_text(request: NormalizedGenerationRequest) -> str:
    _validate_request(request)
    session = _create_session(request)

    if request.json_schema is None:
        return await session.respond(request.conversation_prompt)

    generated = await session.respond(
        request.conversation_prompt,
        json_schema=request.json_schema,
    )
    return generated.to_json()


async def stream_response_deltas(
    request: NormalizedGenerationRequest,
) -> AsyncIterator[str]:
    _validate_request(request)

    if request.json_schema is not None:
        raise GatewayError(
            status_code=400,
            message=(
                "stream=true is not supported when response_format.type is json_schema."
            ),
            code="unsupported_combination",
            param="stream",
        )

    session = _create_session(request)

    previous_snapshot = ""
    async for snapshot in session.stream_response(request.conversation_prompt):
        if snapshot.startswith(previous_snapshot):
            delta = snapshot[len(previous_snapshot) :]
        else:
            delta = snapshot

        previous_snapshot = snapshot

        if delta:
            yield delta


def _validate_request(request: NormalizedGenerationRequest) -> None:
    if request.resolved_model != CANONICAL_MODEL_ID:
        raise GatewayError(
            status_code=400,
            message=(
                f"Unknown model '{request.resolved_model}'. "
                f"This server currently supports only '{CANONICAL_MODEL_ID}'."
            ),
            code="model_not_found",
            param="model",
        )


def _create_session(
    request: NormalizedGenerationRequest,
) -> "fm_types.LanguageModelSession":
    if not HAS_APPLE_FM_SDK:
        raise GatewayError(
            status_code=503,
            message=("Foundation model SDK is not installed in this environment."),
            code="model_unavailable",
        )

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()

    if not is_available:
        reason_name = getattr(reason, "name", str(reason) if reason else "UNKNOWN")
        raise GatewayError(
            status_code=503,
            message=(
                "Foundation model is unavailable on this machine "
                f"(reason={reason_name})."
            ),
            code="model_unavailable",
        )

    if request.instructions:
        return fm.LanguageModelSession(instructions=request.instructions, model=model)

    return fm.LanguageModelSession(model=model)
