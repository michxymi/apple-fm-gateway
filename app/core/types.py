from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CANONICAL_MODEL_ID = "apple.fm.system"


@dataclass(slots=True)
class NormalizedGenerationRequest:
    resolved_model: str
    instructions: str | None
    conversation_prompt: str
    stream: bool = False
    json_schema: dict[str, Any] | None = None
