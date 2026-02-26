from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayError(Exception):
    status_code: int
    message: str
    code: str | None = None
    param: str | None = None

    def __str__(self) -> str:
        return self.message
