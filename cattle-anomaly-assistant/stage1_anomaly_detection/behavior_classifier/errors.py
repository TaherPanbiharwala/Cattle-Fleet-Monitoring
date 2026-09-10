"""Stable, safe-to-display failures for the behavior CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BehaviorError(Exception):
    code: str
    message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def fail(code: str, message: str, **details: Any) -> None:
    raise BehaviorError(code=code, message=message, details=details)
