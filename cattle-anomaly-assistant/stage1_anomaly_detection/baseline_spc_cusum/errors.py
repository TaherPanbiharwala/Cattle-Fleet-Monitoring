"""Named, user-actionable validation failures for the detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectorError(Exception):
    """A validation failure safe to render directly in the command line."""

    code: str
    message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def fail(code: str, message: str, **details: Any) -> None:
    raise DetectorError(code=code, message=message, details=details)
