"""Per-query, per-stage structured logging. See LLM_Diagnostic_Assistant_PRD.md
Section 12: "every stage's decision... logged per query" — this is the raw
data source for the M2b ablation and Section 3's differentiation claim.

Uses Python's stdlib logging module rather than writing files directly, so
whoever eventually builds api/server.py controls handlers/rotation/output
without this module changing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger("stage2_rag_assistant.audit")


def log_event(*, query_id: str, stage: str, **fields: Any) -> None:
    record = {
        "query_id": query_id,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    _logger.info(json.dumps(record, default=str))
