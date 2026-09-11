"""Eval set record, one per line of a JSONL golden file. See LLM_Diagnostic_Assistant_PRD.md Section 6.5.

No `gold_condition` / `requires_disambiguation` here — there is no disease label
in this system, so the fever/heat-stress/mastitis disambiguation the old
disease-diagnosis PRD needed does not apply. `injection_type` is what makes a
case checkable instead: since the shift was deliberately injected, whether the
system caught and correctly attributed it is a mechanical comparison.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ._base import StrictModel
from .anomaly_record import AnomalyRecord

Category = Literal["core", "injected", "adversarial"]
Source = Literal[
    "mmcows_real",
    "db_cow_walking_real",
    "synthetic_injection",
    "synthetic_adversarial",
]


class GoldenCase(StrictModel):
    case_id: str
    category: Category
    source: Source
    # New historical/injection builders always set these immutable values.
    # The defaults retain read support for the superseded checked-in mock
    # fixtures until a user stages and builds the public-data corpus.
    scenario_id: str | None = None
    detector_run_sha256: str | None = Field(default=None, min_length=16)
    behavior_model_sha256: str | None = Field(default=None, min_length=16)
    source_config_sha256: str | None = Field(default=None, min_length=16)
    injection_type: str | None = None
    input_record: AnomalyRecord
    query_text: str | None = None
    gold_anomaly_flag: bool
    gold_driving_signals: list[str]
    gold_key_facts: list[str]
    reviewed_by: str
