"""Stage 3 of 5. See LLM_Diagnostic_Assistant_PRD.md Section 7-9, FR-5/FR-6/FR-7.
No LLM call — retrieval is deterministic SQL, never embedding similarity
(PRD Section 9's explicit instruction).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from shared.schemas import CitedFact
from shared.schemas._base import StrictModel


class ThresholdRow(StrictModel):
    signal_name: str
    threshold_value: float
    threshold_direction: str
    sustained_days: int | None
    notes: str | None


class RetrievedCategory(StrictModel):
    category_id: int
    name: str
    description: str
    thresholds: list[ThresholdRow]
    cited_facts: list[CitedFact]


class RetrievalResult(StrictModel):
    matched_categories: list[RetrievedCategory]
    is_empty: bool


def _keyword_matches(keyword: str, lowered_text: str) -> bool:
    """Word-boundary match, not a raw substring check — a single-word
    keyword like 'thi' must not match inside 'this'/'think', and (found via
    testing) a keyword like the now-removed 'rest' must not match inside
    unrelated text like 'the rest of the herd'."""
    return re.search(rf"\b{re.escape(keyword.lower())}\b", lowered_text) is not None


def retrieve_facts(driving_signals: list[str], raw_text: str | None, *, db_path: str | Path) -> RetrievalResult:
    """FR-5: driving_signals matched via exact SQL equality against
    shift_thresholds.signal_name (see kb/seed_data/shift_categories.yaml's
    header comment for the naming convention that makes this an exact
    match, not a fuzzy one). Free-text entities (when raw_text is present)
    matched via a plain substring check against each category's keywords —
    also deterministic, no embeddings anywhere in this function.

    A category matched by either mechanism returns ALL of its threshold
    rows, not just the one that matched — this is the retrieval mechanism
    FR-10 depends on: matching "cbt" alone still returns the thi_context
    row, because they share a category_id.

    FR-6: every CitedFact carries a source pointing back to its
    literature_links row. FR-7: this function is a pure SQL read with no
    LLM involvement; callers (the orchestrator) log it independently via
    audit_log.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        category_ids: set[int] = set()

        if driving_signals:
            placeholders = ",".join("?" for _ in driving_signals)
            rows = conn.execute(
                f"SELECT DISTINCT category_id FROM shift_thresholds WHERE signal_name IN ({placeholders})",
                driving_signals,
            ).fetchall()
            category_ids.update(row["category_id"] for row in rows)

        if raw_text:
            lowered = raw_text.lower()
            for row in conn.execute("SELECT id, keywords FROM shift_categories"):
                keywords = json.loads(row["keywords"])
                if any(_keyword_matches(keyword, lowered) for keyword in keywords):
                    category_ids.add(row["id"])

        matched = [_load_category(conn, category_id) for category_id in sorted(category_ids)]
        return RetrievalResult(matched_categories=matched, is_empty=not matched)
    finally:
        conn.close()


def _load_category(conn: sqlite3.Connection, category_id: int) -> RetrievedCategory:
    category_row = conn.execute("SELECT * FROM shift_categories WHERE id = ?", (category_id,)).fetchone()
    threshold_rows = conn.execute(
        "SELECT * FROM shift_thresholds WHERE category_id = ?", (category_id,)
    ).fetchall()
    link_rows = conn.execute("SELECT * FROM literature_links WHERE category_id = ?", (category_id,)).fetchall()

    thresholds = [
        ThresholdRow(
            signal_name=row["signal_name"],
            threshold_value=row["threshold_value"],
            threshold_direction=row["threshold_direction"],
            sustained_days=row["sustained_days"],
            notes=row["notes"],
        )
        for row in threshold_rows
    ]
    cited_facts = [
        CitedFact(
            fact_id=f"fact-{row['id']}",
            source=f"literature_links:{row['id']}",
            text=row["summary_text"],
        )
        for row in link_rows
    ]
    return RetrievedCategory(
        category_id=category_id,
        name=category_row["name"],
        description=category_row["description"],
        thresholds=thresholds,
        cited_facts=cited_facts,
    )
