"""Builds the SQLite knowledge base from schema.sql + seed_data/shift_categories.yaml.
See LLM_Diagnostic_Assistant_PRD.md Section 9.

The YAML file is the single editable source of truth (PRD Open Question 1 —
categories/thresholds/citations are expected to change after human review);
this script is idempotent by rebuilding from it, never appending to a stale
database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
DEFAULT_SCHEMA_PATH = _HERE / "schema.sql"
DEFAULT_SEED_PATH = _HERE / "seed_data" / "shift_categories.yaml"
DEFAULT_DB_PATH = _HERE / "kb.sqlite3"


def build_kb(
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    overwrite: bool = False,
) -> None:
    db_path = Path(db_path)
    if db_path.exists():
        if not overwrite:
            raise FileExistsError(f"{db_path} already exists; pass overwrite=True to rebuild")
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(Path(schema_path).read_text())
        seed = yaml.safe_load(Path(seed_path).read_text()) or {}
        for category in seed.get("categories", []):
            category_id = _insert_category(conn, category)
            for threshold in category.get("thresholds", []):
                _insert_threshold(conn, category_id, threshold)
            for link in category.get("literature_links", []):
                _insert_literature_link(conn, category_id, link)
        conn.commit()
    finally:
        conn.close()


def _insert_category(conn: sqlite3.Connection, category: dict) -> int:
    cursor = conn.execute(
        "INSERT INTO shift_categories (name, description, sources, keywords) VALUES (?, ?, ?, ?)",
        (
            category["name"],
            category["description"].strip(),
            json.dumps(category.get("sources", [])),
            json.dumps(category.get("keywords", [])),
        ),
    )
    return cursor.lastrowid


def _insert_threshold(conn: sqlite3.Connection, category_id: int, threshold: dict) -> None:
    conn.execute(
        """INSERT INTO shift_thresholds
           (category_id, signal_name, threshold_value, threshold_direction, sustained_days, notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            category_id,
            threshold["signal_name"],
            threshold["threshold_value"],
            threshold["threshold_direction"],
            threshold.get("sustained_days"),
            (threshold.get("notes") or "").strip() or None,
        ),
    )


def _insert_literature_link(conn: sqlite3.Connection, category_id: int, link: dict) -> None:
    conn.execute(
        "INSERT INTO literature_links (category_id, citation, summary_text) VALUES (?, ?, ?)",
        (category_id, link["citation"], link["summary_text"].strip()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 2 knowledge base SQLite file.")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_kb(schema_path=args.schema_path, seed_path=args.seed_path, db_path=args.db_path, overwrite=args.overwrite)
    print(f"Built KB at {args.db_path}")


if __name__ == "__main__":
    main()
