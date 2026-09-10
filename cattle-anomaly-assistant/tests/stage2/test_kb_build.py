from __future__ import annotations

import sqlite3

import pytest

from stage2_rag_assistant.kb.build_kb import DEFAULT_SCHEMA_PATH, DEFAULT_SEED_PATH, build_kb


def test_build_kb_loads_all_seed_categories(tmp_kb_db):
    conn = sqlite3.connect(tmp_kb_db)
    try:
        (category_count,) = conn.execute("SELECT COUNT(*) FROM shift_categories").fetchone()
        (threshold_count,) = conn.execute("SELECT COUNT(*) FROM shift_thresholds").fetchone()
        (literature_count,) = conn.execute("SELECT COUNT(*) FROM literature_links").fetchone()
        assert category_count == 5
        assert threshold_count >= 5  # temperature category alone has 2
        assert literature_count == 5
    finally:
        conn.close()


def test_build_kb_enforces_threshold_direction_check(tmp_path):
    db_path = tmp_path / "kb.sqlite3"
    build_kb(schema_path=DEFAULT_SCHEMA_PATH, seed_path=DEFAULT_SEED_PATH, db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO shift_thresholds (category_id, signal_name, threshold_value, threshold_direction) "
                "VALUES (1, 'cbt', 1.0, 'sideways')"
            )
    finally:
        conn.close()


def test_build_kb_refuses_to_overwrite_by_default(tmp_path):
    db_path = tmp_path / "kb.sqlite3"
    build_kb(schema_path=DEFAULT_SCHEMA_PATH, seed_path=DEFAULT_SEED_PATH, db_path=db_path)
    with pytest.raises(FileExistsError):
        build_kb(schema_path=DEFAULT_SCHEMA_PATH, seed_path=DEFAULT_SEED_PATH, db_path=db_path)


def test_build_kb_is_idempotent_with_overwrite(tmp_path):
    db_path = tmp_path / "kb.sqlite3"
    build_kb(schema_path=DEFAULT_SCHEMA_PATH, seed_path=DEFAULT_SEED_PATH, db_path=db_path)
    build_kb(schema_path=DEFAULT_SCHEMA_PATH, seed_path=DEFAULT_SEED_PATH, db_path=db_path, overwrite=True)
    conn = sqlite3.connect(db_path)
    try:
        (category_count,) = conn.execute("SELECT COUNT(*) FROM shift_categories").fetchone()
        assert category_count == 5  # not 10 — rebuilt, not appended
    finally:
        conn.close()


def test_cbt_and_thi_context_share_a_category(tmp_kb_db):
    """The mechanism FR-10 depends on: a 'cbt' lookup must pull back the
    'thi_context' row because they share a category_id, even though
    'thi_context' is never itself a driving signal."""
    conn = sqlite3.connect(tmp_kb_db)
    try:
        rows = conn.execute(
            "SELECT signal_name, category_id FROM shift_thresholds WHERE signal_name IN ('cbt', 'thi_context')"
        ).fetchall()
        by_signal = dict(rows)
        assert by_signal["cbt"] == by_signal["thi_context"]
    finally:
        conn.close()
