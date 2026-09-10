-- Knowledge base schema. See LLM_Diagnostic_Assistant_PRD.md Section 9.
-- Plain structured tables, no vector index — retrieval is exact-match SQL,
-- not embedding similarity (PRD Section 9's explicit instruction).

CREATE TABLE shift_categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    sources     TEXT NOT NULL,  -- JSON array of citation keys, stored as TEXT (sqlite3 has no native JSON column)
    keywords    TEXT NOT NULL   -- JSON array; addition beyond PRD Section 9's literal example, needed for
                                -- FR-5's free-text entity matching (plain substring check, still no embeddings)
);

CREATE TABLE shift_thresholds (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id         INTEGER NOT NULL REFERENCES shift_categories(id),
    signal_name         TEXT NOT NULL,  -- matches AnomalyRecord.driving_signals' short vocabulary exactly
                                        -- (e.g. 'cbt', 'lying_time'), plus the synthetic 'thi_context' value
                                        -- for context-only rows never looked up directly (see build_kb.py)
    threshold_value     REAL NOT NULL,
    threshold_direction TEXT NOT NULL CHECK (threshold_direction IN ('above', 'below')),
    sustained_days      INTEGER,
    notes               TEXT
);
CREATE INDEX idx_shift_thresholds_signal_name ON shift_thresholds(signal_name);

CREATE TABLE literature_links (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id  INTEGER NOT NULL REFERENCES shift_categories(id),
    citation     TEXT NOT NULL,
    summary_text TEXT NOT NULL
);
