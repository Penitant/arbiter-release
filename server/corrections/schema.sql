-- Correction log schema; every statement is idempotent so re-running is harmless.

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_text       TEXT NOT NULL,
    predicted_tier  TEXT NOT NULL,
    predicted_model TEXT NOT NULL,
    confidence      REAL NOT NULL,
    corrected_tier  TEXT,              -- NULL if accepted as-is
    corrected_model TEXT,              -- NULL if accepted as-is
    session_id      TEXT NOT NULL,
    created_at      TEXT NOT NULL      -- ISO 8601, UTC
);

CREATE INDEX IF NOT EXISTS idx_corrections_created_at ON corrections(created_at);
CREATE INDEX IF NOT EXISTS idx_corrections_session_id ON corrections(session_id);
CREATE INDEX IF NOT EXISTS idx_corrections_was_overridden
    ON corrections(corrected_tier)
    WHERE corrected_tier IS NOT NULL;

-- Single-row cursor for the incremental fine-tune; advanced only after a successful update.
CREATE TABLE IF NOT EXISTS finetune_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    last_processed_id   INTEGER NOT NULL DEFAULT 0,
    last_run_at         TEXT
);

INSERT OR IGNORE INTO finetune_state (id, last_processed_id) VALUES (1, 0);
