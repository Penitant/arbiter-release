-- Ferry session-state schema. Operational state, not a permanent log;
-- separate from the corrections table. Every statement is idempotent.

CREATE TABLE IF NOT EXISTS session_model_state (
    session_id      TEXT PRIMARY KEY,
    last_model      TEXT NOT NULL,
    parent_session  TEXT,              -- NULL for top-level sessions
    updated_at      TEXT NOT NULL      -- ISO 8601, UTC
);
