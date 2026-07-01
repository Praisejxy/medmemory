"""
Structured fact store (SQLite).

Every remembered fact about a user lives here as a row with:
  - a fact_type (medication, allergy, diagnosis, symptom, preference, other)
  - an importance weight (0-1, set at extraction time)
  - a permanent flag (allergies / chronic diagnoses should NEVER decay away)
  - confidence bookkeeping used by decay.py to age facts out of relevance
  - a superseded_by pointer used for contradiction resolution, so we keep
    full history instead of silently overwriting facts.
"""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    permanent INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_reinforced_at REAL NOT NULL,
    reinforce_count INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(active);
"""


class FactStore:
    def __init__(self, db_path: str = "medmemory.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def insert_fact(self, user_id: str, fact_type: str, content: str,
                     importance: float = 0.5, permanent: bool = False) -> str:
        fact_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO facts
                   (id, user_id, fact_type, content, importance, permanent,
                    created_at, last_reinforced_at, reinforce_count, active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1)""",
                (fact_id, user_id, fact_type, content, importance,
                 int(permanent), now, now),
            )
        return fact_id

    def reinforce_fact(self, fact_id: str):
        """Bump last_reinforced_at + count when a fact is mentioned again,
        resetting its decay clock."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE facts SET last_reinforced_at = ?,
                   reinforce_count = reinforce_count + 1 WHERE id = ?""",
                (time.time(), fact_id),
            )

    def supersede_fact(self, old_fact_id: str, new_fact_id: str):
        """Mark an old fact inactive because a newer, contradicting fact
        replaced it. History is preserved (row stays, just deactivated)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE facts SET active = 0, superseded_by = ? WHERE id = ?",
                (new_fact_id, old_fact_id),
            )

    def get_active_facts(self, user_id: str, fact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if fact_type:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE user_id = ? AND active = 1 AND fact_type = ?",
                    (user_id, fact_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE user_id = ? AND active = 1",
                    (user_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return dict(row) if row else None

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Full history including superseded facts — useful for the demo,
        to show the agent's reasoning over time, not just current state."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE user_id = ? ORDER BY created_at ASC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
