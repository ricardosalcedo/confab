"""SQLite history storage."""

import json
import os
import sqlite3
from pathlib import Path

from confab.engine import Claim

DB_PATH = Path(os.environ.get("CONFAB_DB", Path.home() / ".confab" / "history.db"))

_SCHEMA = """CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY,
    timestamp TEXT DEFAULT (datetime('now')),
    command TEXT,
    prompt TEXT,
    model TEXT,
    samples INTEGER,
    elapsed REAL,
    claims_json TEXT,
    verdict TEXT
)"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.execute(_SCHEMA)
    db.commit()
    return db


def save_check(command: str, prompt: str, model: str, samples: int, elapsed: float,
               claims: list[Claim] = None, verdict: str = None):
    """Persist a check or verify result."""
    claims_json = json.dumps([
        {"text": c.text, "confidence": c.confidence, "level": c.level, "support": c.support_count}
        for c in (claims or [])
    ])
    db = _connect()
    db.execute(
        "INSERT INTO checks (command, prompt, model, samples, elapsed, claims_json, verdict) VALUES (?,?,?,?,?,?,?)",
        (command, prompt, model, samples, elapsed, claims_json, verdict),
    )
    db.commit()
    db.close()


def get_history(limit: int = 20) -> list[dict]:
    """Return recent history entries as dicts."""
    db = _connect()
    rows = db.execute(
        "SELECT id, timestamp, command, prompt, model, samples, elapsed, claims_json, verdict "
        "FROM checks ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    db.close()
    return [
        dict(zip(("id", "timestamp", "command", "prompt", "model", "samples", "elapsed", "claims_json", "verdict"), r))
        for r in rows
    ]


def clear_history():
    """Delete all history."""
    db = _connect()
    db.execute("DELETE FROM checks")
    db.commit()
    db.close()
