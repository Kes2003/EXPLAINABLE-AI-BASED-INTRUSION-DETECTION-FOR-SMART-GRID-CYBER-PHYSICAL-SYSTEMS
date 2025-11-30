from __future__ import annotations

from pathlib import Path
import json
import random
import os
import sqlite3
from typing import Any, Dict, Optional
from contextlib import contextmanager

def ensure_dir(p: Path | str):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path | str):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def project_path(*parts: str) -> Path:
    """Return a Path relative to the project root (repo root)."""
    base = Path(__file__).resolve().parents[1]
    return base.joinpath(*parts)


# -----------------------------
# Simple SQLite-based audit trail
# -----------------------------
@contextmanager
def get_audit_connection(db_path: Path | str):
    """
    Context manager for SQLite connections to ensure proper cleanup.
    """
    dbp = Path(db_path)
    ensure_dir(dbp.parent)
    conn = sqlite3.connect(str(dbp), timeout=10.0)
    try:
        yield conn
    finally:
        conn.close()


def init_audit_db(db_path: Path | str):
    """
    Initialize the audit SQLite database (create table if missing) and indexes.
    Table schema:
      - id INTEGER PRIMARY KEY AUTOINCREMENT
      - created_at TEXT (ISO timestamp)
      - model_type TEXT
      - features_json TEXT
      - predicted_label TEXT
      - probability REAL
      - action_json TEXT
    """
    with get_audit_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_type TEXT,
                features_json TEXT,
                predicted_label TEXT,
                probability REAL,
                action_json TEXT
            )
            """
        )
        # useful indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(predicted_label)")
        conn.commit()
    return Path(db_path)


def log_audit_entry(
    db_path: Path | str,
    model_type: str | None,
    features: Dict[str, Any],
    predicted_label: str | None,
    probability: float | None,
    action: Optional[Dict[str, Any]] = None,
):
    """
    Insert a prediction / action record into the audit DB.
    Uses context manager to ensure proper connection cleanup.
    """
    with get_audit_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (created_at, model_type, features_json, predicted_label, probability, action_json)
            VALUES (datetime('now'), ?, ?, ?, ?, ?)
            """,
            (
                model_type or "unknown",
                json.dumps(features, ensure_ascii=False),
                str(predicted_label) if predicted_label is not None else None,
                float(probability) if probability is not None else None,
                json.dumps(action, ensure_ascii=False) if action is not None else None,
            ),
        )
        conn.commit()