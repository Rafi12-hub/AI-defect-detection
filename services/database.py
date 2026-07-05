"""
database.py
===========
SQLite storage for persistent inspection history.
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
DB_PATH = BASE / "data" / "inspections.db"


def get_conn():
    os.makedirs(str(DB_PATH.parent), exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT,
            model       TEXT,
            verdict     TEXT,
            confidence  REAL DEFAULT 0,
            num_defects INTEGER DEFAULT 0,
            details     TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_inspections_timestamp
        ON inspections(timestamp DESC)
    """)
    conn.commit()
    conn.close()


def add_inspection(filename, model, verdict, confidence, num_defects, details=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO inspections (timestamp, filename, model, verdict, confidence, num_defects, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, model, verdict, confidence, num_defects,
         json.dumps(details) if details else None)
    )
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return row_id


def get_history(limit=100, offset=0):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, filename, model, verdict, confidence, num_defects FROM inspections ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history_count():
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    conn.close()
    return count


def get_history_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    passed = conn.execute("SELECT COUNT(*) FROM inspections WHERE verdict NOT IN ('FAIL', 'DEFECTIVE', 'ANOMALY')").fetchone()[0]
    failed = total - passed
    avg_conf = conn.execute("SELECT COALESCE(AVG(confidence), 0) FROM inspections").fetchone()[0]
    conn.close()
    return {"total": total, "passed": passed, "failed": failed, "avg_confidence": round(avg_conf, 4)}


def clear_history():
    conn = get_conn()
    conn.execute("DELETE FROM inspections")
    conn.commit()
    conn.close()


def export_history_csv():
    import csv, io
    rows = get_history(limit=10000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "Timestamp", "File", "Model", "Verdict", "Confidence", "Defects"])
    for r in rows:
        w.writerow([r["id"], r["timestamp"], r["filename"], r["model"], r["verdict"], r["confidence"], r["num_defects"]])
    return buf.getvalue()


def get_recent_for_dashboard(limit=10):
    return get_history(limit=limit)


init_db()