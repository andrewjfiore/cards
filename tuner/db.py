"""
db.py — SQLite persistence for the card-crop tuner.

Tables:
  configs      – JSON configuration definitions
  arms         – Thompson-sampling state per config
  batches      – one row per batch run
  batch_items  – one row per image processed in a batch
  votes        – user feedback per batch_item
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "tuner.db"

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Return a thread-local connection with WAL mode and foreign keys."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create tables if they don't already exist."""
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS configs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        json_cfg    TEXT    NOT NULL,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS arms (
        config_id      INTEGER PRIMARY KEY REFERENCES configs(id),
        alpha          REAL    NOT NULL DEFAULT 1.0,
        beta           REAL    NOT NULL DEFAULT 1.0,
        last_tested_at TEXT
    );
    CREATE TABLE IF NOT EXISTS batches (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id   INTEGER NOT NULL REFERENCES configs(id),
        config_id_b INTEGER REFERENCES configs(id),
        started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        finished_at TEXT,
        input_dir   TEXT    NOT NULL,
        output_root TEXT    NOT NULL,
        mode        TEXT    NOT NULL DEFAULT 'single'
    );
    CREATE TABLE IF NOT EXISTS batch_items (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id    INTEGER NOT NULL REFERENCES batches(id),
        filename    TEXT    NOT NULL,
        output_path TEXT,
        output_path_b TEXT,
        debug_path  TEXT,
        debug_path_b TEXT,
        status      TEXT    NOT NULL DEFAULT 'pending',
        status_b    TEXT,
        strategy    TEXT,
        strategy_b  TEXT
    );
    CREATE TABLE IF NOT EXISTS votes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_item_id INTEGER NOT NULL REFERENCES batch_items(id),
        vote          TEXT    NOT NULL,
        confidence    TEXT    NOT NULL DEFAULT 'sure',
        reason_tags   TEXT    NOT NULL DEFAULT '[]',
        pairwise_winner TEXT,
        created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """)
    conn.commit()


# ── helpers ──────────────────────────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def dict_from_row(row):
    if row is None:
        return None
    return dict(row)


def dicts_from_rows(rows):
    return [dict(r) for r in rows]


# ── configs ──────────────────────────────────────────────────────────────────

def insert_config(cfg: dict) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO configs (json_cfg, created_at) VALUES (?, ?)",
        (json.dumps(cfg, sort_keys=True), _now()),
    )
    config_id = cur.lastrowid
    conn.execute(
        "INSERT INTO arms (config_id, alpha, beta) VALUES (?, 1.0, 1.0)",
        (config_id,),
    )
    conn.commit()
    return config_id


def get_configs():
    return dicts_from_rows(_conn().execute("""
        SELECT c.id, c.json_cfg, c.created_at,
               a.alpha, a.beta, a.last_tested_at
        FROM configs c JOIN arms a ON c.id = a.config_id
        ORDER BY c.id
    """).fetchall())


def get_config(config_id: int):
    return dict_from_row(_conn().execute(
        "SELECT c.id, c.json_cfg, c.created_at, a.alpha, a.beta, a.last_tested_at "
        "FROM configs c JOIN arms a ON c.id = a.config_id WHERE c.id = ?",
        (config_id,),
    ).fetchone())


def config_exists(cfg: dict) -> bool:
    key = json.dumps(cfg, sort_keys=True)
    row = _conn().execute(
        "SELECT 1 FROM configs WHERE json_cfg = ?", (key,)
    ).fetchone()
    return row is not None


# ── arms ─────────────────────────────────────────────────────────────────────

def get_all_arms():
    return dicts_from_rows(_conn().execute(
        "SELECT config_id, alpha, beta, last_tested_at FROM arms"
    ).fetchall())


def update_arm(config_id: int, d_alpha: float, d_beta: float):
    conn = _conn()
    conn.execute(
        "UPDATE arms SET alpha = alpha + ?, beta = beta + ?, last_tested_at = ? WHERE config_id = ?",
        (d_alpha, d_beta, _now(), config_id),
    )
    conn.commit()


def reset_arms():
    conn = _conn()
    conn.execute("UPDATE arms SET alpha = 1.0, beta = 1.0, last_tested_at = NULL")
    conn.execute("DELETE FROM votes")
    conn.commit()


# ── batches ──────────────────────────────────────────────────────────────────

def create_batch(config_id: int, input_dir: str, output_root: str,
                 config_id_b: int = None, mode: str = "single") -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO batches (config_id, config_id_b, input_dir, output_root, started_at, mode) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (config_id, config_id_b, input_dir, output_root, _now(), mode),
    )
    conn.commit()
    return cur.lastrowid


def finish_batch(batch_id: int):
    conn = _conn()
    conn.execute(
        "UPDATE batches SET finished_at = ? WHERE id = ?", (_now(), batch_id)
    )
    conn.commit()


def get_batch(batch_id: int):
    return dict_from_row(_conn().execute(
        "SELECT * FROM batches WHERE id = ?", (batch_id,)
    ).fetchone())


def get_batches():
    return dicts_from_rows(_conn().execute(
        "SELECT * FROM batches ORDER BY id DESC"
    ).fetchall())


def get_last_batch_config_id():
    row = _conn().execute(
        "SELECT config_id FROM batches ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["config_id"] if row else None


# ── batch_items ──────────────────────────────────────────────────────────────

def add_batch_item(batch_id: int, filename: str, output_path: str = "",
                   debug_path: str = "", status: str = "pending",
                   strategy: str = "",
                   output_path_b: str = "", debug_path_b: str = "",
                   status_b: str = None, strategy_b: str = None) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO batch_items "
        "(batch_id, filename, output_path, debug_path, status, strategy,"
        " output_path_b, debug_path_b, status_b, strategy_b) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, filename, output_path, debug_path, status, strategy,
         output_path_b, debug_path_b, status_b, strategy_b),
    )
    conn.commit()
    return cur.lastrowid


def update_batch_item(item_id: int, **kwargs):
    conn = _conn()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [item_id]
    conn.execute(f"UPDATE batch_items SET {sets} WHERE id = ?", vals)
    conn.commit()


def get_batch_items(batch_id: int):
    return dicts_from_rows(_conn().execute(
        "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY id", (batch_id,)
    ).fetchall())


# ── votes ────────────────────────────────────────────────────────────────────

def add_vote(batch_item_id: int, vote: str, confidence: str = "sure",
             reason_tags: list = None, pairwise_winner: str = None) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO votes (batch_item_id, vote, confidence, reason_tags, pairwise_winner, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (batch_item_id, vote, confidence,
         json.dumps(reason_tags or []), pairwise_winner, _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_votes_for_batch(batch_id: int):
    return dicts_from_rows(_conn().execute("""
        SELECT v.* FROM votes v
        JOIN batch_items bi ON v.batch_item_id = bi.id
        WHERE bi.batch_id = ?
        ORDER BY v.id
    """, (batch_id,)).fetchall())


def get_all_votes():
    return dicts_from_rows(_conn().execute("""
        SELECT v.id, v.batch_item_id, bi.batch_id, bi.filename,
               b.config_id, v.vote, v.confidence, v.reason_tags,
               v.pairwise_winner, v.created_at
        FROM votes v
        JOIN batch_items bi ON v.batch_item_id = bi.id
        JOIN batches b ON bi.batch_id = b.id
        ORDER BY v.id
    """).fetchall())


def get_vote_stats():
    """Per-config aggregated vote stats."""
    return dicts_from_rows(_conn().execute("""
        SELECT b.config_id,
               COUNT(*)                                          AS total_votes,
               SUM(CASE WHEN v.vote = 'up' THEN 1 ELSE 0 END)  AS ups,
               SUM(CASE WHEN v.vote = 'down' THEN 1 ELSE 0 END) AS downs,
               SUM(CASE WHEN v.vote = 'failure' THEN 1 ELSE 0 END) AS failures,
               SUM(CASE WHEN v.vote = 'uncertain' THEN 1 ELSE 0 END) AS uncertains,
               SUM(CASE WHEN v.vote = 'skip' THEN 1 ELSE 0 END) AS skips
        FROM votes v
        JOIN batch_items bi ON v.batch_item_id = bi.id
        JOIN batches b ON bi.batch_id = b.id
        GROUP BY b.config_id
    """).fetchall())
