"""Persistent history of simulation runs (SQLite).

Every simulation the API serves is recorded so results survive a refresh and can
be listed later via GET /runs. Recording must never break a request: failures are
logged and swallowed *here only*, never at the call sites.
"""
import json
import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("HARP_DB_PATH") or os.path.join(os.path.dirname(__file__), "results", "harp_wdn.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    kind          TEXT NOT NULL,
    network       TEXT NOT NULL,
    method        TEXT,
    scenario      TEXT,
    failed_pipes  TEXT,
    cds_overall   REAL,
    cds_critical  REAL,
    pvc           REAL,
    runtime_ms    REAL,
    escalated     INTEGER,
    extra         TEXT
)
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _conn() as conn:
        conn.execute(_SCHEMA)


def record_run(kind: str, network: str, method: str = None, scenario: str = None,
               failed_pipes=None, cds_overall: float = None, cds_critical: float = None,
               pvc: float = None, runtime_ms: float = None, escalated: bool = None,
               extra: dict = None):
    """Insert one run; returns its id, or None if the database is unavailable."""
    try:
        init()
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs (ts, kind, network, method, scenario, failed_pipes, cds_overall,"
                " cds_critical, pvc, runtime_ms, escalated, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), kind, network, method, scenario,
                 json.dumps(list(failed_pipes or [])), cds_overall, cds_critical, pvc, runtime_ms,
                 None if escalated is None else int(bool(escalated)),
                 json.dumps(extra) if extra else None),
            )
            return cur.lastrowid
    except Exception as exc:  # noqa: BLE001 - history must never fail a request
        logger.warning("Could not record run: %s", exc)
        return None


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["failed_pipes"] = json.loads(d["failed_pipes"] or "[]")
    d["extra"] = json.loads(d["extra"]) if d.get("extra") else None
    d["escalated"] = None if d["escalated"] is None else bool(d["escalated"])
    return d


def list_runs(limit: int = 50, network: str = None, kind: str = None) -> list:
    init()
    q, args = "SELECT * FROM runs", []
    where = []
    if network:
        where.append("network = ?"); args.append(network)
    if kind:
        where.append("kind = ?"); args.append(kind)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(max(1, min(int(limit), 500)))
    with _conn() as conn:
        return [_row(r) for r in conn.execute(q, args).fetchall()]


def get_run(run_id: int):
    init()
    with _conn() as conn:
        r = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row(r) if r else None
