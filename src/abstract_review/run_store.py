"""SQLite snapshots and leases for pausing and resuming reviews."""

import json
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, report_version_id TEXT NOT NULL, report_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS snapshots (run_id TEXT NOT NULL, version_id TEXT NOT NULL, report_json TEXT NOT NULL, PRIMARY KEY(run_id, version_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS run_leases (run_id TEXT PRIMARY KEY, token TEXT NOT NULL, expires REAL NOT NULL)")
            yield connection

    def read(self, run_id: str, version_id: str | None = None) -> dict:
        if not self.path.is_file():
            raise ValueError(f"unknown review run {run_id!r}")
        with closing(sqlite3.connect(self.path)) as connection:
            names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            row = None
            if version_id is not None and "snapshots" in names:
                row = connection.execute(
                    "SELECT report_json FROM snapshots WHERE run_id=? AND version_id=?", (run_id, version_id)
                ).fetchone()
            if row is None and "runs" in names:
                active = connection.execute(
                    "SELECT report_version_id,report_json FROM runs WHERE run_id=?", (run_id,)
                ).fetchone()
                if active is not None:
                    if version_id is not None and active[0] != version_id:
                        raise ValueError(f"report version {version_id!r} does not belong to run {run_id!r}")
                    row = active[1:]
        if row is None:
            raise ValueError(f"unknown review run {run_id!r}")
        return json.loads(row[0])

    def save(self, report: dict, token: str | None = None, *, new: bool = False) -> None:
        run_id, version = report["run_id"], report["report_version_id"]
        content = json.dumps(report, ensure_ascii=False)
        with self._write() as db:
            if token is not None:
                row = db.execute("SELECT token FROM run_leases WHERE run_id=? AND expires>?", (run_id, time.time())).fetchone()
                if row is None or row[0] != token:
                    raise ValueError("the review lease expired or changed")
                db.execute("UPDATE run_leases SET expires=? WHERE run_id=?", (time.time() + 150, run_id))
            existing = db.execute("SELECT report_version_id,report_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is not None and existing[0] != version:
                db.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)", (run_id, *existing))
            if new:
                db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, version, content))
            else:
                db.execute("UPDATE runs SET report_version_id=?,report_json=? WHERE run_id=?", (version, content, run_id))
            db.execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)", (run_id, version, content))

    @contextmanager
    def acquire(self, run_id: str):
        token = uuid4().hex
        with self._write() as db:
            now = time.time()
            owner = db.execute("SELECT expires FROM run_leases WHERE run_id=?", (run_id,)).fetchone()
            if owner is not None and owner[0] > now:
                raise ValueError("this review is already being processed")
            db.execute("INSERT OR REPLACE INTO run_leases VALUES (?, ?, ?)", (run_id, token, now + 150))
        try:
            yield token
        finally:
            with self._write() as db:
                db.execute("DELETE FROM run_leases WHERE run_id=? AND token=?", (run_id, token))
