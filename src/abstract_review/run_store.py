"""SQLite snapshots and durable jobs for local review execution."""

import json
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


class RunBusy(ValueError):
    """Another execution is already scheduled or owns this review."""


class LeaseLost(ValueError):
    """An expired or superseded worker can no longer commit."""


@dataclass(frozen=True, slots=True)
class QueuedJob:
    run_id: str
    kind: str
    payload: dict
    token: str


class RunStore:
    def __init__(self, path: Path, *, lease_seconds: float = 150) -> None:
        self.path = path
        self.lease_seconds = lease_seconds

    @contextmanager
    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, report_version_id TEXT NOT NULL, report_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS snapshots (run_id TEXT NOT NULL, version_id TEXT NOT NULL, report_json TEXT NOT NULL, PRIMARY KEY(run_id, version_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS run_leases (run_id TEXT PRIMARY KEY, token TEXT NOT NULL, expires REAL NOT NULL)")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS review_jobs (
                    run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    state TEXT NOT NULL, queued_at REAL NOT NULL, error TEXT
                )"""
            )
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

    @staticmethod
    def _job(db, run_id: str) -> tuple | None:
        return db.execute("SELECT kind,payload,state,error FROM review_jobs WHERE run_id=?", (run_id,)).fetchone()

    @staticmethod
    def _queue(db, run_id: str, kind: str, payload: dict) -> None:
        db.execute(
            "INSERT OR REPLACE INTO review_jobs VALUES (?, ?, ?, 'queued', ?, NULL)",
            (run_id, kind, json.dumps(payload, ensure_ascii=False), time.time()),
        )

    @staticmethod
    def _run(db, run_id: str) -> tuple | None:
        return db.execute("SELECT report_version_id,report_json FROM runs WHERE run_id=?", (run_id,)).fetchone()

    @staticmethod
    def _ref(report: dict) -> dict:
        return {"run_id": report["run_id"], "report_version_id": report["report_version_id"],
                "run_status": report.get("run_status", "report_ready")}

    def status(self, run_id: str) -> dict:
        if not self.path.is_file():
            raise ValueError(f"unknown review run {run_id!r}")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            job = self._job(db, run_id) if "review_jobs" in tables else None
            snapshot = self._run(db, run_id) if "runs" in tables else None
        if snapshot is not None:
            state = self._ref(json.loads(snapshot[1]))
            if job is not None and job[2] in {"queued", "running"}:
                state["run_status"] = job[2]
            return state
        if job is None:
            raise ValueError(f"unknown review run {run_id!r}")
        state = {"run_id": run_id, "report_version_id": None, "run_status": job[2]}
        if job[3] is not None:
            state["runtime_error"] = job[3]
        return state

    def _renew(self, db, run_id: str, token: str, now: float) -> bool:
        updated = db.execute(
            "UPDATE run_leases SET expires=? WHERE run_id=? AND token=? AND expires>?",
            (now + self.lease_seconds, run_id, token, now),
        )
        return updated.rowcount == 1

    def renew(self, run_id: str, token: str) -> bool:
        with self._write() as db:
            return self._renew(db, run_id, token, time.time())

    def _save(self, db, report: dict, *, new: bool = False) -> None:
        run_id, version = report["run_id"], report["report_version_id"]
        content = json.dumps(report, ensure_ascii=False)
        existing = self._run(db, run_id)
        if existing is not None and existing[0] != version:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)", (run_id, *existing))
        if new:
            db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, version, content))
        elif existing is None:
            raise ValueError(f"unknown review run {run_id!r}")
        else:
            db.execute("UPDATE runs SET report_version_id=?,report_json=? WHERE run_id=?", (version, content, run_id))
        db.execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)", (run_id, version, content))

    def save(
        self, report: dict, token: str | None = None, *, new: bool = False,
        enqueue: tuple[str, dict] | None = None,
    ) -> None:
        run_id = report["run_id"]
        with self._write() as db:
            if token is not None and not self._renew(db, run_id, token, time.time()):
                raise LeaseLost("the review lease expired or changed")
            if enqueue is not None and token is None:
                raise ValueError("queueing a new version requires a review lease")
            self._save(db, report, new=new)
            if enqueue is not None:
                self._queue(db, run_id, *enqueue)

    def queue_start(self, run_id: str, payload: dict) -> dict:
        with self._write() as db:
            if self._run(db, run_id) is not None or self._job(db, run_id) is not None:
                raise ValueError("review already exists")
            self._queue(db, run_id, "start", payload)
        return {"run_id": run_id, "report_version_id": None, "run_status": "queued"}

    def queue_resume(self, run_id: str, request_id: str | None, add_requests: int, add_seconds: int) -> dict:
        if add_requests < 0 or add_seconds < 0 or ((add_requests or add_seconds) and not request_id):
            raise ValueError("positive budget extensions require a request ID")
        payload = {"request_id": request_id, "add_requests": add_requests, "add_seconds": add_seconds}
        with self._write() as db:
            row = self._run(db, run_id)
            job = self._job(db, run_id)
            if job is not None and job[2] in {"queued", "running"}:
                if not (add_requests or add_seconds) or (job[0] == "resume" and json.loads(job[1]) == payload):
                    return {"run_id": run_id, "report_version_id": row[0] if row else None, "run_status": job[2]}
                raise RunBusy("this review is already queued or running")
            if row is None:
                raise ValueError("this review has no resumable report")
            report = json.loads(row[1])
            if report.get("configuration") is None:
                raise ValueError("this review was started without an evaluation engine")
            if report.get("run_status") == "failed":
                raise ValueError("this review has an unrecoverable failure")
            if (add_requests or add_seconds) and request_id in report.get("budget_extensions", {}):
                if report["budget_extensions"][request_id] != [add_requests, add_seconds]:
                    raise ValueError("this request ID was already used for a different budget extension")
                return self._ref(report)
            if report.get("run_status") == "report_ready":
                return self._ref(report)
            owner = db.execute("SELECT 1 FROM run_leases WHERE run_id=? AND expires>?", (run_id, time.time())).fetchone()
            if owner is not None:
                raise RunBusy("this review is already being processed")
            report["run_status"] = "queued"
            self._save(db, report)
            self._queue(db, run_id, "resume", payload)
            return self._ref(report)

    def claim_next(self) -> QueuedJob | None:
        with self._write() as db:
            now = time.time()
            row = db.execute(
                """SELECT job.run_id,job.kind,job.payload
                   FROM review_jobs AS job
                   LEFT JOIN run_leases AS lease ON lease.run_id=job.run_id
                   WHERE job.state IN ('queued','running') AND (lease.expires IS NULL OR lease.expires<=?)
                   ORDER BY job.queued_at, job.run_id LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            run_id, kind, payload = row
            token = uuid4().hex
            db.execute("INSERT OR REPLACE INTO run_leases VALUES (?, ?, ?)", (run_id, token, now + self.lease_seconds))
            db.execute("UPDATE review_jobs SET state='running' WHERE run_id=?", (run_id,))
            return QueuedJob(run_id, kind, json.loads(payload), token)

    def finish(self, job: QueuedJob, *, error: str | None = None) -> bool:
        with self._write() as db:
            if not self._renew(db, job.run_id, job.token, time.time()):
                return False
            if error is not None:
                row = self._run(db, job.run_id)
                if row is not None:
                    report = json.loads(row[1])
                    report.update(run_status="failed", runtime_error=error)
                    self._save(db, report)
                db.execute(
                    "UPDATE review_jobs SET state='failed',payload='{}',error=? WHERE run_id=?", (error, job.run_id),
                )
            else:
                db.execute("DELETE FROM review_jobs WHERE run_id=?", (job.run_id,))
            db.execute("DELETE FROM run_leases WHERE run_id=? AND token=?", (job.run_id, job.token))
            return True

    @contextmanager
    def acquire(self, run_id: str):
        token = uuid4().hex
        with self._write() as db:
            now = time.time()
            job = self._job(db, run_id)
            if job is not None and job[2] in {"queued", "running"}:
                raise RunBusy("this review is already queued or running")
            owner = db.execute("SELECT expires FROM run_leases WHERE run_id=?", (run_id,)).fetchone()
            if owner is not None and owner[0] > now:
                raise RunBusy("this review is already being processed")
            db.execute("INSERT OR REPLACE INTO run_leases VALUES (?, ?, ?)", (run_id, token, now + self.lease_seconds))
        try:
            yield token
        finally:
            with self._write() as db:
                db.execute("DELETE FROM run_leases WHERE run_id=? AND token=?", (run_id, token))
