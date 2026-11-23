from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from abstract_review.run_store import LeaseLost, RunBusy, RunStore


def report(run_id, version, status):
    return {"run_id": run_id, "report_version_id": version, "run_status": status,
            "configuration": {"max_requests": 80, "max_seconds": 900}}


def test_queue_claim_is_atomic_and_expired_owner_is_fenced(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("abstract_review.run_store.time.time", lambda: clock[0])
    database = tmp_path / "state.sqlite3"
    store = RunStore(database, lease_seconds=15)
    store.queue_start("run", {"pdf": "main/paper.pdf"})
    barrier = Barrier(2)

    def claim():
        barrier.wait()
        return RunStore(database, lease_seconds=15).claim_next()

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lambda _: claim(), range(2)))
    old, = [job for job in jobs if job is not None]
    assert store.status("run")["run_status"] == "running"
    clock[0] += 14
    assert store.renew(old.run_id, old.token)
    clock[0] += 14
    assert store.claim_next() is None
    clock[0] += 2
    new = store.claim_next()
    assert new.run_id == old.run_id and new.token != old.token and new.payload == old.payload
    assert not store.renew(old.run_id, old.token)
    assert not store.finish(old, error="stale failure")
    with pytest.raises(LeaseLost):
        store.save(report("run", "old", "report_ready"), old.token, new=True)
    store.save(report("run", "new", "report_ready"), new.token, new=True)
    assert store.finish(new)
    assert store.status("run")["report_version_id"] == "new"
    assert store.claim_next() is None


def test_pending_selection_and_resume_job_commit_together(tmp_path):
    store = RunStore(tmp_path / "state.sqlite3")
    store.save(report("run", "original", "report_ready"), new=True)
    with store.acquire("run") as token:
        store.save(report("run", "selected", "queued"), token, enqueue=("resume", {}))
        assert store.read("run", "selected")["run_status"] == "queued"
        assert store.read("run", "original")["run_status"] == "report_ready"
        assert store.claim_next() is None

    new = store.claim_next()
    assert new.run_id == "run" and new.kind == "resume" and new.payload == {}
    with pytest.raises(RunBusy):
        with store.acquire("run"):
            pytest.fail("the active selection job must own the run")
    store.save(report("run", "selected", "report_ready"), new.token)
    assert store.status("run")["run_status"] == "running"
    assert store.finish(new)
    assert store.status("run")["run_status"] == "report_ready"


def test_resume_payload_is_idempotent_and_failed_intake_keeps_diagnostics(tmp_path):
    store = RunStore(tmp_path / "state.sqlite3")
    store.save(report("paused", "v1", "paused"), new=True)
    first = store.queue_resume("paused", "extend", 3, 15)
    assert first == store.queue_resume("paused", "extend", 3, 15)
    with pytest.raises(RunBusy):
        store.queue_resume("paused", "extend", 4, 15)
    job = store.claim_next()
    assert job.payload == {"request_id": "extend", "add_requests": 3, "add_seconds": 15}
    assert store.finish(job, error="unexpected error")
    assert store.read("paused", "v1")["runtime_error"] == "unexpected error"

    store.queue_start("invalid", {"pdf": "main/bad.pdf"})
    bad = store.claim_next()
    assert store.finish(bad, error="invalid PDF")
    assert store.status("invalid") == {
        "run_id": "invalid", "report_version_id": None, "run_status": "failed", "runtime_error": "invalid PDF",
    }
