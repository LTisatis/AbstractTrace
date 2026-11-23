"""Consume durable local review jobs without blocking web requests."""

import logging
import shutil
import sqlite3
from pathlib import Path
from threading import Event, Thread

from .review import PdfReviewInput, ReviewService
from .run_store import LeaseLost, QueuedJob


_LOG = logging.getLogger(__name__)


class LocalWorker:
    def __init__(self, service: ReviewService, data_dir: Path, wake: Event) -> None:
        self.service = service
        self.store = service.store
        self.intake = data_dir / "intake"
        self.wake = wake
        self.stopping = Event()
        self.thread = Thread(target=self._run, name="abstract-review-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                job = self.store.claim_next()
                if job is not None:
                    self._process(job)
                    continue
            except sqlite3.Error:
                _LOG.warning("Local review queue temporarily unavailable", exc_info=True)
            self.wake.wait(0.25)
            self.wake.clear()

    def _heartbeat(self, job: QueuedJob, finished: Event) -> None:
        interval = min(10.0, self.store.lease_seconds / 3)
        while not finished.wait(interval) and not self.stopping.is_set():
            try:
                if not self.store.renew(job.run_id, job.token):
                    return
            except sqlite3.Error:
                _LOG.warning("Could not renew local review lease", exc_info=True)

    def _process(self, job: QueuedJob) -> None:
        finished = Event()
        heartbeat = Thread(target=self._heartbeat, args=(job, finished), daemon=True)
        heartbeat.start()
        error = None
        try:
            if job.kind == "start":
                self._start(job)
                self.service.resume(job.run_id, lease_token=job.token)
            elif job.kind == "resume":
                self.service.resume(job.run_id, lease_token=job.token, **job.payload)
            else:
                raise ValueError("unknown review queue operation")
        except LeaseLost:
            # A later worker controls the job and its intake files.
            return
        except sqlite3.Error:
            # Leave the job recoverable. It becomes claimable after expiry.
            raise
        except (OSError, ValueError) as exc:
            error = str(exc)
        except Exception:
            _LOG.exception("Review worker failed for run %s", job.run_id)
            error = "处理任务时发生内部错误。"
        finally:
            finished.set()
            heartbeat.join(timeout=1)
        if self.store.finish(job, error=error) and job.kind == "start":
            try:
                shutil.rmtree(self._intake(job.run_id))
            except FileNotFoundError:
                pass
            except (OSError, ValueError):
                _LOG.warning("Could not clear staged files for run %s", job.run_id, exc_info=True)

    def _intake(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
            raise ValueError("invalid intake identifier")
        path = self.intake / run_id
        if not path.resolve().is_relative_to(self.intake.resolve()):
            raise ValueError("intake directory escapes local review storage")
        return path

    def _start(self, job: QueuedJob) -> None:
        try:
            self.store.read(job.run_id)
        except ValueError:
            pass
        else:
            return
        root = self._intake(job.run_id).resolve()

        def upload(relative: str) -> Path:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("staged review input is unavailable")
            return path

        parameters = dict(job.payload)
        parameters["pdf"] = upload(parameters["pdf"])
        parameters["supplements"] = tuple(upload(path) for path in parameters["supplements"])
        request = PdfReviewInput(**parameters)
        self.service.start(request, run_id=job.run_id, lease_token=job.token, execute=False)
