from __future__ import annotations

import threading
import time
from collections import OrderedDict

from ..core.errors import StoreFullError
from .models import Job


class JobStore:
    def __init__(
        self,
        max_size: int,
        ttl_seconds: float | None = None,
        result_ttl_seconds: float | None = None,
    ) -> None:
        self._store: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = (
            ttl_seconds if ttl_seconds is not None else result_ttl_seconds
        )
        if self._ttl is None:
            raise ValueError("A result TTL must be configured")

    def _cleanup(self) -> None:
        now = time.time()
        for job_id, job in list(self._store.items()):
            finished = (
                job.completed_at
                if job.completed_at is not None
                else job.finished_at
            )
            if (
                job.status in {"completed", "failed"}
                and finished is not None
                and now - finished >= self._ttl
            ):
                self._store.pop(job_id, None)

    def put(self, job: Job) -> None:
        with self._lock:
            self._cleanup()
            while len(self._store) >= self._max_size:
                removable = next(
                    (
                        key
                        for key, candidate in self._store.items()
                        if candidate.status in {"completed", "failed"}
                    ),
                    None,
                )
                if removable is None:
                    raise StoreFullError("Job result store is full")
                self._store.pop(removable)
            self._store[job.id] = job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._store.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._cleanup()
            return self._store.get(job_id)

    def mark_processing(self, job_id: str, worker_id: str) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job and job.status == "queued":
                job.status = "processing"
                job.started_at = time.time()
                job.worker_id = worker_id

    def mark_completed(
        self,
        job_id: str,
        result: str,
        inference_seconds: float,
        metrics: dict | None = None,
        runtime: dict | None = None,
    ) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if not job:
                return
            now = time.time()
            job.status = "completed"
            job.result = result
            job.inference_seconds = inference_seconds
            job.metrics = metrics or {}
            job.runtime = runtime or {}
            job.image = None
            job.completed_at = now
            job.finished_at = now
            job.done.set()

    def mark_failed(
        self,
        job_id: str,
        public_error: str,
        internal_error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if not job:
                return
            now = time.time()
            job.status = "failed"
            job.public_error = public_error
            job.internal_error = internal_error
            job.image = None
            job.completed_at = now
            job.finished_at = now
            job.done.set()

    def fail_active_for_worker(self, worker_id: str, reason: str) -> int:
        count = 0
        with self._lock:
            active_ids = [
                job.id
                for job in self._store.values()
                if job.status == "processing" and job.worker_id == worker_id
            ]
            for job_id in active_ids:
                self.mark_failed(
                    job_id,
                    "TPU worker stopped during inference",
                    reason,
                )
                count += 1
        return count

    def fail_pending(
        self,
        public_error: str,
        internal_error: str | None = None,
    ) -> int:
        """Fail every queued/processing job when no worker recovery remains."""
        with self._lock:
            pending_ids = [
                job.id
                for job in self._store.values()
                if job.status in {"queued", "processing"}
            ]
            for job_id in pending_ids:
                self.mark_failed(job_id, public_error, internal_error)
            return len(pending_ids)

    def pending_count(self) -> int:
        with self._lock:
            return sum(
                job.status in {"queued", "processing"}
                for job in self._store.values()
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._cleanup()
            counts = {
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
            }
            for job in self._store.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            counts["total"] = len(self._store)
            return counts
