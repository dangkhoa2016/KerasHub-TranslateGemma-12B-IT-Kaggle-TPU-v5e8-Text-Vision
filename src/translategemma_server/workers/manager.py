from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
import uuid

from ..core.errors import (
    QueueFullError,
    ServiceUnavailableError,
    WorkerNotReadyError,
)
from ..jobs.models import Job
from ..jobs.store import JobStore
from .worker import model_worker_main

logger = logging.getLogger("translategemma_server")


class TranslationManager:
    """Coordinate HTTP jobs with one spawned TPU model process."""

    WORKER_ID = "tpu-0"

    def __init__(self, config) -> None:
        self.config = config
        self.store = JobStore(
            config.max_store_size,
            config.result_ttl_seconds,
        )
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=config.max_queue_size)
        self.result_queue = self.ctx.Queue()
        self.shutdown_event = self.ctx.Event()

        self._worker = None
        self._generation = 0
        self._worker_status: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._accepting = True
        self._shutting_down = threading.Event()

        self._collector_stop = threading.Event()
        self._collector_thread = None
        self._monitor_thread = None
        self._startup_thread = None
        self._load_watchdog_thread = None
        self._restart_pending = False

    def start_async(self) -> None:
        if self._startup_thread is not None:
            return

        self._collector_thread = threading.Thread(
            target=self._collect,
            name="tpu-results",
            daemon=True,
        )
        self._collector_thread.start()

        self._startup_thread = threading.Thread(
            target=self._start_worker,
            name="tpu-start",
            daemon=True,
        )
        self._startup_thread.start()

    def _start_worker(self) -> None:
        self._generation += 1
        self._restart_pending = False
        generation = self._generation
        worker_id = self.WORKER_ID

        with self._lock:
            self._worker_status[worker_id] = {
                "worker_id": worker_id,
                "state": "starting",
                "generation": generation,
                "started_at": time.time(),
                "load_timeout_seconds": self.config.worker_load_timeout,
            }

        process = self.ctx.Process(
            target=model_worker_main,
            args=(
                worker_id,
                generation,
                self.config.worker_payload(),
                self.task_queue,
                self.result_queue,
                self.shutdown_event,
            ),
            name=f"translategemma-{worker_id}",
        )
        process.start()
        self._worker = process

        self._monitor_thread = threading.Thread(
            target=self._monitor,
            args=(process, generation),
            name="tpu-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

        self._load_watchdog_thread = threading.Thread(
            target=self._watch_worker_load,
            args=(process, generation),
            name="tpu-load-watchdog",
            daemon=True,
        )
        self._load_watchdog_thread.start()

    def _can_restart_generation(self, generation: int) -> bool:
        return (generation - 1) < self.config.max_worker_restarts

    def _expire_worker_load(self, process, generation: int) -> bool:
        """Atomically fail a still-loading generation and terminate its process."""
        with self._lock:
            status = self._worker_status.get(self.WORKER_ID, {})
            if status.get("generation") != generation:
                return False
            if status.get("state") not in {"starting", "loading"}:
                return False
            status.update(
                state="failed",
                load_timed_out=True,
                error=(
                    "TPU worker load timeout after "
                    f"{self.config.worker_load_timeout:g}s"
                ),
            )
            self._restart_pending = self._can_restart_generation(generation)

        if process.is_alive():
            process.terminate()
        return True

    def _watch_worker_load(self, process, generation: int) -> None:
        deadline = time.monotonic() + self.config.worker_load_timeout
        while not self._shutting_down.is_set() and time.monotonic() < deadline:
            with self._lock:
                status = self._worker_status.get(self.WORKER_ID, {})
                if status.get("generation") != generation:
                    return
                if status.get("state") not in {"starting", "loading"}:
                    return
            time.sleep(0.25)

        if not self._shutting_down.is_set():
            if self._expire_worker_load(process, generation):
                logger.error(
                    "TPU worker generation %s exceeded load timeout %.1fs",
                    generation,
                    self.config.worker_load_timeout,
                )

    def _collect(self) -> None:
        while not self._collector_stop.is_set():
            try:
                message = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle(message)

    def _handle(self, message: dict) -> None:
        worker_id = message.get("worker_id", self.WORKER_ID)
        message_type = message.get("type")

        with self._lock:
            status = self._worker_status.setdefault(
                worker_id,
                {"worker_id": worker_id},
            )
            if message.get("generation") != status.get("generation"):
                return

            status["pid"] = message.get("pid")

            if message_type == "worker_state":
                status["state"] = message["state"]
            elif message_type == "worker_ready":
                status.update(
                    state="ready",
                    metadata=message.get("metadata", {}),
                )
            elif message_type == "worker_load_error":
                status.update(
                    state="failed",
                    error=message.get("error"),
                )
                self._restart_pending = self._can_restart_generation(
                    int(status.get("generation", 0))
                )
            elif message_type == "job_started":
                status["state"] = "busy"
                status["active_job_id"] = message["job_id"]
                self.store.mark_processing(message["job_id"], worker_id)
            elif message_type == "job_completed":
                status["state"] = "ready"
                status.pop("active_job_id", None)
                self.store.mark_completed(
                    message["job_id"],
                    message["result"],
                    message["inference_seconds"],
                    message.get("metrics"),
                    message.get("runtime"),
                )
            elif message_type == "job_failed":
                status["state"] = "ready"
                status.pop("active_job_id", None)
                self.store.mark_failed(
                    message["job_id"],
                    "Translation failed",
                    message.get("error"),
                )
            elif message_type == "worker_stopped":
                status["state"] = "stopped"

    def _monitor(self, process, generation: int) -> None:
        process.join()
        if self._shutting_down.is_set():
            return

        with self._lock:
            status = self._worker_status.get(self.WORKER_ID, {})
            current_generation = status.get("generation") == generation
            if not current_generation:
                return
            status["state"] = "failed"
            status["exit_code"] = process.exitcode
            status.setdefault(
                "error",
                f"TPU worker exited code {process.exitcode}",
            )
            should_restart = self._can_restart_generation(generation)
            self._restart_pending = should_restart

        self.store.fail_active_for_worker(
            self.WORKER_ID,
            f"worker exited code {process.exitcode}",
        )

        if should_restart:
            logger.warning(
                "TPU worker exited (code=%s); restarting (%s/%s)",
                process.exitcode,
                generation,
                self.config.max_worker_restarts,
            )
            time.sleep(1)
            if not self._shutting_down.is_set():
                self._start_worker()
        else:
            with self._lock:
                self._restart_pending = False
            self.store.fail_pending(
                "TPU worker unavailable",
                f"worker exited code {process.exitcode}; restart budget exhausted",
            )

    def has_ready_worker(self) -> bool:
        with self._lock:
            return any(
                status.get("state") in {"ready", "busy"}
                and (status.get("metadata") or {}).get("device_count")
                == self.config.expected_tpu_devices
                for status in self._worker_status.values()
            )

    def submit(self, payload: dict, request_id: str | None = None) -> Job:
        if not self._accepting or self._shutting_down.is_set():
            raise ServiceUnavailableError("Server is shutting down")
        if not self.has_ready_worker():
            raise WorkerNotReadyError(
                "TPU worker is not ready yet",
                health=self.health(),
            )

        job = Job(
            id=f"job-{uuid.uuid4().hex[:16]}",
            text=payload.get("text", ""),
            src=payload["src"],
            tgt=payload["tgt"],
            max_tokens=payload["max_tokens"],
            request_id=request_id,
            image=payload.get("image"),
            src_code=payload.get("src_code"),
            tgt_code=payload.get("tgt_code"),
        )
        self.store.put(job)

        task = {
            "job_id": job.id,
            "text": job.text,
            "src": job.src,
            "tgt": job.tgt,
            "max_tokens": job.max_tokens,
            "image": job.image,
            "src_code": job.src_code,
            "tgt_code": job.tgt_code,
        }
        try:
            self.task_queue.put_nowait(task)
        except queue.Full as exc:
            self.store.delete(job.id)
            raise QueueFullError("Translation queue is full") from exc

        # The multiprocessing queue owns its serialized copy. Do not retain
        # decoded images in the coordinator-side job store.
        job.image = None
        return job

    def health(self) -> dict:
        with self._lock:
            workers = [dict(value) for value in self._worker_status.values()]

        ready_workers = sum(
            worker.get("state") in {"ready", "busy"}
            and (worker.get("metadata") or {}).get("device_count")
            == self.config.expected_tpu_devices
            for worker in workers
        )

        if ready_workers:
            state = "ready"
        elif self._restart_pending:
            state = "restarting"
        elif not workers or any(
            worker.get("state") in {"starting", "loading"}
            for worker in workers
        ):
            state = "loading"
        else:
            state = "unavailable"

        return {
            "state": state,
            "ready": ready_workers == 1,
            "ready_workers": ready_workers,
            "expected_workers": 1,
            "restart_pending": self._restart_pending,
            "worker_generation": self._generation,
            "worker_load_timeout_seconds": self.config.worker_load_timeout,
            "accepting_jobs": (
                self._accepting and not self._shutting_down.is_set()
            ),
            "jobs": self.store.stats(),
            "workers": workers,
            "runtime": {
                "model": "translategemma_12b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": self.config.expected_tpu_devices,
                "mesh": list(self.config.mesh_shape),
            },
        }

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.pending_count() == 0:
                return True
            time.sleep(0.2)
        return self.store.pending_count() == 0

    def shutdown(self, wait_for_jobs: bool, timeout: float) -> bool:
        if self._shutting_down.is_set():
            return self.store.pending_count() == 0

        self._shutting_down.set()
        self._accepting = False
        idle = self.wait_idle(timeout) if wait_for_jobs else True
        self.shutdown_event.set()

        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass

        process = self._worker
        if process and process.is_alive():
            process.join(timeout=min(timeout, 30))
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)

        self._collector_stop.set()
        if self._collector_thread:
            self._collector_thread.join(timeout=2)
        return idle
