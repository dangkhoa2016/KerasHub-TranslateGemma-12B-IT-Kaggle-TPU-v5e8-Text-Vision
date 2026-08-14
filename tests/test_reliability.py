import importlib.util
import tempfile
import unittest
import queue
import types
from unittest import mock
from dataclasses import replace
from pathlib import Path

from translategemma_server.core.config import Config
from translategemma_server.workers.manager import TranslationManager
from translategemma_server.jobs.models import Job
from translategemma_server.jobs.store import JobStore
from translategemma_server.workers.worker import (
    fatal_worker_load_exit_code,
    validate_tpu_fallback,
    model_worker_main,
)


def make_config(**overrides):
    return replace(Config.for_tests(), **overrides)


class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive
        self.terminated = False
        self.exitcode = None

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15


class WorkerReliabilityTests(unittest.TestCase):
    def test_kaggle_v5litepod_8_alias_is_accepted(self):
        validate_tpu_fallback(
            True,
            {
                "TG_TPU_FALLBACK_APPLIED": "true",
                "TPU_ACCELERATOR_TYPE": "v5litepod-8",
            },
        )

    def test_model_load_failure_uses_nonzero_process_exit(self):
        self.assertNotEqual(fatal_worker_load_exit_code(), 0)

    def test_worker_entrypoint_exits_nonzero_after_preflight_failure(self):
        result_queue = queue.Queue()
        fake_jax = types.SimpleNamespace()
        fake_keras = types.SimpleNamespace()
        config = {
            "jax_compilation_cache_dir": None,
            "require_v5e8": True,
        }
        env = {
            "TG_TPU_FALLBACK_APPLIED": "true",
            "TPU_ACCELERATOR_TYPE": "v4-8",
        }
        with mock.patch.dict("sys.modules", {"jax": fake_jax, "keras": fake_keras}):
            with mock.patch.dict("os.environ", env, clear=False):
                # This test intentionally forces preflight failure. Suppress only
                # the expected logger.exception traceback so setup output stays clear.
                with mock.patch("translategemma_server.workers.worker.logger.exception"):
                    with self.assertRaises(SystemExit) as raised:
                        model_worker_main(
                            "tpu-0",
                            1,
                            config,
                            queue.Queue(),
                            result_queue,
                            types.SimpleNamespace(is_set=lambda: False),
                        )
        self.assertNotEqual(raised.exception.code, 0)
        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())
        self.assertTrue(any(m.get("type") == "worker_load_error" for m in messages))


class ManagerReliabilityTests(unittest.TestCase):
    def test_load_timeout_marks_generation_failed_and_terminates_process(self):
        manager = TranslationManager(make_config(worker_load_timeout=1))
        process = FakeProcess(alive=True)
        manager._worker_status[manager.WORKER_ID] = {
            "worker_id": manager.WORKER_ID,
            "generation": 1,
            "state": "loading",
        }

        applied = manager._expire_worker_load(process, generation=1)

        self.assertTrue(applied)
        self.assertTrue(process.terminated)
        status = manager._worker_status[manager.WORKER_ID]
        self.assertEqual(status["state"], "failed")
        self.assertTrue(status["load_timed_out"])
        self.assertIn("timeout", status["error"].lower())

    def test_load_timeout_does_not_kill_ready_worker(self):
        manager = TranslationManager(make_config(worker_load_timeout=1))
        process = FakeProcess(alive=True)
        manager._worker_status[manager.WORKER_ID] = {
            "worker_id": manager.WORKER_ID,
            "generation": 1,
            "state": "ready",
        }
        self.assertFalse(manager._expire_worker_load(process, generation=1))
        self.assertFalse(process.terminated)

    def test_health_reports_restarting_instead_of_unavailable_during_retry(self):
        manager = TranslationManager(make_config())
        manager._worker_status[manager.WORKER_ID] = {
            "worker_id": manager.WORKER_ID,
            "generation": 1,
            "state": "failed",
            "error": "boom",
        }
        manager._restart_pending = True
        health = manager.health()
        self.assertEqual(health["state"], "restarting")
        self.assertTrue(health["restart_pending"])


class StoreReliabilityTests(unittest.TestCase):
    def test_fail_pending_marks_queued_and_processing_jobs_done(self):
        store = JobStore(max_size=4, ttl_seconds=60)
        queued = Job(id="q", text="q", src="English", tgt="Vietnamese", max_tokens=4)
        active = Job(id="a", text="a", src="English", tgt="Vietnamese", max_tokens=4)
        store.put(queued)
        store.put(active)
        store.mark_processing("a", "tpu-0")

        count = store.fail_pending("TPU worker unavailable", "restart budget exhausted")

        self.assertEqual(count, 2)
        self.assertEqual(store.get("q").status, "failed")
        self.assertEqual(store.get("a").status, "failed")
        self.assertTrue(store.get("q").done.is_set())
        self.assertTrue(store.get("a").done.is_set())


class ReadinessScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts/wait_ready.py"
        spec = importlib.util.spec_from_file_location("wait_ready", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_loading_503_continues(self):
        decision = self.module.readiness_decision(
            503,
            {"state": "loading", "ready": False},
        )
        self.assertEqual(decision, "continue")

    def test_restarting_503_continues(self):
        decision = self.module.readiness_decision(
            503,
            {"state": "restarting", "ready": False},
        )
        self.assertEqual(decision, "continue")

    def test_unavailable_503_fails_fast(self):
        decision = self.module.readiness_decision(
            503,
            {"state": "unavailable", "ready": False},
        )
        self.assertEqual(decision, "failed")

    def test_ready_requires_exact_eight_device_mesh(self):
        decision = self.module.readiness_decision(
            200,
            {
                "state": "ready",
                "ready": True,
                "workers": [{
                    "state": "ready",
                    "metadata": {"device_count": 8, "mesh_shape": [1, 8]},
                }],
            },
        )
        self.assertEqual(decision, "ready")


class ScriptContractTests(unittest.TestCase):
    def test_start_script_verifies_liveness_before_success(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/start.sh").read_text()
        self.assertIn("/health/live", text)
        self.assertIn("curl", text)
        self.assertIn("START_LIVENESS_TIMEOUT", text)

    def test_notebook_uses_shared_wait_ready_script(self):
        import json
        root = Path(__file__).resolve().parents[1]
        path = root / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"
        nb = json.loads(path.read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        self.assertIn("scripts/wait_ready.py", code)
        self.assertNotIn("except (urllib.error.HTTPError, urllib.error.URLError):\n            pass", code)


if __name__ == "__main__":
    unittest.main()
