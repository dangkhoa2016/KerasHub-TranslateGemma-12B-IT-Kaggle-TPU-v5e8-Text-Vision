import base64
import io
import os
import queue
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from PIL import Image
import importlib.util
HAS_FLASK = importlib.util.find_spec("flask") is not None

from translategemma_server.core.config import Config, discover_model_path, weights_path
from translategemma_server.core.errors import QueueFullError, WorkerNotReadyError
from translategemma_server.core.validation import parse_image_translation_payload, parse_translation_payload
from translategemma_server.jobs.models import Job
from translategemma_server.jobs.store import JobStore
from translategemma_server.tpu.generation import language_code, plan_generation, translation_prompt
from translategemma_server.workers.manager import TranslationManager
from translategemma_server.workers.worker import sanitize_public_metrics, validate_tpu_fallback
if HAS_FLASK:
    from translategemma_server.api.app import Runtime, create_app, parse_restart_options
else:
    Runtime = create_app = None
    def parse_restart_options(data, config):
        wait = data.get("wait_for_jobs", True)
        if not isinstance(wait, bool):
            raise ValueError("wait_for_jobs must be boolean")
        raw = data.get("timeout", config.shutdown_timeout)
        try:
            timeout = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be numeric") from exc
        return wait, timeout


def make_config(**overrides):
    base = Config.for_tests()
    return replace(base, **overrides)


class ConfigTests(unittest.TestCase):
    def test_discovers_latest_complete_sharded_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for version in ("1", "2"):
                p = base / version
                (p / "assets/tokenizer").mkdir(parents=True)
                for name in ("config.json", "preprocessor.json", "assets/tokenizer/vocabulary.spm"):
                    (p / name).write_text("x")
                (p / "model.weights.json").write_text('{"weight_map": {"/layers/x": "model_00001.weights.h5"}}')
                (p / "model_00001.weights.h5").write_bytes(b"x")
            self.assertEqual(discover_model_path(base), base / "2")
            self.assertEqual(weights_path(base / "2").name, "model.weights.json")

    def test_defaults_are_single_tpu_worker_mesh_1x8(self):
        cfg = make_config()
        self.assertEqual(cfg.expected_tpu_devices, 8)
        self.assertEqual(cfg.mesh_shape, (1, 8))
        self.assertEqual(cfg.mesh_axis_names, ("batch", "model"))
        self.assertEqual(cfg.worker_count, 1)
        self.assertEqual(cfg.model_dtype, "bfloat16")
        self.assertTrue(cfg.generation_split_compile)


class ValidationTests(unittest.TestCase):
    def test_text_payload_accepts_language_names_and_codes(self):
        payload = parse_translation_payload({
            "text": "Hello",
            "source_lang": "English",
            "target_lang": "Vietnamese",
            "source_lang_code": "en",
            "target_lang_code": "vi",
            "max_new_tokens": 32,
        }, make_config())
        self.assertEqual(payload["src"], "English")
        self.assertEqual(payload["tgt_code"], "vi")

    def test_image_payload_decodes_base64_and_enforces_pixels(self):
        image = Image.new("RGB", (8, 8), "white")
        buf = io.BytesIO(); image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        payload = parse_image_translation_payload({
            "image_base64": encoded,
            "source_lang": "English",
            "target_lang": "Vietnamese",
        }, make_config())
        self.assertEqual(payload["image"].size, (8, 8))


class GenerationTests(unittest.TestCase):
    def test_generation_bucket_matches_v10_shape_policy(self):
        plan = plan_generation(110, 128, buckets=(256, 512, 1024), bucket_step=512, bucketing=True)
        self.assertEqual(plan.max_length, 256)

    def test_prompt_uses_structured_translategemma_format(self):
        prompt = translation_prompt("Hello", "English", "Vietnamese")
        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("English (en)", prompt)
        self.assertIn("Vietnamese (vi)", prompt)
        self.assertEqual(language_code("Vietnamese"), "vi")


class StoreTests(unittest.TestCase):
    def test_store_expires_finished_jobs(self):
        store = JobStore(max_size=2, ttl_seconds=0.01)
        job = Job(id="j1", text="x", src="English", tgt="Vietnamese", max_tokens=4)
        job.status = "completed"; job.finished_at = time.time() - 1
        store.put(job)
        self.assertIsNone(store.get("j1"))


class ManagerTests(unittest.TestCase):
    def test_health_stays_loading_until_single_worker_reports_eight_tpus(self):
        manager = TranslationManager(make_config())
        manager._target_worker_count = 1
        manager._worker_status["tpu-0"] = {"worker_id": "tpu-0", "state": "loading"}
        self.assertEqual(manager.health()["state"], "loading")
        manager._worker_status["tpu-0"] = {
            "worker_id": "tpu-0", "state": "ready",
            "metadata": {"device_count": 8, "mesh_shape": [1, 8]},
        }
        health = manager.health()
        self.assertEqual(health["state"], "ready")
        self.assertEqual(health["expected_workers"], 1)

    def test_submit_rejects_when_worker_not_ready(self):
        manager = TranslationManager(make_config())
        with self.assertRaises(WorkerNotReadyError):
            manager.submit({"text":"x","src":"English","tgt":"Vietnamese","max_tokens":8})

    def test_bounded_queue_rejects_excess_job(self):
        manager = TranslationManager(make_config(max_queue_size=1))
        manager.task_queue = queue.Queue(maxsize=1)
        manager._worker_status["tpu-0"] = {
            "worker_id": "tpu-0",
            "state": "ready",
            "metadata": {"device_count": 8, "mesh_shape": [1, 8]},
        }
        payload = {"text": "x", "src": "English", "tgt": "Vietnamese", "max_tokens": 8}
        manager.submit(payload)
        with self.assertRaises(QueueFullError):
            manager.submit(payload)




class RestartOptionTests(unittest.TestCase):
    def test_restart_options_reject_non_boolean_wait_and_bad_timeout(self):
        cfg = make_config(shutdown_timeout=30)
        self.assertEqual(parse_restart_options({}, cfg), (True, 30.0))
        with self.assertRaises(ValueError):
            parse_restart_options({"wait_for_jobs": "yes"}, cfg)
        with self.assertRaises(ValueError):
            parse_restart_options({"timeout": "not-a-number"}, cfg)


class WorkerContractTests(unittest.TestCase):
    def test_public_metrics_do_not_echo_internal_prompt(self):
        metrics = sanitize_public_metrics({"prompt": "secret/internal prompt", "max_length": 256})
        self.assertNotIn("prompt", metrics)
        self.assertEqual(metrics["max_length"], 256)

    def test_v5e8_fallback_validation_matches_v10_preflight_contract(self):
        validate_tpu_fallback(True, {"TG_TPU_FALLBACK_APPLIED": "true", "TPU_ACCELERATOR_TYPE": "v5e-8"})
        with self.assertRaises(RuntimeError):
            validate_tpu_fallback(True, {"TG_TPU_FALLBACK_APPLIED": "true", "TPU_ACCELERATOR_TYPE": "v4-8"})
        # When the fallback was not used, v10 trusts JAX device discovery instead of guessing from env labels.
        validate_tpu_fallback(True, {"TG_TPU_FALLBACK_APPLIED": "false", "TPU_ACCELERATOR_TYPE": ""})

class FakeManager:
    def __init__(self, cfg, ready=True, submit_error=None):
        self.config = cfg
        self.store = JobStore(cfg.max_store_size, cfg.result_ttl_seconds)
        self.ready = ready
        self.submit_error = submit_error

    def health(self):
        return {
            "state": "ready" if self.ready else "loading",
            "ready": self.ready,
            "ready_workers": 1 if self.ready else 0,
            "expected_workers": 1,
            "accepting_jobs": True,
            "jobs": self.store.stats(),
            "workers": [],
            "runtime": {
                "model": "translategemma_12b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": 8,
                "mesh": [1, 8],
            },
        }

    def submit(self, payload, request_id=None):
        if self.submit_error:
            raise self.submit_error
        if not self.ready:
            raise WorkerNotReadyError("TPU worker not ready", health=self.health())
        job = Job(
            id="job-test",
            text=payload.get("text", ""),
            src=payload["src"],
            tgt=payload["tgt"],
            max_tokens=payload["max_tokens"],
            request_id=request_id,
        )
        job.status = "completed"
        job.result = "Xin chào"
        job.runtime = {"backend": "jax", "tpu_devices": 8, "mesh": [1, 8]}
        job.done.set()
        self.store.put(job)
        return job

    def shutdown(self, wait_for_jobs, timeout):
        return True


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this offline build container")
class ApiTests(unittest.TestCase):
    def test_live_is_200_while_ready_is_503_during_load(self):
        cfg = make_config()
        app = create_app(Runtime(cfg, FakeManager(cfg, ready=False)))
        client = app.test_client()
        self.assertEqual(client.get("/health/live").status_code, 200)
        self.assertEqual(client.get("/health/ready").status_code, 503)

    def test_translate_requires_api_key_and_returns_runtime(self):
        cfg = make_config(api_key="secret")
        app = create_app(Runtime(cfg, FakeManager(cfg)))
        client = app.test_client()
        payload = {
            "text": "Hi",
            "source_lang": "English",
            "target_lang": "Vietnamese",
        }
        self.assertEqual(client.post("/translate", json=payload).status_code, 401)
        response = client.post(
            "/translate",
            headers={"Authorization": "Bearer secret"},
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["runtime"]["tpu_devices"], 8)

    def test_invalid_json_is_400(self):
        cfg = make_config(api_key="secret")
        client = create_app(Runtime(cfg, FakeManager(cfg))).test_client()
        response = client.post(
            "/translate",
            headers={"Authorization": "Bearer secret", "Content-Type": "text/plain"},
            data="not-json",
        )
        self.assertEqual(response.status_code, 400)

    def test_async_accepts_job_and_result_endpoint_returns_completion(self):
        cfg = make_config(api_key="secret")
        client = create_app(Runtime(cfg, FakeManager(cfg))).test_client()
        headers = {"Authorization": "Bearer secret"}
        response = client.post(
            "/translate/async",
            headers=headers,
            json={
                "text": "Hi",
                "source_lang": "English",
                "target_lang": "Vietnamese",
            },
        )
        self.assertEqual(response.status_code, 202)
        result = client.get("/result/job-test", headers=headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.get_json()["translation"], "Xin chào")

    def test_queue_full_is_429(self):
        cfg = make_config(api_key="secret")
        manager = FakeManager(cfg, submit_error=QueueFullError("Translation queue is full"))
        client = create_app(Runtime(cfg, manager)).test_client()
        response = client.post(
            "/translate",
            headers={"Authorization": "Bearer secret"},
            json={
                "text": "Hi",
                "source_lang": "English",
                "target_lang": "Vietnamese",
            },
        )
        self.assertEqual(response.status_code, 429)

    def test_worker_unavailable_is_503(self):
        cfg = make_config(api_key="secret")
        client = create_app(Runtime(cfg, FakeManager(cfg, ready=False))).test_client()
        response = client.post(
            "/translate",
            headers={"Authorization": "Bearer secret"},
            json={
                "text": "Hi",
                "source_lang": "English",
                "target_lang": "Vietnamese",
            },
        )
        self.assertEqual(response.status_code, 503)

    def test_oversize_request_is_413(self):
        cfg = make_config(api_key="secret", max_request_bytes=128)
        client = create_app(Runtime(cfg, FakeManager(cfg))).test_client()
        response = client.post(
            "/translate",
            headers={"Authorization": "Bearer secret"},
            json={
                "text": "x" * 1000,
                "source_lang": "English",
                "target_lang": "Vietnamese",
            },
        )
        self.assertEqual(response.status_code, 413)

    def test_restart_requires_separate_secret(self):
        cfg = make_config(api_key="secret", restart_secret="restart-secret")
        client = create_app(Runtime(cfg, FakeManager(cfg))).test_client()
        self.assertEqual(client.post("/restart", json={}).status_code, 401)



class SourceContractTests(unittest.TestCase):
    def test_coordinator_entrypoint_does_not_import_jax_keras(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "src/server.py").read_text() if (root / "src/server.py").exists() else ""
        forbidden = ("import jax", "import keras", "import keras_hub")
        self.assertFalse(any(item in text for item in forbidden))


if __name__ == '__main__':
    unittest.main()
