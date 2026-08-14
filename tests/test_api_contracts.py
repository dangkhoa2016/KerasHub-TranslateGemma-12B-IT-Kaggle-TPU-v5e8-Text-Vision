import io
import json
import re
import unittest
from pathlib import Path

from PIL import Image

from translategemma_server.api import app as app_module
from translategemma_server.core import validation
from translategemma_server.core.config import Config
from translategemma_server.jobs.models import Job
from translategemma_server.workers.manager import TranslationManager


def make_config():
    return Config.for_tests()


class MultipartValidationTests(unittest.TestCase):
    def test_binary_image_parser_exists_and_matches_base64_contract(self):
        parser = getattr(validation, "parse_image_translation_binary", None)
        self.assertTrue(callable(parser), "binary image parser is missing")
        image = Image.new("RGB", (9, 7), "white")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        payload = parser(
            buf.getvalue(),
            {
                "source_lang": "English",
                "target_lang": "Vietnamese",
                "max_new_tokens": "32",
            },
            make_config(),
        )
        self.assertEqual(payload["image"].size, (9, 7))
        self.assertEqual(payload["src"], "English")
        self.assertEqual(payload["tgt"], "Vietnamese")
        self.assertEqual(payload["max_tokens"], 32)

    def test_binary_image_parser_rejects_empty_or_invalid_images(self):
        parser = getattr(validation, "parse_image_translation_binary", None)
        self.assertTrue(callable(parser), "binary image parser is missing")
        form = {"source_lang": "English", "target_lang": "Vietnamese"}
        with self.assertRaises(Exception):
            parser(b"", form, make_config())
        with self.assertRaises(Exception):
            parser(b"not-an-image", form, make_config())


class RequestIdTests(unittest.TestCase):
    def test_request_id_normalization_preserves_safe_values_and_replaces_unsafe(self):
        normalizer = getattr(app_module, "normalize_request_id", None)
        self.assertTrue(callable(normalizer), "request-id normalizer is missing")
        self.assertEqual(normalizer("client-123:abc"), "client-123:abc")
        generated = normalizer("bad request id with spaces")
        self.assertRegex(generated, r"^req-[0-9a-f]{24}$")
        self.assertRegex(normalizer(None), r"^req-[0-9a-f]{24}$")
        self.assertRegex(normalizer("x" * 129), r"^req-[0-9a-f]{24}$")

    def test_job_public_dict_includes_request_id(self):
        job = Job(
            id="job-1",
            text="hello",
            src="English",
            tgt="Vietnamese",
            max_tokens=8,
        )
        self.assertTrue(hasattr(job, "request_id"), "Job.request_id is missing")
        job.request_id = "req-test"
        self.assertEqual(job.public_dict()["request_id"], "req-test")

    def test_manager_submit_accepts_request_id(self):
        manager = TranslationManager(make_config())
        manager.task_queue = __import__("queue").Queue(maxsize=4)
        manager._worker_status["tpu-0"] = {
            "worker_id": "tpu-0",
            "state": "ready",
            "metadata": {"device_count": 8, "mesh_shape": [1, 8]},
        }
        payload = {"text": "x", "src": "English", "tgt": "Vietnamese", "max_tokens": 8}
        try:
            job = manager.submit(payload, request_id="req-abc")
        except TypeError as exc:
            self.fail(f"TranslationManager.submit lacks request_id support: {exc}")
        self.assertEqual(job.request_id, "req-abc")


class ApiContractTests(unittest.TestCase):
    def test_app_source_contains_multipart_info_and_request_id_contracts(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "src/translategemma_server/api/app.py").read_text(encoding="utf-8")
        self.assertIn('request.files.get("image")', text)
        self.assertIn('@app.get("/info")', text)
        self.assertIn('X-Request-ID', text)
        self.assertIn('multipart/form-data', text)

    def test_info_schema_builder_is_safe_and_has_capabilities(self):
        builder = getattr(app_module, "build_info_payload", None)
        self.assertTrue(callable(builder), "safe /info payload builder is missing")
        health = {
            "state": "ready",
            "worker_generation": 2,
            "runtime": {
                "model": "translategemma_12b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": 8,
                "mesh": [1, 8],
                "model_path": "/secret/model/path",
            },
            "workers": [{"error": "secret traceback", "pid": 123}],
        }
        payload = builder(health)
        encoded = json.dumps(payload)
        self.assertEqual(payload["api_version"], "v1.0.0")
        self.assertEqual(payload["state"], "ready")
        self.assertIn("multipart", payload["capabilities"]["image_transports"])
        for forbidden in ("/secret/model/path", "secret traceback", '"pid"'):
            self.assertNotIn(forbidden, encoded)


class TunnelAndNotebookContractTests(unittest.TestCase):
    def test_tunnel_script_has_local_liveness_preflight(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/run_tunnel.sh").read_text(encoding="utf-8")
        self.assertIn("/health/live", text)
        self.assertIn("demo_info.py", text)

    def test_demo_info_script_exists_and_never_reads_secret_value(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts/demo_info.py"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"api_key\.txt.*read_text")
        self.assertIn("clients/python/translategemma_client.py", text)

    def test_public_notebook_contract(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"
        self.assertTrue(path.is_file())
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        for forbidden in ("import jax", "import keras", "import keras_hub"):
            self.assertNotIn(forbidden, code)
        self.assertIn("scripts/wait_ready.py", code)
        self.assertIn("/info", code)
        self.assertIn("--multipart", code)


class EngineIntegrityTests(unittest.TestCase):
    EXPECTED = {
        "engine.py": "121c6adce3f79094dff1a3509fc9592b06a864692a4539d60c39552b3b97d70d",
        "distribution.py": "e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13",
        "generation.py": "5ed47b7760f98064523e38476d11e98ff59feee65f08457f742338e253c4a511",
    }

    def test_proven_tpu_engine_files_are_byte_identical_to_v2(self):
        import hashlib
        root = Path(__file__).resolve().parents[1] / "src/translategemma_server/tpu"
        for name, expected in self.EXPECTED.items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)


if __name__ == "__main__":
    unittest.main()

class PackagingTests(unittest.TestCase):
    def test_clean_source_excludes_private_backup_marker_files(self):
        import importlib.util
        import tempfile
        import zipfile
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("pkg_source_v3", root / "scripts/package_source.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "project"
            (fake / "src").mkdir(parents=True)
            (fake / "src/a.py").write_text("x", encoding="utf-8")
            (fake / "PRIVATE-BACKUP-WARNING.txt").write_text("private", encoding="utf-8")
            (fake / "PRIVATE-BACKUP-MANIFEST.json").write_text("{}", encoding="utf-8")
            out = Path(tmp) / "source.zip"
            module.create_source_zip(out, fake)
            names = zipfile.ZipFile(out).namelist()
            self.assertFalse(any("PRIVATE-BACKUP-" in name for name in names))

    def test_full_backup_helper_is_removed_from_public_source(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "scripts/package_full_backup.py").exists())
        self.assertFalse((root / "restore-private.sh").exists())


import importlib.util as _importlib_util
HAS_FLASK_V3 = _importlib_util.find_spec("flask") is not None
if HAS_FLASK_V3:
    from translategemma_server.api.app import Runtime, create_app
    from translategemma_server.jobs.store import JobStore


class _V3FakeManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = JobStore(cfg.max_store_size, cfg.result_ttl_seconds)

    def health(self):
        return {
            "state": "ready",
            "ready": True,
            "ready_workers": 1,
            "expected_workers": 1,
            "worker_generation": 1,
            "accepting_jobs": True,
            "jobs": self.store.stats(),
            "workers": [],
            "runtime": {
                "model": "translategemma_12b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": 8,
                "mesh": [1, 8],
                "model_path": "/must/not/leak",
            },
        }

    def submit(self, payload, request_id=None):
        job = Job(
            id="job-v3",
            text=payload.get("text", ""),
            src=payload["src"],
            tgt=payload["tgt"],
            max_tokens=payload["max_tokens"],
            request_id=request_id,
        )
        job.status = "completed"
        job.result = "Xin chào"
        job.done.set()
        self.store.put(job)
        return job

    def shutdown(self, wait_for_jobs, timeout):
        return True


@unittest.skipUnless(HAS_FLASK_V3, "Flask is not installed in this offline build container")
class FlaskApiTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.for_tests()
        self.cfg = __import__("dataclasses").replace(self.cfg, api_key="secret")
        self.manager = _V3FakeManager(self.cfg)
        self.client = create_app(Runtime(self.cfg, self.manager)).test_client()
        self.headers = {"Authorization": "Bearer secret"}

    def test_request_id_is_preserved_in_header_and_job_payload(self):
        response = self.client.post(
            "/translate",
            headers={**self.headers, "X-Request-ID": "demo-123"},
            json={"text": "Hi", "source_lang": "English", "target_lang": "Vietnamese"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "demo-123")
        self.assertEqual(response.get_json()["request_id"], "demo-123")

    def test_multipart_image_uses_existing_image_endpoint(self):
        image = Image.new("RGB", (8, 8), "white")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        response = self.client.post(
            "/translate/image",
            headers=self.headers,
            data={
                "image": (buf, "sample.png"),
                "source_lang": "English",
                "target_lang": "Vietnamese",
                "max_new_tokens": "32",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["translation"], "Xin chào")

    def test_info_requires_auth_and_does_not_leak_model_path(self):
        self.assertEqual(self.client.get("/info").status_code, 401)
        response = self.client.get("/info", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        raw = response.get_data(as_text=True)
        self.assertIn('"api_version":"v1.0.0"', raw)
        self.assertNotIn("/must/not/leak", raw)
