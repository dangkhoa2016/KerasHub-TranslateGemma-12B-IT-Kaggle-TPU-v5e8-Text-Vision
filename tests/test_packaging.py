import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_source_zip_excludes_runtime_files(self):
        root = Path(__file__).resolve().parents[1]
        module = load_module(root / "scripts/package_source.py", "package_source")
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "project"
            fake.mkdir()
            (fake / "src").mkdir()
            (fake / "src/a.py").write_text("x")
            (fake / ".env").write_text("SECRET=x")
            (fake / "data").mkdir()
            (fake / "data/api_key.txt").write_text("x")
            (fake / "log").mkdir()
            (fake / "log/server.log").write_text("x")
            (fake / "state").mkdir()
            (fake / "state/server.pid").write_text("1")
            out = Path(tmp) / "source.zip"
            module.create_source_zip(out, fake)
            names = zipfile.ZipFile(out).namelist()
            self.assertTrue(any(name.endswith("/src/a.py") for name in names))
            self.assertFalse(any(name.endswith("/.env") or "/data/api_key.txt" in name or "/log/" in name or "/state/" in name for name in names))


class NotebookContractTests(unittest.TestCase):
    def test_current_notebook_exists_and_does_not_import_jax_in_kernel_cells(self):
        import json
        root = Path(__file__).resolve().parents[1]
        path = root / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"
        self.assertTrue(path.is_file())
        nb = json.loads(path.read_text())
        code = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        self.assertNotIn("import jax", code)
        self.assertIn("scripts/start.sh", code)
        self.assertIn("scripts/wait_ready.py", code)


class SmokeClientTests(unittest.TestCase):
    def test_sync_smoke_client_polls_when_server_returns_202(self):
        root = Path(__file__).resolve().parents[1]
        module = load_module(root / "scripts/api_smoke.py", "api_smoke")
        calls = []
        def fake_request(method, path, payload=None):
            calls.append((method, path))
            if len(calls) == 1:
                return 202, {"job_id": "job-1", "status": "processing"}
            return 200, {"job_id": "job-1", "status": "completed", "translation": "Xin chao"}
        status, data = module.submit_and_wait(fake_request, "/translate", {"text": "Hi"}, timeout=1, poll_interval=0)
        self.assertEqual(status, 200)
        self.assertEqual(data["translation"], "Xin chao")
        self.assertEqual(calls, [("POST", "/translate"), ("GET", "/result/job-1")])


if __name__ == "__main__":
    unittest.main()
