import importlib.util
import io
import unittest
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClientContractTests(unittest.TestCase):
    def test_python_client_exists_and_supports_polling_and_multipart(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "clients/python/translategemma_client.py"
        self.assertTrue(path.is_file())
        module = load_module(path, "tg_client")
        self.assertTrue(hasattr(module, "TranslateGemmaClient"))
        self.assertTrue(hasattr(module, "encode_multipart"))

    def test_python_submit_and_wait_polls_202_result(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "clients/python/translategemma_client.py"
        self.assertTrue(path.is_file())
        module = load_module(path, "tg_client_poll")
        client = module.TranslateGemmaClient("http://example.test", api_key="k", poll_interval=0)
        calls = []

        def fake_request(method, path, *, json_body=None, body=None, headers=None, timeout=None):
            calls.append((method, path))
            if len(calls) == 1:
                return 202, {"job_id": "job-1", "result_url": "/result/job-1"}, {"x-request-id": "req-1"}
            return 200, {"job_id": "job-1", "status": "completed", "translation": "Xin chao"}, {"x-request-id": "req-1"}

        client._request = fake_request
        result = client.translate_text("Hi", "English", "Vietnamese")
        self.assertEqual(result["translation"], "Xin chao")
        self.assertEqual(calls, [("POST", "/translate"), ("GET", "/result/job-1")])

    def test_python_multipart_encoder_contains_fields_and_file(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "clients/python/translategemma_client.py"
        self.assertTrue(path.is_file())
        module = load_module(path, "tg_client_multipart")
        content_type, body = module.encode_multipart(
            {"source_lang": "English", "target_lang": "Vietnamese"},
            "image",
            "sample.png",
            b"PNGDATA",
            "image/png",
        )
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="source_lang"', body)
        self.assertIn(b'name="image"; filename="sample.png"', body)
        self.assertIn(b"PNGDATA", body)

    def test_node_client_uses_builtins_and_no_npm_dependency_manifest(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "clients/node/translategemma-client.mjs"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("fetch(", text)
        self.assertIn("new FormData", text)
        self.assertIn("new Blob", text)
        self.assertFalse((root / "clients/node/package.json").exists())


if __name__ == "__main__":
    unittest.main()
