import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VersionContractTests(unittest.TestCase):
    def test_package_version_is_public_v1_0_0(self):
        import translategemma_server
        self.assertEqual(translategemma_server.__version__, "v1.0.0")

    def test_info_payload_reports_public_v1_0_0(self):
        from translategemma_server.api.app import build_info_payload
        payload = build_info_payload({
            "state": "ready",
            "worker_generation": 1,
            "runtime": {
                "model": "translategemma_12b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": 8,
                "mesh": [1, 8],
            },
        })
        self.assertEqual(payload["api_version"], "v1.0.0")


class DependencyHygieneTests(unittest.TestCase):
    def test_known_keras_nlp_conflict_is_advisory_not_runtime_error(self):
        module = load_module(ROOT / "scripts/check_dependency_hygiene.py", "dep_hygiene_public")
        report = module.analyze_versions({
            "keras": "3.15.1", "keras-hub": "0.31.0", "numpy": "2.5.2",
            "flask": "3.1.3", "pillow": "11.3.0", "keras-nlp": "0.29.1",
        })
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("keras-nlp" in item for item in report["advisories"]))

    def test_proven_runtime_version_mismatch_is_error(self):
        module = load_module(ROOT / "scripts/check_dependency_hygiene.py", "dep_hygiene_public_bad")
        report = module.analyze_versions({
            "keras": "3.15.1", "keras-hub": "0.29.1", "numpy": "2.5.2",
            "flask": "3.1.3", "pillow": "11.3.0",
        })
        self.assertTrue(any("keras-hub" in item for item in report["errors"]))


class PackagingSecurityTests(unittest.TestCase):
    def test_secret_scanner_allows_example_but_rejects_runtime_secret_files(self):
        module = load_module(ROOT / "scripts/secret_scan.py", "secret_scan_public")
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            (project / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
            self.assertEqual(module.scan_tree(project), [])
            (project / ".env").write_text("API_KEY=real-secret-value\n", encoding="utf-8")
            self.assertTrue(any(".env" in issue for issue in module.scan_tree(project)))

    def test_source_zip_contains_sha256_manifest_covering_payload_files(self):
        package = load_module(ROOT / "scripts/package_source.py", "package_source_public")
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "demo-v1.0.0"
            (project / "src").mkdir(parents=True)
            (project / "src/a.py").write_text("print('hello')\n", encoding="utf-8")
            (project / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
            out = Path(tmp) / "source.zip"
            package.create_source_zip(out, project)
            with zipfile.ZipFile(out) as zf:
                names = zf.namelist()
                manifest_name = f"{project.name}/SOURCE-MANIFEST.sha256"
                self.assertIn(manifest_name, names)
                manifest = zf.read(manifest_name).decode("utf-8")
                self.assertIn("src/a.py", manifest)
                self.assertNotIn("SOURCE-MANIFEST.sha256", manifest)


class DocsAndCiTests(unittest.TestCase):
    def test_hardening_docs_and_ci_exist(self):
        for rel in ("LICENSE", "NOTICE", "SECURITY.md", "CONTRIBUTING.md", "docs/API.md", ".github/workflows/ci.yml", "requirements-ci.txt"):
            self.assertTrue((ROOT / rel).is_file(), rel)
        self.assertFalse((ROOT / "docs/PRIVATE_KAGGLE_DEV.md").exists())
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("bash scripts/test_unit.sh", ci)
        self.assertIn("unittest discover -s tests -v", (ROOT / "scripts/test_unit.sh").read_text(encoding="utf-8"))
        self.assertIn("secret_scan.py", ci)
        self.assertNotIn("jax.devices", ci)
        self.assertNotIn("scripts/setup.sh", ci)


class ReleaseWorkflowTests(unittest.TestCase):
    def test_tag_release_workflow_builds_integrity_checked_public_artifacts(self):
        path = ROOT / ".github/workflows/release.yml"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("tags:", text)
        self.assertIn("v*", text)
        self.assertIn("bash scripts/test_unit.sh", text)
        self.assertIn("scripts/package_source.py", text)
        self.assertIn("kaggle-tpu-v5e8-text-vision.ipynb", text)
        self.assertIn("sha256sum", text)
        self.assertIn("gh release", text)
        self.assertNotIn("scripts/setup.sh", text)
        self.assertNotIn("jax.devices", text)


class NotebookContractTests(unittest.TestCase):
    def test_current_notebook_is_git_main_public_test_workflow(self):
        path = ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")
        self.assertIn("git -C \"$ROOT\" fetch origin main", code)
        self.assertIn("git -C \"$ROOT\" reset --hard origin/main", code)
        self.assertIn("scripts/setup.sh", code)
        self.assertIn("scripts/wait_ready.py", code)
        self.assertIn("--multipart", code)
        for forbidden in ("private_dev", ".kaggle-ssh", "NGROK_AUTHTOKEN", "SSH_PUBLIC_KEY", "package_full_backup.py", "restore-private.sh"):
            self.assertNotIn(forbidden, code)
        self.assertNotIn("import jax", code)
        self.assertNotIn("import keras", code)
        self.assertNotIn("import keras_hub", code)


class EngineFrozenTests(unittest.TestCase):
    def test_proven_tpu_engine_checksums_are_unchanged(self):
        expected = {
            "engine.py": "121c6adce3f79094dff1a3509fc9592b06a864692a4539d60c39552b3b97d70d",
            "distribution.py": "e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13",
            "generation.py": "5ed47b7760f98064523e38476d11e98ff59feee65f08457f742338e253c4a511",
        }
        for name, digest in expected.items():
            actual = hashlib.sha256((ROOT / "src/translategemma_server/tpu" / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, name)


if __name__ == "__main__":
    unittest.main()
