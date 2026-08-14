import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"


class PublicDistributionTests(unittest.TestCase):
    def test_private_dev_toolkit_is_removed(self):
        self.assertFalse((ROOT / "scripts/private_dev").exists())
        self.assertFalse((ROOT / "docs/PRIVATE_KAGGLE_DEV.md").exists())
        self.assertFalse((ROOT / "restore-private.sh").exists())
        self.assertFalse((ROOT / "scripts/package_full_backup.py").exists())

    def test_setup_has_no_private_dev_install_hook(self):
        text = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")
        self.assertNotIn("INSTALL_PRIVATE_DEV_TOOLS", text)
        self.assertNotIn("scripts/private_dev", text)

    def test_version_is_public_v1_0_0(self):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import translategemma_server
        self.assertEqual(translategemma_server.__version__, "v1.0.0")

    def test_notebook_uses_stable_checkout_and_has_no_private_dev_cells(self):
        self.assertTrue(NOTEBOOK.exists())
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertGreater(len(code_cells), 0)
        first = code_cells[0]
        self.assertIn('ROOT="/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision"', first)
        self.assertIn("git -C \"$ROOT\" fetch origin main", first)
        self.assertIn("git -C \"$ROOT\" reset --hard origin/main", first)
        joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertIn('"--porcelain"', joined)
        self.assertIn("https://github.com/dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision.git", joined)
        self.assertIn("RUN_TPU_VALIDATION", joined)
        self.assertIn('env["TPU_PREFLIGHT_MODE"] = "required" if RUN_TPU_VALIDATION else "skip"', joined)
        for forbidden in (
            "private_dev", "ENABLE_PRIVATE_DEV_TOOLS", "START_PRIVATE_SSH",
            "START_PRIVATE_TMUX", ".kaggle-ssh", "NGROK_AUTHTOKEN",
            "SSH_PUBLIC_KEY", "%%writefile setup.sh", "apt install tmux zip",
            "package_full_backup.py", "restore-private.sh",
        ):
            self.assertNotIn(forbidden, joined)

    def test_readmes_make_kaggle_github_import_the_primary_quick_start(self):
        expected = {
            "README.md": (
                "File → Import Notebook → GitHub",
                "dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision",
                "notebooks/kaggle-tpu-v5e8-text-vision.ipynb",
                "Recommended",
            ),
            "README.vi.md": (
                "File → Import Notebook → GitHub",
                "dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision",
                "notebooks/kaggle-tpu-v5e8-text-vision.ipynb",
                "Khuyến nghị",
            ),
        }
        for name, phrases in expected.items():
            text = (ROOT / name).read_text(encoding="utf-8")
            for phrase in phrases:
                self.assertIn(phrase, text, f"{name}: missing {phrase}")

    def test_notebook_markdown_explains_github_import_workflow(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertIn("File → Import Notebook → GitHub", markdown)
        self.assertIn("notebooks/kaggle-tpu-v5e8-text-vision.ipynb", markdown)
        self.assertIn("RUN_TPU_VALIDATION=True", markdown)

    def test_public_tree_has_no_temporary_or_private_release_identity(self):
        public_paths = [
            ROOT / "README.md",
            ROOT / "README.vi.md",
            ROOT / "NOTICE",
            ROOT / "SECURITY.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "docs/API.md",
            ROOT / "clients/README.md",
            ROOT / ".env.example",
            ROOT / ".github/workflows/ci.yml",
            ROOT / "scripts/demo_info.py",
            ROOT / "scripts/check_dependency_hygiene.py",
            ROOT / "src/server.py",
            ROOT / "src/translategemma_server/core/__init__.py",
            ROOT / "src/translategemma_server/jobs/__init__.py",
            ROOT / "src/translategemma_server/workers/__init__.py",
            ROOT / "src/translategemma_server/tpu/__init__.py",
            ROOT / "src/translategemma_server/api/__init__.py",
            ROOT / "src/translategemma_server/core/config.py",
            ROOT / "src/translategemma_server/workers/worker.py",
            NOTEBOOK,
        ]
        for path in public_paths:
            text = path.read_text(encoding="utf-8")
            for forbidden in ("jkdaouvsafkukofa", "x11011x", "private-v5.4", "private-v3", "private-v2"):
                self.assertNotIn(forbidden, text, f"{path}: stale {forbidden}")

        private_named_tests = [p.name for p in (ROOT / "tests").glob("test_private_*.py")]
        self.assertEqual(private_named_tests, [])

    def test_internal_process_docs_are_not_shipped(self):
        self.assertFalse((ROOT / "docs/superpowers").exists())

    def test_only_current_kaggle_notebook_is_shipped(self):
        notebooks = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(notebooks, ["kaggle-tpu-v5e8-text-vision.ipynb"])


if __name__ == "__main__":
    unittest.main()
