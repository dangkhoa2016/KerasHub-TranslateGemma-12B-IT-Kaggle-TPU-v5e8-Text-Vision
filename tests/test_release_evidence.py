import importlib.util
import json
import os
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"
FROZEN = {
    "src/translategemma_server/tpu/engine.py": "121c6adce3f79094dff1a3509fc9592b06a864692a4539d60c39552b3b97d70d",
    "src/translategemma_server/tpu/distribution.py": "e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13",
    "src/translategemma_server/tpu/generation.py": "5ed47b7760f98064523e38476d11e98ff59feee65f08457f742338e253c4a511",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReleaseEvidenceTests(unittest.TestCase):
    def test_version_is_public_v1_0_0(self):
        sys.path.insert(0, str(ROOT / "src"))
        import translategemma_server
        self.assertEqual(translategemma_server.__version__, "v1.0.0")

    def test_frozen_tpu_engine_files_are_unchanged(self):
        import hashlib
        for rel, expected in FROZEN.items():
            digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, rel)

    def test_worker_restricts_jax_platforms_before_import(self):
        text = (ROOT / "src/translategemma_server/workers/worker.py").read_text(encoding="utf-8")
        marker = 'os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")'
        self.assertIn(marker, text)
        self.assertLess(text.index(marker), text.index("import jax"))

    def test_worker_installs_only_exact_known_python_warning_filters(self):
        module = load_module(ROOT / "src/translategemma_server/workers/worker.py", "worker_public")
        func = getattr(module, "configure_known_runtime_warning_filters", None)
        self.assertIsNotNone(func)
        if func is None:
            return
        with warnings.catch_warnings():
            warnings.resetwarnings()
            func()
            filters = warnings.filters
        rendered = [
            (
                action,
                getattr(message, "pattern", None),
                getattr(module_re, "pattern", None),
            )
            for action, message, _category, module_re, _lineno in filters
        ]
        self.assertIn(
            (
                "ignore",
                r"Transparent hugepages are not enabled\..*",
                r"jax\._src\.cloud_tpu_init",
            ),
            rendered,
        )
        self.assertIn(
            (
                "ignore",
                r"`build\(\)` was called on layer 'gemma3_causal_lm_1'.*",
                r"keras\.src\.layers\.layer",
            ),
            rendered,
        )

    def test_runtime_log_sanitizer_collapses_known_noise_and_preserves_unknown(self):
        path = ROOT / "scripts/sanitize_runtime_log.py"
        self.assertTrue(path.is_file())
        if not path.is_file():
            return
        module = load_module(path, "sanitize_public")
        raw = "\n".join(
            [
                "/x/jax/_src/cloud_tpu_init.py:88: UserWarning: Transparent hugepages are not enabled. TPU runtime startup noise",
                "  warnings.warn(",
                "WARNING: Logging before InitGoogle() is written to STDERR",
                'E0000 common_lib.cc:648] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8471 in any of the 0 ports provided in `tpu_process_addresses`="local"',
                "=== Source Location Trace: ===",
                "learning/45eac/tfrc/runtime/common_lib.cc:238",
                "/x/keras/src/layers/layer.py:431: UserWarning: `build()` was called on layer 'gemma3_causal_lm_1', however the layer does not have a `build()` method implemented",
                "  warnings.warn(",
                "2026-08-16: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)",
                "E9999 unexpected fatal-looking line that must survive",
            ]
        )
        cleaned, summary = module.sanitize_runtime_log_text(raw)
        self.assertNotIn("E0000", cleaned)
        self.assertNotIn("UNKNOWN ERROR (303)", cleaned)
        self.assertNotIn("UserWarning: Transparent hugepages", cleaned)
        self.assertNotIn("UserWarning: `build()`", cleaned)
        self.assertIn("E9999 unexpected fatal-looking line that must survive", cleaned)
        self.assertEqual(summary["transparent_hugepages"], 1)
        self.assertEqual(summary["slice_builder_8471"], 1)
        self.assertEqual(summary["keras_build_warning"], 1)
        self.assertEqual(summary["cuda_probe_303"], 1)
        cuda_advisory = module.ADVISORIES["cuda_probe_303"]
        self.assertIn("successful TPU-only run", cuda_advisory)
        self.assertIn("raw log retained for audit", cuda_advisory)
        self.assertNotIn("prevent this probe", cuda_advisory)
        self.assertNotIn("JAX_PLATFORMS", cuda_advisory)

    def test_public_docs_do_not_claim_jax_platforms_prevents_cuda_probe(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_vi = (ROOT / "README.vi.md").read_text(encoding="utf-8")
        self.assertNotIn("so JAX does not probe an unused CUDA backend", readme)
        self.assertNotIn("để tránh JAX probe CUDA không cần thiết", readme_vi)
        self.assertIn("cuInit 303", readme)
        self.assertIn("cuInit 303", readme_vi)

    def test_unit_test_wrapper_sets_src_pythonpath(self):
        path = ROOT / "scripts/test_unit.sh"
        self.assertTrue(path.is_file())
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            self.assertIn('PYTHONPATH="$ROOT_DIR/src', text)
            self.assertIn("unittest discover -s tests -v", text)


if __name__ == "__main__":
    unittest.main()
