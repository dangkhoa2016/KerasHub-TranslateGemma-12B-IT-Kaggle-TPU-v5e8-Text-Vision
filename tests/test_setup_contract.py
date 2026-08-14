import importlib.util
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SetupContractTests(unittest.TestCase):
    def test_setup_uses_conflict_aware_noninteractive_pip_flags(self):
        text = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")
        for token in (
            "--no-warn-conflicts",
            "--root-user-action=ignore",
            "--disable-pip-version-check",
            "--no-color",
        ):
            self.assertIn(token, text)

    def test_setup_uses_isolated_tpu_preflight_filter(self):
        text = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/check_tpu_preflight.py", text)
        self.assertNotIn('import jax\nprint("JAX:"', text)


    def test_setup_bootstraps_missing_libtpu_without_replacing_jax(self):
        text = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/ensure_libtpu.py", text)
        self.assertNotIn('pip install "jax[tpu]"', text)
        self.assertNotIn("pip install -U jax", text)
        helper = (ROOT / "scripts/ensure_libtpu.py").read_text(encoding="utf-8")
        self.assertIn("--no-deps", helper)

    def test_libtpu_bootstrap_policy_keeps_existing_runtime_and_installs_only_when_missing(self):
        module = load_module(ROOT / "scripts/ensure_libtpu.py", "ensure_libtpu_public")
        self.assertEqual(module.plan_action(None, "auto"), "install")
        self.assertEqual(module.plan_action("0.0.17", "auto"), "keep")
        self.assertEqual(module.plan_action("0.0.42.1", "auto"), "keep")
        self.assertEqual(module.plan_action(None, "false"), "error")

    def test_setup_supports_required_auto_and_skip_tpu_preflight_modes(self):
        text = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")
        self.assertIn("TPU_PREFLIGHT_MODE", text)
        for mode in ("required", "auto", "skip"):
            self.assertIn(mode, text)

    def test_dependency_hygiene_reclassifies_known_keras_nlp_conflict_as_advisory(self):
        module = load_module(ROOT / "scripts/check_dependency_hygiene.py", "dep_public")
        report = module.analyze_versions(
            {
                "keras": "3.15.1",
                "keras-hub": "0.31.0",
                "numpy": "2.5.2",
                "flask": "3.1.3",
                "pillow": "12.2.0",
                "keras-nlp": "0.29.1",
            }
        )
        self.assertEqual(report["errors"], [])
        self.assertEqual(report.get("warnings", []), [])
        self.assertTrue(any("keras-nlp" in item for item in report["advisories"]))

    def test_pip_check_allows_only_known_keras_nlp_conflict(self):
        module = load_module(ROOT / "scripts/check_dependency_hygiene.py", "dep_public_pip")
        known = "keras-nlp 0.29.1 has requirement keras-hub==0.29.1, but you have keras-hub 0.31.0."
        report = module.classify_pip_check([known])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["advisories"], [known])

        unknown = "keras-hub 0.31.0 has requirement kagglesdk==0.1.28, but you have kagglesdk 0.1.30."
        report = module.classify_pip_check([known, unknown])
        self.assertEqual(report["advisories"], [known])
        self.assertEqual(report["errors"], [unknown])

        unrelated = "some-package 1.0 has requirement numpy==2.4.0, but you have numpy 2.5.2."
        report = module.classify_pip_check([known, unrelated])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["ignored"], [unrelated])

    def test_tpu_preflight_filters_only_known_success_noise(self):
        module = load_module(ROOT / "scripts/check_tpu_preflight.py", "tpu_public")
        stderr = "\n".join(
            [
                "/x/cloud_tpu_init.py:88: UserWarning: Transparent hugepages are not enabled.",
                "  warnings.warn(",
                "WARNING: Logging before InitGoogle() is written to STDERR",
                'E0000 common_lib.cc:648] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8471 in any of the 0 ports provided in `tpu_process_addresses`="local"',
                "=== Source Location Trace: ===",
                "learning/45eac/tfrc/runtime/common_lib.cc:238",
            ]
        )
        filtered = module.filter_known_success_noise(stderr)
        self.assertEqual(filtered["unknown"], [])
        self.assertTrue(filtered["thp"])
        self.assertTrue(filtered["metric_port"])

        unknown = module.filter_known_success_noise(stderr + "\nUNEXPECTED TPU STDERR")
        self.assertEqual(unknown["unknown"], ["UNEXPECTED TPU STDERR"])

    def test_version_is_public_v1_0_0(self):
        env = dict(PYTHONPATH=str(ROOT / "src"))
        proc = subprocess.run(
            ["python3", "-c", "import translategemma_server as m; print(m.__version__)"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.stdout.strip(), "v1.0.0")


if __name__ == "__main__":
    unittest.main()
