#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import subprocess
import sys
from typing import Iterable, Mapping

PROVEN_EXACT = {
    "keras": "3.15.1",
    "keras-hub": "0.31.0",
    "numpy": "2.5.2",
}
REQUIRED_IMPORTS = {
    "flask": "flask",
    "pillow": "PIL",
}
PROJECT_PACKAGES = frozenset((*PROVEN_EXACT, *REQUIRED_IMPORTS))


def analyze_versions(versions: Mapping[str, str]) -> dict[str, list[str]]:
    normalized = {str(k).lower(): str(v) for k, v in versions.items()}
    errors: list[str] = []
    advisories: list[str] = []

    for package, expected in PROVEN_EXACT.items():
        actual = normalized.get(package)
        if actual is None:
            errors.append(f"missing {package}=={expected}")
        elif actual != expected:
            errors.append(f"expected {package}=={expected}, got {actual}")

    for package in REQUIRED_IMPORTS:
        if package not in normalized:
            errors.append(f"missing {package}")

    keras_nlp = normalized.get("keras-nlp")
    keras_hub = normalized.get("keras-hub")
    if keras_nlp and keras_hub == PROVEN_EXACT["keras-hub"]:
        advisories.append(
            "keras-nlp is preinstalled in the Kaggle image and declares an older "
            "keras-hub dependency; this project never imports keras-nlp and keeps "
            f"the Kaggle-proven keras-hub=={PROVEN_EXACT['keras-hub']} runtime."
        )

    return {"errors": errors, "advisories": advisories, "warnings": []}


def _is_known_keras_nlp_conflict(line: str) -> bool:
    value = line.lower()
    return (
        "keras-nlp" in value
        and "keras-hub" in value
        and "0.29.1" in value
        and "0.31.0" in value
    )


def classify_pip_check(lines: Iterable[str]) -> dict[str, list[str]]:
    errors: list[str] = []
    advisories: list[str] = []
    ignored: list[str] = []
    for raw in lines:
        line = str(raw).strip()
        if not line or line.lower() == "no broken requirements found.":
            continue
        if _is_known_keras_nlp_conflict(line):
            advisories.append(line)
            continue
        requiring_package = line.split(None, 1)[0].lower() if line.split() else ""
        if requiring_package in PROJECT_PACKAGES:
            errors.append(line)
        else:
            # Kaggle images can contain unrelated preinstalled-package metadata
            # conflicts, including packages that constrain NumPy differently.
            # They are outside this project's runtime dependency set unless the
            # package declaring the broken requirement is one of ours.
            ignored.append(line)
    return {"errors": errors, "advisories": advisories, "ignored": ignored}


def collect_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (*PROVEN_EXACT, *REQUIRED_IMPORTS, "keras-nlp"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            pass

    # Metadata can exist for a broken install. Confirm the runtime imports that
    # the CPU coordinator/tests actually need without importing JAX/Keras.
    for package, module in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(module) is None:
            versions.pop(package, None)
    return versions


def pip_check_report() -> dict[str, list[str]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = [*proc.stdout.splitlines(), *proc.stderr.splitlines()]
    return classify_pip_check(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TranslateGemma 12B userspace dependency hygiene")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--pip-check",
        action="store_true",
        help="Run pip check and fail only for conflicts involving project runtime packages.",
    )
    args = parser.parse_args()

    versions = collect_versions()
    report = analyze_versions(versions)
    if args.pip_check:
        pip_report = pip_check_report()
        report["errors"].extend(pip_report["errors"])
        report["advisories"].extend(pip_report["advisories"])
        ignored_count = len(pip_report["ignored"])
    else:
        ignored_count = 0

    if args.json:
        print(json.dumps({"versions": versions, "ignored_pip_conflicts": ignored_count, **report}, indent=2, sort_keys=True))
    elif not args.quiet:
        for package in sorted(versions):
            print(f"[deps] {package}={versions[package]}")
        for advisory in dict.fromkeys(report["advisories"]):
            print(f"[deps] ADVISORY: {advisory}")
        if ignored_count:
            print(f"[deps] ADVISORY: ignored {ignored_count} unrelated Kaggle preinstalled-package conflict(s).")
        for error in report["errors"]:
            print(f"[deps] ERROR: {error}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
