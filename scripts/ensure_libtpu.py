#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import os
import subprocess
import sys

DEFAULT_EXPECTED_LIBTPU_VERSION = "0.0.17"
_TRUE = {"true", "1", "yes", "auto"}
_FALSE = {"false", "0", "no"}


def normalize_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode in _TRUE:
        return "auto"
    if mode in _FALSE:
        return "false"
    raise ValueError(
        f"Invalid INSTALL_LIBTPU_IF_MISSING={value!r} "
        "(use auto/true/false)"
    )


def installed_version() -> str | None:
    try:
        return metadata.version("libtpu")
    except metadata.PackageNotFoundError:
        return None


def plan_action(current: str | None, mode: str) -> str:
    normalized = normalize_mode(mode)
    if current:
        return "keep"
    if normalized == "auto":
        return "install"
    return "error"


def install_libtpu(version: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--root-user-action=ignore",
            "--no-warn-conflicts",
            "--no-deps",
            "--no-color",
            "--progress-bar",
            "off",
            f"libtpu=={version}",
        ],
        check=True,
    )


def main() -> int:
    expected = os.environ.get(
        "EXPECTED_LIBTPU_VERSION", DEFAULT_EXPECTED_LIBTPU_VERSION
    ).strip()
    mode = os.environ.get("INSTALL_LIBTPU_IF_MISSING", "auto")
    try:
        action = plan_action(installed_version(), mode)
    except ValueError as exc:
        print(f"[libtpu] {exc}", file=sys.stderr)
        return 2

    current = installed_version()
    if action == "keep":
        print(f"[libtpu] existing runtime retained: {current}")
        if current != expected:
            print(
                f"[libtpu] ADVISORY: proven reference is {expected}, but existing "
                f"libtpu {current} is retained; TPU preflight is the functional gate."
            )
        return 0

    if action == "error":
        print(
            "[libtpu] libtpu is missing and INSTALL_LIBTPU_IF_MISSING=false",
            file=sys.stderr,
        )
        return 1

    print(f"[libtpu] missing; installing proven runtime libtpu=={expected}")
    install_libtpu(expected)
    current = installed_version()
    if current != expected:
        print(
            f"[libtpu] expected {expected} after install, found {current!r}",
            file=sys.stderr,
        )
        return 1
    print(f"[libtpu] installed: {current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
