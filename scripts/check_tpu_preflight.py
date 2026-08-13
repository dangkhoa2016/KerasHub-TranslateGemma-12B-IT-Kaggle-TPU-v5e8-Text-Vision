#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THP_PATH = Path("/sys/kernel/mm/transparent_hugepage/enabled")

CHILD = r'''
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
import jax

devices = jax.devices("tpu")
expected = int(os.environ.get("EXPECTED_TPU_DEVICES", "8"))
print("JAX:", getattr(jax, "__version__", "unknown"))
print("TPU devices:", devices)
if len(devices) != expected:
    raise SystemExit(f"Expected {expected} TPU devices, found {len(devices)}")
'''


def try_enable_transparent_hugepages() -> bool:
    try:
        if not THP_PATH.exists():
            return False
        current = THP_PATH.read_text(encoding="utf-8", errors="replace")
        if "[always]" in current:
            return True
        if os.access(THP_PATH, os.W_OK):
            THP_PATH.write_text("always\n", encoding="utf-8")
            current = THP_PATH.read_text(encoding="utf-8", errors="replace")
            return "[always]" in current
    except OSError:
        return False
    return False


def filter_known_success_noise(stderr: str) -> dict[str, object]:
    unknown: list[str] = []
    thp = False
    metric_port = False
    skip_trace = False
    for raw in stderr.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "Transparent hugepages are not enabled" in line or line == "warnings.warn(":
            thp = True
            continue
        if "Logging before InitGoogle() is written to STDERR" in line:
            continue
        if "Could not set metric server port" in line and "SliceBuilder port 8471" in line:
            metric_port = True
            skip_trace = True
            continue
        if skip_trace and (line == "=== Source Location Trace: ===" or line.startswith("learning/")):
            continue
        skip_trace = False
        unknown.append(line)
    return {"unknown": unknown, "thp": thp, "metric_port": metric_port}


def main() -> int:
    thp_enabled = try_enable_transparent_hugepages()
    proc = subprocess.run(
        [sys.executable, "-c", CHILD],
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        return proc.returncode

    filtered = filter_known_success_noise(proc.stderr)
    for line in filtered["unknown"]:
        print(f"[tpu-preflight] STDERR: {line}", file=sys.stderr)
    if filtered["thp"] and not thp_enabled:
        print(
            "[tpu-preflight] ADVISORY: transparent hugepages could not be enabled in this Kaggle VM; "
            "8-device TPU validation still succeeded."
        )
    elif thp_enabled:
        print("[tpu-preflight] transparent hugepages: enabled")
    if filtered["metric_port"]:
        print(
            "[tpu-preflight] ADVISORY: suppressed the known libtpu SliceBuilder metric-port 8471 "
            "message after successful TPU device discovery."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
