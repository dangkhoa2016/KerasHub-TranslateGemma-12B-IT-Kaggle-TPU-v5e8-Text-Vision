#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

CATEGORIES = (
    "transparent_hugepages",
    "slice_builder_8471",
    "keras_build_warning",
    "cuda_probe_303",
)

ADVISORIES = {
    "transparent_hugepages": (
        "[runtime-advisory] suppressed known JAX TPU transparent-hugepages "
        "startup warning; raw log retained for audit"
    ),
    "slice_builder_8471": (
        "[runtime-advisory] suppressed known successful-run libtpu "
        "InitGoogle/SliceBuilder metric-port 8471 startup noise; raw log retained for audit"
    ),
    "keras_build_warning": (
        "[runtime-advisory] suppressed exact proven-harmless Keras Gemma3 build() "
        "warning; strict checkpoint load and readiness must still be verified"
    ),
    "cuda_probe_303": (
        "[runtime-advisory] suppressed exact CUDA cuInit 303 startup signature "
        "previously observed during a successful TPU-only run; raw log retained for audit"
    ),
}


def _is_thp(line: str) -> bool:
    return "UserWarning: Transparent hugepages are not enabled." in line and "cloud_tpu_init.py" in line


def _is_init_google(line: str) -> bool:
    return line.strip() == "WARNING: Logging before InitGoogle() is written to STDERR"


def _is_slice_builder(line: str) -> bool:
    return (
        "Could not set metric server port" in line
        and "Could not find SliceBuilder port 8471" in line
        and 'tpu_process_addresses`="local"' in line
    )


def _is_keras_build(line: str) -> bool:
    return (
        "keras/src/layers/layer.py" in line
        and "UserWarning: `build()` was called on layer 'gemma3_causal_lm_1'" in line
    )


def _is_cuda_303(line: str) -> bool:
    return (
        "xla/stream_executor/cuda/cuda_platform.cc" in line
        and "failed call to cuInit" in line
        and "UNKNOWN ERROR (303)" in line
    )


def sanitize_runtime_log_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Collapse only exact known-success noise and preserve every unknown line."""
    lines = text.splitlines()
    out: list[str] = []
    summary = {name: 0 for name in CATEGORIES}
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_thp(line):
            summary["transparent_hugepages"] += 1
            out.append(ADVISORIES["transparent_hugepages"])
            i += 1
            if i < len(lines) and lines[i].strip() == "warnings.warn(":
                i += 1
            continue
        if _is_init_google(line) and i + 1 < len(lines) and _is_slice_builder(lines[i + 1]):
            summary["slice_builder_8471"] += 1
            out.append(ADVISORIES["slice_builder_8471"])
            i += 2
            if i < len(lines) and lines[i].strip() == "=== Source Location Trace: ===":
                i += 1
                if i < len(lines) and "runtime/common_lib.cc:" in lines[i]:
                    i += 1
            continue
        if _is_keras_build(line):
            summary["keras_build_warning"] += 1
            out.append(ADVISORIES["keras_build_warning"])
            i += 1
            if i < len(lines) and lines[i].strip() == "warnings.warn(":
                i += 1
            continue
        if _is_cuda_303(line):
            summary["cuda_probe_303"] += 1
            out.append(ADVISORIES["cuda_probe_303"])
            i += 1
            continue
        out.append(line)
        i += 1
    cleaned = "\n".join(out)
    if text.endswith("\n") or cleaned:
        cleaned += "\n"
    return cleaned, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a review-facing runtime log while preserving unknown stderr"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    raw = args.input.read_text(encoding="utf-8", errors="replace")
    cleaned, summary = sanitize_runtime_log_text(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cleaned, encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
