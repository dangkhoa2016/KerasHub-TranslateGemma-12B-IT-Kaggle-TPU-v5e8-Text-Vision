#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def readiness_decision(
    status: int,
    data: dict[str, Any],
    *,
    expected_devices: int = 8,
    expected_mesh: tuple[int, ...] = (1, 8),
) -> str:
    state = data.get("state")
    if status == 200 and data.get("ready") is True:
        workers = data.get("workers") or []
        if len(workers) != 1:
            return "continue"
        worker = workers[0]
        metadata = worker.get("metadata") or {}
        if (
            worker.get("state") in {"ready", "busy"}
            and metadata.get("device_count") == expected_devices
            and metadata.get("mesh_shape") == list(expected_mesh)
        ):
            return "ready"
        return "continue"
    if state in {"loading", "restarting"}:
        return "continue"
    if state == "unavailable":
        return "failed"
    if status in {401, 403}:
        return "failed"
    return "continue"


def request_json(url: str, api_key: str, timeout: float) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw or str(exc.reason)}
        return exc.code, data


def summary(data: dict[str, Any]) -> str:
    workers = data.get("workers") or []
    pieces = [
        f"state={data.get('state')}",
        f"ready={data.get('ready_workers', 0)}/{data.get('expected_workers', 1)}",
        f"generation={data.get('worker_generation', '?')}",
    ]
    if workers:
        worker = workers[0]
        pieces.append(f"worker={worker.get('state')}")
        if worker.get("pid"):
            pieces.append(f"pid={worker.get('pid')}")
        if worker.get("error"):
            pieces.append(f"error={worker.get('error')}")
        if worker.get("exit_code") is not None:
            pieces.append(f"exit={worker.get('exit_code')}")
    return " ".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=10)
    parser.add_argument("--expected-devices", type=int, default=8)
    parser.add_argument("--expected-mesh", default="1,8")
    parser.add_argument("--heartbeat", type=float, default=30)
    args = parser.parse_args()

    key_path = Path(args.api_key_file)
    expected_mesh = tuple(int(part) for part in args.expected_mesh.split(","))
    deadline = time.monotonic() + args.timeout
    started = time.monotonic()
    last_summary = None
    last_print = 0.0

    while time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        if not key_path.is_file() or not key_path.read_text(encoding="utf-8").strip():
            if elapsed - last_print >= args.heartbeat or last_print == 0.0:
                print(f"[{elapsed:06.1f}s] waiting for API key", flush=True)
                last_print = elapsed
            time.sleep(args.interval)
            continue

        api_key = key_path.read_text(encoding="utf-8").strip()
        try:
            status, data = request_json(
                args.base_url.rstrip("/") + "/health/ready?details=1",
                api_key,
                args.request_timeout,
            )
        except urllib.error.URLError as exc:
            current = f"connection-error={exc.reason!r}"
            if current != last_summary or elapsed - last_print >= args.heartbeat:
                print(f"[{elapsed:06.1f}s] {current}", flush=True)
                last_summary = current
                last_print = elapsed
            time.sleep(args.interval)
            continue

        current = f"HTTP={status} {summary(data)}"
        if current != last_summary or elapsed - last_print >= args.heartbeat:
            print(f"[{elapsed:06.1f}s] {current}", flush=True)
            last_summary = current
            last_print = elapsed

        decision = readiness_decision(
            status,
            data,
            expected_devices=args.expected_devices,
            expected_mesh=expected_mesh,
        )
        if decision == "ready":
            print("PASS: TPU worker ready on expected devices/mesh", flush=True)
            return 0
        if decision == "failed":
            print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
            return 2
        time.sleep(args.interval)

    print(f"ERROR: readiness timeout after {args.timeout:g}s", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
