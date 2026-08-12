from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

from .. import __version__

from ..core.errors import (
    QueueFullError,
    ServiceUnavailableError,
    StoreFullError,
    ValidationError,
    WorkerNotReadyError,
)
from ..core.validation import (
    parse_image_translation_binary,
    parse_image_translation_payload,
    parse_translation_payload,
)

logger = logging.getLogger("translategemma_server")
API_VERSION = __version__
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@dataclass
class Runtime:
    config: Any
    manager: Any
    server: Any = None
    restart_lock: threading.Lock = field(default_factory=threading.Lock)
    shutdown_started: threading.Event = field(default_factory=threading.Event)


def normalize_request_id(value: str | None) -> str:
    """Preserve a compact safe client request ID or generate a server ID."""
    candidate = value.strip() if isinstance(value, str) else ""
    if candidate and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return f"req-{uuid.uuid4().hex[:24]}"


def build_info_payload(health: dict) -> dict:
    """Build an authenticated developer-info response without private state."""
    runtime = health.get("runtime") or {}
    safe_runtime_keys = (
        "model",
        "backend",
        "accelerator",
        "expected_tpu_devices",
        "mesh",
    )
    safe_runtime = {
        key: runtime.get(key)
        for key in safe_runtime_keys
        if key in runtime
    }
    return {
        "service": "TranslateGemma 12B IT",
        "api_version": API_VERSION,
        "architecture": "single TPU worker sharded across 8 TPU devices",
        "state": health.get("state"),
        "worker_generation": health.get("worker_generation"),
        "runtime": safe_runtime,
        "capabilities": {
            "text": True,
            "vision": True,
            "async_jobs": True,
            "request_ids": True,
            "image_transports": ["json-base64", "multipart"],
            "cold_compile_polling": True,
        },
        "endpoints": {
            "text": ["/translate", "/translate/async"],
            "image": ["/translate/image", "/translate/image/async"],
            "result": "/result/<job_id>",
            "health": ["/health/live", "/health/ready"],
            "info": "/info",
        },
    }


def parse_restart_options(data: dict, config) -> tuple[bool, float]:
    """Validate the small restart payload independently of Flask."""
    wait_for_jobs = data.get("wait_for_jobs", True)
    if not isinstance(wait_for_jobs, bool):
        raise ValueError("Field 'wait_for_jobs' must be boolean")

    try:
        timeout = float(data.get("timeout", config.shutdown_timeout))
    except (TypeError, ValueError) as exc:
        raise ValueError("Field 'timeout' must be a number") from exc
    if timeout <= 0:
        raise ValueError("Field 'timeout' must be greater than 0")

    return wait_for_jobs, min(timeout, 900.0)


def create_app(runtime: Runtime):
    # Flask/Werkzeug imports remain out of the package import path so local
    # non-serving tools can inspect/configure the project without importing them.
    from flask import Flask, g, jsonify, request
    from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

    config = runtime.config
    manager = runtime.manager
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_request_bytes

    @app.before_request
    def begin_request():
        g.request_id = normalize_request_id(request.headers.get("X-Request-ID"))
        g.request_started = time.monotonic()
        g.job_id = None

    @app.after_request
    def complete_request(response):
        request_id = getattr(g, "request_id", normalize_request_id(None))
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = round(
            (time.monotonic() - getattr(g, "request_started", time.monotonic()))
            * 1000,
            3,
        )
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "job_id": getattr(g, "job_id", None),
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return response

    def provided_api_key() -> str:
        header_key = request.headers.get("X-API-Key", "").strip()
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return header_key

    def valid_api_key() -> bool:
        if not config.api_auth_required:
            return True
        provided = provided_api_key()
        return bool(provided and hmac.compare_digest(provided, config.api_key))

    def auth_required(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not valid_api_key():
                return jsonify({"error": "Unauthorized"}), 401
            return function(*args, **kwargs)

        return wrapped

    def json_body() -> dict:
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        return data

    def image_body() -> dict:
        if request.is_json:
            return parse_image_translation_payload(json_body(), config)
        if request.mimetype == "multipart/form-data":
            upload = request.files.get("image")
            if upload is None:
                raise ValidationError("Multipart field 'image' is required")
            binary = upload.stream.read(config.max_image_bytes + 1)
            if len(binary) > config.max_image_bytes:
                raise ValidationError("Image is too large")
            form = request.form.to_dict(flat=True)
            return parse_image_translation_binary(binary, form, config)
        raise ValidationError(
            "Content-Type must be application/json or multipart/form-data"
        )

    def service_unavailable(exc):
        health = getattr(exc, "health", None) or manager.health()
        payload = {
            "error": str(exc),
            "request_id": g.request_id,
            "state": health.get("state"),
            "ready_workers": health.get("ready_workers", 0),
            "expected_workers": 1,
            "retry_after_seconds": 30,
        }
        return jsonify(payload), 503, {"Retry-After": "30"}

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_exc):
        return jsonify(
            {"error": "Request body is too large", "request_id": g.request_id}
        ), 413

    @app.errorhandler(Exception)
    def unhandled_error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(
                {"error": exc.description, "request_id": g.request_id}
            ), exc.code
        logger.exception("Unhandled HTTP error request_id=%s", g.request_id)
        return jsonify(
            {"error": "Internal server error", "request_id": g.request_id}
        ), 500

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "TranslateGemma 12B IT",
                "api_version": API_VERSION,
                "architecture": "single TPU worker sharded across 8 TPU devices",
                "api_auth_required": config.api_auth_required,
                "image_transports": ["json-base64", "multipart/form-data"],
                "endpoints": [
                    "/health/live",
                    "/health/ready",
                    "/info",
                    "/translate",
                    "/translate/async",
                    "/translate/image",
                    "/translate/image/async",
                    "/result/<job_id>",
                    "/restart",
                ],
            }
        )

    @app.get("/health/live")
    def live():
        return jsonify({"status": "alive", "pid": os.getpid()})

    @app.get("/health")
    @app.get("/health/ready")
    def ready():
        health = manager.health()
        details = request.args.get("details", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if details and not valid_api_key():
            return jsonify({"error": "Unauthorized"}), 401

        if details:
            payload = health
        else:
            payload = {
                key: health[key]
                for key in (
                    "state",
                    "ready",
                    "ready_workers",
                    "expected_workers",
                    "accepting_jobs",
                    "jobs",
                    "runtime",
                )
            }
        return jsonify(payload), 200 if health["ready"] else 503

    @app.get("/info")
    @auth_required
    def info():
        return jsonify(build_info_payload(manager.health()))

    def submit(payload_factory: Callable[[], dict], async_mode: bool):
        try:
            payload = payload_factory()
            job = manager.submit(payload, request_id=g.request_id)
            g.job_id = job.id
        except ValidationError as exc:
            return jsonify({"error": str(exc), "request_id": g.request_id}), 400
        except QueueFullError as exc:
            return jsonify({"error": str(exc), "request_id": g.request_id}), 429
        except (
            WorkerNotReadyError,
            ServiceUnavailableError,
            StoreFullError,
        ) as exc:
            return service_unavailable(exc)

        if async_mode:
            return (
                jsonify(
                    {
                        "job_id": job.id,
                        "request_id": job.request_id,
                        "status": job.status,
                        "result_url": f"/result/{job.id}",
                    }
                ),
                202,
            )

        if not job.done.wait(timeout=config.request_timeout):
            return (
                jsonify(
                    {
                        "job_id": job.id,
                        "request_id": job.request_id,
                        "status": "processing",
                        "result_url": f"/result/{job.id}",
                    }
                ),
                202,
            )

        return (
            jsonify(job.public_dict()),
            500 if job.status == "failed" else 200,
        )

    @app.post("/translate")
    @auth_required
    def translate():
        return submit(
            lambda: parse_translation_payload(json_body(), config),
            False,
        )

    @app.post("/translate/async")
    @auth_required
    def translate_async():
        return submit(
            lambda: parse_translation_payload(json_body(), config),
            True,
        )

    @app.post("/translate/image")
    @auth_required
    def image_translate():
        return submit(image_body, False)

    @app.post("/translate/image/async")
    @auth_required
    def image_translate_async():
        return submit(image_body, True)

    @app.get("/result/<job_id>")
    @auth_required
    def result(job_id: str):
        g.job_id = job_id
        job = manager.store.get(job_id)
        if not job:
            return jsonify(
                {"error": "Job not found or result expired", "request_id": g.request_id}
            ), 404
        if job.status in {"queued", "processing"}:
            return jsonify(job.public_dict(False)), 202
        return (
            jsonify(job.public_dict()),
            500 if job.status == "failed" else 200,
        )

    @app.post("/restart")
    def restart():
        provided = request.headers.get("X-Restart-Secret", "").strip()
        if not provided or not hmac.compare_digest(
            provided,
            config.restart_secret,
        ):
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        try:
            wait_for_jobs, timeout = parse_restart_options(data, config)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not runtime.restart_lock.acquire(blocking=False):
            return jsonify({"error": "Restart already in progress"}), 409

        def do_restart() -> None:
            try:
                time.sleep(0.2)
                runtime.shutdown_started.set()
                manager.shutdown(wait_for_jobs, timeout)
                if runtime.server:
                    runtime.server.shutdown()
                    runtime.server.server_close()
                os.execv(sys.executable, [sys.executable] + sys.argv)
            finally:
                runtime.restart_lock.release()

        threading.Thread(
            target=do_restart,
            daemon=False,
            name="http-restart",
        ).start()
        return jsonify({"status": "restarting", "pid": os.getpid()}), 202

    return app
