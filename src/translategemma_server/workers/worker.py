from __future__ import annotations

import logging
import os
import queue
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger("translategemma_server")

SUPPORTED_KAGGLE_V5E8_LABELS = {"v5e-8", "v5litepod-8"}


def configure_known_runtime_warning_filters() -> None:
    """Hide only two warning signatures proven harmless by Kaggle v5.2.

    C++/libtpu stderr is deliberately not redirected here. Unknown runtime
    warnings and errors must remain visible in the raw server stdout log.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Transparent hugepages are not enabled\..*",
        category=UserWarning,
        module=r"jax\._src\.cloud_tpu_init",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`build\(\)` was called on layer 'gemma3_causal_lm_1'.*",
        category=UserWarning,
        module=r"keras\.src\.layers\.layer",
    )


def fatal_worker_load_exit_code() -> int:
    """Return the non-zero exit code used when TPU/model initialization fails."""
    return 1


def sanitize_public_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return worker metrics safe to expose through the public job response.

    v10 reports the fully rendered prompt as a diagnostic metric. The REST server
    does not need to echo that internal prompt back to clients, so the public API
    strips it while preserving timing/shape metadata useful for debugging.
    """
    public = dict(metrics or {})
    public.pop("prompt", None)
    return public


def validate_tpu_fallback(require_v5e8: bool, environ: Mapping[str, str]) -> None:
    """Mirror the v10 preflight guard for Kaggle's metadata-less TPU fallback."""
    if not require_v5e8:
        return
    if environ.get("TG_TPU_FALLBACK_APPLIED") != "true":
        return
    accelerator = environ.get("TPU_ACCELERATOR_TYPE", "")
    if accelerator not in SUPPORTED_KAGGLE_V5E8_LABELS:
        expected = ", ".join(sorted(SUPPORTED_KAGGLE_V5E8_LABELS))
        raise RuntimeError(
            f"Kaggle TPU fallback expected one of [{expected}], "
            f"got TPU_ACCELERATOR_TYPE={accelerator!r}"
        )


def model_worker_main(
    worker_id: str,
    generation: int,
    worker_config: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
    shutdown_event: Any,
) -> None:
    """Own the entire eight-device TPU mesh and execute jobs serially."""
    os.environ["KERAS_BACKEND"] = "jax"
    # Declare the worker's intended JAX backends explicitly: TPU for inference
    # and CPU for explicit host work. A successful v5.3 Kaggle TPU-only run still
    # emitted one CUDA cuInit 303 line from the shared runtime stack, so this
    # setting is not treated as proof that every CUDA probe is prevented.
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
    configure_known_runtime_warning_filters()

    cache_dir = worker_config.get("jax_compilation_cache_dir")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(
            worker_config.get("jax_persistent_cache_min_compile_time_secs", 1.0)
        )
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(
            worker_config.get("jax_persistent_cache_min_entry_size_bytes", -1)
        )

    def emit(kind: str, **payload: Any) -> None:
        result_queue.put(
            {
                "type": kind,
                "worker_id": worker_id,
                "generation": generation,
                "pid": os.getpid(),
                **payload,
            }
        )

    emit("worker_state", state="loading")

    try:
        # JAX/Keras imports intentionally happen only in the spawned TPU worker.
        import jax
        import keras

        from ..tpu.distribution import build_distribution
        from ..tpu.engine import TranslateGemmaTPUEngine

        validate_tpu_fallback(bool(worker_config["require_v5e8"]), os.environ)

        devices = list(jax.devices("tpu"))
        expected = int(worker_config["expected_tpu_devices"])
        if len(devices) != expected:
            raise RuntimeError(
                f"Expected {expected} TPU devices, found {len(devices)}"
            )

        _mesh, _layout_map, distribution = build_distribution(
            keras,
            jax,
            shape=tuple(worker_config["mesh_shape"]),
            axis_names=tuple(worker_config["mesh_axis_names"]),
            data_axis=worker_config["data_axis"],
            model_axis=worker_config["model_axis"],
        )

        engine = TranslateGemmaTPUEngine(
            worker_config["model_path"],
            worker_config["model_dtype"],
            distribution,
            vision_enabled=worker_config["vision_enabled"],
            generation_bucketing=worker_config["generation_bucketing"],
            generation_length_buckets=tuple(
                worker_config["generation_length_buckets"]
            ),
            generation_bucket_step=worker_config["generation_bucket_step"],
            vision_min_generation_length=worker_config[
                "vision_min_generation_length"
            ],
            split_compile_generation=worker_config["generation_split_compile"],
        )
        metadata = engine.load()
        metadata.update(
            {
                "mesh_shape": list(worker_config["mesh_shape"]),
                "mesh_axis_names": list(worker_config["mesh_axis_names"]),
                "model": "translategemma_12b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
            }
        )
    except Exception as exc:
        logger.exception("TPU model load failed")
        emit("worker_load_error", error=repr(exc))
        raise SystemExit(fatal_worker_load_exit_code())

    emit("worker_ready", metadata=metadata)

    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:
            break

        job_id = task["job_id"]
        emit("job_started", job_id=job_id)
        started = time.time()

        try:
            kwargs = {
                "src": task["src"],
                "tgt": task["tgt"],
                "max_tokens": task["max_tokens"],
                "src_code": task.get("src_code"),
                "tgt_code": task.get("tgt_code"),
            }
            if task.get("image") is not None:
                result, metrics = engine.translate_image(task["image"], **kwargs)
            else:
                result, metrics = engine.translate(task["text"], **kwargs)

            emit(
                "job_completed",
                job_id=job_id,
                result=result,
                metrics=sanitize_public_metrics(metrics),
                inference_seconds=time.time() - started,
                runtime={
                    "model": "translategemma_12b_it",
                    "backend": "jax",
                    "accelerator": "TPU v5e-8",
                    "tpu_devices": metadata["device_count"],
                    "mesh": metadata["mesh_shape"],
                    "dtype": metadata["dtype"],
                    "generation_mode": metadata["generation_mode"],
                },
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            emit("job_failed", job_id=job_id, error=repr(exc))

    emit("worker_stopped", state="stopped")
