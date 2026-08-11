from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import DATA_DIR, LOG_DIR, STATE_DIR
from .secrets import load_or_create_secret

REQUIRED_MODEL_FILES = (
    "config.json",
    "preprocessor.json",
    "assets/tokenizer/vocabulary.spm",
)
WEIGHTS_MONOLITHIC = "model.weights.h5"
WEIGHTS_SHARDED_INDEX = "model.weights.json"
WEIGHTS_SHARDED_GLOB = "model_*.weights.h5"


def weights_path(path: Path) -> Path:
    monolithic = path / WEIGHTS_MONOLITHIC
    if monolithic.is_file():
        return monolithic

    index = path / WEIGHTS_SHARDED_INDEX
    if index.is_file() and list(path.glob(WEIGHTS_SHARDED_GLOB)):
        return index

    raise FileNotFoundError(f"No strict Keras weights entry found under {path}")


def model_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / relative).is_file() for relative in REQUIRED_MODEL_FILES):
        return False
    try:
        weights_path(path)
    except FileNotFoundError:
        return False
    return True


def discover_model_path(base: Path) -> Path:
    base = base.expanduser()
    if model_complete(base):
        return base
    if not base.is_dir():
        raise FileNotFoundError(f"Model base does not exist: {base}")

    candidates = [path for path in base.iterdir() if model_complete(path)]
    if not candidates:
        raise FileNotFoundError(
            f"No complete TranslateGemma preset found under {base}"
        )

    def version_key(path: Path) -> tuple[int, int | str]:
        try:
            return (1, int(path.name))
        except ValueError:
            return (0, path.name)

    return sorted(candidates, key=version_key, reverse=True)[0]


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean")


def _int(name: str, default: int, minimum: int | None = None) -> int:
    value = int(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float, minimum: float | None = None) -> float:
    value = float(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip())
        for item in os.environ.get(name, default).split(",")
        if item.strip()
    )


def _strs(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in os.environ.get(name, default).split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Config:
    model_path: str
    host: str
    port: int
    worker_count: int
    expected_tpu_devices: int
    require_v5e8: bool
    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    data_axis: str
    model_axis: str
    vision_enabled: bool
    model_dtype: str
    generation_split_compile: bool
    generation_bucketing: bool
    generation_length_buckets: tuple[int, ...]
    generation_bucket_step: int
    vision_min_generation_length: int
    max_worker_restarts: int
    worker_load_timeout: float
    max_queue_size: int
    max_store_size: int
    result_ttl_seconds: float
    max_input_chars: int
    max_image_bytes: int
    max_image_pixels: int
    default_output_tokens: int
    max_output_tokens: int
    request_timeout: float
    shutdown_timeout: float
    max_request_bytes: int
    jax_compilation_cache_dir: Optional[str]
    jax_persistent_cache_min_compile_time_secs: float
    jax_persistent_cache_min_entry_size_bytes: int
    api_auth_required: bool
    api_key: str
    restart_secret: str

    @classmethod
    def from_env(cls) -> "Config":
        for directory in (DATA_DIR, LOG_DIR, STATE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        model_base = Path(
            os.environ.get("MODEL_PATH")
            or os.environ.get(
                "MODEL_BASE",
                "/kaggle/input/models/keras/translategemma/keras/translategemma_12b_it",
            )
        )
        model_path = discover_model_path(model_base)

        mesh_shape = _ints("MESH_SHAPE", "1,8")
        mesh_axis_names = _strs("MESH_AXIS_NAMES", "batch,model")
        if (
            len(mesh_shape) != len(mesh_axis_names)
            or not mesh_shape
            or any(value <= 0 for value in mesh_shape)
        ):
            raise ValueError("Invalid TPU mesh")

        expected_tpu_devices = _int("EXPECTED_TPU_DEVICES", 8, 1)
        if math.prod(mesh_shape) != expected_tpu_devices:
            raise ValueError(
                "MESH_SHAPE product must equal EXPECTED_TPU_DEVICES"
            )

        default_output_tokens = _int("DEFAULT_OUTPUT_TOKENS", 128, 1)
        max_output_tokens = _int("MAX_OUTPUT_TOKENS", 1024, 1)
        if default_output_tokens > max_output_tokens:
            raise ValueError(
                "DEFAULT_OUTPUT_TOKENS must be <= MAX_OUTPUT_TOKENS"
            )

        cache = os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            "/kaggle/working/.cache/translategemma-12b-jax",
        ).strip() or None
        api_auth_required = _bool("API_AUTH_REQUIRED", True)

        return cls(
            model_path=str(model_path),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_int("PORT", 7860, 1),
            # The single-worker TPU design intentionally ignores multi-worker tuning knobs.
            worker_count=1,
            expected_tpu_devices=expected_tpu_devices,
            require_v5e8=_bool("REQUIRE_V5E8", True),
            mesh_shape=mesh_shape,
            mesh_axis_names=mesh_axis_names,
            data_axis=os.environ.get("DATA_PARALLEL_AXIS", "batch"),
            model_axis=os.environ.get("MODEL_PARALLEL_AXIS", "model"),
            vision_enabled=_bool("VISION_ENABLED", True),
            model_dtype=os.environ.get("MODEL_DTYPE", "bfloat16"),
            generation_split_compile=_bool("GENERATION_SPLIT_COMPILE", True),
            generation_bucketing=_bool("GENERATION_BUCKETING", True),
            generation_length_buckets=tuple(
                sorted(set(_ints("GENERATION_LENGTH_BUCKETS", "256,512,1024")))
            ),
            generation_bucket_step=_int("GENERATION_BUCKET_STEP", 512, 1),
            vision_min_generation_length=_int(
                "VISION_MIN_GENERATION_LENGTH", 512, 1
            ),
            max_worker_restarts=_int("MAX_WORKER_RESTARTS", 1, 0),
            worker_load_timeout=_float("WORKER_LOAD_TIMEOUT", 1200, 1),
            max_queue_size=_int("MAX_QUEUE_SIZE", 16, 1),
            max_store_size=_int("MAX_STORE_SIZE", 500, 1),
            result_ttl_seconds=_float("RESULT_TTL_SECONDS", 3600, 1),
            max_input_chars=_int("MAX_INPUT_CHARS", 20000, 1),
            max_image_bytes=_int("MAX_IMAGE_BYTES", 5_242_880, 1024),
            max_image_pixels=_int("MAX_IMAGE_PIXELS", 20_000_000, 64),
            default_output_tokens=default_output_tokens,
            max_output_tokens=max_output_tokens,
            request_timeout=_float("REQUEST_TIMEOUT", 600, 1),
            shutdown_timeout=_float("SHUTDOWN_TIMEOUT", 300, 1),
            max_request_bytes=_int("MAX_REQUEST_BYTES", 8_388_608, 1024),
            jax_compilation_cache_dir=cache,
            jax_persistent_cache_min_compile_time_secs=_float(
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", 1.0, 0
            ),
            jax_persistent_cache_min_entry_size_bytes=_int(
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", -1, -1
            ),
            api_auth_required=api_auth_required,
            api_key=load_or_create_secret(
                "API_KEY",
                DATA_DIR / "api_key.txt",
                api_auth_required,
            ),
            restart_secret=load_or_create_secret(
                "RESTART_SECRET",
                DATA_DIR / "restart_secret.txt",
                True,
            ),
        )

    @classmethod
    def for_tests(cls) -> "Config":
        return cls(
            model_path="/tmp/model",
            host="127.0.0.1",
            port=7860,
            worker_count=1,
            expected_tpu_devices=8,
            require_v5e8=True,
            mesh_shape=(1, 8),
            mesh_axis_names=("batch", "model"),
            data_axis="batch",
            model_axis="model",
            vision_enabled=True,
            model_dtype="bfloat16",
            generation_split_compile=True,
            generation_bucketing=True,
            generation_length_buckets=(256, 512, 1024),
            generation_bucket_step=512,
            vision_min_generation_length=512,
            max_worker_restarts=1,
            worker_load_timeout=5,
            max_queue_size=4,
            max_store_size=20,
            result_ttl_seconds=60,
            max_input_chars=1000,
            max_image_bytes=524_288,
            max_image_pixels=20_000_000,
            default_output_tokens=32,
            max_output_tokens=128,
            request_timeout=0.05,
            shutdown_timeout=1,
            max_request_bytes=4096,
            jax_compilation_cache_dir=None,
            jax_persistent_cache_min_compile_time_secs=1.0,
            jax_persistent_cache_min_entry_size_bytes=-1,
            api_auth_required=True,
            api_key="test-api-key",
            restart_secret="test-restart-secret",
        )

    def worker_payload(self) -> dict[str, object]:
        fields = (
            "model_path",
            "expected_tpu_devices",
            "require_v5e8",
            "mesh_shape",
            "mesh_axis_names",
            "data_axis",
            "model_axis",
            "vision_enabled",
            "model_dtype",
            "generation_split_compile",
            "generation_bucketing",
            "generation_length_buckets",
            "generation_bucket_step",
            "vision_min_generation_length",
            "jax_compilation_cache_dir",
            "jax_persistent_cache_min_compile_time_secs",
            "jax_persistent_cache_min_entry_size_bytes",
        )
        return {field: getattr(self, field) for field in fields}
