from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..core.config import weights_path
from .generation import (
    GenerationPlan,
    plan_generation,
    scalar_text,
    translation_prompt,
    vision_translation_prompt,
    VISION_PROMPT_PROFILE,
)

logger = logging.getLogger("translategemma_tpu")


def _checkpoint_is_legacy(entry: Any) -> bool:
    """True when a sharded checkpoint uses the old backbone-relative layout.

    keras_hub < 0.31 / keras < 3.14 saved sharded weight maps with paths like
    "/layers/gemma3_vision_encoder/...", while newer versions walk from the
    CausalLM root and prefix every path with "gemma3_backbone/". A monolithic
    H5 (no weight_map) is always loadable at any root.
    """
    try:
        import json

        index = Path(str(entry))
        if index.name != "model.weights.json":
            return False
        weight_map = json.loads(index.read_text())["weight_map"]
    except Exception:
        return False
    keys = list(weight_map.keys())
    if not keys:
        return False
    sample = keys[0].lstrip("/")
    return "gemma3_backbone" not in sample


class TranslateGemmaTPUEngine:
    """KerasHub Gemma3 engine with split prefill/decode compilation.

    The split path intentionally mirrors the successful v1 debugging workaround:
    one JIT for prefill, one JIT for a decode step, and a Python autoregressive
    loop. This prevents XLA from compiling prefill + decode loop as one giant graph.
    """

    def __init__(
        self,
        preset_path: str,
        dtype: str,
        distribution: Any,
        *,
        vision_enabled: bool = True,
        generation_bucketing: bool = True,
        generation_length_buckets: tuple[int, ...] = (256, 512, 1024),
        generation_bucket_step: int = 512,
        vision_min_generation_length: int = 512,
        split_compile_generation: bool = True,
        phase_callback: Any | None = None,
    ):
        self.preset_path = preset_path
        self.dtype = dtype
        self.distribution = distribution
        self.vision_enabled = vision_enabled
        self.generation_bucketing = generation_bucketing
        self.generation_length_buckets = tuple(sorted(set(generation_length_buckets)))
        self.generation_bucket_step = generation_bucket_step
        self.vision_min_generation_length = vision_min_generation_length
        self.split_compile_generation = split_compile_generation
        self.phase_callback = phase_callback
        self._active_workload = "generation"
        self.model = None
        self.preprocessor = None
        self._prefill_jit = None
        self._decode_jit = None
        self._generation_state = None
        self._prefill_compiled_shapes: set[tuple] = set()
        self._decode_compiled_shapes: set[tuple] = set()

    def load(self) -> dict[str, Any]:
        import jax
        import keras
        import keras_hub

        start = time.perf_counter()
        keras.mixed_precision.set_global_policy(self.dtype)
        path = Path(self.preset_path)
        entry = weights_path(path)

        logger.info("Building Gemma3 CausalLM inside Keras ModelParallel distribution scope")
        # Build architecture/preprocessor first so the distribution can assign layouts,
        # then load the local checkpoint strictly (no skip_mismatch).
        with self.distribution.scope():
            self.model = keras_hub.models.Gemma3CausalLM.from_preset(
                str(path),
                load_weights=False,
            )
            if _checkpoint_is_legacy(entry):
                # Legacy presets (saved by keras_hub < 0.31 / keras < 3.14) store the
                # sharded weight_map with backbone-relative paths (e.g.
                # "/layers/gemma3_vision_encoder/...") while keras_hub >= 0.31 walks from
                # the CausalLM root, which prefixes every path with "gemma3_backbone/".
                # Load at the backbone root instead so the paths match exactly.
                logger.info("Detected legacy sharded layout; loading weights at backbone level")
                self.model.backbone.load_weights(str(entry), skip_mismatch=False)
            else:
                self.model.load_weights(str(entry), skip_mismatch=False)
        self.model.compile(sampler="greedy")
        self.preprocessor = self.model.preprocessor

        elapsed = time.perf_counter() - start
        metadata = {
            "load_seconds": round(elapsed, 3),
            "dtype": self.dtype,
            "vision_enabled": self.vision_enabled,
            "keras_version": getattr(keras, "__version__", "unknown"),
            "keras_hub_version": getattr(keras_hub, "__version__", "unknown"),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "device_count": len(jax.devices("tpu")),
            "devices": [str(device) for device in jax.devices("tpu")],
            "weights_entry": str(entry),
            "strict_weight_loading": True,
            "model_weight_count": len(self.model.weights),
            "model_trainable_weight_count": len(self.model.trainable_weights),
            "generation_mode": "split_compile" if self.split_compile_generation else "keras_generate",
            "prompt_profile": VISION_PROMPT_PROFILE,
            "decode_semantics": "keras_hub_gemma3_generate_step_v0.31_cache_update_mask",
        }
        logger.info("Model loaded: %s", metadata)
        return metadata

    def _phase(self, name: str) -> None:
        if self.phase_callback is not None:
            try:
                self.phase_callback(name)
            except Exception:
                logger.warning("phase callback failed for %s", name, exc_info=True)

    def _build_generate_paths(self) -> None:
        if self._prefill_jit is not None and self._decode_jit is not None:
            return
        import jax
        import keras

        model = self.model
        if model is None:
            raise RuntimeError("Model is not loaded")

        def _scope(trainable, non_trainable):
            mapping = list(zip(model.trainable_variables, trainable))
            mapping.extend(zip(model.non_trainable_variables, non_trainable))
            return keras.StatelessScope(state_mapping=mapping)

        @jax.jit
        def prefill(
            token_ids,
            padding_mask,
            images,
            vision_mask,
            vision_indices,
            trainable,
            non_trainable,
        ):
            with _scope(trainable, non_trainable):
                image_embeddings = None
                if images is not None:
                    image_embeddings = model.backbone.vision_encoder(images)
                _, cache = model._build_cache(
                    token_ids,
                    image_embeddings,
                    vision_mask,
                    padding_mask,
                    vision_indices,
                )
            return cache

        @jax.jit
        def decode_step(token_ids, cache, index, cache_update_mask, trainable, non_trainable):
            # Mirror Gemma3CausalLM.generate_step(): during decode the native
            # path does not pass padding_mask/vision_mask again. It only uses a
            # one-token cache_update_mask sliced from the *original* prompt mask
            # so prefilled prompt cache entries are preserved while generated
            # token entries are written on later steps.
            with _scope(trainable, non_trainable):
                logits, _, cache = model.call_with_cache(
                    token_ids=token_ids,
                    cache=cache,
                    cache_update_index=index,
                    cache_update_mask=cache_update_mask,
                )
            return logits, cache

        self._prefill_jit = prefill
        self._decode_jit = decode_step
        self._generation_state = None
        logger.info("Split-compile generation paths built (prefill + decode jits)")

    def _generation_variable_state(self):
        if self._generation_state is None:
            self._generation_state = (
                [variable.value for variable in self.model.trainable_variables],
                [variable.value for variable in self.model.non_trainable_variables],
            )
        return self._generation_state

    def _stop_token_ids(self) -> tuple[int, ...]:
        tokenizer = self.model.preprocessor.tokenizer
        values = []
        for name in ("end_token_id", "end_token2_id", "end_token_id_2"):
            value = getattr(tokenizer, name, None)
            if value is not None:
                try:
                    values.append(int(value))
                except (TypeError, ValueError):
                    pass
        token_to_id = getattr(tokenizer, "token_to_id", None)
        unk_id = getattr(tokenizer, "unk_token_id", getattr(tokenizer, "_unk_token_id", None))
        if token_to_id is not None:
            end_of_turn = token_to_id("<end_of_turn>")
            if end_of_turn is not None and (unk_id is None or int(end_of_turn) != int(unk_id)):
                values.append(int(end_of_turn))
        return tuple(dict.fromkeys(values))

    def _prompt_token_count(self, inputs: Any, *, probe_length: int) -> int:
        import keras

        prepared = self.model.preprocessor.generate_preprocess(inputs, sequence_length=probe_length)
        mask = prepared["padding_mask"]
        return int(keras.ops.sum(mask))

    def plan(self, inputs: Any, max_new_tokens: int, *, minimum_length: int = 0) -> GenerationPlan:
        probe_length = max(max(self.generation_length_buckets), minimum_length, 1024)
        prompt_tokens = self._prompt_token_count(inputs, probe_length=probe_length)
        return plan_generation(
            prompt_tokens,
            max_new_tokens,
            buckets=self.generation_length_buckets,
            bucket_step=self.generation_bucket_step,
            bucketing=self.generation_bucketing,
            minimum_length=minimum_length,
        )

    @staticmethod
    def _block(value: Any) -> None:
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()

    def _generate_manual(self, inputs: Any, plan: GenerationPlan, stop_token_ids: Any = "auto") -> tuple[str, dict]:
        """Manual autoregressive loop over the sharded model.

        It mirrors KerasHub generation semantics closely enough for the validated
        single-example flow while avoiding a fused XLA while-loop compile.
        """
        import keras

        model = self.model
        self._build_generate_paths()
        preprocess_started = time.perf_counter()
        pre = model.preprocessor.generate_preprocess(inputs, sequence_length=plan.max_length)
        preprocessing_seconds = time.perf_counter() - preprocess_started
        token_ids = keras.ops.expand_dims(pre["token_ids"], axis=0)
        padding_mask = keras.ops.expand_dims(pre["padding_mask"], axis=0)
        images = pre.get("images", None)
        if images is not None:
            images = keras.ops.expand_dims(images, axis=0)
        vision_mask = pre.get("vision_mask", None)
        if vision_mask is not None:
            vision_mask = keras.ops.expand_dims(vision_mask, axis=0)
        vision_indices = pre.get("vision_indices", None)
        if vision_indices is not None:
            vision_indices = keras.ops.expand_dims(vision_indices, axis=0)
        if images is None or images.shape[1] == 0:
            images = None
            vision_mask = None
            vision_indices = None

        preprocess_diagnostics = {
            "token_ids_shape": [int(v) for v in token_ids.shape],
            "padding_mask_shape": [int(v) for v in padding_mask.shape],
            "padding_mask_true_count": int(keras.ops.sum(keras.ops.cast(padding_mask, "int32"))),
            "images_shape": [int(v) for v in images.shape] if images is not None else None,
            "vision_mask_shape": [int(v) for v in vision_mask.shape] if vision_mask is not None else None,
            "vision_mask_true_count": (
                int(keras.ops.sum(keras.ops.cast(vision_mask, "int32")))
                if vision_mask is not None else None
            ),
            "vision_indices_shape": [int(v) for v in vision_indices.shape] if vision_indices is not None else None,
            "num_images": int(images.shape[1]) if images is not None else 0,
        }

        state = self._generation_variable_state()
        prefill_key = (
            int(plan.max_length),
            images is not None,
            tuple(getattr(images, "shape", ())) if images is not None else (),
        )
        prefill_was_compiled = prefill_key in self._prefill_compiled_shapes
        self._phase(f"{self._active_workload}_prefill_{'run' if prefill_was_compiled else 'compile'}")
        started = time.perf_counter()
        cache = self._prefill_jit(
            token_ids, padding_mask, images, vision_mask, vision_indices, *state
        )
        self._block(cache)
        prefill_seconds = time.perf_counter() - started
        self._prefill_compiled_shapes.add(prefill_key)

        if stop_token_ids == "auto":
            stop_ids = self._stop_token_ids()
        elif stop_token_ids is None:
            stop_ids = ()
        else:
            stop_ids = tuple(int(v) for v in stop_token_ids)

        # Preserve the preprocessor's original prompt mask exactly. Native
        # Gemma3 generate_step closes over this immutable mask and slices
        # ~padding_mask at index-1 on every decode iteration.
        original_padding_mask = padding_mask
        index = int(keras.ops.sum(keras.ops.cast(original_padding_mask, "int32"), axis=-1)[0])
        prompt_index = index
        generated: list[int] = []
        first_token_seconds = None
        decode_total_seconds = 0.0
        decode_first_call_seconds = 0.0
        decode_key = (1, plan.max_length)
        decode_was_compiled = decode_key in self._decode_compiled_shapes

        for step in range(min(plan.max_new_tokens, plan.max_length - index)):
            if index >= plan.max_length:
                break
            cache_update_index = keras.ops.array(index - 1, dtype="int32")
            token = keras.ops.slice(token_ids, [0, index - 1], [1, 1])
            cache_update_mask = keras.ops.slice(
                keras.ops.logical_not(original_padding_mask), [0, index - 1], [1, 1]
            )
            if step == 0:
                self._phase(f"{self._active_workload}_decode_{'run' if decode_was_compiled else 'compile'}")
            else:
                self._phase(f"{self._active_workload}_decode_run")
            step_started = time.perf_counter()
            logits, cache = self._decode_jit(
                token, cache, cache_update_index, cache_update_mask, *state
            )
            self._block(logits)
            step_seconds = time.perf_counter() - step_started
            decode_total_seconds += step_seconds
            if step == 0:
                decode_first_call_seconds = step_seconds
                first_token_seconds = prefill_seconds + step_seconds
                self._decode_compiled_shapes.add(decode_key)

            next_id = int(keras.ops.argmax(logits[0, 0], axis=-1))
            generated.append(next_id)

            replacement = keras.ops.array([[next_id]], dtype=token_ids.dtype)
            token_ids = keras.ops.slice_update(token_ids, [0, index], replacement)
            index += 1
            if next_id in stop_ids:
                break

        stop_token_id = generated[-1] if generated and generated[-1] in stop_ids else None
        raw_generated_tokens = len(generated)
        if stop_token_id is not None:
            generated = generated[:-1]
        generated_tensor = keras.ops.array(generated, dtype="int32")
        detokenized = model.preprocessor.tokenizer.detokenize(generated_tensor)
        text = scalar_text(detokenized)
        compile_prefill = round(prefill_seconds, 6) if not prefill_was_compiled else 0.0
        compile_decode = round(decode_first_call_seconds, 6) if not decode_was_compiled else 0.0
        metrics = {
            "prompt_tokens": prompt_index,
            "generated_tokens": len(generated),
            "raw_generated_tokens_including_stop": raw_generated_tokens,
            "max_new_tokens_requested": plan.max_new_tokens,
            "stop_token_ids": list(stop_ids),
            "stop_token_id": stop_token_id,
            "stopped_on_stop_token": stop_token_id is not None,
            "preprocess_diagnostics": preprocess_diagnostics,
            "preprocessing_seconds": round(preprocessing_seconds, 6),
            "image_preprocessing_seconds": round(preprocessing_seconds, 6) if images is not None else None,
            "prefill_seconds": round(prefill_seconds, 6),
            "decode_total_seconds": round(decode_total_seconds, 6),
            "decode_first_call_seconds": round(decode_first_call_seconds, 6),
            "time_to_first_token_seconds": round(first_token_seconds, 6) if first_token_seconds is not None else None,
            "tokens_per_second": round(len(generated) / decode_total_seconds, 6) if decode_total_seconds > 0 else None,
            # These compile fields are first-call compile-inclusive measurements.
            # JAX does not expose pure compiler time through this path.
            "compile_prefill_seconds": compile_prefill,
            "compile_decode_seconds": compile_decode,
            "compile_measurement": "first_call_compile_inclusive",
            "prefill_runtime_seconds": round(prefill_seconds, 6) if prefill_was_compiled else None,
            "decode_runtime_seconds": round(decode_total_seconds, 6) if decode_was_compiled else (
                round(max(0.0, decode_total_seconds - decode_first_call_seconds), 6)
            ),
            "vision_encoder_seconds": None,
            "vision_encoder_timing_note": (
                "Vision encoder is fused into the prefill JIT to preserve the validated split architecture."
                if images is not None else None
            ),
            "generation_mode": "split_compile",
            "decode_semantics": "keras_hub_gemma3_generate_step_v0.31_cache_update_mask",
        }
        return text, metrics

    def _generate_native(self, inputs: Any, plan: GenerationPlan) -> tuple[str, dict]:
        started = time.perf_counter()
        value = self.model.generate(
            inputs,
            max_length=plan.max_length,
            strip_prompt=True,
        )
        self._block(value)
        elapsed = time.perf_counter() - started
        return scalar_text(value), {
            "total_seconds": round(elapsed, 6),
            "generation_mode": "keras_generate",
            "warning": "Native fused generation can exceed Kaggle host RAM for the 12B v5e-8 profile.",
        }

    def _generate(self, inputs: Any, plan: GenerationPlan) -> tuple[str, dict]:
        if self.split_compile_generation:
            return self._generate_manual(inputs, plan, stop_token_ids="auto")
        return self._generate_native(inputs, plan)

    def translate(
        self,
        text: str,
        src: str,
        tgt: str,
        max_tokens: int,
        *,
        src_code: str | None = None,
        tgt_code: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._active_workload = "text"
        prompt = translation_prompt(
            text, src, tgt, source_lang_code=src_code, target_lang_code=tgt_code
        )
        plan = self.plan(prompt, max_tokens)
        started = time.perf_counter()
        output, metrics = self._generate(prompt, plan)
        metrics["total_seconds"] = round(time.perf_counter() - started, 6)
        metrics["plan"] = plan.as_dict()
        metrics["prompt"] = prompt
        return scalar_text(output), metrics

    def translate_with_prompt(
        self,
        prompt: str,
        max_tokens: int,
        *,
        workload: str = "text_custom",
    ) -> tuple[str, dict[str, Any]]:
        """Generate from an already-built text prompt without changing JIT internals."""
        self._active_workload = workload
        plan = self.plan(prompt, max_tokens)
        started = time.perf_counter()
        output, metrics = self._generate(prompt, plan)
        metrics["total_seconds"] = round(time.perf_counter() - started, 6)
        metrics["plan"] = plan.as_dict()
        metrics["prompt"] = prompt
        return scalar_text(output), metrics


    def translate_image(
        self,
        image: Any,
        src: str,
        tgt: str,
        max_tokens: int,
        *,
        src_code: str | None = None,
        tgt_code: str | None = None,
        prompt: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._active_workload = "vision"
        if not self.vision_enabled:
            raise RuntimeError("Vision is disabled")
        if prompt is None:
            prompt = vision_translation_prompt(
                src, tgt, source_lang_code=src_code, target_lang_code=tgt_code
            )
        return self.generate_vision_with_prompt(
            image,
            prompt,
            max_tokens,
            workload="vision",
        )

    def generate_vision_with_prompt(
        self,
        image: Any,
        prompt: str,
        max_tokens: int,
        *,
        workload: str = "vision",
    ) -> tuple[str, dict[str, Any]]:
        """Generate from an arbitrary image prompt without changing TPU JIT internals."""
        self._active_workload = workload
        if not self.vision_enabled:
            raise RuntimeError("Vision is disabled")
        inputs = {"prompts": prompt, "images": image}
        plan = self.plan(inputs, max_tokens, minimum_length=self.vision_min_generation_length)
        started = time.perf_counter()
        output, metrics = self._generate(inputs, plan)
        metrics["total_seconds"] = round(time.perf_counter() - started, 6)
        metrics["plan"] = plan.as_dict()
        metrics["prompt"] = prompt
        return scalar_text(output), metrics
