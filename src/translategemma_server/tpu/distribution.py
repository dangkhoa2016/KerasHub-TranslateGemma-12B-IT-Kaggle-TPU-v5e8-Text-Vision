"""Experimental Gemma3 model-parallel layout for KerasHub 0.31.0.

KerasHub 0.31.0 does not expose a stable Gemma3 layout helper for this exact
TranslateGemma configuration. Keep these rules isolated so an upstream layout
can replace them without rewriting the experiment runner.
"""
from __future__ import annotations

import math
from typing import Any


def build_layout_map(keras_module: Any, device_mesh: Any, data_axis: str, model_axis: str):
    layout_map = keras_module.distribution.LayoutMap(device_mesh)
    layout_map["token_embedding/embeddings"] = (model_axis, data_axis)
    layout_map[r"decoder_block.*attention.*query.kernel"] = (model_axis, data_axis, None)
    # K/V head counts can make 8-way head sharding awkward; keep that head dimension replicated.
    layout_map[r"decoder_block.*attention.*(key|value).kernel"] = (None, data_axis, None)
    layout_map[r"decoder_block.*attention_output.kernel"] = (model_axis, None, data_axis)
    layout_map[r"decoder_block.*ffw_gating.kernel"] = (data_axis, model_axis)
    layout_map[r"decoder_block.*ffw_gating_2.kernel"] = (data_axis, model_axis)
    layout_map[r"decoder_block.*ffw_linear.kernel"] = (model_axis, data_axis)

    # Gemma3 vision encoder. Biases/norms/position embeddings/patch conv stay replicated.
    layout_map[r"image_encoder.*multi_head_attention.*(query_proj|key_proj|value_proj).kernel"] = (
        data_axis, model_axis
    )
    layout_map[r"image_encoder.*multi_head_attention.*out_proj.kernel"] = (model_axis, data_axis)
    layout_map[r"image_encoder.*mlp_dense_1.kernel"] = (data_axis, model_axis)
    layout_map[r"image_encoder.*mlp_dense_2.kernel"] = (model_axis, data_axis)
    layout_map[r"vision_output_encoder.*vision_input_projection.kernel"] = (model_axis, data_axis)
    return layout_map


def build_distribution(
    keras_module: Any,
    jax_module: Any,
    *,
    shape: tuple[int, ...],
    axis_names: tuple[str, ...],
    data_axis: str,
    model_axis: str,
):
    devices = list(jax_module.devices("tpu"))
    required = math.prod(shape)
    if len(devices) != required:
        raise RuntimeError(f"DeviceMesh requires {required} TPU devices, found {len(devices)}")
    mesh = keras_module.distribution.DeviceMesh(
        shape=shape,
        axis_names=axis_names,
        devices=devices,
    )
    layout_map = build_layout_map(keras_module, mesh, data_axis, model_axis)
    distribution = keras_module.distribution.ModelParallel(
        layout_map=layout_map,
        batch_dim_name=data_axis,
    )
    return mesh, layout_map, distribution
