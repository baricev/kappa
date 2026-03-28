"""Load Qwen3 params from Orbax (Simply ``Qwen2Format`` layout)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from kappa.checkpoint.qwen_hf_convert import hf_flat_to_qwen3_params, is_huggingface_flat
from kappa.checkpoint.qwen_flat import load_qwen3_flat_params
from kappa.qwen3.architecture import ModelPreset, Qwen3Config, qwen3_config_for_preset
from kappa.qwen3.weights import Qwen3Params, params_from_flat


def load_qwen3_unsharded(
    checkpoint_dir: str | Path,
    *,
    preset: ModelPreset,
    dtype: jnp.dtype | None = None,
    config: Qwen3Config | None = None,
) -> tuple[Qwen3Config, Qwen3Params]:
    cfg = config or qwen3_config_for_preset(preset)
    flat = load_qwen3_flat_params(checkpoint_dir, dtype=dtype)
    if is_huggingface_flat(flat):
        params = hf_flat_to_qwen3_params(flat, cfg, preset=preset)
    else:
        params = params_from_flat(flat, num_layers=cfg.num_layers, use_moe=cfg.use_moe)
    if not cfg.use_tied_embedding and params.lm_head is None:
        raise ValueError("untied model requires output_layer.w in checkpoint")
    return cfg, params
