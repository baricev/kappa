"""High-level load: checkpoint → config + Gemma3DenseParams."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from kappa.checkpoint.orbax_flat import load_gemma3_flat_params
from kappa.gemma3.architecture import Gemma3DenseConfig, ModelSizeB, gemma3_dense_config
from kappa.gemma3.weights import Gemma3DenseParams, params_from_flat


def load_gemma3_dense_unsharded(
    checkpoint_dir: str | Path,
    *,
    model_size: ModelSizeB = 4,
    config: Gemma3DenseConfig | None = None,
    dtype: jnp.dtype | None = None,
    drop_siglip: bool = True,
) -> tuple[Gemma3DenseConfig, Gemma3DenseParams]:
    cfg = config or gemma3_dense_config(model_size=model_size)
    flat = load_gemma3_flat_params(
        checkpoint_dir, dtype=dtype, drop_siglip=drop_siglip
    )
    params = params_from_flat(flat, num_layers=cfg.num_layers)
    return cfg, params
