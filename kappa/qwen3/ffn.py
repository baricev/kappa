"""Dense SwiGLU FFN (gate_proj / up_proj / down_proj)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kappa.qwen3.quant import Weight, to_compute_dtype


def swiglu_ffn(
    x: Array,
    gate_w: Weight,
    up_w: Weight,
    down_w: Weight,
) -> Array:
    """``gate_w``, ``up_w``: ``[model_dim, intermediate]``; ``down_w``: ``[intermediate, model_dim]``."""
    gate = jnp.einsum("btd,df->btf", x, to_compute_dtype(gate_w, x.dtype))
    up = jnp.einsum("btd,df->btf", x, to_compute_dtype(up_w, x.dtype))
    a = jax.nn.silu(gate) * up
    return jnp.einsum("btf,fd->btd", a, to_compute_dtype(down_w, x.dtype))
