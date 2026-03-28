"""Dense SwiGLU FFN (gate_proj / up_proj / down_proj)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def swiglu_ffn(
    x: Array,
    gate_w: Array,
    up_w: Array,
    down_w: Array,
) -> Array:
    """``gate_w``, ``up_w``: ``[model_dim, intermediate]``; ``down_w``: ``[intermediate, model_dim]``."""
    gate = jnp.einsum("btd,df->btf", x, gate_w.astype(x.dtype))
    up = jnp.einsum("btd,df->btf", x, up_w.astype(x.dtype))
    a = jax.nn.silu(gate) * up
    return jnp.einsum("btf,fd->btd", a, down_w.astype(x.dtype))
