"""RMSNorm: Qwen3 uses ``x * scale`` (no ``1 + scale``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def rms_norm(x: Array, scale: Array, *, epsilon: float = 1e-6, out_dtype: jnp.dtype | None = None) -> Array:
    """RMSNorm with learnable ``scale`` (not Gemma ``1+scale``)."""
    out_dtype = out_dtype or x.dtype
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    normed = xf * jax.lax.rsqrt(var + epsilon)
    return (normed * scale.astype(jnp.float32)).astype(out_dtype)
