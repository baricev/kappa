"""RMSNorm (Gemma / Flax-compatible scaling: ``(1 + scale)``)."""

import jax
import jax.numpy as jnp
from jax import Array


def rms_norm(x: Array, scale: Array, *, epsilon: float = 1e-6, out_dtype: jnp.dtype | None = None) -> Array:
    out_dtype = out_dtype or x.dtype
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    normed = xf * jax.lax.rsqrt(var + epsilon) * (1 + scale.astype(jnp.float32))
    return normed.astype(out_dtype)
