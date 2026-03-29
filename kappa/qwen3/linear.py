"""Attention projections (same layout as Gemma / Simply ``EinsumLinear``)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kappa.qwen3.quant import Weight, to_compute_dtype


def project_q(x: Array, q_w: Weight) -> Array:
    """``q_w`` ``[num_heads, model_dim, head_dim]``."""
    return jnp.einsum("btd,ndh->btnh", x, to_compute_dtype(q_w, x.dtype))


def project_kv(x: Array, k_w: Weight, v_w: Weight) -> tuple[Array, Array]:
    """``k_w``, ``v_w`` each ``[num_kv, model_dim, head_dim]``."""
    k = jnp.einsum("btd,kdh->btkh", x, to_compute_dtype(k_w, x.dtype))
    v = jnp.einsum("btd,kdh->btkh", x, to_compute_dtype(v_w, x.dtype))
    return k, v


def project_attn_out(encoded: Array, o_w: Weight) -> Array:
    """``o_w`` ``[num_heads, head_dim, model_dim]``."""
    return jnp.einsum("btnh,nhd->btd", encoded, to_compute_dtype(o_w, encoded.dtype))
