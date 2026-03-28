"""Attention projections (same layout as Gemma / Simply ``EinsumLinear``)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def project_q(x: Array, q_w: Array) -> Array:
    """``q_w`` ``[num_heads, model_dim, head_dim]``."""
    return jnp.einsum("btd,ndh->btnh", x, q_w.astype(x.dtype))


def project_kv(x: Array, k_w: Array, v_w: Array) -> tuple[Array, Array]:
    """``k_w``, ``v_w`` each ``[num_kv, model_dim, head_dim]``."""
    k = jnp.einsum("btd,kdh->btkh", x, k_w.astype(x.dtype))
    v = jnp.einsum("btd,kdh->btkh", x, v_w.astype(x.dtype))
    return k, v


def project_attn_out(encoded: Array, o_w: Array) -> Array:
    """``o_w`` ``[num_heads, head_dim, model_dim]``."""
    return jnp.einsum("btnh,nhd->btd", encoded, o_w.astype(encoded.dtype))
