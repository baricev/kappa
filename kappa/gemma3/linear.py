"""Gemma-style Einsum projections (GQA): Q, KV split, output projection."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def project_q(x: Array, q_w: Array) -> Array:
    """``x`` [B, T, D], ``q_w`` [num_heads, D, head_dim] -> [B, T, num_heads, head_dim]."""
    return jnp.einsum("btd,ndh->btnh", x, q_w.astype(x.dtype))


def project_kv(x: Array, kv_w: Array) -> tuple[Array, Array]:
    """``kv_w`` [2, num_kv, D, head_dim] -> k, v each [B, T, num_kv, head_dim]."""
    k = jnp.einsum("btd,kdh->btkh", x, kv_w[0].astype(x.dtype))
    v = jnp.einsum("btd,kdh->btkh", x, kv_w[1].astype(x.dtype))
    return k, v


def project_attn_out(encoded: Array, o_w: Array) -> Array:
    """``encoded`` [B, T, H, head_dim], ``o_w`` [H, head_dim, D] -> [B, T, D]."""
    return jnp.einsum("btnh,nhd->btd", encoded, o_w.astype(encoded.dtype))
