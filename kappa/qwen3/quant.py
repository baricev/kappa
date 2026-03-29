"""W8 symmetric per-tensor quantization for functional Qwen3 (PTQ-style inference).

Weights are stored as :class:`Q8Weight` (int8 values + scalar scale). Matmuls dequantize to the
activation dtype (fake quant). No NNX/Flax — plain :func:`jax.numpy.einsum`.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias, TypeGuard

import jax.numpy as jnp
from jax import Array


class Q8Weight(NamedTuple):
    """Symmetric int8 weight tensor with a single scalar scale (``dequant ≈ values * scale``)."""

    values: Array
    scale: Array


Weight: TypeAlias = Array | Q8Weight


def is_q8_weight(w: Weight) -> TypeGuard[Q8Weight]:
    return isinstance(w, Q8Weight)


def weight_param_dtype(w: Weight) -> jnp.dtype:
    """Dtype for activations / KV cache (embedding table dtype, or Q8 scale dtype)."""
    return w.scale.dtype if is_q8_weight(w) else w.dtype


def to_compute_dtype(w: Weight, dtype: jnp.dtype) -> Array:
    """Dequantize Q8 weights or cast full-precision weights for matmul/embed."""
    if is_q8_weight(w):
        return w.values.astype(dtype) * w.scale.astype(dtype)
    return w.astype(dtype)


def quantize_weight(w: Array, *, scale_dtype: jnp.dtype = jnp.bfloat16) -> Q8Weight:
    """Per-tensor max-abs symmetric quantization to int8."""
    w_f = w.astype(jnp.float32)
    amax = jnp.max(jnp.abs(w_f))
    scale = amax / jnp.float32(127.0)
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)
    q = jnp.clip(jnp.round(w_f / scale), -128.0, 127.0).astype(jnp.int8)
    return Q8Weight(values=q, scale=scale.astype(scale_dtype))


def gather_embed_tokens(tokens: Array, embed: Weight, *, compute_dtype: jnp.dtype) -> Array:
    """Index embedding rows; Q8 path dequantizes only gathered rows."""
    t = tokens.astype(jnp.int32)
    if is_q8_weight(embed):
        rows = embed.values[t]
        return rows.astype(compute_dtype) * embed.scale.astype(compute_dtype)
    return embed[t]


def moe_take_dequant(
    ffn_w: Weight,
    idx_flat: Array,
    *,
    e: int,
    num_experts_per_tok: int,
    bt: int,
    dtype: jnp.dtype,
) -> Array:
    """``jnp.take`` along expert axis 0 then dequant; shapes match :func:`_moe_gather_einsum` gather."""
    idx_c = jnp.clip(idx_flat, 0, e - 1)
    if is_q8_weight(ffn_w):
        wv = ffn_w.values
        g = jnp.take(wv, idx_c, axis=0).reshape(bt, num_experts_per_tok, *wv.shape[1:])
        return g.astype(dtype) * ffn_w.scale.astype(dtype)
    g = jnp.take(ffn_w, idx_c, axis=0).reshape(bt, num_experts_per_tok, *ffn_w.shape[1:])
    return g.astype(dtype)
