"""Attention masks: causal, segment (packed batch), sliding window (Gemma local layers)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.sampling import neg_inf


def causal_square(seq_len: int) -> Array:
    """Bool [L, L]: query i may attend to key j iff j <= i."""
    t = jnp.arange(seq_len)[:, None]
    s = jnp.arange(seq_len)[None, :]
    return t >= s


def extended_causal_mask(lq: int, lk: int, prefix_len: Array) -> Array:
    """Bool [B, Lq, Lk] for chunked prefill.

    Keys index global positions ``0 .. lk-1``; query row ``i`` (0..lq-1) is at global
    ``prefix_len + i`` and may attend to keys ``j`` with ``j <= prefix_len + i``.
    ``prefix_len`` is int32 [B].
    """
    i = jnp.arange(lq, dtype=jnp.int32)
    j = jnp.arange(lk, dtype=jnp.int32)
    pl = prefix_len[:, None, None]
    return j[None, None, :] <= (pl + i[None, :, None])


def segment_rect_mask(segment_q: Array, segment_k: Array) -> Array:
    """Bool [B, Lq, Lk]: packed-batch mask (same segment only)."""
    return segment_q[:, :, None] == segment_k[:, None, :]


def token_pair_valid_mask(valid: Array) -> Array:
    """Bool [B, L, L]: query i and key j may interact only if both positions are valid (non-padding).

    Matches the idea of ``inputs_mask[...,None] * inputs_mask[...,None,:]`` in packed/causal setups.
    """
    m = valid.astype(jnp.bool_)
    return m[:, :, None] & m[:, None, :]


def local_sliding_extended(prefix_len: Array, lq: int, lk: int, window_size: int) -> Array:
    """Bool [B, Lq, Lk]: |global_q - global_k| <= window_size (global positions along the sequence)."""
    i = jnp.arange(lq, dtype=jnp.int32)
    j = jnp.arange(lk, dtype=jnp.int32)
    gq = prefix_len[:, None] + i[None, :]
    gk = j[None, None, :]
    return jnp.abs(gq[..., None] - gk) <= window_size


def bool_to_additive(m: Array, *, dtype: jnp.dtype = jnp.float32) -> Array:
    """Convert bool mask [B, Lq, Lk] to float bias (0 = keep, large negative = mask)."""
    return jnp.where(m, jnp.array(0.0, dtype=dtype), neg_inf(dtype))


def combine_additive(*masks: Array) -> Array:
    """Sum additive masks (e.g. causal + segment + local)."""
    out = masks[0]
    for m in masks[1:]:
        out = out + m
    return out
