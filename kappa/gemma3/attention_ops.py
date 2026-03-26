"""Dense GQA attention (prefill + decode) — MaxText-style Splash lives in ``splash_prefill``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kappa.gemma3.sampling import neg_inf


def repeat_kv_heads(x: Array, *, num_query_heads: int, num_kv_heads: int) -> Array:
    """Repeat KV heads for GQA: ``[B, L, n_kv, D] -> [B, L, n_q, D]``.

    Uses reshape+broadcast instead of ``jnp.repeat`` so XLA emits
    ``broadcast_in_dim`` (zero-copy alias) rather than a gather.
    """
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    r = num_query_heads // num_kv_heads
    if r == 1:
        return x
    b, l, nk, d = x.shape
    return jnp.broadcast_to(x[:, :, :, None, :], (b, l, nk, r, d)).reshape(b, l, nk * r, d)


def dense_gqa_attention(
    q: Array,
    k: Array,
    v: Array,
    mask_add: Array,
    *,
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Scaled dot-product attention.

    ``q``: ``[B, Lq, n_q, D]``; ``k``, ``v``: ``[B, Lk, n_q, D]`` (KV already repeated).
    ``mask_add``: broadcastable to ``[B, n_q, Lq, Lk]`` (0 keep, large negative drop).

    Query is assumed to be pre-scaled by ``query_pre_attn_scalar`` in the block layer.
    """
    # [B, n_q, Lq, Lk]
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32)
    if attn_logits_soft_cap is not None:
        cap = jnp.asarray(attn_logits_soft_cap, dtype=logits.dtype)
        logits = jnp.tanh(logits / cap) * cap
    logits = logits + mask_add
    weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def decode_attention_dense(
    q: Array,
    k_cache: Array,
    v_cache: Array,
    *,
    kv_lengths: Array,
    attn_logits_soft_cap: float | None = None,
    window_size: int | Array = 0,
) -> Array:
    """Single decode step: ``q`` ``[B, 1, n_q, D]``; caches ``[B, max_len, n_kv, D]`` (not repeated).

    Masks keys past ``kv_lengths[b]-1`` (assumes current token's K/V already written at index length-1).
    ``window_size`` restricts attention to the most recent ``window_size`` keys.  Pass ``0`` or
    ``max_cache_len`` to disable (global layers).  Accepts traced JAX values for ``jax.lax.scan``.
    """
    n_q = q.shape[-2]
    n_kv = k_cache.shape[-2]
    if n_q % n_kv != 0:
        raise ValueError("GQA repeat ratio invalid")
    k = repeat_kv_heads(k_cache, num_query_heads=n_q, num_kv_heads=n_kv)
    v = repeat_kv_heads(v_cache, num_query_heads=n_q, num_kv_heads=n_kv)
    b, lq, _, d = q.shape
    _, lk, _, _ = k.shape
    j = jnp.arange(lk, dtype=jnp.int32)[None, :]
    len_exp = kv_lengths[:, None]
    valid = j < len_exp
    # Always apply window mask; window_size=0 or >=lk means all keys are in window (no-op).
    q_pos = len_exp - 1
    valid = valid & (j >= (q_pos - jnp.asarray(window_size, dtype=jnp.int32)))
    mask_add = jnp.where(valid, 0.0, neg_inf(jnp.float32))
    mask_add = mask_add[:, None, None, :]
    return dense_gqa_attention(
        q, k, v, mask_add, attn_logits_soft_cap=attn_logits_soft_cap
    )


def prefill_attention_dense(
    q: Array,
    k: Array,
    v: Array,
    mask_add: Array,
    *,
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Prefill chunk: ``q,k,v`` same length ``L`` (square causal mask provided by caller)."""
    n_q = q.shape[-2]
    n_kv = k.shape[-2]
    k = repeat_kv_heads(k, num_query_heads=n_q, num_kv_heads=n_kv)
    v = repeat_kv_heads(v, num_query_heads=n_q, num_kv_heads=n_kv)
    return dense_gqa_attention(
        q, k, v, mask_add, attn_logits_soft_cap=attn_logits_soft_cap
    )
