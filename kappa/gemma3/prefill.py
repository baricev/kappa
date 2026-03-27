"""Chunked prefill helpers (dense GQA + masks). Splash autoselect: ``splash_prefill``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import AttentionType
from kappa.gemma3.attention_ops import prefill_attention_dense
from kappa.gemma3.masks import (
    bool_to_additive,
    causal_square,
    extended_causal_mask,
    local_sliding_extended,
    segment_rect_mask,
    token_pair_valid_mask,
)
from kappa.gemma3.splash_prefill import prefill_chunk_autoselect, splash_square_q_len_ok


def mask_prefill_chunk_with_prefix(
    *,
    lq: int,
    lk: int,
    prefix_len: Array,
    attn_type: AttentionType | Array,
    window_size: int,
    segment_q: Array | None,
    segment_k: Array | None,
    token_valid: Array | None = None,
) -> Array:
    """Float additive mask ``[B, 1, Lq, Lk]`` for ``prefill_attention_dense``.

    ``attn_type`` may be a Python ``AttentionType`` or a traced JAX int.
    Sliding window is always computed; global layers use ``lk`` as effective
    window (all-True mask), avoiding a Python branch on ``attn_type``.
    """
    m = extended_causal_mask(lq, lk, prefix_len)
    if segment_q is not None and segment_k is not None:
        m = m & segment_rect_mask(segment_q, segment_k)
    # Effective window: actual window_size for local layers, lk for global (no-op).
    effective_ws = jnp.where(
        jnp.asarray(attn_type == int(AttentionType.LOCAL_SLIDING)),
        jnp.asarray(window_size, dtype=jnp.int32),
        jnp.asarray(lk, dtype=jnp.int32),
    )
    m = m & local_sliding_extended(prefix_len, lq, lk, effective_ws)
    if token_valid is not None:
        pair = token_pair_valid_mask(token_valid)
        m = m & pair[:, :lq, :lk]
    add = bool_to_additive(m)
    return add[:, None, :, :]


def prefill_chunk_dense(
    q: Array,
    k: Array,
    v: Array,
    mask_add: Array,
    *,
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Run dense prefill for one chunk (caller supplies mask)."""
    return prefill_attention_dense(
        q, k, v, mask_add, attn_logits_soft_cap=attn_logits_soft_cap
    )


def prefill_square_chunk_dense(
    q: Array,
    k: Array,
    v: Array,
    *,
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Square causal chunk: ``q,k,v`` each ``[B, L, n_*, D]`` (same ``L``).

    Uses :func:`~kappa.gemma3.splash_prefill.prefill_chunk_autoselect` (Splash on TPU when
    eligible; dense otherwise).
    """
    return prefill_chunk_autoselect(q, k, v, attn_logits_soft_cap=attn_logits_soft_cap)


def prefill_chunk_with_prefix_dense(
    q: Array,
    k: Array,
    v: Array,
    *,
    prefix_len: Array,
    attn_type: AttentionType | Array,
    window_size: int,
    segment_q: Array | None = None,
    segment_k: Array | None = None,
    token_valid: Array | None = None,
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Prefill when ``k,v`` span ``prefix_len + chunk`` along the sequence axis."""
    lq = q.shape[1]
    lk = k.shape[1]
    mask_add = mask_prefill_chunk_with_prefix(
        lq=lq,
        lk=lk,
        prefix_len=prefix_len,
        attn_type=attn_type,
        window_size=window_size,
        segment_q=segment_q,
        segment_k=segment_k,
        token_valid=token_valid,
    )
    dense = lambda: prefill_attention_dense(
        q, k, v, mask_add, attn_logits_soft_cap=attn_logits_soft_cap
    )
    if segment_q is not None or segment_k is not None:
        return dense()
    if lq != lk:
        return dense()
    if attn_logits_soft_cap is not None:
        return dense()
    tv_ok = jnp.asarray(True) if token_valid is None else jnp.all(token_valid)
    splash_pred = (
        jnp.all(prefix_len == 0)
        & tv_ok
        & jnp.asarray(attn_type == int(AttentionType.GLOBAL), dtype=jnp.bool_)
        & jnp.asarray(splash_square_q_len_ok(lq), dtype=jnp.bool_)
    )
    return jax.lax.cond(
        splash_pred,
        lambda: prefill_chunk_autoselect(q, k, v, attn_logits_soft_cap=None),
        dense,
    )


__all__ = [
    "mask_prefill_chunk_with_prefix",
    "prefill_chunk_autoselect",
    "prefill_chunk_dense",
    "prefill_chunk_with_prefix_dense",
    "prefill_square_chunk_dense",
]
