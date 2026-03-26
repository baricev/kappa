"""Sampling and log-probability helpers (Simply-style top-k / top-p, no extra deps)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import categorical


def neg_inf(dtype: jnp.dtype) -> Array:
    """Finite sentinel for masked logits (stable on TPU/MPS/CPU in float32)."""
    finfo = jnp.finfo(jnp.dtype(dtype))
    return jnp.array(finfo.min / 4, dtype=dtype)


def top_k_mask(logits: Array, top_k: int) -> Array:
    """Boolean mask (same shape as logits) True for tokens in top-k by logit."""
    if top_k <= 0:
        return jnp.ones_like(logits, dtype=bool)
    inner = logits.shape[-1]
    k = min(top_k, inner)
    _, indices = jax.lax.top_k(logits, k)
    mask = jnp.zeros(logits.shape, dtype=bool)
    for i in range(k):
        mask = mask | jax.nn.one_hot(indices[..., i], inner).astype(jnp.bool_)
    return mask


def top_p_mask(logits: Array, top_p: float) -> Array:
    """Nucleus mask in original vocab order (Simply ``sampling_lib.top_p_mask``)."""
    if top_p >= 1.0:
        return jnp.ones_like(logits, dtype=bool)
    probs = jax.nn.softmax(logits, axis=-1)
    indices = jnp.argsort(logits, axis=-1, descending=True)
    sorted_probs = jnp.take_along_axis(probs, indices, axis=-1)
    cumsum = jnp.cumulative_sum(sorted_probs, axis=-1, include_initial=True)
    keep_sorted = cumsum[..., :-1] < jnp.asarray(top_p, dtype=cumsum.dtype)
    _, mask = jax.lax.sort_key_val(indices, keep_sorted, dimension=-1)
    return mask


def sampling_mask(
    logits: Array,
    *,
    top_k: int = -1,
    top_p: float = 1.0,
) -> Array:
    """Intersection of top-k and top-p masks when both apply."""
    m = jnp.ones_like(logits, dtype=bool)
    if top_k > 0:
        m = m & top_k_mask(logits, top_k)
    if top_p < 1.0:
        m = m & top_p_mask(logits, top_p)
    return m


def masked_logits(logits: Array, mask: Array) -> Array:
    lf = logits.astype(jnp.float32)
    return jnp.where(mask, lf, neg_inf(jnp.float32))


def sample_from_logits(
    key: Array,
    logits: Array,
    *,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> tuple[Array, Array]:
    """Sample one token per row from final vocab dimension.

    ``logits`` shape ``(..., V)`` (e.g. ``(B, 1, V)`` or ``(B, V)``).

    Greedy when ``temperature == 0`` or ``top_k == 1`` (Simply convention).

    Returns:
        tokens: int32, shape ``(...[:-1])``
        log_probs: float32, same leading shape as tokens
    """
    lf = logits.astype(jnp.float32)
    if temperature == 0.0:
        tok = jnp.argmax(lf, axis=-1).astype(jnp.int32)
        # Greedy decoding still reports likelihood under the untempered model.
        lp = logprob_of_tokens(lf, tok, temperature=1.0, top_k=top_k, top_p=top_p)
        return tok, lp
    if top_k == 1:
        tok = jnp.argmax(lf, axis=-1).astype(jnp.int32)
        lp = logprob_of_tokens(lf, tok, temperature=temperature, top_k=top_k, top_p=top_p)
        return tok, lp

    t = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), jnp.float32(1e-9))
    scaled = lf / t
    mask = sampling_mask(scaled, top_k=top_k, top_p=top_p)
    mlog = masked_logits(scaled, mask)
    tok = categorical(key, mlog, axis=-1).astype(jnp.int32)
    lsm = jax.nn.log_softmax(mlog, axis=-1)
    oh = jax.nn.one_hot(tok, lsm.shape[-1], axis=-1)
    lp = jnp.sum(lsm * oh, axis=-1)
    return tok, lp


def logprob_of_tokens(
    logits: Array,
    tokens: Array,
    *,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> Array:
    """Log-prob of ``tokens`` under the same masked softmax as ``sample_from_logits``."""
    lf = logits.astype(jnp.float32)
    t = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), jnp.float32(1e-9))
    scaled = lf / t
    mask = sampling_mask(scaled, top_k=top_k, top_p=top_p)
    mlog = masked_logits(scaled, mask)
    lsm = jax.nn.log_softmax(mlog, axis=-1)
    # Avoid take_along_axis on last dim (problematic on some backends, e.g. jax-mps).
    oh = jax.nn.one_hot(tokens.astype(jnp.int32), lsm.shape[-1], axis=-1)
    return jnp.sum(lsm * oh, axis=-1)


def sequence_log_likelihood(
    logits_btv: Array,
    tokens_bt: Array,
    *,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    mask_bt: Array | None = None,
) -> Array:
    """Per-sequence sum of token log-probs; optional bool mask excludes padding."""
    lp = logprob_of_tokens(logits_btv, tokens_bt, temperature=temperature, top_k=top_k, top_p=top_p)
    if mask_bt is None:
        return jnp.sum(lp, axis=-1)
    m = mask_bt.astype(lp.dtype)
    return jnp.sum(lp * m, axis=-1)
