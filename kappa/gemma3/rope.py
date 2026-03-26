"""RoPE: on-the-fly (DeepMind Gemma style) and optional sin/cos tables."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from typing import Protocol

from kappa.gemma3.architecture import AttentionType, Gemma3DenseConfig


class _RopeConfig(Protocol):
    head_dim: int
    local_base_frequency: int
    local_scale_factor: float
    global_base_frequency: int
    global_scale_factor: float


def apply_rope(
    x: Array,
    positions: Array,
    *,
    base_frequency: float,
    scale_factor: float = 1.0,
) -> Array:
    """Apply RoPE to ``x`` shaped (B, L, num_heads, head_dim). ``positions`` (B, L)."""
    head_dim = x.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base_frequency**fraction
    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp = sinusoid_inp / scale_factor
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(x, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return jnp.concatenate([first_part, second_part], axis=-1).astype(x.dtype)


def precompute_rope_sin_cos(
    *,
    max_seq_len: int,
    head_dim: int,
    base_frequency: float,
    scale_factor: float = 1.0,
) -> tuple[Array, Array]:
    """Tables (max_seq_len, head_dim // 2) for indexing by position."""
    half = head_dim // 2
    freq = base_frequency ** (-2 * jnp.arange(half) / head_dim)
    positions = jnp.arange(max_seq_len, dtype=jnp.float32) / scale_factor
    sinusoid = jnp.outer(positions, freq)
    return jnp.sin(sinusoid), jnp.cos(sinusoid)


def apply_rope_cached(
    x: Array,
    positions: Array,
    sin_cache: Array,
    cos_cache: Array,
) -> Array:
    """``sin_cache`` / ``cos_cache``: (max_len, head_dim//2). ``positions`` (B, L) int."""
    sin_vals = sin_cache[positions][:, :, None, :]
    cos_vals = cos_cache[positions][:, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], axis=-1).astype(
        x.dtype
    )


# Stacked format: (sin_stack[2, max_len, half], cos_stack[2, max_len, half])
# Index 0 = local, index 1 = global.  Supports traced indexing for ``jax.lax.scan``.
RopeCache = tuple[Array, Array]

_ROPE_INDEX_LOCAL = 0
_ROPE_INDEX_GLOBAL = 1


def build_rope_cache(cfg: Gemma3DenseConfig | _RopeConfig, *, max_seq_len: int) -> RopeCache:
    sl, cl = precompute_rope_sin_cos(
        max_seq_len=max_seq_len,
        head_dim=cfg.head_dim,
        base_frequency=float(cfg.local_base_frequency),
        scale_factor=cfg.local_scale_factor,
    )
    sg, cg = precompute_rope_sin_cos(
        max_seq_len=max_seq_len,
        head_dim=cfg.head_dim,
        base_frequency=float(cfg.global_base_frequency),
        scale_factor=cfg.global_scale_factor,
    )
    return jnp.stack([sl, sg]), jnp.stack([cl, cg])


def apply_rope_for_layer(
    x: Array,
    positions: Array,
    *,
    attn_type: AttentionType | Array,
    cfg: Gemma3DenseConfig | _RopeConfig,
    rope_cache: RopeCache | None,
) -> Array:
    """Apply RoPE using local or global frequencies depending on ``attn_type``.

    ``attn_type`` may be a Python ``AttentionType`` (unrolled loop) **or** a traced
    JAX int (``jax.lax.scan`` over layers).  When ``rope_cache`` is provided the
    dispatch is a single ``dynamic_index_in_dim`` — no Python branching.
    """
    if rope_cache is None:
        # On-the-fly path — requires Python-known attn_type (no traced values).
        if attn_type == AttentionType.GLOBAL:
            return apply_rope(
                x,
                positions,
                base_frequency=float(cfg.global_base_frequency),
                scale_factor=cfg.global_scale_factor,
            )
        return apply_rope(
            x,
            positions,
            base_frequency=float(cfg.local_base_frequency),
            scale_factor=cfg.local_scale_factor,
        )
    sin_stack, cos_stack = rope_cache
    # Works for both Python int/enum and traced JAX int.
    idx = jnp.asarray(attn_type == int(AttentionType.GLOBAL), dtype=jnp.int32)
    return apply_rope_cached(x, positions, sin_stack[idx], cos_stack[idx])
