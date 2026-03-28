"""RoPE caches for Qwen3 (single ``rope_theta``; stacked format matches Gemma helpers)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import AttentionType
from kappa.gemma3.rope import RopeCache, apply_rope_cached, precompute_rope_sin_cos
from kappa.qwen3.architecture import Qwen3Config


def build_qwen3_rope_cache(cfg: Qwen3Config, *, max_seq_len: int) -> RopeCache:
    """Sin/cos stacks (local + global slots); Qwen3 uses the same theta for both."""
    sl, cl = precompute_rope_sin_cos(
        max_seq_len=max_seq_len,
        head_dim=cfg.head_dim,
        base_frequency=float(cfg.rope_theta),
        scale_factor=1.0,
    )
    return jnp.stack([sl, sl]), jnp.stack([cl, cl])


def apply_rope_qwen3(x: Array, positions: Array, rope_cache: RopeCache | None, *, cfg: Qwen3Config) -> Array:
    """Apply RoPE to ``[B, L, H, D]``; uses global slot when cache provided."""
    if rope_cache is None:
        from kappa.gemma3.rope import apply_rope

        return apply_rope(x, positions, base_frequency=float(cfg.rope_theta), scale_factor=1.0)
    sin_stack, cos_stack = rope_cache
    return apply_rope_cached(x, positions, sin_stack[int(AttentionType.GLOBAL)], cos_stack[int(AttentionType.GLOBAL)])
