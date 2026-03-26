"""Gemma 3 feed-forward (gated GeLU), matching checkpoint ``gating_einsum`` layout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def feed_forward(
    x: Array,
    gating_w: Array,
    linear_w: Array,
    *,
    transpose_gating_einsum: bool,
) -> Array:
    """``gating_w``: ``[2, hidden, embed]`` if ``transpose_gating_einsum`` else ``[2, embed, hidden]``.

    ``linear_w``: ``[hidden, embed]``.
    """
    if transpose_gating_einsum:
        gate = jnp.einsum("btd,nhd->btnh", x, gating_w.astype(x.dtype))
    else:
        gate = jnp.einsum("btd,ndh->btnh", x, gating_w.astype(x.dtype))
    a = jax.nn.gelu(gate[..., 0, :]) * gate[..., 1, :]
    return jnp.einsum("bth,hf->btf", a, linear_w.astype(x.dtype))
