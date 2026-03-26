"""Position ids from attention masks (Gemma convention)."""

import jax.numpy as jnp
from jax import Array


def positions_from_mask(mask: Array) -> Array:
    """Cumulative positions for valid (True) tokens; matches common Gemma-3 tokenization helpers."""
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)
