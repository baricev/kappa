"""Dense, fixed-shape KV cache (functional state)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class DenseKVState(NamedTuple):
    """Pre-allocated K/V buffers and per-row valid lengths.

    Shapes:
        ``k``, ``v``: ``[batch, max_len, num_kv_heads, head_dim]``
        ``lengths``: int32 ``[batch]`` — number of tokens written per row (0 .. max_len).
    """

    k: Array
    v: Array
    lengths: Array


def init_dense_kv(
    *,
    batch: int,
    max_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> DenseKVState:
    z = jnp.zeros((batch, max_len, num_kv_heads, head_dim), dtype=dtype)
    return DenseKVState(k=z, v=z, lengths=jnp.zeros((batch,), dtype=jnp.int32))


def write_kv_range(
    state: DenseKVState,
    k_new: Array,
    v_new: Array,
    *,
    start: Array,
) -> DenseKVState:
    """Write ``k_new``, ``v_new`` ``[B, seq_len, n_kv, D]`` at ``start[b]`` along the length axis."""

    def _one(k_row: Array, v_row: Array, kn: Array, vn: Array, st: Array) -> tuple[Array, Array]:
        st_i = st.astype(jnp.int32)
        return (
            jax.lax.dynamic_update_slice_in_dim(k_row, kn, st_i, axis=0),
            jax.lax.dynamic_update_slice_in_dim(v_row, vn, st_i, axis=0),
        )

    k, v = jax.vmap(_one)(state.k, state.v, k_new, v_new, start)
    return DenseKVState(k=k, v=v, lengths=state.lengths)


def advance_lengths(state: DenseKVState, delta: Array) -> DenseKVState:
    """``delta``: int32 ``[B]`` — add to each row's written length (e.g. after a chunk)."""
    return DenseKVState(
        k=state.k,
        v=state.v,
        lengths=state.lengths + delta,
    )


def set_lengths(state: DenseKVState, lengths: Array) -> DenseKVState:
    """Set absolute lengths (e.g. after prefill)."""
    return DenseKVState(k=state.k, v=state.v, lengths=lengths.astype(jnp.int32))


def append_decode_token(state: DenseKVState, k_t: Array, v_t: Array) -> DenseKVState:
    """Write one token per row at ``state.lengths`` and increment lengths.

    ``k_t``, ``v_t``: ``[B, 1, num_kv_heads, head_dim]``.
    """
    st = write_kv_range(state, k_t, v_t, start=state.lengths)
    return advance_lengths(st, jnp.ones_like(state.lengths, dtype=jnp.int32))
