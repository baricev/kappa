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
    shape = (batch, max_len, num_kv_heads, head_dim)
    return DenseKVState(
        k=jnp.zeros(shape, dtype=dtype),
        v=jnp.zeros(shape, dtype=dtype),
        lengths=jnp.zeros((batch,), dtype=jnp.int32),
    )


def _write_kv_decode_token(
    state: DenseKVState,
    k_new: Array,
    v_new: Array,
    *,
    start: Array,
) -> DenseKVState:
    """Single-token write: ``k_new``, ``v_new`` are ``[B, 1, n_kv, D]``.

    One-hot mask + ``jnp.where`` only (no ``vmap(dynamic_update_slice)``, no batched ``gather``).
    The gather-based path materializes ``[B, S, …]`` from a tiny ``k_new`` each step and can exhaust
    MPS allocator limits during long ``lax.scan`` decode.
    """
    _, s, _, _ = state.k.shape
    pos = start.astype(jnp.int32)
    s_idx = jnp.arange(s, dtype=jnp.int32)[None, :]
    mask = s_idx == pos[:, None]
    mask_kv = mask[:, :, None, None]
    kn = k_new[:, 0, :, :]
    vn = v_new[:, 0, :, :]
    k_out = jnp.where(mask_kv, kn[:, None, :, :], state.k)
    v_out = jnp.where(mask_kv, vn[:, None, :, :], state.v)
    return DenseKVState(k=k_out, v=v_out, lengths=state.lengths)


def _write_kv_range_loop(
    state: DenseKVState,
    k_new: Array,
    v_new: Array,
    *,
    start: Array,
    lc: int,
) -> DenseKVState:
    """Multi-token write without scatter or batched gather: one masked merge per offset."""
    _, s, _, _ = state.k.shape

    def body(t: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        k_acc, v_acc = carry
        pos = start.astype(jnp.int32) + t
        s_idx = jnp.arange(s, dtype=jnp.int32)[None, :]
        mask = s_idx == pos[:, None]
        mask_kv = mask[:, :, None, None]
        kn = k_new[:, t, :, :]
        vn = v_new[:, t, :, :]
        k_acc = jnp.where(mask_kv, kn[:, None, :, :], k_acc).astype(k_acc.dtype)
        v_acc = jnp.where(mask_kv, vn[:, None, :, :], v_acc).astype(v_acc.dtype)
        return k_acc, v_acc

    k_out, v_out = jax.lax.fori_loop(0, lc, body, (state.k, state.v))
    return DenseKVState(k=k_out, v=v_out, lengths=state.lengths)


def write_kv_range(
    state: DenseKVState,
    k_new: Array,
    v_new: Array,
    *,
    start: Array,
) -> DenseKVState:
    """Write ``k_new``, ``v_new`` ``[B, seq_len, n_kv, D]`` at ``start[b]`` along the length axis.

    Avoids ``vmap(dynamic_update_slice)`` (batched scatter issues on MPS; see ``MPS_PORTING_NOTES.md``)
    and avoids batched ``gather`` from small ``k_new`` onto full length ``S`` on the **decode**
    path (``seq_len == 1``), which can hit Metal allocator limits during long generation.

    - ``seq_len == 1``: one-hot ``jnp.where`` (decode hot path).
    - ``seq_len > 1``: ``fori_loop`` of single-position writes (prefill chunks; no gather).
    """
    _, lc, nk, hd = k_new.shape
    if v_new.shape != k_new.shape:
        raise ValueError("k_new and v_new must have the same shape")
    _, s, nk_s, hd_s = state.k.shape
    if (nk, hd) != (nk_s, hd_s):
        raise ValueError("KV head dims must match state")

    if lc == 1:
        return _write_kv_decode_token(state, k_new, v_new, start=start)
    return _write_kv_range_loop(state, k_new, v_new, start=start, lc=lc)


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
