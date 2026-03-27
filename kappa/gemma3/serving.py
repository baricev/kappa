"""JetStream/MaxText-style **scheduler-facing** API: chunked prefill + decode on dense fixed-slot KV.

The KV layout remains contiguous per layer (see :class:`~kappa.gemma3.transformer.DenseInferenceState`);
there is no paged block table. Chunked prefill only splits **query/work**; attention is still
dense-prefix (prefix cache + causal within chunk) via :func:`forward_prefill_chunk`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import Gemma3DenseConfig
from kappa.gemma3.rope import RopeCache
from kappa.gemma3.transformer import (
    DenseInferenceState,
    embed_tokens,
    forward_decode_step,
    forward_prefill_chunk,
)
from kappa.gemma3.weights import Gemma3DenseParams


class PrefillState(NamedTuple):
    """Host/scheduler view for one chunked-prefill admission (batch dimension = decode slots)."""

    slot_ids: Array
    """``[B]`` int32 row ids (identity ``arange(B)`` when batch rows map 1:1 to slots)."""

    start_pos: int
    """Uniform offset into the dense cache: prefix already written occupies ``[0, start_pos)``."""

    chunk_lens: Array
    """``[B]`` int32 — valid token count in the current padded chunk tensor (``<=`` chunk width)."""


class DecodeState(NamedTuple):
    """Host view for an autoregressive decode step."""

    active_slot_ids: Array | None
    """Optional ``[B]`` selection of slot indices; ``None`` means all rows ``0 .. B-1`` are active."""

    seq_lens: Array
    """``[B]`` int32 — KV length **before** appending the new token (RoPE position for that token)."""

    kv_cache: DenseInferenceState
    """Per-layer dense KV; each :class:`~kappa.gemma3.kv_cache.DenseKVState.lengths` should match ``seq_lens``."""


def prefill_chunk(
    params: Gemma3DenseParams,
    tokens_chunk: Array,
    slot_ids: Array,
    start_pos: int,
    true_chunk_lens: Array,
    kv_cache: DenseInferenceState,
    *,
    cfg: Gemma3DenseConfig,
    rope_cache: RopeCache | None,
    max_seq_len: int,
) -> tuple[Array, Array, DenseInferenceState]:
    """Run one prefill chunk: write K/V at ``[start_pos, start_pos + chunk)``; dense-prefix attention.

    ``slot_ids`` is accepted for scheduler compatibility; when batch rows are already ordered as
    slots, pass ``jnp.arange(B, dtype=jnp.int32)``.

    ``max_seq_len`` is the compile-time bound; callers should bucket lengths so
    ``start_pos + true_chunk_lens[b] <= max_seq_len`` (overflow is undefined if violated).
    """
    del slot_ids  # reserved for remapped / gathered batches; identity is the common case
    _ = max_seq_len  # part of the public contract for schedulers / bucket selection
    tc = true_chunk_lens.astype(jnp.int32)
    return forward_prefill_chunk(
        tokens_chunk,
        params=params,
        cfg=cfg,
        rope_cache=rope_cache,
        state=kv_cache,
        start_pos=start_pos,
        true_chunk_lens=tc,
        last_logits_only=True,
    )


def decode_step(
    params: Gemma3DenseParams,
    prev_tokens: Array,
    active_slot_ids: Array | None,
    seq_lens: Array,
    kv_cache: DenseInferenceState,
    *,
    cfg: Gemma3DenseConfig,
    rope_cache: RopeCache | None,
) -> tuple[Array, Array, DenseInferenceState]:
    """One decode step: ``prev_tokens`` ``[B, 1]``; RoPE position ``seq_lens`` (length before write).

    ``active_slot_ids``: reserved for subset decode; ``None`` decodes all batch rows. Non-``None``
    subset routing is not implemented yet — pass ``None`` and slice inputs on the host if needed.
    """
    if active_slot_ids is not None:
        raise NotImplementedError(
            "decode_step with active_slot_ids is not implemented; pass None or slice on the host."
        )
    x = embed_tokens(prev_tokens, params.input_embedding_table)
    pos = seq_lens[:, None].astype(jnp.int32)
    return forward_decode_step(
        x,
        pos,
        kv_cache,
        params=params,
        cfg=cfg,
        rope_cache=rope_cache,
    )


__all__ = [
    "DecodeState",
    "PrefillState",
    "decode_step",
    "prefill_chunk",
]
