"""Full Gemma 3 dense transformer: embed, prefill, decode step, logits."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import Gemma3DenseConfig
from kappa.gemma3.block import (
    block_forward_decode,
    block_forward_prefill,
    block_forward_prefill_chunk,
    kv_from_prefill,
)
from kappa.gemma3.positions import positions_from_mask
from kappa.gemma3.kv_cache import DenseKVState, init_dense_kv, set_lengths, write_kv_range
from kappa.gemma3.norms import rms_norm
from kappa.gemma3.rope import RopeCache
from kappa.gemma3.weights import Gemma3DenseParams


class DenseInferenceState(NamedTuple):
    """Per-layer dense KV caches (``num_layers`` entries)."""

    kv: tuple[DenseKVState, ...]


def init_dense_inference_state(
    cfg: Gemma3DenseConfig,
    *,
    batch: int,
    max_len: int,
    dtype: jnp.dtype,
) -> DenseInferenceState:
    return DenseInferenceState(
        tuple(
            init_dense_kv(
                batch=batch,
                max_len=max_len,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                dtype=dtype,
            )
            for _ in range(cfg.num_layers)
        )
    )


def embed_tokens(tokens: Array, embed_table: Array) -> Array:
    """Token ids ``[B, L]`` -> embeddings ``[B, L, E]`` (scaled)."""
    t = tokens.astype(jnp.int32)
    # Avoid ``jnp.asarray`` on device arrays (no-op but can confuse constant folding); convert host once.
    et = embed_table if isinstance(embed_table, Array) else jnp.asarray(embed_table)
    x = et[t]
    e = jnp.asarray(et.shape[-1], dtype=jnp.float32)
    return x * jnp.sqrt(e).astype(x.dtype)


def logits_from_hidden(x: Array, embed_table: Array) -> Array:
    """Tied weights: ``[B, L, E]`` -> ``[B, L, V]`` logits (float32)."""
    et = embed_table if isinstance(embed_table, Array) else jnp.asarray(embed_table)
    return jnp.einsum("btd,vd->btv", x.astype(jnp.float32), et.astype(jnp.float32))


def _maybe_final_logit_softcap(logits: Array, cap: float | None) -> Array:
    if cap is None or cap <= 0:
        return logits
    return jnp.tanh(logits / cap) * cap


def gather_last_valid_logits(logits: Array, valid_mask: Array) -> Array:
    """Per-row logits at the last valid (non-padding) position: ``[B, L, V]`` -> ``[B, V]``.

    Requires at least one valid token per row (typical prompts).
    """
    b, l, _ = logits.shape
    counts = jnp.sum(valid_mask, axis=1).astype(jnp.int32)
    last_idx = jnp.maximum(counts - 1, 0)
    rows = jnp.arange(b, dtype=jnp.int32)
    return logits[rows, last_idx, :]


def forward_prefill(
    tokens: Array,
    positions: Array,
    *,
    params: Gemma3DenseParams,
    cfg: Gemma3DenseConfig,
    rope_cache: RopeCache | None,
    segment_ids: Array | None = None,
    token_valid_mask: Array | None = None,
    max_len: int | None = None,
) -> tuple[Array, Array, DenseInferenceState]:
    """Run full prefill: ``tokens`` ``[B, L]``, ``positions`` ``[B, L]``.

    Returns ``(hidden, logits, inference_state)`` with KV caches seeded for decode.
    ``max_len`` must be >= ``L`` (defaults to ``L``).
    """
    ml = max_len if max_len is not None else int(tokens.shape[1])
    x = embed_tokens(tokens, params.input_embedding_table)
    b = int(tokens.shape[0])
    valid = (
        token_valid_mask
        if token_valid_mask is not None
        else jnp.ones((b, int(tokens.shape[1])), dtype=jnp.bool_)
    )
    kv_lens = jnp.sum(valid, axis=1).astype(jnp.int32)
    kvs: list[DenseKVState] = []
    for i, block_p in enumerate(params.blocks):
        x, k, v = block_forward_prefill(
            x,
            positions,
            segment_ids,
            params=block_p,
            cfg=cfg,
            layer_idx=i,
            rope_cache=rope_cache,
            token_valid_mask=valid,
        )
        kvs.append(kv_from_prefill(k, v, max_len=ml, lengths=kv_lens))
    x = rms_norm(x, params.final_norm_scale)
    logits = logits_from_hidden(x, params.input_embedding_table)
    logits = _maybe_final_logit_softcap(logits, cfg.final_logit_softcap)
    return x, logits, DenseInferenceState(tuple(kvs))


def forward_prefill_chunk(
    tokens_chunk: Array,
    *,
    params: Gemma3DenseParams,
    cfg: Gemma3DenseConfig,
    rope_cache: RopeCache | None,
    state: DenseInferenceState,
    start_pos: int,
    true_chunk_lens: Array,
) -> tuple[Array, Array, DenseInferenceState]:
    """Chunked prefill with **dense-prefix** attention (MaxText-style).

    For each layer, keys/values are ``concat(prefix_cache, chunk)`` so queries in the chunk attend to
    the full prefix ``[0 : start_pos)`` plus causal attention within the chunk. Chunk K/V are written
    into the dense cache at ``start_pos : start_pos + Lc`` (padded tail masked to zero; lengths set to
    ``start_pos + true_chunk_lens``).

    ``start_pos`` is **uniform** across batch rows (length-bucketed compilation). ``true_chunk_lens``
    is int32 ``[B]`` with ``1 <= true_chunk_lens[b] <= Lc`` (valid tokens in this chunk).

    Caller must ensure cache rows already hold a valid prefix when ``start_pos > 0`` (lengths and
    ``[:, :start_pos, ...]`` populated).
    """
    b, lc = tokens_chunk.shape[0], int(tokens_chunk.shape[1])
    valid = jnp.arange(lc, dtype=jnp.int32)[None, :] < true_chunk_lens[:, None]
    positions = jnp.broadcast_to(
        jnp.arange(start_pos, start_pos + lc, dtype=jnp.int32),
        (b, lc),
    )
    x = embed_tokens(tokens_chunk, params.input_embedding_table)
    new_kvs: list[DenseKVState] = []
    start_vec = jnp.full((b,), start_pos, dtype=jnp.int32)
    new_lens = start_pos + true_chunk_lens.astype(jnp.int32)
    for i, block_p in enumerate(params.blocks):
        kv_i = state.kv[i]
        prefix_k = None
        prefix_v = None
        if start_pos > 0:
            prefix_k = kv_i.k[:, :start_pos, :, :]
            prefix_v = kv_i.v[:, :start_pos, :, :]
        x, k_ch, v_ch = block_forward_prefill_chunk(
            x,
            positions,
            params=block_p,
            cfg=cfg,
            layer_idx=i,
            rope_cache=rope_cache,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            start_pos=start_pos,
            token_valid_chunk=valid,
        )
        k_ch = k_ch * valid[..., None, None]
        v_ch = v_ch * valid[..., None, None]
        st = write_kv_range(kv_i, k_ch, v_ch, start=start_vec)
        st = set_lengths(st, new_lens)
        new_kvs.append(st)
    x = rms_norm(x, params.final_norm_scale)
    logits = logits_from_hidden(x, params.input_embedding_table)
    logits = _maybe_final_logit_softcap(logits, cfg.final_logit_softcap)
    return x, logits, DenseInferenceState(tuple(new_kvs))


def forward_decode_step(
    token_embedded: Array,
    positions: Array,
    state: DenseInferenceState,
    *,
    params: Gemma3DenseParams,
    cfg: Gemma3DenseConfig,
    rope_cache: RopeCache | None,
) -> tuple[Array, Array, DenseInferenceState]:
    """One decode step: ``token_embedded`` ``[B, 1, E]``, ``positions`` ``[B, 1]``.

    Returns ``(hidden, logits, new_state)`` — logits ``[B, 1, V]`` for the current step.
    """
    x = token_embedded
    new_kvs: list[DenseKVState] = []
    for i, block_p in enumerate(params.blocks):
        x, kv_i = block_forward_decode(
            x,
            positions,
            state.kv[i],
            params=block_p,
            cfg=cfg,
            layer_idx=i,
            rope_cache=rope_cache,
        )
        new_kvs.append(kv_i)
    x = rms_norm(x, params.final_norm_scale)
    logits = logits_from_hidden(x, params.input_embedding_table)
    logits = _maybe_final_logit_softcap(logits, cfg.final_logit_softcap)
    return x, logits, DenseInferenceState(tuple(new_kvs))
