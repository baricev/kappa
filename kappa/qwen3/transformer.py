"""Qwen3 transformer: embed, prefill, decode."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.kv_cache import DenseKVState, init_dense_kv
from kappa.qwen3.architecture import Qwen3Config
from kappa.qwen3.block import block_forward_decode, block_forward_prefill, kv_from_prefill
from kappa.qwen3.rope import RopeCache
from kappa.qwen3.norms import rms_norm
from kappa.qwen3.weights import Qwen3Params


class Qwen3InferenceState(NamedTuple):
    kv: tuple[DenseKVState, ...]


def init_qwen3_inference_state(
    cfg: Qwen3Config,
    *,
    batch: int,
    max_len: int,
    dtype: jnp.dtype,
) -> Qwen3InferenceState:
    return Qwen3InferenceState(
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
    t = tokens.astype(jnp.int32)
    et = embed_table if isinstance(embed_table, Array) else jnp.asarray(embed_table)
    return et[t]


def logits_from_hidden(x: Array, lm_head: Array) -> Array:
    """``lm_head`` ``[vocab, model_dim]``."""
    et = lm_head if isinstance(lm_head, Array) else jnp.asarray(lm_head)
    dt = jnp.promote_types(x.dtype, et.dtype)
    return jnp.einsum("btd,vd->btv", x.astype(dt), et.astype(dt))


def logits_from_hidden_tied(x: Array, embed_table: Array) -> Array:
    dt = jnp.promote_types(x.dtype, embed_table.dtype)
    return jnp.einsum("btd,vd->btv", x.astype(dt), embed_table.astype(dt))


def logits_from_hidden_last_positions(x: Array, w: Array, last_idx: Array) -> Array:
    """``x`` ``[B, L, D]``; last valid index per row ``[B]`` -> logits ``[B, V]``."""
    b = x.shape[0]
    rows = jnp.arange(b, dtype=jnp.int32)
    h = x[rows, last_idx, :]
    dt = jnp.promote_types(h.dtype, w.dtype)
    return jnp.einsum("bd,vd->bv", h.astype(dt), w.astype(dt))


def forward_prefill(
    tokens: Array,
    positions: Array,
    *,
    params: Qwen3Params,
    cfg: Qwen3Config,
    rope_cache: RopeCache | None,
    token_valid_mask: Array | None = None,
    max_len: int | None = None,
    last_logits_only: bool = False,
) -> tuple[Array, Array, Qwen3InferenceState]:
    ml = max_len if max_len is not None else int(tokens.shape[1])
    x = embed_tokens(tokens, params.embed)
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
            params=block_p,
            cfg=cfg,
            rope_cache=rope_cache,
            token_valid_mask=valid,
        )
        kvs.append(kv_from_prefill(k, v, max_len=ml, lengths=kv_lens))
    x = rms_norm(x, params.final_norm)
    w_head = params.embed if cfg.use_tied_embedding else params.lm_head
    assert w_head is not None
    if last_logits_only:
        last_idx = jnp.maximum(jnp.sum(valid, axis=1).astype(jnp.int32) - 1, 0)
        logits = logits_from_hidden_last_positions(x, w_head, last_idx)
    else:
        logits = (
            logits_from_hidden_tied(x, w_head)
            if cfg.use_tied_embedding
            else logits_from_hidden(x, w_head)
        )
    return x, logits, Qwen3InferenceState(tuple(kvs))


def forward_decode_step(
    token_embedded: Array,
    positions: Array,
    state: Qwen3InferenceState,
    *,
    params: Qwen3Params,
    cfg: Qwen3Config,
    rope_cache: RopeCache | None,
    max_cache_len: int,
) -> tuple[Array, Array, Qwen3InferenceState]:
    x = token_embedded
    new_kvs: list[DenseKVState] = []
    for i, block_p in enumerate(params.blocks):
        x, kv_i = block_forward_decode(
            x,
            positions,
            state.kv[i],
            params=block_p,
            cfg=cfg,
            rope_cache=rope_cache,
            max_cache_len=max_cache_len,
        )
        new_kvs.append(kv_i)
    x = rms_norm(x, params.final_norm)
    w_head = params.embed if cfg.use_tied_embedding else params.lm_head
    assert w_head is not None
    logits = (
        logits_from_hidden_tied(x, w_head)
        if cfg.use_tied_embedding
        else logits_from_hidden(x, w_head)
    )
    return x, logits, Qwen3InferenceState(tuple(new_kvs))
