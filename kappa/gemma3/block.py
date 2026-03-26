"""Single Gemma 3 dense transformer block (prefill + decode)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import AttentionType, Gemma3DenseConfig
from kappa.gemma3.attention_ops import decode_attention_dense
from kappa.gemma3.ffn import feed_forward
from kappa.gemma3.kv_cache import DenseKVState, append_decode_token, set_lengths
from kappa.gemma3.linear import project_attn_out, project_kv, project_q
from kappa.gemma3.norms import rms_norm
from kappa.gemma3.prefill import prefill_chunk_with_prefix_dense
from kappa.gemma3.rope import RopeCache, apply_rope_for_layer
from kappa.gemma3.weights import LayerParams


def _qk_norm(q: Array, k: Array, p: LayerParams, use_qk_norm: bool) -> tuple[Array, Array]:
    if not use_qk_norm:
        return q, k
    q = rms_norm(q, p.attn_query_norm_scale)
    k = rms_norm(k, p.attn_key_norm_scale)
    return q, k


def block_forward_prefill(
    x: Array,
    positions: Array,
    segment_ids: Array | None,
    *,
    params: LayerParams,
    cfg: Gemma3DenseConfig,
    layer_idx: int,
    rope_cache: RopeCache | None,
    token_valid_mask: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Prefill one layer: ``x`` ``[B, L, E]``, ``positions`` ``[B, L]``.

    Returns ``(hidden_out, k, v)`` where ``k,v`` are RoPE-applied KV for cache seeding.
    """
    attn_type = cfg.attention_pattern[layer_idx]
    h = rms_norm(x, params.pre_attention_norm_scale)
    q = project_q(h, params.q_proj)
    k, v = project_kv(h, params.kv_proj)
    q, k = _qk_norm(q, k, params, cfg.use_qk_norm)
    q = apply_rope_for_layer(q, positions, attn_type=attn_type, cfg=cfg, rope_cache=rope_cache)
    k = apply_rope_for_layer(k, positions, attn_type=attn_type, cfg=cfg, rope_cache=rope_cache)
    q = q * jnp.asarray(cfg.query_pre_attn_scalar, dtype=q.dtype)

    b, l, _, _ = q.shape
    prefix_len = jnp.zeros((b,), dtype=jnp.int32)
    sq = segment_ids if segment_ids is not None else None
    sk = segment_ids if segment_ids is not None else None

    attn = prefill_chunk_with_prefix_dense(
        q,
        k,
        v,
        prefix_len=prefix_len,
        attn_type=attn_type,
        window_size=cfg.window_size,
        segment_q=sq,
        segment_k=sk,
        token_valid=token_valid_mask,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
    )
    attn = project_attn_out(attn, params.output_proj)
    if cfg.use_post_attn_norm:
        attn = rms_norm(attn, params.post_attention_norm_scale)
    h = x + attn

    out = rms_norm(h, params.pre_ffw_norm_scale)
    out = feed_forward(
        out,
        params.gating_weights,
        params.output_weights,
        transpose_gating_einsum=cfg.transpose_gating_einsum,
    )
    if cfg.use_post_ffw_norm:
        out = rms_norm(out, params.post_ffw_norm_scale)
    return h + out, k, v


def block_forward_prefill_chunk(
    x_chunk: Array,
    positions_chunk: Array,
    *,
    params: LayerParams,
    cfg: Gemma3DenseConfig,
    layer_idx: int,
    rope_cache: RopeCache | None,
    prefix_k: Array | None,
    prefix_v: Array | None,
    start_pos: int,
    token_valid_chunk: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Prefill one layer on a **chunk** with optional **prefix KV** already in the cache (MaxText-style).

    Queries run only on ``x_chunk`` (length ``Lc``). Keys/values are ``concat(prefix, chunk)`` along
    the sequence axis so attention is dense-prefix: query ``i`` in the chunk attends to keys
    ``j <= start_pos + i`` (``extended_causal_mask`` with ``prefix_len = start_pos``).

    When ``start_pos == 0``, this matches :func:`block_forward_prefill` on the chunk alone.

    ``start_pos`` is **uniform** across batch rows (length-bucketed compilation). ``prefix_k`` /
    ``prefix_v`` must be ``[B, start_pos, n_kv, D]`` when ``start_pos > 0``.
    """
    attn_type = cfg.attention_pattern[layer_idx]
    b, lc, _ = x_chunk.shape
    if start_pos == 0:
        valid = (
            token_valid_chunk
            if token_valid_chunk is not None
            else jnp.ones((b, lc), dtype=jnp.bool_)
        )
        return block_forward_prefill(
            x_chunk,
            positions_chunk,
            None,
            params=params,
            cfg=cfg,
            layer_idx=layer_idx,
            rope_cache=rope_cache,
            token_valid_mask=valid,
        )

    if prefix_k is None or prefix_v is None:
        raise ValueError("prefix_k and prefix_v required when start_pos > 0")
    if int(prefix_k.shape[1]) != start_pos:
        raise ValueError(f"prefix_k length {prefix_k.shape[1]} != start_pos {start_pos}")

    h = rms_norm(x_chunk, params.pre_attention_norm_scale)
    q = project_q(h, params.q_proj)
    k_ch, v_ch = project_kv(h, params.kv_proj)
    q, k_ch = _qk_norm(q, k_ch, params, cfg.use_qk_norm)
    q = apply_rope_for_layer(q, positions_chunk, attn_type=attn_type, cfg=cfg, rope_cache=rope_cache)
    k_ch = apply_rope_for_layer(k_ch, positions_chunk, attn_type=attn_type, cfg=cfg, rope_cache=rope_cache)
    q = q * jnp.asarray(cfg.query_pre_attn_scalar, dtype=q.dtype)

    # Prefix positions 0..start_pos-1 — RoPE was applied when those KV were written; use cached tensors as-is.
    k_full = jnp.concatenate([prefix_k, k_ch], axis=1)
    v_full = jnp.concatenate([prefix_v, v_ch], axis=1)

    prefix_len = jnp.full((b,), start_pos, dtype=jnp.int32)
    sq = None
    sk = None
    valid_ch = token_valid_chunk if token_valid_chunk is not None else jnp.ones((b, lc), dtype=jnp.bool_)
    valid_full = jnp.concatenate(
        [jnp.ones((b, start_pos), dtype=jnp.bool_), valid_ch.astype(jnp.bool_)],
        axis=1,
    )

    attn = prefill_chunk_with_prefix_dense(
        q,
        k_full,
        v_full,
        prefix_len=prefix_len,
        attn_type=attn_type,
        window_size=cfg.window_size,
        segment_q=sq,
        segment_k=sk,
        token_valid=valid_full,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
    )
    attn = project_attn_out(attn, params.output_proj)
    if cfg.use_post_attn_norm:
        attn = rms_norm(attn, params.post_attention_norm_scale)
    h = x_chunk + attn

    out = rms_norm(h, params.pre_ffw_norm_scale)
    out = feed_forward(
        out,
        params.gating_weights,
        params.output_weights,
        transpose_gating_einsum=cfg.transpose_gating_einsum,
    )
    if cfg.use_post_ffw_norm:
        out = rms_norm(out, params.post_ffw_norm_scale)
    return h + out, k_ch, v_ch


def block_forward_decode(
    x: Array,
    positions: Array,
    kv: DenseKVState,
    *,
    params: LayerParams,
    cfg: Gemma3DenseConfig,
    layer_idx: int,
    rope_cache: RopeCache | None,
) -> tuple[Array, DenseKVState]:
    """Decode one layer: ``x`` ``[B, 1, E]``, ``positions`` ``[B, 1]``; returns updated ``kv``."""
    attn_type = cfg.attention_pattern[layer_idx]
    h = rms_norm(x, params.pre_attention_norm_scale)
    q = project_q(h, params.q_proj)
    k, v = project_kv(h, params.kv_proj)
    q, k = _qk_norm(q, k, params, cfg.use_qk_norm)
    q = apply_rope_for_layer(q, positions, attn_type=attn_type, cfg=cfg, rope_cache=rope_cache)
    k = apply_rope_for_layer(k, positions, attn_type=attn_type, cfg=cfg, rope_cache=rope_cache)
    q = q * jnp.asarray(cfg.query_pre_attn_scalar, dtype=q.dtype)

    kv_new = append_decode_token(kv, k, v)
    # Effective window: actual window for local layers, full cache for global (no-op).
    # jnp.where accepts both Python and traced attn_type — scan-friendly.
    max_cache = kv_new.k.shape[1]
    effective_ws = jnp.where(
        jnp.asarray(attn_type == int(AttentionType.LOCAL_SLIDING)),
        jnp.asarray(cfg.window_size, dtype=jnp.int32),
        jnp.asarray(max_cache, dtype=jnp.int32),
    )
    attn = decode_attention_dense(
        q,
        kv_new.k,
        kv_new.v,
        kv_lengths=kv_new.lengths,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        window_size=effective_ws,
    )
    attn = project_attn_out(attn, params.output_proj)
    if cfg.use_post_attn_norm:
        attn = rms_norm(attn, params.post_attention_norm_scale)
    h = x + attn

    out = rms_norm(h, params.pre_ffw_norm_scale)
    out = feed_forward(
        out,
        params.gating_weights,
        params.output_weights,
        transpose_gating_einsum=cfg.transpose_gating_einsum,
    )
    if cfg.use_post_ffw_norm:
        out = rms_norm(out, params.post_ffw_norm_scale)
    return h + out, kv_new


def kv_from_prefill(
    k: Array,
    v: Array,
    *,
    max_len: int,
    lengths: Array | None = None,
) -> DenseKVState:
    """Build cache state from full prefill ``k,v`` ``[B, L, n_kv, D]`` (left-aligned).

    ``lengths`` (int32 ``[B]``): valid token count per row (e.g. sum of non-padding mask).
    Defaults to ``L`` (dense prompts with no padding).

    Uses ``jnp.pad`` directly instead of allocating zeros + vmap'd ``dynamic_update_slice``.
    """
    b, l, _, _ = k.shape
    pad_len = max_len - l
    pad_width = ((0, 0), (0, pad_len), (0, 0), (0, 0))
    k_padded = jnp.pad(k, pad_width)
    v_padded = jnp.pad(v, pad_width)
    lens = jnp.full((b,), l, dtype=jnp.int32) if lengths is None else lengths.astype(jnp.int32)
    return DenseKVState(k=k_padded, v=v_padded, lengths=lens)
