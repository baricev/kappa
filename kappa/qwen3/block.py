"""Qwen3 decoder block: pre-LN, GQA, SwiGLU or MoE."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import AttentionType
from kappa.gemma3.attention_ops import decode_attention_dense
from kappa.gemma3.kv_cache import DenseKVState, append_decode_token
from kappa.gemma3.prefill import prefill_chunk_with_prefix_dense as _prefill_attn
from kappa.qwen3.architecture import Qwen3Config, query_pre_attn_scalar
from kappa.qwen3.ffn import swiglu_ffn
from kappa.qwen3.ffn_moe import moe_swiglu_ffn
from kappa.qwen3.linear import project_attn_out, project_kv, project_q
from kappa.qwen3.norms import rms_norm
from kappa.qwen3.rope import apply_rope_qwen3
from kappa.qwen3.weights import Qwen3DenseBlockParams, Qwen3MoEBlockParams
from kappa.gemma3.rope import RopeCache


def _qk_norm(q: Array, k: Array, q_s: Array, k_s: Array) -> tuple[Array, Array]:
    q = rms_norm(q, q_s)
    k = rms_norm(k, k_s)
    return q, k


def block_forward_prefill(
    x: Array,
    positions: Array,
    *,
    params: Qwen3DenseBlockParams | Qwen3MoEBlockParams,
    cfg: Qwen3Config,
    rope_cache: RopeCache | None,
    token_valid_mask: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Returns ``(hidden, k, v)`` for cache seeding."""
    valid = (
        token_valid_mask
        if token_valid_mask is not None
        else jnp.ones((x.shape[0], x.shape[1]), dtype=jnp.bool_)
    )
    h = rms_norm(x, params.pre_attn_norm)
    q = project_q(h, params.attn.q_proj)
    k, v = project_kv(h, params.attn.k_proj, params.attn.v_proj)
    q, k = _qk_norm(q, k, params.attn.q_norm_scale, params.attn.k_norm_scale)
    q = apply_rope_qwen3(q, positions, rope_cache, cfg=cfg)
    k = apply_rope_qwen3(k, positions, rope_cache, cfg=cfg)
    q = q * jnp.asarray(query_pre_attn_scalar(cfg), dtype=q.dtype)

    b, l, _, _ = q.shape
    prefix_len = jnp.zeros((b,), dtype=jnp.int32)
    attn = _prefill_attn(
        q,
        k,
        v,
        prefix_len=prefix_len,
        attn_type=AttentionType.GLOBAL,
        window_size=l,
        segment_q=None,
        segment_k=None,
        token_valid=valid,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
    )
    attn = project_attn_out(attn, params.attn.o_proj)
    h = x + attn

    h2 = rms_norm(h, params.pre_ffn_norm)
    if isinstance(params, Qwen3MoEBlockParams):
        assert cfg.use_moe
        out = moe_swiglu_ffn(
            h2,
            params.ffn.router,
            params.ffn.gate_proj,
            params.ffn.up_proj,
            params.ffn.down_proj,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
        )
    else:
        out = swiglu_ffn(h2, params.ffn.gate_proj, params.ffn.up_proj, params.ffn.down_proj)
    return h + out, k, v


def block_forward_decode(
    x: Array,
    positions: Array,
    kv: DenseKVState,
    *,
    params: Qwen3DenseBlockParams | Qwen3MoEBlockParams,
    cfg: Qwen3Config,
    rope_cache: RopeCache | None,
    max_cache_len: int,
) -> tuple[Array, DenseKVState]:
    h = rms_norm(x, params.pre_attn_norm)
    q = project_q(h, params.attn.q_proj)
    k, v = project_kv(h, params.attn.k_proj, params.attn.v_proj)
    q, k = _qk_norm(q, k, params.attn.q_norm_scale, params.attn.k_norm_scale)
    q = apply_rope_qwen3(q, positions, rope_cache, cfg=cfg)
    k = apply_rope_qwen3(k, positions, rope_cache, cfg=cfg)
    q = q * jnp.asarray(query_pre_attn_scalar(cfg), dtype=q.dtype)

    kv_new = append_decode_token(kv, k, v)
    attn = decode_attention_dense(
        q,
        kv_new.k,
        kv_new.v,
        kv_lengths=kv_new.lengths,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        window_size=max_cache_len,
    )
    attn = project_attn_out(attn, params.attn.o_proj)
    h = x + attn

    h2 = rms_norm(h, params.pre_ffn_norm)
    if isinstance(params, Qwen3MoEBlockParams):
        out = moe_swiglu_ffn(
            h2,
            params.ffn.router,
            params.ffn.gate_proj,
            params.ffn.up_proj,
            params.ffn.down_proj,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
        )
    else:
        out = swiglu_ffn(h2, params.ffn.gate_proj, params.ffn.up_proj, params.ffn.down_proj)
    return h + out, kv_new


def kv_from_prefill(
    k: Array,
    v: Array,
    *,
    max_len: int,
    lengths: Array | None = None,
) -> DenseKVState:
    b, l, _, _ = k.shape
    pad_len = max_len - l
    pad_width = ((0, 0), (0, pad_len), (0, 0), (0, 0))
    k_padded = jnp.pad(k, pad_width)
    v_padded = jnp.pad(v, pad_width)
    lens = jnp.full((b,), l, dtype=jnp.int32) if lengths is None else lengths.astype(jnp.int32)
    return DenseKVState(k=k_padded, v=v_padded, lengths=lens)
