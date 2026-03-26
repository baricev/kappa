"""Gemma 3 dense: config, weights, KV, attention, full transformer, generation."""

from kappa.gemma3.architecture import (
    AttentionType,
    Gemma3DenseConfig,
    attention_pattern_for_layers,
    gemma3_dense_config,
)
from kappa.gemma3.attention_ops import (
    decode_attention_dense,
    dense_gqa_attention,
    prefill_attention_dense,
    repeat_kv_heads,
)
from kappa.gemma3.block import (
    block_forward_decode,
    block_forward_prefill,
    block_forward_prefill_chunk,
    kv_from_prefill,
)
from kappa.gemma3.ffn import feed_forward
from kappa.gemma3.generate import generate
from kappa.gemma3.jit_inference import (
    jit_forward_decode_step,
    jit_forward_prefill,
    jit_forward_prefill_chunk,
    jit_generate,
)
from kappa.gemma3.kv_cache import (
    DenseKVState,
    advance_lengths,
    append_decode_token,
    init_dense_kv,
    set_lengths,
    write_kv_range,
)
from kappa.gemma3.linear import project_attn_out, project_kv, project_q
from kappa.gemma3.load import load_gemma3_dense_unsharded
from kappa.gemma3.masks import (
    bool_to_additive,
    causal_square,
    extended_causal_mask,
    local_sliding_extended,
    segment_rect_mask,
)
from kappa.gemma3.positions import positions_from_mask
from kappa.gemma3.prefill import (
    mask_prefill_chunk_with_prefix,
    prefill_chunk_autoselect,
    prefill_chunk_dense,
    prefill_chunk_with_prefix_dense,
    prefill_square_chunk_dense,
)
from kappa.gemma3.rope import (
    RopeCache,
    apply_rope,
    apply_rope_for_layer,
    build_rope_cache,
)
from kappa.gemma3.splash_prefill import prefill_square_chunk_splash
from kappa.gemma3.serving import DecodeState, PrefillState, decode_step, prefill_chunk
from kappa.gemma3.transformer import (
    DenseInferenceState,
    embed_tokens,
    forward_decode_step,
    forward_prefill,
    forward_prefill_chunk,
    gather_last_valid_logits,
    init_dense_inference_state,
    logits_from_hidden,
)
from kappa.gemma3.weights import (
    Gemma3DenseParams,
    LayerParams,
    param_tree_leaves_shapes,
    params_from_flat,
)

__all__ = [
    "AttentionType",
    "DenseInferenceState",
    "DenseKVState",
    "Gemma3DenseConfig",
    "Gemma3DenseParams",
    "LayerParams",
    "RopeCache",
    "advance_lengths",
    "append_decode_token",
    "apply_rope",
    "apply_rope_for_layer",
    "attention_pattern_for_layers",
    "DecodeState",
    "PrefillState",
    "block_forward_decode",
    "block_forward_prefill",
    "block_forward_prefill_chunk",
    "bool_to_additive",
    "build_rope_cache",
    "causal_square",
    "decode_attention_dense",
    "decode_step",
    "dense_gqa_attention",
    "embed_tokens",
    "extended_causal_mask",
    "feed_forward",
    "forward_decode_step",
    "forward_prefill",
    "forward_prefill_chunk",
    "gather_last_valid_logits",
    "gemma3_dense_config",
    "generate",
    "init_dense_inference_state",
    "init_dense_kv",
    "jit_forward_decode_step",
    "jit_forward_prefill",
    "jit_forward_prefill_chunk",
    "jit_generate",
    "kv_from_prefill",
    "load_gemma3_dense_unsharded",
    "local_sliding_extended",
    "logits_from_hidden",
    "mask_prefill_chunk_with_prefix",
    "param_tree_leaves_shapes",
    "params_from_flat",
    "positions_from_mask",
    "prefill_attention_dense",
    "prefill_chunk_autoselect",
    "prefill_chunk_dense",
    "prefill_chunk_with_prefix_dense",
    "prefill_square_chunk_dense",
    "prefill_chunk",
    "prefill_square_chunk_splash",
    "project_attn_out",
    "project_kv",
    "project_q",
    "repeat_kv_heads",
    "segment_rect_mask",
    "set_lengths",
    "write_kv_range",
]
