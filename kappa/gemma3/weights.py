"""Gemma 3 dense parameter PyTrees (NamedTuples)."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from kappa.checkpoint.orbax_flat import FlatParams


class LayerParams(NamedTuple):
    attn_key_norm_scale: Array
    attn_query_norm_scale: Array
    output_proj: Array
    kv_proj: Array
    q_proj: Array
    gating_weights: Array
    output_weights: Array
    post_attention_norm_scale: Array
    post_ffw_norm_scale: Array
    pre_attention_norm_scale: Array
    pre_ffw_norm_scale: Array


class Gemma3DenseParams(NamedTuple):
    input_embedding_table: Array
    mm_input_projection: Array
    mm_soft_embedding_norm: Array
    final_norm_scale: Array
    blocks: tuple[LayerParams, ...]


def params_from_flat(flat: FlatParams, *, num_layers: int) -> Gemma3DenseParams:
    """Map flat Orbax-style keys to a Gemma3DenseParams tree."""

    def layer(i: int) -> LayerParams:
        p = f"transformer.layer_{i}."
        return LayerParams(
            flat[p + "attn._key_norm.scale"],
            flat[p + "attn._query_norm.scale"],
            flat[p + "attn.attn_vec_einsum.w"],
            flat[p + "attn.kv_einsum.w"],
            flat[p + "attn.q_einsum.w"],
            flat[p + "mlp.gating_einsum.w"],
            flat[p + "mlp.linear.w"],
            flat[p + "post_attention_norm.scale"],
            flat[p + "post_ffw_norm.scale"],
            flat[p + "pre_attention_norm.scale"],
            flat[p + "pre_ffw_norm.scale"],
        )

    return Gemma3DenseParams(
        flat["transformer.embedder.input_embedding"],
        flat["transformer.embedder.mm_input_projection.w"],
        flat["transformer.embedder.mm_soft_embedding_norm.scale"],
        flat["transformer.final_norm.scale"],
        tuple(layer(i) for i in range(num_layers)),
    )


def param_tree_leaves_shapes(params: Gemma3DenseParams) -> dict[str, tuple[int, ...]]:
    """Host-only introspection for logging / tests."""

    def spec(x: Array) -> tuple[int, ...]:
        return tuple(int(d) for d in x.shape)

    out: dict[str, tuple[int, ...]] = {
        "input_embedding_table": spec(params.input_embedding_table),
        "mm_input_projection": spec(params.mm_input_projection),
        "mm_soft_embedding_norm": spec(params.mm_soft_embedding_norm),
        "final_norm_scale": spec(params.final_norm_scale),
    }
    for i, block in enumerate(params.blocks):
        prefix = f"blocks[{i}]."
        for name in LayerParams._fields:
            out[prefix + name] = spec(getattr(block, name))
    return out
