"""Qwen3 parameter trees (Simply / HF layout after ``Qwen2Format``)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from kappa.qwen3.quant import Q8Weight, Weight, quantize_weight

# Local alias avoids importing ``kappa.checkpoint`` (package ``__init__`` pulls ``qwen_hf_convert``
# → ``weights``) when this module is loaded from ``sharding`` or other early imports.
FlatParams = dict[str, Array]


def _weight_shape(w: Weight) -> tuple[int, ...]:
    return w.values.shape if isinstance(w, Q8Weight) else w.shape


def normalize_lm_head_for_logits(embed: Weight, lm_head: Weight | None) -> Weight | None:
    """``logits_from_hidden`` uses ``einsum(..., vd)`` with ``v=vocab``, ``d=model_dim``.

    HuggingFace ``lm_head.weight`` is ``[vocab, model_dim]``. Simply ``output_layer/w`` is often
    transposed to ``[model_dim, vocab]``.
    """
    if lm_head is None:
        return None
    v_sz, d_sz = _weight_shape(embed)[0], _weight_shape(embed)[1]
    ls = _weight_shape(lm_head)
    if ls[0] == v_sz and ls[1] == d_sz:
        return lm_head
    if ls[0] == d_sz and ls[1] == v_sz:
        if isinstance(lm_head, Q8Weight):
            return Q8Weight(values=jnp.transpose(lm_head.values), scale=lm_head.scale)
        return jnp.transpose(lm_head)
    raise ValueError(
        f"lm_head shape {ls} does not match embed table {(v_sz, d_sz)} "
        "as [vocab, dim] or [dim, vocab]."
    )


class AttnParams(NamedTuple):
    q_proj: Weight
    k_proj: Weight
    v_proj: Weight
    o_proj: Weight
    q_norm_scale: Array
    k_norm_scale: Array


class DenseFfnParams(NamedTuple):
    gate_proj: Weight
    up_proj: Weight
    down_proj: Weight


class MoEFfnParams(NamedTuple):
    router: Weight
    gate_proj: Weight
    up_proj: Weight
    down_proj: Weight


class Qwen3DenseBlockParams(NamedTuple):
    pre_attn_norm: Array
    pre_ffn_norm: Array
    attn: AttnParams
    ffn: DenseFfnParams


class Qwen3MoEBlockParams(NamedTuple):
    pre_attn_norm: Array
    pre_ffn_norm: Array
    attn: AttnParams
    ffn: MoEFfnParams


class Qwen3Params(NamedTuple):
    embed: Weight
    final_norm: Array
    blocks: tuple[Qwen3DenseBlockParams | Qwen3MoEBlockParams, ...]
    lm_head: Weight | None  # if not tied


def params_from_flat(
    flat: FlatParams,
    *,
    num_layers: int,
    use_moe: bool,
) -> Qwen3Params:
    """Load from dot-separated flat dict (see ``qwen_flat.load_qwen3_flat_params``)."""

    def _get(*candidates: str) -> Array:
        for c in candidates:
            if c in flat:
                return flat[c]
        raise KeyError(f"missing key; tried {candidates}")

    embed = _get("params.embed_linear.embed", "params.embed_linear.w", "embed_linear.embed", "embed_linear.w")
    final_norm = _get("params.final_ln.scale", "final_ln.scale")

    lm_head: Array | None
    if "params.output_layer.w" in flat or "output_layer.w" in flat:
        lm_head = _get("params.output_layer.w", "output_layer.w")
    else:
        lm_head = None

    blocks: list[Qwen3DenseBlockParams | Qwen3MoEBlockParams] = []
    for i in range(num_layers):
        p = f"params.block_{i}."
        alt = f"block_{i}."
        pre0 = _get(p + "pre_ln_0.scale", alt + "pre_ln_0.scale")
        pre1 = _get(p + "pre_ln_1.scale", alt + "pre_ln_1.scale")
        attn = AttnParams(
            q_proj=_get(p + "attn.q_proj.w", alt + "attn.q_proj.w"),
            k_proj=_get(p + "attn.k_proj.w", alt + "attn.k_proj.w"),
            v_proj=_get(p + "attn.v_proj.w", alt + "attn.v_proj.w"),
            o_proj=_get(p + "attn.o_proj.w", alt + "attn.o_proj.w"),
            q_norm_scale=_get(p + "attn.q_norm.scale", alt + "attn.q_norm.scale"),
            k_norm_scale=_get(p + "attn.k_norm.scale", alt + "attn.k_norm.scale"),
        )
        if use_moe:
            ffn = MoEFfnParams(
                router=_get(p + "ffn.router.w", alt + "ffn.router.w"),
                gate_proj=_get(p + "ffn.ffn_0_gate.w", alt + "ffn.ffn_0_gate.w"),
                up_proj=_get(p + "ffn.ffn_0.w", alt + "ffn.ffn_0.w"),
                down_proj=_get(p + "ffn.ffn_1.w", alt + "ffn.ffn_1.w"),
            )
            blocks.append(Qwen3MoEBlockParams(pre0, pre1, attn, ffn))
        else:
            ffn = DenseFfnParams(
                gate_proj=_get(p + "ffn.ffn_0_gate.w", alt + "ffn.ffn_0_gate.w"),
                up_proj=_get(p + "ffn.ffn_0.w", alt + "ffn.ffn_0.w"),
                down_proj=_get(p + "ffn.ffn_1.w", alt + "ffn.ffn_1.w"),
            )
            blocks.append(Qwen3DenseBlockParams(pre0, pre1, attn, ffn))

    lm_head = normalize_lm_head_for_logits(embed, lm_head)
    return Qwen3Params(embed=embed, final_norm=final_norm, blocks=tuple(blocks), lm_head=lm_head)


def quantize_qwen3_params(params: Qwen3Params, *, scale_dtype: jnp.dtype = jnp.bfloat16) -> Qwen3Params:
    """PTQ: linear weights → :class:`Q8Weight`; RMS norm scales stay full precision."""

    def q(w: Weight) -> Q8Weight:
        if isinstance(w, Q8Weight):
            return w
        return quantize_weight(w, scale_dtype=scale_dtype)

    def attn(a: AttnParams) -> AttnParams:
        return AttnParams(
            q_proj=q(a.q_proj),
            k_proj=q(a.k_proj),
            v_proj=q(a.v_proj),
            o_proj=q(a.o_proj),
            q_norm_scale=a.q_norm_scale,
            k_norm_scale=a.k_norm_scale,
        )

    def dense_ffn(f: DenseFfnParams) -> DenseFfnParams:
        return DenseFfnParams(gate_proj=q(f.gate_proj), up_proj=q(f.up_proj), down_proj=q(f.down_proj))

    def moe_ffn(f: MoEFfnParams) -> MoEFfnParams:
        return MoEFfnParams(
            router=q(f.router),
            gate_proj=q(f.gate_proj),
            up_proj=q(f.up_proj),
            down_proj=q(f.down_proj),
        )

    new_blocks: list[Qwen3DenseBlockParams | Qwen3MoEBlockParams] = []
    for b in params.blocks:
        ap = attn(b.attn)
        if isinstance(b, Qwen3MoEBlockParams):
            new_blocks.append(Qwen3MoEBlockParams(b.pre_attn_norm, b.pre_ffn_norm, ap, moe_ffn(b.ffn)))
        else:
            new_blocks.append(Qwen3DenseBlockParams(b.pre_attn_norm, b.pre_ffn_norm, ap, dense_ffn(b.ffn)))

    lm = params.lm_head
    lm_q = None if lm is None else q(lm)

    return Qwen3Params(
        embed=q(params.embed),
        final_norm=params.final_norm,
        blocks=tuple(new_blocks),
        lm_head=lm_q,
    )
