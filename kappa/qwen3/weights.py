"""Qwen3 parameter trees (Simply / HF layout after ``Qwen2Format``)."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from kappa.checkpoint.qwen_flat import FlatParams


class AttnParams(NamedTuple):
    q_proj: Array
    k_proj: Array
    v_proj: Array
    o_proj: Array
    q_norm_scale: Array
    k_norm_scale: Array


class DenseFfnParams(NamedTuple):
    gate_proj: Array
    up_proj: Array
    down_proj: Array


class MoEFfnParams(NamedTuple):
    router: Array
    gate_proj: Array
    up_proj: Array
    down_proj: Array


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
    embed: Array
    final_norm: Array
    blocks: tuple[Qwen3DenseBlockParams | Qwen3MoEBlockParams, ...]
    lm_head: Array | None  # if not tied


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

    return Qwen3Params(embed=embed, final_norm=final_norm, blocks=tuple(blocks), lm_head=lm_head)
