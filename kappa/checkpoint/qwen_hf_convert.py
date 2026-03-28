"""Convert raw HuggingFace flat weights (``model.*`` keys) to :class:`~kappa.qwen3.weights.Qwen3Params`.

Tensor layout matches Simply ``Qwen2Format`` + transposes to kappa's ``ndh`` attention weights.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
from jax import Array

from kappa.qwen3.architecture import Qwen3Config
from kappa.qwen3.weights import (
    AttnParams,
    DenseFfnParams,
    MoEFfnParams,
    Qwen3DenseBlockParams,
    Qwen3MoEBlockParams,
    Qwen3Params,
)


def is_huggingface_flat(flat: dict[str, Array]) -> bool:
    return "model.embed_tokens.weight" in flat


def count_hf_decoder_layers(flat: dict[str, Array]) -> int:
    """Largest ``N+1`` such that ``model.layers.{N}.`` keys exist."""
    mx = -1
    for k in flat:
        if m := re.match(r"^model\.layers\.(\d+)\.", k):
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else 0


def hf_flat_has_moe_router(flat: dict[str, Array]) -> bool:
    """Dense SwiGLU uses ``gate_proj``; MoE uses a ``gate`` router (no ``gate_proj``)."""
    return any(re.search(r"^model\.layers\.\d+\.mlp\.gate\.weight$", k) for k in flat)


def suggest_matching_preset(flat: dict[str, Array]) -> str | None:
    """Best-effort preset name from tensor shapes (for error hints)."""
    from kappa.qwen3.architecture import qwen3_config_for_preset

    if not is_huggingface_flat(flat):
        return None
    d = int(flat["model.embed_tokens.weight"].shape[1])
    n = count_hf_decoder_layers(flat)
    moe = hf_flat_has_moe_router(flat)
    for name in ("qwen3-0.6b", "qwen3-4b", "qwen3-30b-a3b"):
        c = qwen3_config_for_preset(name)  # type: ignore[arg-type]
        if c.model_dim == d and c.num_layers == n and c.use_moe == moe:
            return name
    return None


def validate_hf_flat_matches_preset(
    flat: dict[str, Array],
    cfg: Qwen3Config,
    *,
    preset: str,
) -> None:
    """Raise if HF tensors do not match the selected kappa preset (wrong ``--model``)."""
    if not is_huggingface_flat(flat):
        return
    d_ckpt = int(flat["model.embed_tokens.weight"].shape[1])
    n_ckpt = count_hf_decoder_layers(flat)
    moe_ckpt = hf_flat_has_moe_router(flat)
    ok = d_ckpt == cfg.model_dim and n_ckpt == cfg.num_layers and moe_ckpt == cfg.use_moe
    if ok:
        return
    sug = suggest_matching_preset(flat)
    hint = f" Try: --model {sug}" if sug else " No known preset matched this checkpoint."
    raise ValueError(
        "HF checkpoint tensors do not match the selected --model preset.\n"
        f"  preset {preset!r} expects: model_dim={cfg.model_dim}, num_layers={cfg.num_layers}, "
        f"use_moe={cfg.use_moe}\n"
        f"  checkpoint has: model_dim={d_ckpt}, num_layers={n_ckpt}, moe_router={moe_ckpt}\n"
        f"{hint}"
    )


def _split_head(v: Array, per_head_dim: int, *, axis: int = 0) -> Array:
    n = v.shape[axis] // per_head_dim
    new_shape = (*v.shape[:axis], n, per_head_dim, *v.shape[axis + 1 :])
    return jnp.reshape(v, new_shape)


def hf_flat_to_qwen3_params(
    flat: dict[str, Array],
    cfg: Qwen3Config,
    *,
    preset: str = "",
) -> Qwen3Params:
    """Build params from HF-style keys (safetensors / unconverted Orbax)."""
    if preset:
        validate_hf_flat_matches_preset(flat, cfg, preset=preset)
    hd = cfg.head_dim
    if cfg.use_moe:
        return _hf_flat_to_qwen3_moe_params(flat, cfg)
    return _hf_flat_to_qwen3_dense_params(flat, cfg, hd)


def _hf_flat_to_qwen3_dense_params(flat: dict[str, Array], cfg: Qwen3Config, hd: int) -> Qwen3Params:
    embed = flat["model.embed_tokens.weight"]
    final_norm = flat["model.norm.weight"]

    if "lm_head.weight" in flat:
        lm = jnp.transpose(flat["lm_head.weight"])
    else:
        lm = None

    if cfg.use_tied_embedding:
        lm_head: Array | None = None
    else:
        if lm is None:
            raise ValueError("untied model requires lm_head.weight")
        lm_head = lm

    blocks: list[Qwen3DenseBlockParams] = []
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        pre0 = flat[f"{p}.input_layernorm.weight"]
        pre1 = flat[f"{p}.post_attention_layernorm.weight"]
        qn = flat[f"{p}.self_attn.q_norm.weight"]
        kn = flat[f"{p}.self_attn.k_norm.weight"]

        qw = flat[f"{p}.self_attn.q_proj.weight"]
        kw = flat[f"{p}.self_attn.k_proj.weight"]
        vw = flat[f"{p}.self_attn.v_proj.weight"]
        ow = flat[f"{p}.self_attn.o_proj.weight"]

        q_dnh = jnp.einsum("nhd->dnh", _split_head(qw, hd, axis=0))
        q_proj = jnp.transpose(q_dnh, (1, 0, 2))

        k_dnh = jnp.einsum("nhd->dnh", _split_head(kw, hd, axis=0))
        k_proj = jnp.transpose(k_dnh, (1, 0, 2))

        v_dnh = jnp.einsum("nhd->dnh", _split_head(vw, hd, axis=0))
        v_proj = jnp.transpose(v_dnh, (1, 0, 2))

        o_mnh = _split_head(ow, hd, axis=1)
        o_proj = jnp.transpose(o_mnh, (1, 2, 0))

        gate = jnp.transpose(flat[f"{p}.mlp.gate_proj.weight"])
        up = jnp.transpose(flat[f"{p}.mlp.up_proj.weight"])
        down = jnp.transpose(flat[f"{p}.mlp.down_proj.weight"])
        ffn = DenseFfnParams(gate_proj=gate, up_proj=up, down_proj=down)

        attn = AttnParams(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            q_norm_scale=qn,
            k_norm_scale=kn,
        )
        blocks.append(Qwen3DenseBlockParams(pre0, pre1, attn, ffn))

    return Qwen3Params(embed=embed, final_norm=final_norm, blocks=tuple(blocks), lm_head=lm_head)


def _gather_experts(flat: dict[str, Array], layer: int, kind: str) -> Array:
    """Stack expert weights ``[E, ...]`` in order (HF ``mlp.experts.{e}.*``)."""
    pat = re.compile(rf"^model\.layers\.{layer}\.mlp\.experts\.(\d+)\.{kind}\.weight$")
    found: list[tuple[int, Array]] = []
    for k, v in flat.items():
        if m := pat.match(k):
            found.append((int(m.group(1)), v))
    if not found:
        raise KeyError(f"no expert tensors for layer {layer} kind {kind}")
    found.sort(key=lambda x: x[0])
    return jnp.stack([x[1] for x in found], axis=0)


def _hf_flat_to_qwen3_moe_params(flat: dict[str, Array], cfg: Qwen3Config) -> Qwen3Params:
    hd = cfg.head_dim
    embed = flat["model.embed_tokens.weight"]
    final_norm = flat["model.norm.weight"]
    lm_head: Array | None
    if cfg.use_tied_embedding:
        lm_head = None
    else:
        if "lm_head.weight" not in flat:
            raise ValueError("untied MoE model requires lm_head.weight")
        lm_head = jnp.transpose(flat["lm_head.weight"])

    blocks: list[Qwen3MoEBlockParams] = []
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        pre0 = flat[f"{p}.input_layernorm.weight"]
        pre1 = flat[f"{p}.post_attention_layernorm.weight"]
        qn = flat[f"{p}.self_attn.q_norm.weight"]
        kn = flat[f"{p}.self_attn.k_norm.weight"]

        qw = flat[f"{p}.self_attn.q_proj.weight"]
        kw = flat[f"{p}.self_attn.k_proj.weight"]
        vw = flat[f"{p}.self_attn.v_proj.weight"]
        ow = flat[f"{p}.self_attn.o_proj.weight"]

        q_dnh = jnp.einsum("nhd->dnh", _split_head(qw, hd, axis=0))
        q_proj = jnp.transpose(q_dnh, (1, 0, 2))
        k_dnh = jnp.einsum("nhd->dnh", _split_head(kw, hd, axis=0))
        k_proj = jnp.transpose(k_dnh, (1, 0, 2))
        v_dnh = jnp.einsum("nhd->dnh", _split_head(vw, hd, axis=0))
        v_proj = jnp.transpose(v_dnh, (1, 0, 2))
        o_mnh = _split_head(ow, hd, axis=1)
        o_proj = jnp.transpose(o_mnh, (1, 2, 0))

        attn = AttnParams(q_proj, k_proj, v_proj, o_proj, qn, kn)

        router = jnp.transpose(flat[f"{p}.mlp.gate.weight"])
        g_st = _gather_experts(flat, i, "gate_proj")
        u_st = _gather_experts(flat, i, "up_proj")
        d_st = _gather_experts(flat, i, "down_proj")
        gate_e = jnp.einsum("eoi->eio", g_st)
        up_e = jnp.einsum("eoi->eio", u_st)
        down_e = jnp.einsum("eoi->eio", d_st)
        ffn = MoEFfnParams(router=router, gate_proj=gate_e, up_proj=up_e, down_proj=down_e)
        blocks.append(Qwen3MoEBlockParams(pre0, pre1, attn, ffn))

    return Qwen3Params(embed=embed, final_norm=final_norm, blocks=tuple(blocks), lm_head=lm_head)