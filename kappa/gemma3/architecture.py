"""Gemma 3 dense architecture table and runtime config (no Flax)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Literal

import numpy as np

ModelSizeB = Literal[1, 4, 12, 27]


class AttentionType(IntEnum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


_BASE_PATTERN: tuple[AttentionType, ...] = (
    AttentionType.LOCAL_SLIDING,
) * 5 + (AttentionType.GLOBAL,)

# Rows: model_size_B, num_layers, num_heads, num_kv_heads, embed_dim, mlp_hidden, head_dim
_GEMMA3_VARIANTS = np.array(
    [
        [1, 4, 12, 27],
        [26, 34, 48, 62],
        [4, 8, 16, 32],
        [1, 4, 8, 16],
        [1152, 2560, 3840, 5376],
        [6912, 10240, 15360, 21504],
        [256, 256, 256, 128],
    ],
    dtype=np.int32,
)

_ROW = {
    "model_size": 0,
    "num_layers": 1,
    "num_heads": 2,
    "num_kv_heads": 3,
    "embed_dim": 4,
    "hidden_dim": 5,
    "head_dim": 6,
}


def _tile_pattern(n: int) -> tuple[AttentionType, ...]:
    rep, rem = divmod(n, len(_BASE_PATTERN))
    return _BASE_PATTERN * rep + _BASE_PATTERN[:rem]


def attention_pattern_for_layers(num_layers: int) -> tuple[AttentionType, ...]:
    """Gemma 3 local/global tiling (same as dense ``gemma3_dense_config``)."""
    return _tile_pattern(num_layers)


def query_pre_attn_scalar_for(model_size: int) -> float:
    """Pre-attention query scale (matches common Gemma 3 recipes; verify per release)."""
    head_dim_27b = 128
    embed_dim_27b = 5376
    num_heads_27b = 32
    if model_size == 27:
        return float((embed_dim_27b / num_heads_27b) ** -0.5)
    return float((head_dim_27b * 2) ** -0.5)


@dataclass(frozen=True, slots=True)
class Gemma3DenseConfig:
    model_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    hidden_dim: int
    head_dim: int
    attention_pattern: tuple[AttentionType, ...]
    query_pre_attn_scalar: float
    vocab_size: int = 262_144
    window_size: int = 1024
    local_base_frequency: int = 10_000
    local_scale_factor: float = 1.0
    global_base_frequency: int = 1_000_000
    global_scale_factor: float = 8.0
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    use_qk_norm: bool = True
    attn_logits_soft_cap: float | None = None
    final_logit_softcap: float | None = None
    transpose_gating_einsum: bool = True


def gemma3_dense_config(
    *,
    model_size: ModelSizeB,
    **overrides: object,
) -> Gemma3DenseConfig:
    if model_size not in (1, 4, 12, 27):
        raise ValueError("model_size must be one of 1, 4, 12, 27 (billions).")
    col = int(np.where(_GEMMA3_VARIANTS[_ROW["model_size"]] == model_size)[0][0])
    base = {k: int(_GEMMA3_VARIANTS[row, col]) for k, row in _ROW.items() if k != "model_size"}
    cfg = Gemma3DenseConfig(
        model_size=model_size,
        num_layers=base["num_layers"],
        num_heads=base["num_heads"],
        num_kv_heads=base["num_kv_heads"],
        embed_dim=base["embed_dim"],
        hidden_dim=base["hidden_dim"],
        head_dim=base["head_dim"],
        attention_pattern=_tile_pattern(base["num_layers"]),
        query_pre_attn_scalar=query_pre_attn_scalar_for(model_size),
    )
    return replace(cfg, **overrides)  # type: ignore[arg-type]
