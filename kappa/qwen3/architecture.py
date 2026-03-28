"""Qwen3 model hyperparameters (Simply-aligned; single-device inference)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

ModelPreset = Literal[
    "qwen3-0.6b",
    "qwen3-4b",
    "qwen3-30b-a3b",
]


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Dense or MoE decoder-only config."""

    vocab_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    model_dim: int
    intermediate_size: int
    rope_theta: float
    rms_eps: float
    use_tied_embedding: bool
    # Attention
    attn_logits_soft_cap: float | None  # None if disabled (HF uses no softcap)
    # MoE (dense: use_moe=False, experts ignored)
    use_moe: bool
    num_experts: int
    num_experts_per_tok: int
    # Generation defaults (HF Qwen3 chat template)
    pad_token_id: int
    eos_token_id: int


def qwen3_config_for_preset(preset: ModelPreset) -> Qwen3Config:
    """Known public shapes; paths live in ``kappa.defaults``."""
    if preset == "qwen3-0.6b":
        return Qwen3Config(
            vocab_size=151_936,
            num_layers=28,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            model_dim=1024,
            intermediate_size=3072,
            rope_theta=1_000_000.0,
            rms_eps=1e-6,
            use_tied_embedding=True,
            attn_logits_soft_cap=None,
            use_moe=False,
            num_experts=0,
            num_experts_per_tok=0,
            pad_token_id=151_643,
            eos_token_id=151_645,
        )
    if preset == "qwen3-4b":
        return Qwen3Config(
            vocab_size=151_936,
            num_layers=36,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            model_dim=2560,
            intermediate_size=9728,
            rope_theta=1_000_000.0,
            rms_eps=1e-6,
            use_tied_embedding=True,
            attn_logits_soft_cap=None,
            use_moe=False,
            num_experts=0,
            num_experts_per_tok=0,
            pad_token_id=151_643,
            eos_token_id=151_645,
        )
    if preset == "qwen3-30b-a3b":
        return Qwen3Config(
            vocab_size=151_936,
            num_layers=48,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
            model_dim=2048,
            intermediate_size=768,
            rope_theta=1_000_000.0,
            rms_eps=1e-6,
            use_tied_embedding=False,
            attn_logits_soft_cap=None,
            use_moe=True,
            num_experts=128,
            num_experts_per_tok=8,
            pad_token_id=151_643,
            eos_token_id=151_645,
        )
    raise ValueError(f"unknown preset: {preset!r}")


def query_pre_attn_scalar(cfg: Qwen3Config) -> float:
    """1/sqrt(head_dim), Simply-style when query_scale < 0."""
    return float(cfg.head_dim**-0.5)
