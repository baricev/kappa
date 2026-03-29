"""Qwen3 functional JAX (dense + MoE).

Load checkpoints via ``from kappa.qwen3.load import load_qwen3_unsharded`` (avoids import cycles).
"""

from kappa.qwen3.architecture import (
    ModelPreset,
    MoEImpl,
    QuantMode,
    Qwen3Config,
    qwen3_config_for_preset,
)
from kappa.qwen3.hf_tokenizer import QwenHfTokenizer

__all__ = [
    "ModelPreset",
    "MoEImpl",
    "QuantMode",
    "Qwen3Config",
    "QwenHfTokenizer",
    "qwen3_config_for_preset",
]
