"""JIT entry points mirroring ``kappa.gemma3.jit_inference`` (Qwen3)."""

from __future__ import annotations

import jax

from kappa.qwen3.transformer import forward_decode_step, forward_prefill, forward_prefill_chunk

jit_forward_prefill = jax.jit(forward_prefill, static_argnames=("cfg", "max_len", "last_logits_only"))

jit_forward_decode_step = jax.jit(forward_decode_step, static_argnames=("cfg", "max_cache_len"))

jit_forward_prefill_chunk = jax.jit(
    forward_prefill_chunk, static_argnames=("cfg", "start_pos", "last_logits_only")
)

__all__ = [
    "jit_forward_decode_step",
    "jit_forward_prefill",
    "jit_forward_prefill_chunk",
]
