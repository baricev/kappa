"""JIT-compiled inference entry points: one XLA program per call for prefill, decode, and generation."""

from __future__ import annotations

import jax

from kappa.gemma3.generate import generate
from kappa.gemma3.transformer import forward_decode_step, forward_prefill, forward_prefill_chunk

jit_forward_prefill = jax.jit(forward_prefill, static_argnames=("cfg", "max_len", "last_logits_only"))

jit_forward_decode_step = jax.jit(forward_decode_step, static_argnames=("cfg",))

jit_forward_prefill_chunk = jax.jit(
    forward_prefill_chunk, static_argnames=("cfg", "start_pos", "last_logits_only")
)

jit_generate = jax.jit(
    generate,
    static_argnames=(
        "cfg",
        "max_new_tokens",
        "max_cache_len",
        "temperature",
        "top_k",
        "top_p",
        "pad_token_id",
        "decode_scan_chunk_size",
        "stop_token_ids",
        "return_timings",
    ),
)

__all__ = [
    "jit_forward_decode_step",
    "jit_forward_prefill",
    "jit_forward_prefill_chunk",
    "jit_generate",
]
