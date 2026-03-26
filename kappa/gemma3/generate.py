"""Greedy / sampling generation using ``forward_prefill`` + ``forward_decode_step``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kappa.gemma3.architecture import Gemma3DenseConfig
from kappa.gemma3.positions import positions_from_mask
from kappa.gemma3.rope import RopeCache
from kappa.gemma3.sampling import sample_from_logits
from kappa.gemma3.transformer import (
    embed_tokens,
    forward_decode_step,
    forward_prefill,
    gather_last_valid_logits,
)
from kappa.gemma3.weights import Gemma3DenseParams


def generate(
    rng: Array,
    prompt_tokens: Array,
    *,
    params: Gemma3DenseParams,
    cfg: Gemma3DenseConfig,
    rope_cache: RopeCache | None,
    max_new_tokens: int,
    max_cache_len: int,
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
    pad_token_id: int = 0,
    decode_scan_chunk_size: int | None = 128,
) -> Array:
    """Generate up to ``max_new_tokens`` new tokens after ``prompt_tokens`` (``[B, L]``).

    Uses Gemma-style **non-padding** masks (``token != pad_token_id``), RoPE positions from
    ``positions_from_mask`` (same recipe as ``gemma/gm/math/_pos_utils.build_positions_from_mask``),
    attention masks that drop padding query/key pairs, KV cache lengths equal to the **valid** prompt
    length (not the padded tensor width), and logits at the **last valid** prompt position — not
    blindly ``[:, -1]`` — for the first sampled token.

    Decode steps advance a **per-row** position vector (supports batched right-padded prompts).

    ``decode_scan_chunk_size``: split the autoregressive ``jax.lax.scan`` over decode steps into
    multiple scans of at most this length (Python loop between chunks). Same math as one long scan;
    helps JAX MPS avoid resource limits on very long generations. Use ``None`` or ``<= 0`` for a
    single ``lax.scan`` over all ``max_new_tokens - 1`` steps (legacy behavior).

    Returns ``[B, L + max_new_tokens]``. Trimming at EOS is left to callers (e.g. the infer script).
    """
    valid = (prompt_tokens != pad_token_id).astype(jnp.bool_)
    pos0 = positions_from_mask(valid)

    _, logits, state = forward_prefill(
        prompt_tokens,
        pos0,
        params=params,
        cfg=cfg,
        rope_cache=rope_cache,
        segment_ids=None,
        token_valid_mask=valid,
        max_len=max_cache_len,
    )

    logit = gather_last_valid_logits(logits, valid)
    rng, rng_a = jax.random.split(rng)
    if temperature == 0.0:
        first = jnp.argmax(logit, axis=-1).astype(jnp.int32)[:, None]
    else:
        tok, _ = sample_from_logits(
            rng_a,
            logit,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        first = tok[:, None].astype(jnp.int32)

    if max_new_tokens <= 1:
        return jnp.concatenate([prompt_tokens, first], axis=1)

    pos_vec = jnp.sum(valid, axis=1).astype(jnp.int32)

    def step(carry, _):
        rng_i, st, last_tok, pos_b = carry
        x = embed_tokens(last_tok, params.input_embedding_table)
        pos_1 = pos_b[:, None]
        _, lg, st_n = forward_decode_step(
            x,
            pos_1,
            st,
            params=params,
            cfg=cfg,
            rope_cache=rope_cache,
        )
        lt = lg[:, 0, :]
        rng_i, sub = jax.random.split(rng_i)
        if temperature == 0.0:
            nt = jnp.argmax(lt, axis=-1).astype(jnp.int32)[:, None]
        else:
            tok2, _ = sample_from_logits(sub, lt, temperature=temperature, top_k=top_k, top_p=top_p)
            nt = tok2[:, None].astype(jnp.int32)
        return (rng_i, st_n, nt, pos_b + 1), nt

    rng, rng_loop = jax.random.split(rng)
    carry = (rng_loop, state, first, pos_vec)
    total_steps = max_new_tokens - 1
    chunk = decode_scan_chunk_size
    use_chunks = chunk is not None and chunk > 0 and total_steps > chunk

    if not use_chunks:
        _, rest = jax.lax.scan(step, carry, None, length=total_steps)
    else:
        chunks: list[Array] = []
        remaining = total_steps
        while remaining > 0:
            n = min(chunk, remaining)
            carry, chunk_out = jax.lax.scan(step, carry, None, length=n)
            chunks.append(chunk_out)
            remaining -= n
        rest = jnp.concatenate(chunks, axis=0)

    rest = jnp.swapaxes(rest, 0, 1)[..., 0]
    return jnp.concatenate([prompt_tokens, first, rest], axis=1)
