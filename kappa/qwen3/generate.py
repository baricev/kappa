"""Greedy / sampling generation for Qwen3."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from kappa.gemma3.positions import positions_from_mask
from kappa.gemma3.sampling import sample_from_logits
from kappa.qwen3.architecture import Qwen3Config
from kappa.qwen3.rope import RopeCache
from kappa.qwen3.special_tokens import QWEN3_EOS
from kappa.qwen3.quant import weight_param_dtype
from kappa.qwen3.transformer import (
    embed_tokens,
    forward_decode_step,
    forward_prefill,
    forward_prefill_chunk,
    init_qwen3_inference_state,
)
from kappa.qwen3.weights import Qwen3Params


def _block(*xs: Any) -> None:
    for x in xs:
        jax.block_until_ready(x)


def generate(
    rng: Array,
    prompt_tokens: Array,
    *,
    params: Qwen3Params,
    cfg: Qwen3Config,
    rope_cache: RopeCache | None,
    max_new_tokens: int,
    max_cache_len: int,
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    decode_scan_chunk_size: int | None = 128,
    prefill_chunk_size: int | None = None,
    stop_token_ids: tuple[int, ...] | None = None,
    return_timings: bool = False,
) -> Array | tuple[Array, dict[str, float]]:
    """Autoregressive generation (same control flow as ``gemma3.generate``).

    ``prefill_chunk_size``: if set and ``> 0`` and the prompt is longer, run prefill in
    fixed-size chunks (padding the last chunk). Requires **uniform** valid length per batch
    row (or batch size 1); otherwise falls back to one-shot prefill.
    """
    pad = cfg.pad_token_id if pad_token_id is None else pad_token_id
    t_all0 = time.perf_counter()
    valid = (prompt_tokens != pad).astype(jnp.bool_)
    pos0 = positions_from_mask(valid)

    t_pf0 = time.perf_counter()
    b, plen = int(prompt_tokens.shape[0]), int(prompt_tokens.shape[1])
    fc = prefill_chunk_size
    lens = jnp.sum(valid, axis=1)
    uniform = b == 1 or bool(jnp.all(lens == lens[0]))
    use_chunked = (
        fc is not None
        and int(fc) > 0
        and plen > int(fc)
        and uniform
    )
    if use_chunked:
        dtype = weight_param_dtype(params.embed)
        state = init_qwen3_inference_state(cfg, batch=b, max_len=max_cache_len, dtype=dtype)
        fc_i = int(fc)
        offset = 0
        logits = None
        while offset < plen:
            lc_read = min(fc_i, plen - offset)
            raw = prompt_tokens[:, offset : offset + lc_read]
            vraw = valid[:, offset : offset + lc_read]
            if lc_read < fc_i:
                pad_tail = jnp.full((b, fc_i - lc_read), pad, dtype=prompt_tokens.dtype)
                chunk = jnp.concatenate([raw, pad_tail], axis=1)
                vpad = jnp.zeros((b, fc_i - lc_read), dtype=jnp.bool_)
                valid_chunk = jnp.concatenate([vraw, vpad], axis=1)
            else:
                chunk = raw
                valid_chunk = vraw
            true_lens = jnp.sum(valid_chunk, axis=1).astype(jnp.int32)
            last_chunk = offset + fc_i >= plen
            _, logits, state = forward_prefill_chunk(
                chunk,
                params=params,
                cfg=cfg,
                rope_cache=rope_cache,
                state=state,
                start_pos=offset,
                true_chunk_lens=true_lens,
                last_logits_only=last_chunk,
            )
            offset += fc_i
    else:
        _, logits, state = forward_prefill(
            prompt_tokens,
            pos0,
            params=params,
            cfg=cfg,
            rope_cache=rope_cache,
            token_valid_mask=valid,
            max_len=max_cache_len,
            last_logits_only=True,
        )
    _block(logits, state)
    prefill_s = time.perf_counter() - t_pf0

    logit = logits
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
    _block(first)
    ttft_s = time.perf_counter() - t_all0
    t_dec0 = time.perf_counter()

    def _finish(out: Array, *, decode_s: float | None = None) -> Array | tuple[Array, dict[str, float]]:
        _block(out)
        ds = time.perf_counter() - t_dec0 if decode_s is None else decode_s
        total_s = time.perf_counter() - t_all0
        if not return_timings:
            return out
        return out, {
            "prefill_s": prefill_s,
            "ttft_s": ttft_s,
            "decode_s": ds,
            "total_s": total_s,
        }

    if max_new_tokens <= 1:
        out = jnp.concatenate([prompt_tokens, first], axis=1)
        return _finish(out, decode_s=0.0)

    pos_vec = jnp.sum(valid, axis=1).astype(jnp.int32)
    total_steps = max_new_tokens - 1
    if stop_token_ids is None:
        stops = (eos_token_id if eos_token_id is not None else QWEN3_EOS,)
    else:
        stops = stop_token_ids
    use_early_stop = len(stops) > 0 and int(prompt_tokens.shape[0]) == 1
    stop_set = frozenset(stops)

    if use_early_stop and int(first[0, 0]) in stop_set:
        out = jnp.concatenate([prompt_tokens, first], axis=1)
        return _finish(out, decode_s=0.0)

    def step(carry, _):
        rng_i, st, last_tok, pos_b = carry
        x = embed_tokens(last_tok, params.embed)
        pos_1 = pos_b[:, None]
        _, lg, st_n = forward_decode_step(
            x,
            pos_1,
            st,
            params=params,
            cfg=cfg,
            rope_cache=rope_cache,
            max_cache_len=max_cache_len,
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
    chunk = decode_scan_chunk_size
    use_chunks = chunk is not None and chunk > 0 and total_steps > chunk

    if use_early_stop:
        b = int(prompt_tokens.shape[0])
        stop_arr = jnp.array(stops, dtype=jnp.int32)

        def cond(carry_w):
            _rng, _st, _lt, _pb, i, _rb, go = carry_w
            return jnp.logical_and(go, i < total_steps)

        def body(carry_w):
            rng_i, st, last_tok, pos_b, i, rest_buf, go = carry_w
            x = embed_tokens(last_tok, params.embed)
            pos_1 = pos_b[:, None]
            _, lg, st_n = forward_decode_step(
                x,
                pos_1,
                st,
                params=params,
                cfg=cfg,
                rope_cache=rope_cache,
                max_cache_len=max_cache_len,
            )
            lt = lg[:, 0, :]
            rng_i, sub = jax.random.split(rng_i)
            if temperature == 0.0:
                nt = jnp.argmax(lt, axis=-1).astype(jnp.int32)[:, None]
            else:
                tok2, _ = sample_from_logits(sub, lt, temperature=temperature, top_k=top_k, top_p=top_p)
                nt = tok2[:, None].astype(jnp.int32)
            rest_buf = rest_buf.at[i].set(nt[:, 0])
            hit = jnp.isin(nt[0, 0], stop_arr)
            go_new = jnp.logical_not(hit)
            return (rng_i, st_n, nt, pos_b + 1, i + 1, rest_buf, go_new)

        rest_buf0 = jnp.zeros((total_steps, b), dtype=jnp.int32)
        carry_w0 = (rng_loop, state, first, pos_vec, 0, rest_buf0, True)
        _f = jax.lax.while_loop(cond, body, carry_w0)
        _rng, _st, _lt, _pb, n_written, rest_buf, _go = _f
        rest = rest_buf[:n_written, :].T
        out = jnp.concatenate([prompt_tokens, first, rest], axis=1)
        return _finish(out)

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
    out = jnp.concatenate([prompt_tokens, first, rest], axis=1)
    return _finish(out)
