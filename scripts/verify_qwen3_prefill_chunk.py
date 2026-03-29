#!/usr/bin/env python3
"""Assert one-shot prefill matches chunked prefill (last-token logits + KV lengths).

  source mps-python-3.13/bin/activate && python scripts/verify_qwen3_prefill_chunk.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kappa.qwen3.architecture import Qwen3Config
from kappa.qwen3.rope import build_qwen3_rope_cache
from kappa.qwen3.transformer import (
    forward_prefill,
    forward_prefill_chunk,
    init_qwen3_inference_state,
)
from kappa.qwen3.weights import AttnParams, DenseFfnParams, Qwen3DenseBlockParams, Qwen3Params


def _tiny_cfg() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        model_dim=64,
        intermediate_size=128,
        rope_theta=10_000.0,
        rms_eps=1e-6,
        use_tied_embedding=True,
        attn_logits_soft_cap=None,
        use_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        pad_token_id=0,
        eos_token_id=1,
    )


def _random_params(key: jax.Array, cfg: Qwen3Config) -> Qwen3Params:
    keys = jax.random.split(key, 50)
    ki = 0
    d, h, dh = cfg.model_dim, cfg.num_heads, cfg.head_dim
    nkv = cfg.num_kv_heads
    iq = cfg.intermediate_size
    v = cfg.vocab_size

    def take():
        nonlocal ki
        k = keys[ki]
        ki += 1
        return k

    embed = jax.random.normal(take(), (v, d), dtype=jnp.float32) * 0.02
    final_norm = jnp.ones((d,), dtype=jnp.float32)

    blocks: list[Qwen3DenseBlockParams] = []
    for _ in range(cfg.num_layers):
        pre0 = jnp.ones((d,), dtype=jnp.float32)
        pre1 = jnp.ones((d,), dtype=jnp.float32)
        attn = AttnParams(
            q_proj=jax.random.normal(take(), (h, d, dh), dtype=jnp.float32) * 0.02,
            k_proj=jax.random.normal(take(), (nkv, d, dh), dtype=jnp.float32) * 0.02,
            v_proj=jax.random.normal(take(), (nkv, d, dh), dtype=jnp.float32) * 0.02,
            o_proj=jax.random.normal(take(), (h, dh, d), dtype=jnp.float32) * 0.02,
            q_norm_scale=jnp.ones((dh,), dtype=jnp.float32),
            k_norm_scale=jnp.ones((dh,), dtype=jnp.float32),
        )
        ffn = DenseFfnParams(
            gate_proj=jax.random.normal(take(), (d, iq), dtype=jnp.float32) * 0.02,
            up_proj=jax.random.normal(take(), (d, iq), dtype=jnp.float32) * 0.02,
            down_proj=jax.random.normal(take(), (iq, d), dtype=jnp.float32) * 0.02,
        )
        blocks.append(Qwen3DenseBlockParams(pre0, pre1, attn, ffn))
    return Qwen3Params(embed=embed, final_norm=final_norm, blocks=tuple(blocks), lm_head=None)


def main() -> None:
    jax.config.update("jax_enable_x64", False)
    cfg = _tiny_cfg()
    key = jax.random.PRNGKey(0)
    params = _random_params(key, cfg)
    max_len = 256
    rope_cache = build_qwen3_rope_cache(cfg, max_seq_len=max_len)

    plen = 37
    fc = 16
    pad = cfg.pad_token_id
    prompt = jax.random.randint(key, (1, plen), 1, cfg.vocab_size, dtype=jnp.int32)
    valid = jnp.ones((1, plen), dtype=jnp.bool_)
    pos0 = jnp.arange(plen, dtype=jnp.int32)[None, :]

    _, log1, st1 = forward_prefill(
        prompt,
        pos0,
        params=params,
        cfg=cfg,
        rope_cache=rope_cache,
        token_valid_mask=valid,
        max_len=max_len,
        last_logits_only=True,
    )

    state0 = init_qwen3_inference_state(cfg, batch=1, max_len=max_len, dtype=params.embed.dtype)
    offset = 0
    logits_c = None
    st_c = state0
    while offset < plen:
        lc_read = min(fc, plen - offset)
        raw = prompt[:, offset : offset + lc_read]
        vraw = valid[:, offset : offset + lc_read]
        if lc_read < fc:
            pad_tail = jnp.full((1, fc - lc_read), pad, dtype=prompt.dtype)
            chunk = jnp.concatenate([raw, pad_tail], axis=1)
            vpad = jnp.zeros((1, fc - lc_read), dtype=jnp.bool_)
            valid_chunk = jnp.concatenate([vraw, vpad], axis=1)
        else:
            chunk = raw
            valid_chunk = vraw
        true_lens = jnp.sum(valid_chunk, axis=1).astype(jnp.int32)
        last_chunk = offset + fc >= plen
        _, logits_c, st_c = forward_prefill_chunk(
            chunk,
            params=params,
            cfg=cfg,
            rope_cache=rope_cache,
            state=st_c,
            start_pos=offset,
            true_chunk_lens=true_lens,
            last_logits_only=last_chunk,
        )
        offset += fc

    err_l = jnp.max(jnp.abs(log1 - logits_c)).item()
    assert err_l < 1e-4, f"logits mismatch {err_l}"

    for i in range(cfg.num_layers):
        err_k = jnp.max(jnp.abs(st1.kv[i].k - st_c.kv[i].k)).item()
        err_v = jnp.max(jnp.abs(st1.kv[i].v - st_c.kv[i].v)).item()
        assert err_k < 1e-4, f"layer {i} k mismatch {err_k}"
        assert err_v < 1e-4, f"layer {i} v mismatch {err_v}"
        assert jnp.all(st1.kv[i].lengths == st_c.kv[i].lengths), f"layer {i} lengths"

    print("verify_qwen3_prefill_chunk: ok")


if __name__ == "__main__":
    main()
