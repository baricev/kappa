#!/usr/bin/env python3
"""Forward-only sanity: ``gather_einsum`` vs ``ragged_jax`` vs ``fixed_capacity`` (high cap).

Run from repo root:
  source mps-python-3.13/bin/activate && python scripts/verify_qwen3_moe_impls.py
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from kappa.qwen3.architecture import Qwen3Config
from kappa.qwen3.ffn_moe import moe_swiglu_ffn, moe_swiglu_ffn_legacy


def main() -> None:
    jax.config.update("jax_enable_x64", False)
    key = jax.random.PRNGKey(0)
    b, t, d, e, k, i = 2, 3, 16, 8, 2, 32
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x = jax.random.normal(k1, (b, t, d), dtype=jnp.float32)
    router_w = jax.random.normal(k2, (d, e), dtype=jnp.float32) * 0.02
    ffn_gate = jax.random.normal(k3, (e, d, i), dtype=jnp.float32) * 0.02
    ffn_up = jax.random.normal(k4, (e, d, i), dtype=jnp.float32) * 0.02
    ffn_down = jax.random.normal(k5, (e, i, d), dtype=jnp.float32) * 0.02

    base = dataclasses.replace(
        Qwen3Config(
            vocab_size=128,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            model_dim=d,
            intermediate_size=i,
            rope_theta=10_000.0,
            rms_eps=1e-6,
            use_tied_embedding=True,
            attn_logits_soft_cap=None,
            use_moe=True,
            num_experts=e,
            num_experts_per_tok=k,
            pad_token_id=0,
            eos_token_id=1,
        ),
        moe_impl="gather_einsum",
        moe_ragged_decode_token_threshold=0,
    )

    y_ref = moe_swiglu_ffn(x, router_w, ffn_gate, ffn_up, ffn_down, cfg=base)
    y_legacy = moe_swiglu_ffn_legacy(
        x, router_w, ffn_gate, ffn_up, ffn_down, num_experts=e, num_experts_per_tok=k
    )
    err_legacy = jnp.max(jnp.abs(y_ref - y_legacy)).item()
    assert err_legacy < 1e-5, err_legacy

    y_ragged = moe_swiglu_ffn(
        x,
        router_w,
        ffn_gate,
        ffn_up,
        ffn_down,
        cfg=dataclasses.replace(base, moe_impl="ragged_jax"),
    )
    err_r = jnp.max(jnp.abs(y_ref - y_ragged)).item()
    assert err_r < 1e-4, f"ragged_jax vs gather max abs err {err_r}"

    # Large enough capacity so nothing is dropped vs dropless reference
    y_cap = moe_swiglu_ffn(
        x,
        router_w,
        ffn_gate,
        ffn_up,
        ffn_down,
        cfg=dataclasses.replace(
            base,
            moe_impl="fixed_capacity",
            moe_capacity_factor=1_000.0,
            moe_ragged_decode_token_threshold=0,
        ),
    )
    err_c = jnp.max(jnp.abs(y_ref - y_cap)).item()
    assert err_c < 1e-4, f"fixed_capacity (high cap) vs gather max abs err {err_c}"

    @jax.jit
    def run_ragged(x_):
        return moe_swiglu_ffn(
            x_,
            router_w,
            ffn_gate,
            ffn_up,
            ffn_down,
            cfg=dataclasses.replace(base, moe_impl="ragged_jax"),
        )

    y_jit = run_ragged(x)
    err_j = jnp.max(jnp.abs(y_ragged - y_jit)).item()
    assert err_j < 1e-6, err_j

    print("verify_qwen3_moe_impls: ok (legacy, ragged_jax, fixed_capacity high-cap, jit)")


if __name__ == "__main__":
    main()
