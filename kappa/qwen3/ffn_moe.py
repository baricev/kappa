"""MoE FFN: router top-k + softmax on logits (Simply ``MoEFeedForward`` when k>1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def moe_swiglu_ffn(
    x: Array,
    router_w: Array,
    ffn_gate: Array,
    ffn_up: Array,
    ffn_down: Array,
    *,
    num_experts: int,
    num_experts_per_tok: int,
) -> Array:
    """Sparse MoE forward (dropless).

    Shapes: ``router_w`` ``[D, E]``; ``ffn_*`` ``[E, D, I]`` / ``[E, I, D]`` for gate/up/down.
    """
    d = x.shape[-1]
    if router_w.shape != (d, num_experts):
        raise ValueError(f"router_w shape {router_w.shape} != ({d}, {num_experts})")
    logits = jnp.einsum("btd,de->bte", x.astype(jnp.float32), router_w.astype(jnp.float32))
    top_logits, top_idx = jax.lax.top_k(logits, k=num_experts_per_tok)
    weights = jax.nn.softmax(top_logits, axis=-1).astype(x.dtype)

    # [B*T, D], [B*T, k], [B*T, k]
    b, t, _ = x.shape
    x_flat = jnp.reshape(x, (b * t, d))
    idx = jnp.reshape(top_idx, (b * t, num_experts_per_tok))
    w = jnp.reshape(weights, (b * t, num_experts_per_tok))

    idx_flat = jnp.reshape(idx, (-1,))
    e = ffn_gate.shape[0]
    w0 = jnp.take(ffn_gate, jnp.clip(idx_flat, 0, e - 1), axis=0).reshape(
        b * t, num_experts_per_tok, *ffn_gate.shape[1:]
    )
    wu = jnp.take(ffn_up, jnp.clip(idx_flat, 0, e - 1), axis=0).reshape(
        b * t, num_experts_per_tok, *ffn_up.shape[1:]
    )
    wd = jnp.take(ffn_down, jnp.clip(idx_flat, 0, e - 1), axis=0).reshape(
        b * t, num_experts_per_tok, *ffn_down.shape[1:]
    )
    # [BT, k, I]
    g = jnp.einsum("bd,bkdi->bki", x_flat, w0)
    u = jnp.einsum("bd,bkdi->bki", x_flat, wu)
    h = jax.nn.silu(g) * u
    # [BT, k, D]
    y_k = jnp.einsum("bki,bkid->bkd", h, wd)
    y = jnp.sum(y_k * w[..., None], axis=1)
    return jnp.reshape(y, (b, t, d))
