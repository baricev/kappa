"""MoE FFN: router top-k + SwiGLU experts (several compute paths).

Default ``gather_einsum`` matches the historical implementation. Optional
``fixed_capacity`` (token dropping), ``ragged_jax`` (:func:`jax.lax.ragged_dot`),
and ``ragged_tokamax`` (optional ``tokamax`` package, falls back to JAX).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from kappa.qwen3.architecture import MoEImpl, Qwen3Config

# Upper bound on B*T when deriving static fixed_capacity buffer width (slots per expert).
_MOE_FIXED_CAP_MAX_BT = 8192


def _fixed_capacity_buffer_slots(cfg: Qwen3Config) -> int:
    """Compile-time per-expert buffer depth for fixed_capacity (not a JAX value)."""
    if cfg.moe_fixed_capacity_slots is not None:
        return max(1, int(cfg.moe_fixed_capacity_slots))
    k, e = cfg.num_experts_per_tok, cfg.num_experts
    if e <= 0 or k <= 0:
        return 1
    cap = cfg.moe_capacity_factor * _MOE_FIXED_CAP_MAX_BT * k / e
    return max(1, int(math.ceil(cap)))


def _route(
    x: Array,
    router_w: Array,
    *,
    num_experts: int,
    num_experts_per_tok: int,
) -> tuple[Array, Array, Array, int, int, int]:
    d = x.shape[-1]
    if router_w.shape != (d, num_experts):
        raise ValueError(f"router_w shape {router_w.shape} != ({d}, {num_experts})")
    b, t, _ = x.shape
    logits = jnp.einsum("btd,de->bte", x.astype(jnp.float32), router_w.astype(jnp.float32))
    top_logits, top_idx = jax.lax.top_k(logits, k=num_experts_per_tok)
    weights = jax.nn.softmax(top_logits, axis=-1).astype(x.dtype)
    x_flat = jnp.reshape(x, (b * t, d))
    idx = jnp.reshape(top_idx, (b * t, num_experts_per_tok))
    w = jnp.reshape(weights, (b * t, num_experts_per_tok))
    return x_flat, idx, w, b, t, d


def _slot_within_expert(sorted_e: Array) -> Array:
    """Slot index 0,1,... within each expert after sorting by expert id."""

    def step(carry: tuple[Array, Array], e: Array) -> tuple[tuple[Array, Array], Array]:
        prev_e, prev_slot = carry
        same = e == prev_e
        new_slot = jnp.where(same, prev_slot + jnp.int32(1), jnp.int32(0))
        return (e, new_slot), new_slot

    first = sorted_e[0]
    init = (first - jnp.int32(1), jnp.int32(-1))
    _, slots = jax.lax.scan(step, init, sorted_e)
    return slots


def _moe_gather_einsum(
    x_flat: Array,
    idx: Array,
    w: Array,
    ffn_gate: Array,
    ffn_up: Array,
    ffn_down: Array,
    *,
    num_experts_per_tok: int,
    b: int,
    t: int,
    d: int,
) -> Array:
    e = ffn_gate.shape[0]
    bt = b * t
    idx_flat = jnp.reshape(idx, (-1,))
    w0 = jnp.take(ffn_gate, jnp.clip(idx_flat, 0, e - 1), axis=0).reshape(
        bt, num_experts_per_tok, *ffn_gate.shape[1:]
    )
    wu = jnp.take(ffn_up, jnp.clip(idx_flat, 0, e - 1), axis=0).reshape(
        bt, num_experts_per_tok, *ffn_up.shape[1:]
    )
    wd = jnp.take(ffn_down, jnp.clip(idx_flat, 0, e - 1), axis=0).reshape(
        bt, num_experts_per_tok, *ffn_down.shape[1:]
    )
    g = jnp.einsum("bd,bkdi->bki", x_flat, w0)
    u = jnp.einsum("bd,bkdi->bki", x_flat, wu)
    h = jax.nn.silu(g) * u
    y_k = jnp.einsum("bki,bkid->bkd", h, wd)
    y = jnp.sum(y_k * w[..., None], axis=1)
    return jnp.reshape(y, (b, t, d))


def _moe_fixed_capacity(
    x_flat: Array,
    idx: Array,
    w: Array,
    ffn_gate: Array,
    ffn_up: Array,
    ffn_down: Array,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    capacity_factor: float,
    capacity_slots: int,
    b: int,
    t: int,
    d: int,
) -> Array:
    bt = b * t
    m = bt * num_experts_per_tok
    expert_flat = jnp.reshape(idx, (m,))
    token_flat = jnp.broadcast_to(jnp.arange(bt, dtype=jnp.int32)[:, None], (bt, num_experts_per_tok)).reshape(
        m,
    )
    w_flat = jnp.reshape(w, (m,))
    order = jnp.lexsort((token_flat.astype(jnp.int32), expert_flat.astype(jnp.int32)))
    sorted_e = expert_flat[order]
    sorted_x = x_flat[token_flat[order]]
    sorted_w = w_flat[order]
    slot = _slot_within_expert(sorted_e)
    cap_f = jnp.asarray(capacity_factor, dtype=jnp.float32)
    c_dyn = jnp.maximum(
        jnp.int32(1),
        jnp.round(cap_f * jnp.float32(bt * num_experts_per_tok) / jnp.float32(num_experts)).astype(jnp.int32),
    )
    c_buf = jnp.int32(capacity_slots)
    valid = jnp.logical_and(slot < c_dyn, slot < c_buf)
    e = num_experts
    buf = jnp.zeros((e, capacity_slots, d), dtype=x_flat.dtype)
    safe_s = jnp.where(valid, slot, jnp.int32(0))
    buf = buf.at[sorted_e, safe_s].add(jnp.where(valid[:, None], sorted_x, jnp.zeros_like(sorted_x)))
    g = jnp.einsum("ecd,edi->eci", buf, ffn_gate)
    u = jnp.einsum("ecd,edi->eci", buf, ffn_up)
    h = jax.nn.silu(g) * u
    y_b = jnp.einsum("eci,eid->ecd", h, ffn_down)
    y_pick = jnp.where(valid[:, None], y_b[sorted_e, safe_s], jnp.zeros((m, d), dtype=x_flat.dtype))
    inv = jnp.zeros(m, dtype=jnp.int32).at[order].set(jnp.arange(m, dtype=jnp.int32))
    y_assign = y_pick[inv]
    out_flat = jnp.zeros((bt, d), dtype=x_flat.dtype)
    out_flat = out_flat.at[token_flat].add(y_assign * w_flat[:, None])
    return jnp.reshape(out_flat, (b, t, d))


def _ragged_dot(lhs: Array, rhs: Array, group_sizes: Array, *, prefer_tokamax: bool) -> Array:
    if prefer_tokamax:
        try:
            import tokamax  # type: ignore[import-not-found]

            fn = getattr(tokamax, "ragged_dot", None)
            if callable(fn):
                return fn(lhs, rhs, group_sizes)  # type: ignore[misc]
        except Exception:
            pass
    return jax.lax.ragged_dot(lhs, rhs, group_sizes)


def _moe_ragged(
    x_flat: Array,
    idx: Array,
    w: Array,
    ffn_gate: Array,
    ffn_up: Array,
    ffn_down: Array,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    b: int,
    t: int,
    d: int,
    prefer_tokamax: bool,
) -> Array:
    bt = b * t
    m = bt * num_experts_per_tok
    expert_flat = jnp.reshape(idx, (m,))
    token_flat = jnp.broadcast_to(jnp.arange(bt, dtype=jnp.int32)[:, None], (bt, num_experts_per_tok)).reshape(
        m,
    )
    w_flat = jnp.reshape(w, (m,))
    order = jnp.argsort(expert_flat, stable=True)
    sorted_e = expert_flat[order]
    sorted_x = x_flat[token_flat[order]]
    sorted_w = w_flat[order]
    group_sizes = jnp.bincount(sorted_e, length=num_experts).astype(jnp.int32)

    lhs_f = sorted_x.astype(jnp.float32)
    wg = ffn_gate.astype(jnp.float32)
    wu = ffn_up.astype(jnp.float32)
    wd = ffn_down.astype(jnp.float32)

    g_out = _ragged_dot(lhs_f, wg, group_sizes, prefer_tokamax=prefer_tokamax)
    u_out = _ragged_dot(lhs_f, wu, group_sizes, prefer_tokamax=prefer_tokamax)
    h = jax.nn.silu(g_out) * u_out
    y_sorted = _ragged_dot(h, wd, group_sizes, prefer_tokamax=prefer_tokamax)

    inv = jnp.zeros(m, dtype=jnp.int32).at[order].set(jnp.arange(m, dtype=jnp.int32))
    y_orig = y_sorted[inv].astype(x_flat.dtype)
    out_flat = jnp.zeros((bt, d), dtype=x_flat.dtype)
    out_flat = out_flat.at[token_flat].add(y_orig * w_flat[:, None])
    return jnp.reshape(out_flat, (b, t, d))


def _effective_moe_impl(cfg: Qwen3Config, *, total_tokens: int) -> MoEImpl:
    impl = cfg.moe_impl
    if impl in ("ragged_jax", "ragged_tokamax") and total_tokens <= cfg.moe_ragged_decode_token_threshold:
        return "gather_einsum"
    return impl


def moe_swiglu_ffn(
    x: Array,
    router_w: Array,
    ffn_gate: Array,
    ffn_up: Array,
    ffn_down: Array,
    *,
    cfg: Qwen3Config,
) -> Array:
    """MoE SwiGLU forward; implementation selected by ``cfg.moe_impl`` (and decode threshold).

    Shapes: ``router_w`` ``[D, E]``; ``ffn_*`` ``[E, D, I]`` / ``[E, I, D]`` for gate/up/down.
    """
    if not cfg.use_moe:
        raise ValueError("moe_swiglu_ffn requires cfg.use_moe")
    num_experts = cfg.num_experts
    k = cfg.num_experts_per_tok
    x_flat, idx, w, b, t, d = _route(x, router_w, num_experts=num_experts, num_experts_per_tok=k)
    impl = _effective_moe_impl(cfg, total_tokens=b * t)

    if impl == "gather_einsum":
        return _moe_gather_einsum(
            x_flat, idx, w, ffn_gate, ffn_up, ffn_down, num_experts_per_tok=k, b=b, t=t, d=d
        )
    if impl == "fixed_capacity":
        if cfg.moe_capacity_factor <= 0:
            raise ValueError("fixed_capacity requires cfg.moe_capacity_factor > 0")
        slots = _fixed_capacity_buffer_slots(cfg)
        return _moe_fixed_capacity(
            x_flat,
            idx,
            w,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_experts=num_experts,
            num_experts_per_tok=k,
            capacity_factor=cfg.moe_capacity_factor,
            capacity_slots=slots,
            b=b,
            t=t,
            d=d,
        )
    if impl == "ragged_jax":
        return _moe_ragged(
            x_flat,
            idx,
            w,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_experts=num_experts,
            num_experts_per_tok=k,
            b=b,
            t=t,
            d=d,
            prefer_tokamax=False,
        )
    if impl == "ragged_tokamax":
        return _moe_ragged(
            x_flat,
            idx,
            w,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_experts=num_experts,
            num_experts_per_tok=k,
            b=b,
            t=t,
            d=d,
            prefer_tokamax=True,
        )
    raise ValueError(f"unknown moe_impl: {impl!r}")


def moe_swiglu_ffn_legacy(
    x: Array,
    router_w: Array,
    ffn_gate: Array,
    ffn_up: Array,
    ffn_down: Array,
    *,
    num_experts: int,
    num_experts_per_tok: int,
) -> Array:
    """Backward-compatible API: always uses gather/einsum (ignores config flags)."""
    x_flat, idx, w, b, t, d = _route(x, router_w, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
    return _moe_gather_einsum(
        x_flat, idx, w, ffn_gate, ffn_up, ffn_down, num_experts_per_tok=num_experts_per_tok, b=b, t=t, d=d
    )
