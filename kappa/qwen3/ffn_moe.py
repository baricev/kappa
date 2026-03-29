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
from kappa.qwen3.quant import Weight, is_q8_weight, moe_take_dequant, to_compute_dtype

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


def _expert_count(ffn_w: Weight) -> int:
    return int(ffn_w.values.shape[0] if is_q8_weight(ffn_w) else ffn_w.shape[0])


def _route(
    x: Array,
    router_w: Weight,
    *,
    num_experts: int,
    num_experts_per_tok: int,
) -> tuple[Array, Array, Array, int, int, int]:
    d = x.shape[-1]
    rw = to_compute_dtype(router_w, jnp.float32)
    if rw.shape != (d, num_experts):
        raise ValueError(f"router_w shape {rw.shape} != ({d}, {num_experts})")
    b, t, _ = x.shape
    logits = jnp.einsum("btd,de->bte", x.astype(jnp.float32), rw.astype(jnp.float32))
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
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
    *,
    num_experts_per_tok: int,
    b: int,
    t: int,
    d: int,
) -> Array:
    e = _expert_count(ffn_gate)
    bt = b * t
    idx_flat = jnp.reshape(idx, (-1,))
    w0 = moe_take_dequant(
        ffn_gate, idx_flat, e=e, num_experts_per_tok=num_experts_per_tok, bt=bt, dtype=x_flat.dtype
    )
    wu = moe_take_dequant(
        ffn_up, idx_flat, e=e, num_experts_per_tok=num_experts_per_tok, bt=bt, dtype=x_flat.dtype
    )
    wd = moe_take_dequant(
        ffn_down, idx_flat, e=e, num_experts_per_tok=num_experts_per_tok, bt=bt, dtype=x_flat.dtype
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
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
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
    gate_f = to_compute_dtype(ffn_gate, x_flat.dtype)
    up_f = to_compute_dtype(ffn_up, x_flat.dtype)
    down_f = to_compute_dtype(ffn_down, x_flat.dtype)
    g = jnp.einsum("ecd,edi->eci", buf, gate_f)
    u = jnp.einsum("ecd,edi->eci", buf, up_f)
    h = jax.nn.silu(g) * u
    y_b = jnp.einsum("eci,eid->ecd", h, down_f)
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
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
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
    wg = to_compute_dtype(ffn_gate, jnp.float32)
    wu = to_compute_dtype(ffn_up, jnp.float32)
    wd = to_compute_dtype(ffn_down, jnp.float32)

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
    router_w: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
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
    router_w: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
    *,
    num_experts: int,
    num_experts_per_tok: int,
) -> Array:
    """Backward-compatible API: always uses gather/einsum (ignores config flags)."""
    x_flat, idx, w, b, t, d = _route(x, router_w, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
    return _moe_gather_einsum(
        x_flat, idx, w, ffn_gate, ffn_up, ffn_down, num_experts_per_tok=num_experts_per_tok, b=b, t=t, d=d
    )
