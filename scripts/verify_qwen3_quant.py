#!/usr/bin/env python3
"""Sanity check W8 fake-quant vs bf16 for Qwen3 attention projection (no checkpoint)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kappa.qwen3.linear import project_q
from kappa.qwen3.quant import quantize_weight


def main() -> None:
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    b, t, d, nh, hdim = 2, 5, 64, 4, 16
    # float32 activations isolate W8 error (bf16 matmul noise dominates otherwise).
    x = jax.random.normal(k0, (b, t, d), dtype=jnp.float32)
    w_f32 = jax.random.normal(k1, (nh, d, hdim), dtype=jnp.float32)
    w_q = quantize_weight(w_f32, scale_dtype=jnp.float32)
    y0 = project_q(x, w_f32)
    y1 = project_q(x, w_q)
    rel = float(jnp.linalg.norm(y0 - y1) / (jnp.linalg.norm(y0) + jnp.finfo(jnp.float32).tiny))
    print(f"relative L2(project_q f32 vs w8 dequant): {rel:.6g}")
    if rel > 0.02:
        raise SystemExit(f"unexpectedly large relative error for random W8: {rel}")


if __name__ == "__main__":
    main()
