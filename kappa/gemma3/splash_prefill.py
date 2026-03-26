"""TPU Splash vs dense — mirrors MaxText ``autoselected`` length threshold (default 128)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax import Array

from kappa.gemma3.attention_ops import prefill_attention_dense, repeat_kv_heads
from kappa.gemma3.masks import bool_to_additive, causal_square

# Match MaxText ``autoselected`` short-sequence dot-product branch.
_DEFAULT_SPLASH_MIN_LEN = 128


def _platform() -> str:
    try:
        return jax.devices()[0].platform
    except Exception:
        return "cpu"


def prefill_square_chunk_splash(
    q: Array,
    k: Array,
    v: Array,
    *,
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Square causal self-attention via TPU Splash (``make_splash_mha``).

    Expects ``q,k,v`` as ``[B, L, H, D]`` layout (head-major not required here;
    internally transposed to ``[B, H, L, D]``).  Query is assumed pre-scaled by
    ``query_pre_attn_scalar`` in the block layer.
    """
    del attn_logits_soft_cap  # wired when extending to Gemma soft-cap
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

    b, l, num_q, d = q.shape
    num_kv = k.shape[2]
    k = repeat_kv_heads(k, num_query_heads=num_q, num_kv_heads=num_kv)
    v = repeat_kv_heads(v, num_query_heads=num_q, num_kv_heads=num_kv)

    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))

    mask = sam.MultiHeadMask([sam.CausalMask((l, l))] * num_q)
    block_sizes = sak.BlockSizes.get_default()
    kernel = sak.make_splash_mha(
        mask=mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
        attn_logits_soft_cap=None,
    )

    def run(qb: Array, kb: Array, vb: Array) -> Array:
        out = kernel(q=qb, k=kb, v=vb, segment_ids=None)
        return out

    out_t = jax.vmap(run)(qt, kt, vt)
    return jnp.transpose(out_t, (0, 2, 1, 3))


def prefill_chunk_autoselect(
    q: Array,
    k: Array,
    v: Array,
    *,
    splash_min_len: int | None = None,
    force_dense: bool = False,
) -> Array:
    """Square chunk: dense below ``splash_min_len`` (default 128); else Splash on TPU if available.

    Set ``KAPPA_SPLASH_PREFILL=0`` to always use dense.
    """
    if os.environ.get("KAPPA_SPLASH_PREFILL", "1") == "0":
        force_dense = True
    min_len = splash_min_len if splash_min_len is not None else _DEFAULT_SPLASH_MIN_LEN
    l = q.shape[1]
    if force_dense or l < min_len or _platform() != "tpu":
        m = causal_square(l)
        mask_add = bool_to_additive(m)[None, None, :, :]
        return prefill_attention_dense(q, k, v, mask_add)
    try:
        return prefill_square_chunk_splash(q, k, v)
    except Exception:
        m = causal_square(l)
        mask_add = bool_to_additive(m)[None, None, :, :]
        return prefill_attention_dense(q, k, v, mask_add)
