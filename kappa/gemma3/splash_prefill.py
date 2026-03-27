"""TPU Splash vs dense — mirrors MaxText ``autoselected`` length threshold (default 128)."""

from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
from jax import Array

from kappa.gemma3.attention_ops import prefill_attention_dense, repeat_kv_heads
from kappa.gemma3.masks import bool_to_additive, causal_square

# Match MaxText ``autoselected`` short-sequence dot-product branch.
_DEFAULT_SPLASH_MIN_LEN = 128

_logger = logging.getLogger(__name__)


def _splash_q_block_size() -> int:
    """Default Splash block size; ``make_splash_mha`` requires ``q_seq_len`` divisible by this."""
    try:
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak

        bs = sak.BlockSizes.get_default()
        return int(getattr(bs, "q_block_size", 128))
    except Exception:
        return 128


def splash_square_q_len_ok(seq_len: int) -> bool:
    """Whether square Splash prefill is valid at this length (TPU kernel divisor rule)."""
    if seq_len <= 0:
        return False
    qbs = _splash_q_block_size()
    return seq_len % qbs == 0


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

    Attention logit soft-capping is not implemented in the Splash kernel; callers must use dense
    (``prefill_chunk_autoselect`` forces dense when ``attn_logits_soft_cap`` is set).
    """
    if attn_logits_soft_cap is not None:
        raise ValueError(
            "prefill_square_chunk_splash does not support attn_logits_soft_cap; use dense attention."
        )
    b, l, num_q, d = q.shape
    qbs = _splash_q_block_size()
    if l % qbs != 0:
        raise ValueError(
            f"Splash square prefill requires q_seq_len ({l}) divisible by q_block_size ({qbs}); "
            "use prefill_chunk_autoselect (dense fallback) or pad the sequence."
        )
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

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
    attn_logits_soft_cap: float | None = None,
) -> Array:
    """Square chunk: dense below ``splash_min_len`` (default 128); else Splash on TPU if available.

    Splash also requires ``q_seq_len`` divisible by the kernel ``q_block_size`` (typically 128); odd
    lengths use dense without attempting Splash.

    Set ``KAPPA_SPLASH_PREFILL=0`` to always use dense.

    When ``attn_logits_soft_cap`` is not ``None``, always uses dense attention (Splash kernel has no
    soft-cap path).
    """
    if attn_logits_soft_cap is not None:
        force_dense = True
    if os.environ.get("KAPPA_SPLASH_PREFILL", "1") == "0":
        force_dense = True
    min_len = splash_min_len if splash_min_len is not None else _DEFAULT_SPLASH_MIN_LEN
    l = q.shape[1]
    if (
        force_dense
        or l < min_len
        or _platform() != "tpu"
        or not splash_square_q_len_ok(l)
    ):
        m = causal_square(l)
        mask_add = bool_to_additive(m)[None, None, :, :]
        return prefill_attention_dense(q, k, v, mask_add, attn_logits_soft_cap=attn_logits_soft_cap)
    try:
        return prefill_square_chunk_splash(q, k, v, attn_logits_soft_cap=None)
    except Exception as e:
        _logger.warning("Splash prefill failed; falling back to dense: %s", e, exc_info=True)
        m = causal_square(l)
        mask_add = bool_to_additive(m)[None, None, :, :]
        return prefill_attention_dense(q, k, v, mask_add, attn_logits_soft_cap=attn_logits_soft_cap)
