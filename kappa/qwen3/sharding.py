"""Multi-device :class:`~jax.sharding.Mesh` + :class:`~jax.sharding.PartitionSpec` for Qwen3.

Mirrors the pattern in repo-root ``weights.py`` (Gemma): build a parameter PyTree, a matching
:class:`PartitionSpec` PyTree, then :func:`jax.device_put` with :class:`~jax.sharding.NamedSharding`.

**MoE**: expert tensors and router are left **replicated** (``P()``). The current MoE forward
(:func:`~kappa.qwen3.ffn_moe.moe_swiglu_ffn`) assumes full expert weights on each device;
expert parallelism would need a different compute path. Tensor-parallel sharding applies to
embeddings, attention, and dense FFN only, which still reduces HBM vs fully replicated MoE
when combined with numpy Orbax restore + sharded ``device_put``.

**Correctness**: With SPMD, JAX inserts collectives for mixed shardings in einsums where
possible. Validate on your topology; single-device mesh is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from kappa.qwen3.architecture import Qwen3Config
from kappa.qwen3.quant import Q8Weight
from kappa.qwen3.weights import (
    AttnParams,
    DenseFfnParams,
    MoEFfnParams,
    Qwen3DenseBlockParams,
    Qwen3MoEBlockParams,
    Qwen3Params,
    quantize_qwen3_params,
)

_logger = logging.getLogger(__name__)


def create_qwen3_device_mesh(
    shape: tuple[int | None, int | None] | None = None,
    *,
    axis_names: tuple[str, str] = ("data", "tensor"),
) -> Mesh:
    """2D device mesh; defaults to ``(1, N)`` so ``P(..., 'tensor', ...)`` spans all devices.

    Same layout idea as ``weights.py`` ``create_device_mesh``: first axis is unused for pure
    tensor-parallel sharding in the specs below.
    """
    devs = jax.devices()
    if {d.platform for d in devs} == {"cpu"}:
        return Mesh(np.array(devs).reshape((1, 1)), axis_names=axis_names)

    n = len(devs)
    if shape is None or shape == (None, None):
        data, tensor = 1, n
    else:
        data, tensor = shape[0], shape[1]
        if data is None and tensor is None:
            data, tensor = 1, n
        elif data is None:
            if tensor is None or n % tensor:
                raise ValueError(f"{n} devices cannot form mesh (_, {tensor})")
            data = n // tensor
        elif tensor is None:
            if n % data:
                raise ValueError(f"{n} devices cannot form mesh ({data}, _)")
            tensor = n // data
        if data * tensor > n:
            raise ValueError("mesh product exceeds device count")

    arr = np.array(devs[: data * tensor]).reshape((data, tensor))
    mesh = Mesh(arr, axis_names=axis_names)
    _logger.info(
        "Qwen3 mesh shape %s axis_names=%s on %d devices",
        mesh.devices.shape,
        axis_names,
        n,
    )
    return mesh


def _axis_name_tensor(mesh: Mesh) -> str:
    return mesh.axis_names[1] if len(mesh.axis_names) > 1 else mesh.axis_names[0]


def _shard_leading_if_divisible(mesh: Mesh, size: int, rank: int) -> P:
    """``P(tensor, None, ...)`` on first dim if ``size`` divisible by tensor axis length."""
    if rank <= 0:
        return P()
    t_axis = 1 if mesh.devices.ndim > 1 else 0
    n = int(mesh.devices.shape[t_axis])
    if size % n != 0:
        return P()
    name = mesh.axis_names[t_axis]
    rest = (None,) * (rank - 1)
    return P(name, *rest)


def qwen3_params_pspec(cfg: Qwen3Config, mesh: Mesh) -> Qwen3Params:
    """PartitionSpec tree matching :class:`Qwen3Params` (see module doc for MoE caveats)."""
    t = _axis_name_tensor(mesh)
    _rep = P()
    q8 = cfg.quantization == "w8"

    def wps(ps: P) -> P | Q8Weight:
        if not q8:
            return ps
        return Q8Weight(values=ps, scale=_rep)

    nh = cfg.num_heads
    nkv = cfg.num_kv_heads

    def attn_pspec() -> AttnParams:
        return AttnParams(
            wps(_shard_leading_if_divisible(mesh, nh, 3)),
            wps(_shard_leading_if_divisible(mesh, nkv, 3)),
            wps(_shard_leading_if_divisible(mesh, nkv, 3)),
            wps(_shard_leading_if_divisible(mesh, nh, 3)),
            _rep,
            _rep,
        )

    def dense_ffn_pspec() -> DenseFfnParams:
        return DenseFfnParams(
            wps(P(None, t)),  # [model_dim, intermediate] shard intermediate
            wps(P(None, t)),
            wps(P(t, None)),
        )

    def moe_ffn_pspec() -> MoEFfnParams:
        return MoEFfnParams(router=wps(_rep), gate_proj=wps(_rep), up_proj=wps(_rep), down_proj=wps(_rep))

    blocks: list[Qwen3DenseBlockParams | Qwen3MoEBlockParams] = []
    for _ in range(cfg.num_layers):
        ap = attn_pspec()
        if cfg.use_moe:
            blocks.append(Qwen3MoEBlockParams(_rep, _rep, ap, moe_ffn_pspec()))
        else:
            blocks.append(Qwen3DenseBlockParams(_rep, _rep, ap, dense_ffn_pspec()))

    vocab = cfg.vocab_size
    emb_ps = wps(_shard_leading_if_divisible(mesh, vocab, 2))
    lm_ps: P | Q8Weight | None = None if cfg.use_tied_embedding else wps(_shard_leading_if_divisible(mesh, vocab, 2))

    return Qwen3Params(
        embed=emb_ps,
        final_norm=_rep,
        blocks=tuple(blocks),
        lm_head=lm_ps,
    )


def device_put_qwen3_params(params: Qwen3Params, pspec: Qwen3Params, mesh: Mesh) -> Qwen3Params:
    """``jax.device_put`` the PyTree with a tree of :class:`~jax.sharding.NamedSharding` (see ``weights.py``)."""

    def to_named_sharding(s: Any) -> Any:
        if s is None:
            return None
        return NamedSharding(mesh, s)

    # PartitionSpec must stay leaves (do not decompose as tuple nodes).
    def _is_pspec_leaf(x: Any) -> bool:
        return x is None or isinstance(x, P)

    sharding_tree = jax.tree.map(to_named_sharding, pspec, is_leaf=_is_pspec_leaf)
    return jax.device_put(params, sharding_tree)


def load_qwen3_sharded(
    checkpoint_dir: str | Path,
    mesh: Mesh,
    *,
    cfg: Qwen3Config,
    dtype: jnp.dtype | None = None,
    restore_concurrent_gb: int | None = None,
) -> Qwen3Params:
    """Orbax → flat (numpy host) → :class:`Qwen3Params` → sharded ``device_put`` inside ``mesh``."""
    from kappa.checkpoint.qwen_flat import load_qwen3_flat_params
    from kappa.qwen3.weights import params_from_flat

    path = Path(checkpoint_dir).expanduser().resolve()
    flat = load_qwen3_flat_params(
        path,
        dtype=dtype,
        restore_concurrent_gb=restore_concurrent_gb,
        restore_arrays_as_numpy=True,
    )
    params = params_from_flat(flat, num_layers=cfg.num_layers, use_moe=cfg.use_moe)
    if cfg.quantization == "w8":
        sd = dtype if dtype is not None else jnp.bfloat16
        params = quantize_qwen3_params(params, scale_dtype=sd)
    pspec = qwen3_params_pspec(cfg, mesh)
    with mesh:
        out = device_put_qwen3_params(params, pspec, mesh)
        leaves = [x for x in jax.tree.leaves(out) if x is not None]
        if leaves:
            leaves[0].block_until_ready()
    return out
