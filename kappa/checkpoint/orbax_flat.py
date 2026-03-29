"""Orbax checkpoint → flat param dict. Keep Orbax usage here; API changes often."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_logger = logging.getLogger(__name__)

FlatParams = dict[str, jax.Array]


def _restore_pytree_onto_local_devices(ckptr: ocp.PyTreeCheckpointer, path: Path) -> Any:
    """Restore arrays onto ``jax.local_devices()[0]``, ignoring saved shardings.

    Checkpoints saved on another backend (e.g. MPS) embed device ids in sharding
    metadata; default Orbax restore then fails on TPU/GPU with "Device MPS:0 was
    not found". We rebuild :class:`ArrayRestoreArgs` with a local
    :class:`~jax.sharding.SingleDeviceSharding`.
    """
    from orbax.checkpoint._src.metadata.tree import TreeMetadata, build_default_tree_metadata

    step = ckptr.metadata(str(path))
    item_meta = step.item_metadata
    if item_meta is None:
        raise ValueError("checkpoint StepMetadata.item_metadata is None")
    if not isinstance(item_meta, TreeMetadata):
        raise TypeError(
            f"expected TreeMetadata for single-item PyTree checkpoint, got {type(item_meta)}"
        )
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError("jax.local_devices() is empty; cannot restore")
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
    sharding_tm = build_default_tree_metadata(
        jax.tree.map(lambda _: sharding, item_meta.tree),
        use_zarr3=getattr(item_meta, "use_zarr3", False),
    )
    restore_args = ocp.checkpoint_utils.construct_restore_args(item_meta, sharding_tm)
    return ckptr.restore(
        str(path),
        args=ocp.args.PyTreeRestore(item=item_meta, restore_args=restore_args),
    )


def _path_key_to_str(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for k in path:
        parts.append(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))))
    return ".".join(parts)


def flatten_pytree_to_dict(tree: Any) -> FlatParams:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_path_key_to_str(path): v for path, v in flat}


def load_orbax_pytree(path: str | Path) -> Any:
    """Restore the checkpoint root as a PyTree (structure depends on how it was saved).

    Uses explicit local-device shardings when checkpoint metadata is available so
    restores work across backends (e.g. MPS-saved weights on TPU).
    """
    path = Path(path)
    _logger.info("Restoring Orbax checkpoint from %s", path)
    ckptr = ocp.PyTreeCheckpointer()
    try:
        return _restore_pytree_onto_local_devices(ckptr, path)
    except (
        FileNotFoundError,
        ImportError,
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
    ) as e:
        _logger.info("Orbax default restore path (%s)", e)
        return ckptr.restore(str(path))


def load_gemma3_flat_params(
    path: str | Path,
    *,
    dtype: jnp.dtype | None = None,
    drop_siglip: bool = True,
) -> FlatParams:
    """Load Gemma-3 dense checkpoint keys into a single flat dict (dot-separated paths).

    Adapted from the experimental ``gemma-jax`` loader; validate on your checkpoint revision.
    """
    raw = load_orbax_pytree(path)
    flat = flatten_pytree_to_dict(raw)
    if drop_siglip:
        flat = {k: v for k, v in flat.items() if not k.startswith("SigLiPFromPatches_0")}
    flat = {k.replace("/", "."): v for k, v in flat.items()}
    if dtype is not None:
        flat = jax.tree.map(lambda x: x.astype(dtype), flat)
    return flat
