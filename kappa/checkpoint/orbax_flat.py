"""Orbax checkpoint → flat param dict. Keep Orbax usage here; API changes often."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_logger = logging.getLogger(__name__)

FlatParams = dict[str, jax.Array]

_ENV_RESTORE_CONCURRENT_GB = "KAPPA_ORBAX_RESTORE_CONCURRENT_GB"


def _effective_restore_concurrent_gb(explicit: int | None) -> int | None:
    """Orbax ``restore_concurrent_gb`` (None = library default, currently large)."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(_ENV_RESTORE_CONCURRENT_GB, "").strip()
    if not raw:
        return None
    return int(raw)


def _pytree_checkpointer(*, restore_concurrent_gb: int | None) -> ocp.Checkpointer:
    """Match :class:`PyTreeCheckpointer` handler options; optional restore byte cap."""
    gb = _effective_restore_concurrent_gb(restore_concurrent_gb)
    handler = ocp.PyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=False,
        use_compression=True,
        restore_concurrent_gb=gb,
    )
    return ocp.Checkpointer(
        handler,
        multiprocessing_options=ocp.options.MultiprocessingOptions(primary_host=0),
    )


def _restore_with_step_tree_metadata(
    ckptr: ocp.Checkpointer, path: Path, *, as_numpy_leaves: bool
) -> Any:
    """Restore via step metadata; JAX single-device sharding or numpy host leaves."""
    from orbax.checkpoint._src.metadata.tree import TreeMetadata, build_default_tree_metadata

    step = ckptr.metadata(os.fspath(path))
    item_meta = step.item_metadata
    if item_meta is None:
        raise ValueError("StepMetadata.item_metadata is None (PyTree _METADATA missing or unreadable)")
    if not isinstance(item_meta, TreeMetadata):
        raise TypeError(
            f"expected TreeMetadata for single-item PyTree checkpoint, got {type(item_meta)}"
        )
    if as_numpy_leaves:
        noop_sharding = build_default_tree_metadata(
            jax.tree.map(lambda _: None, item_meta.tree),
            use_zarr3=getattr(item_meta, "use_zarr3", False),
        )
        restore_args = ocp.checkpoint_utils.construct_restore_args(item_meta, noop_sharding)
    else:
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
        os.fspath(path),
        args=ocp.args.PyTreeRestore(item=item_meta, restore_args=restore_args),
    )


def _restore_with_inferred_structure(
    ckptr: ocp.Checkpointer, path: Path, *, as_numpy_leaves: bool
) -> Any:
    """Restore when PyTree ``_METADATA`` is missing (OCDBT + aggregate)."""
    from etils import epath

    handler = ckptr._handler  # noqa: SLF001
    directory = epath.Path(os.fspath(path))
    structure, _use_zarr3 = handler._get_internal_metadata(directory)  # noqa: SLF001
    if as_numpy_leaves:
        noop = jax.tree.map(lambda _: None, structure)
        restore_args = ocp.checkpoint_utils.construct_restore_args(structure, noop)
    else:
        devices = jax.local_devices()
        if not devices:
            raise RuntimeError("jax.local_devices() is empty; cannot restore")
        sharding = jax.sharding.SingleDeviceSharding(devices[0])
        sharding_tree = jax.tree.map(lambda _: sharding, structure)
        restore_args = ocp.checkpoint_utils.construct_restore_args(structure, sharding_tree)
    return ckptr.restore(
        os.fspath(path),
        args=ocp.args.PyTreeRestore(item=None, restore_args=restore_args),
    )


def _path_key_to_str(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for k in path:
        parts.append(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))))
    return ".".join(parts)


def flatten_pytree_to_dict(tree: Any) -> FlatParams:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_path_key_to_str(path): v for path, v in flat}


def load_orbax_pytree(
    path: str | Path,
    *,
    restore_concurrent_gb: int | None = None,
    restore_arrays_as_numpy: bool = False,
) -> Any:
    """Restore the checkpoint root as a PyTree (structure depends on how it was saved).

    Orbax may log that ``_CHECKPOINT_METADATA`` is missing; that file is optional
    step-level metadata and does not by itself break restore.

    Unless ``restore_arrays_as_numpy`` is true, we rebuild :class:`ArrayRestoreArgs`
    with :class:`~jax.sharding.SingleDeviceSharding` for ``local_devices()[0]`` so
    checkpoints saved elsewhere (e.g. MPS) work on TPU/GPU. With
    ``restore_arrays_as_numpy=True``, array leaves are read as ``numpy.ndarray`` on
    host (for :func:`jax.device_put` with a :class:`~jax.sharding.Mesh`).

    Args:
        restore_concurrent_gb: Passed to :class:`ocp.PyTreeCheckpointHandler` as
            ``restore_concurrent_gb`` to cap in-flight restore bytes (reduces parallel
            tensor materialization). Use a small value (e.g. 1--4) on TPU if restore
            hits ``RESOURCE_EXHAUSTED`` / HBM OOM. When ``None``, uses env
            ``KAPPA_ORBAX_RESTORE_CONCURRENT_GB`` if set, else Orbax default (~96 GiB
            in-flight), which can overshoot **single-device** HBM for large models.

    Note:
        Single-device JAX restore requires the **full** parameter set to fit one
        chip's HBM during Orbax read. Use ``restore_arrays_as_numpy=True`` before
        sharding with a mesh (see ``kappa.qwen3.sharding``).
    """
    path = Path(path)
    _logger.info("Restoring Orbax checkpoint from %s", path)
    ckptr = _pytree_checkpointer(restore_concurrent_gb=restore_concurrent_gb)
    eff = _effective_restore_concurrent_gb(restore_concurrent_gb)
    if eff is not None:
        _logger.info("Orbax restore_concurrent_gb=%s (caps in-flight restore bytes)", eff)
    if restore_arrays_as_numpy:
        _logger.info("Orbax restore: array leaves as numpy (host memory)")
    _step_meta_errors = (
        FileNotFoundError,
        ImportError,
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
    )
    try:
        return _restore_with_step_tree_metadata(ckptr, path, as_numpy_leaves=restore_arrays_as_numpy)
    except _step_meta_errors as e:
        _logger.info(
            "Orbax restore: step/PyTree metadata path skipped (%s: %s); trying inferred structure",
            type(e).__name__,
            e,
        )
    try:
        return _restore_with_inferred_structure(ckptr, path, as_numpy_leaves=restore_arrays_as_numpy)
    except Exception as e:
        _logger.info(
            "Orbax restore: inferred-structure path skipped (%s: %s); using default restore",
            type(e).__name__,
            e,
        )
    return ckptr.restore(os.fspath(path))


def load_gemma3_flat_params(
    path: str | Path,
    *,
    dtype: jnp.dtype | None = None,
    drop_siglip: bool = True,
    restore_concurrent_gb: int | None = None,
    restore_arrays_as_numpy: bool = False,
) -> FlatParams:
    """Load Gemma-3 dense checkpoint keys into a single flat dict (dot-separated paths).

    Adapted from the experimental ``gemma-jax`` loader; validate on your checkpoint revision.
    """
    raw = load_orbax_pytree(
        path,
        restore_concurrent_gb=restore_concurrent_gb,
        restore_arrays_as_numpy=restore_arrays_as_numpy,
    )
    flat = flatten_pytree_to_dict(raw)
    if drop_siglip:
        flat = {k: v for k, v in flat.items() if not k.startswith("SigLiPFromPatches_0")}
    flat = {k.replace("/", "."): v for k, v in flat.items()}
    if dtype is not None:
        flat = jax.tree.map(lambda x: x.astype(dtype), flat)
    return flat
