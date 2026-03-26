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


def _path_key_to_str(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for k in path:
        parts.append(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))))
    return ".".join(parts)


def flatten_pytree_to_dict(tree: Any) -> FlatParams:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_path_key_to_str(path): v for path, v in flat}


def load_orbax_pytree(path: str | Path) -> Any:
    """Restore the checkpoint root as a PyTree (structure depends on how it was saved)."""
    path = Path(path)
    _logger.info("Restoring Orbax checkpoint from %s", path)
    return ocp.PyTreeCheckpointer().restore(str(path))


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
