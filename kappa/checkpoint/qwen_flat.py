"""Orbax / flat dict loading for Simply ``Qwen2Format``-style checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from kappa.checkpoint.orbax_flat import flatten_pytree_to_dict, load_orbax_pytree

_logger = logging.getLogger(__name__)

FlatParams = dict[str, jax.Array]


def load_qwen3_flat_params(
    path: str | Path,
    *,
    dtype: jnp.dtype | None = None,
) -> FlatParams:
    """Load checkpoint root and flatten to dot-separated string keys.

    Compatible with checkpoints produced via Simply's ``hf_to_orbax`` + ``Qwen2Format``
    (plus ``LegacyFormat`` transforms).
    """
    raw = load_orbax_pytree(path)
    flat = flatten_pytree_to_dict(raw)
    flat = _normalize_flat_keys(flat)
    if dtype is not None:
        flat = {k: v.astype(dtype) for k, v in flat.items()}
    return flat


def _normalize_flat_keys(flat: FlatParams) -> FlatParams:
    """Normalize ``/`` vs ``.`` and optional ``state`` / ``params`` wrappers."""
    out: FlatParams = {}
    for k, v in flat.items():
        key = k.replace("/", ".")
        # Drop common single-segment prefixes from some Orbax trees
        for prefix in ("state.", "data."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = v
    return out


def flat_get(flat: FlatParams, key: str, default: Any = None) -> Any:
    if key in flat:
        return flat[key]
    return default
