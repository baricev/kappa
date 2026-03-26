"""Default paths for local model assets — resolved by platform."""

from __future__ import annotations

import os
from pathlib import Path


def _detect_platform() -> str:
    """Best-effort JAX platform detection without forcing JAX init."""
    env = os.environ.get("JAX_PLATFORM_NAME", "").lower()
    if env:
        return env
    try:
        import jax
        return jax.devices()[0].platform
    except Exception:
        return "cpu"


_COLAB_DRIVE = Path("/content/drive/MyDrive")


def _is_colab() -> bool:
    """Detect Colab: check for the google.colab module, then fall back to drive path."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return _COLAB_DRIVE.exists()


def _default_checkpoint() -> Path:
    if _is_colab():
        return _COLAB_DRIVE / "4b"
    platform = _detect_platform()
    if platform == "mps":
        return Path("/Users/x/workspace/4b")
    else:
        return Path("/Users/x/workspace/4b")


def _default_tokenizer(repo_root: Path) -> Path:
    if _is_colab():
        return _COLAB_DRIVE / "4b" / "tokenizer.model"
    return repo_root / "tokenizer.model"


DEFAULT_GEMMA3_4B_CHECKPOINT: Path = _default_checkpoint()


def default_tokenizer_path(repo_root: Path) -> Path:
    """SentencePiece model path for the current platform."""
    return _default_tokenizer(repo_root)
