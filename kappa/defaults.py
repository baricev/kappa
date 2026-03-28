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


def _default_qwen3_checkpoint(subdir: str) -> Path:
    if _is_colab():
        return _COLAB_DRIVE / subdir
    platform = _detect_platform()
    if platform == "mps":
        return Path(f"/Users/x/workspace/{subdir}")
    return Path(f"/Users/x/workspace/{subdir}")


# Orbax roots after HF→Simply conversion (see ``third_party/simply`` ``hf_to_orbax``).
DEFAULT_QWEN3_0P6B_CHECKPOINT: Path = _default_qwen3_checkpoint("Qwen3-0.6B/ORBAX")
DEFAULT_QWEN3_4B_CHECKPOINT: Path = _default_qwen3_checkpoint("Qwen3-4B/ORBAX")
DEFAULT_QWEN3_30B_A3B_CHECKPOINT: Path = _default_qwen3_checkpoint("Qwen3-30B-A3B/ORBAX")


def default_qwen3_tokenizer_dir() -> Path:
    """HF tokenizer folder (must contain ``tokenizer.json``): Hugging Face snapshot layout."""
    if _is_colab():
        return _COLAB_DRIVE / "qwen3-0.6b"
    return Path.home() / "workspace" / "qwen3-0.6b"


def default_qwen3_tokenizer_path(repo_root: Path) -> Path:
    """Deprecated name: use :func:`default_qwen3_tokenizer_dir`. ``repo_root`` ignored."""
    del repo_root
    return default_qwen3_tokenizer_dir()


def default_tokenizer_path(repo_root: Path) -> Path:
    """SentencePiece model path for the current platform."""
    return _default_tokenizer(repo_root)
