"""Hugging Face–style Qwen tokenizer (``tokenizer.json`` + BPE), Simply ``HuggingFaceVocab``-aligned."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast


@dataclass(frozen=True, slots=True)
class QwenHfTokenizer:
    """Loads Rust ``tokenizers`` from ``tokenizer.json``; specials from ``tokenizer_config.json``."""

    _tok: object  # tokenizers.Tokenizer
    pad_id: int | None
    bos_id: int | None
    eos_id: int | None

    @classmethod
    def from_directory(cls, path: str | Path) -> QwenHfTokenizer:
        """``path`` is a folder containing ``tokenizer.json`` (and usually ``tokenizer_config.json``)."""
        root = Path(path).expanduser().resolve()
        return cls.from_tokenizer_json(root / "tokenizer.json")

    @classmethod
    def from_tokenizer_json(cls, tokenizer_json: str | Path) -> QwenHfTokenizer:
        """Load from a single ``tokenizer.json`` file (parent folder must hold ``tokenizer_config.json`` if present)."""
        try:
            import tokenizers
        except ImportError as e:
            raise ImportError(
                "The `tokenizers` package is required for Qwen HF tokenizers. "
                "Install with: uv pip install -e '.[inference]'"
            ) from e

        p = Path(tokenizer_json).expanduser().resolve()
        raw = p.read_bytes()
        tok = tokenizers.Tokenizer.from_buffer(raw)
        config_path = p.parent / "tokenizer_config.json"
        config: Mapping[str, Any] = {}
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                config = json.load(f)

        def _tid(name: str) -> int | None:
            if name not in config:
                return None
            token = config[name]
            if token is None:
                return None
            if isinstance(token, str):
                s = token
            elif isinstance(token, dict) and "content" in token:
                s = token["content"]
            else:
                raise ValueError(f"unexpected {name} entry: {token!r}")
            if not isinstance(s, str):
                raise ValueError(f"{name} content is not a string: {token!r}")
            tid = tok.token_to_id(s)
            return int(tid) if tid is not None else None

        return cls(
            _tok=tok,
            pad_id=_tid("pad_token"),
            bos_id=_tid("bos_token"),
            eos_id=_tid("eos_token"),
        )

    def encode(self, text: str) -> list[int]:
        enc = self._tok.encode(text)
        return cast(Any, enc).ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return cast(Any, self._tok).decode(token_ids, skip_special_tokens=skip_special_tokens)
