#!/usr/bin/env python3
"""Load Gemma 3 4B (Orbax checkpoint) and run greedy text generation.

Requires optional deps: ``uv pip install -e '.[inference]'`` for SentencePiece.

Example::

    source mps-python-3.13/bin/activate
    python scripts/infer_gemma3_4b.py --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Orbax checkpoint directory (default: kappa.defaults local 4B path)",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="SentencePiece model path (default: <repo>/tokenizer.model or Colab path)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (for default tokenizer path)",
    )
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument(
        "--max-cache-len",
        type=int,
        default=4096,
        help="KV cache length bound (must be >= prompt length + max new tokens)",
    )
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Parameter dtype after load",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling (greedy ignores)")
    p.add_argument(
        "--no-bos",
        action="store_true",
        help="Do not prepend Gemma BOS (id 2); for debugging only",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Wrap prompt in Gemma 3 chat turns (<start_of_turn>user / model); "
            "recommended for instruction-tuned checkpoints"
        ),
    )
    return p.parse_args()


def _load_tokenizer(path: Path):
    try:
        import sentencepiece as spm
    except ImportError as e:
        print(
            "sentencepiece is required. Install with: uv pip install -e '.[inference]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    sp = spm.SentencePieceProcessor()
    sp.load(str(path))
    return sp


def _gemma3_chat_user_model(text: str) -> str:
    """Same structure as ``gemma/gm/text/_template.PROMPT`` (ends before model completion)."""
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


def _encode_prompt(sp, text: str, *, use_bos: bool, bos_id: int, chat: bool) -> list[int]:
    """Gemma uses fixed BOS id (not necessarily ``sp.bos_id()``)."""
    payload = _gemma3_chat_user_model(text) if chat else text
    ids = sp.encode_as_ids(payload)
    return [bos_id] + ids if use_bos else ids


def _decode_generated(sp, full_ids: list[int], prompt_len: int, eos_id: int) -> str:
    """Decode only tokens after the prompt, stopping before the first EOS."""
    gen = full_ids[prompt_len:]
    if eos_id in gen:
        gen = gen[: gen.index(eos_id)]
    return sp.decode_ids(gen)


def main() -> None:
    args = _parse_args()

    from kappa.defaults import DEFAULT_GEMMA3_4B_CHECKPOINT, default_tokenizer_path

    ckpt = args.checkpoint or DEFAULT_GEMMA3_4B_CHECKPOINT
    tok_path = args.tokenizer or default_tokenizer_path(args.repo_root)

    if not ckpt.is_dir():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        raise SystemExit(1)
    if not tok_path.is_file():
        print(f"Tokenizer not found: {tok_path}", file=sys.stderr)
        raise SystemExit(1)

    import jax
    import jax.numpy as jnp

    from kappa.gemma3.generate import generate
    from kappa.gemma3.load import load_gemma3_dense_unsharded
    from kappa.gemma3.rope import build_rope_cache
    from kappa.gemma3.special_tokens import GEMMA3_BOS, GEMMA3_EOS

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    print(f"Loading checkpoint from {ckpt} (this can take a while)...", flush=True)
    cfg, params = load_gemma3_dense_unsharded(ckpt, model_size=4, dtype=dtype)
    print(f"Config: {cfg.num_layers} layers, embed_dim={cfg.embed_dim}", flush=True)

    max_len = args.max_cache_len
    rope_cache = build_rope_cache(cfg, max_seq_len=max_len)

    sp = _load_tokenizer(tok_path)
    ids = _encode_prompt(
        sp,
        args.prompt,
        use_bos=not args.no_bos,
        bos_id=GEMMA3_BOS,
        chat=args.chat,
    )
    prompt = jnp.asarray([ids], dtype=jnp.int32)
    plen = int(prompt.shape[1])
    if plen + args.max_new_tokens > max_len:
        print(
            f"--max-cache-len ({max_len}) must be >= prompt ({plen}) + max_new_tokens ({args.max_new_tokens})",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Prompt ({plen} tokens): {args.prompt!r}", flush=True)
    rng = jax.random.key(args.seed)
    out = generate(
        rng,
        prompt,
        params=params,
        cfg=cfg,
        rope_cache=rope_cache,
        max_new_tokens=args.max_new_tokens,
        max_cache_len=max_len,
        temperature=0.0,
    )
    out_ids = out[0].tolist()
    text = _decode_generated(sp, out_ids, plen, eos_id=GEMMA3_EOS)
    print("\n--- generated (continuation; stops at EOS) ---\n")
    print(text)


if __name__ == "__main__":
    main()
