#!/usr/bin/env python3
"""Load Qwen3 (Orbax checkpoint in Simply ``Qwen2Format`` layout) and run greedy generation.

Uses the Hugging Face ``tokenizer.json`` (Rust ``tokenizers``) pipeline — same idea as Simply's
``HuggingFaceVocab`` / MaxText ``tokenizer_type=huggingface``.

Requires optional deps::

    uv pip install -e '.[inference]'

Example::

    # --model must match the checkpoint (default is qwen3-0.6b only).
    python scripts/infer_qwen3.py --model qwen3-4b --checkpoint ~/workspace/qwen3-4b/ORBAX/1/state \\
      --tokenizer ~/workspace/qwen3-4b
    python scripts/infer_qwen3.py --model qwen3-0.6b --tokenizer ~/workspace/qwen3-0.6b
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default="qwen3-0.6b",
        choices=("qwen3-0.6b", "qwen3-4b", "qwen3-30b-a3b"),
        help="Must match the Orbax checkpoint (embed dim + layer count). Default is 0.6B only.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Orbax checkpoint directory (default: kappa.defaults for preset)",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="HF tokenizer: directory with tokenizer.json, or path to tokenizer.json",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Unused (legacy); default tokenizer dir is ~/.cache or ~/workspace",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Give me a short recipe for banana bread.",
        help="Prompt string (ignored if --prompt-file is set)",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read prompt text from this file (UTF-8); overrides --prompt",
    )
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
        "--decode-chunk-size",
        type=int,
        default=128,
        metavar="N",
        help="Split decode lax.scan into chunks of N steps (0 = single scan).",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Do not stop at EOS (run up to --max-new-tokens every time).",
    )
    p.add_argument(
        "--timings",
        action="store_true",
        help="Print prefill / TTFT / decode / total seconds and tok/s.",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="Wrap prompt in Qwen3 chat turns (im_start/im_end).",
    )
    p.add_argument(
        "--moe-impl",
        type=str,
        default="gather_einsum",
        choices=("gather_einsum", "fixed_capacity", "ragged_jax", "ragged_tokamax"),
        help="MoE expert path for MoE presets (dense models ignore this).",
    )
    p.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        metavar="N",
        help="If N>0 and prompt length > N, prefill in chunks of N (uniform batch lengths or B=1).",
    )
    p.add_argument(
        "--orbax-restore-concurrent-gb",
        type=int,
        default=None,
        metavar="GB",
        help="Orbax PyTree restore in-flight byte cap (GiB). Lower (e.g. 1--4) if TPU HBM OOM during "
        "checkpoint read. Env KAPPA_ORBAX_RESTORE_CONCURRENT_GB overrides when this flag is omitted.",
    )
    p.add_argument(
        "--mesh",
        type=str,
        default="none",
        choices=("none", "auto"),
        help="none: single-device load. auto: (1,N) device mesh, numpy Orbax restore, then "
        "tensor-parallel PartitionSpec + jax.device_put (see kappa.qwen3.sharding).",
    )
    return p.parse_args()


def _load_hf_tokenizer(path: Path):
    from kappa.qwen3.hf_tokenizer import QwenHfTokenizer

    p = path.expanduser().resolve()
    if p.is_dir():
        if not (p / "tokenizer.json").is_file():
            print(f"Not a HF tokenizer directory (missing tokenizer.json): {p}", file=sys.stderr)
            raise SystemExit(1)
        return QwenHfTokenizer.from_directory(p)
    if p.is_file():
        if p.name != "tokenizer.json":
            print(f"Expected tokenizer.json or a directory, got: {p}", file=sys.stderr)
            raise SystemExit(1)
        return QwenHfTokenizer.from_tokenizer_json(p)
    print(f"Tokenizer path not found: {p}", file=sys.stderr)
    raise SystemExit(1)


def _chat_user_assistant(text: str) -> str:
    """Qwen3 chat template (text before assistant completion)."""
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _encode_prompt(tok, text: str, *, chat: bool) -> list[int]:
    payload = _chat_user_assistant(text) if chat else text
    return tok.encode(payload)


def _decode_generated(tok, full_ids: list[int], prompt_len: int, eos_id: int) -> str:
    gen = full_ids[prompt_len:]
    if eos_id in gen:
        gen = gen[: gen.index(eos_id)]
    return tok.decode(gen, skip_special_tokens=False)


def _default_ckpt_for_model(name: str) -> Path:
    from kappa import defaults as d

    if name == "qwen3-0.6b":
        return d.DEFAULT_QWEN3_0P6B_CHECKPOINT
    if name == "qwen3-4b":
        return d.DEFAULT_QWEN3_4B_CHECKPOINT
    if name == "qwen3-30b-a3b":
        return d.DEFAULT_QWEN3_30B_A3B_CHECKPOINT
    raise ValueError(name)


def main() -> None:
    args = _parse_args()

    from kappa.defaults import default_qwen3_tokenizer_dir

    ckpt = args.checkpoint or _default_ckpt_for_model(args.model)
    tok_path = args.tokenizer or default_qwen3_tokenizer_dir()

    if not ckpt.is_dir():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        raise SystemExit(1)

    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        prompt_text = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt_text = args.prompt

    import jax
    import jax.numpy as jnp

    from kappa.qwen3.generate import generate
    from kappa.qwen3.load import load_qwen3_for_mesh, load_qwen3_unsharded
    from kappa.qwen3.rope import build_qwen3_rope_cache
    from kappa.qwen3.special_tokens import QWEN3_EOS

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    print(f"Loading checkpoint from {ckpt} (this can take a while)...", flush=True)
    mesh = None
    if args.mesh == "auto":
        from kappa.qwen3.sharding import create_qwen3_device_mesh

        mesh = create_qwen3_device_mesh()
        cfg, params = load_qwen3_for_mesh(
            ckpt,
            mesh,
            preset=args.model,  # type: ignore[arg-type]
            dtype=dtype,
            restore_concurrent_gb=args.orbax_restore_concurrent_gb,
        )
        print(
            f"  sharded load: mesh shape {mesh.devices.shape} axes={mesh.axis_names}",
            flush=True,
        )
    else:
        cfg, params = load_qwen3_unsharded(
            ckpt,
            preset=args.model,
            dtype=dtype,
            restore_concurrent_gb=args.orbax_restore_concurrent_gb,
        )  # type: ignore[arg-type]
    if cfg.use_moe:
        cfg = dataclasses.replace(cfg, moe_impl=args.moe_impl)  # type: ignore[arg-type]
    print(
        f"Config: {cfg.num_layers} layers, model_dim={cfg.model_dim}, moe={cfg.use_moe}",
        flush=True,
    )
    if cfg.use_moe:
        print(f"  moe_impl={cfg.moe_impl}", flush=True)

    max_len = args.max_cache_len
    rope_cache = build_qwen3_rope_cache(cfg, max_seq_len=max_len)

    print(f"Loading tokenizer from {tok_path}...", flush=True)
    tok = _load_hf_tokenizer(tok_path)
    eos_id = tok.eos_id if tok.eos_id is not None else QWEN3_EOS

    ids = _encode_prompt(tok, prompt_text, chat=args.chat)
    prompt = jnp.asarray([ids], dtype=jnp.int32)
    plen = int(prompt.shape[1])
    if plen + args.max_new_tokens > max_len:
        print(
            f"--max-cache-len ({max_len}) must be >= prompt ({plen}) + max_new_tokens ({args.max_new_tokens})",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.prompt_file is not None:
        print(f"Prompt from {args.prompt_file} ({plen} tokens)", flush=True)
    else:
        print(f"Prompt ({plen} tokens): {prompt_text!r}", flush=True)
    rng = jax.random.key(args.seed)
    decode_chunk = None if args.decode_chunk_size <= 0 else args.decode_chunk_size
    stop_ids: tuple[int, ...] | None = () if args.no_early_stop else None
    pfc = args.prefill_chunk_size if args.prefill_chunk_size > 0 else None
    _gen_kw = dict(
        rng=rng,
        prompt=prompt,
        params=params,
        cfg=cfg,
        rope_cache=rope_cache,
        max_new_tokens=args.max_new_tokens,
        max_cache_len=max_len,
        temperature=0.0,
        decode_scan_chunk_size=decode_chunk,
        prefill_chunk_size=pfc,
        stop_token_ids=stop_ids,
        return_timings=args.timings,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=eos_id,
    )
    if mesh is not None:
        with mesh:
            gen = generate(**_gen_kw)
    else:
        gen = generate(**_gen_kw)
    if args.timings:
        out, timings = gen
        n_new = int(out.shape[1]) - plen
        ttot = timings["total_s"]
        tdec = timings["decode_s"]
        print(
            f"timings: prefill={timings['prefill_s']:.3f}s  "
            f"ttft={timings['ttft_s']:.3f}s  "
            f"decode={tdec:.3f}s  "
            f"total={ttot:.3f}s  "
            f"new_tokens={n_new}",
            flush=True,
        )
        if ttot > 0 and n_new > 0:
            print(f"throughput: {n_new / ttot:.2f} new tok/s (total time)", flush=True)
        if tdec > 0 and n_new > 1:
            print(f"throughput: {(n_new - 1) / tdec:.2f} new tok/s (decode only)", flush=True)
    else:
        out = gen
    out_ids = out[0].tolist()
    text = _decode_generated(tok, out_ids, plen, eos_id=eos_id)
    print("\n--- generated (continuation; stops at EOS) ---\n")
    print(text)


if __name__ == "__main__":
    main()
