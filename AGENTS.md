# kappa

Functional JAX building blocks for Gemma 3, Qwen 3 and Gpt-OSS inference and training (experimental).

Local development on Apple Silicon via `jax-mps` and TPU-focused work on paged KV cache + ragged decode.

This file is the source of truth for environment setup, model asset locations, edit boundaries, and third-party policy.

## Python environment

- **Virtualenv**: `mps-python-3.13`
- **Python**: 3.13 (required by `jax-mps`)
- **Platform**: macOS arm64 (Apple Silicon)

```sh
source /Users/x/workspace/jax-functional/mps-python-3.13/bin/activate
uv pip install -e .   # editable install from repo root
```

## Model assets

| Asset | Path |
|---|---|
| Gemma-3 4B checkpoint (Orbax/OCDBT) | `/Users/x/workspace/4b` |
| Gemma-3N checkpoint | `/Users/x/workspace/3n` |
| Gemma-3 270M checkpoint | `/Users/x/workspace/3-270m/` |
| SentencePiece tokenizer | `./tokenizer.model` (repo root) |

Paths resolve by platform in `kappa.defaults` (detected via `JAX_PLATFORM_NAME` or `jax.devices()`):

| Environment | Checkpoint | Tokenizer |
|---|---|---|
| Local (MPS) | `/Users/x/workspace/4b` | `repo_root/tokenizer.model` |
| Colab (TPU/GPU) | `/content/drive/MyDrive/4b` | `/content/drive/MyDrive/4b/tokenizer.model` |

Colab is auto-detected by checking if `/content/drive/MyDrive` exists (requires `drive.mount()`). To add a new environment, edit `defaults.py`.

## Third-party policy (`third_party/`)

Reference trees are read-only sources for ideas and tests:

- **Do not import** `third_party` code from first-party modules. Copy/adapt into `kappa`.
- **No git submodules.** Optional clones are gitignored; see `third_party/README.md` for clone commands.
- The third_party/ directory contains the following repositories:
    https://github.com/AI-Hypercomputer/maxtext.git
    https://github.com/AI-Hypercomputer/JetStream.git
    https://github.com/jax-ml/jax.git
    https://github.com/google-deepmind/gemma.git
    https://github.com/google-deepmind/simply
    https://github.com/vllm-project/tpu-inference.git

## Qwen 3 (MoE paths, tokenizer, parity)

- **Checkpoints / tokenizer**: Orbax layout via `kappa.qwen3.load`; HF flat conversion in `kappa.checkpoint.qwen_hf_convert`. Use a HF tokenizer directory with `tokenizer.json` (`uv pip install -e '.[inference]'`). **`--model` / preset must match** the checkpoint (embed width, layer count, dense vs MoE).
- **MoE execution** (`Qwen3Config`): `moe_impl` is one of `gather_einsum` (default), `fixed_capacity` (capacity-limited dispatch; can drop routes), `ragged_jax` (`jax.lax.ragged_dot`), `ragged_tokamax` (uses **Tokamax** when installed; otherwise same as `ragged_jax`). Install Tokamax from source, e.g. `uv pip install git+https://github.com/openxla/tokamax.git`. See `moe_capacity_factor`, `moe_ragged_decode_token_threshold` in `kappa/qwen3/architecture.py`. For small token counts (decode), ragged modes automatically use `gather_einsum` when `total_tokens <= moe_ragged_decode_token_threshold`.
- **Parity**: We do **not** treat Hugging Face logits/hidden states as a required reference; prefer internal checks (e.g. `scripts/verify_qwen3_moe_impls.py`) and `Integrating_MaxText_MoE_Strategies.md` for the MoE roadmap.

## Important:
- ignore the `ignore/` directory and all of its contents. Do NOT search or read it.
