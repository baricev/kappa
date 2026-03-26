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

## Important:
- ignore the `ignore/` directory and all of its contents. Do NOT search or read it.
