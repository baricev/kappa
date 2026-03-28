#!/usr/bin/env python3
"""Download arbitrary Hugging Face models to disk for use with kappa.

Example::

    # Download Gemma 3 4B Orbax checkpoint
    python scripts/download_hf_model.py google/gemma-3-4b-it-jax --local-dir ~/workspace/4b-it
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not found. Install with:")
    print("  pip install huggingface_hub")
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repository ID (e.g., 'google/gemma-3-4b-it-jax')",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local directory to download the model to",
    )
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token (optional, or use HF_TOKEN env var)",
    )
    p.add_argument(
        "--include",
        type=str,
        nargs="+",
        help="Glob patterns to include (e.g., '*.model', 'checkpoint*')",
    )
    p.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        help="Glob patterns to exclude",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Downloading {args.repo_id} to {args.local_dir}...")
    
    # Ensure the directory exists
    args.local_dir.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
            token=args.token,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
        )
        print(f"\n[OK] Download complete: {path}")
        
        # Check for Orbax-like structure and warn if it looks nested
        checkpoints = list(args.local_dir.glob("**/checkpoint"))
        if checkpoints:
            print("\nDetected potential Orbax checkpoints at:")
            for cp in checkpoints:
                print(f"  {cp.parent}")
            print("\nUse the path above as --checkpoint in inference scripts.")

        # Check for tokenizer.model
        tokenizers = list(args.local_dir.glob("**/tokenizer.model"))
        if tokenizers:
            print("\nDetected potential tokenizer models at:")
            for t in tokenizers:
                print(f"  {t}")
            print("\nConsider moving one to the repo root to use as the default.")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
