#!/usr/bin/env python3
"""
Download recommended GGUF models for local utterance segmentation.

This script helps you download quantized GGUF models from HuggingFace
that work well with the utterance segmentation pipeline.

Recommended models:
- Phi-3 Mini 4K Instruct (Q4_K_M, ~2.4GB) - Best balance of quality/speed
- Phi-3 Mini 4K Instruct (Q2_K, ~1.4GB) - Faster, lower quality
- Llama 3.2 3B Instruct (Q4_K_M, ~1.9GB) - Alternative option

Usage:
    # Download default model (Phi-3 Mini Q4_K_M)
    python download_local_model.py

    # Download specific model
    python download_local_model.py --model phi3-q2

    # Download to specific directory
    python download_local_model.py --output_dir /path/to/models

    # List available models
    python download_local_model.py --list
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# Available models with HuggingFace URLs
MODELS = {
    "phi3-q4": {
        "name": "Phi-3 Mini 4K Instruct (Q4_K_M)",
        "file": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "size": "2.4 GB",
        "description": "Recommended default - good balance of quality and speed"
    },
    "phi3-q2": {
        "name": "Phi-3 Mini 4K Instruct (Q2_K)",
        "file": "Phi-3-mini-4k-instruct-Q2_K.gguf",
        "url": "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q2_K.gguf",
        "size": "1.4 GB",
        "description": "Faster, lower quality - good for testing"
    },
    "llama32-3b-q4": {
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": "1.9 GB",
        "description": "Alternative model with good performance"
    }
}


def list_models():
    """Display available models."""
    print("\nAvailable models:\n")
    print("=" * 80)
    for model_id, info in MODELS.items():
        print(f"ID:          {model_id}")
        print(f"Name:        {info['name']}")
        print(f"File:        {info['file']}")
        print(f"Size:        {info['size']}")
        print(f"Description: {info['description']}")
        print("-" * 80)
    print()


def download_model(model_id: str, output_dir: Path) -> bool:
    """
    Download a model using urllib.

    Args:
        model_id: Model identifier from MODELS dict
        output_dir: Directory to save the model

    Returns:
        True if successful, False otherwise
    """
    if model_id not in MODELS:
        print(f"Error: Unknown model '{model_id}'", file=sys.stderr)
        print("Use --list to see available models", file=sys.stderr)
        return False

    model_info = MODELS[model_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / model_info['file']

    # Check if already downloaded
    if output_path.exists():
        print(f"Model already exists at: {output_path}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Skipping download.")
            return True

    print(f"\nDownloading {model_info['name']}...")
    print(f"URL:  {model_info['url']}")
    print(f"Size: {model_info['size']}")
    print(f"Saving to: {output_path}")
    print()

    try:
        import urllib.request

        def progress_hook(block_num, block_size, total_size):
            """Display download progress."""
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='')
            else:
                mb_downloaded = downloaded / (1024 * 1024)
                print(f"\rDownloaded: {mb_downloaded:.1f} MB", end='')

        urllib.request.urlretrieve(
            model_info['url'],
            output_path,
            reporthook=progress_hook
        )
        print("\n\nDownload complete!")
        print(f"Model saved to: {output_path}")
        print("\nUsage example:")
        print(f"  python -m utterance_segmentation.cli \\")
        print(f"    --input_asr_json data/asr.json \\")
        print(f"    --output_json output.json \\")
        print(f"    --llm_backend local \\")
        print(f"    --local_model_path {output_path}")
        print("\nOr with standalone script:")
        print(f"  python standalone_utterance_seg.py \\")
        print(f"    --input_asr_json data/asr.json \\")
        print(f"    --output_json output.json \\")
        print(f"    --llm_backend local \\")
        print(f"    --local_model_path {output_path}")
        return True

    except Exception as e:
        print(f"\nError downloading model: {e}", file=sys.stderr)
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download recommended GGUF models for local utterance segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (Phi-3 Mini Q4_K_M)
  python download_local_model.py

  # Download faster, smaller model
  python download_local_model.py --model phi3-q2

  # Download Llama 3.2 3B
  python download_local_model.py --model llama32-3b-q4

  # Download to specific directory
  python download_local_model.py --output_dir /path/to/models

  # List available models
  python download_local_model.py --list
"""
    )

    parser.add_argument(
        "--model",
        default="phi3-q4",
        help="Model to download (default: phi3-q4). Use --list to see options"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the model (default: ./models)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    print("=" * 80)
    print("Local Model Downloader for Utterance Segmentation")
    print("=" * 80)

    success = download_model(args.model, args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
