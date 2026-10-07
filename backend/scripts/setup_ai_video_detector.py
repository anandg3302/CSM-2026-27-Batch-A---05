#!/usr/bin/env python3
"""Stage A — Download and verify the VideoMAE deepfake detector model.

Downloads model.safetensors, config.json, and preprocessor_config.json
from Vansh180/VideoMae-ffc23-deepfake-detector at the pinned revision
ea26518b527d938d66e568c52bcf1c19a86cf295.

Usage:
    python scripts/setup_ai_video_detector.py
"""

import hashlib
import json
import sys
from pathlib import Path

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "storage" / "models"
MODEL_DIR = MODELS_DIR / "Vansh180--VideoMae-ffc23-deepfake-detector"

EXPECTED_SAFETENSORS_SHA256 = (
    "293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29"
)
MODEL_REPO = "Vansh180/VideoMae-ffc23-deepfake-detector"
PINNED_REVISION = "ea26518b527d938d66e568c52bcf1c19a86cf295"

FILES_TO_DOWNLOAD = [
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
]


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_model():
    """Download model files from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES_TO_DOWNLOAD:
        target = MODEL_DIR / filename
        if target.is_file():
            print(f"  [EXISTS] {filename}")
            continue

        print(f"  [DOWNLOADING] {filename} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                revision=PINNED_REVISION,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"  [OK] {filename} -> {downloaded}")
        except Exception as exc:
            print(f"  [FAILED] {filename}: {exc}")
            sys.exit(1)


def verify_model():
    """Verify model.safetensors SHA-256 hash."""
    safetensors_path = MODEL_DIR / "model.safetensors"
    if not safetensors_path.is_file():
        print(f"ERROR: {safetensors_path} not found.")
        return False

    actual = compute_sha256(safetensors_path)
    if actual.lower() != EXPECTED_SAFETENSORS_SHA256.lower():
        print(f"ERROR: SHA-256 mismatch!")
        print(f"  Expected: {EXPECTED_SAFETENSORS_SHA256}")
        print(f"  Actual:   {actual}")
        return False

    print(f"  [VERIFIED] model.safetensors SHA-256: {actual}")
    return True


def main():
    print("=" * 60)
    print("MediaTrace Stage A — Video AI Detector Setup")
    print("=" * 60)
    print(f"Model: {MODEL_REPO}")
    print(f"Revision: {PINNED_REVISION}")
    print(f"Target: {MODEL_DIR}")
    print()

    download_model()
    print()

    if verify_model():
        # Show all artifact hashes
        print()
        print("Artifact hashes:")
        for name in FILES_TO_DOWNLOAD:
            p = MODEL_DIR / name
            if p.is_file():
                h = compute_sha256(p)
                print(f"  {name}: {h}")

        print()
        print("Video AI detector setup: PASSED")
    else:
        print()
        print("Video AI detector setup: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
