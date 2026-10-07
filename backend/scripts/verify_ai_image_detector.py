r"""Step 14.1.1.2 — Standalone Real ONNX Runtime Verification Script.

Loads the locally exported `storage/models/umm-maybe--AI-image-detector/model.onnx`
derived from pinned `pytorch_model.bin` (`c7e223baf11bc40528af364ba7bdea030ef42f9e`),
creates a deterministic synthetic RGB image in memory, runs real ONNX Runtime
inference, and prints the model path, SHA-256, input/output shapes, raw logits,
and selected class.

Usage:
    cd backend
    $env:PYTHONPATH="."; .\.venv\Scripts\python scripts/verify_ai_image_detector.py
"""

from __future__ import annotations

import argparse

from app.core.config import settings
from app.schemas.ai_detection import OFFICIAL_MODEL_ID
from scripts.setup_ai_image_detector import verify_real_onnx_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify real ONNX Runtime inference on installed AI Image Detector")
    parser.add_argument("--model-id", default=settings.ai_image_detector_model_id or OFFICIAL_MODEL_ID)
    args = parser.parse_args()
    verify_real_onnx_inference(args.model_id)
