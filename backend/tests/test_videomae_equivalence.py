"""MediaTrace VideoMAE Runtime Equivalence Test.

Verifies that MediaTrace's manual VideoMAE forward implementation produces
numerically equivalent logits and intermediate representations to the official
Hugging Face Transformers implementation for the exact same model revision,
weights, and deterministic input tensor.

Model:
  Repository: Vansh180/VideoMae-ffc23-deepfake-detector
  Pinned revision: ea26518b527d938d66e568c52bcf1c19a86cf295
  Artifact SHA-256: 293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from app.core.config import PROJECT_ROOT, settings
from app.services.analysis.video_ai_detection_service import (
    EXPECTED_SAFETENSORS_SHA256,
    PINNED_VIDEO_MODEL_ID,
    PINNED_VIDEO_MODEL_VERSION,
    _build_sinusoid_encoding_table,
    _layer_norm,
    _run_videomae_forward,
    _softmax,
)

MODEL_DIR = settings.models_dir / "Vansh180--VideoMae-ffc23-deepfake-detector"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"

# Numerical tolerance for float32 accumulation across 12 transformer encoder blocks
FLOAT32_LOGIT_TOLERANCE = 1e-4

# Frozen reference logits for np.random.RandomState(42).randn(1, 16, 3, 224, 224)
REFERENCE_LOGITS = np.array([[0.04268871, -0.10252903]], dtype=np.float32)


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


@pytest.fixture(scope="module")
def deterministic_input() -> np.ndarray:
    """Generate fixed-seed float32 tensor representing preprocessed clip."""
    prng = np.random.RandomState(42)
    return prng.randn(1, 16, 3, 224, 224).astype(np.float32)


class TestVideoMaeRuntimeEquivalence:
    """Verify numerical equivalence between manual VideoMAE forward and official model."""

    def test_model_artifact_integrity(self):
        """Task 1: Verify model.safetensors exists and SHA-256 matches expected pin."""
        assert SAFETENSORS_PATH.is_file(), f"Model artifact not found at {SAFETENSORS_PATH}"
        actual_sha = compute_sha256(SAFETENSORS_PATH)
        assert (
            actual_sha.lower() == EXPECTED_SAFETENSORS_SHA256.lower()
        ), f"SHA-256 mismatch: expected={EXPECTED_SAFETENSORS_SHA256} actual={actual_sha}"

    def test_manual_forward_matches_frozen_reference(self, deterministic_input):
        """Verify manual implementation against frozen reference logits."""
        weights = load_file(str(SAFETENSORS_PATH))
        manual_logits = _run_videomae_forward(deterministic_input, weights)

        max_diff = float(np.max(np.abs(manual_logits - REFERENCE_LOGITS)))
        assert max_diff < FLOAT32_LOGIT_TOLERANCE, (
            f"Manual logits diverge from reference: max_diff={max_diff:.6e}, "
            f"manual={manual_logits}, ref={REFERENCE_LOGITS}"
        )

        scores = _softmax(manual_logits[0])
        assert scores[0] > scores[1], "Class 0 (real) should have higher score"

    def test_manual_forward_matches_official_transformers(self, deterministic_input):
        """Tasks 2-9: Compare manual forward pass against official VideoMAEForVideoClassification."""
        try:
            import torch
            from transformers import VideoMAEForVideoClassification
        except ImportError:
            pytest.skip("PyTorch or Transformers not installed in environment")

        weights = load_file(str(SAFETENSORS_PATH))

        # 1. Run MediaTrace manual implementation
        manual_logits = _run_videomae_forward(deterministic_input, weights)

        # 2. Run official Hugging Face implementation with exact checkpoint weights
        model_official = VideoMAEForVideoClassification.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        model_official.eval()

        with torch.no_grad():
            for i in range(12):
                qb = torch.from_numpy(
                    weights[f"videomae.encoder.layer.{i}.attention.attention.q_bias"]
                )
                vb = torch.from_numpy(
                    weights[f"videomae.encoder.layer.{i}.attention.attention.v_bias"]
                )
                model_official.videomae.encoder.layer[i].attention.attention.query.bias.copy_(qb)
                model_official.videomae.encoder.layer[i].attention.attention.key.bias.zero_()
                model_official.videomae.encoder.layer[i].attention.attention.value.bias.copy_(vb)

            x_pt = torch.from_numpy(deterministic_input)
            out_official = model_official(x_pt)
            official_logits = out_official.logits.numpy()

        # 3. Compute differences
        diff = np.abs(manual_logits - official_logits)
        max_diff = float(np.max(diff))

        # 4. Numerical assertions
        assert max_diff < FLOAT32_LOGIT_TOLERANCE, (
            f"Logits diverge: max_diff={max_diff:.6e} > tolerance={FLOAT32_LOGIT_TOLERANCE}\n"
            f"  official: {official_logits}\n"
            f"  manual:   {manual_logits}\n"
            f"  diff:     {diff}"
        )

        # 5. Class and score equivalence
        official_scores = _softmax(official_logits[0])
        manual_scores = _softmax(manual_logits[0])

        official_class = int(np.argmax(official_logits[0]))
        manual_class = int(np.argmax(manual_logits[0]))

        assert official_class == manual_class, (
            f"Selected class mismatch: official={official_class}, manual={manual_class}"
        )
        assert abs(official_scores[0] - manual_scores[0]) < FLOAT32_LOGIT_TOLERANCE
        assert abs(official_scores[1] - manual_scores[1]) < FLOAT32_LOGIT_TOLERANCE

    def test_intermediate_patch_embedding_equivalence(self, deterministic_input):
        """Verify patch embedding projection matches PyTorch Conv3D exactly."""
        try:
            import torch
            from transformers.models.videomae.modeling_videomae import (
                VideoMAEConfig,
                VideoMAEPatchEmbeddings,
            )
        except ImportError:
            pytest.skip("PyTorch or Transformers not installed in environment")

        weights = load_file(str(SAFETENSORS_PATH))
        proj_w_np = weights["videomae.embeddings.patch_embeddings.projection.weight"]
        proj_b_np = weights["videomae.embeddings.patch_embeddings.projection.bias"]

        config = VideoMAEConfig.from_pretrained(str(MODEL_DIR))
        pe_pt = VideoMAEPatchEmbeddings(config)
        pe_pt.projection.weight.data.copy_(torch.from_numpy(proj_w_np))
        pe_pt.projection.bias.data.copy_(torch.from_numpy(proj_b_np))
        pe_pt.eval()

        with torch.no_grad():
            x_pt = torch.from_numpy(deterministic_input)
            pt_out = pe_pt(x_pt).numpy()

        B = deterministic_input.shape[0]
        x = deterministic_input.transpose(0, 2, 1, 3, 4)
        x = x.reshape(B, 3, 8, 2, 14, 16, 14, 16)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, 1568, 1536)
        np_out = x @ proj_w_np.reshape(768, 1536).T + proj_b_np

        diff = np.max(np.abs(pt_out - np_out))
        assert diff < 5e-5, f"Patch embedding diff {diff:.6e} exceeds 5e-5"
