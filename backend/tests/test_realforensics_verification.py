"""
Stage A.2.1: RealForensics Pretrained Model Verification Tests.

These tests verify the integrity, standalone architecture reconstruction, state_dict compatibility,
and deterministic execution of the official RealForensics pretrained model (CVPR 2022).

Constraints:
- RESEARCH & MODEL VERIFICATION ONLY.
- Zero modifications to production VideoMAE detector, endpoints, or decision framework.
- If the checkpoint file is not present locally, tests gracefully skip without failing.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

from .realforensics_arch import RealForensicsModel, build_csn_r101, MeanLinear


CHECKPOINT_DIR = Path("D:/finalPro/storage/models/realforensics-verification")
CHECKPOINT_PATH = CHECKPOINT_DIR / "realforensics_ff.pth"
INTEGRITY_MANIFEST_PATH = CHECKPOINT_DIR / "artifact_integrity.json"

EXPECTED_SHA256 = "843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d"
EXPECTED_SIZE_BYTES = 86220575
EXPECTED_PARAM_COUNT = 21396224


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestRealForensicsVerification:

    def test_architecture_reconstruction_without_checkpoint(self):
        """Verify the pure PyTorch CSN-101 + MeanLinear model instantiates with correct dimensions."""
        model = RealForensicsModel()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == EXPECTED_PARAM_COUNT, (
            f"Expected {EXPECTED_PARAM_COUNT:,} parameters, got {param_count:,}"
        )

        # Forward pass on small synthetic input to verify spatial/temporal dimensions
        model.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 5, 56, 56)
            out = model(dummy_input)
            assert out.shape == (1, 1), f"Expected output shape (1, 1), got {out.shape}"
            assert torch.isfinite(out).all(), "Model output contains non-finite values"

    def test_checkpoint_integrity_if_present(self):
        """Verify SHA-256 and byte length of downloaded checkpoint against pinned integrity manifest."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip(f"RealForensics checkpoint not present at {CHECKPOINT_PATH}")

        assert CHECKPOINT_PATH.stat().st_size == EXPECTED_SIZE_BYTES, (
            f"Expected {EXPECTED_SIZE_BYTES} bytes, got {CHECKPOINT_PATH.stat().st_size}"
        )

        sha256 = compute_file_sha256(CHECKPOINT_PATH)
        assert sha256 == EXPECTED_SHA256, (
            f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha256}"
        )

        if INTEGRITY_MANIFEST_PATH.exists():
            with open(INTEGRITY_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest.get("sha256") == EXPECTED_SHA256
            assert manifest.get("file_size_bytes") == EXPECTED_SIZE_BYTES

    def test_state_dict_exact_loading_if_present(self):
        """Verify all 625 parameter and buffer tensors match strictly with zero missing/unexpected keys."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip(f"RealForensics checkpoint not present at {CHECKPOINT_PATH}")

        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        assert len(state_dict) == 625, f"Expected 625 tensors in state_dict, got {len(state_dict)}"

        model = RealForensicsModel()
        load_result = model.load_state_dict(state_dict, strict=True)
        assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
        assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    def test_deterministic_inference_pass_if_present(self):
        """Verify deterministic forward passes with exact numerical logit reproducibility."""
        if not CHECKPOINT_PATH.exists():
            pytest.skip(f"RealForensics checkpoint not present at {CHECKPOINT_PATH}")

        model = RealForensicsModel()
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False), strict=True)
        model.eval()

        with torch.no_grad():
            # 1. Deterministic zeros input (1, 3, 25, 150, 150)
            zeros_input = torch.zeros(1, 3, 25, 150, 150)
            out_zeros = model(zeros_input)
            expected_zeros_logit = 1.866194
            assert pytest.approx(out_zeros.item(), abs=1e-4) == expected_zeros_logit, (
                f"Zeros logit discrepancy: expected {expected_zeros_logit}, got {out_zeros.item()}"
            )

            # 2. Deterministic pseudo-random input (seed=42)
            torch.manual_seed(42)
            rand_input = torch.randn(1, 3, 25, 150, 150)
            out_rand = model(rand_input)
            expected_rand_logit = -0.702618
            assert pytest.approx(out_rand.item(), abs=1e-4) == expected_rand_logit, (
                f"Randn seed=42 logit discrepancy: expected {expected_rand_logit}, got {out_rand.item()}"
            )

    def test_forensic_output_properties(self):
        """Verify forensic neutrality: outputs are single raw logits, not calibrated probabilities."""
        model = RealForensicsModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 25, 112, 112)
        with torch.no_grad():
            logit = model(dummy_input)

        # Output is unbounded real scalar logit (can be positive or negative)
        assert logit.dim() == 2 and logit.shape == (1, 1)
        raw_val = logit.item()
        assert isinstance(raw_val, float)
        # Note: Uncalibrated sigmoid must not be conflated with posterior ground-truth probability
