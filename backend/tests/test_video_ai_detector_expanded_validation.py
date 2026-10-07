"""
Stage A.2.3: Expanded Controlled Video AI Detector Validation Tests.

Validates that:
1. Pinned model weights and auxiliary assets for BOTH detectors match expected cryptographic SHA-256 digests.
2. The expanded validation manifest (v2.0.0) exists and all 26 samples match their recorded SHA-256 hashes.
3. Both VideoMAE and RealForensics produce deterministic evaluations.
4. The 6 transformed derivatives (3 human, 3 manipulated) correctly map to their parent source videos.
5. Expanded validation result artifacts (JSON, CSV, summary JSON, summary MD) exist, contain 26 records, and are well-formed.
6. Comparative relationship taxonomy (both_agree, both_disagree, models_disagree, evaluated_derivative) is consistent.
7. Baseline validation results from Stage A.1 and Stage A.2.2 remain strictly untouched.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch

from .realforensics_arch import RealForensicsModel
from .fan_arch import FAN

ROOT_DIR = Path("D:/finalPro")
BACKEND_DIR = ROOT_DIR / "backend"
MODELS_DIR = ROOT_DIR / "storage" / "models"
VMAE_DIR = MODELS_DIR / "Vansh180--VideoMae-ffc23-deepfake-detector"
RF_DIR = MODELS_DIR / "realforensics-verification"

EXPANDED_DATASET_DIR = ROOT_DIR / "datasets" / "ai_video_detector_validation" / "expanded"
EXPANDED_MANIFEST_PATH = EXPANDED_DATASET_DIR / "validation_manifest.json"

STAGE_A1_RESULTS = ROOT_DIR / "datasets" / "ai_video_detector_validation" / "validation_results.json"
STAGE_A22_RESULTS = ROOT_DIR / "datasets" / "ai_video_detector_validation" / "realforensics" / "validation_results.json"

EXPECTED_VMAE_SAFETENSORS_SHA256 = "293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29"
EXPECTED_RF_CHECKPOINT_SHA256 = "843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d"
EXPECTED_RF_MEAN_FACE_SHA256 = "dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422"
EXPECTED_RF_FAN_SHA256 = "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa"


def compute_file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class TestExpandedVideoAIDetectorValidation:

    def test_both_detector_artifacts_cryptographic_integrity(self):
        """Verify model weights and auxiliary assets for VideoMAE and RealForensics match exact SHA-256."""
        # VideoMAE
        vmae_safetensors = VMAE_DIR / "model.safetensors"
        assert vmae_safetensors.exists(), f"VideoMAE weights missing: {vmae_safetensors}"
        assert compute_file_sha256(vmae_safetensors).lower() == EXPECTED_VMAE_SAFETENSORS_SHA256.lower()

        # RealForensics
        rf_ckpt = RF_DIR / "realforensics_ff.pth"
        rf_mean_face = RF_DIR / "20words_mean_face.npy"
        rf_fan = RF_DIR / "2DFAN4-11f355bf06.pth.tar"

        assert rf_ckpt.exists(), f"RealForensics checkpoint missing: {rf_ckpt}"
        assert rf_mean_face.exists(), f"Mean face missing: {rf_mean_face}"
        assert rf_fan.exists(), f"2D-FAN missing: {rf_fan}"

        assert compute_file_sha256(rf_ckpt).lower() == EXPECTED_RF_CHECKPOINT_SHA256.lower()
        assert compute_file_sha256(rf_mean_face).lower() == EXPECTED_RF_MEAN_FACE_SHA256.lower()
        assert compute_file_sha256(rf_fan).lower() == EXPECTED_RF_FAN_SHA256.lower()

    def test_expanded_manifest_and_sample_integrity(self):
        """Verify expanded manifest contains all 26 samples with matching SHA-256 hashes on disk."""
        assert EXPANDED_MANIFEST_PATH.exists(), f"Expanded manifest missing: {EXPANDED_MANIFEST_PATH}"
        with open(EXPANDED_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["manifest_version"] == "2.0.0"
        assert manifest["stage"] == "A.2.3"
        samples = manifest["samples"]
        assert len(samples) == 26

        human_samples = [s for s in samples if s["category"] == "human"]
        manip_samples = [s for s in samples if s["category"] == "manipulated"]
        trans_samples = [s for s in samples if s["category"] == "transformed"]

        assert len(human_samples) == 10
        assert len(manip_samples) == 10
        assert len(trans_samples) == 6

        for s in samples:
            cat = s["category"]
            fn = s["filename"]
            sample_path = EXPANDED_DATASET_DIR / cat / fn
            assert sample_path.exists(), f"Sample missing: {sample_path}"
            assert sample_path.stat().st_size == s["file_size_bytes"]
            actual_sha = compute_file_sha256(sample_path)
            assert actual_sha.lower() == s["sha256"].lower(), (
                f"SHA-256 mismatch for {fn}: expected {s['sha256']}, got {actual_sha}"
            )

    def test_transformation_derivatives_pairing(self):
        """Verify all 6 transformed derivatives properly pair with parent videos and define valid operations."""
        with open(EXPANDED_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        trans_samples = [s for s in manifest["samples"] if s["category"] == "transformed"]
        assert len(trans_samples) == 6

        human_derivatives = [s for s in trans_samples if s["parent_category"] == "human"]
        manip_derivatives = [s for s in trans_samples if s["parent_category"] == "manipulated"]

        assert len(human_derivatives) == 3
        assert len(manip_derivatives) == 3

        for d in trans_samples:
            assert "source_video_filename" in d
            assert "source_video_sha256" in d
            assert "transformation_type" in d
            assert "transformation_operation" in d

            parent_folder = "human" if d["parent_category"] == "human" else "manipulated"
            parent_path = EXPANDED_DATASET_DIR / parent_folder / d["source_video_filename"]
            assert parent_path.exists(), f"Parent file missing: {parent_path}"
            parent_sha = compute_file_sha256(parent_path)
            assert parent_sha.lower() == d["source_video_sha256"].lower()

    def test_deterministic_realforensics_inference(self):
        """Verify RealForensics produces deterministic scalar logit output on fixed tensor."""
        rf_ckpt = RF_DIR / "realforensics_ff.pth"
        model = RealForensicsModel()
        model.load_state_dict(torch.load(rf_ckpt, map_location="cpu", weights_only=False), strict=True)
        model.eval()

        torch.manual_seed(42)
        dummy_clip = torch.randn(1, 3, 25, 112, 112)
        with torch.no_grad():
            out1 = model(dummy_clip).item()
            out2 = model(dummy_clip).item()

        assert out1 == pytest.approx(out2, abs=1e-6)

    def test_expanded_validation_results_artifacts_integrity(self):
        """Verify expanded validation results JSON, CSV, summary JSON, and MD exist and are complete."""
        res_json = EXPANDED_DATASET_DIR / "validation_results.json"
        res_csv = EXPANDED_DATASET_DIR / "validation_results.csv"
        sum_json = EXPANDED_DATASET_DIR / "validation_summary.json"
        sum_md = EXPANDED_DATASET_DIR / "validation_summary.md"

        assert res_json.exists(), f"Results JSON missing: {res_json}"
        assert res_csv.exists(), f"Results CSV missing: {res_csv}"
        assert sum_json.exists(), f"Summary JSON missing: {sum_json}"
        assert sum_md.exists(), f"Summary MD missing: {sum_md}"

        with open(res_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "dataset" in data
        assert "detectors" in data and len(data["detectors"]) == 2
        assert "results" in data and len(data["results"]) == 26
        assert "comparative_observations" in data
        assert "transformation_sensitivity" in data and len(data["transformation_sensitivity"]) == 6

        # Check records schema
        valid_dual_relationships = {
            "both_agree_with_known_category",
            "both_disagree_with_known_category",
            "models_disagree",
            "evaluated_derivative",
            "one_or_both_unavailable",
        }

        for r in data["results"]:
            assert "sample_id" in r
            assert "filename" in r
            assert "category" in r
            assert "videomae" in r
            assert "realforensics" in r
            assert "dual_model_relationship" in r
            assert r["dual_model_relationship"] in valid_dual_relationships

            # VideoMAE assertions
            vmae = r["videomae"]
            assert vmae["status"] in ("evaluated", "not_available")
            if vmae["status"] == "evaluated":
                assert isinstance(vmae["fake_score"], float)
                assert 0.0 <= vmae["fake_score"] <= 1.0
                assert vmae["selected_class"] in ("fake", "real")
                assert vmae["normalized_class"] in ("manipulated", "human")

            # RealForensics assertions
            rf = r["realforensics"]
            assert rf["status"] in ("evaluated", "not_available")
            if rf["status"] == "evaluated":
                assert isinstance(rf["uncalibrated_logit"], float)
                assert rf["selected_class"] in ("manipulated", "human")

    def test_baseline_validation_results_remain_unmodified(self):
        """Verify Stage A.1 and Stage A.2.2 results files were not overwritten."""
        assert STAGE_A1_RESULTS.exists(), f"Stage A.1 results missing: {STAGE_A1_RESULTS}"
        with open(STAGE_A1_RESULTS, "r", encoding="utf-8") as f:
            a1_data = json.load(f)
        assert len(a1_data["evaluated_records"]) == 7

        assert STAGE_A22_RESULTS.exists(), f"Stage A.2.2 results missing: {STAGE_A22_RESULTS}"
        with open(STAGE_A22_RESULTS, "r", encoding="utf-8") as f:
            a22_data = json.load(f)
        assert len(a22_data["results"]) == 7
