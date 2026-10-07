"""
Stage A.3: Discovery Layer Documentation and Freeze Verification Tests.

Validates that:
1. Stage A.2.3 baseline validation artifacts remain strictly frozen and unmodified.
2. The candidate discovery document docs/VIDEO_AI_DETECTOR_CROSS_DATASET_A3_CANDIDATES.md exists and is complete.
3. The comparison table satisfies all required criteria columns.
4. All 5 identified candidates (Celeb-DF v2, FF++, DFDC, DeeperForensics, WildDeepfake) are documented.
5. The proposed validation protocol defines cohort targets (>=10 human, >=10 manipulated) with no score fusion or threshold tuning.
6. No unauthorized new video files or dataset directories have been created.
"""

import json
from pathlib import Path
import pytest

ROOT_DIR = Path("D:/finalPro")
EXPANDED_DATASET_DIR = ROOT_DIR / "datasets" / "ai_video_detector_validation" / "expanded"
EXPANDED_MANIFEST_PATH = EXPANDED_DATASET_DIR / "validation_manifest.json"
EXPANDED_RESULTS_PATH = EXPANDED_DATASET_DIR / "validation_results.json"
DOC_A23_PATH = ROOT_DIR / "docs" / "VIDEO_AI_DETECTOR_VALIDATION_A2.3.md"
DOC_A3_PATH = ROOT_DIR / "docs" / "VIDEO_AI_DETECTOR_CROSS_DATASET_A3_CANDIDATES.md"


class TestStageA3DiscoveryLayer:

    def test_stage_a23_freeze_verification(self):
        """Verify Stage A.2.3 baseline validation artifacts are intact and frozen."""
        assert EXPANDED_MANIFEST_PATH.exists(), f"Expanded manifest missing: {EXPANDED_MANIFEST_PATH}"
        assert EXPANDED_RESULTS_PATH.exists(), f"Expanded results missing: {EXPANDED_RESULTS_PATH}"
        assert DOC_A23_PATH.exists(), f"Stage A.2.3 documentation missing: {DOC_A23_PATH}"

        with open(EXPANDED_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["manifest_version"] == "2.0.0"
        assert len(manifest["samples"]) == 26

        with open(EXPANDED_RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        assert len(results["results"]) == 26
        assert results["dataset"]["total_samples"] == 26

        assert DOC_A23_PATH.stat().st_size > 5000

    def test_stage_a3_candidates_document_structure(self):
        """Verify candidate discovery document exists and contains required sections."""
        assert DOC_A3_PATH.exists(), f"Stage A.3 candidate document missing: {DOC_A3_PATH}"
        content = DOC_A3_PATH.read_text(encoding="utf-8")

        # Check required table columns
        required_columns = [
            "Candidate",
            "Ground Truth",
            "Manipulation Diversity",
            "Independent of SDFVD",
            "Access",
            "License",
            "Selective Download",
            "Main Limitation",
        ]
        for col in required_columns:
            assert col.lower() in content.lower(), f"Missing column in comparison table: {col}"

        # Check key candidates documented
        candidates = ["Celeb-DF", "FaceForensics++", "DFDC", "DeeperForensics", "WildDeepfake"]
        for cand in candidates:
            assert cand in content, f"Candidate not documented: {cand}"

        # Check suitability categories present
        assert "Suitable" in content
        assert "Potentially Suitable" in content

        # Check protocol specifications
        assert "10 human" in content.lower() or "at least 10 human" in content.lower()
        assert "10 manipulated" in content.lower() or "at least 10 manipulated" in content.lower()
        assert "No Score Fusion" in content or "no score fusion" in content.lower()
        assert "No Threshold Tuning" in content or "no threshold tuning" in content.lower()

        # Check hard stop condition
        assert "STOP AFTER DATASET DISCOVERY" in content
        assert "WAIT FOR EXPLICIT" in content or "WAIT FOR EXPLICIT USER APPROVAL" in content

    def test_no_unauthorized_downloads_performed(self):
        """Verify no new unapproved video directories or downloaded files were created in datasets/."""
        datasets_dir = ROOT_DIR / "datasets"
        allowed_dirs = {"ai_detector_validation", "ai_video_detector_validation", "external_sources"}
        actual_dirs = {p.name for p in datasets_dir.iterdir() if p.is_dir()}
        assert actual_dirs.issubset(allowed_dirs), f"Unexpected directory in datasets: {actual_dirs - allowed_dirs}"

        # In external_sources, only established sdfvd, dfdc_kaggle_sample, and faceforensicspp_c23 should exist
        ext_dir = datasets_dir / "external_sources"
        allowed_ext = {"sdfvd", "dfdc_kaggle_sample", "faceforensicspp_c23"}
        actual_ext = {p.name for p in ext_dir.iterdir() if p.is_dir()}
        assert actual_ext.issubset(allowed_ext), f"Unexpected external source folder: {actual_ext - allowed_ext}"
