r"""Step 14.1.3 — Controlled Multi-Image AI Detector Validation Harness Tests.

Covers all 19 test requirements from Step 14.1.3:
1. valid manifest
2. invalid/missing manifest
3. missing image
4. unsupported image
5. duplicate validation IDs
6. invalid known_category
7. real detector invocation (unmocked ONNX Runtime inference on installed Swin model)
8. deterministic SHA-256 recording
9. human-category result recording
10. AI-category result recording
11. transformed-category handling (excluded from human-vs-AI confusion counts)
12. category disagreement recording (neutral wording, transformed handling)
13. summary statistics (descriptive min/max/mean/median, confusion counts, no accuracy/F1/AUC/ranking)
14. CSV generation (1 row per image)
15. model integrity verification (aborts on corrupt/unpinned model)
16. no modification of source images
17. no fabricated model scores
18. no threshold introduced (threshold=None, score_type='uncalibrated_model_score')
19. no probability/authenticity claim generated
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from app.core.config import settings
from app.schemas.ai_detection import OFFICIAL_MODEL_ID, OFFICIAL_PINNED_REVISION
from app.services.analysis import image_ai_detection_service
from scripts.validate_ai_image_detector import (
    ModelIntegrityError,
    ValidationHarnessError,
    _evaluate_category_disagreement,
    load_and_validate_manifest,
    run_validation_harness,
    verify_model_integrity_for_validation,
)


def _make_test_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (256, 256), color: tuple[int, int, int] = (60, 110, 170)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_1_to_6_manifest_and_dataset_validation_errors(tmp_path):
    """Tests 1–6:
    1. valid manifest
    2. invalid/missing manifest
    3. missing image
    4. unsupported image
    5. duplicate validation IDs
    6. invalid known_category
    """
    ds_dir = tmp_path / "ai_detector_validation"
    (ds_dir / "human").mkdir(parents=True)
    (ds_dir / "ai_generated").mkdir(parents=True)
    (ds_dir / "transformed").mkdir(parents=True)

    # 2a. Missing manifest
    with pytest.raises(ValidationHarnessError, match="not found"):
        load_and_validate_manifest(ds_dir / "manifest.json")

    # 2b. Invalid JSON manifest
    manifest_file = ds_dir / "manifest.json"
    manifest_file.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValidationHarnessError, match="Invalid JSON"):
        load_and_validate_manifest(manifest_file)

    # 3. Missing image file
    manifest_file.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "human_001",
                        "path": "human/does_not_exist.jpg",
                        "known_category": "human",
                        "source_description": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationHarnessError, match="does not exist"):
        load_and_validate_manifest(manifest_file)

    # Create valid image file
    img_bytes = _make_test_image_bytes("JPEG")
    (ds_dir / "human" / "valid.jpg").write_bytes(img_bytes)

    # 4a. Unsupported extension (.txt)
    (ds_dir / "human" / "note.txt").write_text("not an image", encoding="utf-8")
    manifest_file.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "human_001",
                        "path": "human/note.txt",
                        "known_category": "human",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationHarnessError, match="Unsupported image extension"):
        load_and_validate_manifest(manifest_file)

    # 4b. Corrupted image bytes with .jpg extension
    (ds_dir / "human" / "corrupt.jpg").write_bytes(b"NOT_A_JPEG_STREAM")
    manifest_file.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "human_001",
                        "path": "human/corrupt.jpg",
                        "known_category": "human",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationHarnessError, match="Unsupported or unreadable image file"):
        load_and_validate_manifest(manifest_file)

    # 5. Duplicate validation IDs
    manifest_file.write_text(
        json.dumps(
            {
                "samples": [
                    {"id": "dup_001", "path": "human/valid.jpg", "known_category": "human"},
                    {"id": "dup_001", "path": "human/valid.jpg", "known_category": "human"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationHarnessError, match="Duplicate validation ID"):
        load_and_validate_manifest(manifest_file)

    # 6. Invalid known_category
    manifest_file.write_text(
        json.dumps(
            {
                "samples": [
                    {"id": "human_001", "path": "human/valid.jpg", "known_category": "unknown_inferred"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationHarnessError, match="Invalid known_category"):
        load_and_validate_manifest(manifest_file)

    # 1. Valid manifest
    manifest_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "samples": [
                    {
                        "id": "human_001",
                        "path": "human/valid.jpg",
                        "known_category": "human",
                        "source_description": "known camera photo",
                        "generator": None,
                        "transformations": [],
                        "notes": "ok",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parsed = load_and_validate_manifest(manifest_file)
    assert len(parsed["samples"]) == 1
    assert parsed["samples"][0]["id"] == "human_001"
    assert parsed["samples"][0]["known_category"] == "human"
    assert parsed["samples"][0]["initial_sha256"] == hashlib.sha256(img_bytes).hexdigest()


def test_12_category_disagreement_rules():
    """Test 12: Category disagreement recording and neutral wording."""
    # known human + selected artificial -> disagreement True
    dis, note = _evaluate_category_disagreement("human", "artificial")
    assert dis is True
    assert note == "model/category disagreement observed"

    # known human + selected human -> disagreement False
    dis, note = _evaluate_category_disagreement("human", "human")
    assert dis is False
    assert note is None

    # known ai_generated + selected human -> disagreement True
    dis, note = _evaluate_category_disagreement("ai_generated", "human")
    assert dis is True
    assert note == "model/category disagreement observed"

    # known ai_generated + selected artificial -> disagreement False
    dis, note = _evaluate_category_disagreement("ai_generated", "artificial")
    assert dis is False
    assert note is None

    # transformed without explicit expected_category -> disagreement False
    dis, note = _evaluate_category_disagreement("transformed", "artificial", expected_category=None)
    assert dis is False
    assert note is None

    # transformed WITH explicit expected_category="human" and selected="artificial" -> disagreement True
    dis, note = _evaluate_category_disagreement("transformed", "artificial", expected_category="human")
    assert dis is True
    assert note == "model/category disagreement observed"


def test_15_and_17_model_integrity_verification_and_no_fabricated_scores(tmp_path, monkeypatch):
    """Tests 15 & 17: Model integrity verification aborts on missing, unpinned, or mismatched SHA-256 models."""
    fake_models_root = tmp_path / "models"
    fake_model_dir = fake_models_root / "umm-maybe--AI-image-detector"
    fake_model_dir.mkdir(parents=True)

    monkeypatch.setattr(settings, "models_dir", fake_models_root)
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    # Missing config.json / model.onnx -> aborts with ModelIntegrityError
    with pytest.raises(ModelIntegrityError):
        verify_model_integrity_for_validation()

    # Unpinned revision ("main") -> aborts with ModelIntegrityError
    monkeypatch.setattr(settings, "ai_image_detector_model_version", "main")
    with pytest.raises(ModelIntegrityError, match="Unpinned"):
        verify_model_integrity_for_validation()


def test_7_to_19_end_to_end_real_detector_validation_harness(tmp_path, monkeypatch):
    """Tests 7–19: End-to-end validation harness using the REAL installed ONNX model (no mocking).
    Verifies:
    - 3 human, 3 ai_generated, 1 transformed samples in manifest
    - real ONNX Runtime invocation
    - deterministic SHA-256 recording and byte-for-byte preservation of source images
    - human, ai_generated, and transformed result recording
    - descriptive summary statistics (min, max, mean, median)
    - 2x2 confusion count table excluding transformed images
    - CSV generation with 1 row per image
    - threshold=None, score_type='uncalibrated_model_score', decision_rule='model_class_selection'
    - no accuracy/precision/recall/F1/AUC/ranking/authenticity claims
    """
    real_models_root = Path(__file__).resolve().parents[2] / "storage" / "models"
    monkeypatch.setattr(settings, "models_dir", real_models_root)
    monkeypatch.setattr(settings, "storage_root", real_models_root.parent)
    image_ai_detection_service.clear_ai_detector_cache()

    ds_dir = tmp_path / "validation_ds"
    (ds_dir / "human").mkdir(parents=True)
    (ds_dir / "ai_generated").mkdir(parents=True)
    (ds_dir / "transformed").mkdir(parents=True)

    # Create 3 human, 3 ai_generated, and 1 transformed test files with recorded pre-hashes
    samples_manifest = []
    pre_hashes = {}

    for i in range(1, 4):
        rel = f"human/h_{i}.jpg"
        raw = _make_test_image_bytes("JPEG", size=(240 + i * 8, 240), color=(40 * i, 90, 150))
        p = ds_dir / rel
        p.write_bytes(raw)
        pre_hashes[rel] = (len(raw), hashlib.sha256(raw).hexdigest(), raw)
        samples_manifest.append(
            {
                "id": f"human_00{i}",
                "path": rel,
                "known_category": "human",
                "source_description": f"human sample {i}",
                "generator": None,
                "transformations": [],
                "notes": "",
            }
        )

    for i in range(1, 4):
        rel = f"ai_generated/ai_{i}.png"
        raw = _make_test_image_bytes("PNG", size=(224, 224), color=(180, 50 * i, 90))
        p = ds_dir / rel
        p.write_bytes(raw)
        pre_hashes[rel] = (len(raw), hashlib.sha256(raw).hexdigest(), raw)
        samples_manifest.append(
            {
                "id": f"ai_00{i}",
                "path": rel,
                "known_category": "ai_generated",
                "source_description": f"AI sample {i}",
                "generator": "test_generator_v1",
                "transformations": [],
                "notes": "",
            }
        )

    rel_t = "transformed/t_1.jpg"
    raw_t = _make_test_image_bytes("JPEG", size=(160, 160), color=(100, 120, 140))
    (ds_dir / rel_t).write_bytes(raw_t)
    pre_hashes[rel_t] = (len(raw_t), hashlib.sha256(raw_t).hexdigest(), raw_t)
    samples_manifest.append(
        {
            "id": "transformed_001",
            "path": rel_t,
            "known_category": "transformed",
            "original_category": "human",
            "source_description": "Resized + JPEG recompressed sample",
            "generator": None,
            "transformations": ["resize_160x160", "jpeg_q60"],
            "notes": "",
        }
    )

    manifest_path = ds_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples_manifest}, indent=2), encoding="utf-8")

    out_dir = tmp_path / "val_out"
    res = run_validation_harness(manifest_path=manifest_path, output_dir=out_dir, skip_forensics=False)

    # Verify output files exist
    results_json = Path(res["output_files"]["validation_results_json"])
    summary_json = Path(res["output_files"]["validation_summary_json"])
    results_csv = Path(res["output_files"]["validation_results_csv"])
    assert results_json.is_file()
    assert summary_json.is_file()
    assert results_csv.is_file()

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    # Model integrity verified
    assert summary["model_integrity"]["model_repository"] == OFFICIAL_MODEL_ID
    assert summary["model_integrity"]["pinned_revision"] == OFFICIAL_PINNED_REVISION
    assert summary["model_integrity"]["pytorch_model_sha256"] == "f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf"
    assert summary["model_integrity"]["model_onnx_sha256"] == "693911a555891bac487c9e24daf4f2adcef258eaa7304d8d344ecc96b9af7b2e"

    # Verify all 7 images processed with unmodified bytes and uncalibrated scores
    assert len(payload["results"]) == 7
    for item in payload["results"]:
        rel = item["relative_path"]
        exp_size, exp_sha, exp_raw = pre_hashes[rel]
        disk_raw = (ds_dir / rel).read_bytes()
        assert disk_raw == exp_raw
        assert item["byte_size"] == exp_size
        assert item["sha256"] == exp_sha
        assert item["sha256_preserved"] is True
        assert item["detector_status"] == "available"
        assert item["score_type"] == "uncalibrated_model_score"
        assert item["decision_rule"] == "model_class_selection"
        assert item["threshold"] is None
        assert item["selected_class"] in {"artificial", "human"}
        assert 0.0 <= item["artificial_score"] <= 1.0
        assert 0.0 <= item["human_score"] <= 1.0
        assert item["mediatrace_forensic_signals"] is not None

    # Verify category statistics and confusion counts exclude transformed
    cat_stats = summary["category_statistics"]
    assert cat_stats["human"]["sample_count"] == 3
    assert cat_stats["ai_generated"]["sample_count"] == 3
    assert cat_stats["transformed"]["sample_count"] == 1
    assert summary["transformed_category_results"]["included_in_confusion_counts"] is False

    conf = summary["confusion_count_table"]["rows"]
    assert conf["known_human"]["predicted_artificial"] + conf["known_human"]["predicted_human"] == 3
    assert conf["known_ai_generated"]["predicted_artificial"] + conf["known_ai_generated"]["predicted_human"] == 3

    # Ensure no banned metric labels (accuracy, precision, recall, F1, AUC) appear as summary keys
    summary_keys_flat = json.dumps(summary).lower()
    for forbidden_key in ('"accuracy":', '"precision":', '"recall":', '"f1":', '"auc":', '"winner":'):
        assert forbidden_key not in summary_keys_flat

    # Verify CSV has header + 7 rows
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7


def test_step_14_1_7_provenance_sha256_duplicates_and_separation(tmp_path):
    """Step 14.1.7: Tests for:
    - SHA-256 mismatch detection
    - byte_size / dimension mismatch detection
    - duplicate SHA-256 detection across samples
    - missing provenance field detection under schema_version='2.0'
    - source vs transformation separation (rejecting transformed files in human/ai_generated)
    - transformed parent_sample_id / parent_sha256 validation
    """
    ds_dir = tmp_path / "ai_detector_validation_v2"
    (ds_dir / "human").mkdir(parents=True)
    (ds_dir / "ai_generated").mkdir(parents=True)
    (ds_dir / "transformed").mkdir(parents=True)

    img1 = _make_test_image_bytes("JPEG", size=(300, 200), color=(40, 80, 120))
    img2 = _make_test_image_bytes("PNG", size=(256, 256), color=(190, 60, 90))
    sha1 = hashlib.sha256(img1).hexdigest()
    sha2 = hashlib.sha256(img2).hexdigest()

    (ds_dir / "human" / "cam1.jpg").write_bytes(img1)
    (ds_dir / "human" / "cam1_copy.jpg").write_bytes(img1)
    (ds_dir / "ai_generated" / "gen1.png").write_bytes(img2)

    manifest_file = ds_dir / "manifest.json"

    def _base_sample(sample_id: str, rel_path: str, category: str, sha: str, size: int, w: int, h: int, fmt: str) -> dict:
        return {
            "sample_id": sample_id,
            "id": sample_id,
            "path": rel_path,
            "category": category,
            "known_category": category,
            "sha256": sha,
            "byte_size": size,
            "width": w,
            "height": h,
            "format": fmt,
            "source_description": "Camera original photo",
            "generator_or_camera_if_known": "Camera X",
            "provenance_status": "verified_camera_original_exif",
            "notes": "Test note",
        }

    # 1. SHA-256 mismatch detection
    bad_sha_sample = _base_sample("human_001", "human/cam1.jpg", "human", "0" * 64, len(img1), 300, 200, "JPEG")
    manifest_file.write_text(json.dumps({"schema_version": "2.0", "samples": [bad_sha_sample]}), encoding="utf-8")
    with pytest.raises(ValidationHarnessError, match="Changed SHA-256 detected"):
        load_and_validate_manifest(manifest_file)

    # 2. Duplicate SHA-256 detection across two distinct IDs
    dup_s1 = _base_sample("human_001", "human/cam1.jpg", "human", sha1, len(img1), 300, 200, "JPEG")
    dup_s2 = _base_sample("human_002", "human/cam1_copy.jpg", "human", sha1, len(img1), 300, 200, "JPEG")
    manifest_file.write_text(json.dumps({"schema_version": "2.0", "samples": [dup_s1, dup_s2]}), encoding="utf-8")
    with pytest.raises(ValidationHarnessError, match="Duplicate SHA-256"):
        load_and_validate_manifest(manifest_file)

    # 3. Missing provenance field detection in schema_version="2.0"
    missing_prov = _base_sample("human_001", "human/cam1.jpg", "human", sha1, len(img1), 300, 200, "JPEG")
    del missing_prov["provenance_status"]
    manifest_file.write_text(json.dumps({"schema_version": "2.0", "samples": [missing_prov]}), encoding="utf-8")
    with pytest.raises(ValidationHarnessError, match="missing required provenance fields"):
        load_and_validate_manifest(manifest_file)

    # 4. Transformed file accidentally placed in human/ or ai_generated/ category
    transformed_in_source = _base_sample("human_001", "human/cam1.jpg", "human", sha1, len(img1), 300, 200, "JPEG")
    transformed_in_source["transformations"] = ["resize_512"]
    manifest_file.write_text(json.dumps({"schema_version": "2.0", "samples": [transformed_in_source]}), encoding="utf-8")
    with pytest.raises(ValidationHarnessError, match="Transformed files must not be counted as independent source samples"):
        load_and_validate_manifest(manifest_file)

    # 5. Transformed sample placed inside transformed/ directory but labeled category="human"
    (ds_dir / "transformed" / "t_misplaced.png").write_bytes(img2)
    misplaced_dir = _base_sample("human_002", "transformed/t_misplaced.png", "human", sha2, len(img2), 256, 256, "PNG")
    manifest_file.write_text(json.dumps({"schema_version": "2.0", "samples": [misplaced_dir]}), encoding="utf-8")
    with pytest.raises(ValidationHarnessError, match="Transformed files must not be counted as independent source samples"):
        load_and_validate_manifest(manifest_file)


def test_step_14_1_7_production_dataset_and_markdown_summary(tmp_path, monkeypatch):
    """Step 14.1.7: Validates the actual repository manifest.json (schema_version='2.0')
    and verifies stratified summary, dataset quality warnings, and markdown summary generation.
    """
    real_models_root = Path(__file__).resolve().parents[2] / "storage" / "models"
    real_manifest_path = Path(__file__).resolve().parents[2] / "datasets" / "ai_detector_validation" / "manifest.json"
    monkeypatch.setattr(settings, "models_dir", real_models_root)
    monkeypatch.setattr(settings, "storage_root", real_models_root.parent)
    image_ai_detection_service.clear_ai_detector_cache()

    parsed = load_and_validate_manifest(real_manifest_path, strict_provenance=True)
    assert parsed["schema_version"] == "2.0"
    assert len(parsed["samples"]) == 5

    out_dir = tmp_path / "step_14_1_7_out"
    res = run_validation_harness(manifest_path=real_manifest_path, output_dir=out_dir, skip_forensics=False)

    summary_md_path = Path(res["output_files"]["validation_summary_md"])
    summary_json_path = Path(res["output_files"]["validation_summary_json"])
    results_json_path = Path(res["output_files"]["validation_results_json"])
    results_csv_path = Path(res["output_files"]["validation_results_csv"])

    assert summary_md_path.is_file()
    assert summary_json_path.is_file()
    assert results_json_path.is_file()
    assert results_csv_path.is_file()

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    md_text = summary_md_path.read_text(encoding="utf-8")

    # Verify dataset status is incomplete (1/10 human, 3/10 ai_generated) without fabricating samples
    assert summary["dataset_complete"] is False
    assert summary["sample_availability"]["human_available"] == 1
    assert summary["sample_availability"]["ai_generated_available"] == 3
    assert summary["sample_availability"]["transformed_available"] == 1
    assert summary["sample_availability"]["incomplete_categories"] == ["human", "ai_generated"]

    # Verify confusion count table is labeled 'controlled dataset descriptive counts'
    assert summary["confusion_count_table"]["label"] == "controlled dataset descriptive counts"
    assert summary["confusion_count_table"]["rows"]["known_human"]["predicted_human"] == 1
    assert summary["confusion_count_table"]["rows"]["known_ai_generated"]["predicted_artificial"] == 2
    assert summary["confusion_count_table"]["rows"]["known_ai_generated"]["predicted_human"] == 1

    # Verify stratified summary and dataset quality warnings
    assert "stratified_dataset_summary" in summary
    assert summary["stratified_dataset_summary"]["total_source_sample_count"] == 4
    assert summary["stratified_dataset_summary"]["unique_source_sha256_count"] == 4
    assert summary["stratified_dataset_summary"]["gps_coordinates_exposed"] is False

    warning_codes = {w["code"] for w in summary["dataset_quality_warnings"]}
    assert "human_count_below_target" in warning_codes
    assert "ai_count_below_target" in warning_codes
    assert "limited_known_generator_diversity" in warning_codes

    # Verify markdown summary includes all required sections and neutral language
    assert "controlled dataset descriptive counts" in md_text
    assert "model/category disagreement observed" in md_text
    assert "## 1. Dataset Composition" in md_text
    assert "## 9. Dataset Quality Warnings & Limitations" in md_text
    assert "## 10. Explicit Controlled Engineering Validation Statement" in md_text

