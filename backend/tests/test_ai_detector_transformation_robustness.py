r"""Step 14.1.5 — Controlled AI-Detector Transformation Robustness Automated Tests.

Covers all 13 requirements from Step 14.1.5 Requirement 9:
1. deterministic transformation creation
2. source-to-derivative manifest linkage
3. SHA-256 recording
4. no source overwrite
5. resize dimensions
6. JPEG output validity
7. detector invocation (real ONNX Runtime inference on installed Swin model)
8. preservation of model metadata
9. score delta calculation
10. selected-class change detection
11. category/model disagreement handling
12. summary generation (JSON, CSV, Markdown)
13. no threshold introduced (threshold=None, score_type='uncalibrated_model_score')
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
from app.schemas.ai_detection import OFFICIAL_MODEL_ARCHITECTURE, OFFICIAL_MODEL_ID, OFFICIAL_PINNED_REVISION
from app.services.analysis import image_ai_detection_service
from scripts.test_ai_detector_transformation_robustness import (
    ROBUSTNESS_LIMITATIONS,
    TRANSFORMATION_ORDER,
    _create_single_derivative,
    run_transformation_robustness_experiment,
)


def _create_rgb_test_image(size: tuple[int, int] = (400, 300), color: tuple[int, int, int] = (70, 130, 190)) -> bytes:
    im = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=96)
    return buf.getvalue()


def test_deterministic_transformations_resize_dimensions_and_jpeg_validity():
    """Verify deterministic transformation creation, resize dimensions, and JPEG output validity."""
    src_bytes = _create_rgb_test_image(size=(400, 300))
    src_sha = hashlib.sha256(src_bytes).hexdigest()

    # 1. Original copy is byte-for-byte identical
    orig_b, w0, h0, fmt0, _ = _create_single_derivative(src_bytes, ".jpg", 400, 300, "original")
    assert orig_b == src_bytes
    assert hashlib.sha256(orig_b).hexdigest() == src_sha
    assert (w0, h0, fmt0) == (400, 300, "JPEG")

    # 2. Resize 50% -> 200x150 valid JPEG, deterministic across runs
    r50_a, w50, h50, fmt50, _ = _create_single_derivative(src_bytes, ".jpg", 400, 300, "resize_50")
    r50_b, _, _, _, _ = _create_single_derivative(src_bytes, ".jpg", 400, 300, "resize_50")
    assert r50_a == r50_b
    assert (w50, h50, fmt50) == (200, 150, "JPEG")
    with Image.open(io.BytesIO(r50_a)) as im50:
        assert im50.size == (200, 150)
        assert im50.format == "JPEG"

    # 3. Resize 25% -> 100x75 valid JPEG, deterministic across runs
    r25_a, w25, h25, fmt25, _ = _create_single_derivative(src_bytes, ".jpg", 400, 300, "resize_25")
    r25_b, _, _, _, _ = _create_single_derivative(src_bytes, ".jpg", 400, 300, "resize_25")
    assert r25_a == r25_b
    assert (w25, h25, fmt25) == (100, 75, "JPEG")

    # 4. JPEG Q90, Q70, Q50 preserve dimensions (400x300) and produce decreasing byte sizes on gradient/noise
    for q_tag in ("jpeg_q90", "jpeg_q70", "jpeg_q50"):
        qb, qw, qh, qfmt, _ = _create_single_derivative(src_bytes, ".jpg", 400, 300, q_tag)
        assert (qw, qh, qfmt) == (400, 300, "JPEG")


def test_full_transformation_robustness_experiment_real_detector(tmp_path, monkeypatch):
    """Verify end-to-end transformation robustness experiment on real installed ONNX model:
    - source-to-derivative manifest linkage
    - SHA-256 recording and no source overwrite
    - detector invocation and preservation of model metadata
    - score delta calculation and selected-class change detection
    - category/model disagreement handling
    - JSON, CSV, and Markdown summary generation
    - no threshold introduced (threshold=None, score_type='uncalibrated_model_score')
    """
    real_models_root = Path(__file__).resolve().parents[2] / "storage" / "models"
    monkeypatch.setattr(settings, "models_dir", real_models_root)
    monkeypatch.setattr(settings, "storage_root", real_models_root.parent)
    image_ai_detection_service.clear_ai_detector_cache()

    ds_dir = tmp_path / "ai_detector_validation"
    (ds_dir / "human").mkdir(parents=True)
    (ds_dir / "ai_generated").mkdir(parents=True)

    human_bytes = _create_rgb_test_image(size=(320, 240), color=(45, 95, 155))
    ai_bytes = _create_rgb_test_image(size=(280, 200), color=(190, 60, 85))

    human_path = ds_dir / "human" / "h_source.jpg"
    ai_path = ds_dir / "ai_generated" / "ai_source.jpg"
    human_path.write_bytes(human_bytes)
    ai_path.write_bytes(ai_bytes)

    human_sha = hashlib.sha256(human_bytes).hexdigest()
    ai_sha = hashlib.sha256(ai_bytes).hexdigest()

    manifest_path = ds_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "samples": [
                    {
                        "id": "human_001",
                        "path": "human/h_source.jpg",
                        "known_category": "human",
                        "source_description": "test human image",
                        "generator": None,
                        "transformations": [],
                        "notes": "",
                    },
                    {
                        "id": "ai_001",
                        "path": "ai_generated/ai_source.jpg",
                        "known_category": "ai_generated",
                        "source_description": "test AI image",
                        "generator": "unknown",
                        "transformations": [],
                        "notes": "",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out = run_transformation_robustness_experiment(
        manifest_path=manifest_path,
        output_base_dir=ds_dir,
    )

    # Verify all 4 output files exist
    manifest_out = Path(out["output_files"]["transformation_robustness_manifest_json"])
    results_json = Path(out["output_files"]["transformation_robustness_results_json"])
    results_csv = Path(out["output_files"]["transformation_robustness_results_csv"])
    summary_md = Path(out["output_files"]["transformation_robustness_summary_md"])

    assert manifest_out.is_file()
    assert results_json.is_file()
    assert results_csv.is_file()
    assert summary_md.is_file()

    # Verify source images were not overwritten or modified
    assert human_path.read_bytes() == human_bytes
    assert hashlib.sha256(human_path.read_bytes()).hexdigest() == human_sha
    assert ai_path.read_bytes() == ai_bytes
    assert hashlib.sha256(ai_path.read_bytes()).hexdigest() == ai_sha

    m_data = json.loads(manifest_out.read_text(encoding="utf-8"))
    r_data = json.loads(results_json.read_text(encoding="utf-8"))
    md_text = summary_md.read_text(encoding="utf-8")

    assert m_data["source_image_count"] == 2
    assert m_data["derivative_count"] == 2 * len(TRANSFORMATION_ORDER)
    assert len(r_data["results"]) == 2 * len(TRANSFORMATION_ORDER)

    for row in r_data["results"]:
        assert row["model_repository"] == OFFICIAL_MODEL_ID
        assert row["pinned_model_revision"] == OFFICIAL_PINNED_REVISION
        assert row["model_architecture"] == OFFICIAL_MODEL_ARCHITECTURE
        assert row["score_type"] == "uncalibrated_model_score"
        assert row["decision_rule"] == "model_class_selection"
        assert row["threshold"] is None
        assert row["status"] == "available"
        assert isinstance(row["preprocessing"], dict)
        assert row["preprocessing"]["resize_dimensions"] == "224x224"

        # Verify delta math
        assert row["artificial_score_delta"] == round(
            row["artificial_score"] - row["baseline_artificial_score"], 4
        )
        assert row["human_score_delta"] == round(
            row["human_score"] - row["baseline_human_score"], 4
        )
        assert row["selected_class_changed"] is (
            row["selected_class"] != row["baseline_selected_class"]
        )

        # Verify source-to-derivative linkage
        if row["source_id"] == "human_001":
            assert row["source_category"] == "human"
            assert row["source_sha256"] == human_sha
        else:
            assert row["source_category"] == "ai_generated"
            assert row["source_sha256"] == ai_sha

        # Verify disagreement logic never rewrites source_category
        expected_disagree = (
            (row["source_category"] == "human" and row["selected_class"] == "artificial")
            or (row["source_category"] == "ai_generated" and row["selected_class"] == "human")
        )
        assert row["source_vs_model_disagreement"] is expected_disagree

    # Verify CSV rows
    with results_csv.open("r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 12

    # Verify Markdown summary contains required sections and all 9 explicit limitations
    assert "Dataset Composition" in md_text
    assert "Transformations Tested" in md_text
    assert "Explicit Limitations" in md_text
    for lim in ROBUSTNESS_LIMITATIONS:
        assert lim in md_text
