import hashlib
import io
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import settings
from app.main import app
from app.schemas.ai_detection import (
    OFFICIAL_MODEL_ARCHITECTURE,
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_LICENSE,
    OFFICIAL_PINNED_REVISION,
)
from app.services.analysis import image_ai_detection_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_storage_and_detector(tmp_path: Path, monkeypatch):
    """Isolate storage directories and reset AI detector state per test."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    uploads.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "storage_root", tmp_path)
    monkeypatch.setattr(settings, "uploads_dir", uploads)
    monkeypatch.setattr(settings, "results_dir", results)
    monkeypatch.setattr(settings, "reports_dir", reports)
    monkeypatch.setattr(settings, "models_dir", models)

    monkeypatch.setattr(settings, "ai_image_detector_enabled", True)
    monkeypatch.setattr(settings, "ai_image_detector_model_id", OFFICIAL_MODEL_ID)
    monkeypatch.setattr(settings, "ai_image_detector_model_version", OFFICIAL_PINNED_REVISION)
    monkeypatch.setattr(settings, "ai_image_detector_threshold", None)

    image_ai_detection_service.set_custom_predictor_for_testing(None)
    image_ai_detection_service.clear_ai_detector_cache()
    yield
    image_ai_detection_service.set_custom_predictor_for_testing(None)
    image_ai_detection_service.clear_ai_detector_cache()


def _make_test_image(
    fmt: str = "JPEG",
    size: Tuple[int, int] = (320, 240),
    color: Tuple[int, int, int] = (70, 120, 170),
    exif_bytes: bytes = b"",
    splice_patch: bool = False,
) -> bytes:
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=(190, 200, 210), width=2)
    if splice_patch:
        arr = np.array(img, dtype=np.uint8)
        rng = np.random.default_rng(42)
        patch = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        arr[60:140, 80:160] = patch
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    save_kwargs = {}
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = 92
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def test_a_artificial_class_interpretation():
    """Test A: Artificial class interpretation includes required flags, uncalibrated score, and non-provenance guarantees."""
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {
            "score": 0.9963,
            "human_score": 0.0037,
            "raw_class": "artificial",
        }
    )
    img_bytes = _make_test_image(fmt="JPEG", size=(320, 240))
    sha_before = hashlib.sha256(img_bytes).hexdigest()

    up = client.post("/api/media/upload", files={"file": ("art_case.jpg", io.BytesIO(img_bytes), "image/jpeg")})
    assert up.status_code == 201
    case_id = up.json()["case_id"]
    stored_path = settings.uploads_dir / up.json()["stored_filename"]

    res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res.status_code == 200
    det = res.json()["manipulation_detection"]
    ai_det = det["ai_generation_detection"]
    interp = ai_det["ai_detector_interpretation"]

    assert ai_det["status"] == "available"
    assert interp["status"] == "available"
    assert interp["model_output"]["selected_class"] == "artificial"
    assert interp["model_output"]["selected_class_id"] == 0
    assert interp["model_output"]["artificial_score"] == 0.9963
    assert interp["model_output"]["human_score"] == 0.0037
    assert interp["model_output"]["score_type"] == "uncalibrated_model_score"
    assert interp["model_output"]["threshold"] is None
    assert interp["model_output"]["decision_rule"] == "model_class_selection"

    flags = interp["interpretation_flags"]
    assert "model_available" in flags
    assert "uncalibrated_score" in flags
    assert "no_threshold" in flags
    assert "domain_shift_risk" in flags
    assert "transformation_sensitivity" in flags
    assert "not_provenance_evidence" in flags
    assert "not_authenticity_proof" in flags
    assert "single_model_signal" in flags
    assert "model_class_artificial" in flags

    assert "does not establish that the image was AI-generated" in interp["forensic_meaning"]
    assert "establish provenance or authenticity" in interp["forensic_meaning"]
    assert len(interp["limitations"]) >= 10
    assert len(interp["prohibited_inference"]) == 4

    # Original file integrity preserved
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() == sha_before


def test_b_human_class_interpretation():
    """Test B: Human class interpretation includes required flags and never claims camera authenticity."""
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {
            "score": 0.0897,
            "human_score": 0.9103,
            "raw_class": "human",
        }
    )
    img_bytes = _make_test_image(fmt="JPEG", size=(320, 240))
    up = client.post("/api/media/upload", files={"file": ("hum_case.jpg", io.BytesIO(img_bytes), "image/jpeg")})
    case_id = up.json()["case_id"]

    res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res.status_code == 200
    ai_det = res.json()["manipulation_detection"]["ai_generation_detection"]
    interp = ai_det["ai_detector_interpretation"]

    assert interp["status"] == "available"
    assert interp["model_output"]["selected_class"] == "human"
    assert interp["model_output"]["selected_class_id"] == 1
    assert interp["model_output"]["artificial_score"] == 0.0897
    assert interp["model_output"]["human_score"] == 0.9103
    assert interp["model_output"]["score_type"] == "uncalibrated_model_score"
    assert interp["model_output"]["threshold"] is None
    assert interp["model_output"]["decision_rule"] == "model_class_selection"

    flags = interp["interpretation_flags"]
    assert "model_class_human" in flags
    assert "not_provenance_evidence" in flags
    assert "not_authenticity_proof" in flags
    assert "does not establish that the image is camera-original or authentic" in interp["forensic_meaning"]
    assert "does not rule out manipulation or AI generation" in interp["forensic_meaning"]


def test_c_unavailable_model_handling():
    """Test C: Unavailable model runtime returns status='not_available', no fake score, and model_unavailable flag."""
    img_bytes = _make_test_image(fmt="JPEG", size=(320, 240))
    up = client.post("/api/media/upload", files={"file": ("unavail_case.jpg", io.BytesIO(img_bytes), "image/jpeg")})
    case_id = up.json()["case_id"]

    res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res.status_code == 200
    ai_det = res.json()["manipulation_detection"]["ai_generation_detection"]
    interp = ai_det["ai_detector_interpretation"]

    assert ai_det["status"] == "not_available"
    assert ai_det["prediction"] is None
    assert ai_det["artificial_score"] is None
    assert ai_det["human_score"] is None
    assert interp["status"] == "not_available"
    assert interp["model_output"]["artificial_score"] is None
    assert interp["model_output"]["human_score"] is None
    assert "model_unavailable" in interp["interpretation_flags"]
    assert "not_provenance_evidence" in interp["interpretation_flags"]
    assert "not_authenticity_proof" in interp["interpretation_flags"]


def test_d_and_e_transformation_context_with_and_without_evidence():
    """Test D & E: Transformation context is False when no evidence exists, and True when reference/processing evidence exists."""
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.8512, "human_score": 0.1488, "raw_class": "artificial"}
    )

    # E. Single image with no transformation evidence -> transformation_context.documented is False
    clean_bytes = _make_test_image(fmt="PNG", size=(320, 240))
    up_clean = client.post("/api/media/upload", files={"file": ("clean.png", io.BytesIO(clean_bytes), "image/png")})
    clean_id = up_clean.json()["case_id"]

    det_clean = client.post(f"/api/analysis/image/{clean_id}/detect").json()["manipulation_detection"]
    tx_clean = det_clean["ai_generation_detection"]["transformation_context"]
    assert tx_clean["documented"] is False
    assert tx_clean["signals"] == []
    assert "documented_transformation_context" not in det_clean["ai_generation_detection"]["ai_detector_interpretation"]["interpretation_flags"]

    # D. Image with Software EXIF metadata + reference comparison with different dimensions -> transformation_context.documented is True
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 2024"
    edited_bytes = _make_test_image(fmt="JPEG", size=(240, 180), exif_bytes=exif.tobytes())
    ref_bytes = _make_test_image(fmt="JPEG", size=(480, 360))

    up_edited = client.post("/api/media/upload", files={"file": ("edited.jpg", io.BytesIO(edited_bytes), "image/jpeg")})
    edited_id = up_edited.json()["case_id"]

    # Upload reference image and run reference comparison first
    ref_up = client.post(
        f"/api/comparison/image/{edited_id}/reference",
        files={"file": ("ref.jpg", io.BytesIO(ref_bytes), "image/jpeg")},
    )
    assert ref_up.status_code == 201
    cmp_res = client.post(f"/api/comparison/image/{edited_id}")
    assert cmp_res.status_code == 200

    det_edited = client.post(f"/api/analysis/image/{edited_id}/detect").json()["manipulation_detection"]
    tx_edited = det_edited["ai_generation_detection"]["transformation_context"]
    assert tx_edited["documented"] is True
    assert "dimension_change_observed" in tx_edited["signals"]
    assert "reference_difference_observed" in tx_edited["signals"]
    assert "processing_metadata_present" in tx_edited["signals"]
    assert "documented_transformation_context" in det_edited["ai_generation_detection"]["ai_detector_interpretation"]["interpretation_flags"]


def test_f_and_g_cross_analysis_disagreement_and_evidence_restraint():
    """Test F & G: Verify cross-analysis disagreement handling and that AI detector alone never forces potential_manipulation or multiple_forensic_signals."""
    # Case A: AI detector = artificial, classical forensics = no_strong_indicators
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.9963, "human_score": 0.0037, "raw_class": "artificial"}
    )
    exif_a = Image.Exif()
    exif_a[271] = "Canon"
    exif_a[272] = "EOS R5"
    img_a = _make_test_image(fmt="JPEG", size=(320, 240), exif_bytes=exif_a.tobytes())
    up_a = client.post("/api/media/upload", files={"file": ("case_a.jpg", io.BytesIO(img_a), "image/jpeg")})
    case_a = up_a.json()["case_id"]

    det_a = client.post(f"/api/analysis/image/{case_a}/detect").json()["manipulation_detection"]
    assert det_a["classical_forensics"]["overall_assessment"] == "no_strong_indicators"
    # G. AI detector alone must NOT force potential_manipulation
    assert det_a["overall_assessment"] == "no_strong_indicators"
    assert (
        det_a["assessment"]["cross_analysis_interpretation"]
        == "AI detector selected artificial while other available forensic signals do not independently establish manipulation."
    )

    ev_a = client.post(f"/api/evidence/{case_a}/generate").json()["evidence"]
    # G. AI detector alone must NOT force multiple_forensic_signals
    assert ev_a["assessment"] == "no_significant_evidence"
    disagree_a = [item for item in ev_a["items"] if item["type"] == "ai_detector_cross_analysis_disagreement"]
    assert len(disagree_a) == 1
    assert disagree_a[0]["strength"] == "informational"
    assert (
        "AI detector selected artificial while other available forensic signals do not independently establish manipulation."
        in disagree_a[0]["description"]
    )

    # Case B: AI detector = human, localization = localized_anomaly: True
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.0110, "human_score": 0.9890, "raw_class": "human"}
    )
    img_b = _make_test_image(fmt="JPEG", size=(320, 240), splice_patch=True)
    up_b = client.post("/api/media/upload", files={"file": ("case_b.jpg", io.BytesIO(img_b), "image/jpeg")})
    case_b = up_b.json()["case_id"]

    loc_b = client.post(f"/api/analysis/image/{case_b}/localize").json()["localization"]
    assert loc_b["localized_anomaly"] is True

    det_b = client.post(f"/api/analysis/image/{case_b}/detect").json()["manipulation_detection"]
    assert (
        det_b["assessment"]["cross_analysis_interpretation"]
        == "The AI detector selected human; separate localization analysis identified candidate anomalous regions. These are distinct signals and neither result establishes authenticity or manipulation on its own."
    )

    ev_b = client.post(f"/api/evidence/{case_b}/generate").json()["evidence"]
    disagree_b = [item for item in ev_b["items"] if item["type"] == "ai_detector_cross_analysis_disagreement"]
    assert len(disagree_b) == 1
    assert (
        "The AI detector selected human; separate localization analysis identified candidate anomalous regions."
        in disagree_b[0]["description"]
    )


def test_h_and_i_report_pdf_consistency_and_history_reconstruction_safety():
    """Test H & I: Verify JSON report, PDF report, and History reconstruction adhere to Step 14.1.6 hardening rules."""
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.9963, "human_score": 0.0037, "raw_class": "artificial"}
    )
    img_bytes = _make_test_image(fmt="JPEG", size=(320, 240))
    up = client.post("/api/media/upload", files={"file": ("report_case.jpg", io.BytesIO(img_bytes), "image/jpeg")})
    case_id = up.json()["case_id"]

    client.post(f"/api/analysis/image/{case_id}/detect")
    client.post(f"/api/evidence/{case_id}/generate")

    # I. History reconstruction safety
    hist = client.post(f"/api/history/{case_id}/reconstruct").json()["history"]
    prohibited_titles = {"ai generation occurred", "image created by ai", "camera capture confirmed"}
    for ev in hist["events"]:
        assert ev["title"].lower() not in prohibited_titles
        assert "ai generation occurred" not in ev["description"].lower()
        assert "camera capture confirmed" not in ev["description"].lower()

    analysis_events = [
        ev for ev in hist["events"]
        if ev["event_type"] == "forensic_analysis_event" and ev["title"] == "AI image detector analysis performed"
    ]
    assert len(analysis_events) == 1
    assert analysis_events[0]["status"] == "observed"
    assert "not a historical creation event" in analysis_events[0]["reasoning"]

    # H. Report and PDF consistency
    rep = client.post(f"/api/reports/{case_id}/generate").json()
    rep_ai = rep["ai_generation_detection"]
    assert rep_ai["detector"]["model_repository"] == OFFICIAL_MODEL_ID
    assert rep_ai["detector"]["pinned_revision"] == OFFICIAL_PINNED_REVISION
    assert rep_ai["detector"]["license"] == OFFICIAL_MODEL_LICENSE
    assert rep_ai["selected_class"] == "artificial"
    assert rep_ai["artificial_score"] == 0.9963
    assert rep_ai["human_score"] == 0.0037
    assert rep_ai["ai_detector_interpretation"]["model_output"]["score_type"] == "uncalibrated_model_score"
    assert rep_ai["ai_detector_interpretation"]["model_output"]["threshold"] is None
    assert len(rep_ai["limitations"]) >= 10

    pdf_res = client.post(f"/api/reports/{case_id}/generate-pdf")
    assert pdf_res.status_code == 200
    pdf_text = pdf_res.content.decode("latin-1", errors="ignore")
    assert OFFICIAL_MODEL_ID in pdf_text
    assert OFFICIAL_PINNED_REVISION in pdf_text
    assert "CC BY-ND 4.0" in pdf_text
    assert "uncalibrated_model_score" in pdf_text
    assert "threshold" in pdf_text.lower()
    assert "null" in pdf_text.lower()
    assert "0.9963" in pdf_text
    assert "0.0037" in pdf_text



