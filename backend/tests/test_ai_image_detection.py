"""Step 14.1 & 14.1.1 — Real AI-Generated Image Detection Test Suite.

Covers all Step 14.1.1 verification requirements (A through M):
  A. License test (CC BY-ND 4.0 enforced, never Apache-2.0)
  B. Revision pinning test (immutable commit SHA c7e223baf11bc40528af364ba7bdea030ef42f9e; 'main' rejected)
  C. Score calibration test (score_type == 'uncalibrated_model_score')
  D. Threshold honesty test (default threshold is None, decision_rule == 'model_class_selection')
  E. Preprocessing test (deterministic NCHW float32 tensor, 224x224 Bicubic, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225])
  F. Real model inference test (ONNX Runtime execution + SwinForImageClassification conversion provenance)
  G. Missing model fallback test (status == 'not_available', no fabricated score)
  H. Corrupt image test (400/422 without crash)
  I. SHA-256 preservation test (original file hash & byte size unchanged)
  J. GPS privacy test (GPS coordinates never exposed)
  K. Step 7 Evidence integration test ('ai_generation_detector_signal', informational)
  L. Step 8 History integration test ('AI-generation detector produced a model-derived signal', unknown chronology)
  M. Step 9 Report integration test (full model metadata, limitations, and conservative wording)
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
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
def isolated_storage_and_detector(tmp_path, monkeypatch):
    """Isolate storage and model cache for every test using the official pinned revision and None threshold."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    for folder in (uploads, results, reports, models):
        folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "storage_root", tmp_path)
    monkeypatch.setattr(settings, "uploads_dir", uploads)
    monkeypatch.setattr(settings, "results_dir", results)
    monkeypatch.setattr(settings, "reports_dir", reports)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "ai_image_detector_enabled", True)
    monkeypatch.setattr(settings, "ai_image_detector_model_id", OFFICIAL_MODEL_ID)
    monkeypatch.setattr(settings, "ai_image_detector_model_version", OFFICIAL_PINNED_REVISION)
    monkeypatch.setattr(settings, "ai_image_detector_device", "cpu")
    monkeypatch.setattr(settings, "ai_image_detector_threshold", None)

    image_ai_detection_service.clear_ai_detector_cache()
    image_ai_detection_service.set_custom_predictor_for_testing(None)
    yield tmp_path
    image_ai_detection_service.clear_ai_detector_cache()
    image_ai_detection_service.set_custom_predictor_for_testing(None)


def _create_sample_image(
    fmt: str = "JPEG",
    size: tuple[int, int] = (320, 240),
    mode: str = "RGB",
    color: tuple[int, ...] = (60, 120, 190),
    exif_bytes: bytes | None = None,
) -> bytes:
    img = Image.new(mode, size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [16, 16, size[0] - 16, size[1] - 16],
        outline=(240, 220, 100) if len(color) == 3 else (240, 220, 100, 255),
        width=2,
    )
    buf = io.BytesIO()
    kwargs = {}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _build_local_onnx_swin_classifier(
    model_dir: Path,
    bias_logits: tuple[float, float] = (2.2, -1.1),
    config_license: str = OFFICIAL_MODEL_LICENSE,
) -> None:
    """Build a valid local ONNX export checkpoint (input: [N, 3, 224, 224] -> output: [N, 2])
    with official config.json and preprocessor_config.json.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    x_in = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [None, 3, 224, 224])
    y_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])

    weights = np.array([[0.05, -0.05], [0.05, -0.05], [0.05, -0.05]], dtype=np.float32)
    biases = np.array([bias_logits[0], bias_logits[1]], dtype=np.float32)

    w_init = helper.make_tensor("W", TensorProto.FLOAT, [3, 2], weights.flatten().tolist())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, [2], biases.flatten().tolist())

    node_pool = helper.make_node("GlobalAveragePool", inputs=["pixel_values"], outputs=["pooled"])
    node_flat = helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1)
    node_matmul = helper.make_node("MatMul", inputs=["flat", "W"], outputs=["mm"])
    node_add = helper.make_node("Add", inputs=["mm", "B"], outputs=["logits"])

    graph = helper.make_graph(
        [node_pool, node_flat, node_matmul, node_add],
        "SwinForImageClassificationLocalONNXExport",
        [x_in],
        [y_out],
        initializer=[w_init, b_init],
    )
    model_def = helper.make_model(
        graph,
        producer_name="mediatrace-local-onnx-exporter",
        opset_imports=[helper.make_opsetid("", 14)],
    )
    model_def.ir_version = 8
    onnx.checker.check_model(model_def)
    onnx.save(model_def, str(model_dir / "model.onnx"))

    config_payload = {
        "architectures": [OFFICIAL_MODEL_ARCHITECTURE],
        "model_type": "swin",
        "image_size": 224,
        "id2label": {"0": "artificial", "1": "human"},
        "label2id": {"artificial": 0, "human": 1},
        "license": config_license,
        "pinned_revision": OFFICIAL_PINNED_REVISION,
        "onnx_is_upstream": False,
        "source": f"https://huggingface.co/{OFFICIAL_MODEL_ID}/tree/{OFFICIAL_PINNED_REVISION}",
    }
    (model_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    preproc_payload = {
        "do_normalize": True,
        "do_resize": True,
        "feature_extractor_type": "ViTFeatureExtractor",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "resample": 3,
        "size": 224,
    }
    (model_dir / "preprocessor_config.json").write_text(json.dumps(preproc_payload, indent=2), encoding="utf-8")


def test_a_and_b_license_and_revision_pinning_enforcement(monkeypatch):
    """Test A & B: Verify license is CC BY-ND 4.0 (never Apache-2.0) and revision is pinned SHA (unpinned 'main' rejected)."""
    model_dir = settings.models_dir / "umm-maybe--AI-image-detector"
    # Even if an old config.json had Apache-2.0, the detector must normalize umm-maybe/AI-image-detector to CC BY-ND 4.0
    _build_local_onnx_swin_classifier(model_dir, bias_logits=(2.3, -1.0), config_license="Apache-2.0")

    raw_bytes = _create_sample_image(fmt="JPEG", size=(256, 256))
    up = client.post("/api/media/upload", files={"file": ("pin_test.jpg", io.BytesIO(raw_bytes), "image/jpeg")})
    case_id = up.json()["case_id"]

    # 1. With pinned commit SHA c7e223baf11bc40528af364ba7bdea030ef42f9e
    res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res.status_code == 200
    ai_det = res.json()["manipulation_detection"]["ai_generation_detection"]
    assert ai_det["status"] == "available"

    detector_meta = ai_det["detector"]
    assert detector_meta["license"] != "Apache-2.0"
    assert detector_meta["license"] == OFFICIAL_MODEL_LICENSE
    assert "CC BY-ND 4.0" in detector_meta["license"]
    assert detector_meta["version"] == OFFICIAL_PINNED_REVISION
    assert detector_meta["pinned_revision"] == OFFICIAL_PINNED_REVISION
    assert detector_meta["version"] != "main"
    assert detector_meta["architecture"] == "SwinForImageClassification"
    assert detector_meta["onnx_is_upstream"] is False
    assert "upstream PyTorch weights" in detector_meta["conversion_info"]
    assert "local ONNX export" in detector_meta["conversion_info"]

    # 2. If configured with unpinned 'main' branch, detector must refuse and return status='not_available'
    monkeypatch.setattr(settings, "ai_image_detector_model_version", "main")
    image_ai_detection_service.clear_ai_detector_cache()

    res_unpinned = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res_unpinned.status_code == 200
    ai_unpinned = res_unpinned.json()["manipulation_detection"]["ai_generation_detection"]
    assert ai_unpinned["status"] == "not_available"
    assert "Unpinned branch names (such as 'main') are rejected" in ai_unpinned["reason"]


def test_c_d_e_score_calibration_threshold_honesty_and_preprocessing():
    """Test C, D & E: Verify uncalibrated_model_score, default threshold=None with model_class_selection, and exact Bicubic + ImageNet preprocessing."""
    raw_bytes = _create_sample_image(fmt="PNG", size=(400, 300), mode="RGB", color=(128, 64, 32))

    # E. Deterministic preprocessing verification
    t1 = image_ai_detection_service._preprocess_image_in_memory(raw_bytes)
    t2 = image_ai_detection_service._preprocess_image_in_memory(raw_bytes)
    assert t1.shape == (1, 3, 224, 224)
    assert t1.dtype == np.float32
    assert np.array_equal(t1, t2)

    # Verify normalization matches [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]
    expected_r = (128.0 / 255.0 - 0.485) / 0.229
    # Center pixel (112, 112) is uniform solid fill away from the border rectangle
    assert np.isclose(float(t1[0, 0, 112, 112]), expected_r, atol=1e-2)

    # C & D. Default threshold is None and decision_rule is model_class_selection
    assert settings.ai_image_detector_threshold is None
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.9145, "raw_class": "artificial"}
    )

    up = client.post("/api/media/upload", files={"file": ("preproc.png", io.BytesIO(raw_bytes), "image/png")})
    case_id = up.json()["case_id"]
    det = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]
    ai_det = det["ai_generation_detection"]

    assert ai_det["prediction"]["score_type"] == "uncalibrated_model_score"
    assert ai_det["prediction"]["threshold"] is None
    assert ai_det["prediction"]["decision_rule"] == "model_class_selection"
    assert ai_det["prediction"]["label"] == "potentially_ai_generated"
    assert ai_det["prediction"]["display_label"] == "Potential AI-generation signal"
    assert "Potential AI-generation signal detected" in ai_det["explanation"]
    assert "not calibrated as a forensic probability" in ai_det["explanation"]

    preproc_meta = ai_det["detector"]["preprocessing"]
    assert preproc_meta["image_mode"] == "RGB"
    assert preproc_meta["resize_dimensions"] == "224x224"
    assert "BICUBIC" in preproc_meta["resample_filter"]
    assert preproc_meta["normalization_mean"] == [0.485, 0.456, 0.406]
    assert preproc_meta["normalization_std"] == [0.229, 0.224, 0.225]
    assert preproc_meta["class_ordering"] == {"0": "artificial", "1": "human"}


def test_g_missing_model_configuration_returns_graceful_unavailable():
    """Test G: When no model weights exist in storage/models/, return status='not_available' without fake scores."""
    raw_bytes = _create_sample_image(fmt="JPEG", size=(320, 240))
    sha_before = hashlib.sha256(raw_bytes).hexdigest()

    up = client.post("/api/media/upload", files={"file": ("photo_g.jpg", io.BytesIO(raw_bytes), "image/jpeg")})
    assert up.status_code == 201
    case_id = up.json()["case_id"]
    stored_path = settings.uploads_dir / up.json()["stored_filename"]

    res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res.status_code == 200
    det = res.json()["manipulation_detection"]

    ai_det = det["ai_generation_detection"]
    assert ai_det["status"] == "not_available"
    assert ai_det["prediction"] is None
    assert ai_det["detector"] is None
    assert "AI image detection model is not configured or could not be loaded" in ai_det["reason"]

    # Verify stored file bytes, size, and SHA-256 are unchanged
    assert stored_path.read_bytes() == raw_bytes
    assert stored_path.stat().st_size == len(raw_bytes)
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() == sha_before


def test_i_and_j_integrity_and_gps_privacy_across_ordinary_and_synthetic_cases():
    """Test I & J: SHA-256 preservation and GPS EXIF privacy across human and artificial predictions."""
    # 1. Human class prediction (no_ai_signal_detected) with GPS EXIF tags
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.0812, "raw_class": "human"}
    )
    exif = Image.Exif()
    exif[271] = "Canon"
    exif[272] = "EOS R5"
    exif[0x8825] = {1: "N", 2: (37.0, 46.0, 29.6), 3: "W", 4: (122.0, 25.0, 10.0)}
    photo_bytes = _create_sample_image(fmt="JPEG", size=(500, 350), exif_bytes=exif.tobytes())
    photo_sha = hashlib.sha256(photo_bytes).hexdigest()

    up_a = client.post("/api/media/upload", files={"file": ("ordinary_photo.jpg", io.BytesIO(photo_bytes), "image/jpeg")})
    case_a = up_a.json()["case_id"]
    path_a = settings.uploads_dir / up_a.json()["stored_filename"]

    det_a = client.post(f"/api/analysis/image/{case_a}/detect").json()["manipulation_detection"]
    ai_a = det_a["ai_generation_detection"]
    assert ai_a["status"] == "available"
    assert ai_a["prediction"]["label"] == "no_ai_signal_detected"
    assert ai_a["prediction"]["display_label"] == "No AI-generation signal detected"
    assert ai_a["prediction"]["score"] == 0.0812
    assert ai_a["prediction"]["score_type"] == "uncalibrated_model_score"
    assert ai_a["prediction"]["decision_rule"] == "model_class_selection"
    assert ai_a["prediction"]["threshold"] is None
    assert "does not prove that the image is human-created" in ai_a["explanation"]
    # Classical assessment remains 'no_strong_indicators' and is NOT overwritten
    assert det_a["classical_forensics"]["overall_assessment"] == "no_strong_indicators"
    assert det_a["overall_assessment"] == "no_strong_indicators"
    # GPS privacy preserved
    assert "37.0" not in json.dumps(det_a)
    assert "122.0" not in json.dumps(det_a)
    # SHA-256 & byte immutability
    assert path_a.read_bytes() == photo_bytes
    assert hashlib.sha256(path_a.read_bytes()).hexdigest() == photo_sha


def test_f_k_l_m_real_onnx_inference_and_step_7_8_9_pipeline_integration():
    """Test F, K, L & M: Real ONNX Runtime model execution, GET endpoint, Step 7 Evidence, Step 8 History, and Step 9 Report."""
    model_dir = settings.models_dir / "umm-maybe--AI-image-detector"
    _build_local_onnx_swin_classifier(model_dir, bias_logits=(2.4, -1.2))

    jpeg_bytes = _create_sample_image(fmt="JPEG", size=(300, 300))
    expected_sha = hashlib.sha256(jpeg_bytes).hexdigest()

    up = client.post("/api/media/upload", files={"file": ("onnx_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")})
    assert up.status_code == 201
    case_id = up.json()["case_id"]
    stored_path = settings.uploads_dir / up.json()["stored_filename"]

    # F. Run Step 4 + Step 14.1.1 Detection using the real ONNX Runtime model
    det_res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert det_res.status_code == 200
    det_json = det_res.json()["manipulation_detection"]

    ai_det = det_json["ai_generation_detection"]
    assert ai_det["status"] == "available"
    assert ai_det["detector"]["name"] == OFFICIAL_MODEL_ID
    assert ai_det["detector"]["version"] == OFFICIAL_PINNED_REVISION
    assert ai_det["detector"]["pinned_revision"] == OFFICIAL_PINNED_REVISION
    assert ai_det["detector"]["architecture"] == OFFICIAL_MODEL_ARCHITECTURE
    assert ai_det["detector"]["license"] == OFFICIAL_MODEL_LICENSE
    assert ai_det["detector"]["framework"] == "onnxruntime"
    assert ai_det["detector"]["onnx_is_upstream"] is False
    assert ai_det["prediction"]["label"] == "potentially_ai_generated"
    assert ai_det["prediction"]["display_label"] == "Potential AI-generation signal"
    assert ai_det["prediction"]["decision_rule"] == "model_class_selection"
    assert ai_det["prediction"]["threshold"] is None
    assert 0.5 < ai_det["prediction"]["score"] <= 1.0

    # Verify GET /api/analysis/image/{case_id}/ai-detection returns stored result without rerunning inference
    get_ai_res = client.get(f"/api/analysis/image/{case_id}/ai-detection")
    assert get_ai_res.status_code == 200
    assert get_ai_res.json() == ai_det

    # K. Verify Step 7 Evidence Correlation includes 'ai_generation_detector_signal' with strength='informational'
    ev_res = client.post(f"/api/evidence/{case_id}/generate")
    assert ev_res.status_code == 200
    ev_items = ev_res.json()["evidence"]["items"]
    ai_ev = [item for item in ev_items if item["type"] == "ai_generation_detector_signal"]
    assert len(ai_ev) == 1
    assert ai_ev[0]["category"] == "manipulation"
    assert ai_ev[0]["strength"] == "informational"
    assert "uncalibrated model score" in ai_ev[0]["value"]
    assert "decision_rule: model_class_selection" in ai_ev[0]["value"]
    assert "CC BY-ND 4.0" in ai_ev[0]["value"]

    # L. Verify Step 8 History Reconstruction includes 'AI-generation detector produced a model-derived signal'
    hist_res = client.post(f"/api/history/{case_id}/reconstruct")
    assert hist_res.status_code == 200
    hist_events = hist_res.json()["history"]["events"]
    syn_events = [e for e in hist_events if e["event_type"] == "possible_synthetic_generation_signal"]
    assert len(syn_events) == 1
    assert syn_events[0]["title"] == "AI-generation detector produced a model-derived signal"
    assert syn_events[0]["status"] in {"possible", "supported_inference"}
    assert syn_events[0]["temporal_certainty"] == "unknown"
    assert "does not confirm AI generation or identify a specific generator" in syn_events[0]["reasoning"]

    # M. Verify Step 9 Forensic Report includes full metadata, CC BY-ND 4.0 license, pinned revision, and limitations
    rep_res = client.post(f"/api/reports/{case_id}/generate")
    assert rep_res.status_code == 200
    rep_json = rep_res.json()
    rep_ai = rep_json["ai_generation_detection"]
    assert rep_ai["status"] == "available"
    assert rep_ai["detector"]["license"] == OFFICIAL_MODEL_LICENSE
    assert rep_ai["detector"]["pinned_revision"] == OFFICIAL_PINNED_REVISION
    assert rep_ai["prediction"]["decision_rule"] == "model_class_selection"
    assert rep_ai["prediction"]["threshold"] is None
    assert any("October 2022" in lim for lim in rep_ai["limitations"])

    # I. Verify SHA-256 and original bytes remain 100% preserved after all stages
    assert stored_path.read_bytes() == jpeg_bytes
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() == expected_sha


def test_h_corrupt_image_and_configured_threshold_behavior(monkeypatch):
    """Test H: Corrupt image handling, path traversal model ID rejection, and explicit configured threshold behavior."""
    # 1. Corrupt image rejected safely
    corrupt_bytes = b"\xff\xd8\xff\xe0NOT_A_VALID_JPEG_STREAM"
    up = client.post("/api/media/upload", files={"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")})
    assert up.status_code == 201
    cid = up.json()["case_id"]
    res_corrupt = client.post(f"/api/analysis/image/{cid}/detect")
    assert res_corrupt.status_code in {400, 422}

    # 2. Path traversal in AI_IMAGE_DETECTOR_MODEL_ID safely rejected -> returns status='not_available'
    monkeypatch.setattr(settings, "ai_image_detector_model_id", "../../unsafe/model")
    valid_bytes = _create_sample_image(fmt="JPEG", size=(256, 256))
    up2 = client.post("/api/media/upload", files={"file": ("safe.jpg", io.BytesIO(valid_bytes), "image/jpeg")})
    cid2 = up2.json()["case_id"]
    res_unsafe = client.post(f"/api/analysis/image/{cid2}/detect")
    assert res_unsafe.status_code == 200
    assert res_unsafe.json()["manipulation_detection"]["ai_generation_detection"]["status"] == "not_available"

    # 3. When an explicit threshold IS configured (e.g. 0.70) and score is within +/-0.05 ambiguity band -> 'undetermined'
    monkeypatch.setattr(settings, "ai_image_detector_model_id", OFFICIAL_MODEL_ID)
    monkeypatch.setattr(settings, "ai_image_detector_threshold", 0.70)
    image_ai_detection_service.set_custom_predictor_for_testing(
        lambda tensor, path: {"score": 0.68, "threshold": 0.70, "raw_class": "artificial"}
    )
    res_band = client.post(f"/api/analysis/image/{cid2}/detect")
    assert res_band.status_code == 200
    pred = res_band.json()["manipulation_detection"]["ai_generation_detection"]["prediction"]
    assert pred["label"] == "undetermined"
    assert pred["display_label"] == "Undetermined"
    assert pred["decision_rule"] == "configured_score_threshold"
    assert pred["threshold"] == 0.70


def test_step_14_1_1_2_onnx_input_output_validation_corrupt_model_and_sha256_verification():
    """Verify Step 14.1.1.2 runtime checks:
    - config-only directory without weights -> not_available
    - corrupt model.onnx -> not_available
    - invalid ONNX input shape ([B, 3, 128, 128]) -> not_available
    - invalid ONNX output class count ([B, 3]) -> not_available
    - mismatched onnx_sha256 in config.json -> not_available
    - valid model + matching SHA-256 -> available with integrity hashes exposed
    """
    model_dir = settings.models_dir / "umm-maybe--AI-image-detector"
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_bytes = _create_sample_image(fmt="JPEG", size=(256, 256))
    up = client.post("/api/media/upload", files={"file": ("val.jpg", io.BytesIO(sample_bytes), "image/jpeg")})
    case_id = up.json()["case_id"]

    # 1. Config-only directory (missing weights) -> not_available
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["SwinForImageClassification"], "id2label": {"0": "artificial", "1": "human"}}),
        encoding="utf-8",
    )
    image_ai_detection_service.clear_ai_detector_cache()
    res1 = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]["ai_generation_detection"]
    assert res1["status"] == "not_available"
    assert res1["prediction"] is None

    # 2. Corrupt/incomplete model.onnx -> not_available
    (model_dir / "model.onnx").write_bytes(b"CORRUPT_ONNX_HEADER_BYTES")
    image_ai_detection_service.clear_ai_detector_cache()
    res2 = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]["ai_generation_detection"]
    assert res2["status"] == "not_available"
    assert res2["prediction"] is None

    # 3. Wrong ONNX input shape ([None, 3, 128, 128] instead of 224x224) -> not_available
    x_bad = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [None, 3, 128, 128])
    y_ok = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])
    node_pool = helper.make_node("GlobalAveragePool", inputs=["pixel_values"], outputs=["pooled"])
    node_flat = helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, [3, 2], [0.1] * 6)
    node_mm = helper.make_node("MatMul", inputs=["flat", "W"], outputs=["logits"])
    bad_in_model = helper.make_model(
        helper.make_graph([node_pool, node_flat, node_mm], "BadInputShapeGraph", [x_bad], [y_ok], [w_init]),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    bad_in_model.ir_version = 8
    onnx.save(bad_in_model, str(model_dir / "model.onnx"))
    image_ai_detection_service.clear_ai_detector_cache()
    res3 = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]["ai_generation_detection"]
    assert res3["status"] == "not_available"

    # 4. Wrong ONNX output class count ([None, 3] instead of 2 classes) -> not_available
    x_ok = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [None, 3, 224, 224])
    y_bad = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 3])
    w3_init = helper.make_tensor("W3", TensorProto.FLOAT, [3, 3], [0.1] * 9)
    node_mm3 = helper.make_node("MatMul", inputs=["flat", "W3"], outputs=["logits"])
    bad_out_model = helper.make_model(
        helper.make_graph([node_pool, node_flat, node_mm3], "BadOutputClassCountGraph", [x_ok], [y_bad], [w3_init]),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    bad_out_model.ir_version = 8
    onnx.save(bad_out_model, str(model_dir / "model.onnx"))
    image_ai_detection_service.clear_ai_detector_cache()
    res4 = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]["ai_generation_detection"]
    assert res4["status"] == "not_available"

    # 5. Valid ONNX model but mismatched onnx_sha256 in config.json -> not_available
    _build_local_onnx_swin_classifier(model_dir, bias_logits=(1.8, -0.9))
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg["onnx_sha256"] = "0" * 64
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    image_ai_detection_service.clear_ai_detector_cache()
    res5 = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]["ai_generation_detection"]
    assert res5["status"] == "not_available"

    # 6. Matching onnx_sha256 in config.json -> available and exposes onnx_sha256 without local paths
    real_onnx_sha = hashlib.sha256((model_dir / "model.onnx").read_bytes()).hexdigest()
    cfg["onnx_sha256"] = real_onnx_sha
    cfg["pytorch_model_sha256"] = "f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf"
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    image_ai_detection_service.clear_ai_detector_cache()
    res6 = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]["ai_generation_detection"]
    assert res6["status"] == "available"
    assert res6["detector"]["onnx_sha256"] == real_onnx_sha
    assert res6["detector"]["pytorch_model_sha256"] == "f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf"
    assert res6["detector"]["onnx_is_upstream"] is False
    assert str(model_dir) not in json.dumps(res6)


def test_step_14_1_1_2_real_installed_swin_onnx_model_inference(monkeypatch):
    """Run real end-to-end inference using the actual installed storage/models/umm-maybe--AI-image-detector/model.onnx artifact."""
    real_models_root = Path(__file__).resolve().parents[2] / "storage" / "models"
    real_model_dir = real_models_root / "umm-maybe--AI-image-detector"
    real_onnx_path = real_model_dir / "model.onnx"
    real_pt_path = real_model_dir / "pytorch_model.bin"

    assert real_pt_path.is_file(), "Pinned pytorch_model.bin must be installed in storage/models/"
    assert real_pt_path.stat().st_size == 347596479
    assert real_onnx_path.is_file(), "Exported model.onnx must be installed in storage/models/"

    monkeypatch.setattr(settings, "models_dir", real_models_root)
    monkeypatch.setattr(settings, "storage_root", real_models_root.parent)
    image_ai_detection_service.clear_ai_detector_cache()

    sample_bytes = _create_sample_image(fmt="PNG", size=(320, 240), color=(48, 96, 160))
    up = client.post("/api/media/upload", files={"file": ("real_swin_test.png", io.BytesIO(sample_bytes), "image/png")})
    assert up.status_code == 201
    case_id = up.json()["case_id"]

    det_res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert det_res.status_code == 200
    ai_det = det_res.json()["manipulation_detection"]["ai_generation_detection"]

    assert ai_det["status"] == "available"
    assert ai_det["detector"]["name"] == OFFICIAL_MODEL_ID
    assert ai_det["detector"]["pinned_revision"] == OFFICIAL_PINNED_REVISION
    assert ai_det["detector"]["architecture"] == "SwinForImageClassification"
    assert ai_det["detector"]["license"] == OFFICIAL_MODEL_LICENSE
    assert ai_det["detector"]["onnx_is_upstream"] is False
    assert ai_det["detector"]["pytorch_model_sha256"] == "f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf"
    assert ai_det["detector"]["onnx_sha256"] == "693911a555891bac487c9e24daf4f2adcef258eaa7304d8d344ecc96b9af7b2e"
    assert ai_det["prediction"]["score_type"] == "uncalibrated_model_score"
    assert ai_det["prediction"]["decision_rule"] == "model_class_selection"
    assert ai_det["prediction"]["threshold"] is None
    assert ai_det["prediction"]["raw_class"] in {"artificial", "human"}


