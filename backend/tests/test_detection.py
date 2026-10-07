import hashlib
import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirect storage directories to a temporary folder during test execution."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    uploads.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "storage_root", tmp_path)
    monkeypatch.setattr(settings, "uploads_dir", uploads)
    monkeypatch.setattr(settings, "results_dir", results)
    monkeypatch.setattr(settings, "reports_dir", reports)
    yield tmp_path


def _make_image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (400, 300),
    mode: str = "RGB",
    exif_bytes: bytes | None = None,
) -> bytes:
    img = Image.new(mode, size, color=(80, 140, 210))
    buf = io.BytesIO()
    kwargs = {}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_detect_valid_jpeg_with_software_exif_and_gps_redaction():
    """
    Covers:
    1. Successful detection of a valid uploaded JPEG
    5. Existing Step 3 analysis remains intact
    6. Detection result is persisted
    7. Original upload file remains unchanged
    8. SHA-256 remains unchanged
    9. AI-generation detector returns 'not_available'
    10. No fake confidence score is returned
    11. Detection endpoint does not expose GPS coordinates
    """
    exif = Image.Exif()
    exif[271] = "Nikon"
    exif[272] = "Z8"
    exif[305] = "Adobe Photoshop 25.2"
    exif[306] = "2026:09:28 14:10:00"
    exif[36867] = "2026:09:27 09:00:00"
    exif[0x8825] = {
        1: "N",
        2: (40.0, 42.0, 46.08),
        3: "W",
        4: (74.0, 0.0, 21.6),
    }

    jpeg_bytes = _make_image_bytes(fmt="JPEG", size=(640, 480), exif_bytes=exif.tobytes())
    expected_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

    # Step 2: Upload
    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("edited_sample.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]
    stored_path = settings.uploads_dir / upload_res.json()["stored_filename"]
    bytes_before = stored_path.read_bytes()

    # Step 3: Basic Analysis
    step3_res = client.post(f"/api/analysis/image/{case_id}")
    assert step3_res.status_code == 200
    step3_analysis_snapshot = step3_res.json()["analysis"]

    # Step 4: Manipulation Detection
    detect_res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert detect_res.status_code == 200
    data = detect_res.json()

    assert data["case_id"] == case_id
    assert data["status"] == "detected"
    assert data["sha256"] == expected_sha256

    # Verify Step 3 analysis remains completely intact
    assert data["analysis"] == step3_analysis_snapshot

    # Verify Step 4 manipulation_detection payload
    detection = data["manipulation_detection"]
    assert detection["status"] == "completed"
    assert detection["overall_assessment"] == "potential_manipulation"
    assert detection["ai_generation"]["status"] == "not_available"
    assert "No trained AI-generation detection model" in detection["ai_generation"]["reason"]

    # Verify software_metadata signal is present with weak strength
    signal_names = {sig["name"]: sig for sig in detection["signals"]}
    assert "software_metadata" in signal_names
    assert signal_names["software_metadata"]["value"] == "Adobe Photoshop 25.2"
    assert signal_names["software_metadata"]["strength"] == "weak"

    # Verify no fake numerical confidence score exists
    assert "confidence" not in detection
    assert "confidence_score" not in detection
    for sig in detection["signals"]:
        assert "confidence" not in sig
        assert sig["strength"] in {"informational", "weak", "moderate"}

    # Verify raw GPS coordinates (46.08 / 21.6) are never exposed anywhere in the payload
    serialized = json.dumps(data)
    assert "46.08" not in serialized
    assert "21.6" not in serialized

    # Verify original file on disk was not modified
    bytes_after = stored_path.read_bytes()
    assert bytes_after == bytes_before
    assert hashlib.sha256(bytes_after).hexdigest() == expected_sha256

    # Verify detection result is persisted in storage/results/<case_id>.json
    persisted = json.loads((settings.results_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "detected"
    assert persisted["manipulation_detection"]["overall_assessment"] == "potential_manipulation"
    assert persisted["analysis"] == step3_analysis_snapshot


def test_detect_valid_png_without_exif_returns_insufficient_evidence():
    """
    Covers:
    2. Successful detection of a valid uploaded PNG
    9. AI-generation detector returns 'not_available'
    """
    png_bytes = _make_image_bytes(fmt="PNG", size=(512, 512), mode="RGBA")
    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("clean_graphic.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]

    detect_res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert detect_res.status_code == 200
    data = detect_res.json()

    assert data["status"] == "detected"
    assert data["analysis"] is not None
    assert data["analysis"]["image"]["format"] == "PNG"

    detection = data["manipulation_detection"]
    assert detection["status"] == "completed"
    assert detection["overall_assessment"] == "insufficient_evidence"
    assert detection["ai_generation"]["status"] == "not_available"


def test_detect_camera_jpeg_without_software_returns_no_strong_indicators():
    """Verify an unmodified camera JPEG with hardware EXIF and no editing tags returns 'no_strong_indicators'."""
    exif = Image.Exif()
    exif[271] = "Sony"
    exif[272] = "ILCE-7RM5"
    exif[36867] = "2026:09:28 10:00:00"

    jpeg_bytes = _make_image_bytes(fmt="JPEG", size=(800, 600), exif_bytes=exif.tobytes())
    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("camera_raw_export.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    case_id = upload_res.json()["case_id"]

    detect_res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert detect_res.status_code == 200
    detection = detect_res.json()["manipulation_detection"]
    assert detection["overall_assessment"] == "no_strong_indicators"


def test_detect_nonexistent_case_returns_404():
    """Covers 3: Nonexistent case returns 404."""
    res = client.post("/api/analysis/image/MT-20260928-CAFEBABE/detect")
    assert res.status_code == 404


def test_detect_non_image_case_rejected():
    """Covers 4: Non-image case is rejected with HTTP 400."""
    case_id = "MT-20260928-AABBCCDD"
    non_image_case = {
        "case_id": case_id,
        "original_filename": "interview.wav",
        "stored_filename": f"{case_id}_interview.wav",
        "media_type": "audio",
        "extension": ".wav",
        "file_size": 2048,
        "sha256": "b" * 64,
        "status": "uploaded",
        "created_at": "2026-09-28T11:00:00+00:00",
    }
    (settings.results_dir / f"{case_id}.json").write_text(json.dumps(non_image_case), encoding="utf-8")

    res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res.status_code == 400
    assert "expected 'image'" in res.json()["detail"]
