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


def _create_image_bytes(
    fmt: str = "PNG",
    size: tuple[int, int] = (320, 240),
    mode: str = "RGB",
    exif_bytes: bytes | None = None,
) -> bytes:
    """Create a real valid image buffer using Pillow."""
    img = Image.new(mode, size, color=(40, 120, 200) if mode != "L" else 128)
    buf = io.BytesIO()
    save_kwargs = {}
    if exif_bytes is not None:
        save_kwargs["exif"] = exif_bytes
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def test_analyze_valid_uploaded_image_and_persist_result():
    """
    Tests 1, 4, 5, 7, 9, 10, 11:
    - Analyze valid uploaded PNG image without EXIF
    - Verify dimensions (320x240) and format (PNG)
    - Verify original file bytes and SHA-256 remain completely unchanged
    - Verify analysis result is persisted in storage/results/<case_id>.json with status='analyzed'
    """
    raw_bytes = _create_image_bytes(fmt="PNG", size=(320, 240), mode="RGBA")
    original_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("sample_rgba.png", io.BytesIO(raw_bytes), "image/png")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]

    upload_file_path = settings.uploads_dir / upload_res.json()["stored_filename"]
    bytes_before_analysis = upload_file_path.read_bytes()

    analyze_res = client.post(f"/api/analysis/image/{case_id}")
    assert analyze_res.status_code == 200
    data = analyze_res.json()

    assert data["case_id"] == case_id
    assert data["status"] == "analyzed"
    assert data["sha256"] == original_sha256

    # Check extracted image technical properties
    assert data["analysis"]["image"]["format"] == "PNG"
    assert data["analysis"]["image"]["width"] == 320
    assert data["analysis"]["image"]["height"] == 240
    assert data["analysis"]["image"]["mode"] == "RGBA"
    assert data["analysis"]["image"]["channels"] == 4

    # Check metadata when no EXIF is present
    assert data["analysis"]["metadata"]["exif_present"] is False
    assert data["analysis"]["metadata"]["gps_present"] is False
    assert data["analysis"]["metadata"]["gps_info"] is None

    # Check original uploaded file on disk was not modified byte-for-byte
    bytes_after_analysis = upload_file_path.read_bytes()
    assert bytes_after_analysis == bytes_before_analysis
    assert hashlib.sha256(bytes_after_analysis).hexdigest() == original_sha256

    # Check persistence inside storage/results/<case_id>.json
    persisted_json = json.loads((settings.results_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    assert persisted_json["status"] == "analyzed"
    assert persisted_json["analysis"]["image"]["width"] == 320
    assert persisted_json["analysis"]["image"]["height"] == 240


def test_analyze_jpeg_with_exif_and_gps_privacy():
    """
    Tests 5, 6, 8:
    - Verify JPEG format & compression inspection
    - Verify EXIF fields (Make, Model, Software, DateTimeOriginal) are extracted
    - Verify GPS presence is detected ('GPS metadata present') while raw GPS coordinates are never exposed
    """
    exif = Image.Exif()
    exif[271] = "Canon"                          # Make
    exif[272] = "EOS R5"                         # Model
    exif[305] = "Adobe Photoshop 25.0"           # Software
    exif[306] = "2026:09:28 12:30:00"            # DateTime
    exif[36867] = "2026:09:27 08:15:42"          # DateTimeOriginal
    # GPS IFD tag (0x8825 = 34853) with secret coordinates (37.7749, -122.4194)
    exif[0x8825] = {
        1: "N",
        2: (37.0, 46.0, 29.64),
        3: "W",
        4: (122.0, 25.0, 9.84),
    }

    jpeg_bytes = _create_image_bytes(
        fmt="JPEG",
        size=(640, 480),
        mode="RGB",
        exif_bytes=exif.tobytes(),
    )

    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("camera_shot.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]

    analyze_res = client.post(f"/api/analysis/image/{case_id}")
    assert analyze_res.status_code == 200
    data = analyze_res.json()

    assert data["analysis"]["image"]["format"] == "JPEG"
    assert data["analysis"]["image"]["width"] == 640
    assert data["analysis"]["image"]["height"] == 480
    assert data["analysis"]["image"]["jpeg_encoding"] in {"baseline", "progressive"}

    meta = data["analysis"]["metadata"]
    assert meta["exif_present"] is True
    assert meta["gps_present"] is True
    assert meta["gps_info"] == "GPS metadata present"
    assert meta["fields"]["make"] == "Canon"
    assert meta["fields"]["model"] == "EOS R5"
    assert meta["fields"]["software"] == "Adobe Photoshop 25.0"
    assert meta["fields"]["datetime_original"] == "2026:09:27 08:15:42"

    # Privacy check: ensure secret coordinates (37.7749 / 29.64 / 122.0 / 9.84) do not appear anywhere in response
    serialized = json.dumps(data)
    assert "29.64" not in serialized
    assert "122.0" not in serialized
    assert "9.84" not in serialized

    # Verify neutral software finding exists
    software_findings = [
        f for f in data["analysis"]["findings"]
        if f["finding"] == "Editing software metadata present"
    ]
    assert len(software_findings) == 1
    assert software_findings[0]["value"] == "Adobe Photoshop 25.0"
    assert software_findings[0]["severity"] == "informational"


def test_analyze_nonexistent_case_returns_404():
    """Test 2: Verify analyzing a case ID that does not exist returns HTTP 404."""
    response = client.post("/api/analysis/image/MT-20260928-DEADBEEF")
    assert response.status_code == 404


def test_analyze_non_image_case_rejected():
    """Test 3: Verify a case whose media_type is not 'image' is rejected with HTTP 400."""
    case_id = "MT-20260928-11223344"
    non_image_case = {
        "case_id": case_id,
        "original_filename": "clip.mp4",
        "stored_filename": f"{case_id}_clip.mp4",
        "media_type": "video",
        "extension": ".mp4",
        "file_size": 1024,
        "sha256": "a" * 64,
        "status": "uploaded",
        "created_at": "2026-09-28T10:00:00+00:00",
    }
    (settings.results_dir / f"{case_id}.json").write_text(json.dumps(non_image_case), encoding="utf-8")

    response = client.post(f"/api/analysis/image/{case_id}")
    assert response.status_code == 400
    assert "expected 'image'" in response.json()["detail"]
