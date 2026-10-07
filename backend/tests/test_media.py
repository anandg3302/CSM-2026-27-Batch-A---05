import hashlib
import io
import json
import pytest
from fastapi.testclient import TestClient

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


def test_successful_image_upload_and_sha256():
    """Verify valid image upload creates case metadata and computes accurate SHA-256."""
    sample_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR-sample-forensic-image-bytes"
    expected_sha256 = hashlib.sha256(sample_bytes).hexdigest()

    response = client.post(
        "/api/media/upload",
        files={"file": ("evidence_photo.png", io.BytesIO(sample_bytes), "image/png")},
    )

    assert response.status_code == 201
    data = response.json()

    assert data["case_id"].startswith("MT-")
    assert data["original_filename"] == "evidence_photo.png"
    assert data["stored_filename"] == f"{data['case_id']}_evidence_photo.png"
    assert data["media_type"] == "image"
    assert data["extension"] == ".png"
    assert data["file_size"] == len(sample_bytes)
    assert data["sha256"] == expected_sha256
    assert data["status"] == "uploaded"

    # Verify physical file and metadata JSON exist on disk
    saved_upload = settings.uploads_dir / data["stored_filename"]
    saved_metadata = settings.results_dir / f"{data['case_id']}.json"

    assert saved_upload.is_file()
    assert saved_upload.read_bytes() == sample_bytes
    assert saved_metadata.is_file()

    stored_json = json.loads(saved_metadata.read_text(encoding="utf-8"))
    assert stored_json["case_id"] == data["case_id"]
    assert stored_json["sha256"] == expected_sha256


def test_unsupported_extension_rejected():
    """Verify unsupported file extensions are rejected with HTTP 400."""
    response = client.post(
        "/api/media/upload",
        files={"file": ("malware.exe", io.BytesIO(b"binary-content"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_missing_file_rejected():
    """Verify POST /api/media/upload without a file returns an HTTP 400 error."""
    response = client.post("/api/media/upload")
    assert response.status_code == 400
    assert "No media file" in response.json()["detail"]


def test_path_traversal_filename_sanitized():
    """Verify path traversal sequences in filenames are stripped before saving."""
    sample_bytes = b"fake-jpeg-bytes"
    response = client.post(
        "/api/media/upload",
        files={"file": ("../../etc/passwd/../suspect_img.JPG", io.BytesIO(sample_bytes), "image/jpeg")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["original_filename"] == "suspect_img.jpg"
    assert "/" not in data["stored_filename"]
    assert "\\" not in data["stored_filename"]
    assert (settings.uploads_dir / data["stored_filename"]).is_file()


def test_get_media_case_by_id():
    """Verify GET /api/media/{case_id} returns stored metadata and 404 for unknown cases."""
    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("camera_frame.webp", io.BytesIO(b"webp-image-bytes"), "image/webp")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]

    get_res = client.get(f"/api/media/{case_id}")
    assert get_res.status_code == 200
    assert get_res.json()["case_id"] == case_id
    assert get_res.json()["original_filename"] == "camera_frame.webp"

    not_found_res = client.get("/api/media/MT-20260928-FFFFFFFF")
    assert not_found_res.status_code == 404


def test_list_media_cases():
    """Verify GET /api/media returns a list of all uploaded cases."""
    client.post(
        "/api/media/upload",
        files={"file": ("first.jpg", io.BytesIO(b"first-image-bytes"), "image/jpeg")},
    )
    client.post(
        "/api/media/upload",
        files={"file": ("second.jpeg", io.BytesIO(b"second-image-bytes"), "image/jpeg")},
    )

    list_res = client.get("/api/media")
    assert list_res.status_code == 200
    cases = list_res.json()
    assert isinstance(cases, list)
    assert len(cases) == 2
    filenames = {item["original_filename"] for item in cases}
    assert filenames == {"first.jpg", "second.jpeg"}
