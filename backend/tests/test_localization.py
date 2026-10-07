import hashlib
import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

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


def _create_spliced_anomaly_image(fmt: str = "JPEG", size: tuple[int, int] = (320, 240), exif_bytes: bytes | None = None) -> bytes:
    """
    Create an image with a smooth background and a distinct high-frequency checkerboard
    patch inside [80:160, 64:144] so the classical localization baseline detects a real
    spatial inconsistency.
    """
    img = Image.new("RGB", size, color=(50, 90, 140))
    draw = ImageDraw.Draw(img)
    for y in range(64, 144, 2):
        for x in range(80, 160, 2):
            color = (250, 250, 250) if ((x // 2 + y // 2) % 2 == 0) else (5, 5, 5)
            draw.rectangle([x, y, x + 1, y + 1], fill=color)

    buf = io.BytesIO()
    kwargs = {}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    if fmt == "JPEG":
        kwargs["quality"] = 95
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_localize_jpeg_with_spliced_patch_preserves_step3_step4_and_files():
    """
    Covers:
    1. Successful localization of JPEG
    5. Step 3 analysis remains intact
    6. Step 4 manipulation_detection remains intact
    7. Localization result is persisted
    8. Original uploaded bytes remain unchanged
    9. Original SHA-256 remains unchanged
    10. Heatmap file is created
    11. Mask file is created
    12. Heatmap dimensions match original image
    13. Region coordinates are within image bounds
    14. No GPS coordinates are exposed
    15. No confidence percentage is returned
    """
    exif = Image.Exif()
    exif[271] = "Fujifilm"
    exif[272] = "X-T5"
    exif[0x8825] = {
        1: "N",
        2: (51.0, 30.0, 26.4),
        3: "W",
        4: (0.0, 7.0, 39.9),
    }

    jpeg_bytes = _create_spliced_anomaly_image(fmt="JPEG", size=(320, 240), exif_bytes=exif.tobytes())
    expected_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

    # Step 2: Upload
    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("spliced_scene.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]
    upload_path = settings.uploads_dir / upload_res.json()["stored_filename"]
    bytes_before = upload_path.read_bytes()

    # Step 3: Basic Analysis
    step3_res = client.post(f"/api/analysis/image/{case_id}")
    assert step3_res.status_code == 200
    step3_snapshot = step3_res.json()["analysis"]

    # Step 4: Manipulation Detection
    step4_res = client.post(f"/api/analysis/image/{case_id}/detect")
    assert step4_res.status_code == 200
    step4_snapshot = step4_res.json()["manipulation_detection"]

    # Step 5: Spatial Localization
    loc_res = client.post(f"/api/analysis/image/{case_id}/localize")
    assert loc_res.status_code == 200
    data = loc_res.json()

    assert data["case_id"] == case_id
    assert data["status"] == "localized"
    assert data["sha256"] == expected_sha256

    # Verify Step 3 and Step 4 payloads are completely preserved
    assert data["analysis"] == step3_snapshot
    assert data["manipulation_detection"] == step4_snapshot

    loc = data["localization"]
    assert loc["status"] == "completed"
    assert loc["method"] == "classical_localization_baseline"
    assert loc["localized_anomaly"] is True
    assert loc["image_dimensions"] == {"width": 320, "height": 240}
    assert len(loc["regions"]) >= 1

    # Verify region bounds and anomaly score properties (no confidence field)
    for region in loc["regions"]:
        assert 0 <= region["x"] < 320
        assert 0 <= region["y"] < 240
        assert 1 <= region["width"] <= 320
        assert 1 <= region["height"] <= 240
        assert region["x"] + region["width"] <= 320
        assert region["y"] + region["height"] <= 240
        assert region["area"] == region["width"] * region["height"]
        assert 0.0 <= region["mean_anomaly_score"] <= 1.0
        assert 0.0 <= region["max_anomaly_score"] <= 1.0
        assert "confidence" not in region
        assert "probability" not in region

    # Verify heatmap and mask files exist on disk and match original dimensions (320x240)
    heatmap_path = settings.results_dir / case_id / "localization" / "heatmap.png"
    mask_path = settings.results_dir / case_id / "localization" / "mask.png"
    assert heatmap_path.is_file()
    assert mask_path.is_file()

    with Image.open(heatmap_path) as hm_img:
        assert hm_img.size == (320, 240)
    with Image.open(mask_path) as mk_img:
        assert mk_img.size == (320, 240)

    # Verify visualization HTTP GET endpoint serves heatmap.png
    vis_http = client.get(f"/api/analysis/image/{case_id}/localization/heatmap.png")
    assert vis_http.status_code == 200
    assert vis_http.headers["content-type"] == "image/png"

    # Verify original file on disk was not modified
    bytes_after = upload_path.read_bytes()
    assert bytes_after == bytes_before
    assert hashlib.sha256(bytes_after).hexdigest() == expected_sha256

    # Verify no GPS coordinates are exposed anywhere in the payload
    serialized = json.dumps(data)
    assert "26.4" not in serialized
    assert "39.9" not in serialized

    # Verify persisted JSON in storage/results/<case_id>.json
    persisted = json.loads((settings.results_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "localized"
    assert persisted["localization"]["localized_anomaly"] is True


def test_localize_uniform_png_returns_no_localized_anomaly():
    """
    Covers:
    2. Successful localization of PNG
    16. Empty/no-anomaly result is handled correctly (uniform image does not produce false regions)
    """
    uniform_img = Image.new("RGB", (256, 256), color=(110, 130, 150))
    buf = io.BytesIO()
    uniform_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("uniform_flat.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert upload_res.status_code == 201
    case_id = upload_res.json()["case_id"]

    loc_res = client.post(f"/api/analysis/image/{case_id}/localize")
    assert loc_res.status_code == 200
    loc = loc_res.json()["localization"]

    assert loc["status"] == "completed"
    assert loc["localized_anomaly"] is False
    assert loc["regions"] == []
    assert (settings.results_dir / case_id / "localization" / "heatmap.png").is_file()
    assert (settings.results_dir / case_id / "localization" / "mask.png").is_file()


def test_localize_small_image_returns_insufficient_data():
    """Verify images smaller than 64x64 return status='insufficient_data' without crashing."""
    tiny_img = Image.new("RGB", (32, 32), color=(90, 90, 90))
    buf = io.BytesIO()
    tiny_img.save(buf, format="PNG")

    upload_res = client.post(
        "/api/media/upload",
        files={"file": ("tiny_icon.png", io.BytesIO(buf.getvalue()), "image/png")},
    )
    case_id = upload_res.json()["case_id"]

    loc_res = client.post(f"/api/analysis/image/{case_id}/localize")
    assert loc_res.status_code == 200
    loc = loc_res.json()["localization"]
    assert loc["status"] == "insufficient_data"
    assert loc["localized_anomaly"] is False
    assert loc["regions"] == []
    assert "insufficient" in loc["message"].lower()


def test_localize_nonexistent_case_returns_404():
    """Covers 3: Nonexistent case returns 404."""
    res = client.post("/api/analysis/image/MT-20260928-00000000/localize")
    assert res.status_code == 404


def test_localize_non_image_case_rejected():
    """Covers 4: Non-image case is rejected with HTTP 400."""
    case_id = "MT-20260928-99887766"
    non_image_case = {
        "case_id": case_id,
        "original_filename": "report.pdf",
        "stored_filename": f"{case_id}_report.pdf",
        "media_type": "document",
        "extension": ".pdf",
        "file_size": 4096,
        "sha256": "c" * 64,
        "status": "uploaded",
        "created_at": "2026-09-28T12:00:00+00:00",
    }
    (settings.results_dir / f"{case_id}.json").write_text(json.dumps(non_image_case), encoding="utf-8")

    res = client.post(f"/api/analysis/image/{case_id}/localize")
    assert res.status_code == 400
    assert "expected 'image'" in res.json()["detail"]
