import hashlib
import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import settings
from app.main import app
from app.services import media_service

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


def _make_png_bytes(
    size: tuple[int, int] = (320, 240),
    base_color: tuple[int, int, int] = (60, 100, 150),
    edit_box: tuple[int, int, int, int] | None = None,
    exif_bytes: bytes | None = None,
) -> bytes:
    img = Image.new("RGB", size, color=base_color)
    if edit_box is not None:
        draw = ImageDraw.Draw(img)
        draw.rectangle(list(edit_box), fill=(245, 30, 30))
    buf = io.BytesIO()
    kwargs = {}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    img.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def test_full_comparison_workflow_with_edited_region_and_step_preservation():
    """
    Covers:
    1. Reference image upload succeeds
    5. Reference SHA-256 correct
    6. Reference image stored separately
    8. Comparison succeeds for changed images
    10. Changed regions are within image bounds
    11. Difference image generated
    12. Difference mask generated
    13. Artifact dimensions match comparison dimensions
    14. Comparison result persisted
    15. Step 3 analysis preserved
    16. Step 4 manipulation detection preserved
    17. Step 5 localization preserved
    18. Original case image remains byte-for-byte unchanged
    19. Original case image SHA-256 unchanged
    20. Reference image SHA-256 unchanged
    21. GPS coordinates never exposed
    22. No confidence percentage returned
    """
    exif = Image.Exif()
    exif[271] = "Leica"
    exif[272] = "Q3"
    exif[305] = "Adobe Photoshop 25.4"
    exif[0x8825] = {
        1: "N",
        2: (48.0, 51.0, 29.52),
        3: "E",
        4: (2.0, 17.0, 40.2),
    }

    # Case image has a red edited box in [100:160, 80:130]
    case_bytes = _make_png_bytes(
        size=(320, 240),
        edit_box=(100, 80, 159, 129),
        exif_bytes=exif.tobytes(),
    )
    # Reference image is the clean unedited version
    ref_bytes = _make_png_bytes(size=(320, 240), edit_box=None, exif_bytes=exif.tobytes())

    case_sha256 = hashlib.sha256(case_bytes).hexdigest()
    ref_sha256 = hashlib.sha256(ref_bytes).hexdigest()

    # Step 2: Upload case image
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("edited_case.png", io.BytesIO(case_bytes), "image/png")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]
    case_upload_path = settings.uploads_dir / up_res.json()["stored_filename"]
    case_bytes_before = case_upload_path.read_bytes()

    # Step 3, Step 4, Step 5
    step3_snapshot = client.post(f"/api/analysis/image/{case_id}").json()["analysis"]
    step4_snapshot = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]
    step5_snapshot = client.post(f"/api/analysis/image/{case_id}/localize").json()["localization"]

    # Step 6a: Upload Reference Image
    ref_res = client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("original_reference.png", io.BytesIO(ref_bytes), "image/png")},
    )
    assert ref_res.status_code == 201
    ref_data = ref_res.json()["reference_image"]
    assert ref_data["original_filename"] == "original_reference.png"
    assert ref_data["stored_filename"] == f"{case_id}_reference_original_reference.png"
    assert ref_data["sha256"] == ref_sha256
    assert ref_data["width"] == 320
    assert ref_data["height"] == 240

    ref_upload_path = settings.uploads_dir / ref_data["stored_filename"]
    assert ref_upload_path.is_file()
    assert ref_upload_path.resolve() != case_upload_path.resolve()
    ref_bytes_before = ref_upload_path.read_bytes()

    # Step 6b: Run Comparison
    comp_res = client.post(f"/api/comparison/image/{case_id}")
    assert comp_res.status_code == 200
    body = comp_res.json()

    assert body["status"] == "compared"
    # Verify Steps 3, 4, and 5 are preserved
    assert body["analysis"] == step3_snapshot
    assert body["manipulation_detection"] == step4_snapshot
    assert body["localization"] == step5_snapshot

    comp = body["comparison"]
    assert comp["status"] == "completed"
    assert comp["comparison_mode"] == "direct"
    assert comp["dimensions"]["same"] is True
    assert comp["assessment"] == "localized_visual_differences_detected"
    assert comp["pixel_difference"]["changed_pixel_percentage"] > 0.0
    assert len(comp["changed_regions"]) == 1

    region = comp["changed_regions"][0]
    assert region["id"] == "change-1"
    assert 0 <= region["x"] < 320
    assert 0 <= region["y"] < 240
    assert region["x"] + region["width"] <= 320
    assert region["y"] + region["height"] <= 240
    assert region["changed_pixel_count"] > 0
    assert 0.0 < region["changed_pixel_percentage"] <= 100.0
    assert "confidence" not in region

    # Verify artifacts exist and match dimensions (320x240)
    diff_path = settings.results_dir / case_id / "comparison" / "difference.png"
    mask_path = settings.results_dir / case_id / "comparison" / "difference_mask.png"
    assert diff_path.is_file()
    assert mask_path.is_file()

    with Image.open(diff_path) as d_img:
        assert d_img.size == (320, 240)
    with Image.open(mask_path) as m_img:
        assert m_img.size == (320, 240)

    # Verify HTTP serving of artifacts
    art_res = client.get(f"/api/comparison/image/{case_id}/difference.png")
    assert art_res.status_code == 200
    assert art_res.headers["content-type"] == "image/png"

    # Verify source files and SHA-256 hashes are untouched
    assert case_upload_path.read_bytes() == case_bytes_before
    assert hashlib.sha256(case_upload_path.read_bytes()).hexdigest() == case_sha256
    assert ref_upload_path.read_bytes() == ref_bytes_before
    assert hashlib.sha256(ref_upload_path.read_bytes()).hexdigest() == ref_sha256

    # Verify GPS coordinates (29.52 / 40.2) are never exposed
    serialized = json.dumps(body)
    assert "29.52" not in serialized
    assert "40.2" not in serialized
    assert "confidence" not in comp

    # Verify persistence in storage/results/<case_id>.json
    persisted = json.loads((settings.results_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "compared"
    assert persisted["comparison"]["assessment"] == "localized_visual_differences_detected"


def test_comparison_identical_images_returns_no_significant_difference():
    """Covers 7: Comparison succeeds for identical images."""
    img_bytes = _make_png_bytes(size=(200, 200), base_color=(90, 120, 170))
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("img_a.png", io.BytesIO(img_bytes), "image/png")},
    )
    case_id = up_res.json()["case_id"]

    client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("img_b.png", io.BytesIO(img_bytes), "image/png")},
    )

    comp_res = client.post(f"/api/comparison/image/{case_id}")
    assert comp_res.status_code == 200
    comp = comp_res.json()["comparison"]
    assert comp["assessment"] == "no_significant_difference"
    assert comp["pixel_difference"]["mean_absolute_difference"] == 0.0
    assert comp["pixel_difference"]["max_pixel_difference"] == 0
    assert comp["pixel_difference"]["changed_pixel_percentage"] == 0.0
    assert comp["changed_regions"] == []


def test_comparison_different_dimensions_handled_correctly():
    """Covers 9: Different dimensions handled correctly via normalized_resize."""
    case_bytes = _make_png_bytes(size=(300, 200))
    ref_bytes = _make_png_bytes(size=(600, 400))

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("small_case.png", io.BytesIO(case_bytes), "image/png")},
    )
    case_id = up_res.json()["case_id"]

    client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("large_ref.png", io.BytesIO(ref_bytes), "image/png")},
    )

    comp_res = client.post(f"/api/comparison/image/{case_id}")
    assert comp_res.status_code == 200
    comp = comp_res.json()["comparison"]
    assert comp["dimensions"]["same"] is False
    assert comp["comparison_mode"] == "normalized_resize"
    assert comp["assessment"] == "different_dimensions"


def test_reference_validation_errors_and_artifact_path_traversal(monkeypatch):
    """
    Covers:
    2. Unsupported reference extension rejected
    3. Oversized reference image rejected
    4. Invalid reference image rejected
    23. Path traversal blocked on artifact endpoint
    """
    valid_bytes = _make_png_bytes(size=(128, 128))
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("base.png", io.BytesIO(valid_bytes), "image/png")},
    )
    case_id = up_res.json()["case_id"]

    # 2. Unsupported extension
    bad_ext = client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("bad_ref.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert bad_ext.status_code == 400

    # 4. Invalid image content with valid extension
    corrupt_img = client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("corrupt.png", io.BytesIO(b"not-a-valid-png"), "image/png")},
    )
    assert corrupt_img.status_code == 400

    # 3. Oversized reference image (patch MAX_UPLOAD_SIZE_BYTES to 100 bytes for fast testing)
    from app.services.comparison import image_comparison_service
    monkeypatch.setattr(image_comparison_service, "MAX_UPLOAD_SIZE_BYTES", 100)
    oversized = client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("big.png", io.BytesIO(valid_bytes), "image/png")},
    )
    assert oversized.status_code == 413

    # 23. Path traversal / arbitrary filename blocked on artifact endpoint
    traversal_1 = client.get(f"/api/comparison/image/{case_id}/../../etc/passwd")
    assert traversal_1.status_code in {400, 404}

    traversal_2 = client.get(f"/api/comparison/image/{case_id}/secret.json")
    assert traversal_2.status_code == 400
