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


def _create_image_with_patch(
    size: tuple[int, int] = (320, 240),
    patch_box: tuple[int, int, int, int] | None = None,
    exif_bytes: bytes | None = None,
) -> bytes:
    img = Image.new("RGB", size, color=(55, 95, 145))
    if patch_box is not None:
        x0, y0, x1, y1 = patch_box
        draw = ImageDraw.Draw(img)
        for y in range(y0, y1, 2):
            for x in range(x0, x1, 2):
                color = (250, 250, 250) if ((x // 2 + y // 2) % 2 == 0) else (10, 10, 10)
                draw.rectangle([x, y, x + 1, y + 1], fill=color)

    buf = io.BytesIO()
    kwargs = {"quality": 95}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def test_evidence_generation_correlates_steps_and_detects_spatial_overlap():
    """
    Covers:
    1. Evidence generation succeeds (POST /api/evidence/{case_id}/generate and GET /api/evidence/{case_id})
    3. Evidence report persists
    4. Step 3 evidence is included
    5. Step 4 evidence is included
    6. Step 5 localization evidence is included
    7. Step 6 comparison evidence is included
    8. Integrity evidence is included
    9. GPS coordinates are never exposed
    10. AI detector 'not_available' does not become a fake evidence item
    11. Cross-analysis overlap is detected when regions overlap
    13. No numerical manipulation probability is returned
    14. Previous analysis fields remain unchanged
    """
    exif = Image.Exif()
    exif[271] = "Sony"
    exif[272] = "A7IV"
    exif[305] = "Adobe Photoshop 25.5"
    exif[36867] = "2026:09:27 11:20:30"
    exif[0x8825] = {
        1: "N",
        2: (35.0, 41.0, 77.22),
        3: "E",
        4: (139.0, 41.0, 88.31),
    }

    # Case image has high-frequency patch in [80:160, 64:144]
    case_bytes = _create_image_with_patch(
        size=(320, 240),
        patch_box=(80, 64, 160, 144),
        exif_bytes=exif.tobytes(),
    )
    # Reference image is clean without the patch -> comparison changed_region will also be [80:160, 64:144]
    ref_bytes = _create_image_with_patch(
        size=(320, 240),
        patch_box=None,
        exif_bytes=exif.tobytes(),
    )

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("case_img.jpg", io.BytesIO(case_bytes), "image/jpeg")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]

    # Run Steps 3, 4, 5, 6
    s3_snapshot = client.post(f"/api/analysis/image/{case_id}").json()["analysis"]
    s4_snapshot = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]
    s5_snapshot = client.post(f"/api/analysis/image/{case_id}/localize").json()["localization"]

    client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("ref_img.jpg", io.BytesIO(ref_bytes), "image/jpeg")},
    )
    s6_snapshot = client.post(f"/api/comparison/image/{case_id}").json()["comparison"]

    # Step 7: Generate Evidence Report
    gen_res = client.post(f"/api/evidence/{case_id}/generate")
    assert gen_res.status_code == 200
    full_case = gen_res.json()

    # Verify previous analysis fields remain completely unchanged
    assert full_case["analysis"] == s3_snapshot
    assert full_case["manipulation_detection"] == s4_snapshot
    assert full_case["localization"] == s5_snapshot
    assert full_case["comparison"] == s6_snapshot

    evidence = full_case["evidence"]
    assert evidence["case_id"] == case_id
    assert evidence["assessment"] == "multiple_forensic_signals"

    # Verify summary counts and stages
    summary = evidence["summary"]
    assert set(summary["analysis_stages_available"]) == {
        "basic_forensics",
        "manipulation_detection",
        "localization",
        "comparison",
    }
    assert summary["total_evidence_items"] == len(evidence["items"])
    assert (
        summary["informational"] + summary["weak"] + summary["moderate"]
        == summary["total_evidence_items"]
    )

    # Verify sources from all stages are included
    sources = {item["source"] for item in evidence["items"]}
    assert "step2_upload" in sources
    assert "integrity_check" in sources
    assert "step3_basic_forensics" in sources
    assert "step4_manipulation_detection" in sources
    assert "step5_localization" in sources
    assert "step6_comparison" in sources
    assert "cross_analysis" in sources

    # Verify cross-analysis overlap item exists and has overlap_percentage > 0
    overlaps = [item for item in evidence["items"] if item["type"] == "cross_analysis_overlap"]
    assert len(overlaps) >= 1
    assert overlaps[0]["value"]["overlap_percentage"] > 0.0
    assert overlaps[0]["strength"] == "moderate"

    # Verify AI detector 'not_available' did not create any false AI-generation evidence item
    for item in evidence["items"]:
        assert "ai_generation" not in item["type"].lower()
        assert item["strength"] in {"informational", "weak", "moderate"}
        assert "confidence" not in item
        assert "probability" not in item

    # Verify GPS coordinates (77.22 / 139.0 / 88.31) are never exposed anywhere
    serialized = json.dumps(full_case)
    assert "77.22" not in serialized
    assert "139.0" not in serialized
    assert "88.31" not in serialized

    # Verify GET /api/evidence/{case_id} returns the stored EvidenceReport
    get_res = client.get(f"/api/evidence/{case_id}")
    assert get_res.status_code == 200
    assert get_res.json() == evidence

    # Verify persistence on disk
    persisted = json.loads((settings.results_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    assert persisted["evidence"] == evidence


def test_no_cross_analysis_overlap_when_regions_are_disjoint():
    """Covers 12: No cross-analysis overlap when localization and comparison regions do not overlap."""
    # Case image has high-frequency patch in top-left [16:80, 16:80]
    case_bytes = _create_image_with_patch(size=(320, 240), patch_box=(16, 16, 80, 80))
    # Reference image also has that same top-left patch (so comparison sees 0 diff in top-left),
    # PLUS a solid color change in bottom-right [220:290, 160:220] so comparison changed_region is in bottom-right!
    ref_img = Image.open(io.BytesIO(case_bytes)).convert("RGB")
    draw = ImageDraw.Draw(ref_img)
    draw.rectangle([220, 160, 290, 220], fill=(240, 20, 20))
    ref_buf = io.BytesIO()
    ref_img.save(ref_buf, format="PNG")
    case_png_buf = io.BytesIO()
    Image.open(io.BytesIO(case_bytes)).save(case_png_buf, format="PNG")

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("disjoint_case.png", io.BytesIO(case_png_buf.getvalue()), "image/png")},
    )
    case_id = up_res.json()["case_id"]

    client.post(f"/api/analysis/image/{case_id}/localize")
    client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("disjoint_ref.png", io.BytesIO(ref_buf.getvalue()), "image/png")},
    )
    client.post(f"/api/comparison/image/{case_id}")

    gen_res = client.post(f"/api/evidence/{case_id}/generate")
    assert gen_res.status_code == 200
    evidence = gen_res.json()["evidence"]

    overlaps = [item for item in evidence["items"] if item["type"] == "cross_analysis_overlap"]
    assert overlaps == []


def test_evidence_nonexistent_case_returns_404():
    """Covers 2: Nonexistent case returns 404 on both POST and GET."""
    post_res = client.post("/api/evidence/MT-20260928-12345678/generate")
    assert post_res.status_code == 404

    get_res = client.get("/api/evidence/MT-20260928-12345678")
    assert get_res.status_code == 404
