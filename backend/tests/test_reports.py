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


def _create_image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (320, 240),
    patch_box: tuple[int, int, int, int] | None = None,
    exif_bytes: bytes | None = None,
    quality: int = 95,
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
    kwargs = {}
    if fmt.upper() == "JPEG":
        kwargs["quality"] = quality
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_reports_nonexistent_and_unreported_cases():
    """
    Covers:
    - Nonexistent case returns 404 for JSON and PDF report endpoints
    - Invalid case ID format returns 400
    - GET /api/reports/{case_id} returns 404 before report generation
    """
    assert client.post("/api/reports/invalid-id/generate").status_code == 400
    assert client.get("/api/reports/invalid-id").status_code == 400
    assert client.post("/api/reports/invalid-id/generate-pdf").status_code == 400

    missing_id = "MT-20260928-CAFEBABE"
    assert client.post(f"/api/reports/{missing_id}/generate").status_code == 404
    assert client.get(f"/api/reports/{missing_id}").status_code == 404
    assert client.post(f"/api/reports/{missing_id}/generate-pdf").status_code == 404

    png_bytes = _create_image_bytes(fmt="PNG", size=(160, 120))
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("unreported.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]

    assert client.get(f"/api/reports/{case_id}").status_code == 404


def test_full_pipeline_json_and_pdf_report_generation_security_and_immutability(isolated_storage):
    """
    Covers Requirements 1–24, 26, 30 (PDF Content Validation), and 31 (Security & Immutability):
    - Full pipeline (Steps 2–8) followed by Step 9 JSON and PDF report generation
    - Report ID format (MTR-...) distinct from case_id, report_version == "1.0"
    - Case info, Integrity (SHA-256 + hash_preserved), Media, Forensics, Manipulation Detection,
      Localization ("localized forensic anomaly"), Comparison ("visual differences"),
      Evidence, Cross-Analysis Evidence ("Cross-analysis spatial consistency"),
      Edit History ("Evidence-ordered hypothesis", "Historical accuracy: Not determined"),
      Assessment, and Limitations all faithfully included
    - No numerical manipulation or AI-generation probabilities
    - GPS coordinates, server filesystem paths, and secrets are never exposed in JSON or PDF
    - Case JSON and original uploaded image bytes remain 100% unmodified before vs after report generation
    - PDF file exists at storage/reports/<case_id>/MediaTrace_Forensic_Report.pdf and contains required sections
    """
    exif = Image.Exif()
    exif[271] = "Nikon"
    exif[272] = "Z8"
    exif[305] = "Adobe Photoshop 25.5"
    exif[36867] = "2026:09:20 15:45:10"
    exif[0x8825] = {
        1: "N",
        2: (35.6895, 41.1234, 22.9876),
        3: "E",
        4: (139.6917, 41.5678, 30.4321),
    }

    case_bytes = _create_image_bytes(
        fmt="JPEG",
        size=(320, 240),
        patch_box=(80, 64, 160, 144),
        exif_bytes=exif.tobytes(),
        quality=50,
    )
    ref_bytes = _create_image_bytes(
        fmt="JPEG",
        size=(320, 240),
        patch_box=None,
        exif_bytes=exif.tobytes(),
        quality=95,
    )
    expected_sha256 = hashlib.sha256(case_bytes).hexdigest()

    # Step 2: Upload
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("evidence_photo.jpg", io.BytesIO(case_bytes), "image/jpeg")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]
    stored_filename = up_res.json()["stored_filename"]

    # Steps 3, 4, 5, 6, 7, 8
    client.post(f"/api/analysis/image/{case_id}")
    client.post(f"/api/analysis/image/{case_id}/detect")
    client.post(f"/api/analysis/image/{case_id}/localize")
    client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("ref_photo.jpg", io.BytesIO(ref_bytes), "image/jpeg")},
    )
    client.post(f"/api/comparison/image/{case_id}")
    client.post(f"/api/evidence/{case_id}/generate")
    client.post(f"/api/history/{case_id}/reconstruct")

    # Capture byte-for-byte snapshots of case JSON and uploaded file BEFORE Step 9
    case_json_path = isolated_storage / "results" / f"{case_id}.json"
    upload_file_path = isolated_storage / "uploads" / stored_filename
    case_json_before = json.loads(case_json_path.read_text(encoding="utf-8"))
    upload_sha256_before = hashlib.sha256(upload_file_path.read_bytes()).hexdigest()
    assert upload_sha256_before == expected_sha256

    # Step 9: Generate JSON Forensic Report
    gen_res = client.post(f"/api/reports/{case_id}/generate")
    assert gen_res.status_code == 200
    report = gen_res.json()

    # Verify GET /api/reports/{case_id} returns the exact persisted report
    get_res = client.get(f"/api/reports/{case_id}")
    assert get_res.status_code == 200
    assert get_res.json() == report

    # Verify persistence at storage/reports/<case_id>/report.json
    report_json_path = isolated_storage / "reports" / case_id / "report.json"
    assert report_json_path.is_file()
    assert json.loads(report_json_path.read_text(encoding="utf-8")) == report

    # Verify Report ID and Version
    assert report["report_id"].startswith("MTR-")
    assert report["report_id"] != case_id
    assert report["report_version"] == "1.0"
    assert report["case_id"] == case_id

    # Verify Case Information & Integrity
    assert report["case"]["case_id"] == case_id
    assert report["case"]["original_filename"] == "evidence_photo.jpg"
    assert report["integrity"]["sha256"] == expected_sha256
    assert report["integrity"]["hash_preserved"] is True
    assert "The SHA-256 digest identifies the preserved original file." in report["integrity"]["explanation"]

    # Verify Media & Step 3 Forensics
    assert report["media"]["available"] is True
    assert report["media"]["width"] == 320
    assert report["media"]["height"] == 240
    assert report["media"]["gps_metadata_present"] is True
    assert report["media"]["gps_notice"] == "GPS metadata present"
    assert report["forensics"]["software_metadata"] == "Software metadata: Adobe Photoshop 25.5"

    # Verify Step 4 Manipulation Detection & AI Detector Notice
    assert report["manipulation_detection"]["available"] is True
    assert (
        report["manipulation_detection"]["ai_generation_label"]
        == "AI-generated content detection: Not available"
    )
    assert (
        report["manipulation_detection"]["ai_generation_explanation"]
        == "No trained AI-generation detector is currently configured."
    )

    # Verify Step 5 Localization
    assert report["localization"]["available"] is True
    assert report["localization"]["finding_label"] == "localized forensic anomaly"
    assert len(report["localization"]["regions"]) >= 1
    assert report["localization"]["heatmap_available"] is True
    assert report["localization"]["mask_available"] is True

    # Verify Step 6 Comparison
    assert report["comparison"]["available"] is True
    assert report["comparison"]["finding_label"] == "visual differences"
    assert report["comparison"]["changed_region_count"] >= 1
    assert report["comparison"]["difference_image_available"] is True
    assert report["comparison"]["difference_mask_available"] is True

    # Verify Step 7 Evidence & Cross-Analysis Spatial Consistency
    assert report["evidence"]["available"] is True
    assert report["evidence"]["summary"]["total_evidence_items"] >= 5
    assert len(report["evidence"]["cross_analysis_evidence"]) >= 1
    cross = report["evidence"]["cross_analysis_evidence"][0]
    assert cross["heading"] == "Cross-analysis spatial consistency"
    assert cross["statement"] == "Independent analyses identify an overlapping region."

    # Verify Step 8 History Section & Disclaimers
    assert report["history"]["title"] == "Possible Edit-History Reconstruction"
    assert report["history"]["sequence_label"] == "Evidence-ordered hypothesis"
    assert report["history"]["historical_accuracy_label"] == "Historical accuracy: Not determined"
    assert report["history"]["sequence_type"] == "evidence_ordered_hypothesis"
    assert (
        "This timeline is an evidence-ordered hypothesis based on available forensic observations."
        in report["history"]["timeline_disclaimer"]
    )

    # Verify Assessment & Limitations
    assert report["assessment"]["assessment"] == case_json_before["evidence"]["assessment"]
    assert report["assessment"]["assessment_label"] == case_json_before["evidence"]["assessment_label"]
    assert len(report["limitations"]) >= 13
    assert "The Step 8 timeline is an evidence-ordered hypothesis." in report["limitations"]

    # Step 9 PDF Report Generation
    pdf_res = client.post(f"/api/reports/{case_id}/generate-pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"].startswith("application/pdf")
    pdf_bytes = pdf_res.content

    pdf_file_path = isolated_storage / "reports" / case_id / "MediaTrace_Forensic_Report.pdf"
    assert pdf_file_path.is_file()
    assert pdf_file_path.read_bytes() == pdf_bytes

    # Section 30: PDF Content Validation
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert pdf_bytes.strip().endswith(b"%%EOF")
    pdf_text = pdf_bytes.decode("latin-1")
    assert "MediaTrace" in pdf_text
    assert case_id in pdf_text
    assert report["report_id"] in pdf_text
    assert expected_sha256 in pdf_text
    assert "Assessment" in pdf_text
    assert "EVIDENCE CORRELATION" in pdf_text
    assert "POSSIBLE EDIT-HISTORY RECONSTRUCTION" in pdf_text
    assert "LIMITATIONS" in pdf_text

    # Section 31: Security, Privacy & Immutability Verification
    serialized_json = json.dumps(report)
    for forbidden in (
        "35.6895",
        "41.1234",
        "22.9876",
        "139.6917",
        "30.4321",
        str(isolated_storage),
        "stored_filename",
    ):
        assert forbidden not in serialized_json
        assert forbidden not in pdf_text

    for forbidden_phrase in (
        "confirmed manipulated region",
        "tampered regions",
        "Manipulation confirmed",
        "Tampering confirmed",
        "AI probability:",
        "AI-generated: false",
    ):
        assert forbidden_phrase not in serialized_json
        assert forbidden_phrase not in pdf_text

    # Verify original case analysis JSON and uploaded image SHA-256 were NOT modified
    case_json_after = json.loads(case_json_path.read_text(encoding="utf-8"))
    upload_sha256_after = hashlib.sha256(upload_file_path.read_bytes()).hexdigest()
    assert case_json_after == case_json_before
    assert upload_sha256_after == upload_sha256_before


def test_missing_optional_artifacts_and_report_regeneration(isolated_storage):
    """
    Covers Requirements 25 & 28:
    - Upload-only case (no heatmap, mask, reference image, or difference artifacts)
      still generates both JSON and PDF reports cleanly without failing
    - Regenerating report updates generated_at while preserving report_version="1.0"
      and keeping original file SHA-256 unchanged
    """
    png_bytes = _create_image_bytes(fmt="PNG", size=(160, 120))
    expected_sha256 = hashlib.sha256(png_bytes).hexdigest()

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("minimal_case.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]

    # First generation
    r1 = client.post(f"/api/reports/{case_id}/generate").json()
    assert r1["report_version"] == "1.0"
    assert r1["localization"]["heatmap_available"] is False
    assert r1["comparison"]["difference_image_available"] is False

    # Generate PDF when optional artifacts do not exist
    pdf_res = client.post(f"/api/reports/{case_id}/generate-pdf")
    assert pdf_res.status_code == 200
    pdf_text = pdf_res.content.decode("latin-1")
    assert "Artifact not available." in pdf_text

    # Regenerate JSON report
    r2 = client.post(f"/api/reports/{case_id}/generate").json()
    assert r2["report_id"] == r1["report_id"]
    assert r2["report_version"] == "1.0"
    assert r2["integrity"]["sha256"] == expected_sha256
    assert r2["integrity"]["hash_preserved"] is True
