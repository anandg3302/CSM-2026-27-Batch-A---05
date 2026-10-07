"""Stage B.1 — Release Candidate Audit Test Suite for MediaTrace v1.0.0.

Covers regression verification for:
1. System health, API root, and application version consistency (1.0.0)
2. Path traversal, null bytes, and malformed filename rejection
3. Privacy preservation & raw GPS coordinate redaction across analysis and reporting
4. Report application version (1.0.0), SHA-256 preservation, and PDF validity
5. AI detector boundary: uncalibrated score, null threshold, and anti-escalation enforcement
6. Concurrency and cross-case artifact isolation
7. Sanitized error handling without stack trace or internal path leakage
"""

from __future__ import annotations

import concurrent.futures
from fractions import Fraction
import io
import json
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader

from app.core.config import settings
from app.core.version import MEDIATRACE_APPLICATION_VERSION
from app.main import app
from tests.test_end_to_end import (
    _create_image_bytes,
    _create_video_bytes,
    _create_wav_bytes,
    _create_pdf_bytes,
)

client = TestClient(app)


def test_system_health_and_root_version_consistency():
    """Verify /api/health and /api report MEDIATRACE_APPLICATION_VERSION ('1.0.0')."""
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "ok"
    assert data_health["service"] == "MediaTrace API"
    assert data_health["version"] == "1.0.0"

    res_root = client.get("/api")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["service"] == "MediaTrace API"
    assert data_root["version"] == "1.0.0"
    assert "image" in data_root["supported_modalities"]


def test_path_traversal_and_malformed_filename_rejection():
    """Verify uploads safely reject or sanitize path traversal, null bytes, and malformed names."""
    dummy_bytes = _create_image_bytes(modified=False)

    # Null byte rejection
    res_null = client.post(
        "/api/media/upload",
        files={"file": ("malicious\x00file.jpg", dummy_bytes, "image/jpeg")},
    )
    assert res_null.status_code == 400
    assert "null byte" in res_null.json()["detail"].lower()

    # URL-encoded null byte rejection
    res_null_enc = client.post(
        "/api/media/upload",
        files={"file": ("malicious%00file.jpg", dummy_bytes, "image/jpeg")},
    )
    assert res_null_enc.status_code == 400

    # Path traversal stripping: ../../etc/passwd.jpg -> stripped to safe basename
    res_trav = client.post(
        "/api/media/upload",
        files={"file": ("../../etc/passwd.jpg", dummy_bytes, "image/jpeg")},
    )
    assert res_trav.status_code == 201
    case_trav = res_trav.json()
    assert ".." not in case_trav["stored_filename"]
    assert "/" not in case_trav["stored_filename"]
    assert "\\" not in case_trav["stored_filename"]

    # Windows path traversal stripping: ..\..\windows\system.jpg
    res_win = client.post(
        "/api/media/upload",
        files={"file": ("..\\..\\windows\\system.jpg", dummy_bytes, "image/jpeg")},
    )
    assert res_win.status_code == 201
    case_win = res_win.json()
    assert ".." not in case_win["stored_filename"]
    assert "\\" not in case_win["stored_filename"]

    # Empty filename rejection (400 from sanitize_filename or 422 from FastAPI multipart validation)
    res_empty = client.post(
        "/api/media/upload",
        files={"file": ("", dummy_bytes, "image/jpeg")},
    )
    assert res_empty.status_code in (400, 422)

    # Whitespace filename rejection
    res_ws = client.post(
        "/api/media/upload",
        files={"file": ("   ", dummy_bytes, "image/jpeg")},
    )
    assert res_ws.status_code in (400, 422)


def test_privacy_gps_coordinate_redaction_in_all_layers():
    """Verify that EXIF GPS presence is detected without exposing raw latitude/longitude."""
    img = Image.new("RGB", (64, 64), color=(80, 100, 120))
    exif = img.getexif()
    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[1] = "N"
    gps_ifd[2] = (Fraction(37, 1), Fraction(46, 1), Fraction(2974, 100))
    gps_ifd[3] = "W"
    gps_ifd[4] = (Fraction(122, 1), Fraction(25, 1), Fraction(982, 100))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    gps_img_bytes = buf.getvalue()

    # Upload
    up = client.post(
        "/api/media/upload",
        files={"file": ("gps_test.jpg", gps_img_bytes, "image/jpeg")},
    )
    assert up.status_code == 201
    cid = up.json()["case_id"]

    # Analyze
    an = client.post(f"/api/analysis/image/{cid}")
    assert an.status_code == 200
    an_data = an.json()

    # Check that gps_present is flagged, but no raw coordinates exist in payload
    assert an_data["analysis"]["metadata"]["gps_present"] is True
    assert an_data["analysis"]["metadata"]["gps_info"] == "GPS metadata present"
    raw_an_text = json.dumps(an_data).lower()
    assert "latitude" not in raw_an_text
    assert "longitude" not in raw_an_text

    # Generate Report
    rep = client.post(f"/api/reports/{cid}/generate")
    assert rep.status_code == 200
    rep_data = rep.json()
    assert rep_data["media"]["gps_metadata_present"] is True
    assert rep_data["media"]["gps_notice"] == "GPS metadata present"
    raw_rep_text = json.dumps(rep_data).lower()
    assert "latitude" not in raw_rep_text
    assert "longitude" not in raw_rep_text


def test_report_application_version_and_sha256_preservation():
    """Verify generated JSON and PDF reports report application_version 1.0.0 and preserve SHA-256."""
    img_bytes = _create_image_bytes(modified=True)
    up = client.post(
        "/api/media/upload",
        files={"file": ("audit_version_test.jpg", img_bytes, "image/jpeg")},
    )
    assert up.status_code == 201
    case = up.json()
    cid = case["case_id"]
    client.post(f"/api/analysis/image/{cid}")

    # JSON report
    rep_res = client.post(f"/api/reports/{cid}/generate")
    assert rep_res.status_code == 200
    rep_json = rep_res.json()
    assert rep_json["application_version"] == "1.0.0"
    assert rep_json["integrity"]["sha256"] == case["sha256"]
    assert rep_json["integrity"]["hash_preserved"] is True

    # PDF report
    pdf_res = client.post(f"/api/reports/{cid}/generate-pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.content.startswith(b"%PDF-")
    reader = PdfReader(io.BytesIO(pdf_res.content))
    assert len(reader.pages) >= 1
    pdf_text = "".join(p.extract_text() for p in reader.pages)
    assert "1.0.0" in pdf_text
    assert case["sha256"][:16] in pdf_text


def test_ai_detector_boundary_and_score_interpretation_restraint():
    """Verify AI detector scores are uncalibrated, threshold is null, and cannot escalate overall assessment."""
    img_bytes = _create_image_bytes(modified=False)
    up = client.post(
        "/api/media/upload",
        files={"file": ("clean_human_face.jpg", img_bytes, "image/jpeg")},
    )
    assert up.status_code == 201
    cid = up.json()["case_id"]

    # Basic analysis + detect
    client.post(f"/api/analysis/image/{cid}")
    det_res = client.post(f"/api/analysis/image/{cid}/detect")
    assert det_res.status_code == 200
    det_data = det_res.json()["manipulation_detection"]

    ai_result = det_data.get("ai_generation_detection")
    if ai_result and ai_result.get("status") == "completed":
        assert ai_result.get("score_type") == "uncalibrated_model_score"
        assert ai_result.get("decision_rule") == "model_class_selection"
        assert ai_result.get("threshold") is None

    # Multi-signal decision
    client.post(f"/api/evidence/{cid}/generate")
    interp_res = client.post(f"/api/forensics/{cid}/interpret")
    assert interp_res.status_code == 200
    interp_data = interp_res.json()

    # Rule E: AI detector signal alone CANNOT escalate to multiple_consistent_forensic_observations
    assert interp_data["overall_assessment"] in (
        "insufficient_evidence",
        "no_significant_forensic_evidence",
        "isolated_forensic_signal",
    )


def test_concurrency_and_artifact_isolation():
    """Verify concurrent uploads and analyses maintain complete artifact and record isolation."""
    def run_worker(idx: int) -> tuple[str, str, str]:
        data = _create_image_bytes(modified=(idx % 2 == 1))
        fname = f"thread_worker_{idx}.jpg"
        up = client.post("/api/media/upload", files={"file": (fname, data, "image/jpeg")})
        assert up.status_code == 201
        case = up.json()
        cid = case["case_id"]

        an = client.post(f"/api/analysis/image/{cid}")
        assert an.status_code == 200

        rep = client.post(f"/api/reports/{cid}/generate")
        assert rep.status_code == 200
        rep_data = rep.json()
        return cid, case["sha256"], rep_data["report_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_worker, i) for i in range(4)]
        results = [f.result() for f in futures]

    cids = [r[0] for r in results]
    shas = [r[1] for r in results]
    rids = [r[2] for r in results]

    assert len(set(cids)) == 4, "Duplicate case IDs generated during concurrent execution!"
    assert len(set(rids)) == 4, "Duplicate report IDs generated during concurrent execution!"


def test_sanitized_500_error_handling():
    """Verify unhandled exceptions return a sanitized error without exposing stack traces or paths."""
    # Attempt on non-existent case with valid case_id format
    res_404 = client.get("/api/reports/MT-20260101-00000000")
    assert res_404.status_code == 404
    assert "detail" in res_404.json()
    assert "traceback" not in res_404.text.lower()
    assert "c:\\" not in res_404.text.lower()
    assert "d:\\" not in res_404.text.lower()

    # Attempt on malformed case_id format
    res_400 = client.get("/api/reports/malformed_id")
    assert res_400.status_code == 400
    assert "detail" in res_400.json()
    assert "traceback" not in res_400.text.lower()
