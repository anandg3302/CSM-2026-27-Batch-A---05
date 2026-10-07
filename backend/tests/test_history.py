import hashlib
import io
import json
import re
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import settings
from app.main import app

client = TestClient(app)

ALLOWED_CONFIDENCES = {"high", "moderate", "low", "undetermined"}
ALLOWED_STATUSES = {"observed", "supported_inference", "possible", "unknown"}
ALLOWED_TEMPORAL = {"exact", "approximate", "relative", "unknown"}


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


def _assert_common_history_invariants(history: dict, expected_sha256: str) -> None:
    """Verify structural, confidence, evidence-linking, and deterministic sequence invariants."""
    assert history["reconstructed_sequence"] is True
    assert history["sequence_type"] == "evidence_ordered_hypothesis"
    assert history["historical_accuracy"] in {"not_determined", "partially_supported"}
    assert history["integrity"]["original_sha256"] == expected_sha256
    assert history["integrity"]["hash_preserved"] is True

    # Requirement 17: unknowns and limitations lists are non-empty
    assert len(history["unknowns"]) > 0
    assert len(history["limitations"]) > 0

    events = history["events"]
    assert len(events) >= 2

    # Requirement 8: First timeline event is always case recording / file ingestion
    assert events[0]["id"] == "EH-001"
    assert events[0]["sequence"] == 1
    assert events[0]["event_type"] == "original_capture_or_creation"
    assert events[0]["status"] == "observed"
    assert events[0]["confidence"] == "high"
    assert events[0]["temporal_certainty"] == "exact"

    # Requirement 21: Event sequence numbers are deterministic (1..N)
    for idx, ev in enumerate(events, start=1):
        assert ev["sequence"] == idx
        assert ev["id"] == f"EH-{idx:03d}"
        # Requirement 15: Every event contains evidence_ids
        assert isinstance(ev["evidence_ids"], list)
        assert len(ev["evidence_ids"]) > 0
        # Requirement 16: Every event contains reasoning and limitations
        assert isinstance(ev["reasoning"], str) and len(ev["reasoning"].strip()) > 0
        assert isinstance(ev["limitations"], list) and len(ev["limitations"]) > 0
        # Requirement 20: Qualitative confidence only, never numerical probabilities
        assert ev["confidence"] in ALLOWED_CONFIDENCES
        assert ev["status"] in ALLOWED_STATUSES
        assert ev["temporal_certainty"] in ALLOWED_TEMPORAL
        assert not re.match(r"^\d+(\.\d+)?%?$", str(ev["confidence"]))


def test_history_invalid_and_missing_cases():
    """
    Covers:
    2. Invalid case ID returns 400
    3. Non-existent case ID returns 404
    4. GET /api/history/{case_id} before reconstruction returns 404
    """
    # Invalid case ID format -> 400
    assert client.post("/api/history/invalid-id/reconstruct").status_code == 400
    assert client.get("/api/history/invalid-id").status_code == 400

    # Valid format but non-existent case -> 404
    missing_id = "MT-20260928-DEADBEEF"
    assert client.post(f"/api/history/{missing_id}/reconstruct").status_code == 404
    assert client.get(f"/api/history/{missing_id}").status_code == 404

    # Upload a real case, then call GET /api/history/{case_id} before reconstruction -> 404
    png_bytes = _create_image_bytes(fmt="PNG", size=(160, 120))
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("clean.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]

    pre_get = client.get(f"/api/history/{case_id}")
    assert pre_get.status_code == 404


def test_edge_case_b_upload_only_without_prior_stages():
    """
    Covers Edge Case B:
    - Upload image
    - Do NOT run Steps 3–6
    - Call POST /api/history/{case_id}/reconstruct
    - Verify reconstruction still works safely, includes upload observation, and lists missing stages in unknowns.
    """
    png_bytes = _create_image_bytes(fmt="PNG", size=(160, 120))
    expected_sha256 = hashlib.sha256(png_bytes).hexdigest()

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("only_upload.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]

    rec_res = client.post(f"/api/history/{case_id}/reconstruct")
    assert rec_res.status_code == 200
    case_payload = rec_res.json()
    history = case_payload["history"]

    _assert_common_history_invariants(history, expected_sha256)
    assert history["historical_accuracy"] == "not_determined"
    assert history["inferred_events"] == []

    unknown_text = " ".join(history["unknowns"])
    assert "Step 3" in unknown_text
    assert "Step 5" in unknown_text
    assert "Step 6" in unknown_text


def test_edge_case_a_clean_png_no_anomalies():
    """
    Covers Edge Case A & Requirement 19:
    - Upload clean PNG
    - Run analysis / detection / localization / evidence / history
    - Verify known facts exist (image dimensions and SHA-256 integrity)
    - Verify no unsupported visual edit claims and AI detector 'not_available' does not create an AI event
    """
    png_bytes = _create_image_bytes(fmt="PNG", size=(200, 150))
    expected_sha256 = hashlib.sha256(png_bytes).hexdigest()

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("clean_sample.png", io.BytesIO(png_bytes), "image/png")},
    )
    case_id = up_res.json()["case_id"]

    client.post(f"/api/analysis/image/{case_id}")
    client.post(f"/api/analysis/image/{case_id}/detect")
    client.post(f"/api/analysis/image/{case_id}/localize")
    client.post(f"/api/evidence/{case_id}/generate")

    rec_res = client.post(f"/api/history/{case_id}/reconstruct")
    assert rec_res.status_code == 200
    history = rec_res.json()["history"]

    _assert_common_history_invariants(history, expected_sha256)

    # Requirement 9: Known facts include image dimensions and SHA-256 integrity
    facts_str = " ".join(f["fact"] for f in history["known_facts"])
    assert "200x150" in facts_str
    assert expected_sha256 in facts_str

    # No unsupported visual edit claims or AI generation events
    event_types = [ev["event_type"] for ev in history["events"]]
    assert "possible_visual_edit" not in event_types
    serialized_history = json.dumps(history).lower()
    assert "confirmed_ai_generated" not in serialized_history
    assert "ai_generated_event" not in serialized_history


def test_edge_cases_c_d_e_f_g_full_pipeline_correlation_and_persistence(isolated_storage):
    """
    Covers:
    1. History reconstruction succeeds (POST and GET)
    5. History report is persisted inside storage/results/<case_id>.json
    6. Existing fields remain unchanged (analysis, manipulation_detection, localization, comparison, evidence)
    7. Original uploaded file SHA-256 remains unchanged
    10. Software metadata (Adobe Photoshop / GIMP) creates metadata_processing event with supported_inference (Case C)
    11. Compression/recompression indicators create possible_recompression event with possible status
    12. Localization regions create possible_visual_edit event with possible status and low confidence (Case D)
    13. Reference comparison differences create reference_difference_observed event with observed status (Case E)
    14. Overlapping localization + comparison regions create possible_visual_edit event with supported_inference and moderate confidence (Case F)
    18. GPS coordinates are never exposed in history output (Case G)
    """
    exif = Image.Exif()
    exif[271] = "Canon"
    exif[272] = "EOS R5"
    exif[305] = "GIMP 2.10.36"
    exif[36867] = "2026:08:14 09:30:00"
    exif[0x8825] = {
        1: "N",
        2: (37.0, 46.0, 29.6),
        3: "W",
        4: (122.0, 25.0, 10.0),
    }

    # Create PNG bytes but upload with .jpg extension so extension_format_mismatch triggers possible_recompression,
    # or create JPEG with localized patch + EXIF software + low quality recompression
    case_bytes = _create_image_bytes(
        fmt="JPEG",
        size=(320, 240),
        patch_box=(80, 64, 160, 144),
        exif_bytes=exif.tobytes(),
        quality=45,
    )
    ref_bytes = _create_image_bytes(
        fmt="JPEG",
        size=(320, 240),
        patch_box=None,
        exif_bytes=exif.tobytes(),
        quality=95,
    )
    expected_sha256 = hashlib.sha256(case_bytes).hexdigest()

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("edited_case.jpg", io.BytesIO(case_bytes), "image/jpeg")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]
    stored_filename = up_res.json()["stored_filename"]

    s3_snapshot = client.post(f"/api/analysis/image/{case_id}").json()["analysis"]
    s4_snapshot = client.post(f"/api/analysis/image/{case_id}/detect").json()["manipulation_detection"]
    s5_snapshot = client.post(f"/api/analysis/image/{case_id}/localize").json()["localization"]

    client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("reference_orig.jpg", io.BytesIO(ref_bytes), "image/jpeg")},
    )
    s6_snapshot = client.post(f"/api/comparison/image/{case_id}").json()["comparison"]
    s7_snapshot = client.post(f"/api/evidence/{case_id}/generate").json()["evidence"]

    # Step 8: Reconstruct Possible Edit History
    rec_res = client.post(f"/api/history/{case_id}/reconstruct")
    assert rec_res.status_code == 200
    full_case = rec_res.json()

    # Requirement 6: Existing fields remain unchanged
    assert full_case["analysis"] == s3_snapshot
    assert full_case["manipulation_detection"] == s4_snapshot
    assert full_case["localization"] == s5_snapshot
    assert full_case["comparison"] == s6_snapshot
    assert full_case["evidence"] == s7_snapshot

    # Requirement 7: Original uploaded file SHA-256 remains unchanged on disk
    upload_file_path = isolated_storage / "uploads" / stored_filename
    assert hashlib.sha256(upload_file_path.read_bytes()).hexdigest() == expected_sha256

    # Requirement 1 & 5: GET endpoint matches persisted JSON inside storage/results/<case_id>.json
    get_res = client.get(f"/api/history/{case_id}")
    assert get_res.status_code == 200
    history = get_res.json()
    assert history == full_case["history"]

    persisted_path = isolated_storage / "results" / f"{case_id}.json"
    with open(persisted_path, "r", encoding="utf-8") as f_in:
        persisted_json = json.load(f_in)
    assert persisted_json["history"] == history

    _assert_common_history_invariants(history, expected_sha256)
    assert history["historical_accuracy"] == "partially_supported"

    events = history["events"]

    # Requirement 10 & Case C: Software metadata creates metadata_processing event with supported_inference
    sw_events = [
        ev
        for ev in events
        if ev["event_type"] == "metadata_processing" and ev["status"] == "supported_inference"
    ]
    assert len(sw_events) == 1
    assert "GIMP 2.10.36" in sw_events[0]["title"]
    assert sw_events[0]["confidence"] == "moderate"

    # Requirement 11: Compression/recompression indicators create possible_recompression event with possible status
    recomp_events = [
        ev for ev in events if ev["event_type"] == "possible_recompression"
    ]
    assert len(recomp_events) == 1
    assert recomp_events[0]["status"] == "possible"
    assert recomp_events[0]["confidence"] == "low"

    # Requirement 12 & Case D: Localization regions create possible_visual_edit event with possible status and low confidence
    loc_events = [
        ev
        for ev in events
        if ev["event_type"] == "possible_visual_edit" and ev["source"] == "step5_localization"
    ]
    assert len(loc_events) == 1
    assert loc_events[0]["status"] == "possible"
    assert loc_events[0]["confidence"] == "low"

    # Requirement 13 & Case E: Reference comparison differences create reference_difference_observed event with observed status
    comp_events = [
        ev for ev in events if ev["event_type"] == "reference_difference_observed"
    ]
    assert len(comp_events) == 1
    assert comp_events[0]["status"] == "observed"
    assert comp_events[0]["confidence"] == "high"
    assert comp_events[0]["temporal_certainty"] == "relative"

    # Requirement 14 & Case F: Overlapping localization + comparison regions create possible_visual_edit with supported_inference & moderate confidence
    overlap_events = [
        ev
        for ev in events
        if ev["event_type"] == "possible_visual_edit"
        and ev["source"] == "step7_evidence_correlation"
    ]
    assert len(overlap_events) == 1
    assert overlap_events[0]["status"] == "supported_inference"
    assert overlap_events[0]["confidence"] == "moderate"

    # Requirement 18 & Case G: GPS coordinates are never exposed in history output
    serialized = json.dumps(history)
    assert "37.0" not in serialized
    assert "46.0" not in serialized
    assert "122.0" not in serialized
    assert "29.6" not in serialized
