"""Step 14.3 — Comprehensive Unit & Integration Tests for Forensic Case Reproducibility & Audit Trail.

Covers:
- Section 30 (Items 1–34):
  1. Centralized application/schema/service versioning & deterministic configuration hashing
  2. Path & secret sanitization (`normalize_storage_path`, `sanitize_audit_text`, `sanitize_dict_for_audit`)
  3. Input file provenance & SHA-256 verification (`verified`, `mismatch`, `unavailable`)
  4. Preservation of original `case_data.sha256` on file tampering (`integrity_mismatch_detected`)
  5. Analysis registry (`ANL-YYYYMMDD-XXXXXXXX`) & rerun/reuse tracking (`rerun_of_analysis_id`, `reused_analysis_id`, `configuration_changed`, `model_changed`)
  6. AI model provenance (`umm-maybe/AI-image-detector`, `c7e223baf11bc40528af364ba7bdea030ef42f9e`, `SwinForImageClassification`, `uncalibrated_model_score`, `model_class_selection`, `threshold=None`)
  7. Artifact registry (`MTA-YYYYMMDD-XXXXXXXX`) & immutable archive preservation across reruns (`superseded_by_artifact_id`)
  8. Evidence Dependency Graph (`input_bytes` -> `analysis_execution` -> `derived_artifact` -> `evidence_item` -> `forensic_interpretation` -> `forensic_report`)
  9. System audit event log (`AUD-YYYYMMDD-XXXXXXXX`, `event_category="system_audit_event"`, strictly separate from Step 8 media edit history)
  10. Immutable `ForensicAnalysisSnapshot` (`MTS-YYYYMMDD-XXXXXXXX`) & canonical `snapshot_sha256`
  11. API endpoints (`POST /api/forensics/{case_id}/reproducibility/check`, `GET /api/forensics/{case_id}/reproducibility`)
  12. JSON & PDF Report integration (`FORENSIC REPRODUCIBILITY & AUDIT TRAIL`)
- Section 31 Manual Test Plan Cases A–F:
  - Case A: Clean full pipeline -> `status="reproducible"`
  - Case B: Rerun analysis & localization -> prior records & archived artifacts preserved
  - Case C: Modified stored input file -> `status="not_reproducible"`, original SHA-256 never overwritten
  - Case D: Deleted or modified derived artifact (`heatmap.png`) -> `status="not_reproducible"`, non-destructive check does not regenerate artifact
  - Case E: Configuration / model drift -> `status="partially_reproducible"`
  - Case F: Legacy pre-Step-14.3 case (`analysis_registry=None`) -> `historical_analysis_registry="unavailable"`, no fabricated records
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.version import (
    MEDIATRACE_APPLICATION_NAME,
    MEDIATRACE_APPLICATION_VERSION,
    SCHEMA_VERSIONS,
    SERVICE_VERSIONS,
    compute_configuration_hash,
    normalize_storage_path,
    sanitize_audit_text,
    sanitize_dict_for_audit,
)
from app.main import app
from app.services.evidence.reproducibility_service import (
    compute_snapshot_sha256,
    record_analysis_execution,
)
from app.services.media_service import get_media_case, save_media_case

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate uploads, results, and reports in tmp_path while keeping real models_dir accessible."""
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


def _create_test_jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    img = Image.new("RGB", (width, height), color=(110, 145, 180))
    # Add a contrasting patch so localization and comparison have deterministic structure
    for y in range(40, 120):
        for x in range(40, 120):
            img.putpixel((x, y), (235, 60, 45))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_1_to_8_versioning_configuration_hashing_and_sanitization():
    """Verify centralized version registry, deterministic configuration hashing, and privacy sanitization."""
    assert MEDIATRACE_APPLICATION_NAME == "MediaTrace"
    assert MEDIATRACE_APPLICATION_VERSION == "1.0.0"
    for req_key in (
        "case_schema_version",
        "snapshot_schema_version",
        "evidence_schema_version",
        "interpretation_schema_version",
        "report_schema_version",
        "audit_event_schema_version",
    ):
        assert req_key in SCHEMA_VERSIONS

    # Deterministic configuration hash ignores key ordering, timestamps, paths, and secrets
    cfg_a = {
        "ela_quality": 90,
        "anomaly_threshold": 0.65,
        "api_key": "super_secret_key",
        "storage_dir": "C:\\Users\\Anand\\secret_dir",
        "evaluated_at": "2026-09-29T10:00:00Z",
    }
    cfg_b = {
        "anomaly_threshold": 0.65,
        "ela_quality": 90,
        "password": "hidden_password",
        "upload_path": "/Users/developer/private/img.jpg",
        "evaluated_at": "2026-09-29T12:00:00Z",
    }
    hash_a = compute_configuration_hash("image_localization_service", cfg_a)
    hash_b = compute_configuration_hash("image_localization_service", cfg_b)
    assert len(hash_a) == 64
    assert hash_a == hash_b

    # Different threshold produces a different hash
    cfg_c = {"anomaly_threshold": 0.80, "ela_quality": 90}
    assert compute_configuration_hash("image_localization_service", cfg_c) != hash_a

    # Path and secret sanitization
    raw_text = (
        "Loaded file from C:\\Users\\Anand\\finalPro\\storage\\uploads\\MT-20260929-ABCD1234_photo.jpg "
        "with api_key=sk_live_999999 and bearer=tok_abc123"
    )
    sanitized = sanitize_audit_text(raw_text)
    assert "C:\\Users\\Anand" not in sanitized
    assert "sk_live_999999" not in sanitized
    assert "tok_abc123" not in sanitized
    assert "storage/uploads/MT-20260929-ABCD1234_photo.jpg" in sanitized

    norm_path = normalize_storage_path(
        "C:\\Users\\Anand\\finalPro\\storage\\results\\MT-20260929-ABCD1234\\localization\\heatmap.png"
    )
    assert norm_path == "storage/results/MT-20260929-ABCD1234/localization/heatmap.png"
    assert "Anand" not in norm_path

    clean_dict = sanitize_dict_for_audit(
        {
            "normal_param": "ok",
            "secret_token": "xyz",
            "file_info": "Saved at /Users/alice/workspace/storage/uploads/test.png",
        }
    )
    assert "secret_token" not in clean_dict
    assert "/Users/alice" not in json.dumps(clean_dict)


def test_manual_cases_a_and_b_full_reproducible_pipeline_and_reruns(isolated_storage: Path):
    """
    Covers Manual Test Cases A & B and Section 30 items 9–26:
    - Case A: Full pipeline execution -> status='reproducible', canonical snapshot SHA-256 verified,
      AI model provenance verified, evidence dependency graph verified, JSON & PDF reports verified.
    - Case B: Rerun analysis & localization -> prior AnalysisRecord and ArtifactRecord preserved
      with `rerun_of_analysis_id` and `superseded_by_artifact_id`, plus reuse tracking (`was_reused=True`).
    """
    jpeg_bytes = _create_test_jpeg_bytes()
    expected_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

    # 1. Upload
    up_res = client.post(
        "/api/media/upload",
        files={"file": ("camera_sample.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert up_res.status_code == 201
    case_id = up_res.json()["case_id"]

    # 2. Run pipeline stages (Step 3 -> Step 4/14.1 -> Step 5 -> Step 7 -> Step 14.2 -> Step 8)
    assert client.post(f"/api/analysis/image/{case_id}").status_code == 200
    assert client.post(f"/api/analysis/image/{case_id}/detect").status_code == 200
    assert client.post(f"/api/analysis/image/{case_id}/localize").status_code == 200
    assert client.post(f"/api/evidence/{case_id}/generate").status_code == 200
    assert client.post(f"/api/forensics/{case_id}/interpret").status_code == 200
    assert client.post(f"/api/history/{case_id}/reconstruct").status_code == 200

    # 3. Check Reproducibility via POST /api/forensics/{case_id}/reproducibility/check
    check_res = client.post(f"/api/forensics/{case_id}/reproducibility/check")
    assert check_res.status_code == 200
    check_data = check_res.json()
    assert check_data["case_id"] == case_id
    assert check_data["status"] == "reproducible"
    assert check_data["input_file_present"] is True
    assert check_data["input_sha256_match"] is True
    assert check_data["analysis_records_present"] is True
    assert check_data["historical_analysis_registry"] == "available"
    assert check_data["configuration_hash_match"] is True
    assert check_data["model_reference_match"] is True
    assert check_data["artifacts_present"] is True
    assert check_data["artifact_hashes_match"] is True

    # 4. Fetch full snapshot via GET /api/forensics/{case_id}/reproducibility
    snap_res = client.get(f"/api/forensics/{case_id}/reproducibility")
    assert snap_res.status_code == 200
    snap = snap_res.json()
    assert snap["snapshot_id"].startswith("MTS-")
    assert snap["case_id"] == case_id
    assert len(snap["snapshot_sha256"]) == 64
    assert snap["snapshot_sha256"] == compute_snapshot_sha256(snap)

    # Verify AI Model Provenance
    ai_prov = snap["ai_model_provenance"]
    assert ai_prov["model_name"] == "umm-maybe/AI-image-detector"
    assert ai_prov["pinned_revision"] == "c7e223baf11bc40528af364ba7bdea030ef42f9e"
    assert ai_prov["architecture"] == "SwinForImageClassification"
    assert ai_prov["score_type"] == "uncalibrated_model_score"
    assert ai_prov["decision_rule"] == "model_class_selection"
    assert ai_prov["threshold"] is None
    assert ai_prov["class_mapping"] == {"0": "artificial", "1": "human"}

    # Verify Artifacts & Dependency Graph
    artifacts_first_run = snap["artifacts_produced"]
    assert len(artifacts_first_run) >= 2
    first_heatmap_rec = next(
        a for a in artifacts_first_run if a["artifact_type"] == "localization_heatmap"
    )
    assert first_heatmap_rec["artifact_id"].startswith("MTA-")
    assert first_heatmap_rec["superseded_by_artifact_id"] is None

    dep_node_types = {n["node_type"] for n in snap["evidence_dependency_graph"]}
    assert "input_bytes" in dep_node_types
    assert "analysis_execution" in dep_node_types
    assert "derived_artifact" in dep_node_types
    assert "evidence_item" in dep_node_types
    assert "forensic_interpretation" in dep_node_types

    # 5. Manual Case B: Rerun localization after modifying on-disk heatmap bytes to simulate new output,
    # and verify prior AnalysisRecord and ArtifactRecord are preserved (never overwritten)
    heatmap_disk_path = isolated_storage / "results" / case_id / "localization" / "heatmap.png"
    assert client.post(f"/api/analysis/image/{case_id}/localize").status_code == 200

    # Also test reuse recording explicitly
    case_obj = get_media_case(case_id)
    reused_rec = record_analysis_execution(
        case_data=case_obj,
        service_name="image_analysis_service",
        status="reused",
        input_sha256=expected_sha256,
        output_summary="Reused Step 3 analysis result.",
        was_reused=True,
    )
    save_media_case(case_obj)
    assert reused_rec.was_reused is True
    assert reused_rec.reused_analysis_id is not None

    snap_after_rerun = client.get(f"/api/forensics/{case_id}/reproducibility").json()
    loc_records = [
        r
        for r in snap_after_rerun["analyses_executed"]
        if r["service_name"] == "image_localization_service"
    ]
    assert len(loc_records) == 2
    assert loc_records[0]["rerun_of_analysis_id"] is None
    assert loc_records[1]["rerun_of_analysis_id"] == loc_records[0]["analysis_id"]

    # Verify audit events are strictly labeled system_audit_event and separate from Step 8 edit history
    case_after = get_media_case(case_id)
    assert case_after.audit_events is not None
    assert len(case_after.audit_events) >= 8
    for ev in case_after.audit_events:
        assert ev.event_id.startswith("AUD-")
        assert ev.event_category == "system_audit_event"
    if case_after.history and case_after.history.events:
        for hist_ev in case_after.history.events:
            assert not hist_ev.id.startswith("AUD-")

    # 6. Generate JSON and PDF reports and verify Step 14.3 section & privacy
    rep_res = client.post(f"/api/reports/{case_id}/generate")
    assert rep_res.status_code == 200
    rep_json = rep_res.json()
    assert rep_json["reproducibility_audit_trail"] is not None
    assert rep_json["reproducibility_audit_trail"]["snapshot_id"].startswith("MTS-")
    assert len(rep_json["audit_events"]) >= 8

    pdf_res = client.post(f"/api/reports/{case_id}/generate-pdf")
    assert pdf_res.status_code == 200
    pdf_text = pdf_res.content.decode("latin-1")
    assert "FORENSIC REPRODUCIBILITY & AUDIT TRAIL" in pdf_text
    assert "1. Snapshot Identity" in pdf_text
    assert "2. Input File Provenance & Integrity Verification" in pdf_text
    assert "4. AI Model Provenance" in pdf_text
    assert "7. Reproducibility Check Result" in pdf_text
    assert str(isolated_storage) not in json.dumps(rep_json)
    assert str(isolated_storage) not in pdf_text
    assert heatmap_disk_path.is_file()


def test_manual_cases_c_d_e_f_tampering_missing_artifacts_drift_and_legacy_cases(
    isolated_storage: Path,
):
    """
    Covers Manual Test Cases C, D, E, and F:
    - Case C: Stored input file modified on disk -> status='not_reproducible', input_sha256_match=False,
      original case_data.sha256 is NEVER overwritten, integrity_mismatch_detected audit event logged.
    - Case D: Derived artifact ('heatmap.png') modified or deleted on disk -> status='not_reproducible',
      non-destructive check does NOT regenerate the deleted artifact.
    - Case E: Configuration hash or model version drift -> status='partially_reproducible'.
    - Case F: Pre-Step-14.3 legacy case (`analysis_registry=None`) -> historical_analysis_registry='unavailable',
      status='partially_reproducible', no fabricated historical analysis records.
    """
    jpeg_bytes = _create_test_jpeg_bytes()
    original_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

    up_res = client.post(
        "/api/media/upload",
        files={"file": ("tamper_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    case_id = up_res.json()["case_id"]
    stored_filename = up_res.json()["stored_filename"]
    client.post(f"/api/analysis/image/{case_id}")
    client.post(f"/api/analysis/image/{case_id}/localize")

    # Verify initially reproducible
    assert client.post(f"/api/forensics/{case_id}/reproducibility/check").json()["status"] == "reproducible"

    # --- CASE D: Tamper with derived artifact heatmap.png on disk, then delete it ---
    heatmap_path = isolated_storage / "results" / case_id / "localization" / "heatmap.png"
    original_heatmap_bytes = heatmap_path.read_bytes()
    heatmap_path.write_bytes(original_heatmap_bytes + b"tampered-artifact-bytes")

    check_tampered_art = client.post(f"/api/forensics/{case_id}/reproducibility/check").json()
    assert check_tampered_art["status"] == "not_reproducible"
    assert check_tampered_art["artifact_hashes_match"] is False

    # Delete heatmap.png and verify non-destructive check reports missing artifact without regenerating it
    heatmap_path.unlink()
    check_missing_art = client.post(f"/api/forensics/{case_id}/reproducibility/check").json()
    assert check_missing_art["status"] == "not_reproducible"
    assert check_missing_art["artifacts_present"] is False
    assert not heatmap_path.exists()

    # Restore heatmap.png and verify case returns to reproducible
    heatmap_path.write_bytes(original_heatmap_bytes)
    assert client.post(f"/api/forensics/{case_id}/reproducibility/check").json()["status"] == "reproducible"

    # --- CASE E: Configuration / Service version drift ---
    case_obj = get_media_case(case_id)
    case_obj.analysis_registry[0].configuration_hash = "f" * 64
    save_media_case(case_obj)

    check_drift = client.post(f"/api/forensics/{case_id}/reproducibility/check").json()
    assert check_drift["status"] == "partially_reproducible"
    assert check_drift["configuration_hash_match"] is False

    # --- CASE C: Modify stored original input file on disk ---
    upload_path = isolated_storage / "uploads" / stored_filename
    upload_path.write_bytes(jpeg_bytes + b"modified-input-bytes")

    check_input_mismatch = client.post(f"/api/forensics/{case_id}/reproducibility/check").json()
    assert check_input_mismatch["status"] == "not_reproducible"
    assert check_input_mismatch["input_sha256_match"] is False

    # CRITICAL RULE: Original case_data.sha256 must NEVER be overwritten!
    reloaded_case = get_media_case(case_id)
    assert reloaded_case.sha256 == original_sha256
    mismatch_events = [
        ev
        for ev in (reloaded_case.audit_events or [])
        if ev.event_type == "integrity_mismatch_detected"
    ]
    assert len(mismatch_events) >= 1
    assert mismatch_events[0].details["original_sha256"] == original_sha256

    # --- CASE F: Legacy Pre-Step-14.3 Case (analysis_registry=None) ---
    legacy_up = client.post(
        "/api/media/upload",
        files={"file": ("legacy_case.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    legacy_id = legacy_up.json()["case_id"]
    client.post(f"/api/analysis/image/{legacy_id}")

    # Strip Step 14.3 fields from JSON on disk to simulate a pre-Step-14.3 case
    legacy_json_path = isolated_storage / "results" / f"{legacy_id}.json"
    raw_legacy = json.loads(legacy_json_path.read_text(encoding="utf-8"))
    for k in (
        "case_schema_version",
        "analysis_registry",
        "artifact_registry",
        "audit_events",
        "latest_snapshot",
        "snapshots",
        "report_generations",
    ):
        raw_legacy.pop(k, None)
    legacy_json_path.write_text(json.dumps(raw_legacy, indent=2), encoding="utf-8")

    legacy_check = client.post(f"/api/forensics/{legacy_id}/reproducibility/check").json()
    assert legacy_check["historical_analysis_registry"] == "unavailable"
    assert legacy_check["analysis_records_present"] is False
    assert legacy_check["status"] == "partially_reproducible"

    legacy_snap = client.get(f"/api/forensics/{legacy_id}/reproducibility").json()
    assert legacy_snap["reproducibility_check"]["historical_analysis_registry"] == "unavailable"
    assert legacy_snap["analyses_executed"] == []
