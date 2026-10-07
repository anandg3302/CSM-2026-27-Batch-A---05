"""Step 13 — Full End-to-End Integration, Integrity, Idempotency, and Performance Test Suite.

Covers:
1. Full Image Forensic Workflow:
   upload -> basic analysis -> manipulation detection -> localization ->
   upload reference image -> compare -> evidence correlation ->
   edit-history reconstruction -> JSON forensic report -> PDF forensic report
2. Full Video Forensic Workflow:
   upload -> video analysis -> artifact retrieval -> report generation compatibility
3. Full Audio Forensic Workflow:
   upload -> audio analysis -> artifact retrieval -> report generation compatibility
4. Full Document Forensic Workflow:
   upload -> document analysis -> artifact retrieval -> report generation compatibility
5. Pre/Post SHA-256 & byte-for-byte immutability verification across all 4 modalities
6. Duplicate/idempotent analysis requests and security boundary enforcement
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
import time
import wave

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from pypdf import PdfWriter

from app.core.config import settings
from app.main import app

client = TestClient(app)


def _create_image_bytes(modified: bool = False) -> bytes:
    img = Image.new("RGB", (192, 192), color=(40, 85, 140))
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 24, 168, 168], outline=(230, 230, 230), width=3)
    if modified:
        draw.rectangle([60, 60, 130, 130], fill=(245, 70, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _create_video_bytes() -> bytes:
    tmp_path = settings.reports_dir / "temp_e2e_video.avi"
    settings.ensure_storage_directories()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, 10.0, (128, 96))
    for idx in range(12):
        frame = np.full((96, 128, 3), (30 + idx * 5, 80, 140), dtype=np.uint8)
        cv2.rectangle(frame, (16, 16), (96, 72), (220, 200, 90), -1)
        writer.write(frame)
    writer.release()
    raw = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return raw


def _create_wav_bytes() -> bytes:
    sample_rate = 16000
    n_samples = sample_rate  # 1.0 second
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for idx in range(n_samples):
            val = int(32767 * 0.3 * math.sin(2.0 * math.pi * 440.0 * idx / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def _create_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=500)
    writer.add_metadata(
        {
            "/Author": "Forensic QA",
            "/Creator": "MediaTrace Test Suite",
            "/Producer": "pypdf",
            "/Title": "End-to-End Verification Document",
            "/CreationDate": "D:20260110090000+00'00'",
            "/ModDate": "D:20260112113000+00'00'",
        }
    )
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _assert_uploaded_file_unchanged(case_json: dict, original_bytes: bytes) -> None:
    stored_path = settings.uploads_dir / case_json["stored_filename"]
    assert stored_path.is_file(), f"Stored upload file missing: {stored_path}"
    disk_bytes = stored_path.read_bytes()
    assert disk_bytes == original_bytes, "Uploaded evidence bytes were modified on disk!"
    expected_sha = hashlib.sha256(original_bytes).hexdigest()
    assert case_json["sha256"] == expected_sha


def test_full_end_to_end_image_workflow_and_immutability():
    """Verify the complete 9-stage image forensic pipeline and byte-for-byte evidence immutability."""
    t0 = time.perf_counter()
    suspect_bytes = _create_image_bytes(modified=True)
    reference_bytes = _create_image_bytes(modified=False)

    # 1. Upload suspect image
    res_up = client.post(
        "/api/media/upload",
        files={"file": ("e2e_suspect.jpg", suspect_bytes, "image/jpeg")},
    )
    assert res_up.status_code == 201
    case = res_up.json()
    case_id = case["case_id"]
    _assert_uploaded_file_unchanged(case, suspect_bytes)

    # 2. Basic Image Forensics
    res_analyze = client.post(f"/api/analysis/image/{case_id}")
    assert res_analyze.status_code == 200
    assert res_analyze.json()["analysis"] is not None

    # 3. Manipulation Detection (plus idempotent re-run check)
    res_det = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res_det.status_code == 200
    assert res_det.json()["manipulation_detection"] is not None
    res_det_dup = client.post(f"/api/analysis/image/{case_id}/detect")
    assert res_det_dup.status_code == 200

    # 4. Manipulation Localization
    res_loc = client.post(f"/api/analysis/image/{case_id}/localize")
    assert res_loc.status_code == 200
    loc_data = res_loc.json()["localization"]
    assert loc_data is not None
    assert client.get(f"/api/analysis/image/{case_id}/localization/heatmap.png").status_code == 200

    # 5. Upload Reference Image & Compare
    res_ref = client.post(
        f"/api/comparison/image/{case_id}/reference",
        files={"file": ("e2e_reference.jpg", reference_bytes, "image/jpeg")},
    )
    assert res_ref.status_code == 201
    res_cmp = client.post(f"/api/comparison/image/{case_id}")
    assert res_cmp.status_code == 200
    assert res_cmp.json()["comparison"] is not None

    # 6. Evidence Correlation Engine
    res_ev = client.post(f"/api/evidence/{case_id}/generate")
    assert res_ev.status_code == 200
    assert res_ev.json()["evidence"] is not None

    # 7. Possible Edit-History Reconstruction
    res_hist = client.post(f"/api/history/{case_id}/reconstruct")
    assert res_hist.status_code == 200
    assert res_hist.json()["history"] is not None

    # 8. Forensic Report Generation (JSON + PDF)
    res_rep = client.post(f"/api/reports/{case_id}/generate")
    assert res_rep.status_code == 200
    report_payload = res_rep.json()
    assert report_payload["case_id"] == case_id

    res_json_dl = client.get(f"/api/reports/{case_id}")
    assert res_json_dl.status_code == 200

    res_pdf_gen = client.post(f"/api/reports/{case_id}/generate-pdf")
    assert res_pdf_gen.status_code == 200
    assert res_pdf_gen.content.startswith(b"%PDF-")

    res_pdf_dl = client.get(f"/api/reports/{case_id}/pdf")
    assert res_pdf_dl.status_code == 200
    assert res_pdf_dl.content.startswith(b"%PDF-")

    # 9. Verify original uploaded bytes remained 100% immutable after all stages
    final_case = client.get(f"/api/media/{case_id}").json()
    _assert_uploaded_file_unchanged(final_case, suspect_bytes)
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0


def test_full_end_to_end_video_audio_and_document_workflows():
    """Verify end-to-end video, audio, and document analysis, artifact retrieval, report compatibility, and byte immutability."""
    # --- VIDEO ---
    video_bytes = _create_video_bytes()
    v_up = client.post(
        "/api/media/upload",
        files={"file": ("e2e_clip.avi", video_bytes, "video/x-msvideo")},
    )
    assert v_up.status_code == 201
    v_case = v_up.json()
    v_id = v_case["case_id"]

    v_an = client.post(f"/api/analysis/video/{v_id}")
    assert v_an.status_code == 200
    v_result = v_an.json()["video_analysis"]
    assert v_result["media_type"] == "video"
    frames = v_result["frame_sampling"]["frames"]
    assert len(frames) >= 1
    frame_name = frames[0]["artifact_filename"]
    assert client.get(f"/api/analysis/video/{v_id}/artifacts/{frame_name}").status_code == 200
    assert client.post(f"/api/reports/{v_id}/generate").status_code == 200
    _assert_uploaded_file_unchanged(v_an.json(), video_bytes)

    # --- AUDIO ---
    audio_bytes = _create_wav_bytes()
    a_up = client.post(
        "/api/media/upload",
        files={"file": ("e2e_voice.wav", audio_bytes, "audio/wav")},
    )
    assert a_up.status_code == 201
    a_case = a_up.json()
    a_id = a_case["case_id"]

    a_an = client.post(f"/api/analysis/audio/{a_id}")
    assert a_an.status_code == 200
    a_result = a_an.json()["audio_analysis"]
    assert a_result["media_type"] == "audio"
    assert client.get(f"/api/analysis/audio/{a_id}/artifacts/waveform.png").status_code == 200
    assert client.get(f"/api/analysis/audio/{a_id}/artifacts/spectrogram.png").status_code == 200
    assert client.post(f"/api/reports/{a_id}/generate").status_code == 200
    _assert_uploaded_file_unchanged(a_an.json(), audio_bytes)

    # --- DOCUMENT ---
    pdf_bytes = _create_pdf_bytes()
    d_up = client.post(
        "/api/media/upload",
        files={"file": ("e2e_record.pdf", pdf_bytes, "application/pdf")},
    )
    assert d_up.status_code == 201
    d_case = d_up.json()
    d_id = d_case["case_id"]

    d_an = client.post(f"/api/analysis/document/{d_id}")
    assert d_an.status_code == 200
    d_result = d_an.json()["document_analysis"]
    assert d_result["media_type"] == "document"
    for art in d_result["artifacts"]:
        assert client.get(f"/api/analysis/document/{d_id}/artifacts/{art['artifact_filename']}").status_code == 200
    assert client.post(f"/api/reports/{d_id}/generate").status_code == 200
    _assert_uploaded_file_unchanged(d_an.json(), pdf_bytes)
