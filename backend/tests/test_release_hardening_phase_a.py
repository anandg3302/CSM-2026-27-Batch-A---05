import io
import math
import struct
import wave
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture()
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    return tmp_path


def _make_valid_jpeg_bytes() -> bytes:
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[:, :, 0] = 120
    arr[20:60, 20:60, 1] = 210
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_valid_mp4_bytes(tmp_path: Path) -> bytes:
    mp4_path = tmp_path / "sample_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, 10.0, (128, 96))
    for i in range(12):
        frame = np.full((96, 128, 3), fill_value=(i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return mp4_path.read_bytes()


def _make_valid_wav_bytes() -> bytes:
    buf = io.BytesIO()
    sample_rate = 16000
    duration_s = 1.0
    n_samples = int(sample_rate * duration_s)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(12000 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def _make_valid_docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/></Types>',
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>MediaTrace v1.0 Phase A validation document.</w:t></w:r></w:p></w:body>"
            "</w:document>",
        )
    return buf.getvalue()


def test_input_validation_null_bytes_long_filenames_unicode_and_truncated_jpeg(
    isolated_storage: Path,
):
    """
    Verify Phase A Step 3 & Step 6 hardening:
    - Null bytes in filenames rejected with 400
    - Very long filenames (300+ chars) safely bounded and accepted without OS errors
    - Unicode filenames sanitized safely
    - Truncated JPEG (valid SOI/APP0 header but missing scan data) rejected with 400 at analyze_image
      without marking the case as 'analyzed'
    """
    valid_jpeg = _make_valid_jpeg_bytes()

    # 1. Null byte filename rejected
    res_null = client.post(
        "/api/media/upload",
        files={"file": ("photo\x00.jpg", io.BytesIO(valid_jpeg), "image/jpeg")},
    )
    assert res_null.status_code == 400
    assert "null byte" in res_null.json()["detail"].lower()

    res_pct00 = client.post(
        "/api/media/upload",
        files={"file": ("photo%00.jpg", io.BytesIO(valid_jpeg), "image/jpeg")},
    )
    assert res_pct00.status_code == 400

    # 2. Very long filename (320 chars) handled safely without OS/MAX_PATH crash
    long_name = ("camera_evidence_" * 20) + ".jpg"
    assert len(long_name) > 300
    res_long = client.post(
        "/api/media/upload",
        files={"file": (long_name, io.BytesIO(valid_jpeg), "image/jpeg")},
    )
    assert res_long.status_code == 201
    long_case = res_long.json()
    assert len(long_case["original_filename"]) <= 105
    assert long_case["original_filename"].endswith(".jpg")

    # 3. Unicode filename handled safely
    res_uni = client.post(
        "/api/media/upload",
        files={"file": ("фото_証拠_2026.jpg", io.BytesIO(valid_jpeg), "image/jpeg")},
    )
    assert res_uni.status_code == 201
    assert res_uni.json()["original_filename"].endswith(".jpg")

    # 4. Truncated JPEG header (valid 24-byte header, missing image stream) rejected with 400
    truncated_jpeg = valid_jpeg[:24]
    res_trunc_up = client.post(
        "/api/media/upload",
        files={"file": ("truncated.jpg", io.BytesIO(truncated_jpeg), "image/jpeg")},
    )
    assert res_trunc_up.status_code == 201
    trunc_case_id = res_trunc_up.json()["case_id"]

    res_trunc_anl = client.post(f"/api/analysis/image/{trunc_case_id}")
    assert res_trunc_anl.status_code == 400
    # Ensure failed analysis did not corrupt or mark case as 'analyzed'
    case_after = client.get(f"/api/media/{trunc_case_id}").json()
    assert case_after["status"] == "uploaded"
    assert case_after["analysis"] is None


def test_multimodal_reproducibility_and_reporting_video_audio_document_history(
    isolated_storage: Path,
):
    """
    Verify Phase A Step 2, Step 5, Step 9 & Step 14.3 integration across Video, Audio,
    Document, and Image History workflows:
    - Live execution records (`ANL-*`), artifact records (`MTA-*`), snapshots (`MTS-*`),
      reproducibility checks (`status == 'reproducible'`), JSON reports, and PDF reports.
    """
    # --- 1. VIDEO WORKFLOW ---
    mp4_bytes = _make_valid_mp4_bytes(isolated_storage)
    v_up = client.post(
        "/api/media/upload",
        files={"file": ("road_cam.mp4", io.BytesIO(mp4_bytes), "video/mp4")},
    )
    assert v_up.status_code == 201
    v_case_id = v_up.json()["case_id"]

    assert client.post(f"/api/analysis/video/{v_case_id}").status_code == 200
    v_repro = client.post(f"/api/forensics/{v_case_id}/reproducibility/check")
    assert v_repro.status_code == 200
    assert v_repro.json()["status"] == "reproducible"

    v_snap = client.get(f"/api/forensics/{v_case_id}/reproducibility").json()
    assert any(a["service_name"] == "video_analysis_service" for a in v_snap["analyses_executed"])
    assert len(v_snap["artifacts_produced"]) > 0

    assert client.post(f"/api/reports/{v_case_id}/generate").status_code == 200
    v_pdf = client.post(f"/api/reports/{v_case_id}/generate-pdf")
    assert v_pdf.status_code == 200
    assert v_pdf.content.startswith(b"%PDF-")

    # --- 2. AUDIO WORKFLOW ---
    wav_bytes = _make_valid_wav_bytes()
    a_up = client.post(
        "/api/media/upload",
        files={"file": ("voice_note.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert a_up.status_code == 201
    a_case_id = a_up.json()["case_id"]

    assert client.post(f"/api/analysis/audio/{a_case_id}").status_code == 200
    a_repro = client.post(f"/api/forensics/{a_case_id}/reproducibility/check")
    assert a_repro.status_code == 200
    assert a_repro.json()["status"] == "reproducible"

    a_snap = client.get(f"/api/forensics/{a_case_id}/reproducibility").json()
    assert any(a["service_name"] == "audio_analysis_service" for a in a_snap["analyses_executed"])
    art_types = {art["artifact_type"] for art in a_snap["artifacts_produced"]}
    assert "audio_waveform" in art_types
    assert "audio_spectrogram" in art_types

    assert client.post(f"/api/reports/{a_case_id}/generate").status_code == 200
    a_pdf = client.post(f"/api/reports/{a_case_id}/generate-pdf")
    assert a_pdf.status_code == 200
    assert a_pdf.content.startswith(b"%PDF-")

    # --- 3. DOCUMENT WORKFLOW ---
    docx_bytes = _make_valid_docx_bytes()
    d_up = client.post(
        "/api/media/upload",
        files={
            "file": (
                "statement.docx",
                io.BytesIO(docx_bytes),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert d_up.status_code == 201
    d_case_id = d_up.json()["case_id"]

    assert client.post(f"/api/analysis/document/{d_case_id}").status_code == 200
    d_repro = client.post(f"/api/forensics/{d_case_id}/reproducibility/check")
    assert d_repro.status_code == 200
    assert d_repro.json()["status"] == "reproducible"

    d_snap = client.get(f"/api/forensics/{d_case_id}/reproducibility").json()
    assert any(a["service_name"] == "document_analysis_service" for a in d_snap["analyses_executed"])

    assert client.post(f"/api/reports/{d_case_id}/generate").status_code == 200
    d_pdf = client.post(f"/api/reports/{d_case_id}/generate-pdf")
    assert d_pdf.status_code == 200
    assert d_pdf.content.startswith(b"%PDF-")

    # --- 4. IMAGE + HISTORY WORKFLOW ---
    jpg_bytes = _make_valid_jpeg_bytes()
    i_up = client.post(
        "/api/media/upload",
        files={"file": ("scene.jpg", io.BytesIO(jpg_bytes), "image/jpeg")},
    )
    assert i_up.status_code == 201
    i_case_id = i_up.json()["case_id"]

    assert client.post(f"/api/history/{i_case_id}/reconstruct").status_code == 200
    i_snap = client.get(f"/api/forensics/{i_case_id}/reproducibility").json()
    assert any(a["service_name"] == "history_service" for a in i_snap["analyses_executed"])
    for ev in i_snap["audit_events"]:
        assert ev["event_category"] == "system_audit_event"
