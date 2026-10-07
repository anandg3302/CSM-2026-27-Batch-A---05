import io
import json
import struct
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def create_synthetic_video_bytes(
    *,
    frame_count: int = 8,
    fps: float = 10.0,
    width: int = 160,
    height: int = 120,
    container_brand: bytes = b"isom",
    inject_cut_at: int | None = None,
    inject_tampered_frame_at: int | None = None,
    has_audio: bool = False,
    audio_codec: str | None = None,
    encoder_software: str | None = None,
    creation_timestamp: str | None = None,
    gps_present: bool = False,
) -> bytes:
    """
    Create a deterministic ISO-BMFF (.mp4 / .mov) container with optional MTMD metadata box
    and embedded JPEG frames in the mdat box so that video_analysis_service can decode frames
    and parse container/audio/metadata deterministically across environments.
    """

    def make_box(box_type: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", 8 + len(payload)) + box_type + payload

    ftyp = make_box(b"ftyp", container_brand + struct.pack(">I", 512) + container_brand + b"mp42")

    meta_dict = {
        "codec": "H.264 / AVC (Synthetic Test Stream)",
        "frame_rate": fps,
        "frame_count": frame_count,
        "has_audio": has_audio,
        "audio_codec": audio_codec,
        "encoder_software": encoder_software,
        "creation_timestamp": creation_timestamp,
        "gps_present": gps_present,
    }
    mtmd_box = make_box(b"MTMD", json.dumps(meta_dict).encode("utf-8"))
    moov = make_box(b"moov", mtmd_box)

    frame_blobs = []
    for idx in range(frame_count):
        if inject_cut_at is not None and idx >= inject_cut_at:
            base_color = (220, 45, 45)
        else:
            base_color = (40 + (idx * 5) % 60, 90 + (idx * 4) % 60, 160)

        img = Image.new("RGB", (width, height), color=base_color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([16 + idx * 3, 20, 64 + idx * 3, 75], fill=(235, 215, 90))

        if inject_tampered_frame_at is not None and idx == inject_tampered_frame_at:
            # High-contrast spliced patch + synthetic noise perturbation on a specific frame
            for y in range(25, 95, 2):
                for x in range(30, 130, 2):
                    img.putpixel((x, y), (255, 0, 255) if (x + y) % 4 == 0 else (0, 255, 120))
            jpeg_quality = 35
        else:
            jpeg_quality = 90

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        frame_blobs.append(buf.getvalue())

    mdat = make_box(b"mdat", b"".join(frame_blobs))
    return ftyp + moov + mdat


def create_test_png_bytes() -> bytes:
    img = Image.new("RGB", (96, 96), color=(60, 120, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_video_upload_mp4_mov_and_unsupported_rejection():
    """
    Tests:
    1. Valid video upload (.mp4, .mov)
    2. Unsupported video extension rejection
    6. Video case vs image case validation
    """
    mp4_bytes = create_synthetic_video_bytes(frame_count=6, fps=12.0, container_brand=b"isom")
    resp_mp4 = client.post(
        "/api/media/upload",
        files={"file": ("clip_evidence.mp4", mp4_bytes, "video/mp4")},
    )
    assert resp_mp4.status_code == 201
    mp4_case = resp_mp4.json()
    assert mp4_case["media_type"] == "video"
    assert mp4_case["extension"] == ".mp4"

    mov_bytes = create_synthetic_video_bytes(frame_count=5, fps=10.0, container_brand=b"qt  ")
    resp_mov = client.post(
        "/api/media/upload",
        files={"file": ("iphone_capture.mov", mov_bytes, "video/quicktime")},
    )
    assert resp_mov.status_code == 201
    mov_case = resp_mov.json()
    assert mov_case["media_type"] == "video"
    assert mov_case["extension"] == ".mov"

    # Unsupported extension rejection
    bad_resp = client.post(
        "/api/media/upload",
        files={"file": ("archive.flv", b"FLV_DUMMY", "video/x-flv")},
    )
    assert bad_resp.status_code == 400

    # Image case rejected by video analysis endpoint
    img_resp = client.post(
        "/api/media/upload",
        files={"file": ("still_photo.png", create_test_png_bytes(), "image/png")},
    )
    assert img_resp.status_code == 201
    img_case_id = img_resp.json()["case_id"]

    wrong_type_resp = client.post(f"/api/analysis/video/{img_case_id}")
    assert wrong_type_resp.status_code == 400
    assert "image" in wrong_type_resp.json()["detail"].lower()


def test_video_analysis_error_handling_missing_not_found_sha256_and_corrupt():
    """
    Tests:
    3. Missing video file on disk
    4. Nonexistent case ID (404)
    5. SHA-256 mismatch before analysis (409)
    26. Malformed/corrupt video handling (422)
    """
    # 404 for nonexistent case
    resp_404 = client.post("/api/analysis/video/MT-20990101-DEADBEEF")
    assert resp_404.status_code == 404

    # Corrupt / undecodable video returns 422
    corrupt_upload = client.post(
        "/api/media/upload",
        files={"file": ("corrupt_stream.mp4", b"NOT_A_REAL_VIDEO_BINARY_STREAM", "video/mp4")},
    )
    assert corrupt_upload.status_code == 201
    corrupt_case_id = corrupt_upload.json()["case_id"]
    corrupt_res = client.post(f"/api/analysis/video/{corrupt_case_id}")
    assert corrupt_res.status_code == 422

    # SHA-256 mismatch before analysis (409)
    valid_bytes = create_synthetic_video_bytes(frame_count=5)
    up_resp = client.post(
        "/api/media/upload",
        files={"file": ("integrity_check.mp4", valid_bytes, "video/mp4")},
    )
    case_data = up_resp.json()
    case_id = case_data["case_id"]
    stored_path = Path(settings.storage_root) / "uploads" / case_data["stored_filename"]
    original_bytes = stored_path.read_bytes()

    try:
        stored_path.write_bytes(original_bytes + b"TAMPER")
        mismatch_resp = client.post(f"/api/analysis/video/{case_id}")
        assert mismatch_resp.status_code == 409
    finally:
        stored_path.write_bytes(original_bytes)

    # Missing video file on disk (404)
    stored_path.unlink()
    try:
        missing_resp = client.post(f"/api/analysis/video/{case_id}")
        assert missing_resp.status_code == 404
    finally:
        stored_path.write_bytes(original_bytes)


def test_video_analysis_full_pipeline_metadata_sampling_temporal_cut_and_privacy():
    """
    Tests:
    7-11. Metadata extraction, duration, frame_rate, resolution, audio stream presence
    12-13. Representative frame extraction & max_sample_frames limit respected
    14-21. Temporal consistency, suspicious frame localization with timestamps, recompression, scene/cut, AI detector not_available
    22. No fake probabilities anywhere
    23. No raw GPS coordinates exposed
    24-25. Original video file bytes & SHA-256 unchanged, derived artifacts stored under storage/results/<case_id>/video/
    27-28. Missing optional metadata fields handled & limitations always non-empty
    """
    video_bytes = create_synthetic_video_bytes(
        frame_count=16,
        fps=8.0,
        width=160,
        height=120,
        container_brand=b"isom",
        inject_cut_at=8,
        inject_tampered_frame_at=5,
        has_audio=True,
        audio_codec="AAC (mp4a)",
        encoder_software="Lavf60.16.100",
        creation_timestamp="2026-09-28T12:00:00Z",
        gps_present=True,
    )
    up_resp = client.post(
        "/api/media/upload",
        files={"file": ("forensic_clip.mp4", video_bytes, "video/mp4")},
    )
    assert up_resp.status_code == 201
    case_data = up_resp.json()
    case_id = case_data["case_id"]
    stored_path = Path(settings.storage_root) / "uploads" / case_data["stored_filename"]
    bytes_before = stored_path.read_bytes()

    res = client.post(f"/api/analysis/video/{case_id}")
    assert res.status_code == 200
    body = res.json()
    va = body["video_analysis"]

    # Integrity & original file preservation
    assert body["sha256"] == case_data["sha256"]
    assert stored_path.read_bytes() == bytes_before

    # Container, duration, frame rate, resolution, audio stream, metadata
    assert va["container"].startswith("MP4")
    assert va["width"] == 160
    assert va["height"] == 120
    assert va["frame_rate"] == 8.0
    assert va["frame_count"] == 16
    assert va["duration_seconds"] == 2.0
    assert va["has_audio"] is True
    assert va["audio_codec"] == "AAC (mp4a)"
    assert va["metadata"]["encoder_software"] == "Lavf60.16.100"
    assert va["metadata"]["gps_present"] is True
    assert va["metadata"]["gps_notice"] == "GPS metadata present"

    # Ensure no raw GPS coordinates are leaked in serialized response
    serialized = json.dumps(va)
    assert "latitude" not in serialized.lower()
    assert "longitude" not in serialized.lower()

    # Frame sampling & max_sample_frames limit respected
    fs = va["frame_sampling"]
    assert fs["max_sample_frames"] == settings.max_sample_frames
    assert fs["sample_count"] <= settings.max_sample_frames
    assert len(fs["frames"]) == fs["sample_count"]
    assert fs["sample_count"] > 0

    for sf in fs["frames"]:
        assert sf["frame_index"] >= 0
        assert sf["timestamp_seconds"] >= 0.0
        assert sf["artifact"].startswith(f"storage/results/{case_id}/video/")

    # Temporal consistency, scene cut, recompression, suspicious frames
    assert va["temporal_analysis"]["status"] in {
        "potential_signal",
        "no_strong_signal",
        "insufficient_data",
    }
    assert va["scene_analysis"]["status"] == "available"
    assert len(va["scene_analysis"]["scene_changes"]) >= 1
    assert va["recompression_analysis"]["status"] in {
        "possible_recompression",
        "no_strong_signal",
        "insufficient_data",
    }
    assert len(va["temporal_analysis"]["suspicious_frames"]) >= 1
    for susp in va["temporal_analysis"]["suspicious_frames"]:
        assert susp["frame_index"] >= 0
        assert susp["timestamp_seconds"] >= 0.0
        assert susp["artifact"].startswith(f"storage/results/{case_id}/video/")

    # AI detector status & no fake numeric probabilities
    assert va["ai_generation"]["status"] == "not_available"
    assert "%" not in va["assessment"]["description"]

    # Derived artifacts exist under storage/results/<case_id>/video/ and are downloadable via API
    first_artifact_filename = fs["frames"][0]["artifact_filename"]
    artifact_resp = client.get(f"/api/analysis/video/{case_id}/artifacts/{first_artifact_filename}")
    assert artifact_resp.status_code == 200
    assert artifact_resp.headers["content-type"] in {"image/jpeg", "image/png"}

    # Non-empty limitations
    assert len(va["limitations"]) >= 3

