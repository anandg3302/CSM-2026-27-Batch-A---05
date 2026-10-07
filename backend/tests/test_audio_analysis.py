import io
import json
from pathlib import Path
import wave
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import soundfile as sf

from app.core.config import settings
from app.main import app

client = TestClient(app)


def create_synthetic_wav_bytes(
    *,
    sample_rate: int = 16000,
    channels: int = 2,
    duration_sec: float = 1.5,
    include_silence_gap: bool = True,
    include_discontinuity: bool = True,
    mtad_meta: dict | None = None,
) -> bytes:
    """Create a deterministic PCM 16-bit WAV file with optional silence gap, waveform discontinuity, and metadata."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # Base 440 Hz + 880 Hz harmonic tone
    signal = 0.35 * np.sin(2.0 * np.pi * 440.0 * t) + 0.15 * np.sin(2.0 * np.pi * 880.0 * t)

    if include_silence_gap:
        # Insert a 0.35s low-amplitude region from 0.40s to 0.75s
        s0 = int(0.40 * sample_rate)
        s1 = int(0.75 * sample_rate)
        signal[s0:s1] = 0.001 * np.sin(2.0 * np.pi * 120.0 * t[s0:s1])

    if include_discontinuity:
        # Insert an abrupt high-amplitude step jump + high-frequency burst around 1.05s
        d_idx = int(1.05 * sample_rate)
        if d_idx + 20 < n_samples:
            signal[d_idx] = -0.95
            signal[d_idx + 1] = 0.98
            signal[d_idx + 2 : d_idx + 60] = 0.85 * np.sin(2.0 * np.pi * 3800.0 * t[d_idx + 2 : d_idx + 60])

    pcm16 = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
    if channels > 1:
        stereo = np.column_stack([pcm16 for _ in range(channels)]).flatten()
    else:
        stereo = pcm16

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    raw = buf.getvalue()
    if mtad_meta:
        trailer = b"MTADJSON:" + json.dumps(mtad_meta).encode("utf-8") + b":ENDMTAD"
        raw = raw + trailer
    return raw


def create_synthetic_flac_bytes(sample_rate: int = 16000, duration_sec: float = 1.0) -> bytes:
    """Create a real lossless FLAC stream in memory using soundfile."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    signal = 0.3 * np.sin(2.0 * np.pi * 520.0 * t)
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def create_synthetic_container_audio_bytes(
    *,
    header_magic: bytes = b"ID3\x04\x00\x00\x00\x00\x00\x00",
    mtad_meta: dict | None = None,
) -> bytes:
    """Create a synthetic MP3/M4A container with embedded MTAD metadata and RIFF/WAVE stream."""
    wav_payload = create_synthetic_wav_bytes(
        sample_rate=16000,
        channels=1,
        duration_sec=1.2,
        include_silence_gap=True,
        include_discontinuity=True,
        mtad_meta=mtad_meta,
    )
    return header_magic + wav_payload


def test_audio_uploads_wav_mp3_flac_m4a_and_rejections():
    """
    Tests:
    - valid WAV upload
    - valid MP3 upload
    - valid FLAC upload
    - valid M4A upload
    - unsupported audio extension
    - image case sent to audio endpoint (400)
    - video case sent to audio endpoint (400)
    """
    # 1. Valid WAV
    wav_resp = client.post(
        "/api/media/upload",
        files={"file": ("interview_clip.wav", create_synthetic_wav_bytes(), "audio/wav")},
    )
    assert wav_resp.status_code == 201
    wav_case = wav_resp.json()
    assert wav_case["media_type"] == "audio"
    assert wav_case["extension"] == ".wav"

    # 2. Valid MP3
    mp3_bytes = create_synthetic_container_audio_bytes(
        mtad_meta={"codec": "MP3", "encoder": "LAME3.100", "bitrate": 96000}
    )
    mp3_resp = client.post(
        "/api/media/upload",
        files={"file": ("voice_note.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert mp3_resp.status_code == 201
    mp3_case = mp3_resp.json()
    assert mp3_case["media_type"] == "audio"
    assert mp3_case["extension"] == ".mp3"

    # 3. Valid FLAC
    flac_resp = client.post(
        "/api/media/upload",
        files={"file": ("master_take.flac", create_synthetic_flac_bytes(), "audio/flac")},
    )
    assert flac_resp.status_code == 201
    flac_case = flac_resp.json()
    assert flac_case["media_type"] == "audio"
    assert flac_case["extension"] == ".flac"

    # 4. Valid M4A
    m4a_bytes = create_synthetic_container_audio_bytes(
        header_magic=b"\x00\x00\x00\x14ftypM4A \x00\x00\x00\x00M4A ",
        mtad_meta={"codec": "AAC (Advanced Audio Coding)", "encoder": "Lavf60.16.100"},
    )
    m4a_resp = client.post(
        "/api/media/upload",
        files={"file": ("memo_recording.m4a", m4a_bytes, "audio/mp4")},
    )
    assert m4a_resp.status_code == 201
    m4a_case = m4a_resp.json()
    assert m4a_case["media_type"] == "audio"
    assert m4a_case["extension"] == ".m4a"

    # 5. Unsupported extension rejected
    bad_resp = client.post(
        "/api/media/upload",
        files={"file": ("legacy.mid", b"MThd0000", "audio/midi")},
    )
    assert bad_resp.status_code == 400

    # 6. Image case sent to audio endpoint -> 400
    img_buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_buf, format="PNG")
    img_resp = client.post(
        "/api/media/upload",
        files={"file": ("photo.png", img_buf.getvalue(), "image/png")},
    )
    img_case_id = img_resp.json()["case_id"]
    assert client.post(f"/api/analysis/audio/{img_case_id}").status_code == 400

    # 7. Video case sent to audio endpoint -> 400
    vid_resp = client.post(
        "/api/media/upload",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x14ftypisom", "video/mp4")},
    )
    vid_case_id = vid_resp.json()["case_id"]
    assert client.post(f"/api/analysis/audio/{vid_case_id}").status_code == 400


def test_audio_analysis_errors_missing_not_found_sha256_and_corrupt():
    """
    Tests:
    - nonexistent case (404)
    - missing audio on disk (404)
    - SHA-256 mismatch before analysis (409)
    - corrupt audio handled safely (422)
    """
    # Nonexistent case -> 404
    assert client.post("/api/analysis/audio/MT-20990101-DEADBEEF").status_code == 404

    # Corrupt audio -> 422
    corrupt_up = client.post(
        "/api/media/upload",
        files={"file": ("corrupt.wav", b"NOT_A_VALID_AUDIO_STREAM_BYTES", "audio/wav")},
    )
    corrupt_id = corrupt_up.json()["case_id"]
    assert client.post(f"/api/analysis/audio/{corrupt_id}").status_code == 422

    # SHA-256 mismatch & missing file on disk
    valid_up = client.post(
        "/api/media/upload",
        files={"file": ("tamper_check.wav", create_synthetic_wav_bytes(), "audio/wav")},
    )
    case_data = valid_up.json()
    case_id = case_data["case_id"]
    stored_path = Path(settings.storage_root) / "uploads" / case_data["stored_filename"]
    original_bytes = stored_path.read_bytes()

    try:
        stored_path.write_bytes(original_bytes + b"CORRUPTED_BYTE")
        assert client.post(f"/api/analysis/audio/{case_id}").status_code == 409
    finally:
        stored_path.write_bytes(original_bytes)

    stored_path.unlink()
    try:
        assert client.post(f"/api/analysis/audio/{case_id}").status_code == 404
    finally:
        stored_path.write_bytes(original_bytes)


def test_audio_analysis_full_pipeline_waveform_silence_spectral_discontinuity_privacy_and_immutability():
    """
    Tests:
    - metadata extraction, duration, sample_rate, channels, bit_depth
    - waveform analysis & waveform.png generation
    - silence/gap analysis ('low_amplitude_region')
    - spectral analysis & spectrogram.png generation
    - discontinuity analysis ('possible_audio_discontinuity', 'potentially_suspicious')
    - recompression analysis ('possible_recompression', limitation message)
    - AI detector returns 'not_available' & no fake probabilities
    - GPS privacy ('gps_present': True, 'gps_notice': 'GPS metadata present', no raw coordinates)
    - original SHA-256 & file bytes unchanged
    - derived artifacts stored under storage/results/<case_id>/audio/ and downloadable via API
    """
    wav_bytes = create_synthetic_wav_bytes(
        sample_rate=16000,
        channels=2,
        duration_sec=1.5,
        include_silence_gap=True,
        include_discontinuity=True,
        mtad_meta={
            "encoder": "Lavf60.16.100",
            "software": "Audacity 3.4",
            "creation_time": "2026-09-28T14:00:00Z",
            "title": "Field Recording 01",
            "artist": "Investigator A",
            "gps_present": True,
        },
    )
    up_resp = client.post(
        "/api/media/upload",
        files={"file": ("evidence_audio.wav", wav_bytes, "audio/wav")},
    )
    assert up_resp.status_code == 201
    case_data = up_resp.json()
    case_id = case_data["case_id"]
    stored_path = Path(settings.storage_root) / "uploads" / case_data["stored_filename"]
    bytes_before = stored_path.read_bytes()

    res = client.post(f"/api/analysis/audio/{case_id}")
    assert res.status_code == 200
    body = res.json()
    aa = body["audio_analysis"]

    # Preservation & SHA-256 verification
    assert body["sha256"] == case_data["sha256"]
    assert stored_path.read_bytes() == bytes_before

    # Technical properties & metadata
    assert aa["media_type"] == "audio"
    assert aa["format"] == "WAV"
    assert aa["sample_rate"] == 16000
    assert aa["channels"] == 2
    assert aa["bit_depth"] == 16
    assert aa["duration_seconds"] == 1.5
    assert aa["metadata"]["encoder"] == "Lavf60.16.100"
    assert aa["metadata"]["gps_present"] is True
    assert aa["metadata"]["gps_notice"] == "GPS metadata present"

    # Ensure no raw GPS coordinates are leaked
    serialized = json.dumps(aa)
    assert "latitude" not in serialized.lower()
    assert "longitude" not in serialized.lower()

    # Waveform analysis & artifact
    wf = aa["waveform_analysis"]
    assert wf["peak_amplitude"] > 0.0
    assert wf["rms_level"] > 0.0
    assert wf["dynamic_range_db"] >= 0.0
    assert wf["artifact"].startswith(f"storage/results/{case_id}/audio/waveform.png")
    wf_resp = client.get(f"/api/analysis/audio/{case_id}/artifacts/waveform.png")
    assert wf_resp.status_code == 200
    assert wf_resp.headers["content-type"] == "image/png"

    # Silence / Low-Amplitude Region Analysis
    sil = aa["silence_analysis"]
    assert sil["region_count"] >= 1
    for reg in sil["regions"]:
        assert reg["type"] == "low_amplitude_region"
        assert reg["start_seconds"] >= 0.0
        assert reg["end_seconds"] > reg["start_seconds"]
        assert reg["duration_seconds"] > 0.0

    # Spectral analysis & spectrogram.png
    spec = aa["spectral_analysis"]
    assert spec["spectral_centroid_mean_hz"] > 0.0
    assert spec["spectral_rolloff_mean_hz"] > 0.0
    assert spec["artifact"].startswith(f"storage/results/{case_id}/audio/spectrogram.png")
    spec_resp = client.get(f"/api/analysis/audio/{case_id}/artifacts/spectrogram.png")
    assert spec_resp.status_code == 200
    assert spec_resp.headers["content-type"] == "image/png"

    # Discontinuity analysis
    disc = aa["discontinuity_analysis"]
    assert disc["status"] == "potential_signal"
    assert disc["candidate_count"] >= 1
    for cand in disc["candidates"]:
        assert cand["finding"] == "possible_audio_discontinuity"
        assert cand["status"] == "potentially_suspicious"
        assert cand["start_seconds"] >= 0.0
        assert cand["end_seconds"] >= cand["start_seconds"]

    # Recompression analysis
    recomp = aa["recompression_analysis"]
    assert recomp["status"] == "possible_recompression"
    assert recomp["limitation"] == "Possible recompression is not equivalent to manipulation."

    # AI detection & assessment
    assert aa["ai_generation"]["status"] == "not_available"
    assert aa["assessment"]["status"] in {
        "no_strong_indicators",
        "potential_audio_anomaly",
        "possible_discontinuity",
        "possible_recompression",
        "insufficient_evidence",
    }
    assert "%" not in aa["assessment"]["description"]

    # Full limitations list
    assert len(aa["limitations"]) >= 9
