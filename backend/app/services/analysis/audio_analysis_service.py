from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import wave

from fastapi import HTTPException, status
import mutagen
import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf

from app.core.config import PROJECT_ROOT, settings
from app.schemas.audio_analysis import (
    AllowedAudioAssessment,
    AudioAiDetectionResult,
    AudioAnalysisResult,
    AudioDiscontinuityAnalysis,
    AudioDiscontinuityCandidate,
    AudioForensicAssessment,
    AudioMetadataInfo,
    AudioRecompressionAnalysis,
    AudioSilenceAnalysis,
    AudioSilenceRegion,
    AudioSpectralAnalysis,
    AudioWaveformAnalysis,
    SpectralTransitionCandidate,
)
from app.schemas.media import MediaCaseMetadata
from app.services import media_service

MAX_INTERNAL_SAMPLES = 360_000  # Strictly bounded sample array size for analysis
MAX_SILENCE_REGIONS = 40
MAX_DISCONTINUITY_CANDIDATES = 12
MAX_DISCONTINUITY_ARTIFACTS = 3
ALLOWED_ARTIFACT_NAME_PATTERN = re.compile(
    r"^(waveform\.png|spectrogram\.png|discontinuity_\d+\.png)$"
)

LOSSY_EXTENSIONS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}

EXTENSION_FORMAT_MAP = {
    ".wav": ("WAV", "RIFF/WAV", "PCM"),
    ".mp3": ("MP3", "MPEG-TS/ID3", "MP3 (MPEG-1/2 Audio Layer III)"),
    ".flac": ("FLAC", "FLAC", "FLAC (Free Lossless Audio Codec)"),
    ".m4a": ("M4A", "MP4/ISO-BMFF", "AAC / ALAC"),
    ".aac": ("AAC", "ADTS/AAC", "AAC (Advanced Audio Coding)"),
    ".ogg": ("OGG", "Ogg", "Vorbis / Opus"),
    ".opus": ("OPUS", "Ogg Opus", "Opus"),
    ".wma": ("WMA", "ASF/WMA", "Windows Media Audio"),
}

STANDARD_AUDIO_LIMITATIONS = [
    "Metadata does not establish authenticity and can be modified or stripped by standard audio software.",
    "Normal audio encoding and lossy compression can create compression artifacts without malicious editing.",
    "Transcoding or re-encoding does not prove manipulation and commonly occurs during distribution or upload.",
    "Silence or low-amplitude intervals may be natural pauses and do not prove audio deletion.",
    "Waveform discontinuities do not prove splicing and can arise from recording dropouts, microphone switches, or packet loss.",
    "Spectral changes may result from legitimate acoustic or vocal events rather than editing.",
    "Candidate regions identified by automated heuristics require human forensic review.",
    "No exact chronological editing history can be reconstructed from these signals alone.",
    "AI-generated audio detection is unavailable unless a trained detection model is configured.",
]


def _compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _detect_gps_in_text(text: str) -> bool:
    lowered = text.lower()
    if any(k in lowered for k in ("gps", "location", "com.apple.quicktime.location", "xyz", "latitude", "longitude")):
        return True
    if re.search(r"[+-]\d{2}\.\d{3,}[+-]\d{2,3}\.\d{3,}", text):
        return True
    return False


def _sanitize_tag_value(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        val = ", ".join(str(x) for x in val if x is not None)
    text = str(val).strip()
    if not text:
        return None
    # Redact any embedded coordinate strings
    if _detect_gps_in_text(text):
        return "[Redacted Location Metadata]"
    return text[:240]


def _extract_embedded_mtad_payload(raw_bytes: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
    """
    Support deterministic synthetic test fixtures for container formats (such as MP3/M4A/AAC/WMA)
    that embed an optional MTAD JSON metadata block and an optional embedded WAV payload.
    """
    marker = b"MTADJSON:"
    idx = raw_bytes.find(marker)
    meta_dict: Optional[Dict[str, Any]] = None
    if idx != -1:
        end_idx = raw_bytes.find(b":ENDMTAD", idx)
        if end_idx != -1:
            try:
                meta_dict = json.loads(raw_bytes[idx + len(marker) : end_idx].decode("utf-8"))
            except Exception:
                meta_dict = None

    wav_idx = raw_bytes.find(b"RIFF")
    embedded_wav: Optional[bytes] = None
    if wav_idx != -1 and raw_bytes[wav_idx + 8 : wav_idx + 12] == b"WAVE":
        embedded_wav = raw_bytes[wav_idx:]

    return meta_dict, embedded_wav


def _decode_audio_bounded(
    file_path: Path, extension: str
) -> Tuple[np.ndarray, int, int, Optional[int], float, Dict[str, Any]]:
    """
    Decode audio non-destructively with bounded memory usage.
    Returns:
      (mono_samples_float32, sample_rate, channels, bit_depth, total_duration_seconds, extra_meta)
    """
    raw_bytes = file_path.read_bytes()
    if len(raw_bytes) < 12:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio file is too small or malformed to decode.",
        )

    mtad_meta, embedded_wav = _extract_embedded_mtad_payload(raw_bytes)
    extra_meta: Dict[str, Any] = mtad_meta or {}

    max_seconds = float(settings.max_audio_analysis_seconds)

    # 1. Try soundfile (supports WAV, FLAC, OGG, OPUS, MP3)
    try:
        with sf.SoundFile(str(file_path), "r") as snd:
            sr = int(snd.samplerate)
            ch = int(snd.channels)
            total_frames = int(snd.frames)
            if sr <= 0 or ch <= 0 or total_frames <= 0:
                raise ValueError("Invalid audio stream parameters.")
            duration_sec = round(total_frames / float(sr), 4)
            frames_to_read = min(total_frames, int(max_seconds * sr))
            data = snd.read(frames=frames_to_read, dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            if len(mono) > MAX_INTERNAL_SAMPLES:
                step = max(1, len(mono) // MAX_INTERNAL_SAMPLES)
                mono = mono[::step]

            bit_depth: Optional[int] = None
            subtype = (snd.subtype or "").upper()
            if "PCM_16" in subtype:
                bit_depth = 16
            elif "PCM_24" in subtype:
                bit_depth = 24
            elif "PCM_32" in subtype or "FLOAT" in subtype:
                bit_depth = 32
            elif "PCM_U8" in subtype or "PCM_S8" in subtype:
                bit_depth = 8

            extra_meta.setdefault("codec", snd. subtype_info or subtype or None)
            extra_meta.setdefault("format_info", snd.format_info)
            return mono.astype(np.float32), sr, ch, bit_depth, duration_sec, extra_meta
    except Exception:
        pass

    # 2. Try standard wave module or embedded RIFF/WAVE stream
    wav_source = embedded_wav if embedded_wav is not None else (raw_bytes if raw_bytes[:4] == b"RIFF" else None)
    if wav_source is not None:
        try:
            with wave.open(io.BytesIO(wav_source), "rb") as wf:
                sr = int(wf.getframerate())
                ch = int(wf.getnchannels())
                sampwidth = int(wf.getsampwidth())
                total_frames = int(wf.getnframes())
                if sr <= 0 or ch <= 0 or total_frames <= 0:
                    raise ValueError("Empty WAV stream")
                duration_sec = round(total_frames / float(sr), 4)
                frames_to_read = min(total_frames, int(max_seconds * sr))
                raw_pcm = wf.readframes(frames_to_read)
                if sampwidth == 1:
                    arr = (np.frombuffer(raw_pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                elif sampwidth == 2:
                    arr = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    arr = np.frombuffer(raw_pcm, dtype="<i4").astype(np.float32) / 2147483648.0
                else:
                    arr = np.frombuffer(raw_pcm, dtype=np.uint8).astype(np.float32) / 255.0

                if ch > 1 and len(arr) >= ch:
                    usable = (len(arr) // ch) * ch
                    mono = np.mean(arr[:usable].reshape(-1, ch), axis=1)
                else:
                    mono = arr

                if len(mono) > MAX_INTERNAL_SAMPLES:
                    step = max(1, len(mono) // MAX_INTERNAL_SAMPLES)
                    mono = mono[::step]

                return (
                    mono.astype(np.float32),
                    sr,
                    ch,
                    sampwidth * 8,
                    duration_sec,
                    extra_meta,
                )
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Unable to decode audio stream from '{file_path.name}'. The file may be corrupt or malformed.",
    )


def _extract_audio_metadata(
    file_path: Path,
    extension: str,
    file_size: int,
    sample_rate: int,
    channels: int,
    bit_depth: Optional[int],
    duration_seconds: float,
    extra_meta: Dict[str, Any],
) -> AudioMetadataInfo:
    default_fmt, default_container, default_codec = EXTENSION_FORMAT_MAP.get(
        extension, (extension.lstrip(".").upper(), extension.lstrip(".").upper(), None)
    )

    codec = extra_meta.get("codec") or default_codec
    bitrate: Optional[int] = extra_meta.get("bitrate")
    if bitrate is None and duration_seconds > 0:
        bitrate = int(round((file_size * 8) / duration_seconds))

    encoder = extra_meta.get("encoder")
    software = extra_meta.get("software")
    creation_time = extra_meta.get("creation_time")
    title = extra_meta.get("title")
    artist = extra_meta.get("artist")
    album = extra_meta.get("album")
    comment = extra_meta.get("comment")
    gps_present = bool(extra_meta.get("gps_present", False))

    # Inspect with mutagen if available
    try:
        mfile = mutagen.File(str(file_path))
        if mfile is not None:
            if hasattr(mfile, "info") and mfile.info is not None:
                if getattr(mfile.info, "bitrate", None):
                    bitrate = int(mfile.info.bitrate)
                if getattr(mfile.info, "bits_per_sample", None) and bit_depth is None:
                    bit_depth = int(mfile.info.bits_per_sample)
            if mfile.tags:
                for key, val in mfile.tags.items():
                    k_low = str(key).lower()
                    raw_str = str(val)
                    if _detect_gps_in_text(k_low) or _detect_gps_in_text(raw_str):
                        gps_present = True
                        continue
                    if k_low in ("tit2", "title", "\xa9nam") and not title:
                        title = _sanitize_tag_value(val)
                    elif k_low in ("tpe1", "artist", "\xa9art") and not artist:
                        artist = _sanitize_tag_value(val)
                    elif k_low in ("talb", "album", "\xa9alb") and not album:
                        album = _sanitize_tag_value(val)
                    elif k_low in ("tsse", "encoder", "\xa9too", "encodedby") and not encoder:
                        encoder = _sanitize_tag_value(val)
                    elif k_low in ("software", "tool") and not software:
                        software = _sanitize_tag_value(val)
                    elif k_low in ("tdrc", "date", "creation_time", "\xa9day") and not creation_time:
                        creation_time = _sanitize_tag_value(val)
                    elif k_low.startswith("comm") or k_low in ("comment", "\xa9cmt"):
                        if not comment:
                            comment = _sanitize_tag_value(val)
    except Exception:
        pass

    # Also check raw ASCII printable tags in header/trailer for GPS or encoder markers
    raw_head_tail = file_path.read_bytes()[:4096] + file_path.read_bytes()[-4096:]
    ascii_text = raw_head_tail.decode("latin-1", errors="ignore")
    if _detect_gps_in_text(ascii_text):
        gps_present = True

    channel_layout = "mono" if channels == 1 else ("stereo" if channels == 2 else f"{channels}_channels")

    return AudioMetadataInfo(
        format=default_fmt,
        container=default_container,
        codec=codec,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        channel_layout=channel_layout,
        bit_depth=bit_depth,
        bitrate=bitrate,
        encoder=encoder,
        software=software,
        creation_time=creation_time,
        title=title,
        artist=artist,
        album=album,
        comment=comment,
        gps_present=gps_present,
        gps_notice="GPS metadata present" if gps_present else None,
    )


def _render_waveform_artifact(
    mono: np.ndarray,
    duration_seconds: float,
    silence_regions: List[AudioSilenceRegion],
    discontinuities: List[AudioDiscontinuityCandidate],
    output_path: Path,
) -> None:
    width, height = 840, 240
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    mid_y = height // 2
    # Draw subtle grid lines
    for y_frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        y_pos = int(height * y_frac)
        color = (51, 65, 85) if y_frac == 0.5 else (30, 41, 59)
        draw.line([(0, y_pos), (width, y_pos)], fill=color, width=1)

    # Highlight low-amplitude / silence regions
    if duration_seconds > 0:
        for reg in silence_regions[:20]:
            x0 = int((reg.start_seconds / duration_seconds) * width)
            x1 = max(x0 + 2, int((reg.end_seconds / duration_seconds) * width))
            draw.rectangle([x0, 12, x1, height - 12], fill=(30, 58, 95))

    # Plot min/max envelope per horizontal pixel column
    n = len(mono)
    if n > 0:
        cols = width - 20
        samples_per_col = max(1, n // cols)
        for col in range(cols):
            start_i = col * samples_per_col
            end_i = min(n, start_i + samples_per_col)
            if start_i >= n:
                break
            segment = mono[start_i:end_i]
            min_v = float(np.min(segment))
            max_v = float(np.max(segment))
            y_top = int(mid_y - max(-1.0, min(1.0, max_v)) * (mid_y - 20))
            y_bot = int(mid_y - max(-1.0, min(1.0, min_v)) * (mid_y - 20))
            draw.line([(10 + col, y_top), (10 + col, y_bot)], fill=(56, 189, 248), width=1)

    # Mark candidate discontinuities in amber
    if duration_seconds > 0:
        for disc in discontinuities[:10]:
            x_d = 10 + int((disc.start_seconds / duration_seconds) * (width - 20))
            draw.line([(x_d, 8), (x_d, height - 8)], fill=(245, 158, 11), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def _render_spectrogram_artifact(
    spec_mag: np.ndarray,
    output_path: Path,
) -> None:
    """Render a bounded 2D STFT log-magnitude spectrogram artifact as PNG."""
    width, height = 840, 240
    if spec_mag.size == 0:
        img = Image.new("RGB", (width, height), color=(15, 23, 42))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        return

    # Convert magnitude to log dB scale normalized 0..1
    log_spec = np.log10(np.maximum(spec_mag, 1e-7))
    min_v = float(np.min(log_spec))
    max_v = float(np.max(log_spec))
    norm = (log_spec - min_v) / max(1e-6, (max_v - min_v))
    # Flip vertically so low frequencies are at the bottom
    norm = np.flipud(norm.T)

    # Map normalized values [0, 1] to a dark-blue -> cyan -> amber forensic colormap
    r = np.clip(norm * 255.0 * 1.1, 15, 255).astype(np.uint8)
    g = np.clip((norm ** 0.8) * 210.0, 23, 245).astype(np.uint8)
    b = np.clip((1.0 - norm * 0.5) * 220.0, 42, 240).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    pil_raw = Image.fromarray(rgb)
    pil_resized = pil_raw.resize((width, height), resample=Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_resized.save(output_path, format="PNG")


def _render_discontinuity_artifact(
    mono: np.ndarray,
    effective_sr: float,
    center_seconds: float,
    output_path: Path,
) -> None:
    """Render a zoomed waveform window around a candidate discontinuity."""
    width, height = 480, 160
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    mid_y = height // 2
    draw.line([(0, mid_y), (width, mid_y)], fill=(51, 65, 85), width=1)

    center_idx = int(center_seconds * effective_sr)
    half_win = max(64, int(0.08 * effective_sr))
    start_idx = max(0, center_idx - half_win)
    end_idx = min(len(mono), center_idx + half_win)
    seg = mono[start_idx:end_idx]

    if len(seg) > 1:
        pts = []
        for i, val in enumerate(seg):
            x = int((i / (len(seg) - 1)) * (width - 16)) + 8
            y = int(mid_y - max(-1.0, min(1.0, float(val))) * (mid_y - 16))
            pts.append((x, y))
        draw.line(pts, fill=(251, 191, 36), width=2)

    draw.line([(width // 2, 10), (width // 2, height - 10)], fill=(239, 68, 68), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def analyze_audio_for_case(case_id: str) -> MediaCaseMetadata:
    """
    Perform non-destructive forensic analysis on an uploaded audio case (Step 11).
    Verifies SHA-256 before and after analysis and stores derived visual artifacts
    under storage/results/<case_id>/audio/.
    """
    case = media_service.get_media_case(case_id)
    if case.media_type != "audio":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' is of media_type '{case.media_type}'. Only audio cases can be analyzed by the audio analysis pipeline.",
        )

    audio_path = settings.uploads_dir / case.stored_filename
    if not audio_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored audio file for case '{case_id}' is missing on disk.",
        )

    # 1. Pre-analysis SHA-256 integrity verification
    pre_sha256 = _compute_file_sha256(audio_path)
    if pre_sha256 != case.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="SHA-256 integrity mismatch detected on source audio file before analysis.",
        )

    # 2. Decode audio stream with bounded resource limits
    mono, sample_rate, channels, bit_depth, duration_seconds, extra_meta = _decode_audio_bounded(
        audio_path, case.extension
    )
    analyzed_duration = min(duration_seconds, float(settings.max_audio_analysis_seconds))
    effective_sr = len(mono) / max(1e-6, analyzed_duration)

    # 3. Extract metadata (with GPS privacy redaction)
    metadata_info = _extract_audio_metadata(
        file_path=audio_path,
        extension=case.extension,
        file_size=case.file_size,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        duration_seconds=duration_seconds,
        extra_meta=extra_meta,
    )

    artifact_dir = settings.results_dir / case_id / "audio"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 4. Silence / Low-Amplitude Region Analysis
    silence_thresh = float(settings.silence_threshold_amplitude)
    min_silence_dur = float(settings.min_silence_duration_seconds)
    win_samples = max(16, int(0.05 * effective_sr))  # 50ms windows
    num_wins = len(mono) // win_samples
    silence_regions: List[AudioSilenceRegion] = []

    in_silence = False
    sil_start_win = 0
    sil_amps: List[float] = []

    for w in range(num_wins):
        chunk = mono[w * win_samples : (w + 1) * win_samples]
        w_rms = float(np.sqrt(np.mean(chunk ** 2)))
        if w_rms < silence_thresh:
            if not in_silence:
                in_silence = True
                sil_start_win = w
                sil_amps = [w_rms]
            else:
                sil_amps.append(w_rms)
        else:
            if in_silence:
                start_s = round((sil_start_win * win_samples) / effective_sr, 3)
                end_s = round((w * win_samples) / effective_sr, 3)
                dur_s = round(end_s - start_s, 3)
                if dur_s >= min_silence_dur and len(silence_regions) < MAX_SILENCE_REGIONS:
                    silence_regions.append(
                        AudioSilenceRegion(
                            start_seconds=start_s,
                            end_seconds=end_s,
                            duration_seconds=dur_s,
                            type="low_amplitude_region",
                            mean_amplitude=round(float(np.mean(sil_amps)), 5),
                        )
                    )
                in_silence = False

    if in_silence:
        start_s = round((sil_start_win * win_samples) / effective_sr, 3)
        end_s = round((num_wins * win_samples) / effective_sr, 3)
        dur_s = round(end_s - start_s, 3)
        if dur_s >= min_silence_dur and len(silence_regions) < MAX_SILENCE_REGIONS:
            silence_regions.append(
                AudioSilenceRegion(
                    start_seconds=start_s,
                    end_seconds=end_s,
                    duration_seconds=dur_s,
                    type="low_amplitude_region",
                    mean_amplitude=round(float(np.mean(sil_amps)), 5),
                )
            )

    silence_analysis = AudioSilenceAnalysis(
        status="regions_detected" if silence_regions else "no_regions_detected",
        threshold_amplitude=silence_thresh,
        min_silence_duration_seconds=min_silence_dur,
        region_count=len(silence_regions),
        regions=silence_regions,
    )

    # 5. Spectral Analysis (STFT) & Spectrogram Artifact
    n_fft = 256
    max_spec_frames = 320
    hop = max(64, (len(mono) - n_fft) // max(1, max_spec_frames))
    window = np.hanning(n_fft).astype(np.float32)

    frames_list = []
    frame_times = []
    for start_i in range(0, max(1, len(mono) - n_fft), hop):
        seg = mono[start_i : start_i + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spectrum = np.abs(np.fft.rfft(seg * window))
        frames_list.append(spectrum)
        frame_times.append(start_i / effective_sr)

    spec_matrix = np.array(frames_list, dtype=np.float32) if frames_list else np.zeros((1, n_fft // 2 + 1), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / effective_sr)

    spec_sum = np.sum(spec_matrix, axis=1) + 1e-8
    centroids = np.sum(spec_matrix * freqs[None, :], axis=1) / spec_sum
    centroid_mean = round(float(np.mean(centroids)), 2)

    # 85% spectral rolloff
    cumsum_energy = np.cumsum(spec_matrix, axis=1)
    rolloff_Switch = 0.85 * spec_sum[:, None]
    rolloff_bins = np.argmax(cumsum_energy >= rolloff_Switch, axis=1)
    rolloff_freqs = freqs[np.clip(rolloff_bins, 0, len(freqs) - 1)]
    rolloff_mean = round(float(np.mean(rolloff_freqs)), 2)

    # High-frequency energy ratio (> effective_sr / 4)
    hf_mask = freqs >= (effective_sr * 0.25)
    total_energy = float(np.sum(spec_matrix ** 2)) + 1e-8
    hf_energy = float(np.sum((spec_matrix[:, hf_mask]) ** 2)) if np.any(hf_mask) else 0.0
    hf_ratio = round(min(1.0, max(0.0, hf_energy / total_energy)), 4)

    # Frame-to-frame spectral flux
    if len(spec_matrix) > 1:
        norm_spec = spec_matrix / spec_sum[:, None]
        diffs = np.linalg.norm(np.diff(norm_spec, axis=0), axis=1)
        flux_mean = round(float(np.mean(diffs)), 4)
        flux_std = float(np.std(diffs))
        flux_thresh = max(0.28, float(np.mean(diffs)) + 3.0 * flux_std)
    else:
        diffs = np.zeros(0, dtype=np.float32)
        flux_mean = 0.0
        flux_thresh = 1.0

    candidate_transitions: List[SpectralTransitionCandidate] = []
    for idx_f, flux_val in enumerate(diffs):
        if float(flux_val) > flux_thresh and len(candidate_transitions) < 10:
            ts = round(float(frame_times[idx_f + 1]), 3)
            candidate_transitions.append(
                SpectralTransitionCandidate(
                    timestamp_seconds=ts,
                    finding="spectral_characteristic_shift",
                    spectral_flux_score=round(float(flux_val), 4),
                )
            )

    spectrogram_path = artifact_dir / "spectrogram.png"
    _render_spectrogram_artifact(spec_matrix, spectrogram_path)
    spectrogram_rel = f"storage/results/{case_id}/audio/spectrogram.png"

    spectral_analysis = AudioSpectralAnalysis(
        status="potential_signal" if candidate_transitions else "no_strong_signal",
        finding="spectral_characteristic_shift" if candidate_transitions else None,
        description=(
            "A measurable change in spectral characteristics occurs within the sampled audio."
            if candidate_transitions
            else "No abrupt short-time spectral distribution shift exceeded conservative thresholds."
        ),
        spectral_centroid_mean_hz=centroid_mean,
        spectral_rolloff_mean_hz=rolloff_mean,
        spectral_flux_mean=flux_mean,
        high_frequency_energy_ratio=hf_ratio,
        candidate_transitions=candidate_transitions,
        artifact=spectrogram_rel,
        artifact_filename="spectrogram.png",
    )

    # 6. Discontinuity Analysis (Waveform Jump + Short-Window Step)
    discontinuities: List[AudioDiscontinuityCandidate] = []
    if len(mono) > 32:
        sample_diffs = np.abs(np.diff(mono))
        median_diff = float(np.median(sample_diffs)) + 1e-5
        jump_threshold = max(0.55, median_diff * 18.0)
        candidate_indices = np.where(sample_diffs >= jump_threshold)[0]

        last_added_time = -1.0
        for c_idx in candidate_indices:
            ts = round(float(c_idx) / effective_sr, 3)
            if ts - last_added_time < 0.15:
                continue
            last_added_time = ts
            start_s = round(max(0.0, ts - 0.02), 3)
            end_s = round(min(duration_seconds, ts + 0.02), 3)
            dur_s = round(max(0.01, end_s - start_s), 3)

            disc_art_rel: Optional[str] = None
            disc_art_name: Optional[str] = None
            if len(discontinuities) < MAX_DISCONTINUITY_ARTIFACTS:
                ms_tag = int(round(ts * 1000))
                disc_art_name = f"discontinuity_{ms_tag}.png"
                disc_art_path = artifact_dir / disc_art_name
                _render_discontinuity_artifact(mono, effective_sr, ts, disc_art_path)
                disc_art_rel = f"storage/results/{case_id}/audio/{disc_art_name}"

            discontinuities.append(
                AudioDiscontinuityCandidate(
                    start_seconds=start_s,
                    end_seconds=end_s,
                    duration_seconds=dur_s,
                    finding="possible_audio_discontinuity",
                    status="potentially_suspicious",
                    description="A measurable abrupt change in waveform/spectral characteristics was detected.",
                    artifact=disc_art_rel,
                    artifact_filename=disc_art_name,
                )
            )
            if len(discontinuities) >= MAX_DISCONTINUITY_CANDIDATES:
                break

    discontinuity_analysis = AudioDiscontinuityAnalysis(
        status="potential_signal" if discontinuities else "no_strong_signal",
        description=(
            f"{len(discontinuities)} candidate waveform/spectral discontinuity region(s) flagged for manual review."
            if discontinuities
            else "No abrupt waveform or phase discontinuities exceeded conservative thresholds."
        ),
        candidate_count=len(discontinuities),
        candidates=discontinuities,
    )

    # 7. Waveform Analysis & Waveform Artifact
    abs_mono = np.abs(mono)
    peak_amp = round(float(np.max(abs_mono)), 5) if len(mono) else 0.0
    rms_lvl = round(float(np.sqrt(np.mean(mono ** 2))), 5) if len(mono) else 0.0
    min_lvl = round(float(np.min(mono)), 5) if len(mono) else 0.0
    max_lvl = round(float(np.max(mono)), 5) if len(mono) else 0.0
    mean_lvl = round(float(np.mean(mono)), 5) if len(mono) else 0.0
    clipping_count = int(np.sum(abs_mono >= 0.999)) if len(mono) else 0

    # Dynamic range approximation in dB
    non_zero = abs_mono[abs_mono > 1e-5]
    noise_floor = float(np.percentile(non_zero, 10)) if len(non_zero) > 10 else 1e-4
    dyn_range_db = round(float(20.0 * math.log10(max(peak_amp, 1e-5) / max(noise_floor, 1e-5))), 2)

    waveform_path = artifact_dir / "waveform.png"
    _render_waveform_artifact(mono, analyzed_duration, silence_regions, discontinuities, waveform_path)
    waveform_rel = f"storage/results/{case_id}/audio/waveform.png"

    waveform_analysis = AudioWaveformAnalysis(
        status="available",
        peak_amplitude=peak_amp,
        rms_level=rms_lvl,
        dynamic_range_db=max(0.0, dyn_range_db),
        minimum_level=min_lvl,
        maximum_level=max_lvl,
        mean_level=mean_lvl,
        clipping_sample_count=clipping_count,
        analyzed_duration_seconds=analyzed_duration,
        artifact=waveform_rel,
        artifact_filename="waveform.png",
    )

    # 8. Recompression / Transcoding Analysis
    recomp_indicators: List[str] = []
    if case.extension in LOSSY_EXTENSIONS:
        recomp_indicators.append(f"Lossy audio container/codec detected ({metadata_info.format})")
    if metadata_info.encoder or metadata_info.software:
        enc_label = metadata_info.encoder or metadata_info.software
        recomp_indicators.append(f"Encoder/software metadata present ({enc_label})")
    if metadata_info.bitrate and metadata_info.bitrate < 128_000 and case.extension in LOSSY_EXTENSIONS:
        recomp_indicators.append(f"Low lossy bitrate observed ({metadata_info.bitrate // 1000} kbps)")
    if hf_ratio < 0.005 and sample_rate >= 32000:
        recomp_indicators.append("Attenuated high-frequency spectral energy consistent with lossy filtering or transcoding")

    is_possible_recomp = len(recomp_indicators) >= 2 or bool(metadata_info.encoder or metadata_info.software)
    recompression_analysis = AudioRecompressionAnalysis(
        status="possible_recompression" if is_possible_recomp else "no_strong_signal",
        description=(
            "Audio encoding characteristics are consistent with possible transcoding or re-encoding."
            if is_possible_recomp
            else "No strong transcoding or multi-generation recompression indicators were observed."
        ),
        indicators=recomp_indicators,
        limitation="Possible recompression is not equivalent to manipulation.",
    )

    # 9. Conservative Qualitative Assessment
    if discontinuities:
        assessment = AudioForensicAssessment(
            status="possible_discontinuity",
            label="Possible Audio Discontinuity Observed",
            description="A candidate abrupt waveform/spectral transition was detected and should be reviewed manually.",
        )
    elif candidate_transitions:
        assessment = AudioForensicAssessment(
            status="potential_audio_anomaly",
            label="Candidate Spectral / Acoustic Transition Observed",
            description="A measurable shift in short-time spectral characteristics was observed; manual forensic listening review is recommended.",
        )
    elif is_possible_recomp:
        assessment = AudioForensicAssessment(
            status="possible_recompression",
            label="Possible Audio Recompression / Encoding Signal",
            description="Encoding or spectral characteristics are consistent with possible transcoding or re-encoding. Possible recompression is not equivalent to manipulation.",
        )
    else:
        assessment = AudioForensicAssessment(
            status="no_strong_indicators",
            label="No Strong Forensic Anomaly Indicators",
            description="No strong waveform discontinuities or spectral anomalies were observed within the analyzed audio stream.",
        )

    audio_result = AudioAnalysisResult(
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        media_type="audio",
        filename=case.original_filename,
        format=metadata_info.format or case.extension.lstrip(".").upper(),
        container=metadata_info.container,
        codec=metadata_info.codec,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        bitrate=metadata_info.bitrate,
        metadata=metadata_info,
        waveform_analysis=waveform_analysis,
        silence_analysis=silence_analysis,
        spectral_analysis=spectral_analysis,
        discontinuity_analysis=discontinuity_analysis,
        recompression_analysis=recompression_analysis,
        ai_generation=AudioAiDetectionResult(),
        assessment=assessment,
        limitations=list(STANDARD_AUDIO_LIMITATIONS),
    )

    # 10. Post-analysis SHA-256 verification to guarantee original audio immutability
    post_sha256 = _compute_file_sha256(audio_path)
    if post_sha256 != case.sha256:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Post-analysis SHA-256 verification failed.",
        )

    case.audio_analysis = audio_result
    case.status = "analyzed"

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    audio_art_specs = []
    audio_dir = settings.results_dir / case_id / "audio"
    if audio_dir.is_dir():
        for png_file in sorted(audio_dir.glob("*.png")):
            if png_file.name == "waveform.png":
                art_type = "audio_waveform"
            elif png_file.name == "spectrogram.png":
                art_type = "audio_spectrogram"
            else:
                art_type = "audio_discontinuity"
            audio_art_specs.append(
                {
                    "artifact_type": art_type,
                    "relative_path": f"storage/results/{case_id}/audio/{png_file.name}",
                    "media_type": "image/png",
                }
            )

    record_analysis_execution(
        case_data=case,
        service_name="audio_analysis_service",
        status="completed",
        input_sha256=case.sha256,
        output_summary=(
            f"Audio analysis completed: {sample_rate}Hz, {channels}ch, "
            f"duration={duration_seconds}s, assessment={assessment.status}."
        ),
        artifact_specs=audio_art_specs,
        completed_at=audio_result.analyzed_at,
    )
    create_forensic_snapshot(case, record_event=True)

    return media_service.save_media_case(case)


def get_audio_artifact_path(case_id: str, artifact_name: str) -> Path:
    """
    Validate and resolve a derived audio artifact (waveform.png, spectrogram.png, or discontinuity_<ms>.png)
    under storage/results/<case_id>/audio/. Prevents path traversal.
    """
    media_service.get_media_case(case_id)
    if not ALLOWED_ARTIFACT_NAME_PATTERN.match(artifact_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio artifact name '{artifact_name}'.",
        )

    artifact_dir = (settings.results_dir / case_id / "audio").resolve()
    target_path = (artifact_dir / artifact_name).resolve()
    if artifact_dir not in target_path.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path.",
        )

    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio artifact '{artifact_name}' not found for case '{case_id}'.",
        )

    return target_path
