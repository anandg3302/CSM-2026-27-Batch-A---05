from datetime import datetime, timezone
import hashlib
import io
import json
import os
from typing import Any, List, Optional

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.analysis import (
    ExifFields,
    ForensicFinding,
    ImageAnalysisResult,
    ImageMetadataInfo,
    ImageTechnicalInfo,
)
from app.schemas.media import MediaCaseMetadata
from app.services.media_service import CASE_ID_PATTERN

# Standard EXIF tag IDs
TAG_MAKE = 271
TAG_MODEL = 272
TAG_ORIENTATION = 274
TAG_SOFTWARE = 305
TAG_DATETIME = 306
TAG_EXIF_IFD = 0x8769
TAG_GPS_IFD = 0x8825
TAG_DATETIME_ORIGINAL = 36867

ORIENTATION_LABELS = {
    1: "Horizontal (normal)",
    2: "Mirror horizontal",
    3: "Rotate 180",
    4: "Mirror vertical",
    5: "Mirror horizontal and rotate 270 CW",
    6: "Rotate 90 CW",
    7: "Mirror horizontal and rotate 90 CW",
    8: "Rotate 270 CW",
}


def _clean_exif_string(value: Any) -> Optional[str]:
    """Convert raw EXIF value into a clean printable string, or None if empty."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8", errors="replace").rstrip("\x00").strip()
            return decoded or None
        except Exception:
            return None
    text = str(value).rstrip("\x00").strip()
    return text or None


def _extract_exif_and_gps(img: Image.Image) -> ImageMetadataInfo:
    """
    Safely extract EXIF metadata and detect whether GPS metadata is present.
    Raw GPS coordinates are strictly omitted for privacy.
    """
    try:
        exif = img.getexif()
    except Exception:
        exif = None

    if not exif:
        return ImageMetadataInfo(
            exif_present=False,
            gps_present=False,
            gps_info=None,
            fields=ExifFields(),
        )

    # Check sub-IFDs for Exif (DateTimeOriginal) and GPSInfo
    exif_ifd = {}
    gps_ifd = {}
    try:
        exif_ifd = exif.get_ifd(TAG_EXIF_IFD) or {}
    except Exception:
        exif_ifd = {}

    try:
        gps_ifd = exif.get_ifd(TAG_GPS_IFD) or {}
    except Exception:
        gps_ifd = {}

    has_gps = bool(gps_ifd) or (TAG_GPS_IFD in exif and bool(exif.get(TAG_GPS_IFD)))

    make = _clean_exif_string(exif.get(TAG_MAKE))
    model = _clean_exif_string(exif.get(TAG_MODEL))
    software = _clean_exif_string(exif.get(TAG_SOFTWARE))
    dt = _clean_exif_string(exif.get(TAG_DATETIME))
    dt_orig = _clean_exif_string(exif_ifd.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME_ORIGINAL))

    raw_orientation = exif.get(TAG_ORIENTATION)
    orientation_str: Optional[str] = None
    if raw_orientation is not None:
        if isinstance(raw_orientation, int) and raw_orientation in ORIENTATION_LABELS:
            orientation_str = f"{ORIENTATION_LABELS[raw_orientation]} ({raw_orientation})"
        else:
            orientation_str = _clean_exif_string(raw_orientation)

    exif_has_any_content = bool(len(exif) > 0 or len(exif_ifd) > 0 or has_gps)

    return ImageMetadataInfo(
        exif_present=exif_has_any_content,
        gps_present=has_gps,
        gps_info="GPS metadata present" if has_gps else None,
        fields=ExifFields(
            make=make,
            model=model,
            software=software,
            datetime=dt,
            datetime_original=dt_orig,
            orientation=orientation_str,
        ),
    )


def _build_forensic_findings(
    case_data: MediaCaseMetadata,
    tech_info: ImageTechnicalInfo,
    meta_info: ImageMetadataInfo,
) -> List[ForensicFinding]:
    """
    Construct objective, neutral forensic observations from extracted properties.
    Does not make manipulation or AI-generation claims.
    """
    findings: List[ForensicFinding] = [
        ForensicFinding(
            category="file",
            finding="File integrity SHA-256 verified",
            value=case_data.sha256,
            severity="informational",
        ),
        ForensicFinding(
            category="image",
            finding="Image geometry and color structure extracted",
            value=f"{tech_info.format or 'UNKNOWN'} {tech_info.width}x{tech_info.height} px, mode {tech_info.mode} ({tech_info.channels or 'unknown'} channels)",
            severity="informational",
        ),
    ]

    if tech_info.format == "JPEG":
        q_desc = (
            f"{tech_info.quantization_tables_count} quantization table(s)"
            if tech_info.quantization_tables_count is not None
            else "quantization tables unknown"
        )
        findings.append(
            ForensicFinding(
                category="compression",
                finding="JPEG compression structure inspected",
                value=f"{(tech_info.jpeg_encoding or 'unknown').capitalize()} encoding, {q_desc}",
                severity="informational",
            )
        )

    if meta_info.exif_present:
        findings.append(
            ForensicFinding(
                category="metadata",
                finding="EXIF metadata container present",
                value="Standard EXIF header detected in file",
                severity="informational",
            )
        )
        if meta_info.fields.software:
            findings.append(
                ForensicFinding(
                    category="metadata",
                    finding="Editing software metadata present",
                    value=meta_info.fields.software,
                    severity="informational",
                )
            )
        if meta_info.fields.make or meta_info.fields.model:
            camera_label = " ".join(
                part for part in (meta_info.fields.make, meta_info.fields.model) if part
            )
            findings.append(
                ForensicFinding(
                    category="metadata",
                    finding="Capture device metadata present",
                    value=camera_label,
                    severity="informational",
                )
            )
        if meta_info.fields.datetime_original or meta_info.fields.datetime:
            findings.append(
                ForensicFinding(
                    category="metadata",
                    finding="Timestamp metadata present",
                    value=meta_info.fields.datetime_original or meta_info.fields.datetime or "",
                    severity="informational",
                )
            )
        if meta_info.gps_present:
            findings.append(
                ForensicFinding(
                    category="metadata",
                    finding="GPS metadata present",
                    value="GPS metadata present",
                    severity="informational",
                )
            )
    else:
        findings.append(
            ForensicFinding(
                category="metadata",
                finding="EXIF metadata not present",
                value="No EXIF tags found in container",
                severity="informational",
            )
        )

    return findings


def analyze_image(case_id: str) -> MediaCaseMetadata:
    """
    Run foundational forensic analysis on a stored image case without modifying
    the original uploaded file, and persist the result in storage/results/<case_id>.json.
    """
    if not CASE_ID_PATTERN.match(case_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID format.",
        )

    result_path = settings.results_dir / f"{case_id}.json"
    if not result_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forensic case '{case_id}' was not found.",
        )

    try:
        with open(result_path, "r", encoding="utf-8") as file_in:
            raw_case_dict = json.load(file_in)
        case_data = MediaCaseMetadata(**raw_case_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read metadata for case '{case_id}'.",
        ) from exc

    if case_data.media_type != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' has media_type '{case_data.media_type}', expected 'image'.",
        )

    upload_path = settings.uploads_dir / case_data.stored_filename
    if not upload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original uploaded file for case '{case_id}' was not found.",
        )

    # Read original bytes in read-only mode and verify chain-of-custody hash
    file_bytes = upload_path.read_bytes()
    current_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if current_sha256 != case_data.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="File integrity check failed: stored image SHA-256 does not match case record.",
        )

    # Inspect image from in-memory buffer so the file on disk is never modified
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img_format = img.format or "UNKNOWN"
            width, height = img.size
            if width <= 0 or height <= 0 or (width * height) > 64_000_000:
                raise ValueError("Image dimensions are invalid or exceed safe pixel bounds.")
            mode = img.mode
            bands = img.getbands()
            channels = len(bands) if bands else None

            jpeg_encoding: Optional[str] = None
            qtables_count: Optional[int] = None
            if img_format == "JPEG":
                is_prog = bool(img.info.get("progressive") or img.info.get("progression"))
                jpeg_encoding = "progressive" if is_prog else "baseline"
                qtables = getattr(img, "quantization", None)
                if isinstance(qtables, dict):
                    qtables_count = len(qtables)

            tech_info = ImageTechnicalInfo(
                format=img_format,
                width=width,
                height=height,
                mode=mode,
                channels=channels,
                jpeg_encoding=jpeg_encoding,
                quantization_tables_count=qtables_count,
            )

            meta_info = _extract_exif_and_gps(img)
            img.load()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid decodable image container.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to inspect image contents.",
        ) from exc

    findings = _build_forensic_findings(case_data, tech_info, meta_info)

    analysis_result = ImageAnalysisResult(
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        image=tech_info,
        metadata=meta_info,
        findings=findings,
    )

    updated_case = case_data.model_copy(
        update={
            "status": "analyzed",
            "analysis": analysis_result,
        }
    )

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    record_analysis_execution(
        case_data=updated_case,
        service_name="image_analysis_service",
        status="completed",
        input_sha256=updated_case.sha256,
        output_summary=(
            f"Inspected {tech_info.format} image ({tech_info.width}x{tech_info.height}, {tech_info.mode}); "
            f"EXIF present={meta_info.exif_present}, findings={len(findings)}."
        ),
        completed_at=analysis_result.analyzed_at,
    )
    create_forensic_snapshot(updated_case, record_event=True)

    # Write atomically via a temporary file so the existing JSON is never corrupted on failure
    temp_path = settings.results_dir / f"{case_id}.json.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as tmp_out:
            json.dump(updated_case.model_dump(), tmp_out, indent=2)
        os.replace(temp_path, result_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist analysis results.",
        ) from exc

    return updated_case
