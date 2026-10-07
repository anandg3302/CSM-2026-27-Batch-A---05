from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
from typing import List
import uuid

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.media import MediaCaseMetadata

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt"}
ALLOWED_MEDIA_EXTENSIONS = (
    ALLOWED_IMAGE_EXTENSIONS
    | ALLOWED_VIDEO_EXTENSIONS
    | ALLOWED_AUDIO_EXTENSIONS
    | ALLOWED_DOCUMENT_EXTENSIONS
)
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB for images
MAX_VIDEO_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB for videos
MAX_AUDIO_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB for audio
MAX_DOCUMENT_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB for documents
CASE_ID_PATTERN = re.compile(r"^MT-\d{8}-[A-F0-9]{8}$")
SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


MAX_FILENAME_STEM_LENGTH = 100


def sanitize_filename(raw_filename: str) -> str:
    """
    Strip directory traversal paths (both POSIX and Windows), reject null bytes,
    bound filename length, and sanitize characters to produce a safe basename.
    """
    if not raw_filename or not raw_filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    if "\x00" in raw_filename or "%00" in raw_filename.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename contains invalid null byte characters.",
        )

    # Extract leaf name across both Windows and POSIX separators
    leaf_win = PureWindowsPath(raw_filename.strip()).name
    leaf_name = PurePosixPath(leaf_win).name

    if not leaf_name or leaf_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    path_obj = Path(leaf_name)
    stem = SAFE_FILENAME_CHARS.sub("_", path_obj.stem).strip("._")
    if len(stem) > MAX_FILENAME_STEM_LENGTH:
        stem = stem[:MAX_FILENAME_STEM_LENGTH].rstrip("._")
    raw_suffix = path_obj.suffix.lower()
    suffix = "." + SAFE_FILENAME_CHARS.sub("", raw_suffix.lstrip(".")) if raw_suffix else ""

    if not stem:
        stem = "media"

    return f"{stem}{suffix}"


def validate_image_extension(filename: str) -> str:
    """Verify the file has an allowed image extension and return the lowercase extension."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{extension or 'none'}'. Allowed extensions: {allowed_str}",
        )
    return extension


def validate_media_extension(filename: str) -> tuple[str, str]:
    """Verify the file has an allowed image, video, audio, or document extension and return (extension, media_type)."""
    extension = Path(filename).suffix.lower()
    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return extension, "image"
    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return extension, "video"
    if extension in ALLOWED_AUDIO_EXTENSIONS:
        return extension, "audio"
    if extension in ALLOWED_DOCUMENT_EXTENSIONS:
        return extension, "document"

    allowed_str = ", ".join(sorted(ALLOWED_MEDIA_EXTENSIONS))
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported file extension '{extension or 'none'}'. Allowed extensions: {allowed_str}",
    )


def generate_unique_case_id() -> str:
    """
    Generate a unique readable case ID in the format MT-YYYYMMDD-<8-hex-chars>.
    Ensures no collision with existing result JSON files.
    """
    settings.ensure_storage_directories()
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")

    for _ in range(10):
        unique_suffix = uuid.uuid4().hex[:8].upper()
        candidate = f"MT-{date_part}-{unique_suffix}"
        result_path = settings.results_dir / f"{candidate}.json"
        if not result_path.exists():
            return candidate

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Could not allocate a unique case ID.",
    )


async def create_media_case(upload_file: UploadFile | None) -> MediaCaseMetadata:
    """
    Validate an uploaded image, video, audio, or document file, calculate its SHA-256 digest,
    persist the original bytes under storage/uploads/, and create case metadata in storage/results/.
    """
    if upload_file is None or not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No media file was provided.",
        )

    safe_filename = sanitize_filename(upload_file.filename)
    extension, media_type = validate_media_extension(safe_filename)

    file_bytes = await upload_file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if media_type == "video":
        max_allowed_bytes = MAX_VIDEO_UPLOAD_SIZE_BYTES
    elif media_type == "audio":
        max_allowed_bytes = MAX_AUDIO_UPLOAD_SIZE_BYTES
    elif media_type == "document":
        max_allowed_bytes = MAX_DOCUMENT_UPLOAD_SIZE_BYTES
    else:
        max_allowed_bytes = MAX_UPLOAD_SIZE_BYTES
    if file_size > max_allowed_bytes:
        max_mb = max_allowed_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb} MB.",
        )

    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    settings.ensure_storage_directories()
    case_id = generate_unique_case_id()
    stored_filename = f"{case_id}_{safe_filename}"

    upload_target = settings.uploads_dir / stored_filename
    result_target = settings.results_dir / f"{case_id}.json"

    # Exclusive file creation ('xb' / 'x') ensures we never overwrite existing files
    try:
        with open(upload_target, "xb") as file_out:
            file_out.write(file_bytes)
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Target upload file already exists.",
        ) from exc

    from app.core.version import SCHEMA_VERSIONS
    from app.services.evidence.reproducibility_service import append_audit_event

    created_iso = datetime.now(timezone.utc).isoformat()
    metadata = MediaCaseMetadata(
        case_id=case_id,
        original_filename=safe_filename,
        stored_filename=stored_filename,
        media_type=media_type,
        extension=extension,
        file_size=file_size,
        sha256=sha256_hash,
        status="uploaded",
        created_at=created_iso,
        case_schema_version=SCHEMA_VERSIONS["case_schema_version"],
        analysis_registry=[],
        artifact_registry=[],
        audit_events=[],
        snapshots=[],
        report_generations=[],
    )
    append_audit_event(
        case_data=metadata,
        event_type="case_uploaded",
        summary=f"Uploaded original media file '{safe_filename}' ({file_size} bytes, SHA-256: {sha256_hash[:12]}...).",
        details={
            "original_filename": safe_filename,
            "safe_filename": stored_filename,
            "relative_storage_path": f"storage/uploads/{stored_filename}",
            "media_type": media_type,
            "byte_size": file_size,
            "sha256": sha256_hash,
        },
    )

    try:
        with open(result_target, "x", encoding="utf-8") as json_out:
            json.dump(metadata.model_dump(), json_out, indent=2)
    except Exception as exc:
        # Roll back uploaded file if metadata write fails
        if upload_target.exists():
            upload_target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save forensic case metadata.",
        ) from exc

    return metadata


def get_media_case(case_id: str) -> MediaCaseMetadata:
    """Retrieve stored case metadata JSON by case_id."""
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
            raw_data = json.load(file_in)
        return MediaCaseMetadata(**raw_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metadata for case '{case_id}'.",
        ) from exc


def list_media_cases() -> List[MediaCaseMetadata]:
    """Return all stored forensic cases ordered newest first."""
    settings.ensure_storage_directories()
    cases: List[MediaCaseMetadata] = []

    for json_file in settings.results_dir.glob("MT-*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as file_in:
                raw_data = json.load(file_in)
            cases.append(MediaCaseMetadata(**raw_data))
        except Exception:
            # Ignore malformed non-case files
            continue

    cases.sort(key=lambda item: item.created_at, reverse=True)
    return cases


def atomic_write_text(target_path: Path, content: str) -> None:
    """Atomically write UTF-8 text to target_path using a temporary file and os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.{secrets.token_hex(4)}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def save_media_case(case_data: MediaCaseMetadata) -> MediaCaseMetadata:
    """Atomically persist updated MediaCaseMetadata to storage/results/<case_id>.json."""
    settings.ensure_storage_directories()
    result_path = settings.results_dir / f"{case_data.case_id}.json"
    atomic_write_text(result_path, case_data.model_dump_json(indent=2))
    return case_data

