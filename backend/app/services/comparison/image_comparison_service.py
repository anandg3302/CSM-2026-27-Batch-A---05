from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from typing import List, Tuple

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageChops, ImageDraw, ImageStat, UnidentifiedImageError

from app.core.config import settings
from app.schemas.analysis import ImageDimensions
from app.schemas.comparison import (
    AllowedComparisonAssessment,
    ChangedRegion,
    ComparisonEndpointSummary,
    ComparisonVisualization,
    DimensionComparisonInfo,
    ImageComparisonResult,
    MetadataComparisonInfo,
    PixelDifferenceInfo,
    ReferenceImageMetadata,
)
from app.schemas.media import MediaCaseMetadata
from app.services.analysis import image_analysis_service
from app.services.media_service import (
    CASE_ID_PATTERN,
    MAX_UPLOAD_SIZE_BYTES,
    sanitize_filename,
    validate_image_extension,
)

# Documented per-channel intensity threshold (0..255) above which a pixel is counted as changed.
DIFFERENCE_THRESHOLD = 10

# Documented minimum changed-pixel count and bounding-box area (in pixels) to filter tiny isolated noise.
MIN_CHANGED_PIXELS_IN_REGION = 16
MIN_CHANGED_REGION_AREA_PX = 36

ASSESSMENT_LABELS = {
    "no_significant_difference": "No significant pixel differences detected",
    "visual_differences_detected": "Images differ across broad visual areas",
    "localized_visual_differences_detected": "Localized visual differences detected",
    "different_dimensions": "Images have different dimensions (compared via normalized resize)",
    "insufficient_data": "Insufficient data for comparison",
}


def _load_case(case_id: str) -> Tuple[MediaCaseMetadata, Path]:
    """Validate case_id and load MediaCaseMetadata from storage/results/<case_id>.json."""
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
        return MediaCaseMetadata(**raw_data), result_path
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metadata for case '{case_id}'.",
        ) from exc


def _atomic_save_case(case_data: MediaCaseMetadata, result_path: Path) -> None:
    """Atomically write updated MediaCaseMetadata to storage/results/<case_id>.json."""
    temp_path = result_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as tmp_out:
            json.dump(case_data.model_dump(), tmp_out, indent=2)
        os.replace(temp_path, result_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist updated case metadata.",
        ) from exc


async def upload_reference_image(
    case_id: str,
    upload_file: UploadFile | None,
) -> MediaCaseMetadata:
    """
    Validate and store a separate reference image under:
      storage/uploads/<case_id>_reference_<safe_filename>
    Never overwrites or modifies the original case image.
    """
    case_data, result_path = _load_case(case_id)

    if case_data.media_type != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' has media_type '{case_data.media_type}', expected 'image'.",
        )

    case_upload_path = settings.uploads_dir / case_data.stored_filename
    if not case_upload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original uploaded file for case '{case_id}' was not found.",
        )

    if upload_file is None or not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No reference image file was provided.",
        )

    safe_filename = sanitize_filename(upload_file.filename)
    extension = validate_image_extension(safe_filename)

    file_bytes = await upload_file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded reference file is empty.",
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Reference file exceeds maximum allowed size of {max_mb} MB.",
        )

    # Verify the bytes represent a valid decodable image
    try:
        with Image.open(io.BytesIO(file_bytes)) as ref_img:
            ref_img.load()
            img_format = (ref_img.format or "UNKNOWN").upper()
            width, height = ref_img.size
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded reference file is not a valid decodable image.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode reference image.",
        ) from exc

    ref_sha256 = hashlib.sha256(file_bytes).hexdigest()
    stored_filename = f"{case_id}_reference_{safe_filename}"
    ref_target_path = settings.uploads_dir / stored_filename

    # Ensure we never overwrite the original case upload path
    if ref_target_path.resolve() == case_upload_path.resolve():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Reference storage path conflicts with original case image path.",
        )

    # Remove previous reference file if a reference image is being replaced for this case
    if case_data.reference_image:
        old_ref_path = settings.uploads_dir / case_data.reference_image.stored_filename
        if old_ref_path.is_file() and old_ref_path.resolve() != case_upload_path.resolve():
            old_ref_path.unlink(missing_ok=True)

    with open(ref_target_path, "wb") as ref_out:
        ref_out.write(file_bytes)

    # Verify original case image SHA-256 remains untouched
    if hashlib.sha256(case_upload_path.read_bytes()).hexdigest() != case_data.sha256:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Original case image integrity check failed.",
        )

    ref_metadata = ReferenceImageMetadata(
        original_filename=safe_filename,
        stored_filename=stored_filename,
        extension=extension,
        file_size=file_size,
        sha256=ref_sha256,
        format=img_format,
        width=width,
        height=height,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )

    updated_case = case_data.model_copy(update={"reference_image": ref_metadata})
    _atomic_save_case(updated_case, result_path)
    return updated_case


def _extract_changed_regions(
    mask_img: Image.Image,
    img_width: int,
    img_height: int,
) -> List[ChangedRegion]:
    """
    Identify connected regions of changed pixels from the binary difference mask (0 or 255).
    Uses an 8-connected grid traversal followed by exact bounding-box tightening (`getbbox`).
    """
    cell_size = 16 if min(img_width, img_height) >= 128 else 8
    cols = max(1, (img_width + cell_size - 1) // cell_size)
    rows = max(1, (img_height + cell_size - 1) // cell_size)

    active_grid = [[False] * cols for _ in range(rows)]
    for r_idx in range(rows):
        y0 = r_idx * cell_size
        y1 = min(img_height, y0 + cell_size)
        for c_idx in range(cols):
            x0 = c_idx * cell_size
            x1 = min(img_width, x0 + cell_size)
            tile_hist = mask_img.crop((x0, y0, x1, y1)).histogram()
            if tile_hist[255] > 0:
                active_grid[r_idx][c_idx] = True

    visited = [[False] * cols for _ in range(rows)]
    regions: List[ChangedRegion] = []

    for r_start in range(rows):
        for c_start in range(cols):
            if visited[r_start][c_start] or not active_grid[r_start][c_start]:
                continue

            queue = deque([(r_start, c_start)])
            visited[r_start][c_start] = True
            min_r, max_r = r_start, r_start
            min_c, max_c = c_start, c_start

            while queue:
                curr_r, curr_c = queue.popleft()
                min_r = min(min_r, curr_r)
                max_r = max(max_r, curr_r)
                min_c = min(min_c, curr_c)
                max_c = max(max_c, curr_c)

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = curr_r + dr, curr_c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr][nc]
                            and active_grid[nr][nc]
                        ):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            coarse_x0 = max(0, min_c * cell_size)
            coarse_y0 = max(0, min_r * cell_size)
            coarse_x1 = min(img_width, (max_c + 1) * cell_size)
            coarse_y1 = min(img_height, (max_r + 1) * cell_size)

            sub_mask = mask_img.crop((coarse_x0, coarse_y0, coarse_x1, coarse_y1))
            tight_bbox = sub_mask.getbbox()
            if not tight_bbox:
                continue

            bx0 = coarse_x0 + tight_bbox[0]
            by0 = coarse_y0 + tight_bbox[1]
            bx1 = coarse_x0 + tight_bbox[2]
            by1 = coarse_y0 + tight_bbox[3]

            reg_w = max(1, bx1 - bx0)
            reg_h = max(1, by1 - by0)
            reg_area = reg_w * reg_h

            tight_crop = mask_img.crop((bx0, by0, bx1, by1))
            changed_px_count = int(tight_crop.histogram()[255])

            if changed_px_count < MIN_CHANGED_PIXELS_IN_REGION or reg_area < MIN_CHANGED_REGION_AREA_PX:
                continue

            changed_pct = round((changed_px_count / reg_area) * 100.0, 2)
            regions.append(
                ChangedRegion(
                    id=f"change-{len(regions) + 1}",
                    x=bx0,
                    y=by0,
                    width=reg_w,
                    height=reg_h,
                    area=reg_area,
                    changed_pixel_count=changed_px_count,
                    changed_pixel_percentage=changed_pct,
                )
            )

    regions.sort(key=lambda item: item.changed_pixel_count, reverse=True)
    for idx, reg in enumerate(regions, start=1):
        reg.id = f"change-{idx}"

    return regions


def _generate_comparison_visualizations(
    case_id: str,
    case_rgb: Image.Image,
    mask_img: Image.Image,
    regions: List[ChangedRegion],
) -> ComparisonVisualization:
    """
    Save binary difference mask ('difference_mask.png') and visual difference highlight
    ('difference.png') under storage/results/<case_id>/comparison/.
    """
    comp_dir = settings.results_dir / case_id / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    width, height = case_rgb.size

    # Create dimmed grayscale background of the case image so changed regions stand out clearly
    gray_base = case_rgb.convert("L")
    dimmed_gray = gray_base.point(lambda p: int(p * 0.45))
    visual_base = Image.merge("RGB", (dimmed_gray, dimmed_gray, dimmed_gray))

    # Highlight changed pixels (mask == 255) in high-visibility crimson-magenta
    highlight_layer = Image.new("RGB", (width, height), color=(235, 40, 95))
    visual_diff = Image.composite(highlight_layer, visual_base, mask_img)

    if regions:
        draw = ImageDraw.Draw(visual_diff)
        for reg in regions:
            x1 = max(0, min(width - 1, reg.x))
            y1 = max(0, min(height - 1, reg.y))
            x2 = max(x1, min(width - 1, reg.x + reg.width - 1))
            y2 = max(y1, min(height - 1, reg.y + reg.height - 1))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)

    diff_path = comp_dir / "difference.png"
    mask_path = comp_dir / "difference_mask.png"

    visual_diff.save(diff_path, format="PNG")
    mask_img.save(mask_path, format="PNG")

    return ComparisonVisualization(
        difference="comparison/difference.png",
        mask="comparison/difference_mask.png",
    )


def compare_case_with_reference(case_id: str) -> MediaCaseMetadata:
    """
    Compare the case image with its uploaded reference image.
    Preserves Step 3 analysis, Step 4 manipulation_detection, and Step 5 localization.
    Verifies both files' SHA-256 digests before and after execution.
    """
    case_data, result_path = _load_case(case_id)

    if case_data.media_type != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' has media_type '{case_data.media_type}', expected 'image'.",
        )

    if case_data.reference_image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No reference image has been uploaded for case '{case_id}'.",
        )

    case_upload_path = settings.uploads_dir / case_data.stored_filename
    ref_upload_path = settings.uploads_dir / case_data.reference_image.stored_filename

    if not case_upload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original uploaded file for case '{case_id}' was not found.",
        )
    if not ref_upload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference image file for case '{case_id}' was not found.",
        )

    # Record and verify SHA-256 digests BEFORE comparison
    case_bytes = case_upload_path.read_bytes()
    ref_bytes = ref_upload_path.read_bytes()
    case_sha256_before = hashlib.sha256(case_bytes).hexdigest()
    ref_sha256_before = hashlib.sha256(ref_bytes).hexdigest()

    if case_sha256_before != case_data.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Case image SHA-256 integrity check failed before comparison.",
        )
    if ref_sha256_before != case_data.reference_image.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Reference image SHA-256 integrity check failed before comparison.",
        )

    try:
        with Image.open(io.BytesIO(case_bytes)) as case_img, Image.open(io.BytesIO(ref_bytes)) as ref_img:
            case_img.load()
            ref_img.load()

            case_w, case_h = case_img.size
            ref_w, ref_h = ref_img.size
            case_fmt = (case_img.format or "UNKNOWN").upper()
            ref_fmt = (ref_img.format or "UNKNOWN").upper()
            case_mode = case_img.mode
            ref_mode = ref_img.mode

            case_meta = image_analysis_service._extract_exif_and_gps(case_img)
            ref_meta = image_analysis_service._extract_exif_and_gps(ref_img)

            same_dims = (case_w == ref_w) and (case_h == ref_h)
            case_ratio = case_w / max(1, case_h)
            ref_ratio = ref_w / max(1, ref_h)
            same_aspect = abs(case_ratio - ref_ratio) < 0.01

            comparison_mode = "direct" if same_dims else "normalized_resize"

            case_rgb = case_img.convert("RGB")
            if same_dims:
                comp_ref_rgb = ref_img.convert("RGB")
            else:
                comp_ref_rgb = ref_img.convert("RGB").resize(
                    (case_w, case_h),
                    resample=Image.Resampling.BILINEAR,
                )

            diff_img = ImageChops.difference(case_rgb, comp_ref_rgb)
            r_diff, g_diff, b_diff = diff_img.split()
            max_ch_diff = ImageChops.lighter(ImageChops.lighter(r_diff, g_diff), b_diff)

            extrema = max_ch_diff.getextrema()
            max_pixel_diff = int(extrema[1]) if extrema else 0

            stat_diff = ImageStat.Stat(diff_img)
            mean_abs_diff = round(sum(float(m) for m in stat_diff.mean) / 3.0, 2)
            pixel_similarity = round(max(0.0, min(1.0, 1.0 - (mean_abs_diff / 255.0))), 4)

            # Binary difference mask: 255 where per-channel difference > DIFFERENCE_THRESHOLD, else 0
            mask_img = max_ch_diff.point(lambda p: 255 if p > DIFFERENCE_THRESHOLD else 0)
            total_pixels = max(1, case_w * case_h)
            changed_pixels_total = int(mask_img.histogram()[255])
            changed_pixel_pct = round((changed_pixels_total / total_pixels) * 100.0, 2)

            changed_regions = _extract_changed_regions(mask_img, case_w, case_h)
            vis = _generate_comparison_visualizations(case_id, case_rgb, mask_img, changed_regions)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare case and reference images.",
        ) from exc

    # Determine conservative comparison assessment
    if not same_dims:
        assessment: AllowedComparisonAssessment = "different_dimensions"
    elif changed_pixel_pct < 0.05 and len(changed_regions) == 0:
        assessment = "no_significant_difference"
    elif len(changed_regions) > 0 and changed_pixel_pct <= 55.0:
        assessment = "localized_visual_differences_detected"
    else:
        assessment = "visual_differences_detected"

    case_camera = " ".join(p for p in (case_meta.fields.make, case_meta.fields.model) if p) or None
    ref_camera = " ".join(p for p in (ref_meta.fields.make, ref_meta.fields.model) if p) or None
    case_ts = case_meta.fields.datetime_original or case_meta.fields.datetime
    ref_ts = ref_meta.fields.datetime_original or ref_meta.fields.datetime

    meta_comp = MetadataComparisonInfo(
        format_changed=(case_fmt != ref_fmt),
        dimensions_changed=(not same_dims),
        exif_presence_changed=(case_meta.exif_present != ref_meta.exif_present),
        software_changed=(case_meta.fields.software != ref_meta.fields.software),
        camera_changed=(case_camera != ref_camera),
        timestamp_changed=(case_ts != ref_ts),
        case_software=case_meta.fields.software,
        reference_software=ref_meta.fields.software,
    )

    comp_result = ImageComparisonResult(
        status="completed",
        compared_at=datetime.now(timezone.utc).isoformat(),
        comparison_mode=comparison_mode,
        case_image=ComparisonEndpointSummary(
            width=case_w,
            height=case_h,
            format=case_fmt,
            mode=case_mode,
            sha256=case_sha256_before,
        ),
        reference_image=ComparisonEndpointSummary(
            width=ref_w,
            height=ref_h,
            format=ref_fmt,
            mode=ref_mode,
            sha256=ref_sha256_before,
        ),
        dimensions=DimensionComparisonInfo(
            same=same_dims,
            same_dimensions=same_dims,
            same_aspect_ratio=same_aspect,
            same_format=(case_fmt == ref_fmt),
            same_mode=(case_mode == ref_mode),
            case_dimensions=ImageDimensions(width=case_w, height=case_h),
            reference_dimensions=ImageDimensions(width=ref_w, height=ref_h),
        ),
        pixel_difference=PixelDifferenceInfo(
            mean_absolute_difference=mean_abs_diff,
            max_pixel_difference=max_pixel_diff,
            changed_pixel_percentage=changed_pixel_pct,
            pixel_similarity=pixel_similarity,
            difference_threshold=DIFFERENCE_THRESHOLD,
        ),
        changed_regions=changed_regions,
        metadata_comparison=meta_comp,
        visualization=vis,
        assessment=assessment,
        assessment_label=ASSESSMENT_LABELS[assessment],
    )

    # Verify both source files' SHA-256 digests AFTER comparison
    case_sha256_after = hashlib.sha256(case_upload_path.read_bytes()).hexdigest()
    ref_sha256_after = hashlib.sha256(ref_upload_path.read_bytes()).hexdigest()
    if case_sha256_after != case_sha256_before or ref_sha256_after != ref_sha256_before:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Source file integrity check failed after comparison.",
        )

    updated_case = case_data.model_copy(
        update={
            "status": "compared",
            "comparison": comp_result,
        }
    )

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    comp_artifacts = []
    if comp_result.visualization and getattr(comp_result.visualization, "difference", None):
        comp_artifacts.append(
            {
                "artifact_type": "comparison_difference",
                "relative_path": f"storage/results/{case_id}/{comp_result.visualization.difference}",
                "media_type": "image/png",
            }
        )
    if comp_result.visualization and getattr(comp_result.visualization, "mask", None):
        comp_artifacts.append(
            {
                "artifact_type": "comparison_mask",
                "relative_path": f"storage/results/{case_id}/{comp_result.visualization.mask}",
                "media_type": "image/png",
            }
        )

    record_analysis_execution(
        case_data=updated_case,
        service_name="image_comparison_service",
        status="completed",
        input_sha256=updated_case.sha256,
        output_summary=(
            f"Reference comparison assessment={comp_result.assessment}; "
            f"changed_pixel_percentage={comp_result.pixel_difference.changed_pixel_percentage}%, "
            f"changed_regions={len(comp_result.changed_regions)}."
        ),
        artifact_specs=comp_artifacts,
        completed_at=comp_result.compared_at,
    )
    create_forensic_snapshot(updated_case, record_event=True)

    _atomic_save_case(updated_case, result_path)
    return updated_case


def get_comparison_artifact_path(case_id: str, artifact_name: str) -> Path:
    """
    Validate and return path to difference.png or difference_mask.png.
    Strictly blocks path traversal or arbitrary filenames.
    """
    if not CASE_ID_PATTERN.match(case_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID format.",
        )
    if artifact_name not in {"difference.png", "difference_mask.png"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported comparison artifact requested.",
        )

    artifact_path = settings.results_dir / case_id / "comparison" / artifact_name
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison artifact '{artifact_name}' not found for case '{case_id}'.",
        )
    return artifact_path
