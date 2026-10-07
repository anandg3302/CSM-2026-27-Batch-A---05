from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
from typing import List, Tuple

from fastapi import HTTPException, status
from PIL import Image, ImageChops, ImageFilter, ImageStat, UnidentifiedImageError

from app.core.config import settings
from app.schemas.analysis import (
    ImageDimensions,
    ImageLocalizationResult,
    LocalizationSignal,
    SuspiciousRegion,
)
from app.schemas.media import MediaCaseMetadata
from app.services.analysis import image_analysis_service
from app.services.analysis.localization_visualizer import (
    generate_localization_visualizations,
)
from app.services.media_service import CASE_ID_PATTERN

# Minimum dimensions (pixels) required for meaningful tile-based spatial localization
MIN_LOCALIZATION_DIMENSION = 64

# Documented conservative anomaly score threshold (0.0 -> 1.0).
# A tile must exceed both this normalized threshold and a minimum physical signal contrast
# above the image's background median to prevent uniform numerical noise from triggering regions.
DEFAULT_ANOMALY_THRESHOLD = 0.75

# Minimum physical contrast floors (in 0-255 pixel intensity units) required to treat
# spatial variation as a meaningful forensic signal rather than uniform sensor/rounding noise.
MIN_NOISE_SPREAD_FLOOR = 2.5
MIN_ELA_SPREAD_FLOOR = 2.0


def _choose_tile_size(width: int, height: int) -> int:
    """Select an adaptive tile size in pixels based on image resolution."""
    min_dim = min(width, height)
    if min_dim >= 512:
        return 32
    if min_dim >= 160:
        return 16
    return 8


def _compute_tile_maps(
    img: Image.Image,
    tile_size: int,
) -> Tuple[List[List[float]], float, float]:
    """
    Compute two explainable spatial grids across (rows x cols) tiles:
      1. Local high-frequency noise inconsistency (deviation from global median tile noise)
      2. Controlled JPEG recompression difference (in-memory Q=90 error level deviation)

    Returns:
      (combined_anomaly_grid, max_noise_contrast, max_ela_contrast)
    """
    width, height = img.size
    rgb_img = img.convert("RGB")
    gray_img = rgb_img.convert("L")

    # 1. High-pass residual image for local noise estimation
    smoothed_gray = gray_img.filter(ImageFilter.BoxBlur(1))
    noise_residual_img = ImageChops.difference(gray_img, smoothed_gray)

    # 2. Controlled in-memory JPEG recompression (Quality = 90)
    recomp_buffer = io.BytesIO()
    rgb_img.save(recomp_buffer, format="JPEG", quality=90)
    recomp_buffer.seek(0)
    with Image.open(recomp_buffer) as recomp_jpeg:
        ela_diff_img = ImageChops.difference(rgb_img, recomp_jpeg.convert("RGB"))

    cols = max(1, (width + tile_size - 1) // tile_size)
    rows = max(1, (height + tile_size - 1) // tile_size)

    raw_noise_grid: List[List[float]] = [[0.0] * cols for _ in range(rows)]
    raw_ela_grid: List[List[float]] = [[0.0] * cols for _ in range(rows)]
    flat_noise: List[float] = []
    flat_ela: List[float] = []

    for r_idx in range(rows):
        y0 = r_idx * tile_size
        y1 = min(height, y0 + tile_size)
        for c_idx in range(cols):
            x0 = c_idx * tile_size
            x1 = min(width, x0 + tile_size)
            box = (x0, y0, x1, y1)

            # Local noise statistic: combine high-pass mean residual + grayscale stddev
            res_stat = ImageStat.Stat(noise_residual_img.crop(box))
            gray_stat = ImageStat.Stat(gray_img.crop(box))
            noise_val = float(res_stat.mean[0]) + 0.35 * float(gray_stat.stddev[0])

            # Local recompression difference statistic: mean across RGB channels
            ela_stat = ImageStat.Stat(ela_diff_img.crop(box))
            ela_val = sum(float(m) for m in ela_stat.mean) / 3.0

            raw_noise_grid[r_idx][c_idx] = noise_val
            raw_ela_grid[r_idx][c_idx] = ela_val
            flat_noise.append(noise_val)
            flat_ela.append(ela_val)

    median_noise = statistics.median(flat_noise) if flat_noise else 0.0
    median_ela = statistics.median(flat_ela) if flat_ela else 0.0

    # Measure absolute deviation from the background median
    noise_dev_grid: List[List[float]] = [[0.0] * cols for _ in range(rows)]
    ela_dev_grid: List[List[float]] = [[0.0] * cols for _ in range(rows)]
    max_noise_dev = 0.0
    max_ela_dev = 0.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            n_dev = abs(raw_noise_grid[r_idx][c_idx] - median_noise)
            e_dev = abs(raw_ela_grid[r_idx][c_idx] - median_ela)
            noise_dev_grid[r_idx][c_idx] = n_dev
            ela_dev_grid[r_idx][c_idx] = e_dev
            if n_dev > max_noise_dev:
                max_noise_dev = n_dev
            if e_dev > max_ela_dev:
                max_ela_dev = e_dev

    # Normalize with physical floors so uniform/flat images stay near 0.0
    norm_noise_denom = max(max_noise_dev, MIN_NOISE_SPREAD_FLOOR)
    norm_ela_denom = max(max_ela_dev, MIN_ELA_SPREAD_FLOOR)

    # Dampen scores when the entire image has almost zero physical variation
    noise_activity_gain = min(1.0, max_noise_dev / MIN_NOISE_SPREAD_FLOOR)
    ela_activity_gain = min(1.0, max_ela_dev / MIN_ELA_SPREAD_FLOOR)

    combined_grid: List[List[float]] = [[0.0] * cols for _ in range(rows)]
    for r_idx in range(rows):
        for c_idx in range(cols):
            n_norm = (noise_dev_grid[r_idx][c_idx] / norm_noise_denom) * noise_activity_gain
            e_norm = (ela_dev_grid[r_idx][c_idx] / norm_ela_denom) * ela_activity_gain
            # Combine both signals and retain strong single-channel localized anomalies
            blended = 0.55 * n_norm + 0.45 * e_norm
            peak_assisted = max(blended, 0.88 * max(n_norm, e_norm))
            combined_grid[r_idx][c_idx] = round(max(0.0, min(1.0, peak_assisted)), 4)

    return combined_grid, max_noise_dev, max_ela_dev


def _extract_candidate_regions(
    anomaly_grid: List[List[float]],
    tile_size: int,
    img_width: int,
    img_height: int,
    threshold: float,
) -> List[SuspiciousRegion]:
    """
    Extract bounding boxes of 8-connected components where anomaly_score >= threshold.
    Filters out tiny regions below MIN_REGION_AREA_PX (at least 0.5% of image area or 256 px).
    """
    rows = len(anomaly_grid)
    cols = len(anomaly_grid[0]) if rows > 0 else 0
    visited = [[False] * cols for _ in range(rows)]

    # Documented minimum region area filter: at least 0.5% of total image area (or 256 px),
    # and never smaller than 1 tile.
    min_area_px = max(tile_size * tile_size, min(256, int(0.005 * img_width * img_height)))

    raw_regions: List[SuspiciousRegion] = []

    for r_start in range(rows):
        for c_start in range(cols):
            if visited[r_start][c_start] or anomaly_grid[r_start][c_start] < threshold:
                continue

            # BFS flood-fill over 8-connected anomalous tiles
            queue = deque([(r_start, c_start)])
            visited[r_start][c_start] = True
            component_scores: List[float] = []
            min_r, max_r = r_start, r_start
            min_c, max_c = c_start, c_start

            while queue:
                curr_r, curr_c = queue.popleft()
                component_scores.append(anomaly_grid[curr_r][curr_c])
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
                            and anomaly_grid[nr][nc] >= threshold
                        ):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            x0 = max(0, min_c * tile_size)
            y0 = max(0, min_r * tile_size)
            x1 = min(img_width, (max_c + 1) * tile_size)
            y1 = min(img_height, (max_r + 1) * tile_size)

            box_w = max(1, x1 - x0)
            box_h = max(1, y1 - y0)
            box_area = box_w * box_h

            # Reject isolated single-tile speckles if the image has many tiles and area < min_area_px
            if box_area < min_area_px or (len(component_scores) == 1 and rows * cols > 16 and max(component_scores) < 0.88):
                continue

            mean_score = round(sum(component_scores) / len(component_scores), 3)
            max_score = round(max(component_scores), 3)

            raw_regions.append(
                SuspiciousRegion(
                    id=f"region-{len(raw_regions) + 1}",
                    x=x0,
                    y=y0,
                    width=box_w,
                    height=box_h,
                    area=box_area,
                    mean_anomaly_score=mean_score,
                    max_anomaly_score=max_score,
                )
            )

    # Sort regions by max_anomaly_score descending and re-index IDs cleanly
    raw_regions.sort(key=lambda r: (r.max_anomaly_score, r.area), reverse=True)
    for idx, reg in enumerate(raw_regions, start=1):
        reg.id = f"region-{idx}"

    return raw_regions


def localize_image_manipulation(case_id: str) -> MediaCaseMetadata:
    """
    Run Step 5 baseline spatial manipulation localization on an image case.
    Preserves Step 3 and Step 4 results, verifies original file SHA-256 before and after,
    writes heatmap.png and mask.png under storage/results/<case_id>/localization/,
    and updates storage/results/<case_id>.json with status='localized'.
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

    # Ensure Step 3 analysis exists without overwriting existing Step 3 or Step 4 data
    if case_data.analysis is None:
        refreshed = image_analysis_service.analyze_image(case_id)
        case_data = case_data.model_copy(update={"analysis": refreshed.analysis})

    # Verify original file SHA-256 BEFORE localization
    file_bytes = upload_path.read_bytes()
    sha256_before = hashlib.sha256(file_bytes).hexdigest()
    if sha256_before != case_data.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="File integrity check failed before localization: stored image SHA-256 does not match case record.",
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.load()
            width, height = img.size
            img_format = (img.format or "UNKNOWN").upper()

            if width < MIN_LOCALIZATION_DIMENSION or height < MIN_LOCALIZATION_DIMENSION:
                loc_result = ImageLocalizationResult(
                    status="insufficient_data",
                    localized_at=datetime.now(timezone.utc).isoformat(),
                    method="classical_localization_baseline",
                    localized_anomaly=False,
                    threshold=DEFAULT_ANOMALY_THRESHOLD,
                    image_dimensions=ImageDimensions(width=width, height=height),
                    regions=[],
                    signals=[],
                    visualization=None,
                    message="Image dimensions are insufficient for reliable baseline localization.",
                )
            else:
                tile_size = _choose_tile_size(width, height)
                anomaly_grid, max_noise_dev, max_ela_dev = _compute_tile_maps(img, tile_size)
                regions = _extract_candidate_regions(
                    anomaly_grid=anomaly_grid,
                    tile_size=tile_size,
                    img_width=width,
                    img_height=height,
                    threshold=DEFAULT_ANOMALY_THRESHOLD,
                )
                localized_anomaly = len(regions) > 0

                vis_paths = generate_localization_visualizations(
                    case_id=case_id,
                    original_img=img,
                    anomaly_grid=anomaly_grid,
                    threshold=DEFAULT_ANOMALY_THRESHOLD,
                    regions=regions,
                )

                noise_strength = (
                    "moderate"
                    if localized_anomaly and max_noise_dev >= MIN_NOISE_SPREAD_FLOOR * 2.0
                    else "weak"
                    if max_noise_dev >= MIN_NOISE_SPREAD_FLOOR
                    else "informational"
                )
                ela_strength = (
                    "moderate"
                    if localized_anomaly and max_ela_dev >= MIN_ELA_SPREAD_FLOOR * 2.0
                    else "weak"
                    if max_ela_dev >= MIN_ELA_SPREAD_FLOOR
                    else "informational"
                )

                recomp_note = (
                    "Localized differences were observed after controlled in-memory JPEG recompression (Q=90)."
                    if img_format == "JPEG"
                    else f"Controlled JPEG recompression difference heuristic evaluated on {img_format} container (heuristic signal only, not proof of editing)."
                )

                signals = [
                    LocalizationSignal(
                        name="local_noise_inconsistency",
                        description=f"Evaluated block-level high-frequency residual variance across {tile_size}x{tile_size} px tiles.",
                        strength=noise_strength,
                    ),
                    LocalizationSignal(
                        name="jpeg_recompression_difference",
                        description=recomp_note,
                        strength=ela_strength,
                    ),
                ]

                loc_result = ImageLocalizationResult(
                    status="completed",
                    localized_at=datetime.now(timezone.utc).isoformat(),
                    method="classical_localization_baseline",
                    localized_anomaly=localized_anomaly,
                    threshold=DEFAULT_ANOMALY_THRESHOLD,
                    image_dimensions=ImageDimensions(width=width, height=height),
                    regions=regions,
                    signals=signals,
                    visualization=vis_paths,
                    message=(
                        f"{len(regions)} potentially suspicious region(s) localized above anomaly threshold {DEFAULT_ANOMALY_THRESHOLD}."
                        if localized_anomaly
                        else "No localized anomaly detected by the baseline forensic method."
                    ),
                )
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid decodable image container.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run spatial localization on image.",
        ) from exc

    # Verify original file SHA-256 AFTER localization to guarantee immutability
    sha256_after = hashlib.sha256(upload_path.read_bytes()).hexdigest()
    if sha256_after != case_data.sha256:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Integrity violation detected after localization.",
        )

    updated_case = case_data.model_copy(
        update={
            "status": "localized",
            "localization": loc_result,
        }
    )

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    loc_artifacts = []
    if loc_result.visualization and getattr(loc_result.visualization, "heatmap", None):
        loc_artifacts.append(
            {
                "artifact_type": "localization_heatmap",
                "relative_path": f"storage/results/{case_id}/{loc_result.visualization.heatmap}",
                "media_type": "image/png",
            }
        )
    if loc_result.visualization and getattr(loc_result.visualization, "mask", None):
        loc_artifacts.append(
            {
                "artifact_type": "localization_mask",
                "relative_path": f"storage/results/{case_id}/{loc_result.visualization.mask}",
                "media_type": "image/png",
            }
        )

    record_analysis_execution(
        case_data=updated_case,
        service_name="image_localization_service",
        status="completed",
        input_sha256=updated_case.sha256,
        output_summary=(
            f"Localized anomaly={loc_result.localized_anomaly}; candidate regions={len(loc_result.regions)}."
        ),
        artifact_specs=loc_artifacts,
        completed_at=getattr(loc_result, "localized_at", None),
    )
    create_forensic_snapshot(updated_case, record_event=True)

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
            detail="Failed to persist localization results.",
        ) from exc

    return updated_case


def get_localization_artifact_path(case_id: str, artifact_name: str) -> Path:
    """Resolve and validate path to heatmap.png or mask.png for a given case."""
    if not CASE_ID_PATTERN.match(case_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID format.",
        )
    if artifact_name not in {"heatmap.png", "mask.png"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported localization artifact requested.",
        )

    artifact_path = settings.results_dir / case_id / "localization" / artifact_name
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Localization artifact '{artifact_name}' not found for case '{case_id}'.",
        )
    return artifact_path
