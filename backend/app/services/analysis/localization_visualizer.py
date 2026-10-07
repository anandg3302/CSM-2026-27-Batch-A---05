from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

from app.core.config import settings
from app.schemas.analysis import LocalizationVisualization, SuspiciousRegion


def _score_to_heatmap_rgb(score: float) -> Tuple[int, int, int]:
    """
    Map a normalized anomaly score in [0.0, 1.0] to a classic jet/thermal RGB color:
    0.0 -> Deep Blue (15, 40, 160)
    0.5 -> Cyan/Green-Yellow (40, 210, 120)
    0.75 -> Orange (245, 150, 20)
    1.0 -> Bright Red (235, 25, 25)
    """
    clamped = max(0.0, min(1.0, score))
    if clamped < 0.35:
        t = clamped / 0.35
        r = int(15 + t * 25)
        g = int(40 + t * 150)
        b = int(160 + t * 50)
    elif clamped < 0.70:
        t = (clamped - 0.35) / 0.35
        r = int(40 + t * 205)
        g = int(190 + t * 20)
        b = int(210 - t * 180)
    else:
        t = (clamped - 0.70) / 0.30
        r = int(245 - t * 10)
        g = int(210 - t * 185)
        b = int(30 - t * 5)
    return (r, g, b)


def generate_localization_visualizations(
    case_id: str,
    original_img: Image.Image,
    anomaly_grid: List[List[float]],
    threshold: float,
    regions: List[SuspiciousRegion],
) -> LocalizationVisualization:
    """
    Create a heatmap overlay PNG and an anomaly mask PNG from the tile-level
    anomaly_grid without modifying the original image on disk.

    Saved under:
      storage/results/<case_id>/localization/heatmap.png
      storage/results/<case_id>/localization/mask.png
    """
    width, height = original_img.size
    rows = len(anomaly_grid)
    cols = len(anomaly_grid[0]) if rows > 0 else 0

    loc_dir: Path = settings.results_dir / case_id / "localization"
    loc_dir.mkdir(parents=True, exist_ok=True)

    # Build small grid images for fast bilinear upsampling to exact (width, height)
    grid_color = Image.new("RGB", (max(1, cols), max(1, rows)))
    grid_mask = Image.new("L", (max(1, cols), max(1, rows)))

    for r_idx in range(rows):
        for c_idx in range(cols):
            score = max(0.0, min(1.0, anomaly_grid[r_idx][c_idx]))
            grid_color.putpixel((c_idx, r_idx), _score_to_heatmap_rgb(score))
            mask_val = 255 if score >= threshold else int(score * 160)
            grid_mask.putpixel((c_idx, r_idx), mask_val)

    # Upsample smoothly to match exact original image dimensions
    upsampled_heatmap = grid_color.resize((width, height), resample=Image.Resampling.BILINEAR)
    upsampled_mask = grid_mask.resize((width, height), resample=Image.Resampling.NEAREST)

    # Blend original image (converted to RGB copy in memory) with heatmap overlay
    base_rgb = original_img.convert("RGB")
    blended = Image.blend(base_rgb, upsampled_heatmap, alpha=0.48)

    # Draw bounding boxes for candidate regions on the blended heatmap copy
    if regions:
        draw = ImageDraw.Draw(blended)
        for region in regions:
            x1 = max(0, min(width - 1, region.x))
            y1 = max(0, min(height - 1, region.y))
            x2 = max(x1, min(width - 1, region.x + region.width - 1))
            y2 = max(y1, min(height - 1, region.y + region.height - 1))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)

    heatmap_path = loc_dir / "heatmap.png"
    mask_path = loc_dir / "mask.png"

    blended.save(heatmap_path, format="PNG")
    upsampled_mask.save(mask_path, format="PNG")

    return LocalizationVisualization(
        heatmap="localization/heatmap.png",
        mask="localization/mask.png",
    )
