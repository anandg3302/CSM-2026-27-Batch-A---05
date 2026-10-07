from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from app.schemas.analysis import ImageDimensions

AllowedComparisonAssessment = Literal[
    "no_significant_difference",
    "visual_differences_detected",
    "localized_visual_differences_detected",
    "different_dimensions",
    "insufficient_data",
]


class ReferenceImageMetadata(BaseModel):
    """Metadata for a reference/original image uploaded for comparison against a case image."""

    original_filename: str = Field(..., description="Sanitized original filename of the reference image")
    stored_filename: str = Field(..., description="Filename saved inside storage/uploads/ (<case_id>_reference_<filename>)")
    extension: str = Field(..., description="Normalized lowercase extension")
    file_size: int = Field(..., ge=1, description="File size in bytes")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA-256 hex digest of reference image bytes")
    format: str = Field(..., description="Decoded image container format (JPEG, PNG, WEBP)")
    width: int = Field(..., ge=1, description="Reference image width in pixels")
    height: int = Field(..., ge=1, description="Reference image height in pixels")
    uploaded_at: str = Field(..., description="UTC ISO-8601 timestamp when reference image was uploaded")


class ComparisonEndpointSummary(BaseModel):
    """Summary of an image participating in a comparison."""

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    format: Optional[str] = None
    mode: Optional[str] = None
    sha256: str = Field(..., min_length=64, max_length=64)


class DimensionComparisonInfo(BaseModel):
    """Structural and dimension comparison between case and reference images."""

    same: bool = Field(..., description="True if width and height are identical")
    same_dimensions: bool = Field(..., description="True if width and height are identical")
    same_aspect_ratio: bool = Field(..., description="True if aspect ratios match within tolerance")
    same_format: bool = Field(..., description="True if image container formats match")
    same_mode: bool = Field(..., description="True if color modes (e.g. RGB, RGBA) match")
    case_dimensions: ImageDimensions
    reference_dimensions: ImageDimensions


class PixelDifferenceInfo(BaseModel):
    """Aggregate pixel-level difference metrics."""

    mean_absolute_difference: float = Field(..., ge=0.0, le=255.0)
    max_pixel_difference: int = Field(..., ge=0, le=255)
    changed_pixel_percentage: float = Field(..., ge=0.0, le=100.0)
    pixel_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized pixel similarity score (1.0 = identical pixel values)",
    )
    difference_threshold: int = Field(
        default=10,
        description="Documented per-channel intensity delta above which a pixel is counted as changed",
    )


class ChangedRegion(BaseModel):
    """Bounding box and pixel statistics for a connected region of visual differences."""

    id: str = Field(..., description="Changed region identifier (e.g. change-1)")
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    area: int = Field(..., ge=1)
    changed_pixel_count: int = Field(..., ge=1)
    changed_pixel_percentage: float = Field(..., ge=0.0, le=100.0)


class MetadataComparisonInfo(BaseModel):
    """Privacy-safe metadata comparison between the case image and reference image."""

    format_changed: bool
    dimensions_changed: bool
    exif_presence_changed: bool
    software_changed: bool
    camera_changed: bool
    timestamp_changed: bool
    case_software: Optional[str] = None
    reference_software: Optional[str] = None


class ComparisonVisualization(BaseModel):
    """Relative paths to generated comparison visual artifacts under storage/results/<case_id>/."""

    difference: str = Field(default="comparison/difference.png")
    mask: str = Field(default="comparison/difference_mask.png")


class ImageComparisonResult(BaseModel):
    """Structured result for Step 6 Original vs Edited Image Comparison."""

    status: str = Field(default="completed")
    compared_at: str = Field(..., description="UTC ISO-8601 timestamp when comparison completed")
    comparison_mode: Literal["direct", "normalized_resize"] = Field(
        default="direct",
        description="'direct' when dimensions match; 'normalized_resize' when reference was resized for visual alignment",
    )
    case_image: ComparisonEndpointSummary
    reference_image: ComparisonEndpointSummary
    dimensions: DimensionComparisonInfo
    pixel_difference: PixelDifferenceInfo
    changed_regions: List[ChangedRegion] = Field(default_factory=list)
    metadata_comparison: MetadataComparisonInfo
    visualization: ComparisonVisualization
    assessment: AllowedComparisonAssessment
    assessment_label: str
    disclaimer: str = Field(
        default="Pixel differences indicate that the images differ but do not by themselves prove intentional manipulation."
    )
