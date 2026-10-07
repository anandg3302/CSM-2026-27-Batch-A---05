from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ImageTechnicalInfo(BaseModel):
    """Technical properties extracted from the image container and header."""

    format: Optional[str] = Field(default=None, description="Container format (e.g. JPEG, PNG, WEBP)")
    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")
    mode: str = Field(..., description="Color mode (e.g. RGB, RGBA, L)")
    channels: Optional[int] = Field(default=None, description="Number of color channels")
    jpeg_encoding: Optional[str] = Field(
        default=None,
        description="JPEG encoding mode ('baseline' or 'progressive') when format is JPEG",
    )
    quantization_tables_count: Optional[int] = Field(
        default=None,
        description="Number of quantization tables present when format is JPEG",
    )


class ExifFields(BaseModel):
    """Normalized subset of EXIF metadata fields relevant to forensic intake."""

    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    datetime: Optional[str] = None
    datetime_original: Optional[str] = None
    orientation: Optional[str] = None


class ImageMetadataInfo(BaseModel):
    """EXIF and privacy-safe GPS summary extracted from the image."""

    exif_present: bool = Field(..., description="True if an EXIF metadata block was found")
    gps_present: bool = Field(..., description="True if GPS metadata tags exist in EXIF")
    gps_info: Optional[str] = Field(
        default=None,
        description="Privacy-safe GPS indicator ('GPS metadata present' when present, otherwise null)",
    )
    fields: ExifFields = Field(default_factory=ExifFields)


class ForensicFinding(BaseModel):
    """Single objective forensic observation extracted from the file."""

    category: str = Field(..., description="Category: file, image, metadata, or compression")
    finding: str = Field(..., description="Neutral description of the observed forensic property")
    value: str = Field(..., description="Observed value associated with the finding")
    severity: str = Field(default="informational", description="Severity level (e.g. informational)")


class ImageAnalysisResult(BaseModel):
    """Complete basic forensic analysis payload attached to an image case (Step 3)."""

    analyzed_at: str = Field(..., description="UTC ISO-8601 timestamp when analysis completed")
    image: ImageTechnicalInfo
    metadata: ImageMetadataInfo
    findings: List[ForensicFinding] = Field(default_factory=list)


AllowedOverallAssessment = Literal[
    "no_strong_indicators",
    "potential_manipulation",
    "insufficient_evidence",
]

AllowedSignalStrength = Literal[
    "informational",
    "weak",
    "moderate",
]


class DetectionSignal(BaseModel):
    """Individual heuristic or structural indicator observed during manipulation detection."""

    type: str = Field(..., description="Signal domain: consistency, compression, metadata, or recompression")
    name: str = Field(..., description="Machine-readable identifier for the forensic signal")
    description: str = Field(..., description="Objective explanation of the observed signal and its forensic meaning")
    value: str = Field(..., description="Observed value or state from the image")
    strength: AllowedSignalStrength = Field(
        ...,
        description="Strength of the indicator ('informational', 'weak', or 'moderate')",
    )


from app.schemas.ai_detection import AiGenerationDetectionResult, AiTransformationContext


class ClassicalForensicsSection(BaseModel):
    """Structured classical image forensics section (metadata, JPEG compression, consistency)."""

    overall_assessment: AllowedOverallAssessment
    assessment_label: str
    recompression_status: str = Field(default="Not evaluated")
    metadata_signals: List[DetectionSignal] = Field(default_factory=list)
    compression_signals: List[DetectionSignal] = Field(default_factory=list)
    consistency_signals: List[DetectionSignal] = Field(default_factory=list)
    signals: List[DetectionSignal] = Field(default_factory=list)


class CombinedForensicAssessmentSection(BaseModel):
    """Separated classical vs AI detector assessment without any fabricated numerical probability."""

    overall_assessment: AllowedOverallAssessment
    assessment_label: str
    classical_forensic_observations: str
    ai_detector_output: str
    combined_interpretation: str
    cross_analysis_interpretation: Optional[str] = Field(
        default=None,
        description="Explicit neutral cross-analysis statement when AI detector output and classical/localization findings disagree",
    )
    transformation_context: Optional[AiTransformationContext] = Field(
        default=None,
        description="Documented transformation context from case evidence",
    )



class ManipulationDetectionResult(BaseModel):
    """Structured result for Step 4 & Step 14.1 Image Manipulation + AI-Generated Detection."""

    status: str = Field(default="completed", description="Execution state of the detection pipeline")
    evaluated_at: str = Field(..., description="UTC ISO-8601 timestamp when detection completed")
    overall_assessment: AllowedOverallAssessment = Field(
        ...,
        description="Conservative forensic assessment ('no_strong_indicators', 'potential_manipulation', or 'insufficient_evidence')",
    )
    assessment_label: str = Field(
        ...,
        description="Human-readable conservative interpretation of the overall assessment",
    )
    recompression_status: str = Field(
        default="Not evaluated",
        description="Double-compression / recompression evaluation state",
    )
    signals: List[DetectionSignal] = Field(default_factory=list)
    ai_generation: AiGenerationDetectionResult = Field(
        default_factory=AiGenerationDetectionResult
    )
    classical_forensics: Optional[ClassicalForensicsSection] = Field(
        default=None,
        description="Clearly separated classical forensic findings (metadata, compression, consistency)",
    )
    ai_generation_detection: AiGenerationDetectionResult = Field(
        default_factory=AiGenerationDetectionResult,
        description="Clearly separated real AI-generated image detection result (Step 14.1)",
    )
    assessment: Optional[CombinedForensicAssessmentSection] = Field(
        default=None,
        description="Distinguished summary of classical observations, AI detector output, and combined interpretation",
    )


class ImageDimensions(BaseModel):
    """Width and height of the analyzed image in pixels."""

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class SuspiciousRegion(BaseModel):
    """Bounding box and anomaly score statistics for a candidate region."""

    id: str = Field(..., description="Unique region identifier within the case (e.g. region-1)")
    x: int = Field(..., ge=0, description="Left coordinate in original image pixels")
    y: int = Field(..., ge=0, description="Top coordinate in original image pixels")
    width: int = Field(..., ge=1, description="Region width in pixels")
    height: int = Field(..., ge=1, description="Region height in pixels")
    area: int = Field(..., ge=1, description="Region area in pixels (width * height)")
    mean_anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean normalized anomaly score (0.0 to 1.0) across the candidate region",
    )
    max_anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum normalized anomaly score (0.0 to 1.0) within the candidate region",
    )


class LocalizationSignal(BaseModel):
    """Explainable baseline signal contributing to spatial localization."""

    name: str = Field(..., description="Signal identifier (e.g. local_noise_inconsistency, jpeg_recompression_difference)")
    description: str = Field(..., description="Explanation of the spatial signal")
    strength: AllowedSignalStrength = Field(..., description="Signal strength ('informational', 'weak', or 'moderate')")


class LocalizationVisualization(BaseModel):
    """Relative paths to generated heatmap and mask images under storage/results/<case_id>/."""

    heatmap: str = Field(..., description="Relative path to heatmap overlay PNG (e.g. localization/heatmap.png)")
    mask: str = Field(..., description="Relative path to binary/grayscale anomaly mask PNG (e.g. localization/mask.png)")


class ImageLocalizationResult(BaseModel):
    """Structured result for Step 5 Image Manipulation Localization."""

    status: Literal["completed", "insufficient_data"] = Field(
        default="completed",
        description="Execution state of localization ('completed' or 'insufficient_data')",
    )
    localized_at: str = Field(..., description="UTC ISO-8601 timestamp when localization completed")
    method: str = Field(
        default="classical_localization_baseline",
        description="Identifier of the localization algorithm",
    )
    localized_anomaly: bool = Field(
        ...,
        description="True if one or more candidate regions exceeded the conservative anomaly threshold",
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Conservative anomaly score cutoff used to segment candidate regions",
    )
    image_dimensions: ImageDimensions
    regions: List[SuspiciousRegion] = Field(default_factory=list)
    signals: List[LocalizationSignal] = Field(default_factory=list)
    visualization: Optional[LocalizationVisualization] = None
    message: Optional[str] = None
    disclaimer: str = Field(
        default="Localized anomalies are forensic signals and do not by themselves prove image manipulation.",
        description="Forensic interpretation disclaimer",
    )
