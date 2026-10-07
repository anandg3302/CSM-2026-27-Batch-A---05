from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.version import MEDIATRACE_APPLICATION_VERSION
from app.schemas.ai_detection import AiGenerationDetectionResult
from app.schemas.analysis import DetectionSignal, ForensicFinding, SuspiciousRegion
from app.schemas.comparison import ChangedRegion, MetadataComparisonInfo
from app.schemas.evidence import (
    AllowedEvidenceAssessment,
    EvidenceItem,
    EvidenceSummary,
)
from app.schemas.forensic_decision import ForensicDecisionResult
from app.schemas.history import (
    InferredEventItem,
    KnownFactItem,
    ReconstructedHistoryEvent,
)
from app.schemas.reproducibility import ForensicAnalysisSnapshot, SystemAuditEvent


class ReportCaseInfo(BaseModel):
    """Privacy-safe case metadata included in the forensic report (no server paths)."""

    case_id: str
    original_filename: str
    media_type: str
    extension: str
    file_size: int
    created_at: str
    status: str


class ReportIntegritySection(BaseModel):
    """Cryptographic chain-of-custody integrity section."""

    sha256: str = Field(..., min_length=64, max_length=64)
    hash_preserved: bool
    explanation: str = Field(
        default="The SHA-256 digest identifies the preserved original file."
    )
    scope_notice: str = Field(
        default=(
            "Hash integrity means the stored file has not changed. It does not prove "
            "that the image is authentic, unedited historically, captured by a claimed "
            "device, or that embedded metadata is truthful."
        )
    )


class ReportMediaSection(BaseModel):
    """Privacy-safe summary of Step 3 media and metadata properties."""

    available: bool
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: Optional[str] = None
    file_type: str
    exif_present: bool = False
    software_metadata_present: bool = False
    camera_metadata_present: bool = False
    timestamp_metadata_present: bool = False
    gps_metadata_present: bool = False
    gps_notice: Optional[str] = Field(
        default=None,
        description="Reported strictly as 'GPS metadata present' when GPS exists; raw coordinates are never included",
    )


class ReportForensicsSection(BaseModel):
    """Summary of Step 3 basic forensic observations without speculative conclusions."""

    available: bool
    analyzed_at: Optional[str] = None
    dimensions: Optional[str] = None
    color_mode: Optional[str] = None
    exif_summary: str
    camera_information: Optional[str] = None
    software_metadata: Optional[str] = None
    timestamps: Optional[str] = None
    jpeg_properties: Optional[str] = None
    findings: List[ForensicFinding] = Field(default_factory=list)


class ReportManipulationDetectionSection(BaseModel):
    """Complete meaningful output from Step 4 manipulation & AI-generation detection."""

    available: bool
    detector_status: str
    evaluated_at: Optional[str] = None
    overall_assessment: Optional[str] = None
    assessment_label: Optional[str] = None
    recompression_status: Optional[str] = None
    forensic_signals: List[DetectionSignal] = Field(default_factory=list)
    compression_observations: List[DetectionSignal] = Field(default_factory=list)
    metadata_observations: List[DetectionSignal] = Field(default_factory=list)
    ai_generation_status: str = Field(default="not_available")
    ai_generation_label: str = Field(
        default="AI-generated content detection: Not available"
    )
    ai_generation_explanation: str = Field(
        default="No trained AI-generation detector is currently configured."
    )
    ai_generation_detection: AiGenerationDetectionResult = Field(
        default_factory=AiGenerationDetectionResult,
        description="Structured Step 14.1 AI-generation detection report section",
    )



class ReportLocalizationSection(BaseModel):
    """Step 5 spatial anomaly localization results."""

    available: bool
    status: str
    evaluated_at: Optional[str] = None
    localized_anomaly: bool = False
    finding_label: str = Field(default="localized forensic anomaly")
    method: Optional[str] = None
    summary: Optional[str] = None
    regions: List[SuspiciousRegion] = Field(default_factory=list)
    heatmap_available: bool = False
    mask_available: bool = False
    disclaimer: str = Field(
        default=(
            "Spatial anomaly localization highlights regions with relative visual or compression "
            "inconsistencies (localized forensic anomaly) and does not confirm manipulation."
        )
    )


class ReportComparisonSection(BaseModel):
    """Step 6 reference vs case image comparison section."""

    available: bool
    status: str
    compared_at: Optional[str] = None
    reference_filename: Optional[str] = None
    reference_sha256: Optional[str] = None
    same_dimensions: Optional[bool] = None
    case_dimensions: Optional[str] = None
    reference_dimensions: Optional[str] = None
    changed_pixel_percentage: Optional[float] = None
    changed_region_count: int = 0
    changed_regions: List[ChangedRegion] = Field(default_factory=list)
    metadata_differences: Optional[MetadataComparisonInfo] = None
    difference_image_available: bool = False
    difference_mask_available: bool = False
    finding_label: str = Field(default="visual differences")
    assessment: Optional[str] = None
    assessment_label: Optional[str] = None
    limitation_notice: str = Field(
        default=(
            "Visual differences identify pixel changes between supplied files and may result from "
            "editing, recompression, resizing, format conversion, color processing, or other transformations."
        )
    )


class ReportCrossAnalysisItem(BaseModel):
    """Prominent representation of Step 7 cross-analysis spatial overlap."""

    id: str
    heading: str = Field(default="Cross-analysis spatial consistency")
    statement: str = Field(default="Independent analyses identify an overlapping region.")
    localization_region: str
    comparison_region: str
    overlap_percentage: float
    intersection_area: Optional[int] = None
    description: str
    strength: str


class ReportEvidenceSection(BaseModel):
    """Complete Step 7 Evidence Correlation output."""

    available: bool
    generated_at: Optional[str] = None
    assessment: AllowedEvidenceAssessment
    assessment_label: str
    summary: EvidenceSummary
    items: List[EvidenceItem] = Field(default_factory=list)
    cross_analysis_evidence: List[ReportCrossAnalysisItem] = Field(default_factory=list)
    disclaimer: str


class ReportHistorySection(BaseModel):
    """Complete Step 8 Possible Edit-History Reconstruction output."""

    title: str = Field(default="Possible Edit-History Reconstruction")
    sequence_label: str = Field(default="Evidence-ordered hypothesis")
    historical_accuracy_label: str = Field(default="Historical accuracy: Not determined")
    reconstructed_sequence: bool = Field(default=True)
    sequence_type: str = Field(default="evidence_ordered_hypothesis")
    historical_accuracy: str
    timeline_disclaimer: str = Field(
        default=(
            "This timeline is an evidence-ordered hypothesis based on available forensic observations. "
            "It is not a verified chronological history of the image."
        )
    )
    reference_comparison_disclaimer: str = Field(
        default=(
            "Reference comparison establishes differences between supplied files but does not "
            "independently establish which file was created first."
        )
    )
    events: List[ReconstructedHistoryEvent] = Field(default_factory=list)
    known_facts: List[KnownFactItem] = Field(default_factory=list)
    inferred_events: List[InferredEventItem] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ReportAssessmentSection(BaseModel):
    """Conservative overall assessment derived strictly from Step 7."""

    assessment: AllowedEvidenceAssessment
    assessment_label: str
    note: str = Field(
        default=(
            "Derived strictly from Step 7 Evidence Correlation. MediaTrace does not compute "
            "numerical manipulation probabilities, AI-generation probabilities, or authenticity scores."
        )
    )


class ForensicReport(BaseModel):
    """Structured Step 9 MediaTrace Digital Media Forensic Report."""

    report_id: str = Field(..., description="Unique report ID (e.g. MTR-YYYYMMDD-XXXXXXXX)")
    case_id: str = Field(..., description="Forensic case identifier")
    generated_at: str = Field(..., description="UTC ISO-8601 timestamp when report was generated")
    report_version: str = Field(default="1.0", description="Report schema version")
    application_version: str = Field(
        default=MEDIATRACE_APPLICATION_VERSION,
        description="MediaTrace application version",
    )
    case: ReportCaseInfo
    integrity: ReportIntegritySection
    media: ReportMediaSection
    forensics: ReportForensicsSection
    manipulation_detection: ReportManipulationDetectionSection
    ai_generation_detection: AiGenerationDetectionResult = Field(
        default_factory=AiGenerationDetectionResult,
        description="Dedicated Step 14.1 AI-generation detection section separated from classical forensics",
    )
    localization: ReportLocalizationSection
    comparison: ReportComparisonSection
    evidence: ReportEvidenceSection
    multi_signal_interpretation: Optional[ForensicDecisionResult] = Field(
        default=None,
        description="Dedicated Step 14.2 Multi-Signal Forensic Interpretation & Decision Framework section",
    )
    reproducibility_audit_trail: Optional[ForensicAnalysisSnapshot] = Field(
        default=None,
        description="Dedicated Step 14.3 Forensic Case Reproducibility & Audit Trail section",
    )
    audit_events: List[SystemAuditEvent] = Field(
        default_factory=list,
        description="Append-only system audit event log for Step 14.3 (strictly separate from media edit history)",
    )
    history: ReportHistorySection
    assessment: ReportAssessmentSection
    limitations: List[str] = Field(default_factory=list)
    final_notice: str = Field(
        default=(
            "The report summarizes and exports existing forensic observations. It does not "
            "independently establish authenticity, intent, authorship, or exact historical editing."
        )
    )

