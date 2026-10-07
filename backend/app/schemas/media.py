from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.analysis import (
    ImageAnalysisResult,
    ImageLocalizationResult,
    ManipulationDetectionResult,
)
from app.schemas.comparison import ImageComparisonResult, ReferenceImageMetadata
from app.schemas.evidence import EvidenceReport
from app.schemas.forensic_decision import ForensicDecisionResult
from app.schemas.history import EditHistoryReport
from app.schemas.reproducibility import (
    AnalysisRecord,
    ArtifactRecord,
    ForensicAnalysisSnapshot,
    ReportGenerationAuditRecord,
    SystemAuditEvent,
)
from app.schemas.video_analysis import VideoAnalysisResult
from app.schemas.audio_analysis import AudioAnalysisResult
from app.schemas.document_analysis import DocumentAnalysisResult


class MediaCaseMetadata(BaseModel):
    """Schema representing a forensic media case and its file metadata."""

    case_id: str = Field(..., description="Unique readable forensic case identifier (e.g. MT-YYYYMMDD-XXXXXXXX)")
    case_schema_version: Optional[str] = Field(
        default=None,
        description="Explicit case schema version (None for legacy pre-Step-14.3 cases)",
    )
    original_filename: str = Field(..., description="Original sanitized filename submitted by the user")
    stored_filename: str = Field(..., description="Filename saved inside storage/uploads/")
    media_type: str = Field(default="image", description="Media category (image, video, audio, or document)")
    extension: str = Field(..., description="Normalized lowercase file extension including leading dot")
    file_size: int = Field(..., ge=0, description="Exact file size in bytes")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA-256 hex digest of the uploaded file bytes")
    status: str = Field(default="uploaded", description="Current lifecycle state of the forensic case")
    created_at: str = Field(..., description="UTC ISO-8601 timestamp when the case was created")
    analysis: Optional[ImageAnalysisResult] = Field(
        default=None,
        description="Basic image forensic analysis results once analyzed (Step 3)",
    )
    manipulation_detection: Optional[ManipulationDetectionResult] = Field(
        default=None,
        description="Image manipulation and AI-generation detection results once evaluated (Step 4)",
    )
    localization: Optional[ImageLocalizationResult] = Field(
        default=None,
        description="Spatial anomaly localization results once evaluated (Step 5)",
    )
    reference_image: Optional[ReferenceImageMetadata] = Field(
        default=None,
        description="Reference/original image uploaded for comparison (Step 6)",
    )
    comparison: Optional[ImageComparisonResult] = Field(
        default=None,
        description="Original vs edited image comparison results once evaluated (Step 6)",
    )
    evidence: Optional[EvidenceReport] = Field(
        default=None,
        description="Correlated forensic evidence report once generated (Step 7)",
    )
    forensic_decision: Optional[ForensicDecisionResult] = Field(
        default=None,
        description="Multi-signal forensic interpretation & decision framework result once generated (Step 14.2)",
    )
    history: Optional[EditHistoryReport] = Field(
        default=None,
        description="Possible edit-history reconstruction report once generated (Step 8)",
    )
    video_analysis: Optional[VideoAnalysisResult] = Field(
        default=None,
        description="Video forensic analysis results once evaluated (Step 10)",
    )
    audio_analysis: Optional[AudioAnalysisResult] = Field(
        default=None,
        description="Audio forensic analysis results once evaluated (Step 11)",
    )
    document_analysis: Optional[DocumentAnalysisResult] = Field(
        default=None,
        description="Document forensic analysis results once evaluated (Step 12)",
    )
    analysis_registry: Optional[List[AnalysisRecord]] = Field(
        default=None,
        description="Formal registry of executed analysis operations (Step 14.3)",
    )
    artifact_registry: Optional[List[ArtifactRecord]] = Field(
        default=None,
        description="Formal registry of generated forensic artifacts (Step 14.3)",
    )
    audit_events: Optional[List[SystemAuditEvent]] = Field(
        default=None,
        description="System audit trail separated from Step 8 media edit history (Step 14.3)",
    )
    latest_snapshot: Optional[ForensicAnalysisSnapshot] = Field(
        default=None,
        description="Most recent immutable ForensicAnalysisSnapshot (Step 14.3)",
    )
    snapshots: Optional[List[ForensicAnalysisSnapshot]] = Field(
        default=None,
        description="Immutable collection of ForensicAnalysisSnapshots across reruns (Step 14.3)",
    )
    report_generations: Optional[List[ReportGenerationAuditRecord]] = Field(
        default=None,
        description="Audit records of generated JSON/PDF forensic reports (Step 14.3)",
    )


class MediaCaseListResponse(BaseModel):
    """Schema returning a collection of uploaded forensic cases."""

    total: int
    cases: List[MediaCaseMetadata]
