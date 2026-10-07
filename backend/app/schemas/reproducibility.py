from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.core.version import (
    MEDIATRACE_APPLICATION_NAME,
    MEDIATRACE_APPLICATION_VERSION,
    SCHEMA_VERSIONS,
    SERVICE_VERSIONS,
)

AllowedIntegrityStatus = Literal["verified", "mismatch", "unavailable"]

AllowedAnalysisRecordStatus = Literal[
    "completed",
    "unavailable",
    "failed",
    "skipped",
    "reused",
]

AllowedReproducibilityStatus = Literal[
    "reproducible",
    "partially_reproducible",
    "not_reproducible",
    "cannot_determine",
]

AllowedAuditEventType = Literal[
    "case_created",
    "case_uploaded",
    "media_uploaded",
    "integrity_verified",
    "analysis_started",
    "analysis_completed",
    "analysis_rerun",
    "analysis_failed",
    "analysis_reused",
    "artifact_created",
    "forensic_interpretation_created",
    "snapshot_created",
    "report_generated",
    "report_pdf_generated",
    "reproducibility_checked",
    "integrity_mismatch_detected",
]

DEFAULT_REPRODUCIBILITY_LIMITATIONS: List[str] = [
    "Reproducibility means the same stored input bytes, deterministic configuration, and pinned model revision produce consistent technical outputs.",
    "Reproducibility does NOT prove that the media is authentic, manipulated, AI-generated, or human-created.",
    "Reproducibility does NOT calibrate AI detector scores or turn weak/correlated forensic signals into proof.",
    "System audit events record when MediaTrace executed an operation and are strictly separate from Step 8 media edit history reconstruction.",
]


class InputFileProvenance(BaseModel):
    """Exact provenance of the input file bytes analyzed by MediaTrace."""

    case_id: str
    original_filename: str
    safe_filename: str = ""
    relative_storage_path: str = Field(
        default="",
        description="Privacy-safe normalized storage path (e.g. 'storage/uploads/MT-..._file.jpg')",
    )
    storage_path: Optional[str] = None
    media_type: str
    extension: str = ""
    file_extension: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: int = Field(..., ge=0)
    sha256: str = Field(..., min_length=64, max_length=64)
    original_input_sha256: Optional[str] = None
    uploaded_at: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


class InputIntegrityVerification(BaseModel):
    """Verification comparing current stored original file bytes against recorded original SHA-256."""

    status: AllowedIntegrityStatus
    original_sha256: str = Field(..., min_length=64, max_length=64)
    recomputed_sha256: Optional[str] = None
    original_byte_size: int = Field(..., ge=0)
    recomputed_byte_size: Optional[int] = None
    checked_at: str
    hash_match: bool = False
    byte_size_match: bool = False
    detail: str = ""


class AnalysisEnvironmentInfo(BaseModel):
    """Sanitized runtime environment metadata without usernames, secrets, or local user paths."""

    application_name: str = Field(default=MEDIATRACE_APPLICATION_NAME)
    application_version: str = Field(default=MEDIATRACE_APPLICATION_VERSION)
    python_version: str
    platform_category: str = "unknown"
    execution_device: str = "cpu"
    onnxruntime_version: Optional[str] = None
    pillow_version: Optional[str] = None
    schema_versions: Dict[str, str] = Field(
        default_factory=lambda: dict(SCHEMA_VERSIONS)
    )
    service_versions: Dict[str, str] = Field(
        default_factory=lambda: dict(SERVICE_VERSIONS)
    )


class AiModelProvenanceRecord(BaseModel):
    """Immutable AI model provenance record preserving pinned revision and artifact hashes."""

    available: bool = False
    status: str = "not_available"
    model_name: str = Field(default="umm-maybe/AI-image-detector")
    pinned_revision: str = Field(default="c7e223baf11bc40528af364ba7bdea030ef42f9e")
    architecture: str = Field(default="SwinForImageClassification")
    framework: Optional[str] = None
    device: str = "cpu"
    weights_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 digest of upstream pytorch_model.bin when present on disk",
    )
    onnx_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 digest of local runtime model.onnx when present on disk",
    )
    class_mapping: Dict[str, str] = Field(
        default_factory=lambda: {"0": "artificial", "1": "human"}
    )
    preprocessing_parameters: Dict[str, Any] = Field(default_factory=dict)
    score_type: Literal["uncalibrated_model_score"] = "uncalibrated_model_score"
    decision_rule: Literal["model_class_selection"] = "model_class_selection"
    threshold: Optional[float] = None
    selected_class: Optional[str] = None
    artificial_score: Optional[float] = None
    human_score: Optional[float] = None
    license: Optional[str] = None
    training_domain: Optional[str] = None
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    real_score: Optional[float] = None
    fake_score: Optional[float] = None
    frames_sampled: Optional[int] = None
    usable_detector_frames: Optional[int] = None
    detector_scope: Optional[str] = None


class AnalysisRecord(BaseModel):
    """Formal execution record for a single MediaTrace analysis stage."""

    analysis_id: str = Field(..., description="Unique analysis execution ID (e.g. ANL-YYYYMMDD-XXXXXXXX)")
    service_name: str
    service_version: str
    status: AllowedAnalysisRecordStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    input_sha256: str
    configuration: Dict[str, Any] = Field(default_factory=dict)
    configuration_hash: str
    model_provenance: Optional[AiModelProvenanceRecord] = None
    output_summary: str = ""
    artifact_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    was_reused: bool = False
    reused_analysis_id: Optional[str] = None
    rerun_of_analysis_id: Optional[str] = None
    configuration_changed: bool = False
    model_changed: bool = False


class ArtifactRecord(BaseModel):
    """Formal registry record for a generated forensic artifact with immutable SHA-256 tracking."""

    artifact_id: str = Field(..., description="Unique artifact ID (MTA-YYYYMMDD-XXXXXXXX)")
    case_id: str
    artifact_type: str
    relative_path: str = Field(
        ...,
        description="Normalized relative path under storage/ (never exposing absolute OS paths)",
    )
    immutable_archive_path: Optional[str] = Field(
        default=None,
        description="Versioned immutable archive path preserving artifact bytes across reruns",
    )
    media_type: str
    sha256: str = Field(..., min_length=64, max_length=64)
    derived_artifact_sha256: str = Field(..., min_length=64, max_length=64)
    byte_size: int = Field(..., ge=0)
    created_at: str
    source_analysis_id: str
    source_service: str
    superseded_by_artifact_id: Optional[str] = None


class EvidenceDependencyNode(BaseModel):
    """Traceability node linking input bytes -> analysis -> artifacts -> evidence -> interpretation -> report."""

    node_id: str
    node_type: Literal[
        "input_bytes",
        "analysis_execution",
        "derived_artifact",
        "evidence_item",
        "forensic_interpretation",
        "forensic_report",
    ]
    label: str
    source_service: str
    sha256: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)


class SystemAuditEvent(BaseModel):
    """
    System audit event recording what MediaTrace did (strictly separated from Step 8 media edit history).
    """

    event_id: str = Field(..., description="Unique audit event ID (AUD-YYYYMMDD-XXXXXXXX)")
    case_id: str
    event_type: AllowedAuditEventType
    event_category: Literal["system_audit_event"] = "system_audit_event"
    timestamp: str
    summary: str
    analysis_id: Optional[str] = None
    artifact_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    report_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    scope_notice: str = Field(
        default=(
            "This system audit event records when MediaTrace executed an operation. "
            "It is strictly separate from Step 8 media edit history reconstruction."
        )
    )


class ReproducibilityCheckResult(BaseModel):
    """Non-destructive reproducibility and provenance verification result for a case."""

    case_id: str
    checked_at: str
    status: AllowedReproducibilityStatus
    input_file_present: bool = False
    input_sha256_match: bool = False
    analysis_records_present: bool = False
    historical_analysis_registry: Literal["available", "unavailable"] = "available"
    configuration_hash_match: bool = True
    model_reference_match: bool = True
    artifacts_present: bool = True
    artifact_hashes_match: bool = True
    schema_versions_match: bool = True
    service_versions_match: bool = True
    reasons: List[str] = Field(default_factory=list)
    verification_notes: List[str] = Field(default_factory=list)


class ReportGenerationAuditRecord(BaseModel):
    """Audit record for a generated JSON/PDF forensic report."""

    report_id: str
    case_id: str
    snapshot_id: str
    report_schema_version: str
    generated_at: str
    export_format: Literal["json", "pdf"]
    relative_path: str
    report_sha256: str = Field(..., min_length=64, max_length=64)
    byte_size: int = Field(..., ge=0)


class ForensicAnalysisSnapshot(BaseModel):
    """Immutable forensic analysis snapshot capturing full provenance, dependencies, and reproducibility."""

    snapshot_id: str = Field(..., description="Unique snapshot ID (MTS-YYYYMMDD-XXXXXXXX)")
    snapshot_schema_version: str = Field(
        default=SCHEMA_VERSIONS["snapshot_schema_version"]
    )
    case_id: str
    created_at: str
    snapshot_sha256: str = Field(
        ...,
        description="Canonical SHA-256 digest of the snapshot excluding snapshot_sha256 itself",
    )
    input_file_provenance: InputFileProvenance
    input_integrity: InputIntegrityVerification
    environment_info: AnalysisEnvironmentInfo
    analyses_executed: List[AnalysisRecord] = Field(default_factory=list)
    ai_model_provenance: AiModelProvenanceRecord
    artifacts_produced: List[ArtifactRecord] = Field(default_factory=list)
    evidence_items_consumed: List[str] = Field(default_factory=list)
    evidence_dependency_graph: List[EvidenceDependencyNode] = Field(default_factory=list)
    interpretation_summary: Optional[Dict[str, Any]] = None
    reproducibility_check: ReproducibilityCheckResult
    audit_events: List[SystemAuditEvent] = Field(default_factory=list)
    reproducibility_limitations: List[str] = Field(
        default_factory=lambda: list(DEFAULT_REPRODUCIBILITY_LIMITATIONS)
    )
