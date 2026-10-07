from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

AllowedDocumentAssessment = Literal[
    "no_strong_indicators",
    "metadata_inconsistency",
    "structural_inconsistency",
    "revision_indicators_present",
    "embedded_content_present",
    "potential_document_anomaly",
    "insufficient_evidence",
]


class DocumentParserStatus(BaseModel):
    """Availability status of the forensic parser for the given document format."""

    status: Literal["available", "not_available"] = Field(default="available")
    reason: Optional[str] = None


class DocumentMetadataInfo(BaseModel):
    """Extracted document metadata with GPS privacy redaction."""

    title: Optional[str] = None
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    last_modified_by: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creation_timestamp: Optional[str] = None
    modification_timestamp: Optional[str] = None
    application: Optional[str] = None
    generator: Optional[str] = None
    editor: Optional[str] = None
    language: Optional[str] = None
    pdf_version: Optional[str] = None
    document_version: Optional[str] = None
    revision_number: Optional[str] = None
    company: Optional[str] = None
    gps_present: bool = False
    gps_notice: Optional[str] = Field(
        default=None,
        description="Reported strictly as 'GPS metadata present' when GPS/location metadata exists; raw coordinates are never exposed",
    )


class PageDimensionItem(BaseModel):
    """Page dimensions for paginated formats such as PDF."""

    page_number: int = Field(..., ge=1)
    width_pt: float = Field(..., ge=0.0)
    height_pt: float = Field(..., ge=0.0)


class DocumentStructureInfo(BaseModel):
    """Format-specific structural properties and package/object inventory."""

    status: Literal["available", "not_available", "truncated"] = Field(default="available")
    format_version: Optional[str] = None
    page_count: Optional[int] = Field(default=None, ge=0)
    page_dimensions: List[PageDimensionItem] = Field(default_factory=list)
    encrypted: Optional[bool] = None
    linearized: Optional[bool] = None
    javascript_present: Optional[bool] = None
    macro_present: Optional[bool] = None
    package_components: List[str] = Field(default_factory=list)
    object_count: Optional[int] = Field(default=None, ge=0)
    font_count: Optional[int] = Field(default=None, ge=0)
    fonts: List[str] = Field(default_factory=list)
    paragraph_count: Optional[int] = Field(default=None, ge=0)
    table_count: Optional[int] = Field(default=None, ge=0)
    header_count: Optional[int] = Field(default=None, ge=0)
    footer_count: Optional[int] = Field(default=None, ge=0)
    style_count: Optional[int] = Field(default=None, ge=0)
    relationship_count: Optional[int] = Field(default=None, ge=0)
    truncated: bool = False


class DocumentTextAnalysis(BaseModel):
    """Objective text statistics without stylistic or AI authorship speculation."""

    status: Literal["available", "not_available", "truncated"] = Field(default="available")
    encoding: Optional[str] = None
    bom_present: Optional[bool] = None
    newline_style: Optional[str] = None
    character_count: Optional[int] = Field(default=None, ge=0)
    word_count: Optional[int] = Field(default=None, ge=0)
    line_count: Optional[int] = Field(default=None, ge=0)
    paragraph_count: Optional[int] = Field(default=None, ge=0)
    page_count: Optional[int] = Field(default=None, ge=0)
    language_hint: Optional[str] = None
    repeated_text_indicator: bool = False
    unusual_encoding_indicator: bool = False
    truncated: bool = False


class EmbeddedContentBucket(BaseModel):
    """Count and safe metadata inventory for a specific embedded content category."""

    count: int = Field(default=0, ge=0)
    items: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentEmbeddedContent(BaseModel):
    """Inventory of images, embedded files/attachments, hyperlinks, annotations, forms, and objects."""

    images: EmbeddedContentBucket = Field(default_factory=EmbeddedContentBucket)
    embedded_files: EmbeddedContentBucket = Field(default_factory=EmbeddedContentBucket)
    hyperlinks: EmbeddedContentBucket = Field(default_factory=EmbeddedContentBucket)
    annotations: EmbeddedContentBucket = Field(default_factory=EmbeddedContentBucket)
    forms: EmbeddedContentBucket = Field(default_factory=EmbeddedContentBucket)
    objects: EmbeddedContentBucket = Field(default_factory=EmbeddedContentBucket)
    truncated: bool = False


class DocumentRevisionAnalysis(BaseModel):
    """Conservative revision and change indicator analysis."""

    status: Literal["indicators_present", "no_strong_indicators", "not_available"] = Field(
        default="no_strong_indicators"
    )
    description: str = Field(
        default="No tracked-change or multi-revision indicators were detected."
    )
    tracked_changes_detected: bool = False
    tracked_insertions_count: int = Field(default=0, ge=0)
    tracked_deletions_count: int = Field(default=0, ge=0)
    comments_count: int = Field(default=0, ge=0)
    incremental_updates_detected: bool = False
    incremental_eof_count: int = Field(default=0, ge=0)
    revision_number: Optional[str] = None
    last_modified_by: Optional[str] = None
    indicators: List[str] = Field(default_factory=list)


class TimestampRelationshipSignal(BaseModel):
    """Observed relationship between available document/filesystem timestamps."""

    type: str = Field(default="timestamp_relationship")
    status: Literal["potential_signal", "informational"] = Field(default="informational")
    description: str


class DocumentTimestampAnalysis(BaseModel):
    """Comparison of document and filesystem timestamps without assuming chronological truth."""

    filesystem_created_at: Optional[str] = None
    filesystem_modified_at: Optional[str] = None
    document_created_at: Optional[str] = None
    document_modified_at: Optional[str] = None
    signals: List[TimestampRelationshipSignal] = Field(default_factory=list)
    note: str = Field(
        default="Timestamps may be rewritten by copying, exporting, editing, or software and do not independently prove true creation chronology."
    )


class DocumentConsistencySignal(BaseModel):
    """Metadata or structural inconsistency signal requiring human review."""

    finding: str
    category: Literal["metadata", "structure", "timestamp", "revision"] = Field(default="metadata")
    status: Literal["potential_signal"] = Field(default="potential_signal")
    description: str


class DocumentConsistencyAnalysis(BaseModel):
    """Summary of metadata and structural consistency findings."""

    status: Literal["inconsistencies_detected", "consistent", "not_available"] = Field(
        default="consistent"
    )
    signals: List[DocumentConsistencySignal] = Field(default_factory=list)


class DocumentArtifactItem(BaseModel):
    """Derived visual page preview or summary artifact saved under storage/results/<case_id>/document/."""

    page_number: int = Field(..., ge=1)
    artifact: str
    artifact_filename: str


class DocumentAiDetectionResult(BaseModel):
    """Modular AI-generated document detector status."""

    status: Literal["not_available"] = Field(default="not_available")
    reason: str = Field(
        default="No trained AI-generated document detection model is currently configured."
    )


class DocumentForensicAssessment(BaseModel):
    """Conservative qualitative assessment of the document analysis."""

    status: AllowedDocumentAssessment
    label: str
    description: str


class DocumentAnalysisResult(BaseModel):
    """Structured result for Step 12 Document Forensic Analysis."""

    analyzed_at: str
    media_type: Literal["document"] = Field(default="document")
    filename: str
    format: str
    mime_type: str
    file_size: int = Field(..., ge=0)
    status: Literal["available", "not_available"] = Field(default="available")
    reason: Optional[str] = None
    parser_status: DocumentParserStatus = Field(default_factory=DocumentParserStatus)
    metadata: DocumentMetadataInfo = Field(default_factory=DocumentMetadataInfo)
    structure: DocumentStructureInfo = Field(default_factory=DocumentStructureInfo)
    text_analysis: DocumentTextAnalysis = Field(default_factory=DocumentTextAnalysis)
    embedded_content: DocumentEmbeddedContent = Field(default_factory=DocumentEmbeddedContent)
    revision_analysis: DocumentRevisionAnalysis = Field(default_factory=DocumentRevisionAnalysis)
    timestamp_analysis: DocumentTimestampAnalysis = Field(default_factory=DocumentTimestampAnalysis)
    consistency_analysis: DocumentConsistencyAnalysis = Field(
        default_factory=DocumentConsistencyAnalysis
    )
    artifacts: List[DocumentArtifactItem] = Field(default_factory=list)
    ai_generation: DocumentAiDetectionResult = Field(default_factory=DocumentAiDetectionResult)
    assessment: DocumentForensicAssessment
    limitations: List[str] = Field(default_factory=list)
