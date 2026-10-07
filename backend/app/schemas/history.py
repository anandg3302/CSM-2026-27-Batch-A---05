from typing import List, Literal
from pydantic import BaseModel, Field

AllowedHistoryEventType = Literal[
    "original_capture_or_creation",
    "possible_synthetic_generation_signal",
    "metadata_processing",
    "image_processing",
    "possible_recompression",
    "possible_visual_edit",
    "possible_export",
    "reference_difference_observed",
    "forensic_analysis_event",
    "unknown_processing_event",
]

AllowedHistoryEventStatus = Literal[
    "observed",
    "supported_inference",
    "possible",
    "unknown",
]

AllowedTemporalCertainty = Literal[
    "exact",
    "approximate",
    "relative",
    "unknown",
]

AllowedQualitativeConfidence = Literal[
    "high",
    "moderate",
    "low",
    "undetermined",
]

AllowedHistoricalAccuracy = Literal[
    "not_determined",
    "partially_supported",
]


class HistoryIntegrityInfo(BaseModel):
    """Integrity metadata verifying that the original uploaded file SHA-256 remains unchanged."""

    original_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Original SHA-256 hex digest recorded at Step 2 upload",
    )
    hash_preserved: bool = Field(
        default=True,
        description="Whether the stored original file still matches the Step 2 SHA-256 digest",
    )


class ReconstructedHistoryEvent(BaseModel):
    """Single possible or observed event in the reconstructed edit-history sequence."""

    id: str = Field(..., description="Unique history event identifier (e.g. EH-001)")
    sequence: int = Field(..., ge=1, description="Deterministic 1-based position in the reconstructed sequence")
    event_type: AllowedHistoryEventType = Field(..., description="Controlled event type classification")
    title: str = Field(..., description="Concise neutral title for the timeline event")
    description: str = Field(..., description="Factual or conservative description of the event")
    source: str = Field(..., description="Originating MediaTrace pipeline stage or evidence source")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="List of correlated Step 7 evidence IDs (e.g. ['EV-001']) supporting this event",
    )
    temporal_certainty: AllowedTemporalCertainty = Field(
        ..., description="Temporal certainty level ('exact', 'approximate', 'relative', 'unknown')"
    )
    confidence: AllowedQualitativeConfidence = Field(
        ..., description="Qualitative confidence level ('high', 'moderate', 'low', 'undetermined')"
    )
    status: AllowedHistoryEventStatus = Field(
        ..., description="Epistemic status ('observed', 'supported_inference', 'possible', 'unknown')"
    )
    reasoning: str = Field(..., description="Forensic reasoning linking the underlying evidence to this event")
    limitations: List[str] = Field(
        default_factory=list,
        description="Explicit limitations describing what this event cannot prove",
    )


class KnownFactItem(BaseModel):
    """Objective, directly measured fact extracted from completed pipeline stages."""

    fact: str = Field(..., description="Directly measured factual statement")
    source: str = Field(..., description="Originating analysis stage (e.g. step2_upload, step3_basic_analysis)")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Correlated Step 7 evidence IDs supporting this fact",
    )


class InferredEventItem(BaseModel):
    """Conservative inference or possible processing event derived from forensic signals."""

    title: str = Field(..., description="Concise title of the inferred or possible event")
    event_type: AllowedHistoryEventType = Field(..., description="Controlled event type")
    confidence: AllowedQualitativeConfidence = Field(..., description="Qualitative confidence level")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Correlated Step 7 evidence IDs supporting this inference",
    )
    reasoning: str = Field(..., description="Conservative explanation of why this event is considered possible")
    limitations: List[str] = Field(
        default_factory=list,
        description="Explicit caveats on what remains unconfirmed",
    )


class EditHistoryReport(BaseModel):
    """Structured possible edit-history reconstruction report for a MediaTrace case."""

    case_id: str = Field(..., description="Forensic case identifier")
    generated_at: str = Field(..., description="UTC ISO-8601 timestamp when reconstruction was generated")
    reconstructed_sequence: bool = Field(
        default=True,
        description="Always true to indicate this sequence is an analytical reconstruction",
    )
    sequence_type: Literal["evidence_ordered_hypothesis"] = Field(
        default="evidence_ordered_hypothesis",
        description="Indicates events are ordered as an evidence-backed hypothesis rather than exact history",
    )
    historical_accuracy: AllowedHistoricalAccuracy = Field(
        default="not_determined",
        description="Whether chronological accuracy is 'not_determined' or 'partially_supported'",
    )
    events: List[ReconstructedHistoryEvent] = Field(
        default_factory=list,
        description="Deterministically ordered sequence of observed and possible events",
    )
    known_facts: List[KnownFactItem] = Field(
        default_factory=list,
        description="Directly measured facts with high certainty",
    )
    inferred_events: List[InferredEventItem] = Field(
        default_factory=list,
        description="Possible or inferred processing events derived from forensic anomalies",
    )
    unknowns: List[str] = Field(
        default_factory=list,
        description="Explicit statements of what cannot be determined from available evidence",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Methodological limitations of edit-history reconstruction",
    )
    integrity: HistoryIntegrityInfo = Field(
        ..., description="Original file SHA-256 preservation verification"
    )
